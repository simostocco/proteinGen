"""Validate/export the preserved committed SQLite state using sequential D: reads."""
from pathlib import Path
import array, hashlib, json, os, struct, time
from collections import Counter
import numpy as np

ROOT=Path('D:/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization')
AA=frozenset('ACDEFGHIKLMNPQRSTVWY')
KEY=struct.Struct('<32s32sqQH')
FRAME=np.dtype([('pid','>u4'),('dbsize','>u4'),('salt','S8'),('checksum','S8'),('data','V4096')])

def save(path,obj):
    partial=path.with_suffix(path.suffix+'.partial')
    with partial.open('w',encoding='utf-8') as f:
        json.dump(obj,f,sort_keys=True,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    partial.replace(path)

def varint(buf,pos):
    value=0
    for i in range(8):
        b=buf[pos]; pos+=1; value=(value<<7)|(b&127)
        if b<128:return value,pos
    return (value<<8)|buf[pos],pos+1

def record(payload):
    end,pos=varint(payload,0)
    types=[]
    while pos<end:
        t,pos=varint(payload,pos); types.append(t)
    pos=end; values=[]
    sizes={0:0,1:1,2:2,3:3,4:4,5:6,6:8,7:8,8:0,9:0}
    for t in types:
        size=sizes[t] if t<12 else (t-12)//2
        raw=payload[pos:pos+size]; pos+=size
        if t==0:v=None
        elif t in (8,9):v=t-8
        elif t in range(1,7):v=int.from_bytes(raw,'big',signed=True)
        elif t==7:v=struct.unpack('>d',raw)[0]
        elif t>=12:v=bytes(raw) if t%2==0 else bytes(raw).decode('utf-8')
        else:raise ValueError('Reserved SQLite serial type')
        values.append(v)
    assert pos==len(payload)
    return values

class Pager:
    def __init__(self):
        audit=json.loads((ROOT/'reports/checkpoint_audit.json').read_text())
        self.schema={x['name']:x for x in audit['schema']}
        self.ps=audit['database_metadata']['page_size']; assert self.ps==4096
        self.n=audit['database_metadata']['page_count']
        with (ROOT/'audit/reservoir.sqlite-shm').open('rb') as f:hdr=f.read(96)
        assert hdr[:48]==hdr[48:]
        fields=struct.unpack('<IIIBBHIIIIIIII',hdr[:48])
        assert fields[0]==3007000 and fields[3]==1 and fields[5]==self.ps
        self.mxframe=fields[6]; assert fields[7]==self.n
        self.dbpath=ROOT/'preservation/reservoir.sqlite'
        self.walpath=ROOT/'preservation/reservoir.sqlite-wal'
        self.db=self.dbpath.open('rb',buffering=0)
        self.wal=self.walpath.open('rb',buffering=0)
        self.latest=np.zeros(self.n+1,dtype=np.uint32)
        with self.walpath.open('rb',buffering=8*1024**2) as f:
            header=f.read(32); assert struct.unpack('>I',header[8:12])[0]==self.ps
            self.salt=header[16:24]
            remaining=self.mxframe; index=1
            while remaining:
                take=min(remaining,2048)
                buf=f.read(take*FRAME.itemsize); assert len(buf)==take*FRAME.itemsize
                rows=np.frombuffer(buf,dtype=FRAME)
                assert np.all(rows['pid']>=1) and np.all(rows['pid']<=self.n)
                assert all(bytes(s)==self.salt for s in rows['salt'])
                self.latest[rows['pid']]=np.arange(index,index+take,dtype=np.uint32)
                # Duplicate page numbers inside a block must keep the final frame.
                np.maximum.at(self.latest,rows['pid'],np.arange(index,index+take,dtype=np.uint32))
                remaining-=take; index+=take
            assert int(rows['dbsize'][-1])==self.n
        self.bytes_read=self.mxframe*FRAME.itemsize+32

    def page(self,pid):
        assert 1<=pid<=self.n
        frame=int(self.latest[pid])
        if frame:
            self.wal.seek(32+(frame-1)*FRAME.itemsize+24)
            data=self.wal.read(self.ps)
        else:
            self.db.seek((pid-1)*self.ps); data=self.db.read(self.ps)
        self.bytes_read+=len(data); assert len(data)==self.ps
        return data

    def reachable(self,name):
        mask=np.zeros(self.n+1,dtype=np.uint8)
        stack=[self.schema[name]['rootpage']]
        total=0
        while stack:
            pid=stack.pop(); assert mask[pid]==0,'B-tree cycle/shared page'
            kind=int(self.kinds[pid])
            assert kind in (2,5,10,13)
            mask[pid]=kind; total+=1
            if kind in (2,5):
                stack.extend(self.children[pid])
            if total%100000==0: print(json.dumps({'stage':'tree_inventory','tree':name,'pages':total}),flush=True)
        return mask

    def inventory(self):
        self.kinds=np.zeros(self.n+1,dtype=np.uint8); self.children={}
        start=time.perf_counter(); last=start
        for pid,page in self.logical_selected_pages():
            offset=100 if pid==1 else 0; kind=int(page[offset]); self.kinds[pid]=kind
            if kind in (2,5):
                n=struct.unpack_from('>H',page,offset+3)[0]
                if n>(self.ps-offset-12)//2:continue
                children=[struct.unpack_from('>I',page,offset+8)[0]]
                for i in range(n):
                    pos=struct.unpack_from('>H',page,offset+12+2*i)[0]
                    if pos+4>self.ps:break
                    children.append(struct.unpack_from('>I',page,pos)[0])
                if len(children)==n+1 and all(1<=p<=self.n for p in children):self.children[pid]=children
            if time.perf_counter()-last>=60:
                print(json.dumps({'stage':'sequential_page_inventory','read_bytes':self.bytes_read,
                    'elapsed_seconds':time.perf_counter()-start}),flush=True);last=time.perf_counter()

    def logical_selected_pages(self,seen=None,candidates=None):
        # Each selected logical page is consumed exactly once. WAL overrides are
        # read in frame order, rather than randomly fetching frames per table row.
        with self.dbpath.open('rb',buffering=8*1024**2) as f:
            pid=1
            while buf:=f.read(self.ps*2048):
                assert len(buf)%self.ps==0
                for i in range(len(buf)//self.ps):
                    if (seen is None or seen[pid] or candidates[pid]) and self.latest[pid]==0:
                        yield pid,memoryview(buf)[i*self.ps:(i+1)*self.ps]
                    pid+=1
                self.bytes_read+=len(buf)
        with self.walpath.open('rb',buffering=8*1024**2) as f:
            f.seek(32); remaining=self.mxframe; index=1
            while remaining:
                take=min(remaining,2048); buf=f.read(take*FRAME.itemsize)
                rows=np.frombuffer(buf,dtype=FRAME)
                for i,pid in enumerate(rows['pid']):
                    if (seen is None or seen[pid] or candidates[pid]) and self.latest[pid]==index+i:
                        yield int(pid),memoryview(buf)[i*FRAME.itemsize+24:(i+1)*FRAME.itemsize]
                self.bytes_read+=len(buf); remaining-=take; index+=take

def cells(page,pid):
    start=100 if pid==1 else 0; kind=page[start]
    if kind==5:return
    n=struct.unpack_from('>H',page,start+3)[0]
    header=12 if kind==2 else 8
    for i in range(n):
        pos=struct.unpack_from('>H',page,start+header+2*i)[0]
        if kind==2:pos+=4
        size,pos=varint(page,pos)
        if kind==13:_,pos=varint(page,pos)
        assert pos+size<=len(page),'Unexpected overflow payload in sequence/hash record'
        yield record(page[pos:pos+size])

def main():
    for key in ['TMP','TEMP','TMPDIR']:os.environ[key]=str(ROOT/'tmp')
    import tempfile
    tempfile.tempdir=str(ROOT/'tmp')
    out=ROOT/'exports'; out.mkdir(exist_ok=True)
    if (out/'export.complete.json').exists():raise RuntimeError('Completed export exists; preserve it')
    state=json.loads((ROOT/'audit/authoritative_state.json').read_text())
    assert state['raw']==31_000_000 and state['offset']==15_459_089_980
    assert state['skip_completed_offset']==state['offset'] and state['stage']=='external'
    assert state['unique']==state['canonical']==24_715_247 and state['duplicates']==0
    for key,n in [('raw_stats',state['raw']),('valid_stats',state['canonical'])]:
        s=state[key]
        assert s['n']==n and sum(s['lengths'].values())==n
        assert sum(int(k)*v for k,v in s['lengths'].items())==s['residues']
        assert sum(s['aa'].values())==s['residues']
    pager=Pager(); pager.inventory()
    seen=pager.reachable('exact_hashes'); candidate=pager.reachable('candidates')
    assert not np.any((seen!=0)&(candidate!=0))
    stats={'seen':0,'candidates':0,'residues':0,'lengths':Counter(),'maximum_seen_offset':0}
    start=time.perf_counter(); last=start
    with (out/'seen.bin.partial').open('wb',buffering=8*1024**2) as fs,\
         (out/'candidate_payload.bin.partial').open('wb',buffering=8*1024**2) as fp,\
         (out/'candidate_keys.bin.partial').open('wb',buffering=8*1024**2) as fk:
        for pid,page in pager.logical_selected_pages(seen,candidate):
            for values in cells(page,pid):
                if seen[pid]:
                    h,off=values
                    assert isinstance(h,bytes) and len(h)==32 and 0<=off<=state['offset']
                    fs.write(h+struct.pack('<q',off)); stats['seen']+=1
                    stats['maximum_seen_offset']=max(stats['maximum_seen_offset'],off)
                else:
                    h,off,p,seq,source_id,multiplicity,historical,external=values
                    assert 20<=len(seq)<=500 and set(seq)<=AA
                    assert hashlib.sha256(seq.encode('ascii')).digest()==h
                    assert hashlib.sha256(b'E012-V2B:12014\0'+seq.encode('ascii')).digest()==p
                    assert 0<=off<=state['offset'] and source_id.startswith('UniRef50_')
                    assert multiplicity==1 and historical==0 and external==1
                    position=fp.tell()
                    sid=source_id.encode('utf-8'); sequence=seq.encode('ascii')
                    body=struct.pack('<q32s32sIBBHH',off,h,p,multiplicity,historical,external,len(sid),len(sequence))+sid+sequence
                    fp.write(struct.pack('<I',len(body))+body)
                    fk.write(KEY.pack(p,h,off,position,len(seq)))
                    stats['candidates']+=1; stats['residues']+=len(seq); stats['lengths'][len(seq)]+=1
            if time.perf_counter()-last>=60:
                import psutil
                print(json.dumps({'stage':'sequential_export','seen':stats['seen'],'candidates':stats['candidates'],
                    'elapsed_seconds':time.perf_counter()-start,'rss_bytes':psutil.Process().memory_info().rss,
                    'logical_read_bytes':pager.bytes_read}),flush=True); last=time.perf_counter()
        for f in [fs,fp,fk]:f.flush();os.fsync(f.fileno())
    assert stats['seen']==state['unique']
    assert stats['candidates']==24_000_000
    stats['lengths']=dict(stats['lengths'])
    stats.update({'source_raw':state['raw'],'source_offset':state['offset'],'skip_completed_offset':state['skip_completed_offset'],
        'stage':state['stage'],'reservoir_count':stats['candidates'],'missing_old_reservoir_count_inferred_and_checked':True,
        'read_bytes':pager.bytes_read,'wall_seconds':time.perf_counter()-start,'all_candidate_hashes_priorities_lengths_and_flags_verified':True,
        'sqlite_validated_wal_maxframe':pager.mxframe,'logical_page_count':pager.n,'training_launched':False})
    outputs={}
    for name in ['seen.bin','candidate_payload.bin','candidate_keys.bin']:
        p=out/(name+'.partial'); final=out/name; p.replace(final)
        h=hashlib.sha256()
        with final.open('rb',buffering=8*1024**2) as f:
            while b:=f.read(8*1024**2):h.update(b)
        outputs[name]={'bytes':final.stat().st_size,'sha256':h.hexdigest()}
    save(out/'export.complete.json',{'statistics':stats,'outputs':outputs})
    save(ROOT/'reports/committed_31m_invariants.json',stats)
    print(json.dumps(stats),flush=True)

if __name__=='__main__':main()
