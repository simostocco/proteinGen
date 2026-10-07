"""V2C exact Bloom negatives and immutable-base / sequential-delta reservoir.

All persistent files, checkpoints, caches and benchmark output are on D:.
The original V2B database and 30M JSON mirror are never written.
"""
from pathlib import Path
from collections import Counter,deque
from contextlib import closing
import hashlib, json, math, os, shutil, sqlite3, struct, time
import numpy as np

RAW_COUNT=38_840_027
SEED=12014
CAP=24_000_000
AA=frozenset('ACDEFGHIKLMNPQRSTVWY')
KEY_DTYPE=np.dtype([('p','S32'),('h','S32'),('off','<i8'),('payload','<u8'),('length','<u2')])
SEEN_DTYPE=np.dtype([('h','S32'),('off','<i8')])

def require_d(path):
    path=Path(path).resolve()
    if os.name=='nt':
        assert path.drive.upper()=='D:', f'Non-D artifact path: {path}'
    else:
        assert path.is_relative_to('/mnt/d'),f'Non-D artifact path: {path}'
    return path

def configure_storage(root):
    root=require_d(root); root.mkdir(parents=True,exist_ok=True)
    tmp=root/'tmp';tmp.mkdir(exist_ok=True)
    for k in ['TMP','TEMP','TMPDIR','XDG_CACHE_HOME']:
        os.environ[k]=str(tmp)
    import tempfile
    tempfile.tempdir=str(tmp)
    return root

def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb',buffering=8*1024**2) as f:
        while b:=f.read(8*1024**2):h.update(b)
    return h.hexdigest()

def atomic_json(path,value):
    path=require_d(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def priority(seq):return hashlib.sha256(b'E012-V2B:12014\0'+seq.encode('ascii')).digest()
def sequence_hash(seq):return hashlib.sha256(seq.encode('ascii')).digest()
def fixed(value):return bytes(value).ljust(32,b'\0')
def key_tuple(row):return fixed(row['p']),fixed(row['h']),int(row['off'])

def historical_train_file(root,repo):
    """Copy the frozen historical input to D: without changing its original bytes."""
    expected='75b6861359f6e9526443fbf1d2b475e55086306df18d4144ca0c7b1afcc580a6'
    target=require_d(root)/'preservation/historical_train.parquet'
    if not target.exists():
        source=Path(repo)/'outputs/e012_causal_rope_sequence/pilot_v1/train.parquet'
        assert sha_file(source)==expected,'Historical TRAIN input changed'
        partial=target.with_suffix('.partial');shutil.copyfile(source,partial)
        assert sha_file(partial)==expected
        partial.replace(target)
    assert sha_file(target)==expected
    return target

class Bloom:
    """No false negatives; positive results always require exact string verification.

    Power-of-two size keeps Python/NumPy double hashing identical, including uint64
    overflow. Default memory is exactly 128 MiB, with 13 probes. At 38.84M hashes,
    the theoretical false-positive probability is approximately 3e-6.
    """
    def __init__(self,bits=1<<30,probes=13):
        assert bits>=8 and bits&(bits-1)==0
        self.bits=bits;self.mask=bits-1;self.probes=probes
        self.data=np.zeros(bits//8,dtype=np.uint8)
        self.additions=0

    def locations(self,h):
        a,b=struct.unpack('<QQ',h[:16]);b|=1
        return ((a+i*b)&self.mask for i in range(self.probes))

    def possible(self,h):
        return all(self.data[i>>3] & (1<<(i&7)) for i in self.locations(h))

    def add(self,h):
        for i in self.locations(h):self.data[i>>3]|=1<<(i&7)
        self.additions+=1

    def seed_file(self,path,progress=None):
        path=require_d(path)
        rows=np.memmap(path,dtype=SEEN_DTYPE,mode='r')
        for first in range(0,len(rows),250_000):
            # Copy only a bounded hash batch, never sequence strings.
            hashes=np.ascontiguousarray(rows['h'][first:first+250_000])
            halves=hashes.view('<u8').reshape(-1,4)
            a=halves[:,0];b=halves[:,1]|np.uint64(1)
            for j in range(self.probes):
                loc=(a+np.uint64(j)*b)&np.uint64(self.mask)
                byte=loc>>np.uint64(3)
                bit=np.left_shift(np.uint8(1),(loc&np.uint64(7)).astype(np.uint8))
                np.bitwise_or.at(self.data,byte,bit)
            self.additions+=len(hashes)
            if progress:progress(first+len(hashes),len(rows))

    def save(self,path):
        path=require_d(path)
        tmp=path.with_suffix(path.suffix+'.partial')
        with tmp.open('wb') as f:self.data.tofile(f);f.flush();os.fsync(f.fileno())
        tmp.replace(path)

def partition_prefix(left,right,k):
    """Exact kth order statistic of two sorted key arrays; O(log(min(N,M))) reads."""
    k=min(k,len(left)+len(right))
    low=max(0,k-len(right));high=min(k,len(left))
    while low<=high:
        i=(low+high)//2;j=k-i
        if i and j<len(right) and key_tuple(left[i-1])>key_tuple(right[j]):high=i-1
        elif j and i<len(left) and key_tuple(right[j-1])>key_tuple(left[i]):low=i+1
        else:
            ends=[]
            if i:ends.append(key_tuple(left[i-1]))
            if j:ends.append(key_tuple(right[j-1]))
            return i,j,max(ends) if ends else None
    raise AssertionError('Unsorted/corrupt priority arrays')

def sorted_keys(path,out):
    """One-time compact-key sort; at most ~2.1GB for the 24M immutable base."""
    path=require_d(path);out=require_d(out)
    data=np.fromfile(path,dtype=KEY_DTYPE)
    data.sort(order=['p','h','off'])
    partial=out.with_suffix(out.suffix+'.partial')
    with partial.open('wb') as f:data.tofile(f);f.flush();os.fsync(f.fileno())
    partial.replace(out)
    atomic_json(out.with_suffix('.complete.json'),{'input_sha256':sha_file(path),
        'output_sha256':sha_file(out),'count':len(data),'record_bytes':KEY_DTYPE.itemsize,
        'maximum_array_bytes':data.nbytes,'sort_order':['p','h','off']})

class Progress:
    def __init__(self,root,raw_start,requested):
        import psutil
        self.psutil=psutil;self.proc=psutil.Process();self.start=time.perf_counter()
        self.last=self.start;self.raw_start=raw_start;self.requested=requested
        self.io0=self.proc.io_counters();self.disk0=psutil.disk_io_counters()
        self.log=require_d(root)/'progress.jsonl'
        self.phase='setup'
        self.recent_raw=raw_start;self.recent_time=self.start
        self.history=deque([(self.start,raw_start)],maxlen=256)
        self.free_before=shutil.disk_usage(require_d(root)).free
        self.minimum_free=self.free_before

    def begin_processing(self):
        self.phase='processing'
        self.start=time.perf_counter();self.last=self.start
        self.recent_time=self.start
        self.history=deque([(self.start,self.raw_start)],maxlen=256)

    def emit(self,raw,canonical,reservoir,queries,force=False,**details):
        elapsed=time.perf_counter()-self.start
        if not force and raw%100_000 and time.perf_counter()-self.last<60:return
        done=raw-self.raw_start;rate=done/max(elapsed,1e-9)
        now=time.perf_counter();self.history.append((now,raw))
        while len(self.history)>2 and self.history[1][0]<now-120:self.history.popleft()
        recent_rate=(raw-self.history[0][1])/max(now-self.history[0][0],1e-9)
        io=self.proc.io_counters();disk=self.psutil.disk_io_counters()
        free=shutil.disk_usage(self.log.parent).free;self.minimum_free=min(self.minimum_free,free)
        row={'phase':self.phase,'io_includes_setup':True,'raw':raw,'percent_of_38840027':100*raw/RAW_COUNT,'raw_per_second':rate,
            'canonical_per_second':canonical/max(elapsed,1e-9),'reservoir_count':reservoir,
            'elapsed_seconds':elapsed,'eta_seconds':(self.requested-done)/rate if rate else None,
            'rss_bytes':self.proc.memory_info().rss,'peak_rss_bytes':getattr(self.proc.memory_info(),'peak_wset',self.proc.memory_info().rss),
            'process_read_bytes':io.read_bytes-self.io0.read_bytes,'process_write_bytes':io.write_bytes-self.io0.write_bytes,
            'host_disk_read_bytes':disk.read_bytes-self.disk0.read_bytes,'host_disk_write_bytes':disk.write_bytes-self.disk0.write_bytes,
            'database_queries':queries,'raw_per_second_recent':recent_rate,'cpu_threads':self.psutil.cpu_count(),
            'd_free_bytes':free,'d_free_before_bytes':self.free_before,
            'd_peak_consumption_bytes_sampled':self.free_before-self.minimum_free,
            'disk_peak_includes_unrelated_activity':True,'durable':False,'training_launched':False}
        row.update(details)
        with self.log.open('a',encoding='utf-8') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
        print(json.dumps(row),flush=True);self.last=time.perf_counter()
        self.recent_raw=raw;self.recent_time=self.last

class ExactMembership:
    """Immutable baseline plus small sorted/batched delta index and current interval.

    Bloom negatives do not query SQLite. Newly encountered hashes are immediately
    added to the Bloom and current-interval map, even when priority rejects them.
    A positive checks full sequence strings; hash collisions keep distinct offsets.
    """
    def __init__(self,baseline,delta,bloom,sequence_at):
        self.base=baseline;self.delta=delta;self.bloom=bloom;self.sequence_at=sequence_at
        self.current={};self.queries=0;self.array_queries=0;self.negatives=0;self.positives=0

    def find(self,h,seq):
        if not self.bloom.possible(h):self.negatives+=1;return None
        self.positives+=1
        for off,previous in self.current.get(h,[]):
            if previous==seq:return off
        for db in [self.delta,self.base]:
            if isinstance(db,ArrayBaseline):self.array_queries+=1
            else:self.queries+=1
            for off, in db.execute('SELECT off FROM exact_hashes WHERE h=?',(h,)):
                if self.sequence_at(h,off)==seq:return off
        return None

    def add(self,h,off,seq):
        self.bloom.add(h)
        self.current.setdefault(h,[]).append((off,seq))

    def commit_hashes(self):
        # Sorted key insertion touches only the small delta B-tree, not the 13GB base.
        rows=sorted((h,off) for h,values in self.current.items() for off,_ in values)
        self.delta.executemany('INSERT INTO exact_hashes(h,off) VALUES(?,?)',rows)
        self.current.clear()

class TieredPool:
    """The exact priority prefix of immutable base + sequential sorted delta.

    No per-record writes into the large baseline. Only compact delta keys are sorted
    at interval boundaries; base membership is a shrinking sorted prefix.
    """
    def __init__(self,base_path,root,cap=CAP):
        self.base=np.memmap(require_d(base_path),dtype=KEY_DTYPE,mode='r')
        self.base_count=len(self.base);self.delta=np.empty(0,dtype=KEY_DTYPE)
        self.root=require_d(root);self.cap=cap;self.pending=[]
        self.cutoff=key_tuple(self.base[-1]) if self.base_count>=cap else None

    def accepts(self,p,h,off):return self.cutoff is None or (p,h,off)<=self.cutoff

    def add(self,p,h,off,payload,length):self.pending.append((p,h,off,payload,length))

    def select_interval(self):
        new=np.array(self.pending,dtype=KEY_DTYPE)
        combined=np.concatenate([self.delta,new])
        combined.sort(order=['p','h','off'])
        self.base_count,keep,self.cutoff=partition_prefix(self.base[:self.base_count],combined,self.cap)
        self.delta=combined[:keep].copy();self.pending=[]
        assert self.base_count+len(self.delta)<=self.cap
        if self.base_count+len(self.delta)<self.cap:self.cutoff=None
        return self.base_count+len(self.delta)

    def write_checkpoint(self,path,inputs):
        path=require_d(path);path.mkdir(parents=True,exist_ok=True)
        keys=path/'delta_priority.bin'
        with keys.open('wb') as f:self.delta.tofile(f);f.flush();os.fsync(f.fileno())
        atomic_json(path/'pool.json',{'inputs':inputs,'base_count':self.base_count,
            'delta_count':len(self.delta),'cutoff':[self.cutoff[0].hex(),self.cutoff[1].hex(),self.cutoff[2]] if self.cutoff else None,
            'delta_priority_sha256':sha_file(keys)})

    def selected(self):
        # Used only for compact benchmark equivalence signatures; bounded arrays.
        both=np.concatenate([self.base[:self.base_count],self.delta])
        both.sort(order=['p','h','off'])
        return both

    def close(self):
        self.base._mmap.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

def open_delta(path):
    path=require_d(path)
    db=sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.execute('PRAGMA cache_size=-524288')
    db.execute('CREATE TABLE IF NOT EXISTS exact_hashes(h BLOB,off INTEGER,PRIMARY KEY(h,off)) WITHOUT ROWID')
    db.execute('CREATE TABLE IF NOT EXISTS updates(h BLOB,off INTEGER,multiplicity INTEGER,source_id TEXT,PRIMARY KEY(h,off)) WITHOUT ROWID')
    db.execute('CREATE TABLE IF NOT EXISTS progress(k TEXT PRIMARY KEY,v TEXT)')
    return db

def commit_continuation(db,root,state,inputs,outputs,after_commit=None,previous_outputs=None):
    """Authoritative SQL state follows fsynced derived files; JSON is a mirror.

    An interruption after SQLite commit cannot cause replay against already seen
    hashes. Uncommitted files are never selected merely because they exist.
    """
    root=require_d(root)
    sealed=dict(previous_outputs or {})
    # Carry only previously sealed immutable generations. New paths must be disjoint.
    for p in outputs:
        name=str(require_d(p))
        assert name not in sealed,'Checkpoint generation would overwrite committed output'
        sealed[name]={'sha256':sha_file(p),'bytes':Path(p).stat().st_size}
    checkpoint={'state':state,'inputs':inputs,'outputs':sealed}
    db.execute("INSERT OR REPLACE INTO progress VALUES('continuation_checkpoint',?)",(json.dumps(checkpoint,sort_keys=True),))
    db.commit()
    if after_commit:after_commit()
    atomic_json(root/'continuation_checkpoint.json',checkpoint)
    return checkpoint

def resume_continuation(db,inputs):
    row=db.execute("SELECT v FROM progress WHERE k='continuation_checkpoint'").fetchone()
    if not row:return None
    checkpoint=json.loads(row[0])
    assert checkpoint['inputs']==inputs,'Continuation inputs changed'
    for name,metadata in checkpoint['outputs'].items():
        path=require_d(name)
        assert path.stat().st_size==metadata['bytes'] and sha_file(path)==metadata['sha256']
    return checkpoint

def seed_committed_delta(bloom,db):
    """Required on every resume; cache lag must never introduce false negatives."""
    for h, in db.execute('SELECT h FROM exact_hashes ORDER BY h,off'):
        bloom.add(h)

class ArrayBaseline:
    """Immutable exact (hash, offset) index; avoids opening the large frozen WAL.

    Positives still verify full strings using the immutable source, never hash only.
    Binary search operates on 32-byte keys; collisions retain every distinct offset.
    """
    def __init__(self,path):
        self.rows=np.memmap(require_d(path),mode='r',dtype=SEEN_DTYPE)
    def execute(self,sql,parameters):
        assert sql=='SELECT off FROM exact_hashes WHERE h=?'
        h=parameters[0];key=np.bytes_(h)
        first=int(np.searchsorted(self.rows['h'],key,side='left'))
        last=int(np.searchsorted(self.rows['h'],key,side='right'))
        return ((int(row['off']),) for row in self.rows[first:last] if fixed(row['h'])==h)
    def close(self):self.rows._mmap.close()

def prepare_array_baseline(root):
    root=require_d(root);source=root/'exports/seen.bin';out=root/'exports/seen_hash_sorted.bin'
    cert=out.with_suffix('.complete.json')
    if cert.exists():
        metadata=json.loads(cert.read_text());assert out.stat().st_size==metadata['bytes']
        assert metadata['source_sha256']==json.loads((root/'exports/export.complete.json').read_text())['outputs']['seen.bin']['sha256']
        assert metadata['count']==24_715_247
        assert sha_file(out)==metadata['sha256'];return out
    import psutil
    assert psutil.virtual_memory().available>=2*source.stat().st_size,'Insufficient RAM for bounded compact-key sort'
    rows=np.fromfile(source,dtype=SEEN_DTYPE);rows.sort(order=['h','off'])
    partial=out.with_suffix('.partial')
    with partial.open('wb') as f:rows.tofile(f);f.flush();os.fsync(f.fileno())
    partial.replace(out)
    atomic_json(cert,{'source_sha256':sha_file(source),'sha256':sha_file(out),'bytes':out.stat().st_size,
        'count':len(rows),'bounded_sort_array_bytes':rows.nbytes,'ordering':['h','off']})
    return out
