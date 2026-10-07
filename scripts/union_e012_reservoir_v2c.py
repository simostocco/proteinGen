"""Stream the V2C selected reservoir + historical unique TRAIN into one D: gzip.

Candidate-only output: mandatory protected screening is still outstanding.
No originals are written; no model is imported.
"""
from pathlib import Path
import gzip, importlib.util, json, os, sqlite3, struct, time
import pyarrow.parquet as pq
spec=importlib.util.spec_from_file_location('v2c',Path(__file__).with_name('e012_reservoir_v2c.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ROOT=Path('D:/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization') if os.name=='nt' else Path('/mnt/d/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization')
m.configure_storage(ROOT)
HEADER=struct.Struct('<q32s32sIBBHH')

def base_rows(path):
    with m.require_d(path).open('rb',buffering=8*1024**2) as f:
        while prefix:=f.read(4):
            assert len(prefix)==4
            size=struct.unpack('<I',prefix)[0];body=f.read(size);assert len(body)==size
            off,h,p,mult,hist,ext,ns,nq=HEADER.unpack_from(body)
            assert size==HEADER.size+ns+nq
            yield {'off':off,'h':h.hex(),'p':p.hex(),'multiplicity':mult,'historical':hist,'external':ext,
                'source_id':body[HEADER.size:HEADER.size+ns].decode(),'seq':body[HEADER.size+ns:].decode()}

def run():
    work=ROOT/'continuation';out=work/'union';out.mkdir(exist_ok=True)
    marker=out/'union.complete.json'
    if marker.exists():
        cert=json.loads(marker.read_text());assert m.sha_file(out/'candidates.jsonl.gz')==cert['sha256'];return
    ns=importlib.util.spec_from_file_location('native',Path(__file__).with_name('run_e012_v2b_native.py'))
    native=importlib.util.module_from_spec(ns);ns.loader.exec_module(native);legacy=native.load()
    db=m.open_delta(work/'delta.sqlite')
    row=db.execute("SELECT v FROM progress WHERE k='continuation_checkpoint'").fetchone();assert row
    cp=json.loads(row[0]);cp=m.resume_continuation(db,cp['inputs']);state=cp['state']
    assert state['stage']=='historical' and state['raw']==m.RAW_COUNT
    pool=json.loads((Path(state['pool_root'])/'pool.json').read_text());c=pool['cutoff']
    cutoff=(bytes.fromhex(c[0]),bytes.fromhex(c[1]),c[2]) if c else None
    historical={}
    for batch in pq.ParquetFile(m.historical_train_file(ROOT,legacy.REPO)).iter_batches(columns=['sample_id','sequence']):
        for item in batch.to_pylist():
            seq=item['sequence'];assert legacy.valid(seq)
            historical[seq]=min(str(item['sample_id']),historical.get(seq,str(item['sample_id'])))
    assert len(historical)==87_930
    remaining=set(historical);updates={(h,off):(mult,sid) for h,off,mult,sid in db.execute('SELECT h,off,multiplicity,source_id FROM updates')}
    # Only duplicate events are cached, not the 24M full sequence population.
    def rows():
        yield from base_rows(ROOT/'exports/candidate_payload.bin')
        for filename in cp['outputs']:
            if Path(filename).name=='candidates.jsonl':
                with m.require_d(filename).open(encoding='utf-8') as f:
                    for line in f:yield json.loads(line)
    stats=legacy.Stats();external=overlap=added=0
    progress=m.Progress(out,0,m.CAP+87_930);progress.begin_processing()
    partial=out/'candidates.jsonl.gz.partial'
    with partial.open('wb') as file,gzip.GzipFile(filename='',fileobj=file,mode='wb',mtime=0,compresslevel=3) as gz:
        def emit(item):
            seq=item['seq'];assert legacy.valid(seq)
            assert m.sequence_hash(seq).hex()==item['h'] and m.priority(seq).hex()==item['p']
            stats.add(seq);gz.write((json.dumps(item,sort_keys=True,separators=(',',':'))+'\n').encode())
        for item in rows():
            h=bytes.fromhex(item['h']);p=bytes.fromhex(item['p']);off=item['off']
            if cutoff and (p,h,off)>cutoff:continue
            update=updates.get((h,off))
            if update:item['multiplicity']+=update[0];item['source_id']=min(item['source_id'],update[1])
            if item['seq'] in remaining:
                remaining.remove(item['seq']);item['historical']=1;overlap+=1
            emit(item);external+=1
            progress.emit(external,external,external,0)
        assert external==pool['base_count']+pool['delta_count']==m.CAP
        for ordinal,(seq,sid) in enumerate(sorted(historical.items())):
            if seq not in remaining:continue
            match=state['historical_matches'].get(seq)
            off,source_id,mult=match if match else (-ordinal-1,sid,1)
            emit({'off':off,'source_id':source_id,'multiplicity':mult,'seq':seq,'h':m.sequence_hash(seq).hex(),
                'p':m.priority(seq).hex(),'historical':1,'external':int(match is not None)})
            added+=1
    with partial.open('r+b') as f:f.flush();os.fsync(f.fileno())
    final=out/'candidates.jsonl.gz';partial.replace(final)
    assert added+overlap==87_930
    m.atomic_json(marker,{'input_checkpoint_sha256':m.sha_file(work/'continuation_checkpoint.json'),
        'sha256':m.sha_file(final),'bytes':final.stat().st_size,'external_reservoir':external,
        'historical_added':added,'cross_source':overlap,'union_count':stats.n,'statistics':stats.summary(),
        'protected_screen_complete':False,'corpus_ready':False,'training_launched':False})
    db.close()

if __name__=='__main__':run()
