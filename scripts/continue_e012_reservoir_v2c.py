"""Gated, D:-only immutable-base external-tail continuation. Never writes V2B artifacts.

Stops after external reservoir selection; historical union and protected MMseqs screening remain mandatory.
No dataset is certified training-ready by this command.
"""
from pathlib import Path
import copy, gzip, hashlib, importlib.util, json, os, shutil, sqlite3, struct, sys, time
import numpy as np
import pyarrow.parquet as pq

spec=importlib.util.spec_from_file_location('v2c',Path(__file__).with_name('e012_reservoir_v2c.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ROOT=Path('D:/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization') if os.name=='nt' else Path('/mnt/d/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization')
m.configure_storage(ROOT)

def run():
    # Conservative incremental tail/union budget, not an unused full-clustering estimate.
    assert 30_000_000_000 <= .7*shutil.disk_usage(ROOT).free,'Insufficient D: safety margin for continuation'
    gate=json.loads((ROOT/'BENCHMARK_GATE_PASSED.json').read_text())
    assert gate['report_sha256']==m.sha_file(ROOT/'reports/BENCHMARK_EQUIVALENCE.json')
    report=json.loads((ROOT/'reports/BENCHMARK_EQUIVALENCE.json').read_text())
    assert report['all_required_benchmarks_passed'] and len(report['cases'])==2
    assert report['tests_report_sha256']==m.sha_file(ROOT/'reports/TESTS.json')
    tests=json.loads((ROOT/'reports/TESTS.json').read_text())
    assert tests['passed'] and tests['core_source_sha256']==m.sha_file(Path(__file__).with_name('e012_reservoir_v2c.py'))
    assert all(c['speedup_including_checkpoint']>=5 and c['processing_speedup']>=5 for c in report['cases'])
    ns=importlib.util.spec_from_file_location('native',Path(__file__).with_name('run_e012_v2b_native.py'))
    native=importlib.util.module_from_spec(ns);ns.loader.exec_module(native);legacy=native.load()
    work=m.configure_storage(ROOT/'continuation');db=m.open_delta(work/'delta.sqlite')
    inputs={'baseline_raw':31_000_000,'seed':12014,'gate_sha256':m.sha_file(ROOT/'BENCHMARK_GATE_PASSED.json'),
        'baseline_export_sha256':m.sha_file(ROOT/'exports/export.complete.json')}
    recovered=m.resume_continuation(db,inputs)
    sealed_outputs=dict(recovered['outputs']) if recovered else {}
    state=copy.deepcopy(recovered['state'] if recovered else json.loads((ROOT/'audit/authoritative_state.json').read_text()))
    if state['stage']=='complete':return
    base=sqlite3.connect('file:'+(ROOT/'audit/reservoir.sqlite').as_posix()+'?mode=ro',uri=True)
    base.execute('PRAGMA query_only=ON');base.execute('PRAGMA cache_size=-262144')
    pool=m.TieredPool(ROOT/'exports/base_priority_sorted.bin',work)
    if recovered:
        pp=json.loads((Path(state['pool_root'])/'pool.json').read_text())
        pool.base_count=pp['base_count'];pool.delta=np.fromfile(Path(state['pool_root'])/'delta_priority.bin',dtype=m.KEY_DTYPE)
        c=pp['cutoff'];pool.cutoff=(bytes.fromhex(c[0]),bytes.fromhex(c[1]),c[2]) if c else None
    progress=m.Progress(work,state['raw'],m.RAW_COUNT-state['raw'])
    starting_canonical=state['canonical']
    bloom=m.Bloom();bloom.seed_file(ROOT/'exports/seen.bin',lambda *_:progress.emit(state['raw'],0,pool.base_count+len(pool.delta),0))
    m.seed_committed_delta(bloom,db)
    historical={}
    for batch in pq.ParquetFile(m.historical_train_file(ROOT,legacy.REPO)).iter_batches(columns=['sample_id','sequence']):
        for row in batch.to_pylist():
            seq=row['sequence'];assert legacy.valid(seq)
            historical[seq]=min(str(row['sample_id']),historical.get(seq,str(row['sample_id'])))
    assert len(historical)==87_930
    state.setdefault('historical_matches',{})
    raw_stats,valid_stats=legacy.Stats(state['raw_stats']),legacy.Stats(state['valid_stats'])
    source=ROOT.parent/'raw/uniref50.fasta.gz'
    with source.open('rb',buffering=8*1024**2) as rf,source.open('rb',buffering=8*1024**2) as vf,gzip.GzipFile(fileobj=rf) as reader,gzip.GzipFile(fileobj=vf) as verifier:
        # Full string verification includes discarded historical representatives.
        def seq_at(h,off):
            row=base.execute('SELECT seq FROM candidates WHERE h=? AND off=?',(h,off)).fetchone()
            return row[0] if row else next(legacy.fasta_records(verifier,off))[2]
        membership=m.ExactMembership(base,db,bloom,seq_at)
        payload=None;generation=None
        def begin():
            nonlocal payload,generation
            # Attempt-specific directory: incomplete outputs can never overwrite a committed generation.
            generation=work/'generations'/f'{state["raw"]}_{time.time_ns()}'
            generation.mkdir(parents=True);payload=(generation/'candidates.jsonl').open('w',encoding='utf-8')
        def checkpoint(off):
            nonlocal payload
            payload.flush();os.fsync(payload.fileno());payload.close();payload=None
            membership.commit_hashes();pool.select_interval()
            pool.write_checkpoint(generation/'pool',inputs)
            state.update(offset=off,reservoir_count=pool.base_count+len(pool.delta),raw_stats=raw_stats.checkpoint(),valid_stats=valid_stats.checkpoint(),pool_root=str(generation/'pool'))
            outputs=[generation/'candidates.jsonl',generation/'pool/delta_priority.bin',generation/'pool/pool.json']
            # Prior generations were verified on resume. Rehash only new immutable files.
            new=m.commit_continuation(db,work,state,inputs,outputs,previous_outputs=sealed_outputs)
            sealed_outputs.update(new['outputs'])
            progress.emit(state['raw'],state['canonical']-starting_canonical,state['reservoir_count'],membership.queries,force=True)
            assert shutil.disk_usage(work).free>=10_000_000_000,'D: free-space guard; durable checkpoint retained'
        starting_canonical=state['canonical']
        if state['stage']=='external':
            # Bounded sequential gzip resume with setup telemetry; never replay 30M->31M.
            for position in range(512*1024**2,state['offset'],512*1024**2):
                reader.seek(position);progress.emit(state['raw'],0,pool.base_count+len(pool.delta),0)
            reader.seek(state['offset']);progress.begin_processing()
            begin()
            for off,header,seq,malformed in legacy.fasta_records(reader,state['offset']):
                if off==state.get('skip_completed_offset'):continue
                state['raw']+=1;raw_stats.add(seq)
                if malformed:state['malformed']+=1
                elif 20<=len(seq)<=500:
                    state['length_valid']+=1
                    if set(seq)<=m.AA:
                        state['canonical']+=1;valid_stats.add(seq);sid=header.split()[0]
                        if seq in historical:
                            match=state['historical_matches'].setdefault(seq,[off,sid,0]);match[1]=min(match[1],sid);match[2]+=1
                        h=m.sequence_hash(seq);prior=membership.find(h,seq)
                        if prior is not None:
                            state['duplicates']+=1
                            db.execute('INSERT INTO updates VALUES(?,?,1,?) ON CONFLICT(h,off) DO UPDATE SET multiplicity=multiplicity+1,source_id=min(source_id,excluded.source_id)',(h,prior,sid))
                        else:
                            state['unique']+=1;membership.add(h,off,seq);p=m.priority(seq)
                            # Preserve V2B admission on priority alone, including theoretical ties.
                            if pool.cutoff is None or p<=pool.cutoff[0]:
                                position=payload.tell();payload.write(json.dumps({'h':h.hex(),'off':off,'p':p.hex(),'seq':seq,'source_id':sid,'multiplicity':1,'historical':0,'external':1})+'\n')
                                pool.add(p,h,off,position,len(seq))
                progress.emit(state['raw'],state['canonical']-starting_canonical,pool.base_count+len(pool.delta)+len(pool.pending),membership.queries)
                if state['raw']%1_000_000==0:
                    state['skip_completed_offset']=off;checkpoint(off);begin()
            assert state['raw']==m.RAW_COUNT
            state['stage']='historical';state.pop('skip_completed_offset',None);checkpoint(reader.tell())
        # This command deliberately leaves historical union materialization to a separate
        # validated streaming stage. All historical match ownership/statistics are preserved.
        m.atomic_json(work/'EXTERNAL_TAIL_COMPLETE.json',{'state':state,'checkpoint_sha256':m.sha_file(work/'continuation_checkpoint.json'),
            'next_stage':'historical_union_then_mandatory_protected_screen','training_launched':False,'corpus_ready':False})
    pool.close();base.close();db.close()

if __name__=='__main__':run()
