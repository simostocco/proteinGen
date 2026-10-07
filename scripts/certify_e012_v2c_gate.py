"""Frozen nine-gate certification. Emit explicit V2C-GO/NOGO, all artifacts on D:."""
from pathlib import Path
import importlib.util,json,os,shutil,subprocess,sys,time
spec=importlib.util.spec_from_file_location('v2c',Path(__file__).with_name('e012_reservoir_v2c.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ROOT=Path('D:/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization');m.configure_storage(ROOT)

def main():
    verdict_path=ROOT/'V2C_VERDICT.json'
    if verdict_path.exists():
        saved=json.loads(verdict_path.read_text());print(json.dumps(saved),flush=True)
        if saved['verdict']!='V2C-GO':raise SystemExit(1)
        assert saved['report_sha256']==m.sha_file(ROOT/'reports/BENCHMARK_EQUIVALENCE.json')
        assert saved['restart_report_sha256']==m.sha_file(ROOT/'reports/RESTART_REPLAY.json')
        assert saved['tests_sha256']==m.sha_file(ROOT/'reports/TESTS.json')
        return
    gates={str(i):False for i in range(1,10)};evidence={}
    try:
        with (ROOT/'logs/gate_comparison.log').open('ab') as log:
            subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('report_benchmarks_e012_v2c.py'))],stdout=log,stderr=log,check=True)
        report=json.loads((ROOT/'reports/BENCHMARK_EQUIVALENCE.json').read_text());assert report['all_required_benchmarks_passed']
        restart=json.loads((ROOT/'reports/RESTART_REPLAY.json').read_text());assert restart['passed'] and restart['records']==1_000_000
        a=json.loads((ROOT/'audit/authoritative_state.json').read_text())
        assert (a['raw'],a['offset'],a['stage'],a['unique'])==(31_000_000,15_459_089_980,'external',24_715_247)
        assert restart['frozen_replay_sha256']==m.sha_file(ROOT/'replays/replay_1000000.parquet')
        gates['1']=True
        case=next(c for c in report['cases'] if c['records']==1_000_000)
        gates['2']=restart['exact_selected_keys_match_f52'] and case['all_equivalence_fields_match']
        gates['3']=restart['all_payload_fields_equal'] and case['all_equivalence_fields_match']
        gates['4']=all(restart['state_fields_equal'].values()) and case['resume_checkpoint_and_delta_reseed_verified']
        assert m.sha_file(ROOT.parent/'protected/protected.fasta')=='1f6df25e70e90d41bfb6a86268372d652f9ddccff15d4b0bfaf00899dca419ae'
        gates['5']=restart['historical_bookkeeping_equal_to_v2b']
        adapter_spec=importlib.util.spec_from_file_location('adapter',Path(__file__).with_name('complete_e012_v2c_pipeline.py'))
        adapter=importlib.util.module_from_spec(adapter_spec);adapter_spec.loader.exec_module(adapter)
        legacy,builder=adapter.build()
        old_builder=legacy.Build.__new__(legacy.Build);old_builder.root=ROOT.parent;old_builder.protected=ROOT.parent/'protected/protected.fasta'
        args=[ROOT/'tmp/policy_query.fasta',ROOT/'tmp/policy_hits.tsv',ROOT/'tmp/policy_search']
        assert list(map(str,legacy.Build.search_command(old_builder,*args)))==list(map(str,builder.search_command(*args))),'Historical MMseqs policy changed'
        gates['6']=restart['post_commit_pre_mirror_recovery'] and restart['uncommitted_interval_recovery']
        tests=json.loads((ROOT/'reports/TESTS.json').read_text());assert tests['passed'] and tests['tests']>=10
        assert tests['core_source_sha256']==m.sha_file(Path(__file__).with_name('e012_reservoir_v2c.py'))
        gates['7']=True
        exported=json.loads((ROOT/'exports/export.complete.json').read_text())
        for name,metadata in exported['outputs'].items():assert m.sha_file(ROOT/'exports'/name)==metadata['sha256'],name
        sorted_meta=json.loads((ROOT/'exports/base_priority_sorted.complete.json').read_text())
        assert m.sha_file(ROOT/'exports/base_priority_sorted.bin')==sorted_meta['output_sha256']
        assert m.sha_file(ROOT.parent/'raw/uniref50.fasta.gz')=='95bdf597d2f892295b5aa2925f5da593e1f2e1fcd7a0a4f5e66b9f1c2e83290d'
        times=json.loads((ROOT/'reports/replay_process_times.json').read_text())
        baseline_start=next(x['created'] for x in times if 'v2b' in x['argv'] and '1000000' in x['argv'])
        # Conservative upper bound includes the aborted cold-WAL start, redesign/sort
        # interval, and complete successful 1M replay. This cannot inflate speedup.
        optimized_start=next(x['created'] for x in times if 'uninterrupted' in x['argv'])
        b_seconds=(ROOT/'benchmarks/v2b_1000000/result.json').stat().st_mtime-baseline_start
        c_seconds=restart['uninterrupted_end_mtime']-optimized_start
        assert b_seconds>0 and c_seconds>0
        speedup=b_seconds/c_seconds;gates['8']=speedup>=5
        peaks=[]
        for path in [ROOT/'replay_restart/uninterrupted/progress.jsonl',ROOT/'replay_restart/resumed/progress.jsonl']:
            peaks.extend(json.loads(line)['peak_rss_bytes'] for line in path.read_text().splitlines())
        reclaim=sum(p.stat().st_size for n in [100_000,1_000_000] for p in (ROOT/'benchmarks'/f'v2b_{n}/baseline').glob('reservoir.sqlite*'))
        free=shutil.disk_usage(ROOT).free;after_cleanup=free+reclaim;planned=40_000_000_000
        old=json.loads((ROOT.parent/'stats/resource_plan.json').read_text());floor=old['starting_free_bytes']*3//10
        gates['9']=max(peaks)<=4*1024**3 and planned<=.7*after_cleanup and after_cleanup-planned>=floor
        evidence={'baseline_end_to_end_seconds':b_seconds,'v2c_end_to_end_conservative_upper_bound_seconds':c_seconds,
            'speedup_lower_bound':speedup,'e2e_method':'V2B process birth through result publication (including DB close). V2C earliest cold-start attempt through successful replay publication, including failed WAL setup and compact-index preparation.',
            'v2c_peak_rss_bytes':max(peaks),'d_free_bytes':free,'certified_private_clone_reclaim_bytes':reclaim,
            'planned_incremental_pipeline_peak_bytes':planned,'existing_repository_free_floor_bytes':floor,
            'original_v2b_artifacts_reverified':report['original_files_reverified'],'source_baseline_commit':'f52bdd3'}
        assert all(gates.values()),{'failed_gates':[k for k,v in gates.items() if not v],'evidence':evidence}
        verdict={'verdict':'V2C-GO','gates':gates,'evidence':evidence,'report_sha256':m.sha_file(ROOT/'reports/BENCHMARK_EQUIVALENCE.json'),
            'restart_report_sha256':m.sha_file(ROOT/'reports/RESTART_REPLAY.json'),'tests_sha256':m.sha_file(ROOT/'reports/TESTS.json'),
            'training_launched':False,'timestamp_unix':time.time()}
    except BaseException as exc:
        verdict={'verdict':'V2C-NOGO','gates':gates,'evidence':evidence,'discrepancy':repr(exc),'training_launched':False,'timestamp_unix':time.time()}
    m.atomic_json(ROOT/'V2C_VERDICT.json',verdict);print(json.dumps(verdict),flush=True)
    if verdict['verdict']!='V2C-GO':raise SystemExit(1)

if __name__=='__main__':main()
