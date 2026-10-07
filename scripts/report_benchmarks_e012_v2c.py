"""Certify isolated replay equivalence/speedup; all reports and gate stay on D:."""
from pathlib import Path
from contextlib import closing
import importlib.util,json,sqlite3

ROOT=Path('D:/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization')
spec=importlib.util.spec_from_file_location('v2c',Path(__file__).with_name('e012_reservoir_v2c.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.configure_storage(ROOT)
COMPARE=['records','canonical','novel','duplicates','accepted_before_trim','selected_count',
    'selected_base_prefix','selected_delta_prefix','selected_cutoff','exact_rejected_keyset_sha256',
    'per_record_changes_sha256','per_record_priority_decisions_sha256','multiplicity_update_events']

def main():
    cases=[]
    for n in [100_000,1_000_000]:
        bp=ROOT/'benchmarks'/f'v2b_{n}/result.json';op=ROOT/'benchmarks'/f'v2c_{n}/result.json'
        if not bp.exists() or not op.exists():continue
        b=json.loads(bp.read_text());o=json.loads(op.read_text())
        matches={k:b[k]==o[k] for k in COMPARE}
        assert all(matches.values()),{k:(b[k],o[k]) for k,v in matches.items() if not v}
        working=op.parent
        with closing(m.open_delta(working/'delta.sqlite')) as db:
            checkpoint=m.resume_continuation(db,{'baseline_raw':31_000_000,'seed':12014,'records':n})
            assert checkpoint['state']==o['checkpoint']
            bloom=m.Bloom();m.seed_committed_delta(bloom,db)
            for h, in db.execute('SELECT h FROM exact_hashes'):
                assert bloom.possible(h),'Resume Bloom false negative'
        baseline_total=b['processing_seconds']+b['checkpoint_seconds']
        optimized_total=o['processing_seconds']+o['checkpoint_seconds']
        case={'records':n,'all_equivalence_fields_match':True,'fields':matches,
            'v2b_result_sha256':m.sha_file(bp),'v2c_result_sha256':m.sha_file(op),
            'resume_checkpoint_and_delta_reseed_verified':True,'v2b_seconds':baseline_total,
            'v2c_seconds':optimized_total,'speedup_including_checkpoint':baseline_total/optimized_total,
            'processing_speedup':b['processing_seconds']/o['processing_seconds'],
            'v2b_setup_seconds':b['setup_seconds'],'v2c_setup_seconds':o['setup_seconds'],
            'v2b_metrics':json.loads((bp.parent/'progress.jsonl').read_text().splitlines()[-1]),
            'v2c_metrics':json.loads((op.parent/'progress.jsonl').read_text().splitlines()[-1]),
            'training_launched':False}
        cases.append(case)
    passed=len(cases)==2 and all(c['speedup_including_checkpoint']>=5 and c['processing_speedup']>=5 for c in cases)
    originals=[]
    if passed:
        for item in json.loads((ROOT/'preservation/FROZEN_FILES.json').read_text())['files']:
            path=m.require_d(item['original'])
            assert path.stat().st_size==item['bytes'] and m.sha_file(path)==item['sha256'],'Original V2B artifact changed'
            originals.append({'path':str(path),'sha256':item['sha256'],'unchanged':True})
    report={'cases':cases,'all_required_benchmarks_passed':passed,'minimum_speedup':5,'original_files_reverified':originals,
        'original_v2b_artifacts_modified':False,'scientific_continuation_launched':False,'training_launched':False}
    m.atomic_json(ROOT/'reports/BENCHMARK_EQUIVALENCE.json',report)
    if passed:
        manifest=json.loads((ROOT/'continuation_manifest.json').read_text())
        manifest['gate_passed']=True
        manifest['benchmark_report_sha256']=m.sha_file(ROOT/'reports/BENCHMARK_EQUIVALENCE.json')
        m.atomic_json(ROOT/'continuation_manifest.json',manifest)
        m.atomic_json(ROOT/'BENCHMARK_GATE_PASSED.json',{'report_sha256':manifest['benchmark_report_sha256'],
            'baseline_raw':31_000_000,'seed':12014,'minimum_speedup':5,'training_launched':False})
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
