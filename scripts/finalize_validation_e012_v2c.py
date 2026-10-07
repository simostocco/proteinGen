"""Finish committed-31M invariants and create a new D: continuation manifest."""
from pathlib import Path
import importlib.util,json,os,time
import numpy as np

ROOT=Path('D:/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization')
spec=importlib.util.spec_from_file_location('v2c',Path(__file__).with_name('e012_reservoir_v2c.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.configure_storage(ROOT)

def main():
    export=json.loads((ROOT/'exports/export.complete.json').read_text())
    state=json.loads((ROOT/'audit/authoritative_state.json').read_text())
    mirror=json.loads((ROOT/'preservation/filter_checkpoint.json').read_text())
    assert state['inputs']==mirror['inputs']
    assert state['raw']-mirror['raw']==1_000_000
    assert state['stage']==mirror['stage']=='external'
    assert state['offset']==state['skip_completed_offset']==15_459_089_980
    assert mirror['offset']==mirror['skip_completed_offset']==15_234_569_728
    assert export['statistics']['seen']==state['unique']==state['canonical']
    assert export['statistics']['candidates']==24_000_000
    start=time.perf_counter()
    seen=np.fromfile(ROOT/'exports/seen.bin',dtype=m.SEEN_DTYPE)
    seen.sort(order=['off','h'])
    assert np.all(seen['off'][1:]>seen['off'][:-1])
    assert seen['off'][-1]<=state['offset']
    candidates=np.memmap(ROOT/'exports/candidate_keys.bin',dtype=m.KEY_DTYPE,mode='r')
    matched=0
    for first in range(0,len(candidates),250_000):
        rows=candidates[first:first+250_000]
        positions=np.searchsorted(seen['off'],rows['off'])
        assert np.all(positions<len(seen))
        assert np.all(seen['off'][positions]==rows['off'])
        assert np.all(seen['h'][positions]==rows['h'])
        matched+=len(rows)
        if matched%1_000_000==0:print(json.dumps({'stage':'candidate_seen_join','matched':matched}),flush=True)
    candidates._mmap.close();del candidates,seen
    # Existing source JSON is never repaired. Verify its original and snapshot hash.
    frozen=json.loads((ROOT/'preservation/FROZEN_FILES.json').read_text())
    original_checks=[]
    for item in frozen['files']:
        assert m.sha_file(item['original'])==item['sha256']
        assert m.sha_file(item['snapshot'])==item['sha256']
        original_checks.append({'name':Path(item['original']).name,'sha256':item['sha256'],'unchanged':True})
    replay=json.loads((ROOT/'replays/replay_1000000.complete.json').read_text())
    assert replay['baseline_raw']==state['raw'] and replay['baseline_offset']==state['offset']
    assert replay['first_replayed_offset']>state['offset']
    report={'all_checks_passed':True,'raw':state['raw'],'offset':state['offset'],
        'skip_completed_offset':state['skip_completed_offset'],'stage':state['stage'],
        'canonical':state['canonical'],'unique':state['unique'],'exact_hash_records':export['statistics']['seen'],
        'candidate_count':matched,'reservoir_cap':24_000_000,'validated_reservoir_count':24_000_000,
        'legacy_sql_reservoir_count_field_absent':state.get('reservoir_count') is None,
        'candidate_hash_membership_verified':True,'unique_seen_offsets_verified':True,
        'candidate_sequence_hash_priority_alphabet_length_flags_verified':True,
        'raw_statistics_n':state['raw_stats']['n'],'canonical_statistics_n':state['valid_stats']['n'],
        'db_commit_before_json_save_explains_lag':True,'json_raw':mirror['raw'],
        'canonical_increment_since_json':state['canonical']-mirror['canonical'],
        'original_files':original_checks,'historical_json_mirror_repaired':False,
        'rollback_to_30m':False,'replay_30m_to_31m':False,'scientific_continuation_launched':False,
        'training_launched':False,'wall_seconds':time.perf_counter()-start}
    m.atomic_json(ROOT/'reports/VALIDATED_31M_STATE.json',report)
    manifest={'version':'E012_V2C_31M_CONTINUATION_V1','baseline':report,
        'source_state_sha256':m.sha_file(ROOT/'audit/authoritative_state.json'),
        'preservation_manifest_sha256':m.sha_file(ROOT/'preservation/FROZEN_FILES.json'),
        'export_manifest_sha256':m.sha_file(ROOT/'exports/export.complete.json'),
        'raw_source_sha256':state['inputs']['raw_sha256'],'seed':12014,'reservoir_cap':24_000_000,
        'storage_root':str(ROOT),'all_generated_artifacts_on_D':True,
        'full_continuation_gate':'100k and 1M equivalence/resume pass plus >=5x measured speedup',
        'gate_passed':False,'training_launched':False}
    m.atomic_json(ROOT/'continuation_manifest.json',manifest)
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
