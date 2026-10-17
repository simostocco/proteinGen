import pytest
from partial_report import paired_delta,selected_update

def test_paired_likelihood_comparison_preserves_opposing_components():
 a={'per_identity_AA_geometry_mean':{'a':[2.,-1.],'b':[3.,0.]}};b={'per_identity_AA_geometry_mean':{'a':[1.,0.],'b':[2.,1.]}}
 result=paired_delta(a,b);assert result['right_minus_left_AA_geometry_objective']==pytest.approx([-1.,1.,0.]);assert result['identity_bootstrap_95pct']==[[-1.,1.,0.],[-1.,1.,0.]]
 with pytest.raises(ValueError,match='panels differ'):paired_delta(a,{'per_identity_AA_geometry_mean':{'a':[1.,0.]}})

def test_original_parent_is_additional_update_zero(tmp_path):
 assert selected_update(tmp_path,'J1-FULL',{'selected_checkpoint':str(tmp_path/'old/J1/checkpoint_10000.pt')})==0
 assert selected_update(tmp_path,'J1-FULL',{'selected_checkpoint':str(tmp_path/'runs/J1-FULL/checkpoint_05000.pt')})==5000

def test_full_report_publication_uses_supplied_results(tmp_path):
 import json
 from partial_report import publish
 arms=['J1-FULL','J1-PARTIAL','J2-FULL','J2-PARTIAL'];done={};valid={};generated={}
 for arm in arms:
  done[arm]={'selected_checkpoint':str(tmp_path/'runs'/arm/'checkpoint_20000.pt'),'selected_sha256':'synthetic','additional_updates':20000,'additional_prefix_decisions':1280000}
  v={'AA_NLL':2.,'geometry_valid_component_mean_per_decision':-1.,'selection_objective':1.,'per_identity_AA_geometry_mean':{'synthetic-id':[2.,-1.]}}
  valid[arm]={'0':{'complete':v,'partial':v},'20000':{'complete':v,'partial':v}}
  g={k:0.1 for k in ['clashes_per100_residues','clashes_per10000_nonlocal_pairs','long_contact_fraction','Rg_A','Rg_over_N_cuberoot','degenerate_frames','torsion_positive_fraction','signed_triple_product_mean']};g.update(adjacent_A={'mean':3.8},angle_rad={'mean':1.9})
  generated[arm]=[{'N':64,'status':'COMPLETE','sequence':'A'*64,'seconds':1.,'independent_CA_metrics':g}]
 native={s:{'64':{'selected_identities':1,'available_identities':1,'length_window':[58,70],'rows':[{'metrics':g}]}} for s in ['TRAIN','VALIDATION']}
 (tmp_path/'NATIVE_VS_GENERATED_GEOMETRY.json').write_text(json.dumps({'native':native}))
 counts={s:{'identities':1,'additional_to_complete_identities':1,'observations':1,'distance_targets':18,'angle_targets':17,'torsion_targets':16,'AA_at_missing_CA':1} for s in ['TRAIN','VALIDATION']}
 (tmp_path/'PARTIAL_PDB_ELIGIBILITY_COUNTS.json').write_text(json.dumps({'splits':counts}));(tmp_path/'view').mkdir();(tmp_path/'view/PARTIAL_VIEW_QA.json').write_text('{}')
 (tmp_path/'configs').mkdir();(tmp_path/'configs/VALIDATION_PANELS.json').write_text(json.dumps({'complete':[['synthetic-id','observation',3]]}))
 publish(tmp_path,done,valid,generated)
 study=json.loads((tmp_path/'STUDY_COMPARISON.json').read_text());assert study['TEST_used'] is False;assert study['selected']['J1-PARTIAL']['update']==20000
 assert study['selected_paired_comparisons']['J1_PARTIAL_MINUS_FULL']['complete']['right_minus_left_AA_geometry_objective']==[0.,0.,0.]
 assert study['partial_validation_identity_strata']['J1-PARTIAL']['20000']['COMPLETE_VIEW_IDENTITY_ADDITIONAL_OBSERVATION']['identities']==1
 assert (tmp_path/'FINAL_PARTIAL_SUPERVISION_REPORT.md').exists() and (tmp_path/'FREE_GENERATION_COMPARISON.md').exists()


def test_operational_resource_samples_distinguish_device_and_process(tmp_path):
 from partial_report import resource_samples
 import json
 p=tmp_path/'telemetry.jsonl'
 rows=[{'D_free_bytes':100,'memory':{'MemAvailable':80,'SwapTotal':40,'SwapFree':40},'GPU':'100, 7000, 90, 50, 80'},{'D_free_bytes':90,'memory':{'MemAvailable':70,'SwapTotal':40,'SwapFree':35},'GPU':'200, 6900, 50, 51, 90'}]
 p.write_text('\n'.join(json.dumps(v) for v in rows))
 v=resource_samples([p])
 assert v['samples']==2 and v['device_wide_used_MiB_peak']==200
 assert v['WSL_available_bytes_min']==70 and v['swap_used_bytes_peak']==5
 assert 'not isolated process' in v['scope']
 assert resource_samples([])['samples']==0


def test_schedule_receipts_detect_unpaired_samples_and_equal_exposure():
 from partial_report import sampling_evidence
 def row(u,count,order,stream):return {'update':u,'branch_decisions':{'complete':count},'sample_order_sha256':order,'complete_stream_sha256':stream,'partial_stream_sha256':'partial'}
 c={a:[] for a in ['J1-FULL','J1-PARTIAL','J2-FULL','J2-PARTIAL']}
 for n in ['J1','J2']:
  c[n+'-FULL']=[row(5000,320000,'full-order','same-prefix')]
  c[n+'-PARTIAL']=[row(10000,320000,'partial-order','same-prefix')]
 v=sampling_evidence(c)
 assert v['paired_chemistry_schedules']['FULL']['all_sample_order_hashes_equal']
 assert v['equal_complete_exposure']['J1'][0]['counts_equal']
 assert v['equal_complete_exposure']['J1'][0]['complete_stream_hashes_equal']
 c['J2-FULL'][0]['sample_order_sha256']='different'
 assert not sampling_evidence(c)['paired_chemistry_schedules']['FULL']['all_sample_order_hashes_equal']


def test_partial_validation_separates_new_identity_from_extra_observation():
 from partial_report import partial_identity_strata
 v=partial_identity_strata({'known':[2.0,-1.0],'new1':[3.0,-.5],'new2':[1.0,-.5]}, {'known'})
 assert v['COMPLETE_VIEW_IDENTITY_ADDITIONAL_OBSERVATION']['identities']==1
 assert v['NEW_PARTIAL_ONLY_IDENTITY']['identities']==2
 assert v['NEW_PARTIAL_ONLY_IDENTITY']['AA_NLL']==2.0
 assert v['NEW_PARTIAL_ONLY_IDENTITY']['objective']==1.5
 assert partial_identity_strata({},set())['NEW_PARTIAL_ONLY_IDENTITY']['AA_NLL'] is None
