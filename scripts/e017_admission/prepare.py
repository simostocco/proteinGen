"""Freeze identity panels and launch configuration; no model inference or updates."""
from pathlib import Path
from collections import Counter
import copy,json,os,resource,shutil,subprocess,time
from protein_joint_sac.crisp_v3.data import Population
from protein_joint_sac.crisp_v3.admission import build_panels,verify_panel,BANDS,band
from protein_joint_sac.crisp_v3.prior import file_sha,SHA
from protein_joint_sac.crisp_v3.storage import digest
R=Path(__file__).resolve().parents[3];D=R.parent;old=D/'e017_crisp_v3_20261021_v1/storage_policy_20261021_v1';start=time.monotonic();free0=shutil.disk_usage(R).free
p=Population();limits=json.load(open(D/'e017_crisp_v3_20261021_v1/configs/NATIVE_CALIBRATION.json'));panels=build_panels(p,limits)
checks={}
for split,panel in panels.items():
 c=verify_panel(p,panel,split);c['cells']={'/'.join(k):v for k,v in c['cells'].items()};checks[split]=c
 target=R/'panels'/('FROZEN_VALIDATION_MANIFEST.json' if split=='VALIDATION' else 'MATCHED_TRAIN_DIAGNOSTIC_MANIFEST.json')
 with open(target,'x') as f:json.dump(panel,f,separators=(',',':'))
a={r['identity'] for r in panels['TRAIN']['records']};b={r['identity'] for r in panels['VALIDATION']['records']};assert not a&b
# Existing split's homology/ownership provenance is authoritative. No sequence re-screening or TEST readout.
pg={s:{r['protected_group_id'] for r in panel['records'] if r['protected_group_id'] is not None} for s,panel in panels.items()};assert not pg['TRAIN']&pg['VALIDATION']
membership=p.membership();hist={s:{cat:dict(Counter(band(len(p.base.seqs[h])) for h in ids)) for cat,ids in groups.items()} for s,groups in p.groups.items()}
goal_pool={str(upper):sum(64<=len(p.base.seqs[h])<=upper for h in p.groups['TRAIN']['complete']) for upper in [96,128,256]}
config=json.load(open(old/'configs/research_supervised.json'));config.update(schema='crisp_v3_initialization_admission_v1',admission_protocol='crisp_actor_initialization_20261021_v1',stage='supervised',updates=10000,seed=1703,validation_panel=str(R/'panels/FROZEN_VALIDATION_MANIFEST.json'),matched_TRAIN_panel=str(R/'panels/MATCHED_TRAIN_DIAGNOSTIC_MANIFEST.json'),validation_milestones=[250,1000,2500,5000,10000],validation_every=100000000,checkpoint_every=250,checkpoint_wall_seconds=1800,goal_stream_scale=.25,schedule=dict(warmup=250,total_updates=10000,floor_ratio=.1),early_stop=dict(schema='crisp_initialization_monitor_v1',patience=2,AA_regression_nats=.05,gap_growth_nats=.10,goal_loss_regression=.10,component_regression_nats=.20,completion_floor=.90,futility_after=5000),stream_weights=dict(AA=dict(aa=1.,geometry=.05,kl_aa=.02,kl_geometry=.01,goal=0.),complete_geometry=dict(aa=0.,geometry=.1,kl_aa=0.,kl_geometry=.01,goal=0.),goal=dict(aa=1.,geometry=.1,kl_aa=.02,kl_geometry=.01,goal=.05)))
# Geometry in AA mix uses only actually valid components; AA-only rows have no geometry signal.
config['optimizer']['lr']=5e-5
config['storage_policy'].update(checkpoint_hint_bytes=700*2**20,near_term_growth_bytes=32*2**20,expected_live_replay_bytes=0,expected_log_bytes=64*2**20)
config['initialization_scope']=dict(updated=['actor'],frozen=['J1','q1','q2','high_actor','high_q1','high_q2','reach','all_target_networks'],system_capacity=63389056,actor_parameters=17438114)
config['frozen_artifact_hashes']={str(path):file_sha(path) for path in [Path(config['validation_panel']),Path(config['matched_TRAIN_panel']),Path(config['native_limits'])]}
with open(R/'configs/SEALED_RESEARCH_INITIALIZATION.json','x') as f:json.dump(config,f,indent=2)
# In-source copies allow reproduction without changing the source-bound scientific content.
for name in ['FROZEN_VALIDATION_MANIFEST.json','MATCHED_TRAIN_DIAGNOSTIC_MANIFEST.json']:
 dest=R/'source/configs/e017_v3_admission'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy(R/'panels'/name,dest)
shutil.copy(R/'configs/SEALED_RESEARCH_INITIALIZATION.json',dest.parent/'research_initialization.json')
witness={s:dict(Counter((r['length_band'],r['native_witness_valid']) for r in panel['tasks'])) for s,panel in panels.items()}
witness={s:{str(k):v for k,v in c.items()} for s,c in witness.items()}
expected={}
for n in [250,1000,2500,5000,10000]:
 allocation=dict(complete=int(64*n*.4),partial=int(64*n*.3),aa_only=int(64*n*.3));expected[n]=dict(AA_decisions=64*n,categories=allocation,unique_identities_by_AA_stream={c:min(v,membership['TRAIN']['identities'][c]) for c,v in allocation.items()},qualified_complete_geometry_decisions=32*n,complete_geometry_max_component_targets=96*n,goal_demonstrations=8*n,goal_kind_counts={k:(8*n)//3+int(i<(8*n)%3) for i,k in enumerate(['P0','P1','P2'])},AA_mix_component_target_counts='measured at runtime; prefix masks/partial continuity prohibit a priori exact actual counts',unique_residue_targets='tracked by shuffled per-identity position cycles; repeats only after exhausting that identity positions')
result=dict(base_commit='4374c4a766a088da4daf0b475a26357408876a8c',membership=membership,panel_checks=checks,length_population_counts=hist,TRAIN_goal_pool_metadata_counts=goal_pool,native_witness_validity=witness,expected_exposures=expected,config_sha256=digest(config),panel_sha256={s:file_sha(R/'panels'/('FROZEN_VALIDATION_MANIFEST.json' if s=='VALIDATION' else 'MATCHED_TRAIN_DIAGNOSTIC_MANIFEST.json')) for s in panels},source_inputs=p.input_hashes(),no_TEST_evaluation=True,TRAIN_VALIDATION_identity_overlap=0,TRAIN_VALIDATION_protected_group_overlap=0,optimizer_updates=0,model_forwards=0,seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,free_before_bytes=free0,free_after_bytes=shutil.disk_usage(R).free)
with open(R/'results/DATA_PANEL_AND_SAMPLING_PREFLIGHT.json','x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:result[k] for k in ['panel_checks','TRAIN_goal_pool_metadata_counts','native_witness_validity','seconds','peak_RSS_bytes','free_after_bytes']},indent=2),flush=True)
