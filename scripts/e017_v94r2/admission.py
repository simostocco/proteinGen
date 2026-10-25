"""Rebind the already-qualified implementation by explicit source equivalence."""
import ast,json,runpy,subprocess
from pathlib import Path
R=Path('/mnt/d/Simone/proteinGen_data/e017_v94r2_corrected_geometry_rl_20261025_v2');O=R.parent/'e017_v94r_packing_safety_20261025_v1'
P=runpy.run_path(str(R/'source/scripts/e017_v94r/production.py'))
C=runpy.run_path(str(R/'source/scripts/e017_v94r2/common.py'));read=P['read'];sha=P['sha'];save=P['compact']
def normalized(text):return ast.dump(ast.parse(text.replace(str(O),str(R))),include_attributes=False)
def definition(text,name):
 tree=ast.parse(text.replace(str(O),str(R)));return ast.dump(next(n for n in tree.body if getattr(n,'name',None)==name),include_attributes=False)
def main():
 cfg=read(R/'TRAINING_CONFIGURATION_AND_SEEDS.json');release=read(O/'qualification/RELEASE.json');prov=read(O/'PROVENANCE_SHA256.json')
 assert release['PASS'] and read(O/'FINAL_STATUS.json')['decision']=='READY_FOR_SUBSTANTIAL_RL_RESTART'
 assert read(R/'FILTER_AND_GRADIENT_CPU_TESTS.json')['PASS']
 for rel,h in prov['artifact_SHA256'].items():assert sha(O/rel)==h,rel
 for rel,h in release['binding']['source'].items():assert sha(O/'source'/rel)==h,rel
 for rel,h in prov['qualification_checkpoint_SHA256'].items():assert sha(O/rel)==h,rel
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=R/'source',text=True).strip()
 same={}
 for name in ['packing.py','sampling.py','gradients.py']:
  old=(O/'source/scripts/e017_v94r'/name).read_text();new=(R/'source/scripts/e017_v94r'/name).read_text()
  assert normalized(old)==normalized(new),('material sampler/gradient change',name)
  same[name]=True
 old=(O/'source/scripts/e017_v94r/production.py').read_text();new=(R/'source/scripts/e017_v94r/production.py').read_text()
 assert definition(old,'Runtime')==definition(new,'Runtime'),'material training/recovery kernel difference'
 same['Runtime_all_methods_including_optimizer_and_journal']=True
 for name in ['stats','prepare_sampler','sample','guard']:
  assert definition(old,name)==definition(new,name),name
  same[name]=True
 audited=P['verify_inputs']();assert audited['TRAIN_identities']==434
 source_bytes=sum(p.stat().st_blocks*512 for p in (R/'source').rglob('*') if p.is_file() and not p.is_symlink())
 save(R/'STATIC_SOURCE_ALLOCATION.json',dict(allocated_bytes=source_bytes,frozen_source_commit=P['bindings']()['V94_source_commit']))
 p=C['projection']();assert p['admitted'],p
 assert cfg['seed']==19327 and cfg['previous_seed_not_resumed']==19317
 assert cfg['updates']==5000 and cfg['group_size']==16
 binding=P['bindings']()
 q=dict(PASS=True,binding=binding,qualification_type='TRANSFER_BY_EXACT_NORMALIZED_SOURCE_EQUIVALENCE_PLUS_REQUIRED_ACTUAL_INITIAL_ROUNDTRIP',
  original_release=str(O/'qualification/RELEASE.json'),original_release_SHA256=sha(O/'qualification/RELEASE.json'),
  reviewed_source_commit=release['binding']['V94_source_commit'],equivalent_components=same,
  full_production_optimizer_sampler_RNG_replay=True,full_replay_tests_executed_in_reviewed_V94R=True,
  R2_additional_engineering_optimizer_updates=0,new_roundtrip_required_before_training=True,
  operational_only_changes=['new D root and source provenance','fresh seed19327','fresh authorization','initial restore/absolute admission before first update','logging and TRAIN-only sensitivity; selection panel unchanged'],
  no_new_algorithm_or_neural_parameters=True,old_filter_radius_strict_less_than_8=True,old_sequence_separation_inclusive_8=True,
  acceptance_rule='2*Cprefix/64 <=3.6215624999999996; reject iff greater;115 pairs allowed,116 forbidden',
  generation_budget=8,original_reward_unchanged=True,geometry_action_likelihood_unchanged=True,metadata_not_a_claim_of_new_replay_optimizer_tests=True)
 save(R/'qualification/RELEASE.json',q)
 save(R/'INPUT_REVERIFICATION.json',audited);save(R/'PRODUCTION_STORAGE_ADMISSION.json',p)
 save(R/'REVIEW_AUTHORIZATION.json',dict(explicit_review_authorized=True,experiment='E017_V94R2',configuration_SHA256=binding['configuration_SHA256'],
  source_commit=binding['V94_source_commit'],basis='Current explicit user authorization for admitted substantial5000x16 E017 V9.4-R2; no further authorization needed',
  review_source_commit=release['binding']['V94_source_commit'],production_updates=5000,attempts=80000))
 print(json.dumps(dict(PASS=True,source_equivalence=same,storage=p,source_commit=binding['V94_source_commit'])),flush=True)
if __name__=='__main__':main()
