from core97 import *
import shutil,collections,subprocess
import numpy as np
assert not ROOT.exists();assert WT.is_dir();ROOT.mkdir();(ROOT/'runtime/tmp').mkdir(parents=True);(ROOT/'runtime/cache').mkdir();(ROOT/'control').mkdir();(WT/'e017_v97').mkdir(exist_ok=False)
for p in Path('/mnt/d/Simone/proteinGen/e017_v97').glob('*.py'):shutil.copyfile(p,WT/'e017_v97'/p.name)
script=(PAIR/'run.sh').read_text().replace(str(PAIR),str(ROOT));(ROOT/'run.sh').write_text(script);(ROOT/'run.sh').chmod(0o755)
original=read(PAIR/'FROZEN_SOURCE_AND_INPUTS.json')
for group in ['files','historical_models']:
 for p,h in original[group].items():assert sha(p)==h,p
pool=read(ROOT.parent/'e017_v93_regional_controllability_20261024_v1/manifests/FUTURE_FIXED_TRAIN_POOL.json')['tasks'];byid=collections.defaultdict(list)
for task in pool:
 assert task['native_witness_success'] and not task['goal']['native_coordinates_provided'] and len(task['fixed_sequence'])==64
 byid[task['source_reference_identity']].append(task)
assert len(byid)==434 and all(len(v)==2 for v in byid.values())
assert len({v[0]['goal']['protected_group'] for v in byid.values()})==434
ordered=sorted(byid,key=lambda x:seed('MONITOR_POPULATION',x));monitor_ids=set(ordered[:128]);train_ids=set(ordered[128:]);mon=[t for t in pool if t['source_reference_identity'] in monitor_ids];fit=[t for t in pool if t['source_reference_identity'] in train_ids]
assert len(mon)==256 and len(fit)==612 and not monitor_ids&train_ids
save('MONITOR_POPULATION.json',dict(identities=128,protected_groups=128,tasks=mon,selection='hash ordering of all434 qualified TRAIN identities with V97 namespace; no outcome selection',target='uniform distribution over256 unchanged identity/goal pairs',TRAIN=True))
save('TRAINING_POPULATION.json',dict(identities=306,protected_groups=306,tasks=fit,common_reallocation_from_historical434=True,TRAIN=True))
save('qualified_train/manifests/FUTURE_FIXED_TRAIN_POOL.json',dict(tasks=fit))
# Old16 reserved TRAIN goals are retained only for descriptive K1/K8, KL and original diversity safeguards.
legacy=read(ROOT.parent/'e017_v93_regional_controllability_20261024_v1/manifests/TRAIN_TASKS.json');assert not {t['source_reference_identity'] for t in legacy['tasks']}&set(byid)
save('qualified_train/manifests/TRAIN_TASKS.json',legacy)
from safety97 import N,LOOKS,ALPHA
looks={}
for u in LOOKS:
 rng=np.random.default_rng(seed('MONITOR_IDENTITY_GOAL',u));indices=rng.integers(len(mon),size=N)
 looks[str(u)]=[dict(index=i,task_index=int(k),candidate_seed=seed('MONITOR_CANDIDATE',u,i),goal_id=mon[int(k)]['goal']['goal_id'],identity=mon[int(k)]['source_reference_identity']) for i,k in enumerate(indices)]
save('MONITOR_SCHEDULE.json',dict(looks=looks,draws_per_look=N,draws_with_replacement=True,candidate_batch=1,alpha=ALPHA,independent_rng_from_training=True))
config=read(PAIR.parent/'e017_v96_arm_a_original_20261029_v1/TRAINING_CONFIGURATION_AND_SEEDS.json');config.update(experiment='E017_V97_ENGINEERING_NO_PRODUCTION',seed=19731,TRAIN_identities=306,TRAIN_protected_groups=306,TRAIN_goal_variants=612,production_authorized=False,production_updates=0,engineering_authorized=True,burial_token_enabled=False,gate_version='TANH_SHARED_GEOMETRY_DELTA_1',safety_looks=LOOKS)
config.update(causal_limitation='single matched seed pair; controls extra training; not algorithmic robustness',initializer='certified R3 update5000 tensors; gate0 in B; fresh optimizer',next_stage_auto_launch=False)
save('CONFIGURATION.json',config);save('TRAINING_CONFIGURATION_AND_SEEDS.json',config)
for arm in ['A','B']:
 future=dict(config,arm=arm,production_authorized=False,production_updates=5000,burial_gate_enabled=arm=='B',planned_generation_attempts=80000)
 save('FUTURE_ARM_'+arm+'_CONFIGURATION.json',future)
ids=sorted(train_ids);order=np.random.default_rng(config['seed']).permutation(len(ids)).tolist();schedule=[]
for u in range(1,5001):
 identity=ids[order[(u-1)%len(ids)]];t=byid[identity][((u-1)//len(ids))%2]
 # use exact original S.ns namespace for future full-trace sampling
 import runpy
 if u==1:
  sys.path.insert(0,str(ENG/'source96'));import common96
  P,F,_=common96.load()
 schedule.append(dict(update=u,identity=identity,goal_id=t['goal']['goal_id'],seeds=[P['S']['ns']('SUBSTANTIAL',19731,u,j) for j in range(16)]))
save('TRAINING_SCHEDULE.json',dict(updates=schedule,arms_matched=True,updates_per_arm=5000,attempts_per_arm=80000))
shutil.copyfile(PAIR/'VALIDATION_PANEL_MANIFEST.json',ROOT/'VALIDATION_PANEL_MANIFEST.json')
save('FROZEN_DESIGN.json',dict(version='V97_QUALIFICATION_1',production_authorized=False,gate='tanh scalar exactly0; shared original and qualified burial branch; original AA logits; parameter interpolation not probability mixture',additional_parameters=1,baseline_actor_update=5000,TRAIN_fit=306,TRAIN_monitor=128,legacy_reserved=16,monitor_samples=N,monitor_looks=LOOKS,monitor_alpha=ALPHA,monitor_decision='PASS only lower backbone>=.95 and lower hard>=original bound and all original KL/diversity/resource safeguards; upper below bound BREACH; otherwise INCONCLUSIVE no updates; no extra draws',no_adaptive_extension=True,primary='raw burial S Bfinal-Afinal>=.01 and positive95lower; difference-in-changes>=.01 and positive95lower; B-own positive change; Bfinal>=.01 and positive95lower',bootstrap_draws=10000,bootstrap_seed=19329,matched_secondary_alpha=.05/32,K1_practical_gain=.05,VALIDATION_development_not_untouched=True,TEST=False))
files={str(p):sha(p) for p in (WT/'e017_v97').glob('*.py')};files.update({str(p):sha(p) for p in ROOT.glob('*.json')});files[str(ROOT/'run.sh')]=sha(ROOT/'run.sh');files[str(ENG/'source96/algorithm.py')]=sha(ENG/'source96/algorithm.py')
save('SOURCE_BINDING.json',dict(files=files,models=original['historical_models'],worktree_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WT,text=True).strip(),original_frozen_manifest_SHA256=sha(PAIR/'FROZEN_SOURCE_AND_INPUTS.json')))
save('PREPARATION.json',dict(PASS=True,UTC=now(),D_free_bytes=shutil.disk_usage(ROOT).free,source_binding_SHA256=sha(ROOT/'SOURCE_BINDING.json'),production_updates=0))
print('V97_PREPARED',ROOT,flush=True)
