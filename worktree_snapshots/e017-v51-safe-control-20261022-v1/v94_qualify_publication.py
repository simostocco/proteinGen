"""Fail-fast first production recovery gate, actual V93 checkpoint callback.
A complete CPU initializer/zero-Adam fixture is used; no optimizer is created or
stepped. Source is not changed. On qualification failure user requires stop.
"""
from pathlib import Path
import ast,collections,copy,hashlib,json,os,runpy,shutil,time
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v94_fixed_sequence_geometry_rl_20261024_v1');C=R/'uniref_cleanup_20261024_v1';V=R.parent/'e017_v93_regional_controllability_20261024_v1';Q=R/'recovery_qualification_20261025_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
assert read(C/'BATCH_0001_RESULT.json')['deleted_count']==1
assert shutil.disk_usage(R).free>=8*2**30,'preferred production admission8GiB not met'
assert not Q.exists();Q.mkdir();(Q/'fixture').mkdir();shutil.copyfile(__file__,Q/'qualification_source.py');torch.set_num_threads(1)
from protein_joint_sac.crisp_v3.recovery import atomic_checkpoint,atomic_json,StoragePolicy,state_digest
spec=read(V/'SUBSTANTIAL_CAMPAIGN_SPEC.json');pool=read(V/'manifests/FUTURE_FIXED_TRAIN_POOL.json');ids=sorted({t['source_reference_identity'] for t in pool['tasks']});assert len(ids)==434
index=read(R.parent/'e017_v91_ordered_goal_recovery_20261023_v1/checkpoints/protein/RECOVERY_INDEX.json')['owned']['0'];assert sha(index['path'])==index['sha256'];initial=torch.load(index['path'],map_location='cpu',weights_only=False);assert state_digest(collections.OrderedDict(initial['actor']))=='8e21d75c2fe8f78ee4fd355183f56913f975f5f3cfc38240cc5271cd60c74373'
assert all(float(v['step'])==0 for v in initial['optimizer']['state'].values())
class FixtureComponent:
 def __init__(self,state):self.state=state
 def state_dict(self):return self.state
class FixtureEngine:pass
e=FixtureEngine();e.actor=FixtureComponent(initial['actor']);opt=FixtureComponent(initial['optimizer']);schedule=FixtureComponent(initial['scheduler'])
source=V/'source/scripts/e017_v93/campaign.py';tree=ast.parse(source.read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run');callback=next(n for n in run.body if isinstance(n,ast.FunctionDef) and n.name=='checkpoint');cpu=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='cpu')
namespace=dict(torch=torch,copy=copy,os=os,shutil=shutil,atomic_checkpoint=atomic_checkpoint,atomic_json=atomic_json,e=e,opt=opt,schedule=schedule,rng=lambda:copy.deepcopy(initial['RNG']),update=0,ids=ids,order=np.random.default_rng(spec['seed']).permutation(len(ids)).tolist(),config_hash=sha(V/'SUBSTANTIAL_CAMPAIGN_SPEC.json'),source_hash=sha(source),c=dict(J1SHA=spec['J1_SHA256'],PSHA=spec['parent_SHA256']),beta=spec['trace_KL_beta'],best=None,spec=spec,root=Q/'fixture',policy=StoragePolicy(min_headroom_bytes=2*2**30,near_term_growth_bytes=128*2**20,checkpoint_hint_bytes=spec['measured_checkpoint_bytes']),R=V,sha=sha)
exec(compile(ast.Module(body=[cpu,callback],type_ignores=[]),str(source),'exec'),namespace)
start=time.perf_counter();first=namespace['checkpoint']('QUALIFICATION_INITIAL_NO_OPTIMIZER_UPDATES');assert sha(first['path'])==first['sha256'];loaded=torch.load(first['path'],map_location='cpu',weights_only=False);assert state_digest(loaded)==first['state_digest'];assert loaded['update']==0 and loaded['accepted_attempts']==0
failure=None
try:namespace['checkpoint']('QUALIFICATION_SECOND_PUBLICATION_NO_OPTIMIZER_UPDATES')
except FileExistsError as error:failure=dict(type=type(error).__name__,message=str(error),actual_production_function=str(source),checkpoint_callback_line=callback.lineno,writer='protein_joint_sac.crisp_v3.recovery.atomic_checkpoint',path=str(Q/'fixture/latest.pt'))
if not failure:raise RuntimeError('Expected candidate integration issue was not reproduced; stop for review rather than infer qualification')
result=dict(status='FAILED_PRODUCTION_CHECKPOINT_PUBLICATION',qualified=False,stop_before_training=True,first_publication_and_complete_readback=True,first_checkpoint=first,complete_initial_state_fixture=True,fixture_parent_fingerprint='8e21d75c2fe8f78ee4fd355183f56913f975f5f3cfc38240cc5271cd60c74373',source_commit=read(V/'PROVENANCE_SHA256.json')['source_commit'],source_SHA256=sha(source),actual_callback_AST_executed=True,failure=failure,explanation='The production callback always writes root/latest.pt, but its immutable writer refuses any existing destination. Repeated publication fails without an optimizer step, so periodic recovery at5000updates cannot be admitted.',optimizer_objects_created=0,optimizer_steps=0,production_updates=0,generation_attempts=0,remaining_recovery_tests='NOT_RUN: user instructed stop immediately if recovery qualification fails; full optimizer interruption/journal/hash/partial-write suite cannot be passed by this unrepaired driver.',journal_static_issue='resume does not replay committed journal entries newer than checkpoint; remains unqualified',new_test_files_retained=True,elapsed_seconds=time.perf_counter()-start,D_free_bytes=shutil.disk_usage(R).free)
atomic_json(Q/'RESULT.json',result);print(json.dumps(dict(qualification=result['status'],first_checkpoint_bytes=first['bytes'],failure=failure,production_updates=0,stop_before_training=True)),flush=True)
