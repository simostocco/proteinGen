"""Select only measured execution improvements, then test complete selected-state recovery."""
from pathlib import Path
import json,copy,io,math,time,resource,ast
import torch
from protein_joint_sac.crisp_v3 import admission as A,optimization as O,networks as N
from protein_joint_sac.crisp_v3.execution import set_loader,scientific_config
from protein_joint_sac.crisp_v3.data import Population,CoverageSampler
from protein_joint_sac.crisp_v3.prior import FrozenPrior,file_sha,tensor_sha
from protein_joint_sac.crisp_v3.storage import restore,payload,rng_state,digest
from protein_joint_sac.crisp_v3.recovery import state_digest
R=Path(__file__).resolve().parents[3];D=R.parent;B=json.load(open(R/'results/COMPLETE_MICROBATCH_BENCHMARK.json'));P=json.load(open(R/'results/VERIFIED_OPERATOR_PAUSE_EXIT.json'));baseline=B['summary']['1']['median_seconds'];numeric={r['microbatch']:r['pass_numerical'] for r in B['comparisons']};candidates=[]
for m in [2,4]:
 if numeric[m]:candidates.append(dict(microbatch=m,loader='eager',seconds=B['summary'][str(m)]['median_seconds']))
if all(B['loader_equivalence'].values()) and B['lazy_loader_summary']['1']['speedup_fraction_vs_original']>=.10:
 for m in [1,4]:
  if m==1 or numeric[m]:candidates.append(dict(microbatch=m,loader='arrow_scalar_lru4',seconds=B['lazy_loader_summary'][str(m)]['median_seconds']))
candidates=[{**r,'speedup_fraction':1-r['seconds']/baseline} for r in candidates];eligible=[r for r in candidates if r['speedup_fraction']>=.15];choice=min(eligible,key=lambda r:r['seconds']) if eligible else dict(microbatch=1,loader='eager',seconds=baseline,speedup_fraction=0)
assert all(B['sealed_microbatch1_equivalence'].values())
with open(R/'results/EXECUTION_SELECTION.json','x') as f:json.dump(dict(selected=choice,candidates=candidates,promotion_speedup_floor=.15,loader_independent_improvement_floor=.10,lazy_loader_not_promoted_reason='Only1.3% isolated gain; retain existing loader to minimize execution changes' if B['lazy_loader_summary']['1']['speedup_fraction_vs_original']<.10 else None,meaning='No objective/effective-batch/LR/data/panel/gate change; numeric agreement required separately',eligible_for_optimized_resume=bool(eligible)),f,indent=2)
if not eligible:raise SystemExit('NO_OPTIMIZATION_PROMOTION; resume original sealed worker from verified250')
old=D/'e017_crisp_v3_training_admission_20261021_v1';config=json.load(open(old/'configs/SEALED_RESEARCH_INITIALIZATION.json'));oldrun=Path(P['checkpoint']).parent.parent;config['microbatch']=choice['microbatch'];config['execution_protocol']=dict(schema='e017_microbatch_execution_v1',loader=choice['loader'],scientific_base_source='b62978364dc054850f6b37e888baaa9d1638d4f6',scientific_base_config_sha256='658231a690a28a2b3af76f92cb82ec922e25d97d97a5c8b47f3ce66a40c95157',original_segment=str(oldrun));config['execution_baseline_validation']=str(oldrun/'validation_00000.json');config['execution_parent']=dict(path=P['checkpoint'],sha256=P['checkpoint_sha256'],update=250,source_commit='b62978364dc054850f6b37e888baaa9d1638d4f6',baseline_validation_sha256=file_sha(config['execution_baseline_validation']),validation=P['bundle']['validation'])
with open(R/'configs/OPTIMIZED_EXECUTION.json','x') as f:json.dump(config,f,indent=2)
# Enforce a single GPU worker, including benchmarks.
try:
 if Path('/proc/178643/stat').read_text().rsplit(') ',1)[1].split()[0]!='Z':raise RuntimeError('old worker still alive')
except FileNotFoundError:pass
parent=torch.load(P['checkpoint'],map_location='cpu',weights_only=False);assert scientific_config(config)==scientific_config(parent['config']);set_loader(choice['loader']);torch.use_deterministic_algorithms(True);prior=FrozenPrior('cuda');system=N.System('research').cuda();learner=O.Learner(system,**config['optimizer']);A.configure_initialization(system,learner,config['schedule']);pop=Population();samplers=dict(AA=CoverageSampler(pop),geometry=CoverageSampler(pop),goals=A.InitGoals(pop,1705,json.load(open(config['native_limits']))));restore(P['checkpoint'],system,learner,samplers,parent['config']);system.train()
for k,m in system.models.items():
 if k!='actor':m.eval()
def step():
 with torch.autocast('cuda',dtype=torch.bfloat16):result=A.initialization_update(system,learner,prior,pop,samplers['AA'],samplers['geometry'],samplers['goals'],config)
 torch.cuda.synchronize();return result
def fingerprint():return dict(model=tensor_sha(system),optimizer=state_digest({k:o.state_dict() for k,o in learner.optimizers.items()}),scheduler=state_digest({k:s.state_dict() for k,s in learner.schedulers.items()}),samplers=state_digest({k:s.state_dict() for k,s in samplers.items()}),RNG=state_digest(rng_state()),steps=dict(learner.steps))
first=step();counter=copy.deepcopy(parent['counters']);counter['updates']=251;counter['engineering_fixture_only']=True;test_config={**config,'engineering_fixture_only':True};state=payload(system,learner,samplers,test_config,parent['replay_snapshot'],counter);memory=io.BytesIO();torch.save(state,memory);size=memory.tell();second=step();expected=fingerprint();memory.seek(0);restore(memory,system,learner,samplers,test_config);repeated=step();actual=fingerprint();assert expected==actual,(expected,actual)
assert file_sha(parent['replay_snapshot']['path'])==parent['replay_snapshot']['sha256']
# Actual worst certified long-prefix fixture at selected LIVE microbatch; no optimizer step.
ids=pop.groups['TRAIN']['complete'];h=max(ids,key=lambda h:len(pop.base.seqs[h]));n=len(pop.base.seqs[h]);rows=[('complete',h,n-1,samplers['geometry'].rng) for _ in range(choice['microbatch'])];p,aa,g,mask,_=pop.batch(rows,'cuda');before=tensor_sha(system);torch.cuda.reset_peak_memory_stats();learner.optimizers['actor'].zero_grad(set_to_none=True)
with torch.autocast('cuda',dtype=torch.bfloat16):
 b=A.prepare(p,prior(p),torch.zeros(len(aa),1,36,device='cuda'),torch.zeros(len(aa),1,dtype=torch.bool,device='cuda'));loss,_=A.actor_loss(system,b,aa,g,mask,config['stream_weights']['AA']);loss.backward()
finite=bool(torch.isfinite(loss)) and all(p.grad is None or torch.isfinite(p.grad).all() for p in system.models['actor'].parameters());assert finite;assert tensor_sha(system)==before;prior.verify(learner.optimizers.values());worst=dict(N=n,prefix=n-1,live_microbatch=choice['microbatch'],loss=float(loss.detach()),finite_gradients=finite,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),host_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
result=dict(status='PASS',selected=choice,complete_serialized_checkpoint_bytes=size,serialization='Full model/targets/optimizer/scheduler/samplers/RNG/replay binding serialized to memory; no redundant large untrained checkpoint on D:',restarted_next_complete_update_bitwise_exact=expected==actual,verified_fields=list(expected),replay_snapshot_sha_verified=True,J1_frozen=True,all_nonactor_state_preserved=True,worst_length_fixture=worst,optimizer_calls_for_discarded_recovery_test=3,unique_campaign_updates=0,selected_config_sha256=digest(config),parent_checkpoint=P['checkpoint'],parent_checkpoint_sha256=P['checkpoint_sha256'])
with open(R/'results/SELECTED_EXECUTION_RECOVERY_TEST.json','x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
