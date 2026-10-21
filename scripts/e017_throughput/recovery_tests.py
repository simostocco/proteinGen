from benchlib import *
from protein_joint_sac.crisp_v3.execution import scientific_config
selection=json.load(open(R/'results/FINAL_EXECUTION_SELECTION.json'));choice=selection['selected'];flags={k:choice[k] for k in ['geometry_microbatch','goal_microbatch','cpu_prefetch','geometry_cpu_prefetch','share_prefix_features','compile_actor_rope']};X.QUALIFIED.setup(pop);set_loader(choice['loader']);prior.shared_prefix_features=choice['share_prefix_features'];X.enable_compiled_rope(choice['compile_actor_rope']);restore(bundle['checkpoint'],system,learner,samplers,config)
original=X.initialization_update
def run(capture=False,profile=False):
 def selected(*args,**kw):
  c=copy.deepcopy(args[-1]);c['execution_protocol'].update(flags);return original(*args[:-1],c,**kw)
 with patch.object(X,'initialization_update',selected):return step(choice['microbatch'],True,capture,profile)
def fingerprint():return dict(system=tensor_sha(system),optimizer=state_digest({k:v.state_dict() for k,v in learner.optimizers.items()}),scheduler=state_digest({k:v.state_dict() for k,v in learner.schedulers.items()}),samplers=state_digest({k:v.state_dict() for k,v in samplers.items()}),RNG=state_digest(rng_state()),steps=dict(learner.steps))
# Last matched ablation: keep optional optimizations only when end-to-end faster.
ablation=[]
for shared,geometry_prefetch in [(True,True),(True,False),(False,False)]:
 set_loader(choice['loader']);restore(bundle['checkpoint'],system,learner,samplers,config);prior.shared_prefix_features=shared;flags.update(share_prefix_features=shared,geometry_cpu_prefetch=geometry_prefetch)
 for i in range(4):
  r,_=run(False,profile=(i==0));ablation.append(dict(shared_features=shared,geometry_prefetch=geometry_prefetch,offset=i,seconds=r['seconds'],profile=r['profile']));print('ABLATION',json.dumps({k:v for k,v in ablation[-1].items() if k!='profile'}),flush=True)
summary=[]
for shared,geometry_prefetch in [(True,True),(True,False),(False,False)]:
 rows=[r for r in ablation if r['shared_features']==shared and r['geometry_prefetch']==geometry_prefetch];summary.append(dict(share_prefix_features=shared,geometry_cpu_prefetch=geometry_prefetch,mean_seconds=statistics.mean(r['seconds'] for r in rows),steady_median_seconds=statistics.median(r['seconds'] for r in rows[1:])))
best=min(summary,key=lambda r:r['mean_seconds']);choice.update(best);flags.update(share_prefix_features=choice['share_prefix_features'],geometry_cpu_prefetch=choice['geometry_cpu_prefetch']);prior.shared_prefix_features=choice['share_prefix_features']
(R/'results/FINAL_EXECUTION_ABLATION.json').write_text(json.dumps(dict(trials=ablation,summary=summary,selected=choice),indent=2))
# Final direct comparison against the actual B4/scalar auxiliary parent execution.
set_loader(choice['loader']);restore(bundle['checkpoint'],system,learner,samplers,config);prior.shared_prefix_features=False;old_row,old_cap=step(4,False,True)
restore(bundle['checkpoint'],system,learner,samplers,config);prior.shared_prefix_features=choice['share_prefix_features'];new_row,new_cap=run(True)
a=old_cap['gradient'];b=new_cap['gradient'];rel=float((a-b).norm()/a.norm());cos=float(torch.dot(a,b)/(a.norm()*b.norm()));pd=float((old_cap['parameters']-new_cap['parameters']).abs().max());ld=abs(old_row['loss']-new_row['loss']);passed=rel<=.03 and cos>=.999 and pd<=1e-5 and ld<=max(.003,.001*abs(old_row['loss'])) and old_cap['samplers']==new_cap['samplers'] and old_cap['RNG']==new_cap['RNG'];assert passed
(R/'results/FINAL_NUMERICAL_EQUIVALENCE.json').write_text(json.dumps(dict(passed=passed,gradient_relative_L2=rel,gradient_cosine=cos,parameter_max_abs=pd,loss_absolute_difference=ld,sampler_exact=old_cap['samplers']==new_cap['samplers'],RNG_exact=old_cap['RNG']==new_cap['RNG'],selected=choice),indent=2));del old_cap,new_cap
restore(bundle['checkpoint'],system,learner,samplers,config)
first,_=run();test_config=copy.deepcopy(config);test_config['microbatch']=choice['microbatch'];test_config['execution_protocol'].update(flags,loader=choice['loader']);test_config['engineering_fixture_only']=True;counter=copy.deepcopy(parent['counters']);counter['updates']=373;counter['engineering_fixture_only']=True;state=payload(system,learner,samplers,test_config,parent['replay_snapshot'],counter);memory=io.BytesIO();torch.save(state,memory);serialized_bytes=memory.tell();second,_=run();expected=fingerprint();memory.seek(0);restore(memory,system,learner,samplers,test_config);third,_=run();actual=fingerprint();assert expected==actual,(expected,actual)
# Persist ONE transient full checkpoint for a separately launched GPU process.
# BytesIO holds the immutable update373 bytes, independent of live tensor references.
import os,hashlib
fixture=R/'results/engineering_recovery';fixture.mkdir(exist_ok=True);path=fixture/'transient_full_state.pt';part=path.with_suffix('.part');memory.seek(0)
with open(part,'xb') as f:
 for block in iter(lambda:memory.read(8*2**20),b''):f.write(block)
 f.flush();os.fsync(f.fileno())
os.replace(part,path)
assert file_sha(path)==hashlib.sha256(memory.getvalue()).hexdigest()
(fixture/'EXPECTED_RESTART.json').write_text(json.dumps(dict(checkpoint=str(path),sha256=file_sha(path),expected=expected,config=test_config,selected=choice,transient_created_by_this_task=True),indent=2))
assert file_sha(parent['replay_snapshot']['path'])==parent['replay_snapshot']['sha256']
# Longest-prefix stress at selected live batch, plus a true empty-prefix fixture.
h=max(pop.groups['TRAIN']['complete'],key=lambda h:len(pop.base.seqs[h]));length=len(pop.base.seqs[h]);fixtures=[]
for t in [0,length-1]:
 records=[('complete',h,t,samplers['geometry'].rng) for _ in range(choice['microbatch'])];p,aa,g,mask,_=pop.batch(records,'cuda');learner.optimizers['actor'].zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();before=tensor_sha(system)
 with torch.autocast('cuda',dtype=torch.bfloat16):
  b=A.prepare(p,prior(p),torch.zeros(len(aa),1,36,device='cuda'),torch.zeros(len(aa),1,dtype=torch.bool,device='cuda'));loss,_=A.actor_loss(system,b,aa,g,mask,config['stream_weights']['AA']);loss.backward()
 torch.cuda.synchronize();finite=bool(torch.isfinite(loss)) and all(p.grad is None or torch.isfinite(p.grad).all() for p in system.models['actor'].parameters());assert finite and tensor_sha(system)==before
 fixtures.append(dict(N=length,t=t,live_microbatch=choice['microbatch'],loss=float(loss.detach()),finite=True,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
prior.verify(learner.optimizers.values());assert inactive==({k:tensor_sha(m) for k,m in system.models.items() if k!='actor'}|{'targets':tensor_sha(system.targets)});assert sum(p.numel() for p in system.parameters() if p.requires_grad)==17438114
result=dict(status='PASS',selected=choice,complete_serialized_checkpoint_bytes=serialized_bytes,exact_next_update_restart=expected==actual,fields=list(expected),replay_SHA_verified=True,J1_file_SHA=file_sha(__import__('protein_joint_sac.crisp_v3.prior',fromlist=['MAIN']).MAIN),J1_tensor_SHA=prior.identity,J1_frozen=True,all_nonactor_tensors_exact=True,only_actor_optimizer=True,actor_trainable_parameters=17438114,full_capacity=63389056,fixtures=fixtures,host_peak_RSS=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,scientific_updates=0)
(R/'results/SELECTED_FULL_RECOVERY_TEST.json').write_text(json.dumps(result,indent=2));print('RECOVERY_PASS',json.dumps(result),flush=True)
# Publish a new execution CONFIG, preserving every scientific field exactly.
c=copy.deepcopy(json.load(open(old/'configs/OPTIMIZED_EXECUTION.json')));c['microbatch']=choice['microbatch'];c['execution_protocol'].update(flags,loader=choice['loader'],schema='e017_end_to_end_execution_v2',batched_auxiliary_streams=True,previous_execution='9c244462f3980ae8008e27fd957200e1354c62c3')
# explicit paths: no bound Python callable can accidentally enter provenance.
previous_run=Path(bundle['checkpoint']).parent.parent;initial_run=old.parent/'e017_crisp_v3_training_admission_20261021_v1/training_runs/run_bf2bbb285d0b4d929eb8cef5215063f3';c['execution_protocol']['original_segments']=[str(initial_run),str(previous_run)];c['execution_parent']=dict(path=bundle['checkpoint'],sha256=bundle['sha256'],update=372,source_commit=parent['config']['source_commit'],baseline_validation_sha256=file_sha(c['execution_baseline_validation']),validation=bundle['validation'])
assert scientific_config(c)==scientific_config(parent['config']);assert scientific_config(c)==scientific_config(json.load(open(old.parent/'e017_crisp_v3_training_admission_20261021_v1/configs/SEALED_RESEARCH_INITIALIZATION.json')))
(R/'configs/OPTIMIZED_EXECUTION.json').write_text(json.dumps(c,indent=2));(R/'results/SCIENTIFIC_CONFIG_EQUIVALENCE.json').write_text(json.dumps(dict(status='PASS',base='b629783',previous='9c24446',fields_unchanged='All fields except execution protocol/parent/baseline metadata and microbatch grouping',effective_AA_batch=c['effective_AA_batch'],geometry_decisions=c['complete_geometry_decisions'],goal_decisions=c['goal_demonstrations'],milestones=c['validation_milestones'],global_ceiling=c['updates']),indent=2))
