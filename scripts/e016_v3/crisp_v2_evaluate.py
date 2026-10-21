from crisp_v2_core import *
from crisp_v2_interface import *
import resource,itertools,subprocess
start=time.monotonic();guard();torch.set_num_threads(6);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
assert read('DIRECT_STEERING_TESTS.json')['all_pass'];actor=load_actor();frozen=tensor_sha(actor);adapter=Steering().cuda();adapter.load_state_dict(torch.load(R/'runs/adapter_00080.pt',weights_only=False)['adapter']);adapter.eval();zero=Steering().cuda().eval();tasks=[r for r in read('tasks/PUBLIC_TASKS.json')if r['split']=='VALIDATION'];cfg=read('configs/FROZEN_PROTOCOL.json');limits=read('configs/NATIVE_CALIBRATION.json');rows=[];traces=[]
for ti,task in enumerate(tasks):
 for seed_index in range(2):
  seed=20261700+ti*10+seed_index
  for arm in cfg['comparison_arms']:
   if time.monotonic()-start>cfg['max_GPU_worker_seconds']:raise RuntimeError('1200s GPU worker gate; completed rows remain checkpointed')
   guard();st=time.monotonic();adapt=zero if arm in ['frozen_J1','frozen_rewind']else adapter;aa=arm!='geometry_only';geo=arm!='AA_only';env=Hierarchy(actor,adapt,task,seed,aa,geo);world=ProteinWorld(env);corrections=arm in ['joint_subgoal_rewind','frozen_rewind'];status='COMPLETED';error=None
   try:
    while True:
     if env.ticks>=3*task['N']or time.monotonic()-st>30:status='ADMINISTRATIVE_TRUNCATION';break
     obs=world.observation();reason=None;pos=None
     if env.state.t==env.state.N:
      result=assess(env.state,task,limits)
      if corrections and not result['success']:reason='completed_constraints_or_validity_miss';pos=max(3,env.state.t-16)
      else:world.transition(HighAction(Kind.STOP));break
     elif corrections:
      passed=[int(k)for k,a in task['motifs'].items()if int(k)<env.state.t and env.state.residues[int(k)]!=a]
      if passed:reason='observed_motif_miss';pos=max(3,min(passed)-4)
      w=task['waypoint']
      if not reason and w and w['index']<env.state.t and float(torch.linalg.vector_norm(env.state.ca[w['index']]-env.state.ca.new_tensor(w['position'])))>task['waypoint_tolerance_A']:reason='observed_waypoint_miss';pos=max(3,w['index']-6)
      remaining=env.state.N-env.state.t;distance=float(torch.linalg.vector_norm(env.state.ca[-1]-env.state.ca.new_tensor(task['endpoint'])))
      if not reason and distance>limits['adjacent_high_A']*remaining:reason='necessary_reachability_bound_failed';pos=max(3,env.state.t-8)
     if reason:
      world.transition(HighAction(Kind.REWIND,arguments=dict(position=pos)));env.interventions[-1]['trigger']=reason;world.transition(HighAction(Kind.REGENERATE));continue
     sg='waypoint'if arm=='joint_subgoal_rewind'and task['waypoint']and env.state.t<=task['waypoint']['index']else 'endpoint';world.transition(HighAction(Kind.CONTINUE,sg))
   except Exception as ex:status='NUMERICAL_OR_INTERFACE_FAILURE';error=repr(ex)
   result=assess(env.state,task,limits);row=dict(task=task['id'],identity=task['identity'],kind=task['kind'],N=task['N'],seed=seed,arm=arm,status=status,error=error,ticks=env.ticks,proposals_generated=env.calls,J1_calls=env.calls,rewinds=sum(x['action']=='REWIND'for x in env.interventions),branches=env.branch,seconds=time.monotonic()-st,final_length=env.state.t,sequence=''.join(env.state.residues),assessment=result,discounted_terminal_success=cfg['gamma']**env.ticks*int(result['success'])if status=='COMPLETED'else None)
   rows.append(row);traces.append(dict(task=task['id'],seed=seed,arm=arm,history=env.history,interventions=env.interventions,CA=env.state.ca.cpu().tolist()))
   write('FEASIBILITY_EPISODES.json',rows)
  print('VAL',ti+1,'seed',seed_index+1,'attempts',len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
# Compact trajectory evidence, including actual action log probabilities, masks and branch semantics.
write('results/EPISODE_TRACES.json',traces)
summary={}
for arm in cfg['comparison_arms']:
 rr=[r for r in rows if r['arm']==arm];complete=[r for r in rr if r['status']=='COMPLETED'];good=[r for r in complete if r['assessment']['success']];meas=[r['assessment']for r in complete];summary[arm]=dict(attempts=len(rr),completed=len(complete),administratively_truncated=sum(r['status']=='ADMINISTRATIVE_TRUNCATION'for r in rr),numerical_failures=sum(r['status']=='NUMERICAL_OR_INTERFACE_FAILURE'for r in rr),successes=len(good),success_lower_bound=len(good)/len(rr),success_upper_bound_if_all_censored_succeed=(len(good)+sum(r['status']=='ADMINISTRATIVE_TRUNCATION'for r in rr))/len(rr),motif_satisfaction=sum(r['motifs']for r in meas)/len(meas)if meas else None,validity=sum(r['valid']for r in meas)/len(meas)if meas else None,endpoint_error_mean_A=float(np.mean([r['position_error_A']for r in meas]))if meas else None,clashes_mean_per100=float(np.mean([r['metrics']['clashes_per100_residues']for r in meas]))if meas else None,seconds=sum(r['seconds']for r in rr),J1_calls=sum(r['J1_calls']for r in rr),rewinds=sum(r['rewinds']for r in rr),unique_sequences=len({r['sequence']for r in complete}))
# Parent-level paired uncertainty, 8clusters; two task kinds/seeds remain within cluster.
rng=np.random.default_rng(917);parents=sorted({r['identity']for r in rows});paired={}
for arm in cfg['comparison_arms'][1:]:
 dif=[]
 for p in parents:
  a=[r for r in rows if r['identity']==p and r['arm']==arm];b=[r for r in rows if r['identity']==p and r['arm']=='frozen_J1'];dif.append(np.mean([int(r['assessment']['success'])for r in a])-np.mean([int(r['assessment']['success'])for r in b]))
 boot=np.mean(rng.choice(dif,(4000,len(dif))),axis=1);paired[arm]=dict(success_lower_bound_delta=float(np.mean(dif)),parent_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),parents=len(parents),note='administrative truncations count as non-success only for this lower bound; primary censoring reported separately')
write('FEASIBILITY_COMPARISON.json',dict(summary=summary,paired=paired,seconds=time.monotonic()-start,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),host_max_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,J1_frozen_after=tensor_sha(actor)==frozen,scope='8VAL parents,16P1/P2tasks,2matched streams,6arms; joint_subgoal_rewind and frozen_rewind are fixed heuristics, not learned high-level policies; AA/G gates ablate one joint-supervised adapter, not separately trained arms',all_unresolved_truncations_reported=True))
print('FEASIBILITY',summary,flush=True)
