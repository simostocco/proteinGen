from core97 import *
import torch,numpy as np,copy,time,sys,subprocess,gc,math,types
from gate97 import install,GATE_SCALE
from safety97 import Monitor

def deterministic():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert not subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
def timing(fn,reps=5):
 for _ in range(2):fn()
 torch.cuda.synchronize();t=time.perf_counter()
 for _ in range(reps):fn()
 torch.cuda.synchronize();return (time.perf_counter()-t)/reps

def equivalent():
 verify();deterministic();P,F=factory('B',ROOT/'qualification');e,_=P['K']['load_engine']('C');e.actor.eval().requires_grad_(True);original=e.actor._v97_original_forward;gate_forward=e.actor.forward;fingerprint=P['c']['state_digest'](e.actor.state_dict());count=sum(p.numel() for p in e.actor.parameters());assert count==17438115
 contexts=[];fixtures=[]
 for gi in [0,1,2,3]:
  obj=read(PAIR.parent/f'e017_v96_arm_a_original_20261029_v1/production/diagnostics/DEV_00000_filtered_{gi:02d}.json')
  for N in [32,64,96]:
   for t in [0,1,2,8,16, min(N-1,48)]:
    row=obj['records'][0]
    if t>len(row['geometry']):continue
    state=P['S']['reconstruct'](row,64,t);state=__import__('dataclasses').replace(state,N=N);goal=copy.deepcopy(obj['task']['goal']);goal['N']=N;goal['goal_id']+='_ENGINEERING_N'+str(N)
    if N==32:goal.update(exposed=[4,12],buried=[20,28],motif=obj['task']['fixed_sequence'][4:6])
    w=P['S']['world'](goal,state);prefix=P['p']['collate']([state],'cuda')
    with torch.no_grad():ref=e.prior(prefix);batch=P['p']['public_batch']([w],prefix,ref);a=original(batch);b=gate_forward(batch)
    assert all(torch.equal(x,y) for x,y in zip(a,b));mask=P['c']['mask_for_step'](t);aa=P['AA'].index(obj['task']['fixed_sequence'][t]);u=a[1][0,aa].double()+a[2][0,aa].double().exp()*torch.tensor([.13,-.21,.45],device='cuda',dtype=torch.float64)
    T=P['K']['S']['T'];la=T['base_logprob'](a[1][0,aa],a[2][0,aa],u,mask);lb=T['base_logprob'](b[1][0,aa],b[2][0,aa],u,mask);assert torch.equal(la,lb)
    eps=torch.tensor([.13,-.21,.45],device='cuda',dtype=torch.float64);act=P['S']['corrected_action'];xa,da=act(a[1][0,aa],a[2][0,aa],mask,eps);xb,db=act(b[1][0,aa],b[2][0,aa],mask,eps);assert torch.equal(xa,xb) and da==db
    fixtures.append(dict(TRAIN_goal=gi,N=N,t=t,parameters_identical=True,latent_logprob_identical=True,transformed_actions_identical=True))
    if N==64 and t==8:contexts.append((batch,obj['task'],aa))
 # Gradient at zero and active branch: no neural update.
 gradient_tests=[]
 for batch,task,aa in contexts:
  for value in [0.,.2*GATE_SCALE,-.2*GATE_SCALE]:
   with torch.no_grad():e.actor.burial_gate.fill_(value)
   e.actor.zero_grad(set_to_none=True);_,mu,ls=e.actor(batch);scalar=mu[0,aa].sum();scalar.backward();g=float(e.actor.burial_gate.grad);shared=float(e.actor.encoder.goal.weight.grad.norm());assert math.isfinite(g) and abs(g)>1e-9 and shared>0
   # Differentiate the same gate map in FP64 with fixed measured branch outputs.
   # This avoids cancellation in a float32 sum of geometry/logstd outputs.
   with torch.no_grad():
    _,m0,l0=original(batch);from algorithm import augment
    _,m1,l1=original(augment(batch));delta=(m1[0,aa]-m0[0,aa]).double()
    def evaluate(raw):return float((m0[0,aa].double()+torch.tensor(raw,device='cuda',dtype=torch.float64).div(GATE_SCALE).tanh()*delta).sum())
    eps=1e-4*GATE_SCALE;high=evaluate(value+eps);low=evaluate(value-eps)
   error=abs((high-low)/(2*eps)-g);assert error<max(1e-5,abs(g)*1e-4)
   gradient_tests.append(dict(goal=task['goal']['goal_id'],raw_gate=value,gate_gradient=g,goal_projection_gradient_norm=shared,finite_difference_error=error))
 with torch.no_grad():e.actor.burial_gate.zero_()
 # Actual burial-only terminal utility score gradient from recorded TRAIN on-policy group.
 packet=read(PAIR.parent/'e017_v96_arm_a_original_20261029_v1/production/rollouts/u000001.json');rows=packet['records'];task=packet['task'];rew=np.array([.1*int(r['metrics']['hard_valid'])*r['metrics'].get('utilities',[0,0])[1] for r in rows]);adv=(16*rew-rew.sum())/15;worlds=[P['S']['world'](task['goal']) for _ in rows];e.actor.zero_grad(set_to_none=True);maxerr=0.
 for t in range(64):
  ids=[i for i,r in enumerate(rows) if any(q[0]==t for q in r['proposals'])]
  if not ids:continue
  prefix=P['p']['collate']([worlds[i].state for i in ids],'cuda')
  with torch.no_grad():ref=e.prior(prefix);batch=P['p']['public_batch']([worlds[i] for i in ids],prefix,ref)
  _,mu,ls=e.actor(batch);terms=[]
  for j,i in enumerate(ids):
   aa=P['AA'].index(task['fixed_sequence'][t]);lp=[]
   for q in rows[i]['proposals']:
    if q[0]!=t:continue
    z=P['K']['S']['T']['base_logprob'](mu[j,aa],ls[j,aa],torch.tensor(q[2],device='cuda',dtype=torch.float64),P['c']['mask_for_step'](t));maxerr=max(maxerr,abs(float(z.detach())-q[4]));lp.append(z)
   terms.append(-adv[i]*torch.stack(lp).sum()/16)
  torch.stack(terms).sum().backward()
  for i in ids:
   if t<len(rows[i]['geometry']):worlds[i].state=P['S']['append'](worlds[i].state,task['fixed_sequence'][t],rows[i]['geometry'][t])
 burial_grad=float(e.actor.burial_gate.grad);assert abs(burial_grad)>1e-9 and maxerr<5e-5
 e.actor.zero_grad(set_to_none=True)
 # Verify actual token-branch input path: at nonzero gate it has nonzero gradient.
 from algorithm import augment
 b0,task,aa=contexts[0];b1=augment(b0);b1['goals']=b1['goals'].detach().requires_grad_(True)
 with torch.no_grad():e.actor.burial_gate.fill_(.2*GATE_SCALE)
 a0,m0,l0=original(b0);a1,m1,l1=original(b1);g=(e.actor.burial_gate/GATE_SCALE).tanh();val=(m0+g*(m1-m0))[0,aa].sum();token_grad=torch.autograd.grad(val,b1['goals'])[0][0,-1];assert token_grad.norm()>0
 with torch.no_grad():e.actor.burial_gate.zero_()
 # Raw/filtered complete finite-controller equivalence under same indexed seeds.
 generated=[];seconds={};tasks=read(ROOT/'MONITOR_POPULATION.json')['tasks'][::64][:4]
 for gi,task in enumerate(tasks):
  for mode in ['B','C']:
   seeds=[seed('EQUIVALENCE',gi,mode,j) for j in range(8)]
   e.actor.forward=original;start=time.perf_counter();a=P['sample'](e,task,mode,8,seeds);oldsec=time.perf_counter()-start
   e.actor.forward=gate_forward;start=time.perf_counter();b=P['sample'](e,task,mode,8,seeds);newsec=time.perf_counter()-start
   for ra,rb in zip(a,b):
    for k in ra:
     if k=='seconds':continue
     assert ra[k]==rb[k],('TRAJECTORY_DIFFERENCE',gi,mode,k)
   generated.append(dict(goal=task['goal']['goal_id'],mode=mode,attempts_per_policy=8,original_seconds=oldsec,gated_seconds=newsec,completed=sum(r['metrics']['complete'] for r in b),aborts=sum(r['termination'] is not None for r in b),original_records=a,gated_records=b))
 # Forward/backward costs: actual batch8 constructed from one goal, no extra generation.
 batch,task,aa=contexts[0]
 def bwd(fn):
  e.actor.zero_grad(set_to_none=True);_,m,l=fn(batch);(m.sum()+l.sum()).backward()
 costs=dict(original_forward_seconds=timing(lambda:original(batch)),gated_forward_seconds=timing(lambda:gate_forward(batch)),original_forward_backward_seconds=timing(lambda:bwd(original),3),gated_forward_backward_seconds=timing(lambda:bwd(gate_forward),3),timing_batch=1,stochastic_group=16,trace_replay_microbatch_max=16)
 # Exercise real single-draw monitoring collector on a bounded suffix, never a safety PASS.
 e.actor.zero_grad(set_to_none=True);fp=P['c']['state_digest'](e.actor.state_dict());m=Monitor(ROOT/'qualification/real_monitor_prefix','B',0,read(ROOT/'SOURCE_BINDING.json')['models'][str(R3/'production/checkpoints/milestone_05000.pt')],actor_fingerprint=fp)
 start=time.perf_counter();decision=m.collect(P,e,max_new=16);monitorsec=time.perf_counter()-start;assert decision['status']=='INCOMPLETE' and m.frontier==16
 assert P['c']['state_digest'](e.actor.state_dict())==fingerprint;e.prior.verify();assert P['c']['state_digest'](e.parent.state_dict())==e.fp;assert all(p.grad is None for p in e.prior.model.parameters())
 save('INITIAL_EQUIVALENCE.json',dict(PASS=True,fixtures=fixtures,lengths=[32,64,96],synthetic_lengths_only_for_robustness=True,full_generation_task_N=64,distribution_bitwise_equal=True,trajectory_tests=generated,new_generation_attempts=128,full_trajectories_compared=64,additional_parameters=1,original_parameters=17438114,parameter_total=count,native_future_coordinates=False))
 save('GATE_GRADIENT_TESTS.json',dict(PASS=True,direct_tests=gradient_tests,burial_only_RLOO_gate_gradient=burial_grad,burial_TRACE_logprob_maxerror=maxerr,nonzero_gate_token_input_gradient=float(token_grad.norm()),neural_optimizer_updates=0,interpretation='usable gradient, not scientific steering evidence'))
 save('COMPUTE_COSTS.json',dict(costs=costs,real_monitor_single_candidate_draws=16,real_monitor_seconds=monitorsec,estimated4096_monitor_seconds=monitorsec*256,monitor_backbone_valid=m.backbone,monitor_decision=decision,monitor_full4096_admission_performed=False,peak_GPU_bytes=torch.cuda.max_memory_allocated(),new_generation_attempts=144,J1_unchanged=True,actor_unchanged=True))
 print('GATE_EQUIVALENCE_AND_GRADIENT_PASS',len(fixtures),burial_grad,flush=True)

def draws():return dict(cpu=torch.rand(8).tolist(),cuda=torch.rand(8,device='cuda').cpu().tolist(),np=np.random.random(8).tolist(),python=__import__('random').random())
def recovery(arm):
 verify();deterministic();path=ROOT/f'qualification/recovery_{arm}';path.mkdir(parents=True,exist_ok=True);P,F=factory(arm,path);rt=P['Runtime'](path);initial=rt.state();fp=P['state_digest'](initial);d=draws();rt.restore();assert P['state_digest'](rt.state())==fp and draws()==d;rt.restore()
 expected=read(ROOT/'TRAINING_SCHEDULE.json')['updates'][0];assert rt.task(1)['goal']['goal_id']==expected['goal_id']
 t=time.perf_counter();logical=rt.step(1);secs=time.perf_counter()-t;post=P['state_digest'](rt.state());rt.restore();again=rt.step(1);assert again==logical and P['state_digest'](rt.state())==post
 rt.publish('ENGINEERING_QUALIFIED');post=P['state_digest'](rt.state());d=draws();rt.restore();assert P['state_digest'](rt.state())==post and draws()==d;rt.restore()
 original_binding=rt.binding;rt.binding=dict(rt.binding,WRONG_SOURCE=True)
 try:rt.restore()
 except AssertionError:pass
 else:raise AssertionError('source mismatch accepted')
 rt.binding=original_binding;rt.restore()
 indexhash=sha(path/'RECOVERY_INDEX.json');proc=subprocess.run([str(ROOT/'run.sh'),str(WT/'e017_v97/gpu97.py'),'partial',arm]);assert proc.returncode==75 and sha(path/'RECOVERY_INDEX.json')==indexhash;rt.restore();assert P['state_digest'](rt.state())==post
 # A deliberately corrupted copied checkpoint must never be accepted.
 cp=path/read(path/'RECOVERY_INDEX.json')['latest']['relative'];corrupt=path/'checkpoints/CORRUPT_FIXTURE.pt';data=bytearray(cp.read_bytes());data[len(data)//2]^=1;corrupt.write_bytes(data);assert sha(corrupt)!=sha(cp)
 idx=read(path/'RECOVERY_INDEX.json');bad=copy.deepcopy(idx);bad['latest']['relative']=str(corrupt.relative_to(path));put(path/'RECOVERY_INDEX.json',bad,False)
 try:rt.restore()
 except AssertionError:pass
 else:raise AssertionError('corrupted checkpoint restored')
 put(path/'RECOVERY_INDEX.json',idx,False);rt.restore();assert P['state_digest'](rt.state())==post
 put(path/'CORRUPTION_REJECTION.json',dict(PASS=True,actual_restore_rejected=True,expected=sha(cp),corrupt=sha(corrupt)))
 rt.e.prior.verify();assert P['c']['state_digest'](rt.e.parent.state_dict())==rt.e.fp
 if arm=='B':assert 'burial_gate' in rt.state()['actor'] and abs(float(rt.e.actor.burial_gate))>0
 save(f'RECOVERY_{arm}.json',dict(PASS=True,arm=arm,unique_engineering_updates=1,physical_optimizer_calls=2,new_training_attempts=16,first_update_seconds=secs,checkpoint_bytes=cp.stat().st_size,full_state_RNG_scheduler_exact=True,journal_replay_exact=True,likelihood_replay_maxerror=logical['replay_logprob_maxerror'],source_binding_rejection=True,interrupted_full_checkpoint_PASS=True,corrupt_file_SHA_rejected=True,gate_value=float(rt.e.actor.burial_gate) if arm=='B' else None,peak_GPU_bytes=torch.cuda.max_memory_allocated(),production_updates=0))
 print('RECOVERY_PASS',arm,flush=True)
def partial(arm):
 from protein_joint_sac.crisp_v3.recovery import atomic_checkpoint,StoragePolicy
 import builtins
 path=ROOT/f'qualification/recovery_{arm}';idx=read(path/'RECOVERY_INDEX.json');state=torch.load(path/idx['latest']['relative'],map_location='cpu',weights_only=False);target=path/'checkpoints/INTERRUPTED.pt';opened=builtins.open
 class Fault:
  def __init__(self,f):self.f=f;self.n=0
  def __getattr__(self,n):return getattr(self.f,n)
  def __enter__(self):return self
  def __exit__(self,*a):return self.f.__exit__(*a)
  def write(self,b):
   n=self.f.write(b);self.n+=n
   if self.n>8*2**20:self.f.flush();os.fsync(self.f.fileno());os._exit(75)
   return n
 def patched(p,*args,**kw):
  f=opened(p,*args,**kw);return Fault(f) if str(p)==str(target)+'.part' and args and args[0]=='xb' else f
 builtins.open=patched;atomic_checkpoint(target,state,StoragePolicy(checkpoint_hint_bytes=224*2**20));raise AssertionError('fault not triggered')
if __name__=='__main__':
 {'equivalence':equivalent,'recovery':lambda:recovery(sys.argv[2]),'partial':lambda:partial(sys.argv[2])}[sys.argv[1]]()
