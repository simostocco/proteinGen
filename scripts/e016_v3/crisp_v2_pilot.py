from crisp_v2_core import *
import random,resource,subprocess
from protein_joint_sac.geometry.ca_builder import next_ca
from protein_joint_sac.rl.sequence_rl import gaussian_kl
from protein_joint_sac.models.batched_bc import collate,encode_batch
start=time.monotonic();guard();torch.set_num_threads(6);torch.manual_seed(170021);torch.cuda.manual_seed_all(170021);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
actor=load_actor();baseline_hash=tensor_sha(actor);adapter=Steering().cuda();params={n:dict(shape=list(p.shape),count=p.numel())for n,p in adapter.named_parameters()};assert sum(p.numel()for p in adapter.parameters())==53530
cfg=read('configs/FROZEN_PROTOCOL.json');all_tasks=read('tasks/PUBLIC_TASKS.json');train=[q for q in all_tasks if q['split']=='TRAIN'];native={q['identity']:q for q in read('tasks/HIDDEN_NATIVE_SUPERVISION.json')};limits=read('configs/NATIVE_CALIBRATION.json');states=[];contexts=[];labels=[];targets=[];records=[];rg=np.random.default_rng(170021)
for task in train:
 n=native[task['identity']];N=task['N'];ps=sorted(set([int(k)for k in task['motifs']]+rg.choice(np.arange(3,N),6,replace=False).tolist()))
 for t in ps:
  state=BuildState(tuple(n['sequence'][:t]),torch.tensor(n['CA'][:t],dtype=torch.float64),N);sg=int(rg.integers(0,2));states.append(state);contexts.append(context(state,task,sg));labels.append(AA.index(n['sequence'][t]));targets.append(n['geometry'][t]);records.append(dict(task=task['id'],identity=task['identity'],t=t,subgoal=sg))
# Frozen full-prefix J1 calls; no full future chain is passed to reference().
features=[[],[],[],[]]
for k in range(0,len(states),16):
 out=reference(actor,states[k:k+16])
 for dst,v in zip(features,out):dst.append(v)
h,ref_l,ref_m,ref_s=[torch.cat(v)for v in features];c=torch.stack(contexts).cuda();y=torch.tensor(labels,device='cuda');g=torch.tensor(targets,dtype=torch.float64,device='cuda');pi=math.pi
u=torch.stack([g[:,0].log(),torch.logit(g[:,1]/pi),torch.atanh(g[:,2]/pi)],-1);jac=torch.stack([u[:,0],math.log(pi)-F.softplus(-u[:,1])-F.softplus(u[:,1]),math.log(pi)+2*(math.log(2)-u[:,2]-F.softplus(-2*u[:,2]))],-1)
def losses(ids):
 l,m,s=adapter(h[ids],c[ids],ref_l[ids],ref_m[ids],ref_s[ids]);a=F.cross_entropy(l,y[ids]);ii=torch.arange(len(ids),device='cuda');mm=m[ii,y[ids]];ss=s[ii,y[ids]];per=.5*((u[ids]-mm)/ss.exp()).square()+ss+.5*math.log(2*pi)+jac[ids];gl=per.mean();ka,kg,_=kl(l,m,s,ref_l[ids],ref_m[ids],ref_s[ids]);return a,gl,ka.mean(),kg.mean()/3
# TRAIN-only scale calibration before any optimizer fitting or VAL scoring.
ids=torch.arange(min(16,len(h)),device='cuda');aa,gg,ka,kg=losses(ids)
def normgrad(loss,parameters):
 gs=torch.autograd.grad(loss,list(parameters),retain_graph=True,allow_unused=True);return float(torch.sqrt(sum((v.double().square().sum()for v in gs if v is not None),torch.tensor(0.,device='cuda'))))
an=normgrad(aa,adapter.A.parameters());gn=normgrad(gg,adapter.G.parameters());coefficient=min(.1,.25*an/max(gn,1e-12));assert an>0 and gn>0
write('TRAIN_ONLY_GRADIENT_CALIBRATION.json',dict(AA_norm=an,geometry_norm=gn,geometry_coefficient=coefficient,AA_shared_transformer_updates=0,pre_validation=True))
opt=torch.optim.AdamW(adapter.parameters(),lr=cfg['lr'],weight_decay=.01);rng=np.random.default_rng(170022);curve=[];seen=set();indices=[]
def saved(update):return dict(adapter=copy.deepcopy(adapter.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),update=update,numpy_generator=copy.deepcopy(rng.bit_generator.state),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),seen=sorted(seen),sample_indices=indices.copy(),history=curve.copy(),source_checkpoint=str(MAIN),source_sha256=MAIN_SHA,config=cfg,geometry_coefficient=coefficient,frozen_tensor_sha256=baseline_hash,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip())
def restore(q):
 adapter.load_state_dict(q['adapter']);opt.load_state_dict(q['optimizer']);rng.bit_generator.state=copy.deepcopy(q['numpy_generator']);torch.set_rng_state(q['torch_rng']);torch.cuda.set_rng_state_all(q['cuda_rng'])
def step():
 sel=rng.choice(len(h),16,replace=True);ix=torch.as_tensor(sel,device='cuda');opt.zero_grad(set_to_none=True);a,ge,k1,k2=losses(ix);loss=a+coefficient*ge+.01*k1+.01*k2
 if not torch.isfinite(loss):raise FloatingPointError('nonfinite adapter loss')
 loss.backward();aa_before=float(torch.nn.utils.clip_grad_norm_(adapter.A.parameters(),5.,error_if_nonfinite=True));g_before=float(torch.nn.utils.clip_grad_norm_(adapter.G.parameters(),5.,error_if_nonfinite=True));opt.step();assert all(torch.isfinite(p).all()for p in adapter.parameters());return dict(loss=float(loss.detach()),AA_CE=float(a.detach()),geometry_NLL=float(ge.detach()),KL_A=float(k1.detach()),KL_G_per_component=float(k2.detach()),AA_gradient_norm=aa_before,geometry_gradient_norm=g_before),sel
# Correctness before any training: adapters initially exactly preserve both branches and KL0.
l,m,s=adapter(h[:2],c[:2],ref_l[:2],ref_m[:2],ref_s[:2]);ka0,kg0,_=kl(l,m,s,ref_l[:2],ref_m[:2],ref_s[:2]);tests=dict(initial_logits_exact=torch.equal(l,ref_l[:2]),initial_mu_exact=torch.equal(m,ref_m[:2]),initial_std_exact=torch.equal(s,ref_s[:2]),initial_KL_zero=bool(ka0.abs().max()==0 and kg0.abs().max()==0),both_adapters_nonzero_gradients=an>0 and gn>0,all_J1_parameters_frozen=all(not p.requires_grad for p in actor.parameters()),prefix_shapes_no_future=all(q.ca.shape==(q.t,3)and q.t<q.N for q in states))
for update in range(1,1 if os.environ.get("CRISP_VERIFY_ONLY")=="1" else 81):
 if time.monotonic()-start>cfg['max_GPU_worker_seconds']:raise RuntimeError('bounded pilot walltime')
 row,sel=step();curve.append(dict(update=update,**row));seen.update(int(i)for i in sel);indices.extend(int(i)for i in sel)
 if update==40:
  q=saved(40);tmp=R/'runs/adapter_00040.pt.part';torch.save(q,tmp);tmp.replace(R/'runs/adapter_00040.pt')
  a1,_=step();a2,_=step();after=saved(42);restore(torch.load(R/'runs/adapter_00040.pt',weights_only=False));b1,_=step();b2,_=step();other=saved(42)
  def same(a,b):
   if torch.is_tensor(a):return torch.equal(a,b)
   if isinstance(a,dict):return a.keys()==b.keys()and all(same(a[k],b[k])for k in a)
   if isinstance(a,(list,tuple)):return len(a)==len(b)and all(same(x,y)for x,y in zip(a,b))
   return a==b
  tests['GPU_restart_adapter_and_AdamW_bitwise']=same(after['adapter'],other['adapter'])and same(after['optimizer'],other['optimizer'])and a1==b1 and a2==b2;restore(q)
 if update%20==0:print('ADAPTER',update,row,flush=True)
if os.environ.get('CRISP_VERIFY_ONLY')=='1':
 q=torch.load(R/'runs/adapter_00080.pt',weights_only=False);restore(q);curve=q['history'];seen=set(q['seen']);indices=q['sample_indices']
else:
 q=saved(80);tmp=R/'runs/adapter_00080.pt.part';torch.save(q,tmp);tmp.replace(R/'runs/adapter_00080.pt')
adapter.eval();tests['J1_parameters_and_buffers_bitwise_unchanged']=tensor_sha(actor)==baseline_hash
if os.environ.get('CRISP_VERIFY_ONLY')=='1':tests['GPU_restart_adapter_and_AdamW_bitwise']=read('results/INITIAL_DIRECT_STEERING_TESTS.json')['tests']['GPU_restart_adapter_and_AdamW_bitwise']
# Joint distribution KL verified against MC in physical units at a TRAIN prefix.
l,m,s=adapter(h[0],c[0],ref_l[0],ref_m[0],ref_s[0]);ka,kg,_=kl(l,m,s,ref_l[0],ref_m[0],ref_s[0]);prob=l.softmax(-1);dist=GeometryDistribution(m[0].double(),s[0].double(),(True,True,True));ref=GeometryDistribution(ref_m[0,0].double(),ref_s[0,0].double(),(True,True,True));latent=dist.mu+dist.log_std.exp()*torch.randn(40000,3,device='cuda',dtype=torch.float64);physical=torch.stack([latent[:,0].exp(),pi*latent[:,1].sigmoid(),pi*latent[:,2].tanh()],-1);diff=(dist.base.log_prob(latent)-ref.base.log_prob(latent)).sum(-1);analytic=gaussian_kl(dist,ref);tests['continuous_KL_matches_MC']=abs(float(diff.mean()-analytic))<.01;tests['continuous_support']=bool((physical[:,0]>0).all()and((physical[:,1]>0)&(physical[:,1]<pi)).all()and(physical[:,2].abs()<pi).all());tests['categorical_KL_correct']=bool(torch.allclose(ka,(prob*(l.log_softmax(-1)-ref_l[0].log_softmax(-1))).sum(),atol=1e-7))
z=dist.transform(latent[0]);manual=(dist.base.log_prob(latent[0])-dist.log_jacobian(latent[0])).sum();tests['physical_logprob_Jacobian_inverse']=abs(float(dist.log_prob(z)-manual))<1e-8
# Differentiable physical action to CA builder.
eps=torch.tensor([.1,-.2,.3],device='cuda',dtype=torch.float64);u0=dist.mu+dist.log_std.exp()*eps;action=dist.transform(u0);state=state_for(train[0]);coord=next_ca(state.ca,action);gr=torch.autograd.grad(coord.square().sum(),list(adapter.G.parameters()),retain_graph=True,allow_unused=True);tests['geometry_reparameterized_builder_gradient']=any(v is not None and float(v.abs().sum())>0 for v in gr)
# Counterfactual change only explicit goals, not hidden native arrays.
s0=state_for(train[0]);task2=copy.deepcopy(train[0]);task2['endpoint'][0]+=20;task2['motifs'][str(s0.t)]=AA[(AA.index(s0.residues[-1])+5)%20];cf=context(s0,task2);basec=context(s0,train[0]);hh,ll,mm,ss=reference(actor,[BuildState(s0.residues,s0.ca.cpu(),s0.N)]);o1=adapter(hh[0],basec,ll[0],mm[0],ss[0]);o2=adapter(hh[0],cf,ll[0],mm[0],ss[0]);tests['goals_change_AA_and_geometry']=float((o1[0]-o2[0]).abs().sum())>0 and float((o1[1]-o2[1]).abs().sum())>0
# Discrete AA estimator uses exact expectation, never integer pathwise gradients.
qvals=torch.arange(20,device='cuda',dtype=torch.float32);pg=(o1[0].softmax(-1)*qvals).sum();pa=torch.autograd.grad(pg,list(adapter.A.parameters()),allow_unused=True);tests['categorical_exact_expectation_nonzero_gradient']=any(v is not None and float(v.abs().sum())>0 for v in pa)
# Rewind deterministic replay, then branched regeneration; serialization recovery.
env=Hierarchy(actor,adapter,train[0],8301);env.act('CONTINUE',steps=8);savedca=env.state.ca.clone();saveds=env.state.residues;env.act('REWIND',position=6);env.act('CONTINUE',steps=5);tests['rewind_same_stream_replays_suffix']=torch.equal(savedca,env.state.ca)and saveds==env.state.residues
env.act('REWIND',position=6);env.act('REGENERATE');env.act('CONTINUE',steps=5);tests['regenerate_branch_changes_decisions']=not torch.equal(savedca,env.state.ca)
export=copy.deepcopy(env.export());path=R/'runs/ENV_RECOVERY.pt';torch.save(export,path);env.act('CONTINUE',steps=2);first=env.state.ca.clone();clone=Hierarchy(actor,adapter,train[0],1);clone.restore(torch.load(path,weights_only=False));clone.act('CONTINUE',steps=2);tests['environment_checkpoint_restart_exact']=torch.equal(first,clone.state.ca)and env.state.residues==clone.state.residues
# Native witness demonstrates feasible specified targets without furnishing the full chain to policy.
witness=[]
for task in all_tasks:
 n=native[task['identity']];s=BuildState(tuple(n['sequence']),torch.tensor(n['CA'],dtype=torch.float64),task['N']);witness.append(dict(task=task['id'],split=task['split'],assessment=assess(s,task,limits)))
write('NATIVE_REACHABILITY_WITNESSES.json',witness);tests['TRAIN_native_goal_witnesses_pass']=all(r['assessment']['success']for r in witness if r['split']=='TRAIN')
# Potential telescopes under discount, independent of retry count; terminal potential0.
phis=[-.7,-.2,-.4,0.];gamma=.995;lhs=sum(gamma**i*shaping(phis[i],phis[i+1],gamma,1.)for i in range(3));tests['potential_telescoping']=abs(lhs-(-phis[0]+gamma**3*phis[-1]))<1e-12
write('DIRECT_STEERING_TESTS.json',dict(tests=tests,all_pass=all(tests.values()),parameter_inventory=params,adapter_parameters=53530,J1_parameters=12351746,exact_joint_KL=float(ka+kg),conditional_geometry_KL_analytic=float(analytic),conditional_geometry_KL_MC=float(diff.mean()),frozen_tensor_sha256=baseline_hash,adapter_only_unique_updates=80,discarded_restart_update_calls=4,no_J1_optimizer=True))
write('ADAPTER_TRAINING_CURVE.json',dict(rows=curve,feature_examples=len(records),selected_feature_indices=indices,unique_feature_indices=len(seen),unique_identity_position_pairs=len({(records[i]['identity'],records[i]['t'])for i in seen}),TRAIN_identities=len({records[i]['identity']for i in seen}),feature_manifest=records,teacher_forced_only=True,geometry_coefficient=coefficient,seconds=time.monotonic()-start,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),J1_frozen_tensor_sha256=baseline_hash))
assert all(tests.values()),tests
print('CORRECTNESS PASS',tests,flush=True)
