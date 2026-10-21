"""Bounded CRISP V3 engineering acceptance and measured target-capacity profile."""
from pathlib import Path
import argparse,copy,gc,json,os,random,resource,shutil,subprocess,time,traceback
import numpy as np
import torch
from protein_joint_sac.crisp_v3.networks import System,PROFILES,prepare
from protein_joint_sac.crisp_v3.prior import FrozenPrior,file_sha,tensor_sha,MAIN,SHA
from protein_joint_sac.crisp_v3.optimization import Learner
from protein_joint_sac.crisp_v3.density import log_prob,sample,kl,semimarkov
from protein_joint_sac.crisp_v3.environment import World,goal_tokens,assess,reconstruct,geometric_jacobian,reachable_bound,frame
from protein_joint_sac.crisp_v3.storage import Replay,payload,atomic_save,restore,digest
from protein_joint_sac.models.batched_bc import collate
from protein_joint_sac.models.geometry_inference import corrected_action
from protein_joint_sac.contracts import AA,BuildState
from protein_joint_sac.geometry.ca_builder import next_ca
R=Path(__file__).resolve().parents[3];V2=R.parent/'e017_crisp_v2_20261021_v1'
def write(n,x):
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.part');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def seed(n=17):random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)
def fixture(device='cuda',length=12):
 native=json.load(open(V2/'tasks/HIDDEN_NATIVE_SUPERVISION.json'))[0];x=torch.tensor(native['CA'],dtype=torch.float64);seq=native['sequence'];N=len(seq);t=min(length,N-1)
 state=BuildState(tuple(seq[:t]),x[:t].clone(),N);task=dict(kind='P2',N=N,targets=[dict(index=N-1,position=x[-1].tolist(),tolerance_A=8.,orientation=frame(x).tolist(),orientation_tolerance_deg=45.)],motifs={str(N//2):seq[N//2]})
 return state,task,torch.tensor(native['g'][t] if 'g' in native else [3.8,1.9,.2],device=device,dtype=torch.float64)
def batch(prior,length=12,device='cuda',B=1):
 s,task,g=fixture(device,length);p=collate([s]*B,device);goals,valid=goal_tokens(task,device)
 with torch.autocast(device,dtype=torch.bfloat16,enabled=device=='cuda'):ref=prior(p)
 b=prepare(p,ref,goals.expand(B,-1,-1),valid.expand(B,-1));mask=torch.arange(1,4,device=device)[None]<=p['t'][:,None];return b,mask,g[None].expand(B,3).clone(),task,s

def engineering():
 seed();torch.use_deterministic_algorithms(True);start=time.monotonic();prior=FrozenPrior('cuda');s=System('development').cuda();l=Learner(s);b,mask,g,task,state=batch(prior);aa=torch.tensor([AA.index('A')],device='cuda');hi=torch.ones(1,9,dtype=torch.bool,device='cuda');checks={};measure={};before=prior.identity
 checks['optimizer_J1_disjoint']=not any(id(p) in {id(z) for z in prior.model.parameters()} for o in l.optimizers.values() for pg in o.param_groups for p in pg['params'])
 with torch.no_grad():
  out=s.models['actor'](b);checks['zero_initialized_prior_equal']=all(torch.equal(a,b['reference'][k]) for a,k in zip(out,['logits','mu','ls']));ref0={k:v.clone() for k,v in prior(b['prefix']).items()}
 # Real GPU updates: all seven independent trainable components.
 with torch.autocast('cuda',dtype=torch.bfloat16):
  measure['supervised']=l.supervised(b,aa,g,mask)
  measure['low_critics']=l.critic(b,b,aa,g,mask,mask,torch.tensor([.4],device='cuda'),torch.tensor([3],device='cuda'),torch.tensor([False],device='cuda'))
  measure['high_critics']=l.critic(b,b,aa%9,g,mask,mask,torch.tensor([.2],device='cuda'),torch.tensor([2],device='cuda'),torch.tensor([True],device='cuda'),high=True,high_mask=hi)
  measure['high_actor']=l.high_actor(b,hi);measure['reach']=l.reach(b,torch.tensor([0],device='cuda'),torch.tensor([1.],device='cuda'),torch.tensor([True],device='cuda'))
  measure['low_actor']=l.low_actor(b,mask)
 prior.verify(l.optimizers.values());checks['J1_bitwise_frozen_all_parameters_and_buffers']=tensor_sha(prior.model)==before;checks['J1_gradients_absent']=all(p.grad is None for p in prior.model.parameters());checks['J1_predictions_bitwise_equal']=all(torch.equal(v,prior(b['prefix'])[k]) for k,v in ref0.items())
 # Both residual heads learn; goal-sensitive body gets nonzero gradient after first head update.
 s.zero_grad(set_to_none=True);out=s.models['actor'](b);loss=out[0].square().mean()+out[1].square().mean();loss.backward();checks['shared_RL_transformer_nonzero_gradient']=sum(float(p.grad.square().sum()) for p in s.models['actor'].encoder.parameters() if p.grad is not None)>0;checks['AA_and_geometry_heads_changed']=not torch.equal(out[0],b['reference']['logits']) and not torch.equal(out[1],b['reference']['mu']);s.zero_grad(set_to_none=True)
 # Missing-coordinate placeholders must not leak; all-known AA remains available.
 masked=copy.deepcopy(b['prefix']);masked['valid'].zero_();masked['x'].fill_(1e8)
 with torch.no_grad():r1=prior(masked);masked['x'].fill_(-3e8);r2=prior(masked)
 checks['missing_geometry_placeholder_invariance']=all(torch.equal(r1[k],r2[k]) for k in r1);checks['geometry_unavailable_finite']=all(torch.isfinite(v).all().item() for v in r1.values());checks['prefix_contains_no_future']=b['prefix']['ids'].shape[1]==state.t
 # Density / KL is common-bijection law, masks zero undefined components.
 mu=torch.tensor([[.8,.3,-.4]],device='cuda',dtype=torch.float64);ls=torch.tensor([[-1.,-.3,-.5]],device='cuda',dtype=torch.float64);full=torch.ones_like(mu,dtype=torch.bool);z,lp=sample(mu,ls,full,torch.tensor([[.1,-.2,.3]],device='cuda'));checks['inverse_density_finite']=bool(torch.isfinite(log_prob(mu,ls,z,full)).all());checks['physical_logp_roundtrip']=float((lp-log_prob(mu,ls,z,full)).abs().max())==0
 ref=dict(logits=torch.zeros(1,20,device='cuda'),mu=mu[:,None].expand(1,20,3).float(),ls=ls[:,None].expand(1,20,3).float());ka,kg,_=kl(ref['logits'],ref['mu'],ref['ls'],ref,full);checks['identical_KL_zero']=float(ka.abs().max()+kg.abs().max())==0
 from protein_joint_sac.models.hybrid_distributions import GeometryDistribution
 d=GeometryDistribution(mu[0],ls[0],(True,True,True));checks['historical_density_semantics_equal']=torch.allclose(d.log_prob(z[0]),lp[0],atol=1e-12,rtol=0)
 repairs=0
 for value in [-1000.,-10.,10.,1000.]:
  m=torch.tensor([1.3,0.,value],device='cuda');action,info=corrected_action(m,torch.zeros(3,device='cuda'),(True,True,True),torch.zeros(3,device='cuda'));assert -np.pi<float(action[2])<np.pi;repairs+=sum(info['adjusted_components'])
 checks['FP64_support_extremes']=True;measure['support_repairs']=repairs
 # True terminals zero bootstrap, operational truncation remains bootstrap-capable.
 target=semimarkov(torch.tensor([1.,1.]),.9,torch.tensor([2,2]),torch.tensor([True,False]),torch.tensor([10.,10.]));checks['semimarkov_terminal_duration']=torch.allclose(target,torch.tensor([1.,9.1]))
 old=[p.clone() for p in s.targets['q1'].parameters()];online=[p.clone() for p in s.models['q1'].parameters()];s.polyak(.25);checks['polyak_exact']=all(torch.equal(p,torch.lerp(a,c,.25)) for p,a,c in zip(s.targets['q1'].parameters(),old,online));checks['targets_no_grad']=all(not p.requires_grad and p.grad is None for p in s.targets.parameters())
 # Replay / intervention restoration and local Jacobian.
 limits=json.load(open(V2/'configs/NATIVE_CALIBRATION.json'));w=World(state,task,limits,seed=7);gcpu=g[0].cpu();init=dict(sequence=''.join(state.residues),coordinates=state.ca.tolist(),valid=state.valid.tolist(),N=state.N)
 for _ in range(5):w.step(0,gcpu)
 saved=w.export();w.option(4);w.option(7);w.step(1,gcpu);branch=w.export();w.restore(saved);checks['rewind_restore_exact']=torch.equal(w.state.ca,saved['state'].ca) and w.context_branch==saved['context_branch'];w.restore(branch)
 replay=Replay(R/'data/engineering_replay.sqlite');replay.episode('engineering',init,task,'engineering-v1','SYNTHETIC_ENGINEERING')
 for e in w.events:replay.append('engineering',e)
 ep,events=replay.read('engineering');rw=reconstruct(ep['initial'],events,ep['goal'],limits,7);checks['event_reconstruction']=torch.equal(rw.state.ca,w.state.ca) and rw.state.residues==w.state.residues;replay.snapshot(R/'data/engineering_replay_snapshot.sqlite');replay.close()
 J=geometric_jacobian(state,gcpu);eps=1e-6;fd=torch.stack([(next_ca(state.ca,gcpu+torch.eye(3,dtype=torch.float64)[i]*eps)-next_ca(state.ca,gcpu-torch.eye(3,dtype=torch.float64)[i]*eps))/(2*eps) for i in range(3)],-1);checks['local_Jacobian_finite_difference']=torch.allclose(J,fd,atol=1e-8,rtol=1e-7)
 # Deterministic GPU interruption/restart, all model/target/optimizer/scheduler state + RNG.
 config={'profile':'development','source':'engineering only','j1':SHA};path=R/'checkpoints/engineering_full.pt';state_dict=payload(s,l,{},config,str(R/'data/engineering_replay_snapshot.sqlite'),{'engineering':True});atomic_save(path,state_dict,64*2**20)
 def continuation():
  with torch.autocast('cuda',dtype=torch.bfloat16):return l.supervised(b,aa,g,mask)
 first=continuation();expected=tensor_sha(s);expected_optimizer=digest({k:len(o.state) for k,o in l.optimizers.items()});restore(path,s,l,{},config);second=continuation();checks['GPU_restart_model_target_bitwise_equal']=tensor_sha(s)==expected;checks['GPU_restart_loss_equal']=first==second
 prior.verify(l.optimizers.values());checks['J1_frozen_after_restart']=tensor_sha(prior.model)==before
 result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,measurements=measure,counts=s.counts(),seconds=time.monotonic()-start,checkpoint=str(path),checkpoint_sha256=file_sha(path),J1_tensor_sha256=before,GPU_peak_allocated=torch.cuda.max_memory_allocated(),GPU_peak_reserved=torch.cuda.max_memory_reserved(),host_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,optimizer_steps=l.steps)
 write('ENGINEERING_TESTS.json',result);print(json.dumps(result,indent=2));assert all(checks.values()),[k for k,v in checks.items() if not v]

def profile():
 seed();torch.cuda.reset_peak_memory_stats();start=time.monotonic();counts={}
 for name in PROFILES:
  with torch.device('meta'):counts[name]=System(name).counts()
 write('MODEL_PARAMETER_COUNTS.json',counts)
 prior=FrozenPrior('cuda');s=System('research').cuda();l=Learner(s);b,mask,g,task,state=batch(prior);aa=torch.zeros(1,dtype=torch.long,device='cuda');hi=torch.ones(1,9,dtype=torch.bool,device='cuda');out={}
 for k,m in s.models.items():
  torch.cuda.reset_peak_memory_stats();tick=time.monotonic();o=l.optimizers[k];o.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):
   if k=='actor':z=m(b);loss=F_loss(z[0])+F_loss(z[1])+F_loss(z[2])
   elif k.startswith('high_q'):loss=F_loss(m(b,aa))
   elif k.startswith('q'):loss=F_loss(m(b,aa,g,mask))
   elif k=='high_actor':loss=F_loss(m(b,hi))
   else:loss=F_loss(m(b,aa))
  loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);o.step();o.zero_grad(set_to_none=True);torch.cuda.synchronize()
  out[k]=dict(seconds=time.monotonic()-tick,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),gradient_norm=float(norm),optimizer_parameters=len(o.state))
 # Whole-capacity resident, all seven AdamW states allocated; lengths synthetic exact prefixes.
 lengths={}
 for N in [64,96,128,256,500]:
  p=copy.deepcopy(b['prefix']);ix=torch.arange(N-1,device='cuda');p.update(ids=(ix%20)[None],t=torch.tensor([N-1],device='cuda'),N=torch.tensor([N],device='cuda'),valid=torch.zeros(1,N-1,dtype=torch.bool,device='cuda'),x=torch.zeros(1,N-1,3,dtype=torch.float64,device='cuda'))
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):ref=prior(p)
  bb=prepare(p,ref,b['goals'],b['goal_valid']);torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
  with torch.autocast('cuda',dtype=torch.bfloat16):v=s.models['actor'](bb);loss=sum(F_loss(x) for x in v)
  loss.backward();s.zero_grad(set_to_none=True);torch.cuda.synchronize();lengths[N]=dict(seconds=time.monotonic()-tick,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
 prior.verify(l.optimizers.values());n=counts['research']['total_trainable'];tn=sum(counts['research']['target_nontrainable'].values());full_bytes=12*n+4*tn;free=shutil.disk_usage(R).free
 write('RESOURCE_PROFILE.json',dict(status='PASS_GPU_RESEARCH_MICROBATCH1',seconds=time.monotonic()-start,components=out,lengths=lengths,trainable_parameters=n,target_parameters=tn,estimated_full_checkpoint_tensor_bytes=full_bytes,two_generation_checkpoint_bytes=2*full_bytes,D_free_bytes=free,reserve_bytes=10*2**30,production_two_generation_storage_ready=free>=10*2**30+2*full_bytes+128*2**20,host_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,J1_unchanged=True,substantive_training_updates=0))
 print(json.dumps(out,indent=2))
def F_loss(x):return x.float().square().mean()
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['engineering','profile']);args=parser.parse_args()
 try:globals()[args.mode]()
 except Exception as e:
  write('results/'+args.mode+'_failure_'+str(int(time.time()))+'.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
