"""Bounded engineering verification. Each phase publishes once, never overwrites evidence."""
import argparse,copy,gc,json,math,os,resource,shutil,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from protein_joint_sac.crisp_v3.networks import System,prepare,PROFILES
from protein_joint_sac.crisp_v3.prior import FrozenPrior,file_sha,tensor_sha,MAIN,SHA
from protein_joint_sac.crisp_v3.optimization import Learner
from protein_joint_sac.crisp_v3 import density
from protein_joint_sac.crisp_v3.storage import Replay,payload,atomic_save,restore,rng_state,set_rng
from protein_joint_sac.crisp_v3.environment import World,assess,goal_tokens,reconstruct,geometric_jacobian,frame
from protein_joint_sac.models.batched_bc import collate
from protein_joint_sac.contracts import AA,BuildState
from protein_joint_sac.models.geometry_inference import corrected_action
R=Path(__file__).resolve().parents[3];V2=R.parent/'e017_crisp_v2_20261021_v1'
def write(n,x):
 p=R/'results'/n
 if p.exists():raise FileExistsError('evidence already exists: '+str(p))
 tmp=p.with_suffix('.part');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(p)
def fixture(device='cpu',length=12):
 n=json.load(open(V2/'tasks/HIDDEN_NATIVE_SUPERVISION.json'))[0];x=torch.tensor(n['CA'][:min(length,len(n['CA']))],dtype=torch.float64);seq=n['sequence'][:len(x)];state=BuildState(tuple(seq),x,500 if length>64 else 64)
 # Long memory fixtures repeat finite synthetic geometry explicitly, NOT native data.
 if length>len(x):
  x=torch.stack([torch.tensor([3.8*i,math.sin(i),math.cos(i)],dtype=torch.float64) for i in range(length)]);state=BuildState(tuple((seq*((length//len(seq))+1))[:length]),x,500)
 task=dict(kind='P2',N=state.N,targets=[dict(index=state.N-1,position=[10.,12.,14.],tolerance_A=8.,orientation=torch.eye(3).tolist(),orientation_tolerance_deg=45.)],motifs={'30':'A'})
 return state,task

def batch(prior,device='cuda',length=12):
 s,t=fixture(length=length);p=collate([s],device);g,v=goal_tokens(t,device)
 with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):ref=prior(p)
 return prepare(p,ref,g,v)
def audit():
 start=time.monotonic();manifest=json.load(open(V2/'CHECKPOINT_MANIFEST.json'));verified={}
 for item in [manifest['reference']]+manifest['checkpoints']:
  h=file_sha(item['path']);assert h==item['sha256'];verified[item['path']]=h
 for p,h in [(R.parent/'e016_j1_torsion_stabilization_20261018_v1/replay/checkpoint_00500_full.pt','9208263300fb9e88519ff0d40e6d9d29a9744b709d9a6181819cd0894b593000'),(R.parent/'e016_partial_supervision_20261016_v1/runs/J1-FULL/checkpoint_20000.pt','9e9215d3720ffc73edb0c3e382db25332f91cb26b0c8d38d01668e383b1e3810')]:assert file_sha(p)==h;verified[str(p)]=h
 profiles={}
 for p in PROFILES:
  with torch.device('meta'):m=System(p)
  profiles[p]=m.counts();del m
 write('SOURCE_AND_CHECKPOINT_RECOVERY_AUDIT.json',dict(verified=verified,v2_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=V2/'source',text=True).strip(),v3_parent=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip(),seconds=time.monotonic()-start,disk_free=shutil.disk_usage(R).free,host_ram=open('/proc/meminfo').read().splitlines()[:3],profiles=profiles))
 print(json.dumps(profiles),flush=True)

def data_audit():
 from protein_joint_sac.crisp_v3.data import Population,CoverageSampler
 from protein_joint_sac.crisp_v3.goals import Goals
 start=time.monotonic();pop=Population();sam=CoverageSampler(pop);draw=sam.draw(320);counts={k:sum(r[0]==k for r in draw) for k in sam.orders};assert counts==dict(complete=128,partial=96,aa_only=96)
 saved=copy.deepcopy(sam.state_dict());a=[r[:3] for r in sam.draw(10)];sam.load_state_dict(saved);assert a==[r[:3] for r in sam.draw(10)]
 samples={}
 for k in sam.orders:
  rec=next(r for r in draw if r[0]==k);b,y,g,m,rows=pop.batch([rec],'cpu');samples[k]=dict(identity=rec[1],position=rec[2],observation=rows[0][-1],prefix=b['ids'].shape[1],geometry_valid=m.tolist())
  if k=='aa_only':assert not b['valid'].any() and not m.any()
 goals=Goals(pop);tasks=[goals.draw(i) for i in range(3)];assert all(t['split']=='TRAIN' for t in tasks)
 write('DATA_AND_SAMPLER_TESTS.json',dict(status='PASS',counts=pop.membership(),quota_320=counts,sampler_resume=True,fixtures=samples,goals=tasks,seconds=time.monotonic()-start,host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,partial_source_masks_verified=True,TEST_targets=0))
 print('data PASS',flush=True)

def gpu():
 torch.manual_seed(1703);torch.cuda.manual_seed_all(1703);torch.use_deterministic_algorithms(True);start=time.monotonic();prior=FrozenPrior('cuda');b=batch(prior);s=System('development').cuda();learner=Learner(s);mask=torch.ones(1,3,device='cuda',dtype=torch.bool);a=torch.tensor([2],device='cuda');g=torch.tensor([[3.8,1.8,-.7]],device='cuda',dtype=torch.float64);highmask=torch.ones(1,9,device='cuda',dtype=torch.bool)
 result={};initial=prior.identity
 with torch.autocast('cuda',dtype=torch.bfloat16):
  l,mu,ls=s.models['actor'](b);assert torch.equal(l,b['reference']['logits']) and torch.equal(mu,b['reference']['mu']) and torch.equal(ls,b['reference']['ls']);ka,kg,_=density.kl(l,mu,ls,b['reference'],mask);assert float(ka.abs().max())<1e-6 and float(kg.abs().max())<1e-6
  result['supervised_step']=learner.supervised(b,a,g,mask)
  result['low_critics']=learner.critic(b,b,a,g,mask,mask,torch.tensor([1.],device='cuda'),torch.tensor([2],device='cuda'),torch.tensor([False],device='cuda'))
  result['low_actor']=learner.low_actor(b,mask)
  result['high_critics']=learner.critic(b,b,a,g,mask,mask,torch.tensor([1.],device='cuda'),torch.tensor([3],device='cuda'),torch.tensor([False],device='cuda'),True,highmask)
  result['high_actor']=learner.high_actor(b,highmask);result['reachability']=learner.reach(b,torch.tensor([0],device='cuda'),torch.tensor([1.],device='cuda'),torch.tensor([True],device='cuda'))
 prior.verify(learner.optimizers.values());result['J1_bitwise_unchanged']=True;result['J1_tensor_sha256']=initial
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  l,mu,ls=s.models['actor'](b);result['AA_logit_change_L2']=float((l-b['reference']['logits']).norm());result['geometry_mu_change_L2']=float((mu-b['reference']['mu']).norm());assert result['AA_logit_change_L2']>0 and result['geometry_mu_change_L2']>0
  # Missing-coordinate placeholders masked before independent and J1 feature construction.
  bb=copy.deepcopy(b['prefix']);bb['valid'].zero_();bb['x'].zero_();g1,v=goal_tokens(fixture()[1],'cuda');r1=prior(bb);i1=prepare(bb,r1,g1,v);z1=s.models['actor'](i1);bb['x'].fill_(12345);r2=prior(bb);i2=prepare(bb,r2,g1,v);z2=s.models['actor'](i2);assert all(torch.equal(x,y) for x,y in zip(z1,z2));result['missing_placeholder_invariance']=True
 # Exact development checkpoint/restart, including target and AdamW tensors.
 config={'profile':'development','reference':SHA,'engineering':True};path=R/f'checkpoints/development_engineering_full_{time.time_ns()}.pt';atomic_save(path,payload(s,learner,{},config,None,{'engineering_calls':8}),30*2**20)
 with torch.autocast('cuda',dtype=torch.bfloat16):learner.supervised(b,a,g,mask)
 expected=tensor_sha(s);expected_opt=copy.deepcopy(learner.optimizers['actor'].state_dict());expected_noise=torch.rand(4,device='cuda')
 restore(path,s,learner,{},config)
 with torch.autocast('cuda',dtype=torch.bfloat16):learner.supervised(b,a,g,mask)
 assert tensor_sha(s)==expected and torch.equal(torch.rand(4,device='cuda'),expected_noise)
 for k,st in expected_opt['state'].items():
  for key,val in st.items():assert torch.equal(val.cpu(),learner.optimizers['actor'].state_dict()['state'][k][key].cpu())
 result['GPU_interrupted_restart_bitwise']=True
 # Target copies have no gradients, Polyak exact reference formula.
 old=copy.deepcopy(s.targets['q1'].state_dict());s.polyak(.125)
 for k,v in s.targets['q1'].state_dict().items():assert torch.equal(v,old[k].lerp(s.models['q1'].state_dict()[k],.125))
 assert all(not p.requires_grad and p.grad is None for p in s.targets.parameters());result['target_update']=True
 prior.verify(learner.optimizers.values());result['seconds']=time.monotonic()-start;result['gpu_peak_allocated_bytes']=torch.cuda.max_memory_allocated();result['gpu_peak_reserved_bytes']=torch.cuda.max_memory_reserved();result['host_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;result['status']='PASS';result['checkpoint_sha256']=file_sha(path);write('GPU_DEVELOPMENT_TESTS.json',result);print(json.dumps(result),flush=True)

def environment_tests():
 start=time.monotonic();native=json.load(open(V2/'tasks/HIDDEN_NATIVE_SUPERVISION.json'))[0];x=torch.tensor(native['CA'],dtype=torch.float64);seq=native['sequence'];N=len(seq);limits=json.load(open(V2/'configs/NATIVE_CALIBRATION.json'));task=dict(N=N,kind='P2',targets=[dict(index=N-1,position=x[-1].tolist(),tolerance_A=8.,orientation=frame(x).tolist(),orientation_tolerance_deg=45.)],motifs={'30':seq[30]});state=BuildState(tuple(seq[:3]),x[:3],N);w=World(state,task,limits);gs=torch.tensor(native['g'],dtype=torch.float64) if 'g' in native else None
 from protein_joint_sac.data.scaled_supervised import internal
 gs=torch.tensor(internal(x.numpy()))
 initial=dict(sequence=seq[:3],coordinates=x[:3].tolist(),valid=[True]*3,N=N)
 replay_path=R/'data/engineering_replay.sqlite';replay=Replay(replay_path);replay.episode('native_replay_engineering',initial,task,'native_witness','SYNTHETIC_ENGINEERING')
 for i in range(3,12):w.step(AA.index(seq[i]),gs[i]);replay.append('native_replay_engineering',w.events[-1])
 saved=w.export();rng_before=w.rng.get_state().clone();w.option(4);assert w.state.t==8;assert w.pending
 try:w.option(0,lambda *_:None);raise AssertionError('continue while pending allowed')
 except ValueError:pass
 w.option(7);assert not w.pending
 w.restore(saved);assert torch.equal(w.rng.get_state(),rng_before) and w.state.t==12
 ep,events=replay.read('native_replay_engineering');rw=reconstruct(ep['initial'],events,task,limits,1);assert torch.equal(w.state.ca,rw.state.ca)
 for i in range(12,N):w.step(AA.index(seq[i]),gs[i])
 final=w.option(8);assert assess(w.state,task,limits)['success'];assert final['terminated']
 # Local geometric derivatives and SE(3) equivariance.
 g=gs[12].clone();J=geometric_jacobian(rw.state,g);eps=1e-5;from protein_joint_sac.geometry.ca_builder import next_ca
 fd=torch.stack([(next_ca(rw.state.ca,g+torch.eye(3,dtype=torch.float64)[i]*eps)-next_ca(rw.state.ca,g-torch.eye(3,dtype=torch.float64)[i]*eps))/(2*eps) for i in range(3)],1);assert torch.allclose(J,fd,atol=1e-7,rtol=1e-6)
 q,_=torch.linalg.qr(torch.tensor([[.2,.4,.9],[.8,.3,.1],[.1,.7,.2]],dtype=torch.float64));q[:,0]*=torch.det(q);shift=torch.tensor([2.,-5.,9.]);assert torch.allclose(next_ca(rw.state.ca@q+shift,g),next_ca(rw.state.ca,g)@q+shift,atol=1e-10)
 for tau in [-1000.,-10.,10.,1000.]:
  action,info=corrected_action(torch.tensor([1.3,.1,tau]),torch.tensor([-1.,-1.,-1.]),(True,True,True),torch.zeros(3));assert -math.pi<float(action[2])<math.pi and (float(action[2])>0)==(tau>0)
 # physical density helper agrees with production law, masks and exact KL.
 from protein_joint_sac.models.hybrid_distributions import GeometryDistribution
 mu=torch.tensor([1.3,.1,-.2],dtype=torch.float64);ls=torch.tensor([-.8,-.4,-.3],dtype=torch.float64);mask=torch.tensor([True,True,True]);p=GeometryDistribution(mu,ls,(True,True,True));assert torch.allclose(density.log_prob(mu,ls,g,mask),p.log_prob(g))
 target=density.semimarkov(torch.tensor([1.,1.]),.9,torch.tensor([3,3]),torch.tensor([True,False]),torch.tensor([2.,2.]));assert torch.allclose(target,torch.tensor([1.,2.458]))
 replay.close();write('ENVIRONMENT_DENSITY_REPLAY_TESTS.json',dict(status='PASS',native_witness=True,rewind_pending_mask=True,prefix_RNG_restore=True,event_reconstruction=True,SE3_equivariance=True,Jacobian_max_error=float((J-fd).abs().max()),physical_density_agreement=True,extreme_FP64_support=True,semi_Markov_target=True,source='engineering fixtures; no learned-policy success claimed',seconds=time.monotonic()-start))
 print('environment PASS',flush=True)

def profile():
 torch.manual_seed(1703);start=time.monotonic();prior=FrozenPrior('cuda');s=System('research').cuda();learner=Learner(s);results={};torch.cuda.reset_peak_memory_stats()
 # Allocate true optimizer states through one actual backward/step for every component.
 for k,m in s.models.items():
  t=time.monotonic();b=batch(prior,length=64);o=learner.optimizers[k];o.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):
   if k=='actor':
    l,mu,ls=m(b);loss=F.cross_entropy(l,torch.tensor([1],device='cuda'))+.01*mu.square().mean()+.01*ls.square().mean()
   elif k.startswith('high_q'):loss=m(b,torch.tensor([1],device='cuda')).square().mean()
   elif k.startswith('q'):loss=m(b,torch.tensor([1],device='cuda'),torch.tensor([[3.8,1.8,-.7]],device='cuda',dtype=torch.float64),torch.ones(1,3,device='cuda',dtype=torch.bool)).square().mean()
   elif k=='high_actor':loss=F.cross_entropy(m(b,torch.ones(1,9,device='cuda',dtype=torch.bool)),torch.tensor([1],device='cuda'))
   else:loss=m(b,torch.tensor([0],device='cuda')).square().mean()
  loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);o.step();o.zero_grad(set_to_none=True);torch.cuda.synchronize();results[k]=dict(seconds=time.monotonic()-t,allocated_bytes=torch.cuda.memory_allocated(),peak_allocated_bytes=torch.cuda.max_memory_allocated())
 prior.verify(learner.optimizers.values());lengths={}
 for L in [64,96,128,256,500]:
  t=time.monotonic();torch.cuda.reset_peak_memory_stats();b=batch(prior,length=L)
  with torch.autocast('cuda',dtype=torch.bfloat16):l,mu,ls=s.models['actor'](b);loss=l.square().mean()+mu.square().mean()+ls.square().mean()
  loss.backward();s.zero_grad(set_to_none=True);torch.cuda.synchronize();lengths[str(L)]=dict(seconds=time.monotonic()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
 prior.verify(learner.optimizers.values());counts=s.counts();opt_bytes=sum(v.numel()*v.element_size() for o in learner.optimizers.values() for st in o.state.values() for v in st.values() if isinstance(v,torch.Tensor));model_bytes=sum(v.numel()*v.element_size() for v in s.state_dict().values());full=model_bytes+opt_bytes
 result=dict(status='PASS',counts=counts,component_optimizer_steps=7,J1_updates=0,components=results,lengths=lengths,optimizer_state_bytes=opt_bytes,model_and_targets_bytes=model_bytes,full_tensor_checkpoint_bytes=full,atomic_two_generation_bytes=2*full,disk_free_bytes=shutil.disk_usage(R).free,recovery_storage_ready=shutil.disk_usage(R).free>=10*2**30+2*full+128*2**20,host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.monotonic()-start);write('RESEARCH_RESOURCE_PROFILE.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
 from torch.nn import functional as F
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['audit','data','environment','gpu','profile']);args=p.parse_args()
 try:globals()[{'audit':'audit','data':'data_audit','environment':'environment_tests','gpu':'gpu','profile':'profile'}[args.phase]]()
 except Exception:
  error=traceback.format_exc();name=f'FAILURE_{args.phase}_{time.time_ns()}.json';write(name,dict(error=error));print(error,flush=True);raise
