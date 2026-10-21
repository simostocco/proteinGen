"""Integration and negative tests added after the initial verified component tests."""
import copy,json,math,random,time,uuid
from pathlib import Path
import torch
from torch.nn import functional as F
from engineering import R,V2,write,fixture,batch
from protein_joint_sac.crisp_v3.prior import FrozenPrior,tensor_sha
from protein_joint_sac.crisp_v3.networks import System,prepare
from protein_joint_sac.crisp_v3.optimization import frozen
from protein_joint_sac.crisp_v3.density import sample,kl,log_prob
from protein_joint_sac.crisp_v3.environment import World,goal_tokens,assess,frame,reconstruct
from protein_joint_sac.crisp_v3.causal import intervene,influence_features,ReachabilityOutcome
from protein_joint_sac.crisp_v3.storage import Replay,digest,payload,atomic_save,restore
from protein_joint_sac.crisp_v3.pipeline import collect,replay_example
from protein_joint_sac.crisp_v3.training import authorization,geometric_proximity
from protein_joint_sac.models.batched_bc import collate
from protein_joint_sac.contracts import AA,BuildState
from protein_joint_sac.data.scaled_supervised import internal

start=time.monotonic();torch.manual_seed(910);prior=FrozenPrior('cuda');s=System('development').cuda();b=batch(prior);mask=torch.ones(1,3,dtype=torch.bool,device='cuda');result={}
# Independent AA/geometry gradients, no geometry/AA-specific coupling except via encoder.
m=s.models['actor'];l,mu,ls=m(b);F.cross_entropy(l,torch.tensor([3],device='cuda')).backward();assert m.aa.weight.grad.abs().sum()>0;assert m.geometry[-1].weight.grad is None or m.geometry[-1].weight.grad.abs().sum()==0
m.zero_grad(set_to_none=True);l,mu,ls=m(b);(-log_prob(mu[:,3],ls[:,3],torch.tensor([[3.8,1.8,.5]],device='cuda'),mask)).mean().backward();assert m.geometry[-1].weight.grad.abs().sum()>0 and m.aa.weight.grad is None;m.zero_grad(set_to_none=True);result['independent_heads']=True
# Give the formerly zero heads tiny deterministic, nonzero steering, without optimizer steps.
with torch.no_grad():m.aa.weight.normal_(0,.002);m.geometry[-1].weight.normal_(0,.002)
l,mu,ls=m(b);q=copy.deepcopy(b);q['goals'][:,:,:3]+=.5;la,ma,sa=m(q);assert (l-la).abs().max()>0 and (mu-ma).abs().max()>0;result['goal_sensitivity']=dict(AA_max=float((l-la).abs().max().detach()),mu_max=float((mu-ma).abs().max().detach()))
# Original next-AA prefix cannot access a suffix: only prefix tensors exist in encoder schema.
assert set(b['prefix'])=={'x','valid','ids','t','N'};result['prefix_only_schema']=True
# Correct exact categorical expectation, streaming versus a single summed graph with same noise.
with frozen([s.models['q1'],s.models['q2']]):
 with torch.no_grad():h=[s.models[k].encoder(b) for k in ['q1','q2']]
 def term(a):
  logits,mu,ls=m(b);lp=logits.log_softmax(-1);p=lp.exp();ids=torch.tensor([a],device='cuda');g,lg=sample(mu[:,a],ls[:,a],mask,torch.zeros(1,3,device='cuda'));value=torch.minimum(*[s.models[k].candidate(x,ids,g,mask) for k,x in zip(['q1','q2'],h)]);return (p[:,a]*(.01*lp[:,a]+.001*lg-value)).mean()
 sum(term(a) for a in range(20)).backward();expected={k:p.grad.clone() for k,p in m.named_parameters() if p.grad is not None};m.zero_grad(set_to_none=True)
 for a in range(20):term(a).backward()
 error=max(float((p.grad-expected[k]).abs().max()) for k,p in m.named_parameters() if k in expected);assert error<2e-5;result['streamed_exact_expectation_gradient_max_error']=error;m.zero_grad(set_to_none=True)
# Monte Carlo KL verifies physical and latent conventions, not differential entropy as categorical entropy.
mu=torch.tensor([[1.3,.1,-.3]],device='cuda',dtype=torch.float64);ls=torch.tensor([[-.7,-.2,-.4]],device='cuda',dtype=torch.float64);rmu=mu+.1;rls=ls+.03
G,LP=sample(mu.expand(12000,3),ls.expand(12000,3),mask.expand(12000,3));estimate=(LP-log_prob(rmu,rls,G,mask)).mean();exact=(rls-ls+.5*((ls.exp()/rls.exp()).square()+((mu-rmu)/rls.exp()).square())-.5).sum();assert abs(float(estimate-exact))<.015;result['KL_MC']=dict(exact=float(exact),estimate=float(estimate))
# Read-only native witnesses for all task classes, with strict SE3 goals in same frame.
native=json.load(open(V2/'tasks/HIDDEN_NATIVE_SUPERVISION.json'))[0];seq=native['sequence'];x=torch.tensor(native['CA'],dtype=torch.float64);N=len(seq);g=torch.tensor(internal(x.numpy()));limits=json.load(open(V2/'configs/NATIVE_CALIBRATION.json'));task=dict(N=N,kind='P2',targets=[dict(index=N-1,position=x[-1].tolist(),orientation=frame(x).tolist(),tolerance_A=8.,orientation_tolerance_deg=45.)],motifs={'30':seq[30]})
for kind in ['P0','P1','P2']:
 t=copy.deepcopy(task);t['kind']=kind
 if kind=='P0':t['targets']=[]
 if kind=='P1':t['targets'][0]['orientation']=None
 assert assess(BuildState(tuple(seq),x,N),t,limits)['success']
result['P0_P1_P2_native_witnesses']=True
# Controlled suffix intervention and local influence features. Native replay here is an engineering continuation.
w=World(BuildState(tuple(seq[:8]),x[:8],N),task,limits,seed=11)
def cont(world,delta):
 i=world.state.t;action=g[i].clone()
 if delta is not None:action+=delta
 world.step(AA.index(seq[i]),action)
effect=intervene(w,torch.tensor([0.,0.,.2],dtype=torch.float64),cont,4);f=influence_features(effect);assert float(f[0])>0 and w.state.t==8;result['counterfactual_endpoint_shift_A']=float(f[0]);result['future_derivative_claimed']=False
try:ReachabilityOutcome('p','m','g',None,False,True).target();raise AssertionError('censored target admitted')
except ValueError:pass
# Real generated-fixture collection, persistent transition records, exact reconstruction and RNG.
rp=R/'data'/f'integration_replay_{time.time_ns()}.sqlite';replay=Replay(rp)
# Force CONTINUE followed by STOP on completion through logits; no model optimization.
with torch.no_grad():
 ha=s.models['high_actor'];ha.out.weight.zero_();ha.out.bias.fill_(-30);ha.out.bias[0]=30;ha.out.bias[8]=0
short=copy.deepcopy(task);short['N']=64;short['initial_sequence']=seq[:3];short['initial_CA']=x[:3].tolist();short['targets']=[];short['motifs']={}
ep=collect(s,prior,short,limits,replay,'engineering_policy',seed=912,max_events=8,option_steps=2);assert ep['censored'] and ep['event_count']==8
rows=[json.loads(r[0]) for r in replay.db.execute('SELECT data FROM transitions')];assert rows and all(r['behavior_logp'] is not None for r in rows)
record=next(r for r in rows if r['action']['level']=='low');bb,nb,worlds=replay_example(prior,replay,record,limits,'cuda');assert worlds[1].state.t==worlds[0].state.t+1
meta,events=replay.read(ep['episode']);rw=reconstruct(meta['initial'],events,short,limits,912);assert rw.state.t==11
# Replay RNG reproduces serial consumed sequence of AA-uniform + Normal(3) draws.
rng=torch.Generator().manual_seed(912)
for _ in range(8):torch.rand((),generator=rng);torch.randn(3,dtype=torch.float64,generator=rng)
assert torch.equal(rng.get_state(),rw.rng.get_state());result['generated_replay_RNG_reconstruction']=True
snapshot=R/'data'/f'integration_snapshot_{time.time_ns()}.sqlite';replay.snapshot(snapshot);replay.close();assert snapshot.stat().st_size>0
# No production authorization is present; dry-run and accidental execution are distinct.
try:authorization(None,{});raise AssertionError('production execution bypass')
except PermissionError:pass
prior.verify();result.update(status='PASS',seconds=time.monotonic()-start,optimizer_steps=0,replay_records=len(rows),episode=ep,production_authorization_gate=True,J1_bitwise_unchanged=True);write('EXTENDED_INTEGRATION_TESTS.json',result);print(json.dumps(result),flush=True)
