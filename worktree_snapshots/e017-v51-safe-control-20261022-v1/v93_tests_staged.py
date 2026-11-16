"""CPU-only exact finite-trace/continuous-gradient/engineering-learning tests."""
import copy,itertools,math,runpy,time
from pathlib import Path
import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import ndtr,expit
R=Path('/mnt/d/Simone/proteinGen_data/e017_v93_regional_controllability_20261024_v1');d=runpy.run_path(str(R/'source/scripts/e017_v93/common.py'));save=d['save'];ns=d['ns'];T=runpy.run_path(str(R/'source/scripts/e017_v93/trace.py'));torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
paths=[]
for reject in range(3):
 for accept in [1,2]:paths.append(tuple([0]*reject+[accept]))
paths.append((0,0,0));traces=[]
for a in paths:
 if a[-1]==0:traces.append((a,(),0.))
 else:
  for b in paths:
   reward=1. if a[-1]==2 and b[-1]==1 else .2 if a[-1]==1 and b[-1]==2 else 0.;traces.append((a,b,reward))
def evaluate(theta,wrong=False):
 lp=theta.log_softmax(-1);logs=[]
 for a,b,_ in traces:
  if wrong:logs.append(sum([lp[0,a[-1]]] if a[-1] else [lp[0,0]*0])+sum([lp[1,b[-1]]] if b and b[-1] else []))
  else:logs.append(lp[0,list(a)].sum()+ (lp[1,list(b)].sum() if b else lp[0,0]*0))
 return torch.stack(logs)
def main():
 start=time.perf_counter();torch.manual_seed(9317);rng=np.random.default_rng(9317);theta=torch.tensor([[.3,-.1,.2],[-.2,.4,.1]],requires_grad=True);rewards=torch.tensor([r for a,b,r in traces]);logs=evaluate(theta);prob=logs.exp();assert abs(float(prob.sum())-1)<1e-12;objective=(prob*rewards).sum();exact=torch.autograd.grad(objective,theta,retain_graph=True)[0];score=torch.autograd.grad((prob.detach()*rewards*logs).sum(),theta,retain_graph=True)[0];assert torch.allclose(exact,score,atol=1e-12)
 wronglogs=evaluate(theta,True);wrong=torch.autograd.grad((prob.detach()*rewards*wronglogs).sum(),theta)[0];assert (wrong-exact).abs().max()>1e-3
 finite=np.zeros((2,3));eps=1e-5
 for i in range(2):
  for j in range(3):
   a=theta.detach().clone();b=a.clone();a[i,j]+=eps;b[i,j]-=eps;finite[i,j]=float(((evaluate(a).exp()*rewards).sum()-(evaluate(b).exp()*rewards).sum())/(2*eps))
 assert np.max(np.abs(finite-exact.numpy()))<1e-8
 # Full trace KL = expected proposal KL at all reached nodes, upper-bounds
 # KL of accepted/fallback outputs (data processing).
 ref=torch.zeros((2,3));rl=evaluate(ref);traceKL=(prob.detach()*(logs.detach()-rl)).sum();kl=(theta.detach().softmax(-1)*(theta.detach().log_softmax(-1)-ref.log_softmax(-1))).sum(-1);counts=torch.tensor([[len(a),len(b)] for a,b,_ in traces],dtype=torch.float64);chain=(prob.detach()*(counts@kl)).sum();assert abs(float(traceKL-chain))<1e-12
 po={};qo={}
 for idx,(a,b,_) in enumerate(traces):
  key=(a[-1],b[-1] if b else -1);po[key]=po.get(key,0)+float(prob[idx]);qo[key]=qo.get(key,0)+float(rl[idx].exp())
 outKL=sum(v*math.log(v/qo[k]) for k,v in po.items());assert outKL<=float(traceKL)+1e-12
 # Independent Monte Carlo score-gradient check, exact standard errors.
 indices=rng.choice(len(traces),size=100000,p=prob.detach().numpy());scores=[]
 for a,b,reward in traces:
  z=np.zeros((2,3))
  for stage,actions in enumerate([a,b]):
   for action in actions:z[stage,action]+=1;z[stage]-=theta.detach().softmax(-1).numpy()[stage]
  scores.append(reward*z)
 samples=np.asarray(scores)[indices];mc=samples.mean(0);se=samples.std(0,ddof=1)/math.sqrt(len(samples));assert np.all(np.abs(mc-exact.numpy())<5*se+1e-5)
 # Continuous normal finite-rejection computation, including every rejected u.
 mu=-.25;n=150000;u=rng.normal(mu,1,(n,3));success=u>=0;first=np.where(success.any(1),success.argmax(1),2);active=np.arange(3)[None,:]<=first[:,None];selected=u[np.arange(n),first];Rwd=np.where(success.any(1),expit(selected),0);S=((u-mu)*active).sum(1);est=Rwd*S;mean=float(est.mean());stderr=float(est.std(ddof=1)/math.sqrt(n))
 def expected(m):
  integ=quad(lambda x:math.exp(-.5*(x-m)**2)/math.sqrt(2*math.pi)*expit(x),0,np.inf)[0];return integ*sum(ndtr(-m)**k for k in range(3))
 true=(expected(mu+1e-5)-expected(mu-1e-5))/2e-5;assert abs(mean-true)<5*stderr
 # Physical Jacobian and correct early-residue active-coordinate gradients.
 from protein_joint_sac.crisp_v3.density import log_prob
 from protein_joint_sac.models.geometry_inference import corrected_action
 maxerr=0.;masked=[];repaircheck=None
 for t in range(4):
  mask=[t>=1,t>=2,t>=3];muT=torch.tensor([1.3,.4,-.3],requires_grad=True);ls=torch.tensor([-.8,-.6,-.5],requires_grad=True);uT=muT.detach()+ls.detach().exp()*torch.tensor([.2,-.5,.8]);geo,diag=corrected_action(muT,ls,mask,torch.tensor([.2,-.5,.8]));raw=T['base_logprob'](muT,ls,uT,mask);physical=raw-T['logjac'](uT,mask);inverse=log_prob(muT,ls,geo,torch.tensor(mask));maxerr=max(maxerr,abs(float(physical-inverse)));gr=torch.autograd.grad(raw,muT)[0];assert all(mask[i] or gr[i]==0 for i in range(3));masked.append(gr.tolist())
 assert maxerr<1e-10
 extreme=torch.tensor([1.,50.,-50.]);geo,diag=corrected_action(extreme,torch.zeros(3),[1,1,1],torch.zeros(3));assert any(diag['adjusted_components']);assert diag['latent']==extreme.tolist();repaircheck=dict(raw_latent_retained=diag['latent'],adjusted=diag['adjusted_components'],note='trace uses Gaussian raw-u density; representative action density is not a sampling mass')
 # Same actual finite-rejection environment learns using delayed rewards.
 weights=torch.nn.Parameter(torch.zeros((2,3)));opt=torch.optim.Adam([weights],lr=.04);curve=[];initial=float((evaluate(weights).exp()*rewards).sum());replayed=False
 def update():
  l=evaluate(weights);ids=rng.choice(len(traces),size=256,p=l.detach().exp().numpy());r=rewards[ids];loss=T['trace_loss'](l[ids],evaluate(ref)[ids],r);opt.zero_grad();loss.backward();assert torch.isfinite(weights.grad).all();opt.step();return ids.copy()
 for step in range(1,251):
  if step==101:
   state=dict(weights=weights.detach().clone(),optimizer=copy.deepcopy(opt.state_dict()),numpy=copy.deepcopy(rng.bit_generator.state),torch=torch.get_rng_state(),update=100,config=dict(seed=9317,batch=256,max_rejections=3),traces=traces);path=R/'results/TOY_RECOVERY.pt';torch.save(state,path);expected_ids=update();w=weights.detach().clone();o=copy.deepcopy(opt.state_dict());next_rng=copy.deepcopy(rng.bit_generator.state);loaded=torch.load(path,weights_only=False);weights.data.copy_(loaded['weights']);opt.load_state_dict(loaded['optimizer']);rng.bit_generator.state=loaded['numpy'];torch.set_rng_state(loaded['torch']);actual_ids=update();assert np.array_equal(expected_ids,actual_ids) and torch.equal(weights,w) and d['c']['state_digest'](opt.state_dict())==d['c']['state_digest'](o) and rng.bit_generator.state==next_rng;replayed=True
  else:update()
  if step in [1,25,50,100,101,150,250]:curve.append(dict(update=step,expected_return=float((evaluate(weights).exp()*rewards).sum())))
 final=float((evaluate(weights).exp()*rewards).sum());assert final>initial+.5 and final>.9 and replayed
 save('POLICY_GRADIENT_TESTS.json',dict(PASS=True,seed=9317,exact_traces=len(traces),probability_sum=float(prob.sum()),exact_gradient=exact.tolist(),score_gradient=score.tolist(),wrong_accepted_only_gradient=wrong.tolist(),finite_difference_max_error=float(np.max(abs(finite-exact.numpy()))),MC_gradient=mc.tolist(),MC_standard_error=se.tolist(),MC_samples=100000,continuous_MC_samples=n,continuous_exact_gradient=true,continuous_MC_gradient=mean,continuous_SE=stderr,trace_KL=float(traceKL),chain_rule_KL=float(chain),accepted_output_KL=outKL,physical_density_max_error=maxerr,masks=masked,finite_grid_repair_test=repaircheck,toy_initial_return=initial,toy_final_return=final,toy_updates=250,curve=curve,recovery_bit_exact=True,toy_checkpoint_bytes=(R/'results/TOY_RECOVERY.pt').stat().st_size,protein_updates=0,seconds=time.perf_counter()-start));print('ALL_TESTS_PASS',initial,final,flush=True)
if __name__=='__main__':main()
