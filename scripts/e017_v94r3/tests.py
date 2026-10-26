import collections,hashlib,itertools,json,math,runpy,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v94r3_substantial_geometry_rl_20261026_v1');OLD=R.parent/'e017_v94r2_corrected_geometry_rl_20261025_v2'
F=runpy.run_path(str(R/'source/scripts/e017_v94r3/shield.py'));BASE=F['F'];S=BASE['OLD'];T=runpy.run_path(str(R.parent/'e017_v93_regional_controllability_20261024_v1/source/scripts/e017_v93/trace.py'))
checks=0
def check(ok):
 global checks
 assert ok;checks+=1

def enumeration(theta):
 pi=theta.softmax(0);ref=torch.tensor([.4,.35,.25],dtype=torch.float64);out=[]
 def stage(t,C,prob,lp,rlp,accepted_lp):
  if t==2:out.append((prob,lp,rlp,float(C==2),accepted_lp));return
  def draw(k,pr,ll,rr,aa):
   for a in range(3):
    p=pr*pi[a];l=ll+pi[a].log();r=rr+ref[a].log()
    # Same strengthened semantics: forbid a next-state dead end in addition
    # to the current budget. Next actions each add at least one contact.
    legal=C+a<=2 and not (t==0 and C+a+1>2)
    if legal:stage(t+1,C+a,p,l,r,aa+pi[a].log())
    elif k<7:draw(k+1,p,l,r,aa)
    else:out.append((p,l,r,0.,aa))
  draw(0,prob,lp,rlp,accepted_lp)
 z=theta.sum()*0;stage(0,0,z+1,z,z,z);return out

def main():
 global checks
 torch.set_num_threads(1);rng=np.random.default_rng(94337);torch.manual_seed(94337)
 # Source control forbids using any neural value/future coordinates in the shield.
 text=(R/'source/scripts/e017_v94r3/shield.py').read_text();check('native_future_coordinates=False' in text)
 check(F['coordinate_certificate'](np.zeros((64,3)),contacts=115)['status']=='TERMINAL_NO_EXTENSION_REQUIRED')
 check(not F['coordinate_certificate'](np.zeros((0,3)),contacts=0)['reject'])
 check(F['coordinate_certificate'](np.array([[np.nan,0,0]]),contacts=0)['status']=='UNKNOWN')
 check(F['coordinate_certificate'](np.array([[1e8,0,0]]),contacts=115)['status']=='UNKNOWN')
 # Boundary equality must not be treated as an unavoidable strict-radius contact.
 cut=8-F['DMAX']-F['GUARD_A'];x=np.zeros((9,3));x[0]=[cut,0,0];x[1]=[20,0,0]
 check(not F['coordinate_certificate'](x,contacts=115)['reject'])
 x[0,0]=cut-1e-6;check(F['coordinate_certificate'](x,contacts=115)['reject'])
 x[0,0]=cut+1e-6;check(not F['coordinate_certificate'](x,contacts=115)['reject'])
 x=np.zeros((9,3));check(not F['coordinate_certificate'](x,contacts=113)['reject']);check(F['coordinate_certificate'](x,contacts=114)['reject'])
 # Exact certificate invariance and necessary-condition checks on arbitrary
 # finite prefixes followed by arbitrary legally bounded Cartesian steps.
 for k in range(250):
  x=np.cumsum(rng.normal(size=(64,3))*rng.uniform(.2,5),axis=0)
  Q,_=np.linalg.qr(rng.normal(size=(3,3)));translation=rng.normal(size=3)*20
  for n in [1,8,16,32,48,63,64]:
   cert=F['coordinate_certificate'](x[:n]);rot=F['coordinate_certificate'](x[:n]@Q+translation)
   check(cert['reject']==rot['reject'])
   if cert['reject']:
    for _ in range(8):
     direction=rng.normal(size=3);direction/=np.linalg.norm(direction);y=x[n-1]+rng.uniform(2.8504200470168195,F['DMAX'])*direction
     xx=np.vstack([x[:n],y]);i=np.arange(n+1);D=np.linalg.norm(xx[:,None]-xx[None,:],axis=-1);C=int(((D<8)&(i[:,None]-i[None,:]>=8)).sum())
     check(C>115)
 # Historical development/fitting chains: retain every prefix of all complete
 # hard-valid controls; prove no known valid next action is excluded.
 preserved=0;certified=[]
 for u in [0,250,500]:
  for path in sorted((OLD/'production/diagnostics').glob(f'DEV_{u:05}_filtered_*.json')):
   packet=json.loads(path.read_text())
   for row in packet['records']:
    state=S['c']['BuildState'].empty(64)
    for t,geom in enumerate(row['geometry']):
     state=S['append'](state,packet['task']['fixed_sequence'][t],geom)
     out=F['feasibility'](state)
     if t+1<len(row['geometry']) or row['metrics']['hard_valid']:
      check(not out['reject']);preserved+=1
     if out['reject']:certified.append(dict(update=u,seed=row['seed'],candidate=row['candidate_index'],prefix_length=t+1))
 # Direct old/new rejection and FEASIBLE witness contract on fixed TRAIN fixtures.
 task=json.loads((OLD/'production/rollouts/u000001.json').read_text())['task'];state=S['c']['BuildState'].empty(64)
 after,cc=F['extend_checked'](state,task['fixed_sequence'][0],[0.,0.,0.]);check(cc['accepted'])
 check(F['feasibility'](after,next_aa=task['fixed_sequence'][1],witness_geometry=[3.8,0,0])['status']=='FEASIBLE_WITNESS')
 after2,cc=F['extend_checked'](after,task['fixed_sequence'][1],[9.,0,0]);check(not cc['accepted'] and 'local_spacing' in cc['reasons'])
 # Tractable finite8 normalization and unbiased complete-trace gradient.
 # Enumeration with two stages can be large: use two action categories by
 # aggregating rejected categories analytically per geometric series instead.
 theta=torch.tensor(.23,dtype=torch.float64,requires_grad=True);p=theta.sigmoid();refp=torch.tensor(.43,dtype=torch.float64);tr=[]
 for k in range(8):
  pr=(1-p)**k*p;lp=k*torch.log1p(-p)+p.log();rp=k*torch.log1p(-refp)+refp.log();tr.append((pr,lp,rp,1.))
 tr.append(((1-p)**8,8*torch.log1p(-p),8*torch.log1p(-refp),0.))
 check(abs(float(sum(x[0] for x in tr).detach())-1)<1e-13)
 beta=.03;objective=sum(pr*(reward-beta*(lp-rp)) for pr,lp,rp,reward in tr)
 exact=torch.autograd.grad(objective,theta,retain_graph=True)[0]
 estimator=sum(pr.detach()*((reward-beta*(lp-rp).detach())*lp-beta*(lp-rp)) for pr,lp,rp,reward in tr)
 estimated=torch.autograd.grad(estimator,theta)[0];check(abs(float(exact-estimated))<1e-12)
 # Continuous finite8 model with terminal reward: analytical score MC.
 mu=.3;eps=torch.randn(100000,8,dtype=torch.float64);values=mu+eps;ok=values<=0;accept=ok.any(1);first=ok.double().argmax(1);length=torch.where(accept,first+1,torch.full_like(first,8));mask=torch.arange(8)[None,:]<length[:,None]
 score=(eps*mask).sum(1);est=accept.double()*score;Phi=.5*(1+math.erf(-mu/math.sqrt(2)));phi=math.exp(-mu*mu/2)/math.sqrt(2*math.pi);target=-8*(1-Phi)**7*phi;se=float(est.std()/math.sqrt(len(est)));check(abs(float(est.mean())-target)<5*se)
 # Small delayed-reward learning correctness only, not protein training.
 toy=torch.tensor(-.5,dtype=torch.float64,requires_grad=True);opt=torch.optim.SGD([toy],lr=.5);toycurve=[]
 for _ in range(80):
  opt.zero_grad();p=toy.sigmoid();s=1-(1-p)**8;delayed=s*s;(-delayed).backward();opt.step();toycurve.append(float(delayed.detach()))
 check(toycurve[-1]>toycurve[0]+.01)
 # Same estimator/density masks as actual trace kernel; all early masks.
 u=torch.tensor([.4,-.3,.8],dtype=torch.float64);m=torch.tensor([.2,.1,-.3],dtype=torch.float64,requires_grad=True);ls=torch.tensor([-.2,.3,-.1],dtype=torch.float64)
 for mask in [(False,False,False),(True,False,False),(True,True,False),(True,True,True)]:
  lp=T['base_logprob'](m,ls,u,mask);exp=(torch.distributions.Normal(m,ls.exp()).log_prob(u)*torch.tensor(mask)).sum();check(torch.allclose(lp,exp,atol=1e-12))
 fn=lambda x:torch.stack([x[0].exp(),math.pi*x[1].sigmoid(),math.pi*x[2].tanh()]);J=torch.autograd.functional.jacobian(fn,u);check(abs(float(T['logjac'](u,[True]*3)-torch.log(J.diag().abs()).sum()))<1e-12)
 logits=torch.zeros((1,20),requires_grad=True);T['categorical_score'](logits,torch.tensor([2]),learned=False).sum().backward();check(not logits.grad.any())
 out=dict(PASS=True,checks=checks,known_valid_prefixes_preserved=preserved,certified_historical_prefixes=certified,seed=94337,
 finite8_probability_normalization=True,exact_score_gradient_error=float(abs(exact-estimated)),continuous_MC_gradient=float(est.mean()),continuous_exact_gradient=target,continuous_MC_SE=se,
 toy_engineering_updates=80,toy_initial_success=toycurve[0],toy_final_success=toycurve[-1],protein_optimizer_updates=0,no_GPU=True,proof=F['proof_contract']())
 (R/'FILTER_AND_GRADIENT_CPU_TESTS.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
