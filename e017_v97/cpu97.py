from core97 import *
from safety97 import *
import numpy as np,torch,itertools,math,shutil,time
from scipy.stats import binom,beta,betabinom

def main():
 verify();rng=np.random.default_rng(1973197);checks=[]
 # Gate likelihood with finite rejection and exact RLOO, including rejected proposals.
 from gate97 import GATE_SCALE
 a=torch.tensor(0.,dtype=torch.float64,requires_grad=True);p=(.2+.7*(a/GATE_SCALE).tanh()).sigmoid();q=.5;b=.013;traces=[(1,),(0,1),(0,0)];vals=[]
 for tr in traces:
  lp=sum(torch.log(p if z else 1-p) for z in tr);prob=lp.exp();lr=lp-len(tr)*math.log(q);R=float(any(tr));vals.append((prob,lp,lr,R))
 obj=sum(pr*(r-b*lr) for pr,lp,lr,r in vals);exact=torch.autograd.grad(obj,a,retain_graph=True)[0];score=sum(pr.detach()*((r-b*lr.detach()-b)*lp) for pr,lp,lr,r in vals);est=torch.autograd.grad(score,a,retain_graph=True)[0];assert torch.allclose(exact,est,atol=1e-12)
 rloo=0.
 for i,j in itertools.product(range(3),repeat=2):
  pi,li,ki,ri=vals[i];pj,lj,kj,rj=vals[j];rloo+=pi.detach()*pj.detach()*(((ri-b*ki.detach())-(rj-b*kj.detach())-b)*li+((rj-b*kj.detach())-(ri-b*ki.detach())-b)*lj)/2
 assert torch.allclose(torch.autograd.grad(rloo,a)[0],exact,atol=1e-12);checks.append('gate full-trace finite-rejection and RLOO exact gradients')
 # Transform log-Jacobian under interpolated geometry means: unchanged bijection.
 import runpy
 T=runpy.run_path(str(ROOT.parent/'e017_v93_regional_controllability_20261024_v1/source/scripts/e017_v93/trace.py'))
 u=torch.tensor([.3,-.2,.7],dtype=torch.float64);fn=lambda v:torch.stack([v[0].exp(),math.pi*v[1].sigmoid(),math.pi*v[2].tanh()]);jac=torch.autograd.functional.jacobian(fn,u);assert torch.allclose(T['logjac'](u,[True]*3),jac.diag().abs().log().sum(),atol=1e-12)
 for mask in [[False]*3,[True,False,False],[True,True,False],[True]*3]:
  m=torch.tensor([.1,.2,-.1],dtype=torch.float64,requires_grad=True);ls=torch.tensor([-.2,-.3,-.4],dtype=torch.float64);lp=T['base_logprob'](m,ls,u,mask);expected=(torch.distributions.Normal(m,ls.exp()).log_prob(u)*torch.tensor(mask)).sum();assert torch.allclose(lp,expected,atol=1e-12)
 # Clamp convention: boundary derivative is PyTorch's chosen subgradient1;
 # left/right finite differences differ at a kink and are not a unique derivative.
 for z in [-5.,-4.,2.]:
  c=torch.tensor(z,dtype=torch.float64,requires_grad=True);dc=torch.autograd.grad(c.clamp(-5,2),c)[0];assert dc==1
 checks.append('masked Normal likelihood, transform Jacobian and explicit clamp-boundary subgradient')
 assert classify(3934,3934)['status']=='PASS';assert classify(3933,3933)['status']=='INCONCLUSIVE';assert classify(3845,3845)['status']=='BREACH';assert classify(3846,3846)['status']=='INCONCLUSIVE';assert classify(4096,4096,4095)['status']=='INCOMPLETE';assert classify(4096,4096,KL=1.01)['status']=='BREACH';checks.append('exact boundary, original KL/diversity and incomplete decisions')
 schedule=read(ROOT/'MONITOR_SCHEDULE.json');assert all(len(v)==N and len({d['candidate_seed'] for d in v})==N for v in schedule['looks'].values());assert len({d['candidate_seed'] for v in schedule['looks'].values() for d in v})==7*N
 checks.append('frozen independent candidate seeds; schedules identical across future arms')
 m=Monitor(ROOT/'qualification/monitor_resume','A',0,'0'*64,fixture=True)
 def row(i):return dict(index=i,draw=m.schedule[i],binding=m.binding,complete=i%5!=0,aborted=i%5==0,backbone_valid=i%5!=0,hard_valid=i%5!=0)
 for i in range(10):m.commit(i,row(i))
 m.commit(10,row(10),interrupt='after_result');assert m.frontier==10
 n=Monitor(m.root,'A',0,'0'*64,fixture=True);assert n.frontier==11 and n.backbone==8 and n.decision()['status']=='INCOMPLETE'
 try:n.commit(10,row(10))
 except AssertionError:pass
 else:raise AssertionError('duplicate accepted')
 # Uncommitted partial evidence never advances the cursor.
 put(m.root/'rows/00011.partial_evidence','{interrupted')
 assert Monitor(m.root,'A',0,'0'*64,fixture=True).frontier==11
 corrupt=ROOT/'qualification/monitor_corrupt';shutil.copytree(m.root,corrupt);(corrupt/'rows/00000.json').write_text('{}')
 try:Monitor(corrupt,'A',0,'0'*64,fixture=True)
 except AssertionError:pass
 else:raise AssertionError('corruption accepted')
 missing=ROOT/'qualification/monitor_missing';shutil.copytree(m.root,missing);os.rename(missing/'rows/00000.json',missing/'rows/00000.quarantined_fixture')
 try:Monitor(missing,'A',0,'0'*64,fixture=True)
 except AssertionError:pass
 else:raise AssertionError('missing committed row accepted')
 try:Monitor(m.root,'B',0,'0'*64,fixture=True)
 except AssertionError:pass
 else:raise AssertionError('cross-arm binding accepted')
 try:n.enforce()
 except SafetyPause:pass
 else:raise AssertionError('incomplete monitor permitted optimizer continuation')
 checks.append('monitor orphan adoption, interrupted result, no duplicates, corruption/missing/wrong-arm rejection')
 outcomes=[];rep=20000
 for p in [.99,.98,.97,.966,.96,.95,.94,.93]:
  k=rng.binomial(N,p,rep);status=np.where(k>=3934,0,np.where(k<=3845,1,2));outcomes.append(dict(model='independent randomized single candidate',p=p,replicates=rep,PASS=float(np.mean(status==0)),BREACH=float(np.mean(status==1)),INCONCLUSIVE=float(np.mean(status==2)),exact_PASS=float(binom.sf(3933,N,p)),exact_BREACH=float(binom.cdf(3845,N,p))))
 # Extreme identity heterogeneity, each selected identity succeeds deterministically.
 # Independently resampled identity per candidate => iid even if an identity repeats.
 vals=[]
 for _ in range(20):
  identity=rng.integers(200,size=(200,N));vals.extend((identity<190).sum(1).tolist())
 vals=np.asarray(vals);hetero=dict(groups=200,deterministically_valid_groups=190,replicates=len(vals),PASS=float(np.mean(vals>=3934)),BREACH=float(np.mean(vals<=3845)),INCONCLUSIVE=float(np.mean((vals>3845)&(vals<3934))),valid_under_independent_draw_scheme=True)
 # Deliberately invalid design: eight correlated candidates per group under beta random effect.
 rho=.2;p=.95;ab=(1/rho-1);bad=[]
 for _ in range(20):
  counts=betabinom.rvs(8,p*ab,(1-p)*ab,size=(200,512),random_state=rng).sum(1);bad.extend(counts.tolist())
 bad=np.asarray(bad);negative=dict(replicates=len(bad),p=.95,ICC=rho,batch_cluster_size=8,misapplied_binomial_PASS=float(np.mean(bad>=3934)),misapplied_binomial_BREACH=float(np.mean(bad<=3845)),conclusion='binomial contract explicitly INVALID for shared cluster effects; not a guarantee for arbitrary correlated protein evaluations')
 looks=rng.binomial(N,.95,size=(rep,2,7));repeated=dict(replicates=rep,arms=2,looks=7,hard_and_backbone_identical_for_simulation=True,any_false_PASS=float(np.mean((looks>=3934).any((1,2)))),any_definite_BREACH_at_boundary=float(np.mean((looks<=3845).any((1,2)))),guarantee='union-bound coverage does not require independence across arms/looks; fresh conditional iid draws within each look required',inconclusive_never_allows_update=True)
 save('CPU_AND_SAFETY_QUALIFICATION.json',dict(PASS=True,checks=checks,finite_trace_gate_gradient=float(exact),RLOO_error=float((exact-est).abs()),simulations=outcomes,heterogeneous_identity_model=hetero,correlated_negative_control=negative,repeated_looks=repeated,alpha=ALPHA,production_updates=0,new_inference=0))
 print('CPU_AND_SAFETY_PASS',flush=True)
if __name__=='__main__':main()
