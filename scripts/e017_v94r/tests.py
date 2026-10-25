"""CPU mathematical checks for the new deterministic packing rejection map."""
import itertools,json,math,runpy
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v94r2_corrected_geometry_rl_20261025_v2');O=R.parent/'e017_v94_fixed_sequence_geometry_rl_20261024_v1'
F=runpy.run_path(str(R/'source/scripts/e017_v94r/packing.py'));P=runpy.run_path(str(O/'source/scripts/e017_v94/production.py'));S=P['S'];T=P['K']['S']['T'];save=P['compact'];checks=0

def enum(theta):
    pi=theta.softmax(0);rp=torch.tensor([.4,.35,.25],dtype=torch.float64);out=[]
    def node(stage,spent,prob,lp,ref,accepted):
        if stage==2:out.append((prob,lp,ref,float(spent==2),accepted));return
        def draw(k,p,l,r,a):
            for j in range(3):
                pn=p*pi[j];ln=l+pi[j].log();rn=r+rp[j].log()
                if spent+j<=2:node(stage+1,spent+j,pn,ln,rn,a+pi[j].log())
                elif k==0:draw(1,pn,ln,rn,a)
                else:out.append((pn,ln,rn,0.,a))
        draw(0,prob,lp,ref,accepted)
    z=theta.sum()*0;node(0,0,z+1,z,z,z);return out

def main():
    global checks
    sampler=runpy.run_path(str(R/'source/scripts/e017_v94r/sampling.py'));assert callable(sampler['corrected_action']);checks+=1
    torch.set_num_threads(1);torch.manual_seed(94317);rng=np.random.default_rng(94317)
    class State:
        N=64
        def __init__(self,x):self.ca=torch.tensor(x,dtype=torch.float64)
    for k in range(300):
        x=np.cumsum(rng.normal(size=(64,3))*rng.uniform(.1,6),axis=0);final=F['prefix_budget'](State(x))
        previous=0
        for n in [1,8,16,32,48,63,64]:
            b=F['prefix_budget'](State(x[:n]));assert b['contacts']>=previous and b['contacts']<=final['contacts']
            assert not (b['over_budget'] and not final['over_budget']);previous=b['contacts'];checks+=2
    assert 2*115/64<=F['UPPER'] and 2*116/64>F['UPPER'];checks+=2
    # Historical fitting-TRAIN controls: new predicate must not reject ANY
    # prefix of a completed hard-valid trajectory. No VALIDATION data used.
    controls=0
    for u in range(1,11):
        packet=P['read'](O/f'production/rollouts/u{u:06}.json')
        for row in packet['records']:
            if not row['metrics']['hard_valid']:continue
            state=P['c']['BuildState'].empty(64)
            for t,geom in enumerate(row['geometry']):
                state=S['append'](state,packet['task']['fixed_sequence'][t],geom)
                assert not F['prefix_budget'](state)['over_budget'];checks+=1
            controls+=1
    theta=torch.tensor([.2,-.1,.3],dtype=torch.float64,requires_grad=True);rows=enum(theta)
    assert abs(float(sum(x[0] for x in rows).detach())-1)<1e-12;checks+=1
    beta=.03
    expected=sum(p*(reward-beta*(lp-ref)) for p,lp,ref,reward,_ in rows)
    exact=torch.autograd.grad(expected,theta,retain_graph=True)[0]
    estimator=sum(p.detach()*((reward-beta*(lp-ref).detach())*lp-beta*(lp-ref)) for p,lp,ref,reward,_ in rows)
    score=torch.autograd.grad(estimator,theta,retain_graph=True)[0]
    assert torch.allclose(exact,score,atol=1e-12,rtol=1e-12);checks+=1
    accepted_only=sum(p.detach()*(reward-beta*(lp-ref).detach())*a for p,lp,ref,reward,a in rows)
    wrong=torch.autograd.grad(accepted_only,theta)[0];assert float((wrong-exact).abs().max())>1e-4;checks+=1
    mu=torch.tensor([.2,.1,-.3],dtype=torch.float64,requires_grad=True);ls=torch.tensor([-.2,.3,-.1],dtype=torch.float64,requires_grad=True);u=torch.tensor([.4,-.3,.8],dtype=torch.float64)
    for mask in [(False,False,False),(True,False,False),(True,True,False),(True,True,True)]:
        lp=T['base_logprob'](mu,ls,u,mask);expected_lp=(torch.distributions.Normal(mu,ls.exp()).log_prob(u)*torch.tensor(mask)).sum()
        assert torch.allclose(lp,expected_lp,atol=1e-12);checks+=1
    fn=lambda v:torch.stack((v[0].exp(),math.pi*v[1].sigmoid(),math.pi*v[2].tanh()))
    jac=torch.autograd.functional.jacobian(fn,u);analytic=T['logjac'](u,[True]*3)
    assert abs(float(analytic-torch.log(jac.diag().abs()).sum()))<1e-12;checks+=1
    z=torch.randn(16,dtype=torch.float64);adv=z-T['loo'](z);assert abs(float(adv.sum()))<1e-12;checks+=1
    logits=torch.zeros((1,20),requires_grad=True);fixed=T['categorical_score'](logits,torch.tensor([2]),learned=False);fixed.sum().backward();assert not logits.grad.any();checks+=1
    # Finite Gaussian rejection controller: score MC vs analytic derivative.
    mu0=.3;eps=torch.randn((100000,2),dtype=torch.float64);samples=mu0+eps;first_ok=samples[:,0]<=0;reward=(first_ok|(samples[:,1]<=0)).double();score_terms=eps[:,0]+(~first_ok)*eps[:,1];est=reward*score_terms
    phi=math.exp(-mu0*mu0/2)/math.sqrt(2*math.pi);Phi=.5*(1+math.erf(-mu0/math.sqrt(2)));target=-2*(1-Phi)*phi
    se=float(est.std()/math.sqrt(len(est)));assert abs(float(est.mean())-target)<5*se;checks+=1
    out=dict(PASS=True,checks=checks,TRAIN_controls=controls,all_hard_valid_prefixes_preserved=True,seed=94317,
        probability_normalization=True,exact_full_trace_gradient_error=float((exact-score).abs().max()),
        accepted_only_negative_control_error=float((wrong-exact).abs().max()),
        Gaussian_MC_gradient=float(est.mean()),Gaussian_exact_gradient=target,Gaussian_MC_SE=se,
        transformed_density_Jacobian_PASS=True,fixed_AA_score_zero=True,proof=F['proof_contract'](),
        neural_optimizer_updates=0,no_GPU=True,controller_uses_future_coordinates=False)
    save(R/'FILTER_AND_GRADIENT_CPU_TESTS.json',out);print(json.dumps(out),flush=True)
if __name__=='__main__':main()
