"""Sequence-first hybrid SAC primitives. No production training launcher."""
import contextlib,math
import torch
from torch import nn
from torch.distributions import Normal
from ..contracts import AA,BuildAction,BuildState,digest
from ..geometry.ca_builder import append,mask_for_step
from ..models.hybrid_distributions import GeometryDistribution
from ..models.geometry_inference import corrected_action,PROTOCOL
from ..models.build_critic import TwinCritics

def gaussian_kl(p,q):
    """KL in physical units equals latent KL under the same bijection/masks."""
    if not torch.equal(p.mask,q.mask):raise ValueError('KL masks differ')
    terms=q.log_std-p.log_std+(.5*((p.log_std.exp()/q.log_std.exp()).square()+((p.mu-q.mu)/q.log_std.exp()).square()))-.5
    return terms[p.mask].sum()

def double_distribution(actor,state,residue,h=None):
    p=actor.geometry_distribution(state,residue,h)
    return GeometryDistribution(p.mu.double(),p.log_std.double(),mask_for_step(state.t))

def rsample_physical(p,noise=None):
    u=p.mu+p.log_std.exp()*(torch.randn_like(p.mu) if noise is None else noise.to(p.mu))
    g=p.transform(u)
    # An exactly saturated transform is not an invertible continuous sample.
    inverse=p.log_prob(g)
    if not torch.isfinite(inverse):raise FloatingPointError('noninvertible continuous draw; no silent repair in SAC estimator')
    logp=torch.where(p.mask,p.base.log_prob(u)-p.log_jacobian(u),torch.zeros_like(u)).sum()
    if not torch.isfinite(logp):raise FloatingPointError('nonfinite continuous density')
    return g,logp,u

@torch.no_grad()
def rollout(actor,N,seed,checkpoint_sha,device='cuda'):
    """Actual prefix-only policy; corrected FP64 construction; caller stores trace."""
    actor.eval();rng=torch.Generator(device=device).manual_seed(seed);state=BuildState.empty(N,device=device)
    trace=[];states=[]
    for t in range(N):
        states.append(state)
        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
            h=actor.encode(state);logits=actor.aa_head(h).float()
        probs=logits.softmax(-1);uniform=torch.rand((),device=device,generator=rng);i=int(torch.searchsorted(probs.cumsum(0),uniform).clamp_max(19))
        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
            p=actor.geometry_distribution(state,AA[i],h)
        eps=torch.randn(3,device=device,dtype=torch.float64,generator=rng)
        g,info=corrected_action(p.mu,p.log_std,mask_for_step(t),eps);action=BuildAction(AA[i],g,mask_for_step(t))
        trace.append({'t':t,'prefix_sha256':digest([list(state.residues),state.ca.cpu().tolist(),N]),'residue':AA[i],'geometry':g.cpu().tolist(),'mask':list(action.mask),'AA_logp':float(logits.log_softmax(-1)[i]),'geometry_latent_log_density':info['latent_action_log_density'],'geometry_represented_log_density':info['represented_action_log_density'],'latent':info['latent'],'epsilon':eps.cpu().tolist(),'categorical_uniform':float(uniform),'support_repaired':any(info['adjusted_components']),'policy_sha256':checkpoint_sha,'protocol':PROTOCOL,'random_stream':{'seed':seed,'generator':'torch CUDA Philox' if device=='cuda' else 'torch CPU','draw_order':'uniform then 3 Gaussian per residue'}})
        state=append(state,action)
    return state,trace,states

def regularized_value(actor,critics,state,reference=None,*,targets=False,alpha_A=.01,alpha_G=.001,beta_A=1.,beta_G=1.,estimator='exact',detach_geometry=False):
    if state.t>=state.N:raise ValueError('terminal state has no entropy/action')
    if estimator not in ['exact','score_function']:raise ValueError('estimator')
    h=actor.encode(state);lp=actor.aa_head(h).log_softmax(-1);p=lp.exp();N=state.N
    q1,q2=(critics.target1,critics.target2) if targets else (critics.q1,critics.q2)
    if reference is not None:
        with torch.no_grad():rh=reference.encode(state);rlp=reference.aa_head(rh).log_softmax(-1)
    else:rh=None;rlp=lp.detach()
    indices=range(20) if estimator=='exact' else [int(torch.distributions.Categorical(logits=lp).sample())]
    losses=[];logs=[]
    for i in indices:
        pd=double_distribution(actor,state,AA[i],h);sampling=GeometryDistribution(pd.mu.detach(),pd.log_std.detach(),mask_for_step(state.t)) if detach_geometry else pd;g,lg,u=rsample_physical(sampling);a=BuildAction(AA[i],g,mask_for_step(state.t));q=torch.minimum(q1(state,a),q2(state,a))
        if reference is not None:
            with torch.no_grad():rd=double_distribution(reference,state,AA[i],rh)
            klg=gaussian_kl(pd,rd)
            if detach_geometry:klg=klg.detach()
        else:klg=lg*0
        f=(alpha_A*lp[i]+alpha_G*lg+beta_A*(lp[i]-rlp[i])+beta_G*klg)/N-q
        # Correct categorical score-function term PLUS direct/reparameterized path.
        loss=p[i]*f if estimator=='exact' else f+lp[i]*f.detach()
        losses.append(loss);logs.append((i,lg.detach(),klg.detach()))
    return -torch.stack(losses).sum(),{'AA_KL':(p*(lp-rlp)).sum().detach(),'conditional_geometry':logs}

def bellman_target(reward,done,next_value):
    """Gamma=1, finite horizon encoded in state. Terminal never bootstraps."""
    return reward if done else reward+next_value

def discounted_nstep(rewards,bootstrap=0.,terminal=True):
    return sum(rewards)+(0. if terminal else bootstrap)

def reference_kl(actor,reference,state):
    h=actor.encode(state);lp=actor.aa_head(h).log_softmax(-1);p=lp.exp()
    with torch.no_grad():rh=reference.encode(state);rlp=reference.aa_head(rh).log_softmax(-1)
    g=torch.stack([gaussian_kl(double_distribution(actor,state,a,h),double_distribution(reference,state,a,rh)) for a in AA])
    return (p*(lp-rlp)).sum(),(p.detach()*g).sum()
