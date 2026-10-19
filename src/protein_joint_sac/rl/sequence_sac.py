"""Finite-horizon sequence-first SAC with explicit production gates and PDB hooks."""
import math
import torch
from .sequence_rl import regularized_value,reference_kl,bellman_target,double_distribution,rsample_physical
from ..contracts import AA
from ..geometry.ca_builder import mask_for_step
from .soft_targets import Temperatures

def norm(values):return math.sqrt(sum(float(x.detach().double().square().sum()) for x in values if x is not None))

def backward_sequence_first(actor,sequence_loss,geometry_loss,limits,ratio=.5):
    """Two graph evaluations, aggregate shared cap, disjoint S/A clips. No step."""
    actor.zero_grad(set_to_none=True);seq=sequence_loss();seq.backward();aa={n:None if p.grad is None else p.grad.clone() for n,p in actor.named_parameters()};actor.zero_grad(set_to_none=True);geo=geometry_loss();geo.backward()
    shared=[(n,p) for n,p in actor.named_parameters() if p.requires_grad and not n.startswith('aa_head.')];head=[(n,p) for n,p in actor.named_parameters() if p.requires_grad and n.startswith('aa_head.')]
    if any(p.requires_grad for n,p in actor.named_parameters() if n.startswith(('geom.','selected_aa.'))):raise ValueError('initial sequence-first phase requires frozen geometry outputs')
    if any(p.grad is not None and torch.any(p.grad!=0) for n,p in head):raise ValueError('geometry retention bypasses exclusive AA head')
    a=norm([aa[n] for n,p in shared]);g=norm([p.grad for n,p in shared]);scale=min(1.,ratio*a/max(g,1e-30))
    for n,p in actor.named_parameters():
        if not p.requires_grad:
            if p.grad is not None:raise ValueError('frozen parameter gradient')
            continue
        if p.grad is not None:p.grad.mul_(scale)
        if aa[n] is not None:p.grad=aa[n].clone() if p.grad is None else p.grad.add_(aa[n])
        if p.grad is not None and not torch.isfinite(p.grad).all():raise FloatingPointError('nonfinite actor gradient')
    stats={'sequence_shared_L2':a,'geometry_shared_raw_L2':g,'geometry_shared_scale':scale,'geometry_to_sequence_shared_ratio':scale*g/max(a,1e-30),'whole_model_clip':False}
    for k,items in [('S',shared),('A',head)]:
        before=norm([p.grad for n,p in items]);factor=min(1.,limits[k]/max(before,1e-30))
        for n,p in items:
            if p.grad is not None:p.grad.mul_(factor)
        stats[k]={'before':before,'factor':factor,'after':before*factor}
    return seq.detach(),geo.detach(),stats

class SequenceFirstSAC:
    def __init__(self,actor,reference,critics,config):
        self.actor=actor.eval();self.reference=reference.requires_grad_(False).eval();self.critics=critics;self.config=config;device=next(actor.parameters()).device
        for n,p in actor.named_parameters():p.requires_grad_(not n.startswith(('geom.','selected_aa.')))
        self.actor_opt=torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad],lr=config['actor_lr'],weight_decay=.01)
        self.critic_opt=torch.optim.AdamW(list(critics.q1.parameters())+list(critics.q2.parameters()),lr=config['critic_lr'])
        self.temperatures=Temperatures(config['target_AA_entropy'],config['target_geometry_entropy']).to(device)
        with torch.no_grad():self.temperatures.log_aa.fill_(math.log(config['alpha_A']));self.temperatures.log_geom.fill_(math.log(config['alpha_G']))
        self.alpha_opt=torch.optim.Adam(self.temperatures.parameters(),lr=config['temperature_lr'])
    def update_resolved(self,state,action,reward,next_state,done,*,PDB_AA,PDB_geometry,receipt=None):
        c=self.config
        if not all(c.get(k,False) for k in ['production_authorized','AF_reward_certified','SAC_real_episode_certified','gradient_composer_certified']):raise PermissionError('production R2 gates or separate authorization missing; no optimizer step')
        if reward is None:raise ValueError('pending/failed oracle is unknown, not biological zero')
        expected=c.get('calibration_sha256')
        if not expected or not receipt or receipt.get('status')!='SUCCEEDED' or receipt.get('calibrated') is not True or receipt.get('calibration_sha256')!=expected or receipt.get('evidence_type')!='EXTERNAL_PREDICTION':raise PermissionError('verified calibrated complete-episode receipt required')
        if not math.isfinite(float(reward)):raise FloatingPointError('nonfinite return')
        if c.get('objective',{}).get('beta_G')!=0:raise ValueError('initial sequence-first phase routes geometry KL only through separately capped retention, not the primary AA objective')
        if done!=(next_state.t==next_state.N) or state.N!=next_state.N:raise ValueError('finite-horizon termination mismatch')
        obj=dict(c['objective'],alpha_A=float(self.temperatures.log_aa.exp().detach()),alpha_G=float(self.temperatures.log_geom.exp().detach()))
        with torch.no_grad():nv=0. if done else regularized_value(self.actor,self.critics,next_state,self.reference,targets=True,**obj)[0];target=bellman_target(torch.as_tensor(reward,device=state.ca.device),done,nv)
        qloss=(self.critics.q1(state,action)-target).square()+(self.critics.q2(state,action)-target).square();self.critic_opt.zero_grad(set_to_none=True);qloss.backward();torch.nn.utils.clip_grad_norm_(list(self.critics.q1.parameters())+list(self.critics.q2.parameters()),5.,error_if_nonfinite=True);self.critic_opt.step()
        if not all(torch.isfinite(p).all() for p in list(self.critics.q1.parameters())+list(self.critics.q2.parameters())):raise FloatingPointError('nonfinite critic parameters')
        ps=list(self.critics.q1.parameters())+list(self.critics.q2.parameters());flags=[p.requires_grad for p in ps]
        for p in ps:p.requires_grad_(False)
        try:
            def sequence():return -regularized_value(self.actor,self.critics,state,self.reference,detach_geometry=True,**obj)[0]+PDB_AA(c['BC_weight'])
            def geometry():return c['geometry_KL_weight']*reference_kl(self.actor,self.reference,state)[1]+PDB_geometry(c['PDB_geometry_weight'])
            seq,geo,stats=backward_sequence_first(self.actor,sequence,geometry,c['clip_limits']);self.actor_opt.step()
            if not all(torch.isfinite(p).all() for p in self.actor.parameters()):raise FloatingPointError('nonfinite actor parameters')
        finally:
            for p,flag in zip(ps,flags):p.requires_grad_(flag)
        with torch.no_grad():
            h=self.actor.encode(state);cat=torch.distributions.Categorical(logits=self.actor.aa_head(h));i=cat.sample();g,lg,_=rsample_physical(double_distribution(self.actor,state,AA[int(i)],h));la=cat.log_prob(i)
        valid=sum(mask_for_step(state.t));tloss=self.temperatures.log_aa*(-la-self.temperatures.target_aa)
        if valid:tloss=tloss+self.temperatures.log_geom*(-lg-self.temperatures.target_geom*valid/3)
        self.alpha_opt.zero_grad();tloss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in self.temperatures.parameters()):raise FloatingPointError('nonfinite temperature gradient')
        self.alpha_opt.step()
        if not all(torch.isfinite(p).all() for p in self.temperatures.parameters()):raise FloatingPointError('nonfinite temperatures')
        self.critics.polyak(.005)
        return {'critic_loss':float(qloss.detach()),'sequence_loss':float(seq),'geometry_retention_loss':float(geo),'clipping':stats,'temperature_loss':float(tloss.detach())}
