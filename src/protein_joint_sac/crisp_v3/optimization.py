"""Independent staged optimizers, streamed hybrid SAC, supervised initialization."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F
from .density import sample,log_prob,kl,semimarkov

@contextmanager
def frozen(modules):
 ps=[p for m in modules for p in m.parameters()];flags=[p.requires_grad for p in ps]
 for p in ps:p.requires_grad_(False)
 try:yield
 finally:
  for p,f in zip(ps,flags):p.requires_grad_(f)

def make_optimizers(system,lr=1e-4):
 return {k:torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=.01,foreach=False) for k,m in system.models.items()}
def finish(o,params,clip=1.):
 ps=list(params);norm=torch.nn.utils.clip_grad_norm_(ps,clip,error_if_nonfinite=True);o.step();o.zero_grad(set_to_none=True);return float(norm)

def supervised(system,b,aa,geometry,valid,weights=None,goal_loss=None):
 w={'aa':1.,'geometry':.1,'kl_aa':.02,'kl_geometry':.01,'goal':.05}|(weights or {})
 logits,mu,ls=system.models['actor'](b);idx=torch.arange(len(aa),device=aa.device)
 ce=F.cross_entropy(logits,aa);nll=-log_prob(mu[idx,aa],ls[idx,aa],geometry,valid)/valid.sum(-1).clamp_min(1)
 ka,kg,_=kl(logits,mu,ls,b['reference'],valid)
 goal=logits.new_zeros(()) if goal_loss is None else goal_loss(mu[idx,aa],ls[idx,aa],valid)
 total=w['aa']*ce+w['geometry']*nll.mean()+w['kl_aa']*ka.mean()+w['kl_geometry']*(kg/valid.sum(-1).clamp_min(1)).mean()+w['goal']*goal
 return total,dict(aa=ce,geometry=nll.mean(),kl_aa=ka.mean(),kl_geometry=kg.mean(),goal=goal)

class Learner:
 def __init__(self,system,lr=1e-4,gamma=.995,alpha_aa=.01,alpha_geometry=.001,beta_aa=.02,beta_geometry=.01):
  self.system=system;self.optimizers=make_optimizers(system,lr);self.gamma=gamma;self.aa=alpha_aa;self.geom=alpha_geometry;self.ba=beta_aa;self.bg=beta_geometry;self.steps={k:0 for k in self.optimizers}
  self.schedulers={k:torch.optim.lr_scheduler.LambdaLR(o,lambda step:1.) for k,o in self.optimizers.items()}
 @torch.no_grad()
 def low_value(self,b,mask):
  s=self.system;l,mu,ls=s.models['actor'](b);lp=l.log_softmax(-1);p=lp.exp();h=[s.targets[k].encoder(b) for k in ['q1','q2']];v=torch.zeros(len(l),device=l.device,dtype=torch.float64)
  for a in range(20):
   ids=torch.full((len(l),),a,device=l.device,dtype=torch.long);g,lg=sample(mu[:,a],ls[:,a],mask);q=torch.minimum(*[s.targets[k].candidate(hh,ids,g,mask) for k,hh in zip(['q1','q2'],h)])
   v+=p[:,a]*(q-self.aa*lp[:,a]-self.geom*lg)
  return v
 def critic(self,b,next_b,a,g,mask,next_mask,reward,duration,terminated,high=False,high_mask=None):
  s=self.system;keys=['high_q1','high_q2'] if high else ['q1','q2']
  with torch.no_grad():
   if high:
    lp=s.models['high_actor'](next_b,high_mask).log_softmax(-1);p=lp.exp();values=[]
    for k in keys:
     hh=s.targets[k].encoder(next_b);values.append(torch.stack([s.targets[k].candidate(hh,torch.full_like(a,i)) for i in range(9)],-1))
    v=(p*(torch.minimum(*values)-self.aa*torch.where(high_mask,lp,0.))).sum(-1)
   else:v=self.low_value(next_b,next_mask)
   target=semimarkov(reward,self.gamma,duration,terminated,v).float()
  out={}
  for k in keys:
   o=self.optimizers[k];o.zero_grad(set_to_none=True);q=s.models[k](b,a) if high else s.models[k](b,a,g,mask);loss=F.mse_loss(q.float(),target);loss.backward();out[k]=dict(loss=float(loss.detach()),gradient_norm=finish(o,s.models[k].parameters()),target_mean=float(target.mean()));self.steps[k]+=1;self.schedulers[k].step()
  return out
 def low_actor(self,b,mask,regularizer=None):
  s=self.system;o=self.optimizers['actor'];o.zero_grad(set_to_none=True);report=[]
  # Critic context is independent of candidate g; no actor gradient is lost by caching.
  with frozen([s.models['q1'],s.models['q2']]):
   with torch.no_grad():hh=[s.models[k].encoder(b) for k in ['q1','q2']]
   for a in range(20):
    # Fresh actor graph per categorical action; do not retain twenty encoder graphs.
    l,mu,ls=s.models['actor'](b);lp=l.log_softmax(-1);p=lp.exp();ids=torch.full((len(l),),a,device=l.device,dtype=torch.long);g,lg=sample(mu[:,a],ls[:,a],mask)
    q=torch.minimum(*[s.models[k].candidate(h,ids,g,mask) for k,h in zip(['q1','q2'],hh)])
    _,_,kg=kl(l,mu,ls,b['reference'],mask);r_lp=b['reference']['logits'].log_softmax(-1)
    term=(p[:,a]*(self.aa*lp[:,a]+self.geom*lg-q+self.ba*(lp[:,a]-r_lp[:,a])+self.bg*kg[:,a]/mask.sum(-1).clamp_min(1))).mean();term.backward();report.append(float(term.detach()))
  if regularizer is not None:
   bc=regularizer();bc.backward();report.append(float(bc.detach()))
  norm=finish(o,s.models['actor'].parameters());self.steps['actor']+=1;self.schedulers['actor'].step();return dict(loss=sum(report),gradient_norm=norm,categorical_actions=20)
 def high_actor(self,b,mask):
  s=self.system;o=self.optimizers['high_actor'];o.zero_grad(set_to_none=True)
  with frozen([s.models[k] for k in ['high_q1','high_q2']]):
   with torch.no_grad():
    h=[s.models[k].encoder(b) for k in ['high_q1','high_q2']];q=torch.stack([torch.minimum(*[s.models[k].candidate(x,torch.full((len(mask),),a,device=mask.device,dtype=torch.long)) for k,x in zip(['high_q1','high_q2'],h)]) for a in range(9)],-1)
   lp=s.models['high_actor'](b,mask).log_softmax(-1);p=lp.exp();loss=(p*(self.aa*torch.where(mask,lp,0.)-q)).sum(-1).mean();loss.backward()
  norm=finish(o,s.models['high_actor'].parameters());self.steps['high_actor']+=1;self.schedulers['high_actor'].step();return dict(loss=float(loss.detach()),gradient_norm=norm)
 def reach(self,b,policy,success,resolved):
  if not resolved.all():raise ValueError('censored/unresolved reachability targets excluded')
  m=self.system.models['reach'];o=self.optimizers['reach'];o.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(m(b,policy),success.float());loss.backward();norm=finish(o,m.parameters());self.steps['reach']+=1;self.schedulers['reach'].step();return dict(loss=float(loss.detach()),gradient_norm=norm)
 def supervised(self,b,aa,g,mask,weights=None):
  m=self.system.models['actor'];o=self.optimizers['actor'];o.zero_grad(set_to_none=True);loss,terms=supervised(self.system,b,aa,g,mask,weights);loss.backward();norm=finish(o,m.parameters());self.steps['actor']+=1;self.schedulers['actor'].step();return dict(loss=float(loss.detach()),gradient_norm=norm,**{k:float(v.detach()) for k,v in terms.items()})
