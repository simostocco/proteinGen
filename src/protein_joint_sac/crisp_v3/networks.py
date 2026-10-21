from dataclasses import dataclass,asdict
import copy,math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from protein_joint_sac.models.batched_bc import features_batch,rope,encode_batch

@dataclass
class Shape:
 width:int
 layers:int
 heads:int=8
 checkpoint:bool=True

PROFILES={
 'development':{'actor':Shape(96,2,4),'low_q':Shape(80,2,4),'high_actor':Shape(64,2,4),'high_q':Shape(64,1,4),'reach':Shape(64,2,4)},
 'research':{'actor':Shape(448,7),'low_q':Shape(384,7),'high_actor':Shape(320,5),'high_q':Shape(288,5),'reach':Shape(256,4)},
 'large':{'actor':Shape(512,9),'low_q':Shape(448,8),'high_actor':Shape(384,6),'high_q':Shape(320,5),'reach':Shape(288,5)}}

class Block(nn.Module):
 def __init__(self,c):
  super().__init__();w=c.width;self.heads=c.heads;self.n1=nn.LayerNorm(w);self.qkv=nn.Linear(w,3*w);self.out=nn.Linear(w,w);self.n2=nn.LayerNorm(w);self.ff=nn.Sequential(nn.Linear(w,4*w),nn.GELU(),nn.Linear(4*w,w))
 def forward(self,x,bias,pos):
  B,L,W=x.shape;H=self.heads;qkv=self.qkv(self.n1(x)).reshape(B,L,3,H,W//H);q,k,v=[qkv[:,:,i].transpose(1,2) for i in range(3)]
  q=rope(q,pos);k=rope(k,pos);z=F.scaled_dot_product_attention(q,k,v,attn_mask=bias.to(q.dtype),dropout_p=0.,is_causal=False)
  x=x+self.out(z.transpose(1,2).reshape(B,L,W));return x+self.ff(self.n2(x))

class ContextEncoder(nn.Module):
 """Prefix-only geometry-aware bidirectional encoder, explicit goals as separate tokens.
 J1 query feature 352 + AA20 + conditional Gaussian120 + context64 + progress4.
 Goal token: position3, frame9, motif20, relative index1, three validity bits.
 """
 def __init__(self,c):
  super().__init__();self.c=c;w=c.width;self.aa=nn.Embedding(20,w);self.token=nn.Linear(38,w);self.query=nn.Linear(560,w);self.goal=nn.Linear(36,w);self.pair=nn.Linear(25,c.heads,bias=False);self.blocks=nn.ModuleList([Block(c) for _ in range(c.layers)]);self.norm=nn.LayerNorm(w)
 def forward(self,b):
  ids=b['prefix']['ids'];B,L=ids.shape;f,p,present=b['features'];ref=b['reference'];g=b['goals'];G=g.shape[1]
  progress=torch.stack([b['prefix']['N']/500,b['prefix']['t']/b['prefix']['N'],b['prefix']['valid'].sum(-1)/b['prefix']['t'].clamp_min(1),b['subgoal']/4],-1).float()
  q=torch.cat([ref['h'],ref['logits'],ref['mu'].flatten(1),ref['ls'].flatten(1),b['context'],progress],-1)
  x=torch.cat([self.aa(ids)+self.token(f),self.goal(g),self.query(q)[:,None]],1)
  bias=x.new_zeros(B,self.c.heads,L+G+1,L+G+1)
  if L and p is not None:
   values,pv=p;bias[:,:,:L,:L]=self.pair(values).permute(0,3,1,2)*pv[:,None]
  valid=torch.cat([present,b['goal_valid'],torch.ones(B,1,dtype=torch.bool,device=x.device)],1);bias=bias.masked_fill(~valid[:,None,None,:],float('-inf'))
  pos=torch.cat([torch.arange(L,device=x.device).expand(B,L),g[:,:,32]*b['prefix']['N'][:,None],b['prefix']['t'][:,None]],1)
  for block in self.blocks:
   x=checkpoint(block,x,bias,pos,use_reentrant=False) if self.c.checkpoint and self.training and torch.is_grad_enabled() else block(x,bias,pos)
  return self.norm(x[:,-1])

class LowActor(nn.Module):
 def __init__(self,c):
  super().__init__();self.encoder=ContextEncoder(c);w=c.width;self.aa=nn.Linear(w,20);self.residue=nn.Embedding(20,64);self.geometry=nn.Sequential(nn.Linear(w+64,w),nn.SiLU(),nn.Linear(w,6))
  for m in [self.aa,self.geometry[-1]]:nn.init.zeros_(m.weight);nn.init.zeros_(m.bias)
 def forward(self,b):
  h=self.encoder(b);B=len(h);aa=b['reference']['logits']+3*self.aa(h).float().tanh();ids=torch.arange(20,device=h.device)
  out=self.geometry(torch.cat([h[:,None].expand(B,20,-1),self.residue(ids)[None].expand(B,20,-1)],-1)).float().tanh()
  mu=b['reference']['mu']+out[:,:,:3]*out.new_tensor([.15,1.,1.]);ls=(b['reference']['ls']+.2*out[:,:,3:]).clamp(-5,2)
  return aa,mu,ls

class LowQ(nn.Module):
 def __init__(self,c):
  super().__init__();self.encoder=ContextEncoder(c);self.residue=nn.Embedding(20,32);self.value=nn.Sequential(nn.Linear(c.width+42,c.width),nn.SiLU(),nn.Linear(c.width,1))
 def candidate(self,h,a,g,mask):
  # Undefined components are masked BEFORE inverse transforms; placeholders cannot leak.
  v=torch.where(mask,g,g.new_tensor([1.,math.pi/2,0.]))
  z=torch.stack([v[:,0]/4,v[:,1]/math.pi,v[:,2].sin(),v[:,2].cos(),v[:,0].log(),torch.logit(v[:,1]/math.pi),torch.atanh(v[:,2]/math.pi)],-1)
  z=z*torch.stack([mask[:,0],mask[:,1],mask[:,2],mask[:,2],mask[:,0],mask[:,1],mask[:,2]],-1)
  return self.value(torch.cat([h,self.residue(a),z.float(),mask.float()],-1)).squeeze(-1)
 def forward(self,b,a,g,mask):return self.candidate(self.encoder(b),a,g,mask)

class HighActor(nn.Module):
 def __init__(self,c):super().__init__();self.encoder=ContextEncoder(c);self.out=nn.Linear(c.width,9)
 def forward(self,b,mask):
  if not mask.any(-1).all():raise ValueError('empty high action set')
  return self.out(self.encoder(b)).float().masked_fill(~mask,float('-inf'))
class HighQ(nn.Module):
 def __init__(self,c):super().__init__();self.encoder=ContextEncoder(c);self.action=nn.Embedding(9,32);self.value=nn.Sequential(nn.Linear(c.width+32,c.width),nn.SiLU(),nn.Linear(c.width,1))
 def candidate(self,h,a):return self.value(torch.cat([h,self.action(a)],-1)).squeeze(-1)
 def forward(self,b,a):return self.candidate(self.encoder(b),a)
class Reachability(nn.Module):
 def __init__(self,c):super().__init__();self.encoder=ContextEncoder(c);self.policy=nn.Embedding(256,32);self.out=nn.Linear(c.width+32,1)
 def forward(self,b,policy):return self.out(torch.cat([self.encoder(b),self.policy(policy)],-1)).squeeze(-1)

class System(nn.Module):
 def __init__(self,profile='development',shapes=None):
  super().__init__();c=PROFILES[profile] if shapes is None else {k:Shape(**v) for k,v in shapes.items()};self.profile=profile;
  if set(c)!=set(PROFILES[profile]):raise ValueError('all component shapes required')
  for v in c.values():
   if v.width%v.heads or (v.width//v.heads)%2 or v.layers<1:raise ValueError('width/head/RoPE shape')
  self.models=nn.ModuleDict({'actor':LowActor(c['actor']),'q1':LowQ(c['low_q']),'q2':LowQ(c['low_q']),'high_actor':HighActor(c['high_actor']),'high_q1':HighQ(c['high_q']),'high_q2':HighQ(c['high_q']),'reach':Reachability(c['reach'])})
  self.targets=nn.ModuleDict({k:copy.deepcopy(self.models[k]).requires_grad_(False).eval() for k in ['q1','q2','high_q1','high_q2']})
 def train(self,mode=True):
  super().train(mode);self.targets.eval();return self
 def counts(self):return {'trainable':{k:sum(p.numel() for p in v.parameters()) for k,v in self.models.items()},'target_nontrainable':{k:sum(p.numel() for p in v.parameters()) for k,v in self.targets.items()},'total_trainable':sum(p.numel() for p in self.models.parameters())}
 @torch.no_grad()
 def polyak(self,tau=.005):
  if not 0<=tau<=1:raise ValueError('tau')
  for k,v in self.targets.items():
   for a,b in zip(v.parameters(),self.models[k].parameters(),strict=True):a.lerp_(b,tau)

def prepare(prefix,reference,goals,goal_valid,context=None,subgoal=None):
 B=len(prefix['t']);dev=prefix['ids'].device
 if goals.shape[-1]!=36 or goal_valid.shape!=goals.shape[:2]:raise ValueError('goal schema')
 return dict(prefix=prefix,reference={k:v.detach() for k,v in reference.items()},features=prefix.pop('_execution_actor_features') if '_execution_actor_features' in prefix else features_batch(prefix,chemistry=True),goals=goals,goal_valid=goal_valid,context=torch.zeros(B,64,device=dev) if context is None else context,subgoal=torch.zeros(B,device=dev) if subgoal is None else subgoal)
