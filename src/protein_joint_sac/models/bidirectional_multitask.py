"""Training-only MASK representation/readout; production J1 and prefix encoder unchanged."""
import collections,copy,math,random
import torch
from torch import nn
from torch.nn import functional as F
from .batched_bc import block_forward

class SpanReadout(nn.Module):
 def __init__(self,model):
  super().__init__();D=model.encoder.config.width
  self.mask_embedding=nn.Parameter(model.encoder.aa.weight.detach().mean(0).clone())
  self.readout=nn.Linear(D,20,device=model.aa_head.weight.device,dtype=model.aa_head.weight.dtype)
  self.readout.load_state_dict(model.aa_head.state_dict())

def masked_states(model,aux,corrupted,Ns,activation_checkpoint=False):
 """Only corrupted IDs (-1 is MASK), true lengths; no target labels/coordinates accepted."""
 if corrupted.ndim!=2 or Ns.shape!=(len(corrupted),) or (Ns<1).any():raise ValueError('masked batch')
 B,L=corrupted.shape;enc=model.encoder;device=corrupted.device
 if ((corrupted < -1)|(corrupted>19)).any():raise ValueError('vocabulary')
 present=torch.arange(L,device=device)[None]<Ns[:,None]
 # Complete known sequence positions; all geometry/chemistry unavailable. Same token feature layout.
 feat=torch.zeros(B,L,38,device=device,dtype=enc.tip.dtype)
 feat[:,0,34]=1.;feat[torch.arange(B,device=device),Ns-1,35]=1.
 context=enc.context(torch.stack([Ns/500.,torch.ones_like(Ns)],-1).to(enc.tip.dtype))
 aa=enc.aa(corrupted.clamp_min(0));aa=torch.where((corrupted==-1)[...,None],aux.mask_embedding[None,None],aa)
 x=aa+enc.token(feat)+context[:,None];x=torch.cat([x,(enc.tip+context)[:,None]],1)
 key_valid=torch.cat([present,torch.ones(B,1,device=device,dtype=torch.bool)],1)
 bias=torch.zeros(B,1,1,L+1,device=device,dtype=x.dtype).masked_fill(~key_valid[:,None,None,:],float('-inf'))
 pos=torch.cat([torch.arange(L,device=device).expand(B,L),Ns[:,None]],1)
 for block in enc.blocks:
  if activation_checkpoint and torch.is_grad_enabled():
   from torch.utils.checkpoint import checkpoint
   x=checkpoint(block_forward,block,x,bias,pos,use_reentrant=False)
  else:x=block_forward(block,x,bias,pos)
 return enc.final(x[:,:L])

def span_batch(sequences,plans,device='cuda'):
 from ..contracts import AA
 B=len(plans);N=[len(sequences[h]) for h,positions in plans];L=max(N);ids=torch.zeros(B,L,dtype=torch.long);targets=ids.clone();mask=torch.zeros(B,L,dtype=torch.bool)
 for k,(h,pos) in enumerate(plans):
  y=torch.tensor([AA.index(a) for a in sequences[h]]);ids[k,:len(y)]=y;targets[k,:len(y)]=y;mask[k,list(pos)]=True;ids[k,mask[k]]=-1
 return ids.to(device),torch.tensor(N,device=device),targets.to(device),mask.to(device)

def span_loss(model,aux,batch):
 ids,N,y,mask=batch;h=masked_states(model,aux,ids,N);logits=aux.readout(h).float();ce=F.cross_entropy(logits.transpose(1,2),y,reduction='none');loss=(ce*mask).sum(1)/mask.sum(1).clamp_min(1)
 return loss,logits,mask

class SpanSampler:
 """Uniform identity epochs, length-bucketed within64; nonoverlapping3–8 spans covering15%."""
 def __init__(self,data,seed=2026101941,role='TRAIN'):
  self.data=data;self.rng=random.Random(seed);self.role=role;self.order=sorted(data.base.ids[role]);self.rng.shuffle(self.order);self.cursor=0;self.epoch=0;self.bits={};self.counts=collections.Counter();self.spans=collections.Counter();self.lengths=collections.Counter();self.kinds=collections.Counter();self.lookup={h:k for k,hs in data.populations[role].items() for h in hs};self._bucket()
 def _bucket(self):
  for i in range(0,len(self.order),64):self.order[i:i+64]=sorted(self.order[i:i+64],key=lambda h:len(self.data.seqs[h]))
 def plan(self,h,record=True):
  n=len(self.data.seqs[h]);M=max(3,round(.15*n));old=self.bits.get(h,0);chosen=set();spans=[]
  while len(chosen)<M:
   rem=M-len(chosen);sizes=[k for k in range(3,min(8,rem)+1) if rem-k==0 or rem-k>=3];l=self.rng.choice(sizes)
   starts=[i for i in range(n-l+1) if not any(j in chosen for j in range(max(0,i-1),min(n,i+l+1)))]
   fresh=[i for i in starts if not any(old&(1<<j) for j in range(i,i+l))];pool=fresh or starts
   if not pool:raise ValueError('no valid span position')
   i=self.rng.choice(pool);chosen.update(range(i,i+l));spans.append(l)
  pos=sorted(chosen)
  if record:
   bits=sum(1<<p for p in pos);self.bits[h]=old|bits;self.counts['examples']+=1;self.counts['masked_targets']+=len(pos);self.counts['repeat_identity_positions']+=(old&bits).bit_count();self.spans.update(spans);self.lengths[str(next(k for k,(lo,hi) in enumerate([(20,64),(65,128),(129,256),(257,384),(385,500)]) if lo<=n<=hi))]+=len(pos);self.kinds[self.lookup[h]]+=len(pos)
  return h,pos
 def draw(self,n):
  out=[]
  for _ in range(n):
   if self.cursor==len(self.order):self.rng.shuffle(self.order);self._bucket();self.cursor=0;self.epoch+=1
   h=self.order[self.cursor];self.cursor+=1;out.append(self.plan(h))
  return sorted(out,key=lambda z:len(self.data.seqs[z[0]]))
 def state_dict(self):return {k:copy.deepcopy(v) for k,v in self.__dict__.items() if k not in ['data','rng','lookup']}|{'rng':self.rng.getstate()}
 def load_state_dict(self,state):
  for k,v in state.items():
   if k=='rng':self.rng.setstate(v)
   else:setattr(self,k,copy.deepcopy(v))
 def report(self):return {'counts':dict(self.counts),'distinct_identities':len(self.bits),'distinct_masked_identity_positions':sum(v.bit_count() for v in self.bits.values()),'mask_span_lengths':dict(self.spans),'masked_positions_by_length':dict(self.lengths),'masked_positions_by_population':dict(self.kinds),'epoch':self.epoch,'cursor':self.cursor}
