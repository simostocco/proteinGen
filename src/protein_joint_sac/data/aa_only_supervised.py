"""Existing J1 plus geometry-unavailable AA targets; no new model parameters."""
import collections,copy,hashlib,random
import numpy as np
import torch
from .partial_continuation import PartialData
from ..contracts import AA
from ..models.batched_bc import collate
from ..rl.sequence_first import Coverage,augment

class ExpandedData(PartialData):
 def __init__(self):
  super().__init__();self.populations={}
  for role in ['TRAIN','VALIDATION']:
   c=set(self.base.jointids[role]);p=set(self.ids[role])-c;u=set(self.base.ids[role])-c-p
   self.populations[role]={'complete':sorted(c),'partial_only':sorted(p),'unused':sorted(u)}
  assert [len(self.populations['TRAIN'][k]) for k in ['complete','partial_only','unused']]==[21475,29129,67615]
  assert [len(self.populations['VALIDATION'][k]) for k in ['complete','partial_only','unused']]==[2947,3178,8650]
 def aa_only_batch(self,examples,device='cuda'):
  states=[];ys=[]
  for h,o,t in examples:
   state,y,g=self.base.target(h,None,t,False);assert g is None;states.append(state);ys.append(y)
  b=collate(states,device);b['bond_valid']=torch.zeros_like(b['valid']);b['aa_loss_valid']=torch.ones(len(states),device=device,dtype=torch.bool);b['geometry_target_valid']=torch.zeros(len(states),3,device=device,dtype=torch.bool)
  return b,torch.tensor(ys,device=device),torch.zeros(len(states),3,device=device,dtype=torch.float64)
 def batch(self,examples,device='cuda'):
  if all(o is not None for h,o,t in examples):return super().batch(examples,device)
  states=[];ys=[];gs=[];links=[];masks=[]
  for h,o,t in examples:
   if o is None:
    state,y,_=self.base.target(h,None,t,False);g=torch.zeros(3,dtype=torch.float64);link=np.zeros(max(0,t-1),bool);mask=np.zeros(3,bool)
   else:
    state,y,g,r=super().target(h,o,t);link=np.ones(max(0,t-1),bool) if r is None else r['links'][:max(0,t-1)];mask=np.array([t>=1,t>=2,t>=3]) if r is None else r['mask'][t]
   states.append(state);ys.append(y);gs.append(g);links.append(link);masks.append(mask)
  b=collate(states,device);bv=torch.zeros_like(b['valid'])
  for i,state in enumerate(states):
   if state.t>1:bv[i,1:state.t]=torch.as_tensor(links[i],device=device)&b['valid'][i,:state.t-1]&b['valid'][i,1:state.t]
  b['aa_loss_valid']=torch.ones(len(states),device=device,dtype=torch.bool);b['bond_valid']=bv;b['geometry_target_valid']=torch.as_tensor(np.array(masks),device=device,dtype=torch.bool)
  b['polymer_position']=torch.where(torch.arange(b['ids'].shape[1],device=device)[None]<b['t'][:,None],torch.arange(b['ids'].shape[1],device=device)[None]+1,0)
  return b,torch.tensor(ys,device=device),torch.stack(gs).to(device)

def augment_available(b,p=.075):
 """Only rows containing actual observed coordinates consume augmentation draws."""
 idx=torch.nonzero(b['valid'].any(1),as_tuple=False).flatten()
 out=dict(b);out['ca_observed']=b['valid'].clone();out['ca_aug_visible']=torch.ones_like(b['valid'])
 if not len(idx):return out
 sub={k:v[idx] for k,v in b.items()};changed=augment(sub,p)
 for k in ['valid','bond_valid','ca_observed','ca_aug_visible']:
  if k in changed:
   if k not in out:out[k]=torch.zeros_like(b['valid'])
   else:out[k]=out[k].clone()
   out[k][idx]=changed[k]
 return out

class ExpandedSampler:
 """Five updates:128 complete/96 partial-only/96 unused AA decisions exactly."""
 QUOTAS=[(26,19,19),(25,20,19),(26,19,19),(25,19,20),(26,19,19)]
 def __init__(self,data,parent):
  self.data=data;self.cover=Coverage(42);self.cover.load_state_dict(parent['coverage']);self.cursor=0;self.counts=collections.Counter();self.orders={};self.cursors={};self.rng={};self.visited={k:{} for k in ['complete','partial_only','unused']};self.context={};self.bands=collections.Counter();self.deciles=collections.Counter();self.parent_cursor=parent['cursor'];self.epoch=collections.Counter()
  for k,ids in {**data.populations['TRAIN'],'geometry':data.base.jointids['TRAIN']}.items():
   self.rng[k]=random.Random(int(hashlib.sha256(('expanded-AA-v1:'+k).encode()).hexdigest(),16));self.orders[k]=list(ids);self.rng[k].shuffle(self.orders[k]);self.cursors[k]=0
 def pick(self,k):
  r=self.rng[k];i=self.cursors[k]
  if i==len(self.orders[k]):r.shuffle(self.orders[k]);i=0;self.epoch[k]+=1
  h=self.orders[k][i];self.cursors[k]=i+1
  o=None if k=='unused' else r.choice(self.data.owners['TRAIN'][h] if k=='partial_only' else self.data.base.owners[h]);N=len(self.data.seqs[h]);aa=k!='geometry';t=self.cover.position(h,N,o or 'AA_ONLY',0 if aa else 3,'AA' if aa else 'GEOMETRY');self.counts[k]+=1
  if aa:
   old=self.visited[k].get(h,0);self.counts['repeated_identity_position_'+k]+=bool(old&(1<<t));self.visited[k][h]=old|(1<<t);ck=h+':'+(o or 'AA_ONLY');self.context[ck]=self.context.get(ck,0)|(1<<t)
   self.bands[k+':'+str(next(i for i,(lo,hi) in enumerate([(20,64),(65,128),(129,256),(257,384),(385,500)]) if lo<=N<=hi))]+=1;self.deciles[k+':'+str(min(9,10*t//N))]+=1
   if k=='partial_only' and not self.data.cache[o]['seen'][t]:self.counts['partial_AA_at_missing_CA']+=1
  return h,o,t
 def draw(self):
  aa=[]
  for k,n in zip(['complete','partial_only','unused'],self.QUOTAS[self.cursor%5]):aa.extend(self.pick(k) for _ in range(n))
  ge=[self.pick('geometry') for _ in range(32)];self.cursor+=1
  key=lambda e:(e[2],e[0],e[1] or '')
  return sorted(aa,key=key),sorted(ge,key=key)
 def state_dict(self):return {k:copy.deepcopy(v) for k,v in self.__dict__.items() if k!='data'}|{'cover':self.cover.state_dict(),'rng':{k:v.getstate() for k,v in self.rng.items()}}
 def load_state_dict(self,s):
  for k,v in s.items():
   if k=='cover':self.cover.load_state_dict(v)
   elif k=='rng':
    for name,st in v.items():self.rng[name].setstate(st)
   else:setattr(self,k,copy.deepcopy(v))
 def report(self):
  return {'updates':self.cursor,'AA_targets':64*self.cursor,'complete_geometry_targets':self.counts['geometry'],'counts':dict(self.counts),'categories':{k:{'unique_identities':len(v),'unique_identity_positions':sum(x.bit_count() for x in v.values()),'population':len(self.orders[k]),'identity_coverage_fraction':len(v)/len(self.orders[k]),'repeated_identity_positions':self.counts['repeated_identity_position_'+k]} for k,v in self.visited.items()},'unique_identities':sum(len(v) for v in self.visited.values()),'unique_identity_positions':sum(x.bit_count() for v in self.visited.values() for x in v.values()),'observation_position_contexts':sum(x.bit_count() for x in self.context.values()),'length_bands':dict(self.bands),'position_deciles':dict(self.deciles),'epochs':dict(self.epoch),'historical_position_coverage_preserved':True}
