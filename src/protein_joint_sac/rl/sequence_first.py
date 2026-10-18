"""J1 parameter-free regularization, exact coverage and proper nonlocal energy score."""
import collections,copy,hashlib,math,random
import torch
from torch.nn import functional as F
from ..models.batched_bc import encode_batch
from ..models.hybrid_distributions import GeometryDistribution
from ..geometry.ca_builder import next_ca,mask_for_step
from .global_geometry import conditional_parameters,radius_of_gyration,global_losses
VERSION="j1_sequence_first_v1"
def set_dropout(model,p=.15):
 if not 0<=p<1:raise ValueError("dropout probability")
 for block in model.encoder.blocks:block.dropout_p=p
def augment(batch,p=.075,generator=None):
 """Real mask and input visibility are separate; physical targets remain unchanged."""
 b=dict(batch);observed=batch['valid'].clone();visible=observed.clone()
 eligible=torch.arange(observed.shape[1],device=observed.device)[None]<(batch['t'][:,None]-3)
 hidden=(torch.rand(observed.shape,device=observed.device,generator=generator)<p)&observed&eligible
 visible &= ~hidden
 b['ca_observed']=observed;b['ca_aug_visible']=~hidden;b['valid']=visible
 if 'bond_valid' in batch:
  links=batch['bond_valid'].clone()
  if links.shape[1]>1:links[:,1:] &= visible[:,:-1]&visible[:,1:]
  b['bond_valid']=links
 return b
class Coverage:
 """Shuffled bijections cover all AA positions before a repeat, even across views."""
 def __init__(self,seed=42):
  self.seed=seed;self.cursor={};self.visited={};self.context={};self.counts=collections.Counter();self.bands=collections.Counter();self.deciles=collections.Counter()
 def position(self,h,N,observation,offset=0,stream="AA"):
  key=stream+":"+h;count=self.cursor.get(key,0);size=N-offset
  if size<=0:raise ValueError("no valid positions")
  cycle,k=divmod(count,size);rng=random.Random(int(hashlib.sha256(f"{self.seed}:{key}:{cycle}".encode()).hexdigest(),16))
  a=rng.randrange(1,size+1)
  while math.gcd(a,size)!=1:a=(a%size)+1
  t=offset+(a*k+rng.randrange(size))%size;self.cursor[key]=count+1
  if stream=="AA":
   old=self.visited.get(h,0);self.counts['repeated_identity_positions']+=int(bool(old&(1<<t)));self.visited[h]=old|(1<<t)
   ck=h+":"+observation;self.context[ck]=self.context.get(ck,0)|(1<<t)
   self.counts['AA_targets']+=1;self.deciles[str(min(9,10*t//N))]+=1
   self.bands[str(next(i for i,(lo,hi) in enumerate([(20,64),(65,128),(129,256),(257,384),(385,500)]) if lo<=N<=hi))]+=1
  return t
 def state_dict(self):return copy.deepcopy(self.__dict__)
 def load_state_dict(self,s):self.__dict__.update(copy.deepcopy(s))
 def report(self):
  return {'distinct_identities':len(self.visited),'distinct_identity_positions':sum(x.bit_count() for x in self.visited.values()),'distinct_observation_position_contexts':sum(x.bit_count() for x in self.context.values()),'counts':dict(self.counts),'length_bands':dict(self.bands),'position_deciles':dict(self.deciles)}
def anchors(valid,t,limit=16,seed=42):
 """Position-only stratification; no reference target coordinate used."""
 choices=[j for j in range(t) if bool(valid[j]) and t-j>=3];rng=random.Random(seed+t)
 groups=[[j for j in choices if lo<=t-j<=hi] for lo,hi in [(3,7),(8,23),(24,100000)]]
 out=[]
 while len(out)<min(limit,len(choices)):
  for group in groups:
   if group and len(out)<limit:out.append(group.pop(rng.randrange(len(group))))
  if not any(groups):break
 return sorted(out)
def energy_score(samples,target):
 """Unbiased U-statistic energy score on finite-dimensional Euclidean vectors."""
 if samples.ndim!=2 or target.shape!=samples.shape[1:] or len(samples)<2:raise ValueError("K>=2 vectors required")
 K=len(samples);first=torch.linalg.vector_norm(samples-target,dim=-1).mean()
 d=torch.linalg.vector_norm(samples[:,None]-samples[None,:],dim=-1)
 return first-d.sum()/(2*K*(K-1))
def geometry_parameters(model,h,labels):
 chem=model.chemistry[labels] if model.encoder.config.chemistry else torch.zeros_like(model.chemistry[labels])
 with torch.autocast(h.device.type,enabled=False):
  return model.geom(torch.cat([h.float(),model.selected_aa(labels).float(),chem.float()],-1))
def direct_nonlocal(model,batch,labels,native_targets,K=4,scale=3.8,seed=42):
 """Coordinates at t are loss-only. Last three anchors must remain visible."""
 with torch.autocast(batch['x'].device.type,dtype=torch.bfloat16,enabled=batch['x'].is_cuda):
  h=encode_batch(model,batch)
 p=geometry_parameters(model,h,labels);losses=[];counts=[]
 for i in range(len(labels)):
  t=int(batch['t'][i]);x=batch['x'][i,:t]
  if t<3 or not bool(batch['valid'][i,t-3:t].all()):continue
  js=anchors(batch['valid'][i,:t],t,16,seed)
  if not js:continue
  a=x[js];target=torch.linalg.vector_norm(native_targets[i]-a,dim=-1)/(scale*math.sqrt(len(js)))
  values=[]
  for k in range(K):
   g,_=GeometryDistribution(p[i,:3].double(),p[i,3:].double(),mask_for_step(t)).rsample()
   coord=next_ca(x,g);values.append(torch.linalg.vector_norm(coord-a,dim=-1)/(scale*math.sqrt(len(js))))
  losses.append(energy_score(torch.stack(values),target));counts.append(len(js))
 return (torch.stack(losses).mean() if losses else p.sum()*0),counts
def generated_bundle(model,prefix,ids,N,H,reference,precision="BF16",recompute=True):
 """Two independent full-gradient paths; native suffix is never actor input."""
 from .global_geometry import rollout
 outputs=[rollout(model,prefix,ids,N,H,reference,precision,recompute) for _ in range(2)]
 starts=len(prefix);segments=[r['coordinates'][starts:] for r in outputs];native=reference[starts:]
 rg=torch.stack([F.huber_loss(torch.log((radius_of_gyration(s)+1e-8)/(radius_of_gyration(native)+1e-8)),native.new_zeros(()),delta=.5) for s in segments]).mean()
 clash=torch.stack([r['clash_loss'] for r in outputs]).mean()
 pairs=[(i,j) for i in range(H) for j in range(i+3,H)]
 rng=random.Random(780+H);pairs=rng.sample(pairs,min(32,len(pairs)))
 a=torch.tensor([i for i,j in pairs],device=prefix.device);b=torch.tensor([j for i,j in pairs],device=prefix.device)
 scale=3.8*math.sqrt(len(pairs))
 target=torch.linalg.vector_norm(native[a]-native[b],dim=-1)/scale
 samples=torch.stack([torch.linalg.vector_norm(s[a]-s[b],dim=-1)/scale for s in segments])
 pair=energy_score(samples,target)
 return {'Rg':rg,'clash':clash,'pair':pair,'coordinates':outputs[0]['coordinates'],'paths':2,'residues':2*H}
