"""Execution-only batching: identical ordered sampler decisions and per-decision weights.
No feature persistence or inference through future suffixes. All caches bounded RAM.
"""
import math,time,json
from collections import defaultdict,Counter,OrderedDict
import torch
from . import admission as A, pipeline as P
from .networks import prepare
from .environment import World
from protein_joint_sac.contracts import BuildState,AA
from protein_joint_sac.models.batched_bc import collate
from protein_joint_sac.data.scaled_supervised import canonical,internal

def goal_example(pop,goals):
 task=goals.draw();h=task['source_identity'];o=task['source_observation'];x=canonical(pop.coordinates(pop.base.rows[o]));g=internal(x);seq=pop.base.seqs[h]
 candidates=sorted(set([p['index'] for p in task['targets']]+[int(i) for i in task['motifs']])-{0,1,2})
 t=goals.rng.choice(candidates) if candidates and goals.prefix_cursor%2==0 else goals.rng.randrange(3,len(seq));goals.prefix_cursor+=1
 return BuildState(tuple(seq[:t]),torch.tensor(x[:t]),len(seq)),AA.index(seq[t]),torch.tensor(g[t]),task

def goal_batch(examples,prior,device):
 from .environment import goal_tokens
 p=collate([e[0] for e in examples],device);tokens=[goal_tokens(e[3],device) for e in examples];B=len(examples);G=max(x[0].shape[1] for x in tokens)
 gt=torch.zeros(B,G,36,device=device);gv=torch.zeros(B,G,dtype=torch.bool,device=device)
 for i,(g,v) in enumerate(tokens):gt[i,:g.shape[1]]=g[0];gv[i,:v.shape[1]]=v[0]
 context=torch.zeros(B,64,device=device);context[:,2]=p['t']/p['N']
 b=prepare(p,prior(p),gt,gv,context)
 return b,torch.tensor([e[1] for e in examples],device=device),torch.stack([e[2] for e in examples]).to(device),torch.ones(B,3,device=device,dtype=torch.bool)

def row_goal_loss(b,mu,ls,mask):
 # The sealed scalar loss includes ZERO for each decision with no applicable next goal.
 # A naive batched eligible-only mean would alter its scientific weighting.
 values=[]
 for i in range(len(mu)):
  one={k:({j:v[i:i+1] for j,v in z.items()} if isinstance(z,dict) else z[i:i+1] if torch.is_tensor(z) else z) for k,z in b.items() if k!='features'}
  values.append(A.deterministic_goal_loss(one,mu[i:i+1],ls[i:i+1],mask[i:i+1]))
 return torch.stack(values).mean()

def loss_with_goal(system,b,aa,g,valid,weights):
 from torch.nn import functional as F
 from .density import log_prob,kl
 logits,mu,ls=system.models['actor'](b);idx=torch.arange(len(aa),device=aa.device);sm=mu[idx,aa];sl=ls[idx,aa];ka,kg,_=kl(logits,mu,ls,b['reference'],valid);count=valid.sum(-1).clamp_min(1)
 values=dict(aa=F.cross_entropy(logits,aa),geometry=(-log_prob(sm,sl,g,valid)/count).mean(),kl_aa=ka.mean(),kl_geometry=(kg/count).mean(),goal=row_goal_loss(b,sm,sl,valid))
 return sum(weights[k]*v for k,v in values.items()),values

def initialization_update(system,learner,prior,pop,sampler,geometry_sampler,goals,config):
 from .optimization import finish
 m=system.models['actor'];opt=learner.optimizers['actor'];opt.zero_grad(set_to_none=True);device=next(m.parameters()).device;B=config['effective_AA_batch'];micro=config['microbatch'];totals={};parts=[];counts=Counter();masks=Counter();ep=config.get('execution_protocol',{})
 def backward(stream,b,aa,g,mask,scale,goal=False):
  objective,terms=loss_with_goal(system,b,aa,g,mask,config['stream_weights'][stream]) if goal else A.actor_loss(system,b,aa,g,mask,config['stream_weights'][stream],False)
  weighted=objective*scale;weighted.backward();parts.append(weighted.detach())
  for k,v in terms.items():key=stream+'/'+k;totals[key]=totals.get(key,0)+v.detach()*scale
  # Defer tiny GPU->CPU synchronizations until all backward passes finish.
  key=stream;value=mask.detach().sum(0);masks[key]=masks.get(key,0)+value
 records=sampler.draw(B)
 for start,(p,aa,g,mask,_) in enumerate(prepared_batches(pop,records,micro,device,ep.get('cpu_prefetch',False))):
  subset=records[start*micro:start*micro+len(aa)];b=prepare(p,prior(p),torch.zeros(len(aa),1,36,device=device),torch.zeros(len(aa),1,dtype=torch.bool,device=device));backward('AA',b,aa,g,mask,len(aa)/B);counts.update(r[0] for r in subset)
 # Interleave geometry sampler draws and observation RNG exactly as the sealed loop.
 n=config['complete_geometry_decisions'];gm=ep.get('geometry_microbatch',micro)
 for p,aa,g,mask,rows in geometry_batches(pop,geometry_sampler,n,gm,device,ep.get('geometry_cpu_prefetch',False)):
  b=prepare(p,prior(p),torch.zeros(len(rows),1,36,device=device),torch.zeros(len(rows),1,dtype=torch.bool,device=device));backward('complete_geometry',b,aa,g,mask,len(rows)/n)
 n=config['goal_demonstrations'];gm=ep.get('goal_microbatch',min(micro,n))
 for start in range(0,n,gm):
  examples=[goal_example(pop,goals) for _ in range(min(gm,n-start))];b,aa,g,mask=goal_batch(examples,prior,device);backward('goal',b,aa,g,mask,config['goal_stream_scale']*len(examples)/n,True)
 grad={k:float(torch.stack([p.grad.float().square().sum() for p in group.parameters() if p.grad is not None]).sum().sqrt()) for k,group in [('encoder',m.encoder),('AA',m.aa),('geometry',m.geometry)]}
 if any(p.grad is not None for k,model in system.models.items() if k!='actor' for p in model.parameters()):raise ValueError('inactive-network gradient')
 norm=finish(opt,m.parameters(),config['gradient_clip']);learner.steps['actor']+=1;learner.schedulers['actor'].step()
 mask_report={s+'/'+name:int(values[j]) for s,values in masks.items() for j,name in enumerate(['distance','angle','torsion'])}
 return dict(loss=float(torch.stack(parts).sum()),terms={k:float(v) for k,v in totals.items()},gradient_groups=grad,gradient_norm=norm,AA_decisions=B,AA_categories=dict(counts),valid_geometry_components=mask_report,goal_kind_counts=dict(goals.kind_counts),goal_identities=len(goals.seen),goal_rejections=goals.rejected)

class ByteBoundGroups:
 def __init__(self,limit=256*2**20):self.limit=limit;self.cache=OrderedDict();self.bytes=0;self.hits=0;self.misses=0;self.seconds=0.
 def clear(self):self.cache.clear();self.bytes=0;self.hits=0;self.misses=0;self.seconds=0.
 def get(self,reader,path,index):
  from .execution import ArrowRows
  import pyarrow.parquet as pq
  key=(path,index)
  if key in self.cache:self.hits+=1;value=self.cache.pop(key);self.cache[key]=value;return value
  self.misses+=1;t=time.monotonic();col=pq.ParquetFile(path).read_row_group(index,columns=['selected_calpha_coordinates'],use_threads=False).column(0);self.seconds+=time.monotonic()-t;value=ArrowRows(col);size=col.nbytes
  while self.cache and self.bytes+size>self.limit:
   _,old=self.cache.popitem(last=False);self.bytes-=old.column.nbytes
  if size<=self.limit:self.cache[key]=value;self.bytes+=size
  return value
GROUPS=ByteBoundGroups()
def ram_group(self,path,index):return GROUPS.get(self,path,index)
ram_group.cache_clear=GROUPS.clear

def enable(config):
 ep=config.get('execution_protocol',{})
 if ep.get('schema')=='e017_end_to_end_execution_v2' and not ep.get('batched_auxiliary_streams'):raise ValueError('Execution v2 requires explicit tested update dispatch')
 from .prior import FrozenPrior
 FrozenPrior.shared_prefix_features=config.get('execution_protocol',{}).get('share_prefix_features',False)
 enable_compiled_rope(config.get('execution_protocol',{}).get('compile_actor_rope',False))
 if config.get('execution_protocol',{}).get('batched_auxiliary_streams'):
  A.initialization_update=initialization_update

# Keep only qualified TRAIN rows from each already-read Parquet column. Arrow.take
# copies selected variable-width buffers, avoiding retention of the full source group.
class SparseRows:
 def __init__(self,column,indices,path,index):
  import pyarrow as pa
  self.column=column.take(pa.array(indices,type=pa.int64()));self.lookup={j:i for i,j in enumerate(indices)};self.path=path;self.index=index
 def __getitem__(self,index):
  if index in self.lookup:return self.column[self.lookup[index]].as_py()
  # Validation requests are not admitted to the TRAIN RAM cache. Original reader
  # semantics still apply; no TEST requests are introduced.
  import pyarrow.parquet as pq
  return pq.ParquetFile(self.path).read_row_group(self.index,columns=['selected_calpha_coordinates'],use_threads=False).column(0)[index].as_py()
class QualifiedGroups(ByteBoundGroups):
 def __init__(self):super().__init__(512*2**20);self.indices={};self.group_offsets={}
 def setup(self,pop):
  by={}
  for r in list(pop.base.rows.values())+list(pop.partial.values()):
   if r['split']=='TRAIN' and r['coordinate_kind']=='legacy_cache':by.setdefault(r['coordinate_path'],set()).add(int(r['coordinate_row']))
  import pyarrow.parquet as pq
  for path,rows in by.items():
   pf=pq.ParquetFile(path);offset=0
   for i in range(pf.num_row_groups):
    n=pf.metadata.row_group(i).num_rows;self.indices[(path,i)]=sorted(j-offset for j in rows if offset<=j<offset+n);offset+=n
 def get(self,reader,path,index):
  import pyarrow.parquet as pq
  key=(path,index)
  if key in self.cache:self.hits+=1;value=self.cache.pop(key);self.cache[key]=value;return value
  self.misses+=1;t=time.monotonic();col=pq.ParquetFile(path).read_row_group(index,columns=['selected_calpha_coordinates'],use_threads=False).column(0);value=SparseRows(col,self.indices.get(key,[]),path,index);self.seconds+=time.monotonic()-t;size=value.column.nbytes
  while self.cache and self.bytes+size>self.limit:
   _,old=self.cache.popitem(last=False);self.bytes-=old.column.nbytes
  if size<=self.limit:self.cache[key]=value;self.bytes+=size
  return value
QUALIFIED=QualifiedGroups()
def qualified_group(self,path,index):return QUALIFIED.get(self,path,index)
qualified_group.cache_clear=QUALIFIED.clear

# Single persistent CPU producer; no lookahead survives an optimizer boundary.
# Each queue holds at most two microbatches. Sampler draws/observation choices
# remain serial and ordered; workers never run J1 or any CUDA operation.
from concurrent.futures import ThreadPoolExecutor
_EXECUTOR=None
def prepared_batches(pop,records,micro,device,prefetch=False):
 global _EXECUTOR
 if not prefetch:
  for i in range(0,len(records),micro):yield pop.batch(records[i:i+micro],device)
  return
 if _EXECUTOR is None:_EXECUTOR=ThreadPoolExecutor(max_workers=1,thread_name_prefix='crisp_cpu_loader')
 from collections import deque
 pending=deque();cursor=0
 def fill():
  nonlocal cursor
  while cursor<len(records) and len(pending)<2:
   part=records[cursor:cursor+micro];cursor+=len(part);pending.append(_EXECUTOR.submit(pop.batch,part,'cpu'))
 fill()
 while pending:
  p,aa,g,mask,rows=pending.popleft().result();fill()
  # Each transfer has an owned pinned buffer. CUDA copies use the current stream.
  move=lambda x:x.pin_memory().to(device,non_blocking=True) if x.numel() else x.to(device)
  yield {k:move(v) for k,v in p.items()},move(aa),move(g),move(mask),rows

def resolved_batch(rows,device):
 p=collate([r[0] for r in rows],device);p['bond_valid']=torch.zeros_like(p['valid'])
 for i,r in enumerate(rows):
  if r[0].t>=2:p['bond_valid'][i,1:r[0].t]=torch.as_tensor(r[4],device=device)&p['valid'][i,:r[0].t-1]&p['valid'][i,1:r[0].t]
 return p,torch.tensor([r[1] for r in rows],device=device),torch.stack([r[2] for r in rows]).to(device),torch.stack([r[3] for r in rows]).to(device),rows

def geometry_batches(pop,sampler,n,micro,device,prefetch=False):
 global _EXECUTOR
 def cpu(count,dev):
  rows=[pop.decision(*sampler.geometry(1)[0]) for _ in range(count)]
  return resolved_batch(rows,dev)
 if not prefetch:
  for i in range(0,n,micro):yield cpu(min(micro,n-i),device)
  return
 if _EXECUTOR is None:_EXECUTOR=ThreadPoolExecutor(max_workers=1,thread_name_prefix='crisp_cpu_loader')
 from collections import deque
 pending=deque();cursor=0
 def fill():
  nonlocal cursor
  while cursor<n and len(pending)<2:
   count=min(micro,n-cursor);cursor+=count;pending.append(_EXECUTOR.submit(cpu,count,'cpu'))
 fill()
 while pending:
  p,aa,g,mask,rows=pending.popleft().result();fill()
  move=lambda x:x.pin_memory().to(device,non_blocking=True) if x.numel() else x.to(device)
  yield {k:move(v) for k,v in p.items()},move(aa),move(g),move(mask),rows

_ROPE_EAGER=None
_ROPE_COMPILED=None
def enable_compiled_rope(enabled):
 global _ROPE_EAGER,_ROPE_COMPILED
 from . import networks
 if _ROPE_EAGER is None:_ROPE_EAGER=networks.rope
 if enabled:
  if _ROPE_COMPILED is None:
   import torch._inductor.config
   torch._inductor.config.compile_threads=1
   _ROPE_COMPILED=torch.compile(_ROPE_EAGER,fullgraph=True,dynamic=True)
  networks.rope=_ROPE_COMPILED
 else:networks.rope=_ROPE_EAGER
