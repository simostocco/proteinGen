"""Linked complete/partial views with frozen parent splits and identity-first sampling."""
import collections,gzip,json,random,time,hashlib
from pathlib import Path
import numpy as np
import torch
import pyarrow.parquet as pq
from .scaled_supervised import Data,Sampler
from .partial_geometry import derive
from .certified_unified import file_sha
from ..contracts import AA,BuildState
from ..models.batched_bc import collate
R=Path('/mnt/d/Simone/proteinGen_data/e016_partial_supervision_20261016_v1')
class PartialData:
 def __init__(self):
  self.base=Data(True);self.proposal=self.base.proposal;self.seqs=self.base.seqs;self.rows={};self.cache={};self.owners={s:collections.defaultdict(list) for s in ['TRAIN','VALIDATION']};start=time.monotonic()
  expected=json.load(open(R/'view/VIEW_FROZEN.json'))
  for path,sha in expected['artifacts'].items():
   if file_sha(R/path)!=sha:raise ValueError('expanded view changed')
  rows=pq.read_table(R/'view/PARTIAL_OBSERVATIONS.parquet').to_pylist();sources=json.load(open(R/'view/SOURCE_HASHES.json'))
  by=collections.defaultdict(list)
  for row in rows:
   h=row['sequence_hash'];parent=self.proposal.sequences[h]
   if row['split'] not in ['TRAIN','VALIDATION'] or row['split']!=parent['split'] or len(row['aa_known'])!=len(self.seqs[h]) or not all(row['aa_known']):raise ValueError('partial ownership/sequence contradiction')
   by[row['coordinate_path']].append(row);self.rows[row['observation_id']]=row;self.owners[row['split']][h].append(row['observation_id'])
  for path,items in sorted(by.items()):
   if file_sha(path)!=sources.get(path):raise ValueError('coordinate source changed')
   if items[0]['coordinate_kind']=='legacy_cache':
    wanted={r['coordinate_row']:r for r in items};offset=0
    for b in pq.ParquetFile(path).iter_batches(batch_size=64,columns=['selected_calpha_coordinates'],use_threads=False):
     for j,v in enumerate(b.column(0)):
      row=wanted.get(offset+j)
      if row is not None:self.add(row,json.loads(v.as_py()))
     offset+=len(b)
   else:
    d=json.load(gzip.open(path,'rt'))
    for row in items:
     if d['source_sha256']!=row['source_sha']:raise ValueError('raw source revision mismatch')
     self.add(row,[v['ca'] for v in d['observations'][row['coordinate_row']]['residue_mapping']])
  if len(self.cache)!=len(rows):raise ValueError('partial cache coverage mismatch')
  for s in self.owners:
   for v in self.owners[s].values():v.sort()
  self.ids={s:sorted(self.owners[s]) for s in self.owners}
  self.receipt={'partial_observations':len(self.cache),'partial_cache_bytes':sum(sum(v.nbytes for v in r.values() if isinstance(v,np.ndarray)) for r in self.cache.values()),'seconds_partial_load':time.monotonic()-start,'TEST_observations_admitted':0,'TEST_feature_or_loss_targets':0,'shared_batches_may_decode_unselected_rows':True,'view_sha256':file_sha(R/'view/VIEW_FROZEN.json')}
 def add(self,row,xy):
  oid=row['observation_id'];seen=np.array(row['ca_observed'],dtype=bool);N=len(self.seqs[row['sequence_hash']]);x=np.array([v if v is not None else [0,0,0] for v in xy],dtype=np.float64)
  if len(x)!=N or [v is not None for v in xy]!=seen.tolist():raise ValueError('partial coordinate/mask correspondence changed')
  link=np.asarray(row['continuity'],bool);g,mask,seg=derive(x,seen,link)
  if not np.array_equal(seg,np.array(row['coordinate_segment_id'])):raise ValueError('segment reconstruction mismatch')
  if list(mask.sum(0))!=[row['distance_targets'],row['angle_targets'],row['torsion_targets']]:raise ValueError('geometry target census mismatch')
  self.cache[oid]={'x':x,'seen':seen,'links':link,'g':g,'mask':mask}
 def target(self,h,o,t):
  if o not in self.rows:return self.base.target(h,o,t,True)+(None,)
  row=self.rows[o]
  if row['sequence_hash']!=h:raise ValueError('partial parent ownership')
  r=self.cache[o];seq=self.seqs[h];x=np.where(r['seen'][:t,None],r['x'][:t],0.)
  state=BuildState(tuple(seq[:t]),torch.from_numpy(x.copy()),len(seq),torch.from_numpy(r['seen'][:t].copy()))
  return state,AA.index(seq[t]),torch.from_numpy(r['g'][t].copy()),r
 def batch(self,examples,device='cuda'):
  states=[];ys=[];gs=[];records=[];mask=[]
  for h,o,t in examples:
   state,y,g,r=self.target(h,o,t);states.append(state);ys.append(y);gs.append(g);records.append(r);mask.append(r['mask'][t] if r is not None else np.array([t>=1,t>=2,t>=3]))
  b=collate(states,device);bv=torch.zeros_like(b['valid'])
  for i,(state,r) in enumerate(zip(states,records)):
   if state.t>1:
    link=torch.ones(state.t-1,dtype=torch.bool,device=device) if r is None else torch.as_tensor(r['links'][:state.t-1],device=device)
    bv[i,1:state.t]=link&b['valid'][i,:state.t-1]&b['valid'][i,1:state.t]
  b['aa_loss_valid']=torch.ones(len(examples),dtype=torch.bool,device=device);b['polymer_position']=torch.where(torch.arange(b['ids'].shape[1],device=device)[None]<b['t'][:,None],torch.arange(b['ids'].shape[1],device=device)[None]+1,0);b['bond_valid']=bv;b['geometry_target_valid']=torch.as_tensor(np.asarray(mask),device=device,dtype=torch.bool)
  return b,torch.tensor(ys,device=device),torch.stack(gs).to(device)
 def panel(self,partial=False):
  out=[];ids=self.ids['VALIDATION'] if partial else self.base.jointids['VALIDATION']
  for h in ids:
   o=self.owners['VALIDATION'][h][0] if partial else self.base.owners[h][0];N=len(self.seqs[h]);pos=set(np.rint(np.linspace(0,N-1,8)).astype(int).tolist())
   if partial:
    r=self.cache[o];missing=np.flatnonzero(~r['seen'])
    if len(missing):pos.add(int(missing[len(missing)//2]))
    for c in range(3):
     available=np.flatnonzero(r['mask'][:,c])
     if len(available):pos.add(int(available[len(available)//2]))
    for t in np.rint(np.linspace(0,N-1,16)).astype(int):
     if len(pos)>=12:break
     pos.add(int(t))
   if len(pos)!=(12 if partial else 8):raise ValueError('fixed deterministic targets per validation identity')
   out.extend((h,o,t) for t in sorted(pos))
  return out
class ContinuationSampler:
 def __init__(self,data,parent_state,expanded=False,partial_geometry_oversample=False):
  self.data=data;self.complete=Sampler(data.base);self.complete.load_state_dict(parent_state);self.expanded=expanded;self.oversample=partial_geometry_oversample;self.ri=random.Random(443);self.ro=random.Random(444);self.rt=random.Random(445);self.rm=random.Random(446);self.cursor=0;self.tokens=0;self.seen=set();self.branch_counts=collections.Counter();self.component_counts=np.zeros(3,dtype=np.int64);self.missing_aa=0;self.complete_stream_sha256="0"*64;self.partial_stream_sha256="0"*64
 def draw(self,n):
  if n%2:raise ValueError('fixed50:50 mixture requires even effective batch')
  out=self.complete.draw(n//2 if self.expanded else n)
  for ex in out:self.complete_stream_sha256=hashlib.sha256((self.complete_stream_sha256+json.dumps(ex,separators=(',',':'))).encode()).hexdigest()
  if self.expanded:
   for _ in range(n//2):
    h=self.ri.choice(self.data.ids['TRAIN']);o=self.ro.choice(self.data.owners['TRAIN'][h]);r=self.data.cache[o]
    choices=np.flatnonzero(r['mask'].any(1)) if self.oversample and self.rt.random()<.5 else range(len(self.data.seqs[h]));t=int(self.rt.choice(choices));ex=(h,o,t);out.append(ex);self.partial_stream_sha256=hashlib.sha256((self.partial_stream_sha256+json.dumps(ex,separators=(',',':'))).encode()).hexdigest()
   self.rm.shuffle(out)
  for h,o,t in out:
   partial=o in self.data.rows;self.branch_counts['partial' if partial else 'complete']+=1;self.cursor+=1;self.tokens+=t;self.seen.add(h)
   self.component_counts+=self.data.cache[o]['mask'][t] if partial else np.array([t>=1,t>=2,t>=3])
   self.missing_aa+=int(partial and not self.data.cache[o]['seen'][t])
  return out
 def state_dict(self):return {'complete':self.complete.state_dict(),'rng':[r.getstate() for r in [self.ri,self.ro,self.rt,self.rm]],'complete_stream_sha256':self.complete_stream_sha256,'partial_stream_sha256':self.partial_stream_sha256,'cursor':self.cursor,'tokens':self.tokens,'seen':sorted(self.seen),'branch_counts':dict(self.branch_counts),'component_counts':self.component_counts.tolist(),'AA_at_missing_CA':self.missing_aa,'expanded':self.expanded,'oversample':self.oversample,'partial_ids_sha256':__import__('hashlib').sha256('\n'.join(self.data.ids['TRAIN']).encode()).hexdigest()}
 def load_state_dict(self,x):
  for k in ['expanded','oversample','partial_ids_sha256']:
   if x[k]!=self.state_dict()[k]:raise ValueError('sampler membership/policy mismatch')
  self.complete.load_state_dict(x['complete'])
  for r,s in zip([self.ri,self.ro,self.rt,self.rm],x['rng']):r.setstate(s)
  self.complete_stream_sha256=x['complete_stream_sha256'];self.partial_stream_sha256=x['partial_stream_sha256'];self.cursor=x['cursor'];self.tokens=x['tokens'];self.seen=set(x['seen']);self.branch_counts=collections.Counter(x['branch_counts']);self.component_counts=np.asarray(x['component_counts']);self.missing_aa=x['AA_at_missing_CA']
