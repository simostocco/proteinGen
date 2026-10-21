"""Identity-first, lazy coordinates. Frozen scientific membership, no new split."""
import copy,hashlib,json,random,gzip
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from protein_joint_sac.data.scaled_supervised import Data,internal,canonical,P
from protein_joint_sac.data.accepted_supervised import CoordinateReader
from protein_joint_sac.data.partial_geometry import derive
from protein_joint_sac.contracts import BuildState,AA
from protein_joint_sac.models.batched_bc import collate
from .prior import file_sha
D=Path('/mnt/d/Simone/proteinGen_data');PARTIAL=D/'e016_partial_supervision_20261016_v1/view/PARTIAL_OBSERVATIONS.parquet'
class Population:
 def __init__(self):
  self.base=Data(False);self.reader=CoordinateReader(self.base.proposal.external);self.partial={};self.partial_owners={}
  frozen=json.load(open(PARTIAL.parent/'VIEW_FROZEN.json'))
  for path,h in frozen['artifacts'].items():
   if file_sha(PARTIAL.parent.parent/path)!=h:raise ValueError('partial frozen artifact mismatch')
  self.partial_sources=json.load(open(PARTIAL.parent/'SOURCE_HASHES.json'));self.reader.external.update(self.partial_sources)
  # Lists retained only as compact masks/provenance; no coordinate archive preload.
  cols=['observation_id','sequence_hash','split','coordinate_kind','coordinate_path','coordinate_row','source_sha','aa_known','ca_observed','continuity','distance_targets','angle_targets','torsion_targets','AA_only_targets','polymer_length']
  for batch in pq.ParquetFile(PARTIAL).iter_batches(batch_size=4096,columns=cols):
   for r in batch.to_pylist():
    if r['split'] not in ['TRAIN','VALIDATION']:raise ValueError('unexpected partial split')
    if self.base.proposal.sequences[r['sequence_hash']]['split']!=r['split'] or not all(r['aa_known']) or len(r['aa_known'])!=len(self.base.seqs[r['sequence_hash']]):raise ValueError('partial ownership/sequence')
    self.partial[r['observation_id']]=r;self.partial_owners.setdefault(r['sequence_hash'],[]).append(r['observation_id'])
  self.groups={}
  for split in ['TRAIN','VALIDATION']:
   allids=set(self.base.ids[split]);c=set(self.base.jointids[split]);p={h for h in allids if h in self.partial_owners}-c;u=allids-c-p;self.groups[split]=dict(complete=sorted(c),partial=sorted(p),aa_only=sorted(u))
  assert [len(x) for x in self.groups['TRAIN'].values()]==[21475,29129,67615]
  self.cache={};self.group_sets={k:set(v) for k,v in self.groups['TRAIN'].items()}
 def generated_prefix_target(self,state,identity,device):
  # Valid future AA label only with the same known native AA prefix. Coordinates
  # may be generated context, never an invented experimental geometry target.
  if identity not in self.base.ids['TRAIN'] or state.N!=len(self.base.seqs[identity]) or ''.join(state.residues)!=self.base.seqs[identity][:state.t] or state.t>=state.N:raise ValueError('continuation target correspondence')
  return collate([state],device),torch.tensor([AA.index(self.base.seqs[identity][state.t])],device=device),torch.zeros(1,3,device=device,dtype=torch.float64),torch.zeros(1,3,device=device,dtype=torch.bool)
 def membership(self):
  result={}
  for split,groups in self.groups.items():
   rows=[r for r in self.base.rows.values() if r['split']==split];pr=[r for r in self.partial.values() if r['split']==split]
   result[split]=dict(identities={k:len(v) for k,v in groups.items()},membership_sha256={k:hashlib.sha256('\n'.join(v).encode()).hexdigest() for k,v in groups.items()},complete_observations=len(rows),partial_observations=len(pr),complete_component_targets={k:sum(len(self.base.seqs[r['sequence_hash']])-i for r in rows) for i,k in enumerate(['distance','angle','torsion'],1)},partial_component_targets={k:sum(r[k+'_targets'] for r in pr) for k in ['distance','angle','torsion']},aa_positions={k:sum(len(self.base.seqs[h]) for h in v) for k,v in groups.items()})
  return result
 def coordinates(self,r):
  oid=r['observation_id']
  if oid not in self.cache:
   if oid in self.partial:
    path=self.reader.check(r['coordinate_path']);i=int(r['coordinate_row'])
    if r['coordinate_kind']=='legacy_cache':
     pf=pq.ParquetFile(path);offset=0;xy=None
     for k in range(pf.num_row_groups):
      n=pf.metadata.row_group(k).num_rows
      if offset<=i<offset+n:xy=json.loads(self.reader.group(path,k)[i-offset]);break
      offset+=n
     if xy is None:raise ValueError('partial row missing')
    else:
     doc=self.reader.small(path)
     if doc['source_sha256']!=r['source_sha']:raise ValueError('source revision')
     xy=[v['ca'] for v in doc['observations'][i]['residue_mapping']]
    if [v is not None for v in xy]!=r['ca_observed']:raise ValueError('CA mask correspondence')
    x=np.array([v if v is not None else [0.,0.,0.] for v in xy],dtype=np.float64)
   else:x=self.reader.read(r)
   self.cache.clear();self.cache[oid]=x
  return self.cache[oid]
 def decision(self,category,h,t,rng):
  seq=self.base.seqs[h];N=len(seq)
  if h not in self.group_sets[category] or not 0<=t<N:raise ValueError('TRAIN ownership/position')
  if category=='aa_only':return BuildState(tuple(seq[:t]),torch.zeros(t,3,dtype=torch.float64),N,torch.zeros(t,dtype=torch.bool)),AA.index(seq[t]),torch.zeros(3,dtype=torch.float64),torch.zeros(3,dtype=torch.bool),np.zeros(max(0,t-1),dtype=bool),None
  owners=self.base.owners[h] if category=='complete' else self.partial_owners[h];o=rng.choice(owners);r=self.base.rows[o] if category=='complete' else self.partial[o]
  if r['split']!='TRAIN':raise ValueError('observation split')
  x=self.coordinates(r)
  if category=='complete':g=internal(x);x=canonical(x);valid=np.ones(N,dtype=bool);links=np.ones(N-1,dtype=bool);mask=np.arange(1,4)<=t
  else:
   valid=np.array(r['ca_observed'],dtype=bool);links=np.array(r['continuity'],dtype=bool);x=np.asarray(x,dtype=np.float64);x=np.where(valid[:,None],x,0.);g,mask_all,_=derive(x,valid,links);mask=mask_all[t]
  state=BuildState(tuple(seq[:t]),torch.from_numpy(x[:t].copy()),N,torch.tensor(valid[:t]));return state,AA.index(seq[t]),torch.from_numpy(g[t].copy()),torch.tensor(mask),links[:max(0,t-1)],o
 def batch(self,records,device):
  rows=[self.decision(*r) for r in records];b=collate([r[0] for r in rows],device);b['bond_valid']=torch.zeros_like(b['valid'])
  for i,r in enumerate(rows):
   if r[0].t>=2:b['bond_valid'][i,1:r[0].t]=torch.tensor(r[4],device=device)&b['valid'][i,:r[0].t-1]&b['valid'][i,1:r[0].t]
  return b,torch.tensor([r[1] for r in rows],device=device),torch.stack([r[2] for r in rows]).to(device),torch.stack([r[3] for r in rows]).to(device),rows

class CoverageSampler:
 """One shuffled identity sweep before repetition, independent shuffled positions per identity."""
 def __init__(self,pop,seed=1703):
  self.pop=pop;self.rng=random.Random(seed);self.orders=copy.deepcopy(pop.groups['TRAIN']);self.cursors={k:0 for k in self.orders};self.position={};self.cycles={};self.decisions={k:0 for k in self.orders};self.seen={k:set() for k in self.orders};self.quota_cursor=0
  for x in self.orders.values():self.rng.shuffle(x)
 def one(self,k):
  order=self.orders[k];i=self.cursors[k]
  if i==len(order):self.rng.shuffle(order);i=0
  h=order[i];self.cursors[k]=i+1
  if h not in self.position or not self.position[h]:
   positions=list(range(len(self.pop.base.seqs[h])));self.rng.shuffle(positions);self.position[h]=positions;self.cycles[h]=self.cycles.get(h,-1)+1
  t=self.position[h].pop();self.decisions[k]+=1;self.seen[k].add(h);return k,h,t,self.rng
 def draw(self,n=64):
  # Five target-decision slots: two complete, then partial/AA-only, then balanced
  # tenth slot. Exactly 4/3/3 over each ten, including across batch boundaries.
  cycle=['complete','partial','aa_only','complete','partial','aa_only','complete','partial','aa_only','complete'];out=[]
  for _ in range(n):k=cycle[self.quota_cursor%10];self.quota_cursor+=1;out.append(self.one(k))
  return out
 def geometry(self,n=32):return [self.one('complete') for _ in range(n)]
 def state_dict(self):
  return dict(orders=self.orders,cursors=self.cursors,position=self.position,cycles=self.cycles,decisions=self.decisions,seen=self.seen,quota_cursor=self.quota_cursor,rng=self.rng.getstate(),membership=self.pop.membership())
 def load_state_dict(self,s):
  if s['membership']!=self.pop.membership():raise ValueError('sampler membership changed')
  for k in ['orders','cursors','position','cycles','decisions','seen','quota_cursor']:setattr(self,k,copy.deepcopy(s[k]))
  self.rng.setstate(s['rng'])
