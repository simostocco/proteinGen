"""Accepted supervised data, RAM-only derived CA cache and mode-independent exposure schedules."""
import collections,hashlib,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from .pdb_split_proposal import PDBSplitProposal
from .accepted_supervised import CoordinateReader
from .certified_unified import file_sha
from ..contracts import AA,BuildState

P=Path('/mnt/d/Simone/proteinGen_data/pdb_supervised_split_50id80cov_20261016_v1')
PILOT=Path('/mnt/d/Simone/proteinGen_data/e016_real_pdb_pilots_20261016_v1')

def internal(x):
 b=np.diff(x,axis=0);d=np.linalg.norm(b,axis=1);theta=np.arctan2(np.linalg.norm(np.cross(b[:-1],b[1:]),axis=1),-(b[:-1]*b[1:]).sum(1));u=b[1:-1]/d[1:-1,None];v=-b[:-2];v=v-(v*u).sum(1)[:,None]*u;w=b[2:]-(b[2:]*u).sum(1)[:,None]*u;v/=np.linalg.norm(v,axis=1)[:,None];w/=np.linalg.norm(w,axis=1)[:,None];tau=np.arctan2((np.cross(u,v)*w).sum(1),(v*w).sum(1));g=np.zeros((len(x),3),dtype=np.float64);g[1:,0]=d;g[2:,1]=theta;g[3:,2]=tau
 if not np.isfinite(g).all() or (d<=0).any() or (theta<=0).any() or (theta>=np.pi).any() or (np.abs(tau)>=np.pi).any():raise ValueError('source geometry outside unchanged action support')
 return g

def canonical(x):
 x=np.asarray(x,dtype=np.float64);a=x[1]-x[0];a/=np.linalg.norm(a);b=x[2]-x[1];b-=b.dot(a)*a;b/=np.linalg.norm(b);c=np.cross(a,b);y=(x-x[0])@np.stack([a,b,c]).T;y[0]=0;return y

class Data:
 def __init__(self,coordinates=False,report=None):
  start=time.monotonic();self.proposal=PDBSplitProposal(P);seal=json.load(open(PILOT/'PROVENANCE_SHA256.json'))
  for n in ['ADMISSION.json','PRIOR_MEMBERSHIP_VERIFICATION.json','priors/GEOMETRY_PRIORS.json','priors/SEQUENCE_STATS.json','priors/RAW_EXTREME_SOURCE_CHECK.json']:
   if file_sha(PILOT/n)!=seal['artifact_sha256'][n]:raise ValueError('prior/admission integrity mismatch')
  self.seqs={h:r['canonical_full_sequence'] for h,r in self.proposal.sequences.items()};self.owners=collections.defaultdict(list);self.rows={};self.native={};self.ca={};self.g={}
  for split in ['TRAIN','VALIDATION']:
   for row in self.proposal.iter_joint(split):
    r={k:row[k] for k in ['observation_id','sequence_hash','split','role','coordinate_path','coordinate_row','coordinate_kind','source_sha','entry_id']};self.rows[r['observation_id']]=r;self.owners[r['sequence_hash']].append(r['observation_id'])
  for v in self.owners.values():v.sort()
  if coordinates:
   reader=CoordinateReader(self.proposal.external)
   for i,(row,x) in enumerate(reader.stream(self.rows.values())):
    h=row['sequence_hash'];o=row['observation_id']
    if x.shape!=(len(self.seqs[h]),3):raise ValueError('complete polymer correspondence')
    self.g[o]=internal(x);self.ca[o]=canonical(x)
    if i%10000==0:print(json.dumps({'preloaded_CA_observations':i,'seconds':time.monotonic()-start}),flush=True)
  self.ids={split:sorted(h for h,r in self.proposal.sequences.items() if r['split']==split) for split in ['TRAIN','VALIDATION','TEST']}
  self.jointids={split:[h for h in self.ids[split] if self.proposal.sequences[h]['joint_eligible']] for split in self.ids}
  if [len(self.ids[s]) for s in self.ids]!=[118219,14775,14775] or [len(self.jointids[s]) for s in self.ids]!=[21475,2947,2949]:raise ValueError('frozen population census changed')
  if report is not None:report.update({'seconds':time.monotonic()-start,'coordinates_loaded':len(self.ca),'coordinate_cache_bytes':sum(x.nbytes+self.g[o].nbytes for o,x in self.ca.items()),'proposal_manifest_sha256':file_sha(P/'PROPOSAL_MANIFEST.json'),'geometry_priors_sha256':file_sha(PILOT/'priors/GEOMETRY_PRIORS.json')})
 def target(self,h,o,t,joint):
  s=self.seqs[h];N=len(s)
  if not 0<=t<N:raise ValueError('prefix boundary')
  if joint:
   if o not in self.ca or self.rows[o]['sequence_hash']!=h:raise ValueError('observation ownership/cache')
   if t>=3:x=self.ca[o][:t].copy()
   elif t==2:x=np.array([[0.,0.,0.],[self.g[o][1,0],0.,0.]])
   else:x=np.zeros((t,3))
   state=BuildState(tuple(s[:t]),torch.from_numpy(x),N);g=torch.from_numpy(self.g[o][t].copy())
  else:state=BuildState(tuple(s[:t]),torch.zeros(t,3,dtype=torch.float64),N,torch.zeros(t,dtype=torch.bool));g=None
  return state,AA.index(s[t]),g
 def panel(self,matched=True,per_band=64,seed=2026):
  ids=self.jointids['VALIDATION'] if matched else self.ids['VALIDATION'];rng=random.Random(seed);chosen=[]
  for lo,hi in [(20,64),(65,128),(129,256),(257,384),(385,500)]:
   candidates=[h for h in ids if lo<=len(self.seqs[h])<=hi];rng.shuffle(candidates);groups=set();picked=[]
   for h in candidates:
    g=self.proposal.sequences[h]['protected_group_id']
    if g not in groups:picked.append(h);groups.add(g)
    if len(picked)==per_band:break
   if len(picked)!=per_band:raise ValueError('insufficient stratified independent development groups')
   chosen.extend(picked)
  result=[]
  for h in chosen:
   N=len(self.seqs[h]);o=self.owners[h][0] if h in self.owners else None
   for t in sorted(rng.sample(range(N),4)):result.append((h,o,t))
  return result

class Sampler:
 def __init__(self,data,full=False,seed=42):
  self.data=data;self.identities=data.ids['TRAIN'] if full else data.jointids['TRAIN'];self.ri=random.Random(seed+1);self.rt=random.Random(seed+2);self.ro=random.Random(seed+3);self.cursor=0;self.tokens=0;self.seen=set()
 def draw(self,n):
  out=[]
  for _ in range(n):
   h=self.ri.choice(self.identities);t=self.rt.randrange(len(self.data.seqs[h]));owners=self.data.owners.get(h,[]);o=self.ro.choice(owners) if owners else None;out.append((h,o,t));self.cursor+=1;self.tokens+=t;self.seen.add(h)
  return out
 def state_dict(self):return {'ri':self.ri.getstate(),'rt':self.rt.getstate(),'ro':self.ro.getstate(),'cursor':self.cursor,'tokens':self.tokens,'seen':sorted(self.seen),'identity_sha256':hashlib.sha256('\n'.join(self.identities).encode()).hexdigest()}
 def load_state_dict(self,x):
  if x['identity_sha256']!=self.state_dict()['identity_sha256']:raise ValueError('sampler membership changed')
  self.ri.setstate(x['ri']);self.rt.setstate(x['rt']);self.ro.setstate(x['ro']);self.cursor=x['cursor'];self.tokens=x['tokens'];self.seen=set(x['seen'])
