"""TRAIN-only native goal construction; target poses expressed in initial-anchor gauge."""
import random
import numpy as np
import torch
from protein_joint_sac.data.scaled_supervised import canonical
from .environment import frame,goal_tokens

CURRICULUM=[dict(length=[64,96],distance_A=[0,400],orientation_deg=45,motifs=1,category='P0'),dict(length=[64,96],distance_A=[0,400],orientation_deg=45,motifs=2,category='P1'),dict(length=[64,128],distance_A=[0,500],orientation_deg=35,motifs=2,category='P2'),dict(length=[64,256],distance_A=[0,1000],orientation_deg=25,motifs=3,category='P2')]
class Goals:
 def __init__(self,population,seed=17,limits=None):self.population=population;self.rng=random.Random(seed);self.stage=0;self.draws=0;self.limits=limits;self.rejected=0;self.prefix_cursor=0
 def draw(self,stage=None,_attempt=0):
  if _attempt>=64:raise ValueError('no native-calibrated feasible goal in 64 bounded draws')
  c=CURRICULUM[self.stage if stage is None else stage];p=self.population;ids=[h for h in p.groups['TRAIN']['complete'] if c['length'][0]<=len(p.base.seqs[h])<=c['length'][1]]
  if not ids:raise ValueError('curriculum has no qualified TRAIN structures')
  h=self.rng.choice(ids);o=self.rng.choice(p.base.owners[h]);r=p.base.rows[o]
  if r['split']!='TRAIN':raise ValueError('TRAIN-only goal source')
  x=canonical(p.coordinates(r));seq=p.base.seqs[h];N=len(seq);kind=c['category'];targets=[]
  if kind!='P0':
   indices=[N-1] if kind=='P1' else [N//2,N-1]
   for i in indices:targets.append(dict(index=i,position=x[i].tolist(),tolerance_A=8.,orientation=frame(torch.tensor(x[:i+1])).tolist() if kind=='P2' else None,orientation_tolerance_deg=c['orientation_deg']))
  motifs={str(i):seq[i] for i in self.rng.sample(range(3,N),c['motifs'])};self.draws+=1
  task=dict(schema='crisp_goal_v3',kind=kind,N=N,targets=targets,motifs=motifs,source_identity=h,source_observation=o,source_sha=r['source_sha'],split='TRAIN',initial_sequence=seq[:3],initial_CA=x[:3].tolist(),experimental=True)
  from .environment import assess
  from protein_joint_sac.contracts import BuildState
  distance=float(np.linalg.norm(x[-1]-x[2]))
  if not c['distance_A'][0]<=distance<=c['distance_A'][1] or (self.limits is not None and not assess(BuildState(tuple(seq),torch.tensor(x),N),task,self.limits)['success']):
   self.rejected+=1;return self.draw(stage,_attempt+1)
  return task
 def variants(self,task,n=3):
  import copy
  out=[]
  for i in range(n):
   t=copy.deepcopy(task)
   for p in t['targets']:p['tolerance_A']=[4.,8.,12.][i%3]
   t['motifs']={k:v for j,(k,v) in enumerate(t['motifs'].items()) if j<=i};t['variation']=i;out.append(t)
  return out
 def state_dict(self):return dict(rng=self.rng.getstate(),stage=self.stage,draws=self.draws,rejected=self.rejected,limits=self.limits,prefix_cursor=self.prefix_cursor)
 def load_state_dict(self,s):self.rng.setstate(s['rng']);self.stage=s['stage'];self.draws=s['draws'];self.rejected=s['rejected'];self.prefix_cursor=s['prefix_cursor'];assert self.limits==s['limits']
