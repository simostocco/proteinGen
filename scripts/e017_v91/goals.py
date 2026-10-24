"""Canonical finite-domain ordered-goal features; unchanged production semantics."""
import copy,hashlib,itertools,json,math,runpy
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v91_ordered_goal_recovery_20261023_v1');c=runpy.run_path(str(R/'source/scripts/e017_v91/core.py'));AA=c['AA']
VERSION='E017_ORDERED_GOAL_1';MAX_MOTIF=8;GRID=10000

def integer(x,name):
 if isinstance(x,bool) or not isinstance(x,(int,np.integer)):raise ValueError(name+' must be an integer')
 return int(x)
def threshold(x):
 if isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x):raise ValueError('finite threshold required')
 q=round(x*GRID)
 if not 0<=q<=GRID or abs(x-q/GRID)>1e-12:raise ValueError('threshold outside declared exact1e-4 grid; no silent quantization')
 return q

def canonical(raw):
 """Normalize legacy V9 or explicit canonical goal; omit audit-only metadata."""
 if raw.get('schema')==VERSION:
  allowed={'schema','N','motif','regions','contrast_units'}
  if set(raw)!=allowed:raise ValueError('unknown/missing canonical semantics')
  N=integer(raw['N'],'N');m=copy.deepcopy(raw['motif']);regions=copy.deepcopy(raw['regions']);contrast=integer(raw['contrast_units'],'contrast_units')
  if set(m)!={'id','tokens','allowed_region','absolute_positions','gaps'} or m['id']!='motif_0':raise ValueError('unsupported motif schema')
  if set(regions)!={'exposed','buried'}:raise ValueError('region roles required')
  for region in regions.values():
   if set(region)!={'interval','threshold_units'}:raise ValueError('unknown region semantics')
 else:
  required={'N','motif','exposed','buried','exposure_min','burial_max','contrast_min'}
  metadata={'goal_id','witness_identity','protected_group','motif_scope','native_coordinates_provided','property_signature','condition'}
  if not required<=set(raw) or set(raw)-required-metadata:raise ValueError('unknown/missing legacy semantics; refuse ignored position/spacing overrides')
  if raw.get('motif_scope','anywhere within exposedregion')!='anywhere within exposedregion':raise ValueError('unsupported motif scope')
  if raw.get('native_coordinates_provided',False):raise ValueError('native-future inference forbidden')
  N=integer(raw['N'],'N');tokens=list(raw['motif']);m=dict(id='motif_0',tokens=tokens,allowed_region=list(raw['exposed']),absolute_positions=None,gaps=[[0,0] for _ in tokens[1:]])
  regions=dict(exposed=dict(interval=list(raw['exposed']),threshold_units=threshold(raw['exposure_min'])),buried=dict(interval=list(raw['buried']),threshold_units=threshold(raw['burial_max'])));contrast=threshold(raw['contrast_min'])
 if not 4<=N<=500:raise ValueError('unsupported length')
 tokens=m['tokens']
 if not isinstance(tokens,list) or not 1<=len(tokens)<=MAX_MOTIF or any(not isinstance(a,str) or len(a)!=1 or a not in AA for a in tokens):raise ValueError('canonical AA motif of length1..8 required')
 m['tokens']=list(tokens)
 def interval(v):
  if len(v)!=2:raise ValueError('interval length')
  a,z=[integer(x,'interval') for x in v]
  if not 0<=a<z<=N:raise ValueError('invalid interval')
  return [a,z]
 m['allowed_region']=interval(m['allowed_region']);m['gaps']=[[integer(x,'gap') for x in pair] for pair in m['gaps']]
 if len(m['gaps'])!=len(tokens)-1 or any(len(p)!=2 or not 0<=p[0]<=p[1]<N for p in m['gaps']):raise ValueError('invalid spacing constraints')
 positions=m['absolute_positions']
 if positions is not None:
  if len(positions)!=len(tokens):raise ValueError('position mask mismatch')
  m['absolute_positions']=[None if pos is None else integer(pos,'position') for pos in positions]
  if any(pos is not None and not m['allowed_region'][0]<=pos<m['allowed_region'][1] for pos in m['absolute_positions']):raise ValueError('assigned position outside region')
 for key in ['exposed','buried']:
  regions[key]['interval']=interval(regions[key]['interval']);regions[key]['threshold_units']=integer(regions[key]['threshold_units'],'threshold_units')
  if not 0<=regions[key]['threshold_units']<=GRID:raise ValueError('threshold range')
 if not 0<=contrast<=GRID:raise ValueError('contrast range')
 result=dict(schema=VERSION,N=N,motif=m,regions=regions,contrast_units=contrast)
 if not assignments(result):raise ValueError('infeasible ordering/spacing/position constraints')
 return result

def assignments(goal):
 m=goal['motif'];a,z=m['allowed_region'];L=len(m['tokens']);given=m['absolute_positions'];out=[]
 # Declared domain bounded to regional motifs; refuse combinatorial explosions.
 if math.comb(z-a,L)>100000:raise ValueError('assignment enumeration exceeds bounded schema support')
 for positions in itertools.combinations(range(a,z),L):
  if given is not None and any(p is not None and positions[j]!=p for j,p in enumerate(given)):continue
  if all(lo<=positions[j+1]-positions[j]-1<=hi for j,(lo,hi) in enumerate(m['gaps'])):out.append(positions)
 return out

def serialize(goal):return json.dumps(canonical(goal),sort_keys=True,separators=(',',':'),ensure_ascii=True)
def semantic_sha(goal):return hashlib.sha256(serialize(goal).encode()).hexdigest()
def sequence_progress(seq,goal):
 m=goal['motif'];best=None;best_score=-1
 for positions in assignments(goal):
  hits=[int(pos<len(seq) and seq[pos]==aa) for pos,aa in zip(positions,m['tokens'])];value=sum(hits)/len(hits)
  if value>best_score:best_score=value;best=(positions,hits)
 return dict(utility=best_score,complete=best_score==1,positions=list(best[0]),role_matches=best[1],role_observed=[int(pos<len(seq)) for pos in best[0]])

def production_raw(goal):
 g=canonical(goal);m=g['motif']
 if m['absolute_positions'] is not None or any(pair!=[0,0] for pair in m['gaps']) or m['allowed_region']!=g['regions']['exposed']['interval']:raise ValueError('extended engineering goals are not the original production task')
 return dict(N=g['N'],motif=''.join(m['tokens']),exposed=g['regions']['exposed']['interval'],buried=g['regions']['buried']['interval'],exposure_min=g['regions']['exposed']['threshold_units']/GRID,burial_max=g['regions']['buried']['threshold_units']/GRID,contrast_min=g['contrast_units']/GRID,motif_scope='anywhere within exposedregion')

def public_batch(worlds,prefix,reference):
 from protein_joint_sac.crisp_v3.networks import prepare
 device=prefix['ids'].device;B=len(worlds);canonical_goals=[canonical(w.goal) for w in worlds];G=max(len(g['motif']['tokens']) for g in canonical_goals);features=torch.zeros(B,G,36,device=device);valid=torch.zeros(B,G,dtype=torch.bool,device=device);context=torch.zeros(B,64,device=device)
 for i,(w,g) in enumerate(zip(worlds,canonical_goals)):
  # Only current generated state; audit IDs and witness coordinates never enter.
  assert w.state.N==g['N'],'state/goal horizon mismatch'
  raw=production_raw(g) if g['motif']['absolute_positions'] is None and all(v==[0,0] for v in g['motif']['gaps']) and g['motif']['allowed_region']==g['regions']['exposed']['interval'] else dict(N=g['N'],exposed=g['regions']['exposed']['interval'],buried=g['regions']['buried']['interval'],motif=''.join(g['motif']['tokens']))
  o=c['observe'](''.join(w.state.residues),w.state.ca.numpy(),raw,w.spec,w.interventions,w.mode);N=g['N'];a,z=g['regions']['exposed']['interval'];b,d=g['regions']['buried']['interval'];progress=sequence_progress(o['sequence'],g);m=g['motif'];L=len(m['tokens']);s=c['sequence_stats'](o['sequence'])
  context[i,:11]=torch.tensor([1,g['regions']['exposed']['threshold_units']/GRID,g['regions']['buried']['threshold_units']/GRID,g['contrast_units']/GRID,a/N,z/N,b/N,d/N,w.mode/3,w.state.t/N,1-w.state.t/N],device=device)
  for j,name in enumerate(['exposed','buried']):
   q=o['regions'][name];context[i,11+2*j]=q['current_known_mean'] if q['current_known_mean'] is not None else 0;context[i,12+2*j]=q['generated']/q['total'];context[i,26+j]=q['final_exposure_upper_bound']
  context[i,15]=progress['utility'];context[i,16:20]=torch.tensor([s['entropy_bits']/math.log2(20),s['max_fraction'],s['max_run']/N,s['unique']/20],device=device);context[i,20]=o['clash_count']/max(1,N*.02);context[i,21]=o['irreversible_clash_violation'];context[i,22]=len(w.interventions)/4;context[i,23]=L/MAX_MOTIF;context[i,24]=N/500;context[i,25]=progress['complete'];context[i,28+w.mode]=1
  ex=o['regions']['exposed'];bu=o['regions']['buried'];ex_known=ex['current_known_mean'] is not None;bu_known=bu['current_known_mean'] is not None;context[i,32:40]=torch.tensor([ex_known,bu_known,max(0,g['regions']['exposed']['threshold_units']/GRID-ex['current_known_mean']) if ex_known else 0,max(0,bu['current_known_mean']-g['regions']['buried']['threshold_units']/GRID) if bu_known else 0,ex_known and bu_known,max(0,g['contrast_units']/GRID-(ex['current_known_mean']-bu['current_known_mean'])) if ex_known and bu_known else 0,1-progress['utility'],progress['complete']],device=device)
  for j,aa in enumerate(m['tokens']):
   aa_pos=None if m['absolute_positions'] is None else m['absolute_positions'][j];lo,hi=m['gaps'][j-1] if j else (0,0);u,v=m['allowed_region'];f=features[i,j];f[:12]=torch.tensor([j/MAX_MOTIF,L/MAX_MOTIF,(aa_pos or 0)/N,lo/N,hi/N,u/N,v/N,1,0,progress['role_matches'][j],progress['role_observed'][j],max(0,v-w.state.t)/(v-u)],device=device);f[12+AA.index(aa)]=1
   # Virtual role anchor is explicit encoding, not an imposed motif location.
   f[32]=(aa_pos if aa_pos is not None else (u+v)/2+j)/N;f[33]=1;f[34]=aa_pos is not None;f[35]=1;valid[i,j]=True
 return prepare(prefix,reference,features,valid,context,torch.tensor([w.mode for w in worlds],device=device))
