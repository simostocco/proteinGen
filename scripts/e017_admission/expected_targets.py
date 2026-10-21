from pathlib import Path
import json,numpy as np,pyarrow.parquet as pq
from collections import defaultdict
from protein_joint_sac.crisp_v3.data import Population
R=Path(__file__).resolve().parents[3];p=Population();rates={}
for category in ['complete','partial','aa_only']:
 values=[]
 for h in p.groups['TRAIN'][category]:
  N=len(p.base.seqs[h])
  if category=='complete':values.append([(N-k)/N for k in [1,2,3]])
  elif category=='aa_only':values.append([0.,0.,0.])
  else:
   obs=[p.partial[o] for o in p.partial_owners[h]];values.append([np.mean([r[c+'_targets']/N for r in obs]) for c in ['distance','angle','torsion']])
 rates[category]=np.mean(values,axis=0).tolist()
expect={}
for u in [250,1000,2500,5000,10000]:
 per={}
 for j,c in enumerate(['distance','angle','torsion']):
  primary=64*u*(.4*rates['complete'][j]+.3*rates['partial'][j]);geometry=32*u*rates['complete'][j];goals=8*u;per[c]=dict(primary_AA=primary,separate_complete_geometry=geometry,goal_demonstrations=goals,total=primary+geometry+goals)
 expect[u]=per
out=dict(expected_component_validity_rates_per_identity_first_uniform_position=rates,expected_component_target_counts=expect,interpretation='Analytical expectation, not actual training exposure or an exact deterministic quota. Partial values use frozen recorded per-observation counts with identity-first uniform observation selection; geometry target masks remain authoritative at runtime.',optimizer_updates=0)
with open(R/'EXPECTED_GEOMETRY_TARGET_COUNTS.json','x') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
