"""Read-only CPU support-boundary reproduction; no model or optimizer."""
import json,math
from pathlib import Path
import torch
from protein_joint_sac.models.hybrid_distributions import GeometryDistribution
from protein_joint_sac.geometry.ca_builder import next_ca
R=Path('/mnt/d/Simone/proteinGen_data/e016_j1_sequence_first_20261018_v1')
rows=[]
ca=torch.tensor([[0.,0.,0.],[3.8,0.,0.],[3.8,3.8,0.]],dtype=torch.float64)
for dtype in [torch.float32,torch.float64]:
 u=torch.tensor([math.log(3.8),0.,10.],dtype=dtype)
 d=GeometryDistribution(torch.zeros(3,dtype=dtype),torch.zeros(3,dtype=dtype),(True,True,True))
 g=d.transform(u).double()
 try:
  x=next_ca(ca,g);status='ACCEPTED';error=None
 except ValueError as e:status='REJECTED';error=str(e)
 rows.append({'transform_dtype':str(dtype),'latent_u':u.tolist(),'geometry_widened_to_FP64':g.tolist(),'torsion_minus_math_pi':float(g[2])-math.pi,'builder_status':status,'error':error})
assert rows[0]['error']=='signed torsion [-pi,pi]'
assert rows[1]['builder_status']=='ACCEPTED'
p=R/'TORSION_PRECISION_SUPPORT_FIXTURE.json'
assert not p.exists()
p.write_text(json.dumps({'scope':'CPU mathematical fixture only; no model execution, no training, no modified transforms','rows':rows,'observed_generation_failure':{'checkpoint_additional_update':7500,'length':384,'sample_index':13,'seed':2026140113,'completed_residues':336,'error':'signed torsion [-pi,pi]'},'interpretation':'Existing FP32 transform can exceed Python pi after widening. This reproduces a compatible mechanism; actual failed latent draw was not retained, so exact draw-level attribution remains unverified. No generation sample replaced or retried.'},indent=2)+'\n')
print(p,flush=True)
