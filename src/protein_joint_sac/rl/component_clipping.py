"""AA-first shared cap and disjoint physical-output gradient blocks; no global G clip."""
import math
import torch
from .group_clipping import groups,norm
VERSION='aa_first_geometry_output_blocks_v1'
ROWS={'distance':(0,3),'angle':(1,4),'torsion':(2,5)}

def blocks(model):
 gs,names=groups(model);p=dict(model.named_parameters());w=p['geom.2.weight'];b=p['geom.2.bias']
 if w.shape!=(6,model.encoder.config.width) or b.shape!=(6,):raise ValueError('unexpected geometry head channel mapping')
 out={k:[(n,q,None) for n,q in gs[k]] for k in ['S','A']};out['Gstem']=[(n,q,None) for n,q in gs['G'] if n not in ['geom.2.weight','geom.2.bias']]
 for k,rows in ROWS.items():out[k]=[('geom.2.weight',w,rows),('geom.2.bias',b,rows)]
 counts={k:sum(q.numel() if rows is None else q[list(rows)].numel() for n,q,rows in v) for k,v in out.items()}
 if sum(counts.values())!=sum(q.numel() for q in model.parameters() if q.requires_grad):raise ValueError('gradient entries not completely partitioned')
 if sorted(r for rows in ROWS.values() for r in rows)!=list(range(6)):raise ValueError('geometry row overlap/missing')
 return out,counts

def combine(model,aa,limits=None,ratio=.5):
 gs,_=groups(model);bs,counts=blocks(model)
 for n,p in model.named_parameters():
  for v in [p.grad,aa.get(n)]:
   if v is not None and not bool(torch.isfinite(v).all()):raise FloatingPointError('nonfinite gradient '+n)
 an=norm([aa[n] for n,p in gs['S']]);gn=norm([p.grad for n,p in gs['S']]);alpha=min(1.,ratio*an/max(gn,1e-30));dot=sum(float((aa[n].float()*p.grad.float()).sum()) for n,p in gs['S'] if aa[n] is not None and p.grad is not None)
 if norm([p.grad for n,p in gs['A']])!=0 or norm([aa[n] for n,p in gs['G']])!=0:raise ValueError('exclusive-head gradient bypass')
 with torch.no_grad():
  for k,items in gs.items():
   for n,p in items:
    a=aa[n]
    if k=='S' and p.grad is not None:p.grad.mul_(alpha)
    if a is not None:
     if p.grad is None:p.grad=a.clone()
     else:p.grad.add_(a)
 def measure(v):return norm([p.grad if rows is None else p.grad[list(rows)] for n,p,rows in v if p.grad is not None])
 before={k:measure(v) for k,v in bs.items()};factors={k:min(1.,limits[k]/max(before[k],1e-30)) if limits else 1. for k in bs}
 with torch.no_grad():
  for k,v in bs.items():
   for n,p,rows in v:
    if p.grad is None:continue
    if rows is None:p.grad.mul_(factors[k])
    else:p.grad[list(rows)]=p.grad[list(rows)]*factors[k]
 after={k:measure(v) for k,v in bs.items()}
 return {'policy':VERSION,'AA_shared_L2':an,'weighted_geometry_shared_raw_L2':gn,'geometry_shared_scale':alpha,'geometry_to_AA_shared_ratio':alpha*gn/max(an,1e-30),'AA_geometry_shared_cosine':dot/max(an*gn,1e-30),'group_before':before,'group_after':after,'group_scale':factors,'AA_shared_after_S_clip_L2':an*factors['S'],'AA_head_retention':factors['A'],'G_head_scaled_by_shared_cap':False,'whole_model_clip_applied':False,'entry_counts':counts}
