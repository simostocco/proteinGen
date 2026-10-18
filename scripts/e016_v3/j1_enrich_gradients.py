"""Enrich measured gradients/moments; never construct an optimizer or call step."""
import json,math,collections,gc
from pathlib import Path
import numpy as np
import torch
import j1_torsion_diagnostic as d
R=d.R;g=json.load(open(R/'GRADIENT_DIAGNOSTICS.json'));m=torch.load(d.CP['pilot2500'],map_location='cpu',weights_only=True)['model'];names=list(g['pilot2500']['adam_moments_by_parameter']);counts={k:0 for k in ['S','A','G']};layers=collections.Counter()
for n in names:
 group='S' if n.startswith('encoder.') else 'A' if n.startswith('aa_head.') else 'G';counts[group]+=m[n].numel();layer='.'.join(n.split('.')[:3]) if n.startswith('encoder.blocks.') else n.split('.')[0]+('.pair' if n.startswith('encoder.pair.') else '');layers[layer]+=m[n].numel()
if sum(counts.values())!=12351746:raise ValueError('parameter census mismatch')
for tag,v in g.items():
 v['group_parameter_counts']=counts;v['per_layer_parameter_counts']=dict(layers)
 for row in v['batches']:
  for loss,x in row['component_gradients'].items():
   weight=1 if loss=='AA' else (0.23729081725003282 if loss=='nonlocal' else 0.2368960244440583/3)
   x['raw_RMS']={k:x['raw_L2'][k]/math.sqrt(counts[k]) for k in counts};x['per_layer_RMS']={k:z/math.sqrt(layers[k]) for k,z in x['per_layer_L2'].items()};x['weighted_L2']={k:z*weight for k,z in x['raw_L2'].items()};x['post_cap_L2']={k:z*(1 if loss=='AA' else row['aux_shared_scale']) for k,z in x['weighted_L2'].items()};x['post_group_clip_contribution_L2']={k:z*row['group_clip_scale'][k] for k,z in x['post_cap_L2'].items()}
d.write('GRADIENT_DIAGNOSTICS.json',g);del m;gc.collect();mom={}
for tag in ['original','source5000','pilot1500','pilot2500']:
 st=torch.load(d.CP[tag],map_location='cpu',weights_only=True);opt=st['optimizer'];flat=[i for q in opt['param_groups'] for i in q['params']];rr={}
 for n,i in zip(names,flat):
  if n not in ['geom.2.weight','geom.2.bias']:continue
  p=st['model'][n];v=opt['state'][i]
  if v['exp_avg'].shape!=p.shape:raise ValueError('moment-name shape mapping mismatch')
  rr[n]=[{'component':('mu_distance','mu_angle','mu_torsion','logstd_distance','logstd_angle','logstd_torsion')[j],'moment_RMS':float(v['exp_avg'][j].double().square().mean().sqrt()),'second_moment_mean':float(v['exp_avg_sq'][j].double().mean()),'moment_preconditioned_direction_RMS':float((v['exp_avg'][j].double()/(v['exp_avg_sq'][j].double().sqrt()+opt['param_groups'][0].get('eps',1e-8))).square().mean().sqrt())} for j in range(6)]
 mom[tag]={'output_rows':rr,'learning_rate_at_checkpoint':opt['param_groups'][0]['lr'],'note':'normalized moments describe stored AdamW conditioning, not measured parameter deltas or a hypothetical optimizer step'};del st,opt;gc.collect()
d.write('ADAMW_OUTPUT_ROW_DIAGNOSTICS.json',mom)
updates=[]
for p in (d.O/'runs/J1').glob('updates_*.jsonl'):
 for line in open(p):
  row=json.loads(line)
  if 'parameter_update_L2_by_group' in row:updates.append({'pilot_update':row['pilot_update'],'parameter_update_L2_by_group':row['parameter_update_L2_by_group']})
d.write('HISTORICAL_PARAMETER_UPDATE_DIAGNOSTICS.json',{'observed':updates,'source':'archived per25-update training logs, not recomputed optimizer steps'})
print('GRADIENT_ENRICHMENT_COMPLETE',counts)
