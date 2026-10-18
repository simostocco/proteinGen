"""Identity-bootstrap paired component changes on unchanged frozen panels."""
import json,math
import numpy as np
import j1_torsion_diagnostic as d
out={};panels=json.load(open(d.O/'configs/EVALUATION_PANELS.json'))
for role in ['TRAIN','VALIDATION']:
 panel=sorted(panels[role],key=lambda x:(x[2],x[0],x[1]));ids=np.array([h for h,o,t in panel]);_,ix=np.unique(ids,return_inverse=True);a=np.load(d.R/f'arrays/pilot500_{role}.npz');mask=a['mask'];rng=np.random.default_rng(811);boot=rng.integers(2947,size=(500,2947));result={}
 for tag in ['original','source5000','pilot1000','pilot1500','pilot2500']:
  b=np.load(d.R/f'arrays/{tag}_{role}.npz');change=b['nll']-a['nll'];v={}
  for c,k in enumerate(['distance','angle','torsion']):
   mean=np.bincount(ix[mask[:,c]],weights=change[mask[:,c],c])/7;v[k]={'delta_mean':float(mean.mean()),'identity_bootstrap95':np.quantile(mean[boot].mean(1),[.025,.975]).tolist()}
  result[tag]=v
 out[role]=result
d.write('PAIRED_COMPONENT_UNCERTAINTY.json',out)
print('PAIRED_IDENTITY_BOOTSTRAP_COMPLETE')
