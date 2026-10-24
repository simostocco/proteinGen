"""Future reserved-TRAIN development callback; no optimizer or TEST access."""
import itertools,runpy,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v93_regional_controllability_20261024_v1');S=runpy.run_path(str(R/'source/scripts/e017_v93/study.py'));read=S['read'];sha=S['sha']
class Live:
 def __init__(self,e):self.e=e
 def distribution(self,ws,tag):return self.e.distribution(ws,'D_RL' if tag=='C_FIXED' else tag)
def evaluate(e,root,update):
 path=Path(root)/f'DEVELOPMENT_{update:05}.json'
 if path.exists():return read(path)['summary']
 tasks=read(R/'manifests/TRAIN_TASKS.json')['tasks'];metrics=[];groups=[];div=[];adapter=Live(e);S['generate'].__globals__['START']=time.perf_counter();S['COUNTS'].clear()
 for task in tasks:
  rows=S['generate'](adapter,task,'C',8);groups.append(dict(task=task,records=rows));metrics+=[r['metrics'] for r in rows];coords=[S['reconstruct'](r).ca.numpy() for r in rows if r['metrics']['complete']];rms=[]
  for a,b in itertools.combinations(coords,2):
   a=a-a.mean(0);b=b-b.mean(0);u,_,v=np.linalg.svd(a.T@b);rot=u@np.diag([1,1,np.linalg.det(u@v)])@v;rms.append(np.sqrt(np.square(a@rot-b).sum()/64))
  if rms:div.append(float(np.mean(rms)))
 summary=dict(update=update,identities=len(tasks),candidates=len(metrics),K1=sum(g['records'][0]['metrics']['success'] for g in groups)/len(tasks),K8=sum(any(r['metrics']['success'] for r in g['records']) for g in groups)/len(tasks),hard_valid=float(np.mean([m['hard_valid'] for m in metrics])),backbone_valid=float(np.mean([m['backbone_valid'] for m in metrics])),geometry_KL_per_active=float(np.mean([v for g in groups for r in g['records'] for v in r['geometry_KL']])),structural_pair_RMSD=float(np.mean(div)))
 S['c']['atomic_json'](path,dict(summary=summary,TRAIN_development_only=True,task_manifest_SHA256=sha(R/'manifests/TRAIN_TASKS.json'),behavior_fingerprint=S['c']['state_digest'](e.actor.state_dict()),records=groups));return summary
