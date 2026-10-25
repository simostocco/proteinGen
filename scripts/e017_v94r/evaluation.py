"""Frozen reserved-TRAIN diagnostics; no VALIDATION selection or TEST reads."""
import csv, itertools, json, math, os, runpy, time
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/d/Simone/proteinGen_data/e017_v94r_packing_safety_20261025_v1')
P=runpy.run_path(str(ROOT/'source/scripts/e017_v94r/production.py'))
S=P['S']; V93=P['V93']; read=P['read']; compact=P['compact']

def summarize(groups):
    rows=[r for g in groups for r in g['records']]; m=[r['metrics'] for r in rows]; div=[]
    for g in groups:
        coords=[S['reconstruct'](r).ca.numpy() for r in g['records'] if r['metrics']['complete']]
        values=[]
        for a,b in itertools.combinations(coords,2):
            a=a-a.mean(0); b=b-b.mean(0); u,_,v=np.linalg.svd(a.T@b)
            rot=u@np.diag([1,1,np.linalg.det(u@v)])@v
            values.append(float(np.sqrt(np.square(a@rot-b).sum()/64)))
        if values: div.append(float(np.mean(values)))
    exposed=[];buried=[]; errors=[]
    for g in groups:
        goal=g['task']['goal']
        for r in g['records']:
            v=r['metrics']; e=v.get('exposed_mean',None); b=v.get('buried_mean',None)
            exposed.append(e is not None and e>=goal['exposure_min'])
            buried.append(b is not None and b<=goal['burial_max'])
            if e is not None and b is not None: errors.append(max(0,goal['exposure_min']-e)+max(0,b-goal['burial_max'])+max(0,.1-e+b))
    return dict(identities=len({g['task']['source_reference_identity'] for g in groups}),goals=len(groups),candidates=len(rows),
        K1=sum(g['records'][0]['metrics']['success'] for g in groups)/len(groups),
        K8=sum(any(r['metrics']['success'] for r in g['records'][:8]) for g in groups)/len(groups),
        hard_valid=float(np.mean([v['hard_valid'] for v in m])),backbone_valid=float(np.mean([v['backbone_valid'] for v in m])),
        complete=float(np.mean([v['complete'] for v in m])),exposure_success=float(np.mean(exposed)),burial_success=float(np.mean(buried)),
        full_success=float(np.mean([v['success'] for v in m])),target_error=float(np.mean(errors)) if errors else 3.,
        reward=float(np.mean([v['reward'] for v in m])),geometry_KL_per_active=float(np.mean([v for r in rows for v in r['geometry_KL']])),
        structural_pair_RMSD=float(np.mean(div)) if div else 0.,proposals=sum(len(r['proposals']) for r in rows),
        global_contact_violations=sum(r['metrics']['complete'] and r['metrics']['geometry']['longrange_contact_degree']>S['SPEC']['packing_limits']['64'][1] for r in rows),rejections=sum(not q[6] for r in rows for q in r['proposals']),aborts=sum(r['termination'] is not None for r in rows))

def development(rt,update):
    path=rt.root/f'diagnostics/DEVELOPMENT_{update:05}.json'; fp=P['c']['state_digest'](rt.e.actor.state_dict())
    if path.exists():
        obj=read(path); assert obj['actor_fingerprint']==fp; return obj['summary']
    tasks=read(V93/'manifests/TRAIN_TASKS.json')['tasks']; result={}; start=time.perf_counter()
    for policy,label in [('C','filtered'),('B','underlying_unfiltered')]:
        groups=[]
        for i,task in enumerate(tasks):
            gp=rt.root/f'diagnostics/DEV_{update:05}_{label}_{i:02}.json'
            if gp.exists():
                obj=read(gp); assert obj['actor_fingerprint']==fp
            else:
                seeds=[S['ns']('CANDIDATE',task['goal']['goal_id'],j) for j in range(8)]
                rows=P['sample'](rt.e,task,policy,8,seeds)
                obj=dict(task=task,records=rows,actor_fingerprint=fp); compact(gp,obj)
            groups.append(obj)
        result[label]=summarize(groups)
        if policy=='B':result[label]['invalid_proposal_checks']=result[label]['rejections'];result[label]['rejections']=0
    compact(path,dict(summary=result,actor_fingerprint=fp,seconds=time.perf_counter()-start,TRAIN_development_only=True,
        manifest_SHA256=P['sha'](V93/'manifests/TRAIN_TASKS.json')))
    return result

def curves(root):
    fields=['update','identity','goal_id','attempts','complete','full','hard','backbone','accepted_actions','residue_actions',
        'proposals','rejected','aborts','reward_mean','reward_SD','geometry_KL','trace_log_ratio','gradient_norm','step_KL','lr','seconds','D_free_bytes','peak_GPU_bytes']
    path=ROOT/'RL_TRAINING_CURVES.csv'; tmp=path.with_suffix('.csv.part')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for cp in sorted((Path(root)/'commits').glob('u*.json')):
            item=read(cp); l=item['logical']; row=dict(l['stats'],**{k:l[k] for k in ['update','identity','goal_id','gradient_norm','step_KL','lr']},
                **{k:item[k] for k in ['seconds','D_free_bytes','peak_GPU_bytes']}); w.writerow({k:row[k] for k in fields})
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
