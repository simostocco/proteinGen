"""Publish a stopped-at-admission V9 result without fabricating downstream stages."""
from pathlib import Path
import collections,hashlib,json,os,runpy,shutil,subprocess,time
import numpy as np
R=Path('/mnt/d/Simone/proteinGen_data/e017_v9_property_steering_20261023_v1');c=runpy.run_path(str(R/'source/scripts/e017_v9/core.py'));read=c['read'];save=c['save'];sha=c['sha']
def write(name,text):
 p=R/name;q=p.with_suffix(p.suffix+'.part')
 with q.open('w') as f:f.write(text);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def ci(values):
 a=np.array(values,float)
 if not len(a):return None
 v=a[np.random.default_rng(19917).integers(0,len(a),(10000,len(a)))].mean(1)
 return dict(estimate=float(a.mean()),CI95=np.quantile(v,[.025,.975]).tolist(),independent_goal_identities=len(a),replicates=10000,seed=19917,meaning='identity bootstrap; all-zero intervals do not prove population risk is zero')
def rows_for(panel,policy):return [read(p) for p in sorted((R/'results').glob(f'{panel}_{policy}_[0-9][0-9][0-9].json'))]
def summarize(panel,policy):
 groups=rows_for(panel,policy);rows=[r for g in groups for r in g['records']];metrics=[r['metrics'] for r in rows]
 if not rows:return dict(status='NOT_EXECUTED',reason='prospective evaluation blocked at earlier admission')
 K=min(len(g['records']) for g in groups);out=dict(goals=len(groups),trajectories=len(rows),K_max=K,population=panel,independent_identity_count=len({g['goal']['witness_identity'] for g in groups}),sequence_complete=sum(m['complete'] for m in metrics),backbone_valid=sum(m['backbone_valid'] for m in metrics),sequence_valid=sum(m['sequence_valid'] for m in metrics),packing_valid=sum(m['packing_valid'] for m in metrics),hard_valid=sum(m['hard_valid'] for m in metrics),full_success=sum(m['success'] for m in metrics),mean_reward=float(np.mean([m['reward'] for m in metrics])),reward_SD=float(np.std([m['reward'] for m in metrics])),failure_causes=dict(collections.Counter(f for m in metrics for f in m['geometry']['failures'])),sequence_exact_unique=len({r['sequence'] for r in rows}),support_failures=sum(bool(r['support_failure']) for r in rows),represented_float_repairs=sum(r['repairs'] for r in rows),success_by_K={})
 for k in [1,2,4,8]:
  if k>K:continue
  a=[int(any(r['metrics']['success'] for r in g['records'][:k])) for g in groups];successes=sum(a);n=len(a);from scipy.stats import beta
  exact=[0. if successes==0 else float(beta.ppf(.025,successes,n-successes+1)),1. if successes==n else float(beta.ppf(.975,successes+1,n-successes))];out['success_by_K'][str(k)]=dict(successful_goals=successes,denominator=n,binomial_CP95=exact,**ci(a))
 for key in ['exposed_mean','buried_mean','contrast','motif_utility','compatibility_score']:
  z=[m[key] for m in metrics if m.get(key) is not None];out[key]=dict(mean=float(np.mean(z)),SD=float(np.std(z))) if z else None
 for key in ['AA_KL','geometry_KL_per_active']:
  z=[v for r in rows for v in (r[key][1:] if key.startswith('geometry') else r[key])];out[key]=dict(mean=float(np.mean(z)),max=float(np.max(z))) if z else None
 entropy=[m['sequence']['entropy_bits'] for m in metrics];out['sequence_entropy_bits_mean']=float(np.mean(entropy));pid=[];rms=[]
 for g in groups:
  rs=g['records'];coordinate_cache={}
  for i in range(len(rs)):
   for j in range(i):
    a,b=rs[i],rs[j]
    if len(a['sequence'])!=len(b['sequence']):continue
    pid.append(sum(x==y for x,y in zip(a['sequence'],b['sequence']))/len(a['sequence']))
    def xyz(r):
     
     key=r['seed']
     if key in coordinate_cache:return coordinate_cache[key]
     world=c['PropertyWorld'](g['goal'],read(R/'REWARD_SPECIFICATION.json'),r['seed'])
     for aa,geom in zip(r['sequence'],r['geometry']):world.append(aa,geom)
     coordinate_cache[key]=world.state.ca.numpy();return coordinate_cache[key]
    if len(a['geometry'])==g['goal']['N'] and len(b['geometry'])==g['goal']['N']:
     x,y=xyz(a),xyz(b);x=x-x.mean(0);y=y-y.mean(0);u,_,v=np.linalg.svd(x.T@y);D=np.eye(3);D[2,2]=np.linalg.det(u@v);rms.append(float(np.sqrt(np.square(x@(u@D@v)-y).sum(1).mean())))
 out['within_goal_pairwise_sequence_identity_mean']=float(np.mean(pid)) if pid else None;out['within_goal_Kabsch_CA_RMSD_A_mean']=float(np.mean(rms)) if rms else None
 return out
