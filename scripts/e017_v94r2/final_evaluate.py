"""Frozen post-budget scientific evaluation. No optimizer is constructed."""
import collections,json,runpy,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v94r3_substantial_geometry_rl_20261026_v1')
P=runpy.run_path(str(R/'source/scripts/e017_v94r/production.py'));E=runpy.run_path(str(R/'source/scripts/e017_v94r/evaluation.py'))
S=P['S'];read=P['read'];sha=P['sha'];save=P['compact'];V=P['V93'];OUT=R/'production/confirmation'
def paired(a,b,K):
 groups=collections.defaultdict(list)
 for x,y in zip(a,b):
  assert x['task']==y['task']
  f=lambda z:int(z['records'][0]['metrics']['success']) if K==1 else int(any(r['metrics']['success'] for r in z['records'][:K]))
  groups[x['task']['goal']['protected_group']].append(f(x)-f(y))
 v=np.array([np.mean(z) for z in groups.values()]);rng=np.random.default_rng(19329);bs=v[rng.integers(len(v),size=(10000,len(v)))].mean(1)
 return dict(delta=float(v.mean()),CI95=np.quantile(bs,[.025,.975]).tolist(),independent_groups=len(v),goals=len(a),seed=19329,bootstrap_replicates=10000)
def checkpoint(e,update):
 name=f'milestone_{update:05}' if update in P['SPEC']['named_updates'] else f'development_best_{update:05}'
 meta=read(R/'production/checkpoints'/(name+'.json'));path=R/'production'/meta['relative'];assert sha(path)==meta['SHA256']
 state=torch.load(path,map_location='cpu',weights_only=False);assert P['state_digest'](state)==meta['state_digest'] and state['binding']==P['bindings']()
 e.actor.load_state_dict(state['actor'],strict=True);e.actor.eval().requires_grad_(False);del state
 return dict(path=str(path),SHA256=meta['SHA256'],update=update,actor_fingerprint=P['c']['state_digest'](e.actor.state_dict()))
def group(e,task,policy,label,i,count=8,seeds=None,state=None):
 path=OUT/f'{label}_{i:03}.json';fp=P['c']['state_digest'](e.actor.state_dict())
 if path.exists():
  obj=read(path);assert obj['task']==task and obj['actor_fingerprint']==fp;return obj
 seeds=seeds or [S['ns']('CANDIDATE',task['goal']['goal_id'],j) for j in range(count)]
 before=dict(e.calls);start=time.perf_counter();P['prepare_sampler']()
 records=S['generate'](P['K']['LiveActor'](e),task,policy,count,state=state,seeds=seeds)
 calls={k:e.calls[k]-before.get(k,0) for k in e.calls}
 obj=dict(task=task,records=records,actor_fingerprint=fp,seeds=seeds,policy=policy,seconds=time.perf_counter()-start,
  calls=calls,proposal_extension_checks=sum(len(r['proposals']) for r in records),terminal_evaluations=len(records),initial_prefix_length=state.t if state is not None else 0)
 save(path,obj);return obj
def main():
 done=read(R/'production/TRAINING_COMPLETED.json');assert done['update']==5000 and done['counters']['attempts']==80000
 assert not (R/'production/STOP_RECORD.json').exists()
 release=read(R/'qualification/RELEASE.json');assert release['PASS'] and release['binding']==P['bindings']()
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 OUT.mkdir(exist_ok=True);tasks=read(V/'manifests/PROSPECTIVE_FIXED.json')['tasks'];assert len(tasks)==32 and len({t['source_reference_identity'] for t in tasks})==16
 e,_=P['K']['load_engine']('C');e.actor.eval().requires_grad_(False)
 initfp=P['c']['state_digest'](e.actor.state_dict());assert P['c']['state_digest'](P['CPU'](e.actor.state_dict()))=='884b1e016135f4bf4b3594b6aefc81c1594a525186e09091ad5bd5c023781f80'
 panels={};metadata={};sens={};costs={}
 for policy,label in [('A','J1'),('B','parent_unfiltered'),('C','parent_corrected_filtered')]:
  panels[label]=[group(e,t,policy,label,i) for i,t in enumerate(tasks)]
 for u,label in [(done['best']['update'],'selected'),(5000,'final5000')]:
  metadata[label]=checkpoint(e,u)
  for policy,suffix in [('C','corrected_filtered'),('B','unfiltered')]:
   name=label+'_'+suffix
   alias='parent_'+suffix if metadata[label]['actor_fingerprint']==initfp else ('selected_'+suffix if label=='final5000' and metadata['selected']['actor_fingerprint']==metadata[label]['actor_fingerprint'] else None)
   if alias in panels:
    panels[name]=panels[alias];metadata[label]['panel_reuse_'+suffix]=alias
   else:panels[name]=[group(e,t,policy,name,i) for i,t in enumerate(tasks)]
 # Paired target interventions share sequence and candidate-indexed streams.
 for model,u in [('parent',0),('selected',done['best']['update'])]:
  checkpoint(e,u);sens[model]=[]
  for i,t in enumerate(tasks):
   seeds=[S['ns']('R2_PROSPECTIVE_SENSITIVITY',t['source_reference_identity'],j) for j in range(8)]
   sens[model].append(group(e,t,'C','SENSITIVITY_'+model,i,seeds=seeds))
 # Stronger frozen Bestof16 and bounded generated-prefix suffix search.
 checkpoint(e,0);best16=[];extras=[];search=[]
 for i,t in enumerate(tasks):
  extra=group(e,t,'C','PARENT_EXTRA8',i,seeds=[S['ns']('CANDIDATE',t['goal']['goal_id'],j) for j in range(8,16)])
  extras.append(extra)
  first=panels['parent_corrected_filtered'][i];best16.append(dict(task=t,records=first['records']+extra['records']))
  firstsearch=group(e,t,'C','SEARCH_INITIAL',i,count=1,seeds=[S['ns']('CANDIDATE',t['goal']['goal_id'],0)])
  row=firstsearch['records'][0];records=[row];suffix=None
  if not row['metrics']['success'] and len(row['sequence'])>=32:
   prefix=S['reconstruct'](row,64,32)
   suffix=group(e,t,'C','SEARCH_SUFFIX',i,count=15,state=prefix,seeds=[S['ns']('R2_SUFFIX_SEARCH',t['goal']['goal_id'],j) for j in range(15)])
   records+=suffix['records']
  search.append(dict(task=t,records=records,full_initial_cost=firstsearch['calls'],suffix_cost=suffix['calls'] if suffix else {},
   checks=firstsearch['proposal_extension_checks']+(suffix['proposal_extension_checks'] if suffix else 0),seconds=firstsearch['seconds']+(suffix['seconds'] if suffix else 0),
   provenance='fresh first complete-generation attempt; if not success rewind generated prefix32 and draw15 suffixes; original terminal evaluator; no native coordinates'))
 save(R/'FROZEN_SEARCH_RESULTS.json',dict(records=search))
 summaries={k:E['summarize'](v) for k,v in panels.items()};summaries['parent_Bestof16']=dict(E['summarize'](best16),K16=sum(any(r['metrics']['success'] for r in g['records']) for g in best16)/32,terminal_evaluations=512)
 sensitivity={}
 for model,groups in sens.items():
  byid=collections.defaultdict(dict)
  for g in groups:byid[g['task']['source_reference_identity']][g['task']['goal']['difficulty']]=g
  diffs=[]
  for identity,pair in byid.items():
   a,b=pair['witness'],pair['relaxed'];rows=[]
   for x,y in zip(a['records'],b['records']):
    if x['metrics']['complete'] and y['metrics']['complete']:
     rows.append(dict(exposure_delta=y['metrics']['exposed_mean']-x['metrics']['exposed_mean'],burial_delta=y['metrics']['buried_mean']-x['metrics']['buried_mean'],
      geometry_changed=x['geometry']!=y['geometry']))
   diffs.append(dict(identity=identity,paired_complete=len(rows),mean_exposure_delta=float(np.mean([x['exposure_delta'] for x in rows])) if rows else None,
    mean_burial_delta=float(np.mean([x['burial_delta'] for x in rows])) if rows else None,geometry_changed=sum(x['geometry_changed'] for x in rows),
    witness_target=a['task']['goal'],relaxed_target=b['task']['goal']))
  sensitivity[model]=diffs
 a=summaries['selected_corrected_filtered'];b=summaries['parent_corrected_filtered'];K1=paired(panels['selected_corrected_filtered'],panels['parent_corrected_filtered'],1);K8=paired(panels['selected_corrected_filtered'],panels['parent_corrected_filtered'],8)
 safe=a['hard_valid']>=b['hard_valid']-.05 and a['backbone_valid']>=b['backbone_valid']-.05 and a['geometry_KL_per_active']<=1 and a['structural_pair_RMSD']>=.8*b['structural_pair_RMSD']
 GO=K1['delta']>=.05 and K1['CI95'][0]>0 and K8['delta']>=0 and safe
 for label,groups in panels.items():
  costs[label]=dict(seconds=sum(g['seconds'] for g in groups),proposal_extension_checks=sum(g['proposal_extension_checks'] for g in groups),terminal_evaluations=sum(g['terminal_evaluations'] for g in groups),
   calls={k:sum(g['calls'].get(k,0) for g in groups) for k in {k for g in groups for k in g['calls']}})
 groups=panels['parent_corrected_filtered']+extras
 costs['parent_Bestof16']=dict(seconds=sum(g['seconds'] for g in groups),proposal_extension_checks=sum(g['proposal_extension_checks'] for g in groups),terminal_evaluations=512,
  calls={k:sum(g['calls'].get(k,0) for g in groups) for k in {k for g in groups for k in g['calls']}})
 result=dict(outcome='RL_LEARNING_GO' if GO else 'RL_PERFORMANCE_NO_GO',updates=5000,training_attempts=80000,
  completed_training_trajectories=done['counters']['complete'],aborted_training_attempts=done['counters']['aborts'],
  tasks=32,independent_identities=16,checkpoint_metadata=metadata,summaries=summaries,K1_paired=K1,K8_paired=K8,
  safeguards_pass=safe,goal_sensitivity=sensitivity,costs=costs,
  search=dict(goals_reached=sum(any(r['metrics']['success'] for r in g['records']) for g in search),goals=32,terminal_evaluations=sum(len(g['records']) for g in search),proposal_checks=sum(g['checks'] for g in search),seconds=sum(g['seconds'] for g in search)),
  primary_comparison='selected trained actor vs original actor under IDENTICAL corrected controller',fixed_AA=True,no_native_target_coordinates=True,TEST=False,
  interpretation='coarse regional-property C-alpha control; not amino-acid design, protein foldability or biochemical function',
  goal_sensitivity_interpretation='matched computational target interventions; different utility thresholds can change success without changed geometry; inspect achieved-property/geometry deltas separately')
 e.prior.verify();assert P['c']['state_digest'](e.parent.state_dict())==e.fp
 save(R/'FINAL_PROSPECTIVE_COMPARISON.json',result);print('R2_FINAL_EVALUATION_COMPLETE',result['outcome'],flush=True)
if __name__=='__main__':main()
