"""Final-only CPU analysis for E017 V7.1; no model fitting or GPU inference."""
import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import torch

R=Path('/mnt/d/Simone/proteinGen_data/e017_v71_expected_return_calibration_20261023_v1');V7=R.parent/'e017_v7_amortized_lowlevel_20261023_v1'
st=runpy.run_path(str(R/'source/scripts/e017_v71/study.py'));read=st['read'];save=st['save'];sha=st['sha'];state_digest=st['state_digest'];INPUTS={}
def load(p):INPUTS[str(p)]=sha(p);return read(p)
def lines(p):
 INPUTS[str(p)]=sha(p);return [json.loads(x) for x in Path(p).read_text().splitlines()]
def text(n,value):
 p=R/n;q=p.with_name(p.name+'.part')
 with q.open('w') as f:f.write(value+'\n');f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(str(x) for x in row)+' |' for row in rows])
def f(x):return 'NA' if x is None else f'{x:.5f}'
def boot(values,groups):
 values=np.asarray(values,dtype=float);groups=np.asarray(groups);ids=sorted(set(groups));sums=np.array([values[groups==g].sum() for g in ids]);counts=np.array([(groups==g).sum() for g in ids]);draw=np.random.default_rng(1787171).integers(0,len(ids),(10000,len(ids)));v=sums[draw].sum(1)/counts[draw].sum(1)
 return dict(estimate=float(values.mean()),CI95=np.quantile(v,[.025,.975]).tolist(),groups=len(ids),observations=len(values),replicates=10000,seed=1787171)
def ranks(x):
 x=np.asarray(x);order=np.argsort(x,kind='stable');result=np.empty(len(x));i=0
 while i<len(x):
  j=i+1
  while j<len(x) and x[order[j]]==x[order[i]]:j+=1
  result[order[i:j]]=(i+j-1)/2;i=j
 return result
def corr(x,y):
 return float(np.corrcoef(x,y)[0,1]) if len(x)>1 and np.std(x)>0 and np.std(y)>0 else None
def auc_ap(pred,labels):
 p=np.asarray(pred);y=np.asarray(labels,dtype=bool);P=int(y.sum());N=len(y)-P
 if P==0 or N==0:return dict(AUROC=None,AUPRC=None,positive=P,negative=N,reason='single-class subset')
 auc=(ranks(p)[y].sum()-P*(P-1)/2)/(P*N);order=np.argsort(-p,kind='stable');sortedp=p[order];cum=np.cumsum(y[order]);end=np.r_[np.flatnonzero(sortedp[1:]!=sortedp[:-1]),len(p)-1];rec=cum[end]/P;prec=cum[end]/(end+1);ap=float(((rec-np.r_[0,rec[:-1]])*prec).sum())
 return dict(AUROC=float(auc),AUPRC=ap,positive=P,negative=N,meaning='raw shaped-Q discriminative ranking against binary success; not probability calibration')
def detail(p,y,groups):
 p=np.asarray(p);y=np.asarray(y);error=p-y;bins=[]
 for ii in np.array_split(np.argsort(p,kind='stable'),5):
  if len(ii):bins.append(dict(n=len(ii),predicted_mean=float(p[ii].mean()),reference_mean=float(y[ii].mean()),bias=float(error[ii].mean()),group_count=len(set(np.array(groups)[ii]))))
 return dict(n=len(y),groups=len(set(groups)),MAE=float(abs(error).mean()),RMSE=float(np.sqrt(np.mean(error**2))),MSE=float(np.mean(error**2)),mean_bias=float(error.mean()),bias_CI=boot(error,groups),Spearman=corr(ranks(p),ranks(y)),calibration_quantiles=bins,low_return_bias=float(error[y<.05].mean()) if (y<.05).any() else None,low_return_n=int((y<.05).sum()),high_return_bias=float(error[y>=.5].mean()) if (y>=.5).any() else None,high_return_n=int((y>=.5).sum()),pred_min=float(p.min()),pred_max=float(p.max()))
def rank_summary(pred,rows):
 q=st['rank_metrics'](pred,rows);rr=q['records'];q['accuracy_CI']=boot([r['correct'] for r in rr],[r['group'] for r in rr]) if rr else None;q['by_stage']={}
 for k in range(4):
  subset=[r for r in (q['all_nontied_pairs'] if k==3 else rr) if r['stage']==k];q['by_stage'][str(k)]=dict(pairs=len(subset),groups=len({r['group'] for r in subset}),accuracy_CI=boot([r['correct'] for r in subset],[r['group'] for r in subset]) if subset else None,eligibility='all exact non-tied terminal pairs' if k==3 else 'predeclared supported nonterminal contrasts')
 return q

def main():
 torch.set_num_threads(1);protocol=load(R/'PROTOCOL.json');cfg=protocol['config'];splits=load(R/'manifests/SPLITS.json');ret=load(R/'RETURN_AUDIT.json');final=load(R/'CRITIC_FINAL.json');rows=lines(R/'results/expected_rows.jsonl');bykey={r['key']:r for r in rows};assert len(bykey)==640
 simple=load(R/'SIMPLE_FIT.json');metrics={};decisions={};comparisons={}
 raw=np.load(R/'results/realized.npz');FIT=raw['split']=='FIT';coef=[]
 for k in range(4):
  ii=FIT&(raw['stage']==k);J=np.exp(-.5*(raw['X'][ii,13]**2+(raw['X'][ii,14]*180/35)**2));design=np.c_[np.ones(len(J)),J];coef.append(np.linalg.lstsq(design,raw['y'][ii],rcond=None)[0])
 def immediate_proxy(features,stages):
  X=np.asarray(features);J=np.exp(-.5*(X[:,13]**2+(X[:,14]*180/35)**2));return np.array([coef[int(k)][0]+coef[int(k)][1]*j for k,j in zip(stages,J)])
 for split,part in final['partitions'].items():
  rr=[bykey[k] for k in part['rows']];stage=np.array([r['stage'] for r in rr]);y=np.array([r['mean_return'] for r in rr]);groups=np.array([r['group'] for r in rr]);pred={**part['predictions']['simple'],**{k:v for k,v in part['predictions'].items() if k!='simple'}};pred['immediate_proxy_posthoc']=immediate_proxy([r['features'] for r in rr],stage);metrics[split]={}
  for name,pp in pred.items():
   pp=np.asarray(pp);metrics[split][name]={}
   for k in [0,1,2,3,'nonterminal','all']:
    mask=np.ones(len(rr),dtype=bool) if k=='all' else stage<3 if k=='nonterminal' else stage==k
    metrics[split][name][str(k)]=detail(pp[mask],y[mask],groups[mask])
   metrics[split][name]['action_ranking']=rank_summary(pp,rr)
   metrics[split][name]['distance_strata']=[]
   dist=np.array([r['features'][8]*8 for r in rr])
   for lo,hi in [(0,4),(4,8),(8,12),(12,float('inf'))]:
    ii=(dist>=lo)&(dist<hi)
    if ii.any():metrics[split][name]['distance_strata'].append(dict(lower_A=lo,upper_A=hi if math.isfinite(hi) else None,n=int(ii.sum()),RMSE=float(np.sqrt(np.mean((pp[ii]-y[ii])**2))),bias=float((pp[ii]-y[ii]).mean())))
  nonterminal=stage<3;bestname=min(['constant','ridge','MLP'],key=lambda n:final['partitions']['DEV']['scores'][n]['nonterminal']['MSE']);bpred=np.array(pred[bestname]);comparisons[split]={}
  for name in ['critic1','critic2']:
   comparisons[split][name]=boot((np.asarray(pred[name])[nonterminal]-y[nonterminal])**2-(bpred[nonterminal]-y[nonterminal])**2,groups[nonterminal])
  p1,p2=np.array(pred['critic1']),np.array(pred['critic2']);metrics[split]['twin_disagreement']=dict(mean_absolute=float(abs(p1-p2).mean()),correlation_with_mean_absolute_error=corr(abs(p1-p2),abs((p1+p2)/2-y)),mean_reference_MC_SE=float(np.mean([r['return_MC_SE'] for r in rr])))
  metrics[split]['target_tracking']={name:dict(mean_absolute_lag=float(abs(np.array(pred[name])-np.array(pred['target'+str(i+1)])).mean()),RMSE_lag=float(np.sqrt(np.mean((np.array(pred[name])-np.array(pred['target'+str(i+1)]))**2)))) for i,name in enumerate(['critic1','critic2'])}
 for name in ['critic1','critic2']:
  reasons=[]
  for split in ['CONFIRM','VALIDATION']:
   m=metrics[split][name];bname=min(['constant','ridge','MLP'],key=lambda n:metrics['DEV'][n]['nonterminal']['MSE']);b=metrics[split][bname]
   if m['nonterminal']['RMSE']>.9*b['nonterminal']['RMSE']:reasons.append(split+': <10% RMSE advantage vs DEV-selected simple predictor')
   if comparisons[split][name]['CI95'][1]>=0:reasons.append(split+': paired MSE advantage not supported')
   if abs(m['nonterminal']['mean_bias'])>.1:reasons.append(split+': bias exceeds0.1')
   rank=m['action_ranking'];ci=rank['accuracy_CI']
   if rank['supported_pairs']<10 or rank['supported_groups']<8:reasons.append(split+': insufficient supported action contrasts')
   if ci is None or ci['estimate']<.6 or ci['CI95'][0]<=.5:reasons.append(split+': action ranking not established')
   if any(m[str(k)]['MSE']>1.1*metrics[split]['constant'][str(k)]['MSE'] for k in [0,1,2,3]):reasons.append(split+': stage error exceeds constant allowance')
  decisions[name]=dict(GO=not reasons,reasons=reasons)
 realized=load(R/'REALIZED_CRITIC_CONFIRMATION.json');data=np.load(R/'results/realized.npz');ei=realized['rows'];ep=st['episodes']();rawgroups=np.array([ep[int(data['index'][i][0])]['group'] for i in ei]);ry=data['y'][ei];rs=data['stage'][ei];binary=data['success'][ei];realized_metrics={}
 for label,pp in realized['predictions'].items():
  pp=np.array(pp);realized_metrics[label]={}
  for j in range(2):
   realized_metrics[label][f'critic{j+1}']={}
   for k in range(4):
    ii=rs==k;realized_metrics[label][f'critic{j+1}'][str(k)]={**detail(pp[ii,j],ry[ii],rawgroups[ii]),'binary_discrimination':auc_ap(pp[ii,j],binary[ii])}
 baserows=[dict(features=data['X'][i].tolist(),stage=int(data['stage'][i])) for i in ei];basepred=st['simple_predict'](baserows);realized_metrics['SIMPLE']={}
 for name,pred in basepred.items():
  realized_metrics['SIMPLE'][name]={}
  for k in range(4):
   ii=rs==k;realized_metrics['SIMPLE'][name][str(k)]={**detail(pred[ii],ry[ii],rawgroups[ii]),'binary_discrimination':auc_ap(pred[ii],binary[ii])}
 rwr=rwr_analysis();manifest=verify_checkpoints(protocol);resource=resources(final,manifest);save('CRITIC_METRICS.json',dict(expected=metrics,paired_MSE_vs_simple=comparisons,realized=realized_metrics,decisions=decisions,immediate_proxy_posthoc=dict(coefficients=np.asarray(coef).tolist(),description='FIT-only intercept/slope on immediate post-action J for each stage; exploratory, not used for selection/promotion'),confidence='10k paired protein-cluster bootstrap; expected means16suffix draws, MCSE and supported-pair coverage separate; not exactQ; no nested suffix-resampling interval'))
 census={}
 for part in ['FIT','DEV','CONFIRM','VALIDATION','VALIDATION_SHIFTED']:
  census[part]={}
  for k in range(4):
   rr=[r for r in rows if r['split']==part and r['stage']==k];census[part][str(k)]=dict(state_action_labels=len(rr),identities=len({r['identity'] for r in rr}),suffix_trials=sum(r['samples'] for r in rr),successful_trials=sum(sum(r['success_labels']) for r in rr),labels_with_observed_success=sum(any(r['success_labels']) for r in rr),mean_MC_SE=float(np.mean([r['return_MC_SE'] for r in rr])),mean_within_action_realized_variance=float(np.mean([np.var(r['returns'],ddof=1) if len(r['returns'])>1 else 0 for r in rr])),mean_task_minus_soft_return=float(np.mean([r['mean_return']-r['mean_soft_return'] for r in rr])))
 save('EXPECTED_LABEL_CENSUS.json',census);save('RWR_METRICS.json',rwr);save('CHECKPOINT_AND_RECOVERY_MANIFEST.json',manifest);save('RESOURCE_AND_STORAGE_REPORT.json',resource)
 publish(protocol,ret,simple,final,rows,metrics,comparisons,realized_metrics,decisions,rwr,manifest,resource)

def rwr_analysis():
 likelihood=load(R/'RWR_FIXED_TRAIN_LIKELIHOODS.json');rr=likelihood['records'];results={}
 for label in ['B25','B250']:
  diff=np.array([r[label+'_logprob']-r['parent_logprob'] for r in rr]);groups=np.array([r['group'] for r in rr]);wins=np.array([r['success'] for r in rr]);invalid=np.array([not r['valid'] for r in rr]);task=collections.defaultdict(list)
  for r in rr:task[r['task_id']].append(r)
  importance=[]
  for tid,rows in task.items():
   lr=np.array([r[label+'_logprob']-r['parent_logprob'] for r in rows]);w=np.exp(lr);s=np.array([r['success'] for r in rows]);inv=np.array([not r['valid'] for r in rows]);importance.append(dict(task_id=tid,group=rows[0]['group'],ESS=float(w.sum()**2/(w*w).sum()),success_SNIS=float((w*s).sum()/w.sum()),invalid_SNIS=float((w*inv).sum()/w.sum()),behavior_success=float(s.mean()),behavior_invalid=float(inv.mean()),max_log_ratio=float(lr.max())))
  results[label]=dict(all_logdensity_change=boot(diff,groups),successful_logdensity_change=boot(diff[wins],groups[wins]) if wins.any() else None,invalid_logdensity_change=boot(diff[invalid],groups[invalid]) if invalid.any() else None,importance_tasks=importance,ESS_mean=float(np.mean([r['ESS'] for r in importance])),success_SNIS_change=boot([r['success_SNIS']-r['behavior_success'] for r in importance],[r['group'] for r in importance]),invalid_SNIS_change=boot([r['invalid_SNIS']-r['behavior_invalid'] for r in importance],[r['group'] for r in importance]),interpretation='logdensity is not event probability;8draw/task SNIS is biased finite-sample diagnostic, no clipping, ESS reported')
 allpanels={};paired={}
 for panel in ['CONFIRM','CONFIRM_SHIFTED']:
  A=load(V7/f'results/A_{panel}_K32.json')['records'];B25=load(V7/f'results/B_BEST25_{panel}_K32.json')['records'];B250=load(R/f'results/B250_{panel}.json')['records'];allpanels[panel]={}
  for label,records in [('parent',A),('B25',B25),('B250',B250)]:
   pool=collections.defaultdict(list)
   for r in records:pool[r['task_id']].append(r)
   tableK={}
   for K in [1,2,4,8,16,32]:
    success=[];groups=[];positions=[];orientations=[];selected=[];full=[];div=[]
    for tid,p in sorted(pool.items()):
     p=sorted(p,key=lambda r:r['candidate_index'])[:K];valid=[r for r in p if r['valid']];chosen=p[0] if K==1 else min(valid,key=lambda r:(max(r['position_A']/8,r['orientation_deg']/35),r['candidate_index'])) if valid else None
     success.append(bool(chosen and chosen['success']));groups.append(p[0]['group']);selected.append(chosen);full.extend(p);xx=np.array([r['tail'] for r in p]);ix=np.triu_indices(K,1)
     if K>1:div.extend(np.sqrt(((xx[:,None]-xx[None,:])**2).sum(-1).mean(-1))[ix].tolist())
    present=[r for r in selected if r];first=[min(p,key=lambda r:r['candidate_index']) for p in pool.values()]
    tableK[str(K)]=dict(success=sum(success),tasks=len(pool),success_interval=boot(success,groups),selected_valid=sum(r['valid'] for r in present),abstentions=sum(r is None for r in selected),candidate_valid_fraction=float(np.mean([r['valid'] for r in full])),candidate_invalid_mass_estimate=boot([not r['valid'] for r in full],[r['group'] for r in full]),mean_position_A=float(np.mean([r['position_A'] for r in present])),mean_orientation_deg=float(np.mean([r['orientation_deg'] for r in present])),mean_candidate_KL=float(np.mean([r['KL'] for r in full])),tail_pair_RMSD_A=float(np.mean(div)) if div else None,task_success=success,task_groups=groups)
   allpanels[panel][label]=tableK
  for label in ['B25','B250']:
   paired[panel+'_'+label]={K:boot(np.array(allpanels[panel][label][K]['task_success'],float)-np.array(allpanels[panel]['parent'][K]['task_success'],float),allpanels[panel]['parent'][K]['task_groups']) for K in ['1','2','4','8','16','32']}
 goal250=load(R/'RWR_GOAL_SENSITIVITY.json');prior=load(V7/'CAUSAL_DIAGNOSTICS.json');goal={}
 for label,rr in [('parent',[r for r in prior['goal_response'] if r['arm']=='A']),('B25',[r for r in prior['goal_response'] if r['arm']=='B']),('B250',goal250['records'])]:goal[label]=dict(correct=sum(r['correct'] for r in rr),pairs=len(rr),rate=boot([r['correct'] for r in rr],[r['group'] for r in rr]),own_minus_cross_A=float(np.mean([r['own_minus_cross_A'] for r in rr])))
 base={(r['identity'],r['seed']):r for r in prior['goal_response'] if r['arm']=='A'};rr=goal250['records'];goal['B250_minus_parent']=boot([float(r['correct'])-float(base[(r['identity'],r['seed'])]['correct']) for r in rr],[r['group'] for r in rr])
 return dict(likelihood=results,panels=allpanels,paired_success=paired,goal_sensitivity=goal,scope='fixed actor checkpoints only; V7exposed VALIDATION now diagnostic, not independent confirmation; old scientific stops remain unchanged')

def verify_checkpoints(protocol):
 index=load(R/'checkpoints/INDEX.json');out={}
 for role,rec in index.items():
  if rec is None:out[role]=None;continue
  assert sha(rec['path'])==rec['sha256'];s=torch.load(rec['path'],map_location='cpu',weights_only=False);assert state_digest(s)==rec['state_digest'];assert s['actor_optimizer'] is None and s['parent_actor_fingerprint']==state_digest(s['actor'])
  steps=[sorted({int(v['step']) for v in opt['state'].values()}) for opt in s['optimizers']];assert all(x==[s['update']] for x in steps)
  out[role]=dict(record=rec,SHA256_verified=True,full_semantic_digest_verified=True,critic_optimizer_steps=steps,sampler_cursor=s['sampler_cursor'],rng_keys=list(s['rng']),actor_fingerprint=s['parent_actor_fingerprint'],J1_tensor_SHA=s['J1_tensor_SHA'],actor_updates=0,critic_parameters_each=[sum(t.numel() for t in x.values()) for x in s['critics']],source=s['source']);del s
 return dict(checkpoints=out,recovery_test=load(R/'RECOVERY_TEST.json'),source=protocol['source'],resume='scientific stop blocks further training; full-state restore tested same process, no fresh-process claim',partial_checkpoints=[str(p) for p in (R/'checkpoints').glob('*.part')])

def resources(final,manifest):
 expected=load(R/'EXPECTED_COMPLETED.json');rwr=load(R/'RWR_MONITOR_COMPLETED.json');simple=load(R/'SIMPLE_FIT.json');data=np.load(R/'results/realized.npz');rows=lines(R/'results/critic_updates.jsonl');free=__import__('shutil').disk_usage(R).free
 watch=lines(R/'results/resource_watch.jsonl') if (R/'results/resource_watch.jsonl').exists() else []
 telemetry=dict(samples=len(watch),minimum_D_free_bytes=min(r['D_free_bytes'] for r in watch) if watch else None,minimum_available_RAM_bytes=min(r['available_RAM_bytes'] for r in watch) if watch else None,maximum_swap_used_bytes=max(r['swap_used_bytes'] for r in watch) if watch else None,first_UTC=watch[0]['UTC'] if watch else None,last_UTC=watch[-1]['UTC'] if watch else None,scope='watch started during suffix collection; peak GPU is separately measured per worker')
 gradients=np.array([row['gradient_norms'] for row in rows]);telemetry['critic_optimization']=dict(combined_twin_MSE_mean=float(np.mean([row['loss'] for row in rows])),combined_twin_MSE_first25=float(np.mean([row['loss'] for row in rows[:25]])),combined_twin_MSE_last25=float(np.mean([row['loss'] for row in rows[-25:]])),preclip_gradient_mean=gradients.mean(0).tolist(),preclip_gradient_max=gradients.max(0).tolist(),clip_counts=(gradients>5).sum(0).tolist(),finite=bool(np.isfinite(gradients).all()),distinct_fitting_tasks=len({a['task_id'] for row in rows for a in row['used']}),transitions_per_twin=len(rows)*4,replay_transition_pool=256*8*4,sampling='stage-balanced cyclic identities, seeded episode/expected-label choice; presentations are not unique trajectories')
 return dict(D_free_before_study_bytes=6009012224,D_free_current_bytes=free,required_operational_headroom_bytes=2*2**30,admitted=free>2*2**30+64*2**20,complete_checkpoint_sizes={k:v['record']['bytes'] if v else None for k,v in manifest['checkpoints'].items()},critic_updates_new=final['new_updates'],critic_updates_historical=250,critic_updates_including_historical=final['new_updates']+250,critic_stage_exposures=dict(collections.Counter(str(a['stage']) for row in rows for a in row['used'])),target_kind_exposures=dict(collections.Counter(a['target_kind'] for row in rows for a in row['used'])),expected_collection=expected,critic_training=dict(elapsed_seconds=final['elapsed_seconds'],network_calls=final['network_calls'],peak_GPU=final['peak_GPU']),RWR_monitor=rwr,simple_model=simple,telemetry=telemetry,actor_updates=0,SAC_actor_updates=0,deleted_historical_artifacts=0,raw_realized_rows=len(data['y']),all_new_artifacts_on_D=True)

def publish(protocol,ret,simple,final,rows,metrics,comparisons,realized,decisions,rwr,manifest,resource):
 go=all(v['GO'] for v in decisions.values());decision='GO to propose a separately bounded low-level SAC pilot, subject to soft-target alignment' if go else 'NO-GO for SAC actor updates'
 selected=final['selected'];best_simple=min(['constant','ridge','MLP'],key=lambda n:metrics['DEV'][n]['nonterminal']['MSE']);cfg=protocol['config']
 critical=table(['Critic','Decision','Failed conditions'],[[name,'GO' if d['GO'] else 'NO-GO','; '.join(d['reasons']) or 'all frozen conditions pass'] for name,d in decisions.items()])
 stage_rows=[]
 for part in ['DEV','CONFIRM','VALIDATION','VALIDATION_SHIFTED']:
  for name in ['constant','ridge','MLP','critic1','critic2']:
   for k in ['0','1','2','3']:
    m=metrics[part][name][k];stage_rows.append([part,name,k,m['n'],m['groups'],f(m['MAE']),f(m['RMSE']),f(m['mean_bias']),f(m['Spearman'])])
 stages=table(['Partition','Predictor','Stage','State/action labels','Proteins','MAE','RMSE','Bias','Spearman'],stage_rows)
 compare=table(['Partition','Predictor','Nonterminal RMSE','Bias','Supported action pairs/groups','Ranking accuracy [95%]'],[
  [part,name,f(m['nonterminal']['RMSE']),f(m['nonterminal']['mean_bias']),f"{m['action_ranking']['supported_pairs']}/{m['action_ranking']['supported_groups']}",f(m['action_ranking']['accuracy'])+' '+str(m['action_ranking']['accuracy_CI']['CI95'] if m['action_ranking']['accuracy_CI'] else 'insufficient contrasts')]
  for part in ['DEV','CONFIRM','VALIDATION','VALIDATION_SHIFTED'] for name in ['constant','ridge','MLP','critic1','critic2'] for m in [metrics[part][name]]])
 cirows=table(['Partition','Critic','MSE difference versus '+best_simple,'95% paired identity interval'],[[part,name,f(v['estimate']),str(v['CI95'])] for part,d in comparisons.items() for name,v in d.items()])
 rwrtable=table(['Panel / 64 tasks','Frozen checkpoint','K1','K2','K4','K8','K16','K32','All-32 invalid fraction','All-32 conditional KL'],[
  [part,name,*[m[str(k)]['success'] for k in [1,2,4,8,16,32]],f(m['32']['candidate_invalid_mass_estimate']['estimate']),f(m['32']['mean_candidate_KL'])] for part,d in rwr['panels'].items() for name,m in d.items()])
 likelihood=table(['Checkpoint','Successful TRAIN log-density change','95% identity interval','Mean importance ESS / 8','SNIS success change','SNIS invalid change'],[
  [name,f(d['successful_logdensity_change']['estimate']) if d['successful_logdensity_change'] else 'NA',str(d['successful_logdensity_change']['CI95']) if d['successful_logdensity_change'] else 'NA',f(d['ESS_mean']),f(d['success_SNIS_change']['estimate']),f(d['invalid_SNIS_change']['estimate'])] for name,d in rwr['likelihood'].items()])
 curve=table(['New critic updates','Total including V7','Twin 1 DEV expected MSE','Twin 2 DEV expected MSE','Best simple DEV expected MSE'],[[r['update'],r['update']+250,f(r['scores'][0]['nonterminal']['MSE']),f(r['scores'][1]['nonterminal']['MSE']),f(r['best_simple_MSE'])] for r in final['curves']])
 stop=load(R/'CRITIC_STOP.json') if (R/'CRITIC_STOP.json').exists() else None
 hardest={name:max(range(3),key=lambda k:metrics['CONFIRM'][name][str(k)]['RMSE']) for name in ['critic1','critic2']}
 evidence=f"TRAIN-confirmation nonterminal RMSE: stage constant {metrics['CONFIRM']['constant']['nonterminal']['RMSE']:.4f}, MLP {metrics['CONFIRM']['MLP']['nonterminal']['RMSE']:.4f}, twins {metrics['CONFIRM']['critic1']['nonterminal']['RMSE']:.4f}/{metrics['CONFIRM']['critic2']['nonterminal']['RMSE']:.4f}. The most difficult nonterminal stage by RMSE is {hardest['critic1']} for twin 1 and {hardest['critic2']} for twin 2."
 ranking_evidence=' '.join(f"{part}: {metrics[part]['critic1']['action_ranking']['supported_pairs']} supported nonterminal action pairs across {metrics[part]['critic1']['action_ranking']['supported_groups']} proteins; twin ranking accuracies {f(metrics[part]['critic1']['action_ranking']['accuracy'])}/{f(metrics[part]['critic2']['action_ranking']['accuracy'])}." for part in ['CONFIRM','VALIDATION'])
 target_table=table(['Partition','Twin1 mean absolute EMA lag','Twin2 mean absolute EMA lag','Twin mean disagreement'],[[part,f(metrics[part]['target_tracking']['critic1']['mean_absolute_lag']),f(metrics[part]['target_tracking']['critic2']['mean_absolute_lag']),f(metrics[part]['twin_disagreement']['mean_absolute'])] for part in ['DEV','CONFIRM','VALIDATION','VALIDATION_SHIFTED']])
 rwr_evidence=' '.join(f"{part}: parent/+25/+250 K1 successes are "+'/'.join(str(rwr['panels'][part][name]['1']['success']) for name in ['parent','B25','B250'])+' out of 64; K32 '+ '/'.join(str(rwr['panels'][part][name]['32']['success']) for name in ['parent','B25','B250'])+'. +250 minus parent K1 difference '+f(rwr['paired_success'][part+'_B250']['1']['estimate'])+' (95% paired protein interval '+str(rwr['paired_success'][part+'_B250']['1']['CI95'])+').' for part in ['CONFIRM','CONFIRM_SHIFTED'])
 audit=f'''# Critic target and return audit — E017 V7.1 v1

V7 remains a failed admission experiment. Both critics passed its terminal-holdout threshold (MSE 0.103986 and 0.092914, limit 0.108925), but failed its first-action TRAIN-probe requirement (MSE 0.092002 and 0.091549, limit 0.053628). Its 1,536 fitting transitions covered all four stages equally. Taking stride 12 over that flattened four-stage table selected only stage 0 for the 128-example fit probe. No V7 threshold or result was revised.

The network already has goal-relative pose, remaining/8 and remaining/H in its 64-feature context, global progress, current fixed-AA conditioning and a seven-channel transformed action representation. Thus missing horizon information is not established; no redundant stage variant or larger critic was introduced. The encoder describes the current prefix independently of the candidate action; the action enters its value head. This may make learning forward kinematics less direct than the simple post-action residual baseline, but the measured comparisons, not that hypothesis, determine promotion.

The actual V7 calibration optimized squared error to gamma^(3-t) times the realized terminal task reward. It did not execute SAC actor updates or soft Bellman learning. Terminal rewards are valid*(success+0.25*J), J=exp(-0.5*((position/8)^2+(orientation/35)^2)); earlier rewards are zero, gamma=0.995. Stage return factors are gamma^3, gamma^2, gamma, 1. The full success indicator, shaped component, hard validity and conditional geometry KL are distinct quantities.

A single Monte Carlo return is a noisy target for expected Q at a nonterminal state/action, not its exact value. V7's requirement to explain half of first-action realized-return variance is not by itself a calibrated-value test: unknown stochastic suffixes can impose irreducible variance. Its changing-identity RWR NLL windows also were not a paired learning-progress estimate. V7.1 uses a new baseline/ranking protocol instead of retroactively calling those original gates passes.

The unexecuted V7 SAC target would have used the minimum target critic, zero bootstrap at terminal transitions, and discounted future entropy/KL costs. We record a separate soft-reference return: task return minus the sum over subsequent decisions j>t of gamma^(j-t)*(0.001*logpi_j/3+0.1*conditional_KL_j). The fixed current action's entropy/KL is excluded from Q(s,a). Lambda 0.1 is a declared diagnostic constant, not an estimate for an adapting dual. The newly fitted critics predict task Q only; comparing them directly with soft returns would be a target mismatch. A future SAC pilot must explicitly align or relearn that soft target.

Masks and supports are unchanged: positive distance, angle in (0,pi), signed torsion in (-pi,pi), with transformed Gaussian densities and their Jacobians. Native AA identities are supplied sequentially; unrevealed suffix AAs are absent from features. CPU reconstruction maximum error is {ret['max_coordinate_reconstruction_error_A']:.3g} A. Perturbing hidden future coordinates and unrevealed AAs leaves features unchanged. The simple features include only the current prefix/action and deterministic one-step forward kinematics, never sampled future coordinates.

Warm-start critics are the two original 12,839,497-parameter networks from V7's stopped C checkpoint. Its 48 old fitting identities are restricted to the new FIT partition; its 32 diagnostic identities do not enter the new CONFIRM partition. Fresh AdamW moments are used for this expanded regime. Targets track an EMA (tau=0.005) for stability diagnostics; no bootstrapped target is used in this supervised fixed-policy calibration. Both critic and target state are recovered and verified; neither actor nor J1 has an optimizer.

Task Q is shaped and can exceed 1. It is not a success probability. AUROC/AUPRC use Q as a ranking score against binary complete success; no calibrated reachability classifier or reachability network was trained.
'''
 text('CRITIC_TARGET_AND_RETURN_AUDIT.md',audit)
 text('STAGE_STRATIFIED_CRITIC_CALIBRATION.md','\n\n'.join([
 '# Stage-stratified expected-return calibration — v1',
 'Fitting/development/confirmation subsets contain 256/64/64 distinct TRAIN identities (8 original trajectories each). Another 128 TRAIN identities are unused for critic fitting/selection. Lower-noise references use 32/16/16 identities from these subsets, plus 8 existing protected VALIDATION identities evaluated under native and generated-prefix conditions. Native/shifted conditions share proteins: 80 task conditions represent 72 distinct proteins, not 80 independent biological examples.',
 'For each condition and stage, two actions are drawn at the same observed state. Each nonterminal action receives 16 independent suffix streams under the unchanged parent policy; alternatives use common random numbers for paired comparisons. Terminal state/actions are evaluated once exactly. This produces 640 fixed state/action labels and 7,840 completed suffix evaluations. Prefix states use the candidate-0 trajectory of the frozen parent; terminal and suffix failures are retained without replacement.',
 'The references are finite Monte Carlo means, not exact Q. Per-label standard errors, all suffix rewards, policy hashes and seeds are retained. EXPECTED_LABEL_CENSUS.json reports stage/partition trial counts, positives, label coverage, within-action realized variance and task-versus-soft return differences. With zero successes in 16 draws, zero estimated sample variance does not establish zero population success probability. Confidence intervals resample proteins and condition on the collected references; they do not include a separate nested suffix resampling analysis.',
 curve,f'Selected checkpoint: `{selected["path"]}`, SHA256 `{selected["sha256"]}`. New updates: {final["new_updates"]}; historical V7 updates: 250. Stop record: `{json.dumps(stop)}`. Original V7 failure is unchanged.',stages,
 'MAE/RMSE/bias and quantile calibration are reported by stage. Stages 0–2 are the primary future-return comparison; stage 3 is exact immediate terminal scoring and cannot establish useful long-horizon value prediction. Ranking uses action alternatives at the identical state. A supported nonterminal contrast requires absolute empirical return difference >=0.1 and greater than 1.96 times its paired Monte Carlo standard error. Coverage is reported explicitly; unsupported contrasts are not evidence of bad or good ranking.',compare,cirows,
 'The 2,048 realized-return confirmation transitions comprise all four stages from eight trajectories for 64 held-out TRAIN proteins. Raw shaped-Q discrimination AUROC/AUPRC and high/low-return bias are in CRITIC_METRICS.json; no strided stage-0 probe was used. Calibration curves and ranking plots are in plots/.',
 'Final EMA target lag and twin disagreement (not a Bellman-target calibration test):',target_table,critical
 ]))
 text('SIMPLE_BASELINE_COMPARISON.md','\n\n'.join([
 '# Simple value predictors versus twin Transformers — v1',
 f"Stage constants have four FIT means. Ridge regression uses {simple['features_dim']+1} coefficients with lambda=1. The small MLP has {simple['MLP_parameters']:,} parameters (64/32 hidden units); the legacy Transformers have 12,839,497 each. All simple inputs are state/goal/action, stage/remaining horizon and deterministic post-action residuals/validity features; no future trajectory or identity label is included.",
 'Scaling uses FIT only. The MLP is selected at 100/250/500/1,000 CPU updates by ordinary DEV return MSE. The strongest simple predictor for primary comparisons is chosen once by DEV nonterminal expected-reference MSE, not by confirmation. Models see stage-balanced diagnostics; raw training examples and fixed-pair expected-reference conditions differ but share the fixed continuation policy and target definition.',
 f'DEV-selected simple predictor: **{best_simple}**.',compare,cirows,
 'The simple MLP receives up to 64,000 mini-batch presentations, whereas the Transformer pilot receives four transitions per new update, plus its 250 historical updates. Therefore these are bounded compute/configuration comparisons, not equal-example or architecture-intrinsic sample-efficiency proofs. The deterministic post-action residual features also give the small model a useful kinematic inductive bias which the legacy Transformer value head must infer. No claim that a larger model could never improve is made.',
 'An exploratory immediate-proximity control was added in final analysis: FIT-only intercept and slope per stage map the post-action J score to return. Its eight coefficients use no future features. It does not change predictor selection or the frozen promotion gates. Nonterminal RMSE:',
 table(['Partition','Stage-calibrated immediate-only score','MLP','Critic1','Critic2'],[[part,*[f(metrics[part][n]['nonterminal']['RMSE']) for n in ['immediate_proxy_posthoc','MLP','critic1','critic2']]] for part in ['CONFIRM','VALIDATION','VALIDATION_SHIFTED']]),
 'A stage-aware large variant was not added because original stage/remaining-horizon encoding is already explicit. No parameter count or actor capacity was increased. A falling critic TRAIN loss is not sufficient: promotion additionally requires held-out expected-value improvement, low bias, supported action ranking and a small external VALIDATION check.',critical
 ]))
 text('FIXED_IDENTITY_RWR_MONITORING.md','\n\n'.join([
 '# Fixed-identity RWR monitoring — v1',
 'No RWR training occurred. All eight original parent trajectories for 64 fixed TRAIN identities were evaluated under the parent, saved +25 and stopped +250 actors. Goals, prefixes, physical actions and identity denominators are identical. Parent behavior log-densities are reused from V7; +25/+250 transformed log-densities are recomputed on the same replay states. All checkpoint files remain unchanged.',likelihood,
 'Changes are full four-action joint log-density differences; dividing by 12 gives the mean per-component log-density change. A continuous density is not event probability. Self-normalized trajectory importance weights estimate success/invalid-event mass diagnostically, using the exact product density ratio under fixed-AA behavior. With only eight draws per task the estimator is biased/noisy; effective sample sizes and identity-clustered intervals are reported. No clipping or pretending that a sum of densities is probability mass.',
 rwrtable,
 'Each 64-task validation panel contains 32 protein/protected-group clusters, with two goal tasks per protein. Generated-prefix conditions reuse those same proteins. Paired intervals resample these 32 clusters, never treating the 2,048 candidates as independent biological observations. The TRAIN likelihood panel separately contains 64 identities and 512 trajectories.',
 'Parent and +25 full K32 pools are reused from V7. New +250 pools use precisely the same V7 task and candidate-indexed streams. Smaller K use prefixes of those pools, and K>1 keeps the V6 joint-goal/exact-validity selector. These VALIDATION panels were already exposed in V7: +250 assessment is a new frozen-checkpoint diagnostic, not independent confirmation or retrospective V7 promotion. Paired success-versus-K intervals, invalid-sample fractions, position/orientation, conditional KL and tail diversity are in RWR_METRICS.json.',
 'Goal sensitivity uses the same paired target/noise controls as V7. Parent/+25 controls are reused; +250 uses the corresponding matched streams. Results:',
 '```json\n'+json.dumps(rwr['goal_sensitivity'],indent=2)+'\n```',
 'The original +250 checkpoint stopped because the NLL continuation statistic compared two nonoverlapping 100-identity windows. This monitoring removes that confound for likelihood but does not erase the original stop. A likelihood gain must still translate into held-out sampled full-valid success; more probability on successful replay examples alone is not reliable K1 control. One fixed pool and a single historical training seed limit comparisons.'
 ]))
 shifted=table(['Predictor','Nonterminal shifted RMSE','Shifted bias','Native VAL RMSE','Twin disagreement'],[[name,f(metrics['VALIDATION_SHIFTED'][name]['nonterminal']['RMSE']),f(metrics['VALIDATION_SHIFTED'][name]['nonterminal']['mean_bias']),f(metrics['VALIDATION'][name]['nonterminal']['RMSE']),f(metrics['VALIDATION_SHIFTED']['twin_disagreement']['mean_absolute']) if name.startswith('critic') else 'NA'] for name in ['constant','ridge','MLP','critic1','critic2']])
 text('GENERATED_PREFIX_VALUE_ANALYSIS.md','\n\n'.join([
 '# Generated-prefix value analysis — v1',shifted,
 'Eight protected VALIDATION proteins are checked in both native and V7-generated-prefix conditions. The original V7 shifted prefix manifests are reused unchanged, with original targets retained. These are paired conditions, not 16 independent proteins. The expected-value actions and suffixes use common seed conventions across conditions; sampling remains the fixed parent law at each resulting prefix.',
 'Historical control limitation is preserved: V7 generated these prefixes with immutable V3 weights through its relative-context input adapter; they are not byte-identical to the earlier V3 zero-context decoding control. This does not invalidate within-V7.1 comparisons using the same fixed prefixes, but historical transfer scores cannot be pooled as matched evidence.',
 'The whole prefix participates in validity. Existing nonlocal clashes or spacing failures may make valid completion impossible under the fixed short final length; a regressor should not infer positive task return merely from endpoint proximity. Rg constraints can change as the chain extends. This study does not replace failed suffixes, relabel targets or learn a reachability model.',
 'Generated-prefix RWR proposal outcomes use the larger exposed V7 panel and are separate from this eight-protein value check:',rwrtable,
 'Confidence and task-distance strata are in CRITIC_METRICS.json. Small external samples and finite suffix references limit generalization; no folding or thermodynamic conclusion is supported.'
 ]))
 main=[
 '# E017 V7.1 final scientific report — v1',f'**Decision: {decision}.** No SAC actor, RWR actor, high-level or reachability updates were run.',
 f"Source commit `{protocol['source']}`; isolated D: worktree `e017-v71-expected-return-20261023-v1`. Fixed actor: V5 D+250, file SHA256 `{st['v']['PSHA']}`. Actor parameters: 17,438,114; J1: 12,351,746; critics: 12,839,497 each. The reference actor and J1 remained frozen and passed fingerprints. The stopped V7 critics were reused only as critic warm-starts under a new protocol, not retroactively approved.",
 '## Why V7 failed',
 'The original fit probe sampled only first-action states, while its held-out terminal requirement passed. It compared noisy realized suffix returns with a fixed explained-variance fit threshold, not a lower-noise expected-Q reference. The RWR continuation statistic also compared different identities. These are diagnostic/objective and sampling limitations; they do not prove critics cannot learn. V7 remains NO-GO, with every historical artifact and threshold unchanged.',
 '## Data, targets and independence',
 'Existing 4,096 parent trajectories produce 16,384 stage-tagged realized transitions. New fitting/development/confirmation subsets contain 256/64/64 distinct TRAIN identities. The 48 old critic-fitting identities are forced into FIT and old 32 diagnostic identities excluded from CONFIRM. Expected-reference subsets use 32/16/16 identities plus eight protected VALIDATION proteins in paired native/shifted conditions. The small VALIDATION panel was exposed in V7; it is an external-partition check, not virgin confirmation.',
 'Two actions per identical state, four stages, sixteen nonterminal suffix trials and one exact terminal evaluation give 640 state/action means across 80 conditions/72 proteins. Binary success, shaped task return, validity, conditional KL and soft entropy/KL-adjusted return remain distinct. No future coordinates or unrevealed AAs enter model features. Finite Monte Carlo means and their uncertainty are not exact expected values.',
 '## Critic learning and comparison',evidence,ranking_evidence,curve,f'Selected state: `{selected["path"]}`; SHA256 `{selected["sha256"]}`.',compare,critical,
 'Stage-specific errors, predicted-return quantile calibration, rank correlations, high/low-return bias and distance strata are reported in STAGE_STRATIFIED_CRITIC_CALIBRATION.md and CRITIC_METRICS.json. Primary future-value analysis excludes terminal stage 3. Discriminating complete successes with raw Q-scores does not calibrate success probabilities. Action ranking is assessed only on sufficiently supported paired reference contrasts, with coverage stated.',
 'The small MLP includes deterministic one-step kinematic residuals; the Transformer has the original prefix encoder and action-conditioned value head. Different inductive biases and exposure counts preclude a universal capacity conclusion. No redundant stage-aware variant was trained because the original horizon encoding is explicit. Targets are EMA diagnostics in supervised Monte Carlo fitting; this is not a Bellman/SAC learning experiment.',
 '## Fixed-identity RWR findings',rwr_evidence,likelihood,rwrtable,
 'No new RWR campaign occurred. Checkpoint +250 is read-only diagnostic evidence; the old V7 stop was neither bypassed nor promoted. Teacher log-density and importance estimates are distinct from sampled proposal event rates. The held-out curves are paired on the same already-exposed V7 goals/noise; improvements cannot be called independent confirmation. Goal-direction controls and diversity are recorded separately.',
 '## Recovery and resources',
 'Optimization diagnostics (combined loss is the sum of the two per-twin batch MSEs):\n\n```json\n'+json.dumps(resource['telemetry']['critic_optimization'],indent=2)+'\n```',
 f"New critic updates: {final['new_updates']}; historical: 250; actor updates: zero. Complete checkpoint sizes and incoming staging/headroom admission appear in RESOURCE_AND_STORAGE_REPORT.json. D: free before publication: {resource['D_free_current_bytes']:,} bytes. The 2 GiB operational reserve was retained. Full critic/target/optimizer/sampler/RNG restoration passed exact next-update replay in the selected same-process configuration; no fresh-process guarantee is claimed. Latest/best full bundles and immutable replay hashes are retained.",
 'All source/evidence/caches/temp files remain on D:. No original PDB, E010, E016/E017 report, checkpoint or dataset was deleted or modified. Only new run-owned superseded recovery bundles were retired after verified replacement. One heavy GPU worker ran at a time. Model/corpus hashes, plot hashes, final process exit and GPU release are in provenance.',
 '## Scientific answers and limits',
 'The original V7 failure is explained by its actual targets and probe distribution, not reinterpreted as a pass. Whether future-return learning is useful is determined by the independent TRAIN confirmation and small external check, including action ranking—not declining TRAIN loss. The stage tables identify the difficult decisions. If confidence or reference contrast coverage is insufficient, NO-GO is preserved rather than claiming accurate action values.',
 'RWR likelihood change and sampled K1/Best-of-K must agree before inferring useful amortization. Zero changes under a fixed pool do not prove population equivalence; a single training seed and eight importance samples per TRAIN task limit conclusions. The suffix budget also cannot prove zero reachability after sixteen failures. Q-values are shaped task returns, not protein foldability or stability probabilities.',
 'If all frozen value gates pass, the result supports proposing a separate bounded SAC pilot after explicitly aligning entropy/KL targets and rechecking actor safety. Otherwise the simplest next step is the best supported low-capacity predictor and a targeted fixed-state/action expected-return/ranking diagnostic at the limiting stages. No actor pilot or follow-on experiment is launched here.'
 ]
 text('FINAL_E017_V71_SCIENTIFIC_REPORT.md','\n\n'.join(main))
 text('NEXT_RL_RESEARCH_DECISION.md','\n\n'.join([
 '# Next RL research decision — E017 V7.1 v1',f'**{decision}.**',critical,
 '1. V7 failed the first-action fit threshold; terminal holdout errors passed. Its strided probe and noisy realized-return criterion limited interpretation. The original failed verdict remains unchanged.',
 '2. Meaningful future prediction requires an advantage on nonterminal expected references and action ranking. A good terminal score can be computed from the resulting geometry and is not enough.',evidence,ranking_evidence,compare,
 '3. Stage-specific metrics and supported-pair coverage distinguish learning failure from insufficient action contrast or Monte Carlo precision. No unsupported ranking is promoted.',
 f'4. The DEV-selected low-cost baseline is {best_simple}. If it matches or beats the large critics, use that baseline for the next narrowly scoped value/gradient diagnostic rather than increasing model size or updates automatically. Exposure and feature differences prevent a universal architecture conclusion.',
 '5. Fixed-identity RWR diagnostics measure actual likelihood changes, invalid-event estimates and held-out proposal outcomes. They do not create a new trained actor or retroactively select +250 in V7.',rwr_evidence,rwrtable,
 '6. A future SAC pilot needs target-semantic alignment: task-Q calibration is not automatically soft-Q calibration under an adaptive KL multiplier. Any future actor study must preserve J1, original validity and KL rules with a separate bounded approval. No SAC, hierarchy, reachability training or longer horizon is started.',
 '7. If evidence remains insufficient, the simplest next experiment should isolate the limiting stage with fixed state/action suffix references and a low-capacity predictor, or test action-sensitive value gradients under the existing parent. Do not substitute a more complex planner for an uncalibrated value function.',
 'All workers are stopped after publication. Wait for the next research decision.'
 ]))
 makeplots(final,metrics,rwr)
 provenance(protocol)

def makeplots(final,metrics,rwr):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
 for i in range(2):ax[0].plot([r['update'] for r in final['curves']],[r['scores'][i]['nonterminal']['RMSE'] for r in final['curves']],marker='o',label=f'critic{i+1}')
 ax[0].set_xlabel('new critic updates');ax[0].set_ylabel('DEV expected-return RMSE');ax[0].legend();ax[0].grid(alpha=.2)
 for name in ['constant','ridge','MLP','critic1','critic2']:ax[1].plot(range(4),[metrics['CONFIRM'][name][str(k)]['RMSE'] for k in range(4)],marker='o',label=name)
 ax[1].set_xticks(range(4));ax[1].set_xlabel('decision stage');ax[1].set_ylabel('TRAIN-confirmation expected RMSE');ax[1].legend(fontsize=8);ax[1].grid(alpha=.2);fig.savefig(R/'plots/STAGE_ERROR_AND_LEARNING.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
 for name in ['constant','ridge','MLP','critic1','critic2']:
  bins=metrics['CONFIRM'][name]['nonterminal']['calibration_quantiles'];ax[0].plot([r['predicted_mean'] for r in bins],[r['reference_mean'] for r in bins],marker='o',label=name)
  rank=metrics['CONFIRM'][name]['action_ranking'];ax[1].bar(name,rank['accuracy'] if rank['accuracy'] is not None else float('nan'))
  if rank['accuracy'] is None:ax[1].text(name,.03,'insufficient',rotation=90,ha='center',fontsize=7)
 ax[0].plot([0,1.25],[0,1.25],'k--',alpha=.4);ax[0].set_xlabel('predicted shaped task Q');ax[0].set_ylabel('mean empirical return');ax[0].legend(fontsize=8);ax[0].grid(alpha=.2);ax[1].axhline(.5,color='k',linestyle='--');ax[1].set_ylim(0,1);ax[1].tick_params(axis='x',rotation=30);ax[1].set_ylabel('supported-pair ranking accuracy');fig.savefig(R/'plots/CALIBRATION_AND_RANKING.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
 for axes,part in zip(ax,['CONFIRM','CONFIRM_SHIFTED']):
  for name,values in rwr['panels'][part].items():axes.plot([1,2,4,8,16,32],[values[str(k)]['success']/64 for k in [1,2,4,8,16,32]],marker='o',label=name)
  axes.set_xscale('log',base=2);axes.set_xticks([1,2,4,8,16,32],labels=[1,2,4,8,16,32]);axes.set_title(part);axes.set_ylabel('full-valid query success');axes.set_xlabel('nested K');axes.legend();axes.grid(alpha=.2)
 fig.savefig(R/'plots/FIXED_CHECKPOINT_RWR_SUCCESS.png',dpi=150);plt.close(fig)

def provenance(protocol):
 names=['FINAL_E017_V71_SCIENTIFIC_REPORT.md','CRITIC_TARGET_AND_RETURN_AUDIT.md','STAGE_STRATIFIED_CRITIC_CALIBRATION.md','SIMPLE_BASELINE_COMPARISON.md','FIXED_IDENTITY_RWR_MONITORING.md','GENERATED_PREFIX_VALUE_ANALYSIS.md','CHECKPOINT_AND_RECOVERY_MANIFEST.json','RESOURCE_AND_STORAGE_REPORT.json','NEXT_RL_RESEARCH_DECISION.md','CRITIC_METRICS.json','RWR_METRICS.json','EXPECTED_LABEL_CENSUS.json']+[str(p.relative_to(R)) for p in (R/'plots').glob('*.png')]
 INPUTS.update(protocol['input_SHA256']);INPUTS[str(Path(__file__))]=sha(__file__)
 for p in [R/'PROTOCOL.json',R/'AUXILIARY_ASSESSMENT_PROTOCOL.json',R/'AUXILIARY_RWR_GOAL_PROTOCOL.json',R/'analysis_source/v71_realized_eval.py',R/'analysis_source/v71_rwr_goal_eval.py',R/'analysis_source/v71_finish.py',R/'results/resource_watch.jsonl',R/'results/suffix_samples.jsonl',R/'results/simple_models.pt',R/'results/critic_updates.jsonl']:
  if p.exists():INPUTS[str(p)]=sha(p)
 for p,h in protocol['input_SHA256'].items():assert sha(p)==h,p
 parent=st['v']['PARENT'];assert sha(parent)==st['v']['PSHA'];INPUTS[str(parent)]=st['v']['PSHA']
 j1=R.parent/'e016_j1_bidirectional_multitask_20261019_v1/runs/J1/checkpoint_07500.pt';jsha='881457ff0396975c1b20dd0a8e2caac8521e65b7e1cec76a7bb531dbc437b412';assert sha(j1)==jsha;INPUTS[str(j1)]=jsha
 assert not list((R/'checkpoints').glob('*.part'))
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);assert not gpu.strip()
 save('PROVENANCE_SHA256.json',dict(schema='E017_V71_FINAL_v1',created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=protocol['source'],input_SHA256=INPUTS,output_SHA256={n:sha(R/n) for n in names},actor_updates=0,SAC_actor_updates=0,high_level_updates=0,reachability_updates=0,TEST_evaluated=False,AlphaFold=False,deleted_historical_artifacts=0,J1_modified=False,reference_actor_modified=False,GPU_compute_processes=[],all_artifacts_on_D=True,D_free_bytes=__import__('shutil').disk_usage(R).free,self_hash='excluded to avoid recursion'))
 print(json.dumps(dict(decision_files=names,free_bytes=__import__('shutil').disk_usage(R).free)),flush=True)
if __name__=='__main__':main()
