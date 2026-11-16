from common_review import *
import numpy as np,collections,math
from scipy.stats import binom,beta
rng=np.random.default_rng(310196)
def corr(x,y):
 return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None
def qs(x):return np.quantile(x,[0,.25,.5,.75,.95,.99,1]).tolist()
def grouped_ci(x):
 x=np.asarray(x);vals=[]
 for _ in range(20):vals.extend(x[rng.integers(len(x),size=(500,len(x)))].mean(1))
 return np.quantile(vals,[.025,.975]).tolist()
def dev(arm):
 root=ARMS[arm]/'production';groups={};out={};details=[]
 for p in sorted((root/'diagnostics').glob('DEVELOPMENT_*.json')):
  u=int(p.stem.split('_')[-1]);g=[read(root/f'diagnostics/DEV_{u:05d}_filtered_{i:02d}.json') for i in range(16)];groups[u]=g
  identity_means=np.array([np.mean([r['metrics']['backbone_valid'] for r in b['records']]) for b in g]);y=np.array([[r['metrics']['backbone_valid'] for r in b['records']] for b in g],float)
  within=float(((y-identity_means[:,None])**2).sum()/(16*7));between=float(8*np.var(identity_means,ddof=1));icc=(between-within)/(between+7*within) if between+7*within else None
  s=int(y.sum());aborts=[];reasons=collections.Counter();feasible=collections.Counter();byid=[]
  for b in g:
   rows=b['records'];bad=[r for r in rows if not r['metrics']['backbone_valid']];byid.append(dict(identity=b['task']['source_reference_identity'],goal_id=b['task']['goal']['goal_id'],difficulty=b['task']['goal']['difficulty'],failures=len(bad)))
   for r in bad:
    t=len(r['geometry']);terminal=[q for q in r['proposals'] if q[0]==t]
    aborts.append(t);reasons.update(v for q in terminal for v in q[8]);feasible.update(q[9] for q in terminal if len(q)>9)
    details.append(dict(arm=arm,update=u,identity=b['task']['source_reference_identity'],goal_id=b['task']['goal']['goal_id'],difficulty=b['task']['goal']['difficulty'],candidate_index=r['candidate_index'],seed=r['seed'],termination=r['termination'],prefix_length=t,terminal_proposal_count=len(terminal),all_terminal_rejected=all(not q[6] for q in terminal),reasons=dict(collections.Counter(v for q in terminal for v in q[8])),feasibility=dict(collections.Counter(q[9] for q in terminal if len(q)>9))))
  transitions=None
  if u:
   pairs=[(a,b) for x,z in zip(groups[0],g) for a,b in zip(x['records'],z['records'])];tr=collections.Counter(f"{int(a['metrics']['backbone_valid'])}->{int(b['metrics']['backbone_valid'])}" for a,b in pairs)
   delta=np.array([np.mean([int(b['metrics']['backbone_valid'])-int(a['metrics']['backbone_valid']) for a,b in zip(x['records'],z['records'])]) for x,z in zip(groups[0],g)])
   transitions=dict(counts=dict(tr),paired_identity_CI95=grouped_ci(delta),effect=float(delta.mean()),matched_seed=True)
  out[str(u)]=dict(valid=s,n=128,iid_CP95=[float(beta.ppf(.025,s,129-s)) if s else 0,float(beta.ppf(.975,s+1,128-s)) if s<128 else 1],identity_bootstrap_CI95=grouped_ci(identity_means),ICC_method_of_moments=icc,ICC_warning='Only16 identities; unstable descriptive estimate, not calibration',failures_by_identity=byid,abort_prefix_positions=aborts,terminal_reason_counts=dict(reasons),terminal_feasibility_counts=dict(feasible),initial_transitions=transitions,summary=read(p)['summary'])
 return out,details
outputs={};allbad=[];train={}
for arm in ARMS:
 outputs[arm],bad=dev(arm);allbad+=bad;root=ARMS[arm]/'production';cs=[read(p)['logical'] for p in sorted((root/'commits').glob('*.json'))];rows=[];groups=[];difficulty=collections.defaultdict(list)
 for p in sorted((root/'rollouts').glob('*.json')):
  packet=read(p);g=packet['task']['goal'];rr=packet['records'];r=np.array([v['metrics']['reward'] for v in rr]);llr=np.array([v['trace_logprob']-v['reference_trace_logprob'] for v in rr]);aug=r-.009499081422170746*llr;adv=(16*aug-aug.sum())/15
  b=np.array([v['metrics'].get('buried_mean',np.nan) for v in rr]);bur=np.array([v['metrics'].get('buried_mean',2)<=g['burial_max'] and v['metrics']['hard_valid'] for v in rr]);ex=np.array([v['metrics'].get('exposed_mean',-1)>=g['exposure_min'] and v['metrics']['hard_valid'] for v in rr]);good=np.isfinite(b)
  groups.append(dict(update=packet['update'],identity=packet['task']['source_reference_identity'],difficulty=g['difficulty'],reward_mean=float(r.mean()),reward_SD=float(r.std()),aug_adv_SD=float(adv.std()),reward_burial_corr=corr(r,bur),exposure_burial_corr=corr(ex,bur),informative_burial_group=bool(bur.any() and not bur.all())))
  for i,v in enumerate(rr):
   m=v['metrics'];vals=[.5*int(m['success'])]+[.1*int(m['hard_valid'])*q for q in m.get('utilities',[0]*5)];vals=(vals+[0]*6)[:6]
   row=dict(arm=arm,update=packet['update'],identity=packet['task']['source_reference_identity'],difficulty=g['difficulty'],reward=r[i],advantage=adv[i],llr=llr[i],burial=bool(bur[i]),exposure=bool(ex[i]),hard=m['hard_valid'],prefix_length=len(v['geometry']),proposals=len(v['proposals']),active_dimensions=sum(min(q[0],3) for q in v['proposals']),components=vals)
   rows.append(row);difficulty[g['difficulty']].append(row)
 comps=np.array([v['components'] for v in rows]);rewards=np.array([v['reward'] for v in rows]);norm=[v['gradient_norm'] for v in cs]
 train[arm]=dict(updates=len(cs),gradient_norm_quantiles=qs(norm),clip_factor_quantiles=qs([min(1,5/x) for x in norm]),clip_count=sum(x>5 for x in norm),step_KL_quantiles=qs([v['step_KL'] for v in cs]),lr_quantiles=qs([v['lr'] for v in cs]),entropy_quantiles=qs([v['diagnostics']['latent_Gaussian_entropy_per_active_dimension'] for v in cs]),advantage_quantiles=qs([v['advantage'] for v in rows]),proposal_count_quantiles=qs([v['proposals'] for v in rows]),active_dimensions_quantiles=qs([v['active_dimensions'] for v in rows]),reward_burial_corr=corr(rewards,[v['burial'] for v in rows]),exposure_burial_corr=corr([v['exposure'] for v in rows],[v['burial'] for v in rows]),informative_burial_groups=sum(v['informative_burial_group'] for v in groups),group_reward_burial_correlation_median=float(np.nanmedian([v['reward_burial_corr'] for v in groups if v['reward_burial_corr'] is not None])),components={k:dict(mean=float(comps[:,j].mean()),SD=float(comps[:,j].std()),nonzero=float(np.mean(comps[:,j]>0)),correlation_total=corr(comps[:,j],rewards)) for j,k in enumerate(['success','exposure','burial','contrast','motif','compatibility'])},difficulty={k:dict(attempts=len(v),aborts=sum(not z['hard'] for z in v),reward=float(np.mean([z['reward'] for z in v])),burial=float(np.mean([z['burial'] for z in v]))) for k,v in difficulty.items()},abort_positions=dict(collections.Counter(v['prefix_length'] for v in rows if not v['hard'])),groups=groups)
write('EXISTING_RECORD_ANALYSIS.json',dict(development=outputs,training=train,abort_details=allbad,interpretation='New retrospective descriptive stability analyses; does not replace frozen historical scientific tests',optimizer_updates=0))
curves=[]
for p in [.99,.98,.975,.97,.965,.96,.95,.94,.93,.90]:
 stop=float(binom.cdf(121,128,p));curves.append(dict(true_validity=p,probability_below95_one_iid_panel=stop,six_independent_looks_hypothetical=1-(1-stop)**6))
# Conservative prospective whole-identity bounded Hoeffding interval; candidates inside a group may depend arbitrarily.
alpha=.05/(2*2*7);sizes=[512,1024,2048,4096,8192];eps=[dict(groups=n,epsilon=math.sqrt(math.log(2/alpha)/(2*n)),admit_if_mean_at_least=.95+math.sqrt(math.log(2/alpha)/(2*n))) for n in sizes]
write('MONITOR_DESIGN_CALCULATIONS.json',dict(iid_sensitivity_curves=curves,warning='iid illustration, not fitted false-stop probability; reused deterministic seeds make unchanged-policy repeated panels identical; changing policies and shared identities invalidate independent-look formula',prospective_interval_alpha_per_arm_metric_look=alpha,bounded_identity_Hoeffding=eps,requirements='unchanged population validity floor .95; no continuation on an inconclusive interval; admission lower bound >=.95; fail upper<.95, pause unresolved; independent TRAIN identity groups with no optimization leakage',new_analyses_not_historical_decisions=True))
print('RECORD_ANALYSIS_COMPLETE',flush=True)
