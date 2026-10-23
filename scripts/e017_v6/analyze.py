"""CPU analysis and final publication for the frozen V6 candidate experiment."""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np

R=Path('/mnt/d/Simone/proteinGen_data/e017_v6_frozen_parent_candidates_20261023_v1')
OLD=R.parent/'e017_v51_safe_control_20261022_v1'
V5=R.parent/'e017_v5_goal_observability_20261022_v1'
V4=R.parent/'e017_v4_multistep_20261022_v1'
sys.path.insert(0,str(R/'source/scripts/e017_v6'))
from study import read, save, sha, seed, tasks, free

def text(name,s):
 p=R/name;tmp=p.with_name(p.name+'.part')
 with tmp.open('w') as f:f.write(s+'\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)

def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x) for x in row)+' |' for row in rows])
def f(x):return 'NA' if x is None else f'{x:.4f}'
def corr(a,b):
 return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None
def ranks(x):
 x=np.asarray(x);return np.array([((x<y).sum()+((x==y).sum()-1)/2) for y in x])

def main():
 cfg=read(R/'PROTOCOL.json');completed=read(R/'WORKER_COMPLETED.json');parity=read(R/'PARITY.json')
 assert completed['complete'] and parity['PASS']
 rows=[json.loads(s) for s in (R/'results/candidates.jsonl').read_text().splitlines()];ts=tasks();td={t['task_id']:t for t in ts}
 assert len(rows)==2016 and len({(r['task_id'],r['candidate_index']) for r in rows})==2016
 assert sum(r['reused'] for r in rows)==126
 pools={t['task_id']:sorted([r for r in rows if r['task_id']==t['task_id']],key=lambda r:r['candidate_index']) for t in ts}
 legacy=read(OLD/'results/INITIAL_00000_AUDIT_H4_native.json');lk={(r['task_id'],r['seed']):r for r in legacy['records']}
 for tid,pool in pools.items():
  assert [r['candidate_index'] for r in pool]==list(range(32))
  assert len({r['seed'] for r in pool})==32
  for r in pool:
   assert r['seed']==seed(tid,r['candidate_index'])
   assert r['success']==(r['valid'] and r['goal_hit'])
   if r['reused']:
    a=lk[(tid,r['seed'])]
    assert all(r[k]==a[k] for k in ['tail','valid','success','position_A','orientation_deg','KL'])
 assert sum(r['success'] for r in rows if r['reused'])==20
 groups=sorted({t['protected_group_id'] for t in ts});gid=np.array([groups.index(t['protected_group_id']) for t in ts]);counts=np.bincount(gid)
 draws=np.random.default_rng(cfg['bootstrap_seed']).integers(0,len(groups),(cfg['bootstrap_replicates'],len(groups)))
 def estimate(v):
  v=np.asarray(v,dtype=float);sums=np.bincount(gid,weights=v,minlength=len(groups));samples=sums[draws].sum(1)/counts[draws].sum(1)
  return dict(estimate=float(v.mean()),CI95=np.quantile(samples,[.025,.975]).tolist(),CI99_166667_six_K=np.quantile(samples,[.05/12,1-.05/12]).tolist())
 def selected_stats(selected,success):
  present=[r for r in selected if r is not None]
  out=dict(tasks=63,selected=len(present),abstentions=63-len(present),success_count=sum(success),full_success=estimate(success),
    valid_count=sum(r['valid'] for r in present),collision_count=sum(r['clashes_per100']>2 for r in present),
    position_A_mean_conditional=float(np.mean([r['position_A'] for r in present])) if present else None,
    orientation_deg_mean_conditional=float(np.mean([r['orientation_deg'] for r in present])) if present else None,
    proposal_KL_selected_mean_conditional=float(np.mean([r['KL'] for r in present])) if present else None,
    selected_candidate_indices=[r['candidate_index'] if r else None for r in selected])
  out['valid_rate_unconditional']=estimate([r is not None and r['valid'] for r in selected])
  out['collision_rate_unconditional']=estimate([r is not None and r['clashes_per100']>2 for r in selected])
  # Continuous errors use all selected tasks; this panel never conditions away failures silently.
  if len(present)==63:
   out['position_A']=estimate([r['position_A'] for r in selected]);out['orientation_deg']=estimate([r['orientation_deg'] for r in selected])
  return out
 result={}; taskmetrics={};incremental=[]
 for K in cfg['K']:
  valid=[];full=[];goal=[];sels={s:[] for s in ['A','B','C','C_original_cost']};decomp=collections.Counter();per=[];pairdist=[];tracedist=[];exacts=[]
  for t in ts:
   p=pools[t['task_id']][:K];v=[r for r in p if r['valid']];hits=[r for r in p if r['goal_hit']];wins=[r for r in p if r['success']]
   valid.append(bool(v));full.append(bool(wins));goal.append(bool(hits))
   sels['A'].append(p[0]);sels['B'].append(min(p,key=lambda r:(r['position_A'],r['candidate_index'])))
   c=min(v,key=lambda r:(r['goal_score'],r['original_goal_cost'],r['candidate_index'])) if v else None
   sels['C'].append(c);sels['C_original_cost'].append(min(v,key=lambda r:(r['original_goal_cost'],r['candidate_index'])) if v else None)
   assert bool(c and c['success'])==bool(wins)
   category='4_full_success_present' if wins else '2_goal_hits_all_invalid' if hits else '3_valid_but_no_goal_hit' if v else '1_no_goal_hit_and_no_valid'
   decomp[category]+=1
   a=np.array([r['tail'] for r in p]);ix=np.triu_indices(K,1)
   endpoints=np.linalg.norm(a[:,-1,None,:]-a[:,-1,:][None,:,:],axis=-1)[ix]
   rms=np.sqrt(np.square(a[:,None]-a[None,:]).sum(-1).mean(-1))[ix]
   pairdist.extend(endpoints.tolist());tracedist.extend(rms.tolist());exacts.append(len({tuple(np.round(x,6).flat) for x in a}))
   per.append(dict(task_id=t['task_id'],identity=t['identity'],group=t['protected_group_id'],candidate_count=K,
    valid_candidates=len(v),goal_hit_candidates=len(hits),full_success_candidates=len(wins),category=category,
    valid_but_goal_miss_candidates=sum(r['valid'] and not r['goal_hit'] for r in p),
    invalid_goal_hit_candidates=sum(not r['valid'] and r['goal_hit'] for r in p),
    endpoint_pair_distance_mean_A=float(endpoints.mean()) if len(endpoints) else None,
    trajectory_pair_RMSD_mean_A=float(rms.mean()) if len(rms) else None,
    selected={s:(a[-1]['candidate_index'] if a[-1] else None) for s,a in sels.items()}))
  selected={s:selected_stats(v,[bool(r and r['success']) for r in v]) for s,v in sels.items()}
  csuccess=[bool(r and r['success']) for r in sels['C']];bsuccess=[bool(r and r['success']) for r in sels['B']];asuccess=[bool(r and r['success']) for r in sels['A']]
  newtime=sum(r['seconds'] for p in pools.values() for r in p[:K] if r['seconds'] is not None)
  equivalent_time=newtime+(126 if K>=2 else 63)*legacy['seconds']/126
  result[str(K)]=dict(tasks=63,at_least_one_valid_count=sum(valid),at_least_one_valid=estimate(valid),
   at_least_one_full_success_count=sum(full),at_least_one_full_success=estimate(full),at_least_one_goal_hit_count=sum(goal),at_least_one_goal_hit=estimate(goal),
   no_goal_hit_count=63-sum(goal),partition=dict(decomp),selectors=selected,C_equals_D_verified=True,
   paired_C_minus_B_success=estimate(np.array(csuccess,dtype=float)-np.array(bsuccess,dtype=float)),
   paired_C_minus_A_success=estimate(np.array(csuccess,dtype=float)-np.array(asuccess,dtype=float)),
   paired_C_minus_legacy_two_draw_mean=estimate(np.array(csuccess,dtype=float)-np.array([np.mean([r['success'] for r in pools[t['task_id']][:2]]) for t in ts])),
   diversity=dict(endpoint_pair_distance_mean_A=float(np.mean(pairdist)) if pairdist else None,
    trajectory_pair_RMSD_mean_A=float(np.mean(tracedist)) if tracedist else None,unique_rounded_traces_total=sum(exacts),
    trace_count=63*K,rounding_A=1e-6),
   cost=dict(full_generation_actor_passes_each=63*K*4,full_generation_J1_passes_each=63*K*4,
     new_generation_actor_passes_each=63*max(0,K-2)*4,new_generation_J1_passes_each=63*max(0,K-2)*4,
     new_generation_seconds=newtime,historical_reused_panel_seconds=legacy['seconds'] if K>=2 else legacy['seconds']/2,
     historical_reused_seconds_are_estimated='saved whole126-attempt panel time includes original fifth BC forward per attempt; half allocated to K1 without per-seed timings',
     estimated_reuse_plus_new_seconds=equivalent_time,
     seconds_per_success_task_by_selector={s:(legacy['seconds']/2 if s=='A' else equivalent_time)/q['success_count'] if q['success_count'] else None for s,q in selected.items()},
     seconds_per_full_success_task_amortized=equivalent_time/sum(full) if sum(full) else None))
  taskmetrics[str(K)]=per
 # Plateau: paired incremental gains preserve task nesting.
 for prev,K in zip(cfg['K'][:-1],cfg['K'][1:]):
  delta=np.array([r['full_success_candidates']>0 for r in taskmetrics[str(K)]],dtype=float)-np.array([r['full_success_candidates']>0 for r in taskmetrics[str(prev)]],dtype=float)
  assert np.all(delta>=0)
  incremental.append(dict(K_from=prev,K_to=K,new_success_tasks=int(delta.sum()),paired_delta=estimate(delta),extra_actor_J1_passes_each=63*(K-prev)*4))
 task_difficulty=[]
 for t in ts:
  p=pools[t['task_id']];d=float(np.linalg.norm(np.array(t['target'])-np.array(t['native_prefix'][-1])))
  task_difficulty.append(dict(task_id=t['task_id'],identity=t['identity'],group=t['protected_group_id'],variant=t['variant'],
    initial_target_distance_A=d,success_probability_sample_mean=sum(r['success'] for r in p)/32,
    invalid_probability_sample_mean=sum(not r['valid'] for r in p)/32,all_32_fail=not any(r['success'] for r in p)))
 ps=[x['success_probability_sample_mean'] for x in task_difficulty];ds=[x['initial_target_distance_A'] for x in task_difficulty]
 distancecorr=corr(ds,ps);rankcorr=corr(ranks(ds),ranks(ps))
 # Identity clustered correlation uncertainty; bootstrap complete identity groups.
 taskdraws=[np.flatnonzero(gid==g) for g in range(len(groups))];cr=[]
 for dd in draws[:4000]:
  ii=np.concatenate([taskdraws[g] for g in dd]);value=corr(np.array(ds)[ii],np.array(ps)[ii])
  if value is not None:cr.append(value)
 matched=[]
 for identity in sorted({t['identity'] for t in ts}):
  pp=[x['success_probability_sample_mean'] for x in task_difficulty if x['identity']==identity]
  if len(pp)==2:matched.append(pp)
 paircorr=corr([x[0] for x in matched],[x[1] for x in matched])
 early=[r for r in rows if r['early_rule']]
 early_failures=collections.Counter(reason for r in early for reason in r['early_rule']['reasons'])
 assert not any(r['success'] for r in early)
 causes=collections.Counter(c for r in rows for c in r['validity_failures'])
 failures=dict(total_candidates=len(rows),invalid_candidates=sum(not r['valid'] for r in rows),causes=dict(causes),
  goal_hit_invalid=sum(r['goal_hit'] and not r['valid'] for r in rows),valid_goal_miss=sum(r['valid'] and not r['goal_hit'] for r in rows),
  full_success=sum(r['success'] for r in rows),
  early_rule=dict(flagged=len(early),flagged_success=sum(r['success'] for r in early),causes=dict(early_failures),
     first_step_counts=dict(collections.Counter(str(r['early_rule']['step']) for r in early)),
     potential_actor_J1_steps_saved_each=sum(r['early_rule']['saved_forward_steps'] for r in early),
     interpretation='retrospective exact necessary rejection, no early stopping applied to original pools; not a learned positive reachability prediction'))
 new=[r for r in rows if not r['reused']]
 components=np.array([r['KL_components_by_step'] for r in new]);kl=dict(new_candidates=1890,mean_components_distance_angle_torsion=components.mean((0,1)).tolist(),
  mean_by_step=components.mean((0,2)).tolist(),all_candidate_proposal_KL_mean=float(np.mean([r['KL'] for r in rows])),
  meaning='conditional proposal KL averaged on sampled states; not selection-policy KL, not a promotion certificate')
 taskvec=np.array([sum(r['success'] for r in pools[t['task_id']])/32 for t in ts])
 pbar=float(taskvec.mean());independent_model32=float(np.mean(1-(1-taskvec)**32))
 out=dict(schema='E017_V6_frozen_candidate_results_v1',protocol=cfg,source_generation_commit=cfg['source_commit'],
  source_analysis_commit=subprocess.check_output(['git','-C',str(R/'source'),'rev-parse','HEAD'],text=True).strip(),
  by_K=result,incremental_K=incremental,task_level_by_K=taskmetrics,failures=failures,proposal_KL=kl,
  difficulty=dict(distance_success_Pearson=distancecorr,CI95_identity_clustered=np.quantile(cr,[.025,.975]).tolist(),
   bootstrap_valid_replicates=len(cr),distance_success_Spearman=rankcorr,paired_goal_success_probability_correlation=paircorr,paired_identities=len(matched),
   pooled_candidate_success_probability=pbar,naive_homogeneous_success_at32=1-(1-pbar)**32,
   plug_in_heterogeneous_success_at32=independent_model32,
   caution='pool success fraction and probability use same32 draws; plug-in estimates biased/circular, not independent success forecasts; no zero-probability claim for0/32 tasks',tasks=task_difficulty),
  resource=completed,parity=parity,confidence_intervals='paired protected-identity bootstrap20,000 replicates, seed176617; task-weighted ratios,32groups; six-K bands endpoint-wise; one fixed proposal pool/no training-seed uncertainty',
  limits='reused validation diagnostic; all32 generated per task, no independent pool replicate; selected-law KL unknown; CA plausibility not folding; missing historical per-candidate wall times')
 save('CANDIDATE_SUCCESS_BY_K.json',out)
 save('ANALYSIS_INTERMEDIATE.json',dict(by_K={K:{k:v for k,v in q.items() if k!='selectors'} for K,q in result.items()},failures=failures,KL=kl,difficulty={k:v for k,v in out['difficulty'].items() if k!='tasks'}))
 # Reports are created from these exact values below.
 reports(out,legacy)

def reports(out,legacy):
 by=out['by_K'];cfg=out['protocol'];q=by['32'];N=q['at_least_one_full_success_count'];B=q['selectors']['B']['success_count'];C=q['selectors']['C']['success_count'];A=q['selectors']['A']['success_count']
 primary=q['paired_C_minus_B_success'];lo,hi=primary['CI95'];inc=out['incremental_K'][-1];resource=out['resource'];failure=out['failures']
 def ci(x):return f"{100*x['estimate']:.2f}% [{100*x['CI95'][0]:.2f}, {100*x['CI95'][1]:.2f}]"
 def diff(x):return f"{100*x['estimate']:.2f}pp [{100*x['CI95'][0]:.2f}, {100*x['CI95'][1]:.2f}]"
 overview=table(['K','Any valid /63','Full success exists /63 [95%]','B min-position success','C validity-first success','C−B Δpp [95%]'],[
  [K,f"{v['at_least_one_valid_count']}/63",f"{v['at_least_one_full_success_count']}/63; {ci(v['at_least_one_full_success'])}",f"{v['selectors']['B']['success_count']}/63",f"{v['selectors']['C']['success_count']}/63",diff(v['paired_C_minus_B_success'])] for K,v in by.items()])
 selection=table(['K/selector','Success /63','Valid /63','Collisions /63','Abstain','Mean position Å','Mean rotation °','Selected proposal KL'],[
  [f'{K}/{s}',f"{v['success_count']}/63",f"{v['valid_count']}/63",f"{v['collision_count']}/63",v['abstentions'],f(v['position_A_mean_conditional']),f(v['orientation_deg_mean_conditional']),f(v['proposal_KL_selected_mean_conditional'])]
  for K,a in by.items() for s,v in a['selectors'].items()])
 partition=table(['K','1 No goal/no valid','2 Goal hit(s), all invalid','3 Valid but no goal hit','4 Full success present','Marginal no goal hit'],[
  [K,*[v['partition'].get(c,0) for c in ['1_no_goal_hit_and_no_valid','2_goal_hits_all_invalid','3_valid_but_no_goal_hit','4_full_success_present']],v['no_goal_hit_count']] for K,v in by.items()])
 increments=table(['K change','New successful tasks','Paired gain [95%]','Extra J1+actor calls EACH'],[[f"{x['K_from']}→{x['K_to']}",x['new_success_tasks'],diff(x['paired_delta']),x['extra_actor_J1_passes_each']] for x in out['incremental_K']])
 diversity=table(['K','Unique traces /all','Mean endpoint distance Å','Mean trajectory RMSD Å'],[[K,f"{x['diversity']['unique_rounded_traces_total']}/{x['diversity']['trace_count']}",f(x['diversity']['endpoint_pair_distance_mean_A']),f(x['diversity']['trajectory_pair_RMSD_mean_A'])] for K,x in by.items()])
 costs=table(['K','Actor passes EACH/J1 passes EACH','New passes EACH','New measured seconds','History+new seconds estimate','Seconds/full-success task C'],[
  [K,v['cost']['full_generation_actor_passes_each'],v['cost']['new_generation_actor_passes_each'],f(v['cost']['new_generation_seconds']),f(v['cost']['estimated_reuse_plus_new_seconds']),f(v['cost']['seconds_per_full_success_task_amortized'])] for K,v in by.items()])
 rowdetails=table(['K','A seconds/success','B seconds/success','C seconds/success','Original-cost C seconds/success'],[[K,*[f(v['cost']['seconds_per_success_task_by_selector'][s]) for s in ['A','B','C','C_original_cost']]] for K,v in by.items()])
 # Scientific conclusions are tied to prespecified results, never to lower position loss alone.
 strength='supported on this reused diagnostic panel' if lo>0 and primary['estimate']>=.05 else 'not established by the prespecified ≥5pp/positive paired95% criterion'
 nextchoice='inference-time validity-aware candidate selection as a bounded diagnostic baseline' if C>B and N>=32 else 'improving the proposal distribution, retaining exact validity checks as diagnostics'
 source=out['source_generation_commit'];analysis=out['source_analysis_commit']
 main=[
 '# E017 V6 — frozen-parent candidate selection, v1',
 f"**Completed without training.** The frozen parent supplies full valid H4 successes for **{N}/63 tasks at K32**. B (minimum positional error) succeeds on **{B}/63**; C (validity-first, threshold-aware geometric selection) on **{C}/63**. The paired C−B gain is **{diff(primary)}** and is {strength}. This is finite-sample geometric inference, not learned planning, folding certification or model promotion.",
 '## Frozen source, checkpoint and tasks',
 f"New isolated D: source worktree branch `e017-v6-frozen-candidates-20261023-v1`. Sampling commit `{source}`; final-analysis commit `{analysis}`. Parent: `{cfg['parent']}`, SHA256 `{cfg['parent_SHA256']}`. It is V5 D+250, selected before V5.1, not a failed V5.1 latest checkpoint. Strict actor restoration passed:17,438,114 parameters, all frozen; frozen J1 has12,351,746 parameters. J1 file SHA256 `881457ff0396975c1b20dd0a8e2caac8521e65b7e1cec76a7bb531dbc437b412`. Actor/parent hashes and J1 tensor fingerprint match before/after. Critics/reachability/high-level networks were not constructed or trained.",
 'Existing V5.1 AUDIT native H4:63 tasks,32 protected protein identities/groups, native and valid synthetic alternative goals at prefixes20/28; full deposited sequence length remains model metadata. TEST stays sealed. This reuses scored VALIDATION and is an exploratory proposal/selection diagnostic, not virgin confirmation. Exact task/source/group IDs are retained in the original frozen manifest and candidate JSONL; TRAIN homology-group separation was rechecked. Every admitted native/synthetic target retains its valid witness. No full-chain protein generation is claimed.',
 '## Reuse and exact matching',
 'The parent initialization audit already contained seeds7101/7102 for each of63 tasks. All126 original coordinate tails/outcomes were reused exactly, and only30 additional trajectories/task were generated:1,890 new+126 reused=2,016 total candidates. Repeated trained checkpoints and MATCHED100 copies were never treated as independent parent draws. Two bounded GPU parity draws reproduced saved parent coordinates and position errors exactly, with strict validity/success and KL agreement. Removing the historical fifth native-BC diagnostic forward preserved generation exactly, leaving four actor+four J1 calls per new trajectory.',
 f"The historical20/126 reference is reproduced by the two reused draws per identical task; K1 uses only seed7101 and is **{A}/63**, so it is not relabeled20/126. C32 versus the two-draw-mean historical baseline is {diff(q['paired_C_minus_legacy_two_draw_mean'])}, using task/group-paired data while retaining the distinct per-attempt versus per-query estimands. More candidates increase inference compute; these are not improved neural weights.",
 '## Frozen candidate protocol and selectors',
 'K=1,2,4,8,16,32 are prefixes of one32-candidate pool/task. Candidate0/1 seeds7101/7102 preserve original shared-task noise coupling. For candidate i≥2, seed=int(SHA256(`E017V6|7617|task_id|i`)[:15],16). Seeds are distinct within every pool. No temperature, action transform/support, threshold or distribution change, no replacement of failures. BF16 neural/FP64 geometry, all networks eval/no-grad; only one GPU worker.',
 'A is candidate0 for all K. B minimizes position only, ignoring validity/orientation. C discards hard-invalid candidates, minimizes max(position/8Å,orientation/35°), then original Huber+5sin²(θ/2) score and candidate index; an empty valid set is an explicit abstention/failure. D asks whether any full valid success exists. Because C’s score≤1 exactly iff both goal thresholds pass, **C attains D mathematically**. D is not an independent performance win over C. A secondary original-goal-cost-only valid selector is reported to expose cases where continuous cost ranks a goal miss above a full hit.',
 '## Success scaling and uncertainty',overview,
 'Success/validity rates count63 tasks, not2,016 independent proteins. Paired identity-cluster bootstrap20,000 replicates, seed176617, resampling32 protected groups with all their goals.95% intervals are nominal; endpoint-wise six-K99.166667% bands are in JSON. Primary C−B atK32 was frozen in PROTOCOL.json before sampling; K curves are descriptive. Fixed pool/legacy shared streams and only32groups limit generalization; no training-seed or independent pool-repeat uncertainty is claimed.',
 increments,
 f"The16→32 extension adds{inc['new_success_tasks']} newly successful tasks. This is the measured late-pool gain, not proof of an asymptote; no K>32 experiment or post hoc allocation extension was run. Tables preserve nonmonotonic B success even though B position and D existence are monotonic in K.",
 '## Selection outcomes and the four-way failure decomposition',selection,
 'C conditional mean errors/KL use selected tasks only; abstentions remain failures in the63-task full-success denominator. A’s outcome and query-generation cost remain one candidate despite appearing in every K row; full-pool resource numbers describe the pool shared by B/C/D, not a demand that A generate unused candidates.',
 partition,
 'These four rows are mutually exclusive with priority4→2→3→1. Class1 means neither a goal hit nor any valid candidate; class3 has no goal hit but does have valid candidates. The broader “no candidate satisfies the geometric goal” marginal is1+3; class2 can also contain valid goal-missing candidates. Candidate-level overlapping counts are retained separately. Failure at32 draws does not prove geometric impossibility or zero proposal probability.',
 '## Physical failures, diversity and difficulty',
 f"Across2,016 candidates, invalid={failure['invalid_candidates']}, full successes={failure['full_success']}; failure criteria={failure['causes']}. Target-hitting but invalid candidates={failure['goal_hit_invalid']}; valid goal misses={failure['valid_goal_miss']}. See VALIDITY_VS_GOAL_ANALYSIS.md for exact diagnostics and early-rule analysis.",diversity,
 f"Initial target-tip distance versus per-task success fraction: Pearson r={f(out['difficulty']['distance_success_Pearson'])}, identity-clustered95%CI{out['difficulty']['CI95_identity_clustered']}; Spearman={f(out['difficulty']['distance_success_Spearman'])}. Same-identity paired-goal success-fraction correlation={f(out['difficulty']['paired_goal_success_probability_correlation'])} across{out['difficulty']['paired_identities']} pairs. These are descriptive task-dependence estimates, affected by finite32-draw noise. Shared difficulty means pooled-candidate success cannot be converted to query success by assuming every task has identical probability.",
 '## Historical controls and scientific scope',
 'V5.1 selected100 AUDIT native H4 A/B/C/D full success5/4/9/8 of126 versus parent20/126; all selected100 arms violate native H2/H4 mean KL on that confirmation panel. V6 does not rescue their eligibility or reuse their weights. Same task/criteria comparisons are valid descriptively, but32-candidate queries have a different compute budget/estimand than single attempts. No V5.1 failed250 state is promoted.',
 'V5 H2 independent-confirmation D55/128 versus frozen-V3 A39/128 established useful short-horizon relative-goal conditioning. Its scored32groups differ from V6, so these are context, not paired significance comparisons. V5 C350 H4 stopped at KL2.307407 with12/92full successes,66/92valid, all26invalid due to clashes; exact350weights were not saved. V6 adds no trained alternative.',
 'V4 classical IK64 achieved26/64full native H8 successes (mean position0.1386Å,orientation22.2246°), generated-prefix10/64. Its horizon, identity/task panel and prior semantics differ; it is contextual evidence only. V5 existing H4 confirmation IK achieved63/64queries with up to two starts on its earlier different32groups; no new IK was run and neither baseline is called matched to V6. Frozen V3 remains a historical source/control, not this parent.',
 '## Inference cost and recovery',costs,
 f"One complete worker finished in{resource['elapsed_seconds']:.3f}s including model setup, verification, fsync, reused rows and parity. New candidate timer sums are in JSON; historical reused panel time{legacy['seconds']:.3f}s includes extra BC forwards. Summing new timers+historical panel gives an amortized estimate, not measured end-to-end cold-startK latency. See INFERENCE_COST_ANALYSIS.md for exact passes, timing limits and memory. Peak allocated/reserved CUDA={resource['GPU_peak_allocated_bytes']:,}/{resource['GPU_peak_reserved_bytes']:,}bytes; peak process RSS={resource['max_RSS_KiB']:,}KiB.",
 'Every candidate row was flushed/fsynced before advancing; RECOVERY records complete task boundaries, with indexed seeds allowing deterministic missing-row continuation if interrupted. No optimizer state/checkpoint copy was needed. Completed raw data have all63×32 unique task-index pairs. WORKER_COMPLETED and final frozen actor/J1/parent checks passed. All evidence, source, logs, caches and temporary paths are on D:;2GiB+64MiB admission reserve stayed intact; E010 and historical E016/E017 were not changed.',
 '## Decision',
 f"The next recommendation is **{nextchoice}**. More candidates can reveal valid goal successes already present in the frozen law; validity-first selection is an exactly computable finite-budget baseline, not learned planning. Use it as the comparison target for later proposal improvements. No high-level RL or learned reachability claim follows from this static task experiment.",
 'Selected-law KL is not estimated by averaging proposal KL along selected trajectories. State-dependent ranking changes the sampling law; therefore this study establishes finite-pool feasibility/selection evidence only, not a KL-compliant policy promotion. Cα validity and tolerance hits do not certify biochemical plausibility, foldability or stability. No training/SAC/TEST/AlphaFold/destructive cleanup was performed. All workers exit after finalization; no next phase is launched.'
 ]
 text('FINAL_E017_V6_SCIENTIFIC_REPORT.md','\n\n'.join(main))
 validity=[
 '# Validity versus goal analysis — E017 V6 v1',
 'The full geometric goal is position≤8Å AND frame orientation≤35°. Full success additionally requires all original whole-prefix+tail Cα validity tests. Native/alternative target witnesses were retained. Neither a failed finite pool nor a failed classical IK search proves geometric unreachability.',partition,
 'The partition prioritizes full success, then invalid goal hits, then valid goal misses, then neither. Class2 may contain both invalid goal hits and valid misses. Marginal no-goal-hit counts=class1+class3. All candidate metrics, causes and selected indices are retained without replacing failures.',
 f"All-candidate totals: {json.dumps(failure,indent=2)}",
 '## Exact original local and global tests',
 'Adjacent distances must lie in[2.8504200470168195,4.052024993913842]Å. Nonlocal pairs have index separation≥3 and clash at<2Å; maximum2clashes/100residues. Rg/N^(1/3) bounds[1.950656875000882,10.184757892570111]; all frame cross-product norms>1e−8; finite coordinates. A22/24-residue final fragment fails from even one clash. Whole original prefix participates; generated-action spacing summaries alone are insufficient.',
 'Generated distances exp(u)>0, angles πsigmoid(u)∈(0,π), torsions πtanh(u)∈(−π,π) retain original support. Each fresh step explicitly verified support and finiteness without clamping/redraw. The original validity oracle contains no separately calibrated angle/torsion realism threshold. Their means and generated distance ranges are in each candidate row; validity is not redefined by inventing a torsion preference. Changed torsions can cause contacts, but no causal torsion attribution is established.',
 '## Early prefix diagnostic, exploratory',
 'At each saved intermediate coordinate state, rejectability checks: irreversible already-emitted spacing/frame/collision failure, or necessary contour bound(target distance>remaining×max allowed bond+8Å). A single nonlocal clash cannot become admissible for the fixed short final fragment. The bound is only necessary for a valid position hit, never sufficient for orientation/success. No Rg early rejection was used because later extension can change it. All32 complete continuations were still generated; this analysis never selected extra draws or altered the pool.',
 f"Retrospective flagged candidates={failure['early_rule']['flagged']}; flagged full successes={failure['early_rule']['flagged_success']}; potential skipped actor+J1 steps EACH={failure['early_rule']['potential_actor_J1_steps_saved_each']}. Counts by earliest step={failure['early_rule']['first_step_counts']}, reasons={failure['early_rule']['causes']}. Zero false rejection on this pool validates the implemented necessary tests here; it does not create a learned reachability predictor or identify promising valid goals positively.",
 '## Distribution diagnostics and limitations',
 f"Mean conditional proposal KL over all stored candidates={f(out['proposal_KL']['all_candidate_proposal_KL_mean'])}. New1,890draws provide mean distance/angle/torsion Gaussian component KL={out['proposal_KL']['mean_components_distance_angle_torsion']}; step means={out['proposal_KL']['mean_by_step']}. Legacy reused candidates lack per-component KL and are excluded from that decomposition. These are frozen-parent measurements, not retrospective attributions to V5.1 A/C failures. Averaged selected proposal KL is not the KL of the selected distribution.",
 diversity,
 'Diversity is coordinate diversity under a fixed sequence and prefix: pairwise endpoint distance and tail RMSD in the same frame, no alignment. Unique tails rounded1e−6Å are reported. Distinct trajectories may still share the same geometric failure; uniqueness alone is not useful reachability. No sequence-diversity claim or foldability inference is made.'
 ]
 text('VALIDITY_VS_GOAL_ANALYSIS.md','\n\n'.join(validity))
 text('CANDIDATE_SELECTION_COMPARISON.md','\n\n'.join([
 '# Candidate selection comparison — E017 V6 v1',overview,selection,
 'A retains the first candidate at every K. B ignores orientation and hard validity; decreasing position is not a monotonic guarantee of full success. C first enforces the original hard criteria, then uses max-normalized position/orientation error; no valid candidates means abstention and failure, never resampling. C_original_cost minimizes original Huber-position+5sin²(θ/2) among valid candidates and is secondary.',
 'For any valid candidate, max(position/8,orientation/35)≤1 iff both target conditions hold. Every full hit therefore precedes every valid miss under C. C equals the pool-existence upper bound D exactly at all K, verified for all63tasks. There is no learned planner or independent oracle advantage: the known goal/validity criteria are computable from generated coordinates. Continuous original goal cost need not respect the conjunction, which is why C_original_cost can differ.',
 f'PrimaryK32 paired C−B full-success gain={diff(primary)}. The predeclared ≥5pp and positive95%lower-bound claim is {strength}. All K use nested pools; six-K99.166667% endpoint-wise bands are in CANDIDATE_SUCCESS_BY_K.json. No best K was selected from the outcomes.',
 f'Historical20/126 corresponds exactly to the two reused single draws for each task. First-draw A={A}/63; C32={C}/63. Task-paired C32 versus historical two-draw mean={diff(q["paired_C_minus_legacy_two_draw_mean"])}. Different budgets and per-attempt/per-query estimands remain explicit.',
 'Chosen-candidate proposal KL is descriptive only: deterministic selection among sampled candidates changes the law. No selected-policy KL guarantee or promotion is claimed. Stored candidate indices, full failures, seeds and four-point tails permit deterministic re-analysis without another forward pass.'
 ]))
 text('INFERENCE_COST_ANALYSIS.md','\n\n'.join([
 '# Inference cost — E017 V6 v1',costs,rowdetails,
 'Every fresh continuation makes exactly4 actor and4 J1 calls (four generated steps), with no fifth native-BC diagnostic. Total prospective-equivalent32-pool budget:8,064actor+8,064J1passes; actually new sampling:7,560each. Two parity continuations add8each, so this worker made7,568actor+7,568J1calls. Original126cached outcomes historically used630each including126BC calls; they were reused rather than regenerated. Small K are prefixes of the same pool, not new worker launches.',
 'A-alone cost is one continuation/task:252actor+252J1 calls for63queries; repeating A rows at higher K does not increase its required calls. B/C/D share complete K pools for a compute-matched selector comparison. No learned critic or high-level network is invoked. Selection arithmetic operates on scalar saved metrics and has negligible compute compared with inference, but an exact selection-only microbenchmark was not performed.',
 f"Measured worker wall time={resource['elapsed_seconds']:.6f}s; summed new per-candidate timers={by['32']['cost']['new_generation_seconds']:.6f}s; remaining overhead includes model loads/parity, saved row processing, verification, fsync and publication. Per-candidate timers include full goal/validity scoring and CUDA synchronization, exclude row fsync. Historical126-attempt panel wall time={legacy['seconds']:.6f}s (includes BC); per-seed historical time is unavailable. Estimated pooled times add the whole historical panel forK≥2, half forK1. These are amortized retrospective costs, not measured isolated cold-start latencies.",
 'Seconds/full-success-task equals total estimated panel inference time divided by successful queries for that selector; failed/abstained queries still consume time. It is not latency conditioned on a successful task and does not omit failures. Runtime for an unsuccessful panel is undefined, not zero. Real execution computed all32 at every task; lower-K costs are sums of recorded prefix candidate timings and call counts.',
 increments,
 f"Maximum PyTorch allocated/reserved GPU bytes={resource['GPU_peak_allocated_bytes']:,}/{resource['GPU_peak_reserved_bytes']:,}; peak process RSS={resource['max_RSS_KiB']:,}KiB. Initial WSL RAM16,699,813,888bytes, available~14.1GB; mid-run available~11.95GB; swap use stayed1,161,678,848bytes at those two snapshots. Host GUI/NVIDIA baseline memory is separate from PyTorch allocation. One RTX5060 inference worker only; no training/evaluation campaign ran concurrently.",
 f"Resource admission requires2GiB+64MiB D: free and CUDA allocation below6GiB; actual allocation remained far lower. Completion D: free={resource['free_bytes']:,}bytes; final publication free recorded in provenance. No model checkpoint copies, large tensor exports or dataset materialization; candidate JSONL stores only scalar diagnostics and four×3 coordinate tails. All caches/tmp/configured runtime paths on D:. No historical deletion. Necessary early-rule retrospective potential savings={failure['early_rule']['potential_actor_J1_steps_saved_each']}calls EACH, not realized savings or a new benchmark."
 ]))
 text('NEXT_RL_RESEARCH_DECISION.md','\n\n'.join([
 '# Next research decision after E017 V6 — v1',
 f'**Recommend {nextchoice}.** This recommendation follows the recorded finite-pool outcomes; no next phase is implemented.',
 f'1. **Useful valid mass already present?** Full valid candidates occur in{N}/63tasks at32 draws; total candidate full successes={failure["full_success"]}/2016. This measures availability within this existing VALIDATION panel, not unseen-population certification. Zero-of32tasks can still have small nonzero success probability.',
 f'2. **Does best-of-K improve success or proximity?** Position-only B32 succeeds{B}/63; validity/threshold-aware C32 succeeds{C}/63 and exactly equals D. C−B={diff(primary)}. B’s better positional objective must be assessed with orientation/validity and budget. Neither improvement changes the frozen actor.',
 f'3. **Scaling/plateau?** Gains across K are documented above;16→32 adds{inc["new_success_tasks"]}tasks at the cost of4,032additionalactor and4,032J1passes. This is observed diminishing/remaining gain only;32draws do not identify an asymptote or authorize larger K.',
 f'4. **Failure mechanism?** AtK32 the mutually exclusive partition is{q["partition"]}; all-candidate invalid causes={failure["causes"]}. Coordinate diversity and difficulty correlations are measured separately. Do not call no-hit pools geometrically unreachable; valid native/synthetic witnesses exist. Failure can reflect low proposal support, orientation difficulty or invalid stochastic paths.',
 f'5. **Selection competitive?** It is a directly measurable geometric-control baseline at{C}/63query successes, with about{f(q["cost"]["seconds_per_full_success_task_amortized"])}estimated amortizedseconds per successful query including failed-query cost. It needs up to32samples and known geometric goal/validity computation, and does not certify the selection law’s KL. Preserve cost/error/success tradeoffs before operational adoption.',
 '6. **Where might RL help?** These static fixed-H4 pools provide evidence about proposal quality and inference selection, not an advantage of sequential RL over supervised control. If later authorized, safer proposal distributions or amortizing demonstrated valid actions is a more grounded target than adding high-level SAC immediately. A separate controlled comparison would need to show improved success per J1/actor pass without sacrificing physical validity or KL. Existing V5.1 failures warn against stronger unconstrained goal optimization.',
 f'7. **Evidence for reachability-aware planning?** Early necessary checks flag{failure["early_rule"]["flagged"]}candidates with zero full-success flags in this pool, potentially saving{failure["early_rule"]["potential_actor_J1_steps_saved_each"]}forward steps EACH retrospectively. This supports exact early rejection as a future engineering diagnostic, not a learned positive reachability model. No intervention, long-horizon, adaptive branching or calibration study here establishes benefit for a learned critic/high-level planner.',
 'Among the requested alternatives: use the observed inference-time validity filter as the immediate finite-budget research baseline; improve the proposal distribution later if cost or no-hit tasks remain limiting. A learned validity/reachability critic needs independent calibration and a benefit over exact checks; a high-level RL planner lacks direct evidence here. Returning to unconstrained geometric-control training or simply adding updates is contradicted by V5.1’s safeguard failures. No architecture/safety threshold or TEST change is recommended.',
 'No controller promotion: selected-law KL remains unresolved. No inference-time result proves protein foldability/stability. Stop all workers, preserve the frozen parent/evidence, and wait for a separately authorized research decision.'
 ]))
 # All source/input/output hashes are sealed last; no recursive provenance self-hash.
 names=['FINAL_E017_V6_SCIENTIFIC_REPORT.md','CANDIDATE_SUCCESS_BY_K.json','VALIDITY_VS_GOAL_ANALYSIS.md','CANDIDATE_SELECTION_COMPARISON.md','INFERENCE_COST_ANALYSIS.md','NEXT_RL_RESEARCH_DECISION.md']
 inp=dict(cfg['input_SHA256'])
 for p in [R/'PROTOCOL.json',R/'PARITY.json',R/'WORKER_COMPLETED.json',R/'results/candidates.jsonl',Path(__file__),
  V4/'FINAL_E017_V4_MULTISTEP_REPORT.md',V5/'FINAL_E017_V5_SCIENTIFIC_REPORT.md',OLD/'MATCHED_ARM_FINAL_COMPARISON.json']:
  inp[str(p)]=sha(p)
 for p in [R/'source/scripts/e017_v5/study.py',R/'source/scripts/e017_v5/frozen_science_config.json',
  R/'source/src/protein_joint_sac/crisp_v3/environment.py',R/'source/src/protein_joint_sac/crisp_v3/prior.py',
  R/'source/src/protein_joint_sac/crisp_v3/networks.py',R/'source/src/protein_joint_sac/geometry/ca_builder.py',
  V4/'source/scripts/e017_v4/experiment.py']:
  inp[str(p)]=sha(p)
 assert sha(cfg['parent'])==cfg['parent_SHA256']
 # Worker PID must have exited before scientific publication.
 oldpid=read(R/'RECOVERY.json').get('pid');assert oldpid is not None and not Path(f'/proc/{oldpid}').exists()
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
 assert not gpu.strip(),gpu
 save('RECOVERY.json',dict(status='COMPLETED_AND_IDLE',worker_pid=oldpid,worker_exited=True,candidates=2016,no_training=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 save('PROVENANCE_SHA256.json',dict(schema='E017_V6_provenance_v1',created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  generation_commit=source,analysis_commit=analysis,input_SHA256=inp,output_SHA256={n:sha(R/n) for n in names},
  parent_SHA256=cfg['parent_SHA256'],J1_file_SHA256='881457ff0396975c1b20dd0a8e2caac8521e65b7e1cec76a7bb531dbc437b412',
  actor_state_digest=resource['actor_state_digest'],J1_tensor_SHA=resource['J1_tensor_SHA'],GPU_compute_processes=[],worker_exited=True,
  D_free_bytes_before_provenance=free(),training_updates=0,SAC_updates=0,TEST_access=False,AlphaFold_inference=False,deleted_files=0,
  integrity=dict(strict_parent_load=True,exact_historical_parity=True,all2016_unique_candidates=True,all_C_equals_D=True,all_legacy126_exactly_reused=True,frozen_actor_J1_verified=True)))
 print(json.dumps(dict(by_K={K:dict(C=v['selectors']['C']['success_count'],B=v['selectors']['B']['success_count'],valid=v['at_least_one_valid_count']) for K,v in by.items()},primary=primary,files=names,free_bytes=free())),flush=True)

if __name__=='__main__':main()
