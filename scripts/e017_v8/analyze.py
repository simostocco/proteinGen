"""Sealed CPU analysis of the V8 finite planning comparison."""
import collections,datetime,json,math,os,runpy,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v8_constrained_planning_20261023_v1');st=runpy.run_path(str(R/'source/scripts/e017_v8/study.py'));read=st['read'];save=st['save'];sha=st['sha'];select=st['select'];CFG=st['CFG'];KLIST=CFG['K']
def boot(a,g):
 a=np.asarray(a,float);g=np.asarray(g);ids=sorted(set(g));sums=np.array([a[g==x].sum() for x in ids]);counts=np.array([(g==x).sum() for x in ids]);draw=np.random.default_rng(18817).integers(0,len(ids),(20000,len(ids)));v=sums[draw].sum(1)/counts[draw].sum(1);return dict(estimate=float(a.mean()),CI95=np.quantile(v,[.025,.975]).tolist(),CI98_333333_family3=np.quantile(v,[.05/6,1-.05/6]).tolist(),groups=len(ids),observations=len(a),replicates=20000,seed=18817)
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(str(x) for x in row)+' |' for row in rows])
def f(x):return 'NA' if x is None else f'{x:.4f}'
def text(name,value):
 p=R/name;q=p.with_name(p.name+'.part')
 with q.open('w') as fp:fp.write(value+'\n');fp.flush();os.fsync(fp.fileno())
 os.replace(q,p)
def result(data,method,K):
 a=data['algorithms'];r=a[method] if method in ['A','CEM'] else a[method+'_'+str(K)];snap=r['snapshots'][str(K)] if method in ['A','CEM'] else r;pool=r['candidates'][:K] if method in ['A','CEM'] else r['candidates'];idx=snap['selected'];chosen=next(x for x in pool if x['index']==idx) if idx is not None else None
 return pool,chosen,snap['cost'],snap.get('prior_diagnostic_cost',{})
def tails(t,pool):
 arrays=[]
 for r in pool:
  x=np.array(t['native_prefix'],dtype=float)
  for g in r['geometry']:x=st['b']['s']['append'](x,np.array(g))
  arrays.append(x[-4:])
 return np.array(arrays)
def summarize(ds,method,K):
 per=[];div=[]
 for d in ds:
  pool,chosen,cost,audit=result(d,method,K);t=d['task'];bad=collections.Counter(x for r in pool for x in r['validity_failures']);q=dict(task_id=t['task_id'],identity=t['identity'],group=t['protected_group_id'],variant=t['variant'],condition=t['condition'],success=bool(chosen and chosen['success']),valid_returned=chosen is not None,abstention=chosen is None,translation_hit=bool(chosen and chosen['translation_hit']),rotation_hit=bool(chosen and chosen['rotation_hit']),position_A=chosen['position_A'] if chosen else None,orientation_deg=chosen['orientation_deg'] if chosen else None,candidate_count=len(pool),candidate_valid=sum(r['valid'] for r in pool),candidate_goal_hit=sum(r['translation_hit'] and r['rotation_hit'] for r in pool),candidate_full_success=sum(r['success'] for r in pool),candidate_collision_failures=sum('nonlocal_clashes' in r['validity_failures'] for r in pool),failure_criteria=dict(bad),selected_index=chosen['index'] if chosen else None,selected_prior=chosen.get('prior_audit') if chosen else None,cost=cost,audit_cost=audit,root_bounds=d['root_irreparable_or_unreachable'])
  if pool:
   x=tails(t,pool);ix=np.triu_indices(len(x),1);rms=np.sqrt(np.square(x[:,None]-x[None,:]).sum(-1).mean(-1))[ix];q['unique_traces']=len({tuple(np.round(r['geometry'],6).flat) for r in pool});q['pair_tail_RMSD_A']=float(rms.mean()) if len(rms) else None;div.extend(rms.tolist())
  else:q.update(unique_traces=0,pair_tail_RMSD_A=None)
  if method=='BEAM':q['prefix_rejections']=len(d['algorithms']['BEAM_'+str(K)]['rejections']);q['rejection_causes']=dict(collections.Counter(x for r in d['algorithms']['BEAM_'+str(K)]['rejections'] for x in r['reasons']))
  per.append(q)
 groups=[r['group'] for r in per];present=[r for r in per if not r['abstention']];out=dict(tasks=len(per),groups=len(set(groups)),full_success_count=sum(r['success'] for r in per),full_success=boot([r['success'] for r in per],groups),valid_returns=sum(r['valid_returned'] for r in per),abstentions=sum(r['abstention'] for r in per),translation_hit_count=sum(r['translation_hit'] for r in per),rotation_hit_count=sum(r['rotation_hit'] for r in per),position_A_conditional_mean=float(np.mean([r['position_A'] for r in present])) if present else None,orientation_deg_conditional_mean=float(np.mean([r['orientation_deg'] for r in present])) if present else None,candidate_count=sum(r['candidate_count'] for r in per),candidate_valid_count=sum(r['candidate_valid'] for r in per),candidate_collision_failures=sum(r['candidate_collision_failures'] for r in per),mean_pair_tail_RMSD_A=float(np.mean(div)) if div else None,per_task=per)
 for name in ['wall_seconds','cpu_process_seconds','neural_GPU_seconds','synchronized_neural_call_wall_seconds','actor_examples','actor_batches','simulated_transitions','terminal_evaluations','gradient_objectives','latent_gradient_steps']:out[name]=sum(r['cost'].get(name,0) for r in per)
 for name in ['wall_seconds','cpu_process_seconds','neural_GPU_seconds','actor_examples','actor_batches','simulated_transitions']:out['audit_'+name]=sum(r['audit_cost'].get(name,0) for r in per)
 out['end_to_end_warm_seconds']=out['wall_seconds']+out['audit_wall_seconds'];out['seconds_per_success']=out['wall_seconds']/out['full_success_count'] if out['full_success_count'] else None
 return out
def main():
 torch.set_num_threads(1);st['check']();assert read(R/'VALIDATION_COMPLETED.json')['complete'];files=sorted((R/'results').glob('VALIDATION_[0-9][0-9][0-9]_*.json'));assert len(files)==128;allrows=[read(p) for p in files];conditions={c:sorted([d for d in allrows if d['task']['condition']==c],key=lambda d:d['task']['task_id']) for c in CFG['conditions']};metrics={};paired={};matched={};methods=['A','BEAM','CEM','IK']
 for condition,ds in conditions.items():
  assert len(ds)==64 and len({d['task']['protected_group_id'] for d in ds})==32;metrics[condition]={};paired[condition]={};matched[condition]={}
  for K in KLIST:
   metrics[condition][str(K)]={method:summarize(ds,method,K) for method in methods};base=metrics[condition][str(K)]['A']['per_task'];groups=[r['group'] for r in base];paired[condition][str(K)]={};matched[condition][str(K)]={}
   for method in methods[1:]:
    rr=metrics[condition][str(K)][method]['per_task'];paired[condition][str(K)][method]=boot([int(a['success'])-int(b['success']) for a,b in zip(rr,base)],groups)
    # Use the largest available baseline prefix whose actual resource use
    # does not exceed each planner's resource use, without extrapolation.
    m={}
    for axis in ['simulated_transitions','actor_examples']:
     wins=[];indices=[]
     for d,r in zip(ds,rr):
      available=[n for n in KLIST if d['algorithms']['A']['snapshots'][str(n)]['cost'][axis]<=r['cost'][axis]];n=max(available) if available else None;indices.append(n)
      if n is not None:wins.append(bool(result(d,'A',n)[1] and result(d,'A',n)[1]['success']))
      else:wins.append(None)
     ii=[j for j,z in enumerate(wins) if z is not None];m[axis]=dict(comparable_tasks=len(ii),baseline_prefixes=indices,paired_gain=boot([int(rr[j]['success'])-int(wins[j]) for j in ii],[groups[j] for j in ii]) if ii else None,scope='nearest lower available baseline8/16/32/64, not exact identical resource/FLOPs; no extrapolation below8; IK has zero planner actorcalls')
    matched[condition][str(K)][method]=m
   print('ANALYZE',condition,K,flush=True)
 decisions={}
 for method in methods[1:]:
  fail=[]
  for c in CFG['conditions']:
   q=paired[c]['64'][method]
   if q['estimate']<.1:fail.append(c+': gain below10pp')
   if q['CI98_333333_family3'][0]<=0:fail.append(c+': familywise interval does not support gain')
  decisions[method]=dict(confirmed_planning_improvement=not fail,reasons=fail,not_neural_policy_promotion=True)
 # Failure decomposition is conditional on sampled/optimized pools except
 # for the mathematically necessary root bounds and native valid witnesses.
 unsolved=[]
 for c,ds in conditions.items():
  for d in ds:
   t=d['task'];out={method:bool(result(d,method,64)[1] and result(d,method,64)[1]['success']) for method in methods};pool=[r for method in methods for r in result(d,method,64)[0]];root=d['root_irreparable_or_unreachable'];reason='solved' if any(out.values()) else 'provable_root_obstruction' if root else 'sampled_goal_hits_invalid' if any(r['translation_hit'] and r['rotation_hit'] for r in pool) else 'finite_search_goal_miss';unsolved.append(dict(task_id=t['task_id'],identity=t['identity'],group=t['protected_group_id'],variant=t['variant'],condition=c,methods=out,root_bounds=root,category=reason,native_witness_proves_feasible=c=='NATIVE',no_method_solved=not any(out.values())))
 prior={}
 for c in CFG['conditions']:
  prior[c]={}
  for method in methods:
   rr=metrics[c]['64'][method]['per_task'];p=[r['selected_prior'] for r in rr if r['selected_prior']];prior[c][method]=dict(selected_paths=len(p),mean_parent_logp_latent_per_component=float(np.mean([a['parent_logp_latent']/12 for a in p])) if p else None,mean_max_abs_standardized_latent=float(np.mean([a['max_abs_standardized_latent'] for a in p])) if p else None,fraction_selected_with_any_abs_z_over3=float(np.mean([a['max_abs_standardized_latent']>3 for a in p])) if p else None,mean_conditional_proposal_KL=float(np.mean([a['conditional_proposal_KL'] for a in p])) if p else None)
  cems=[r for d in conditions[c] for r in d['algorithms']['CEM']['candidates'] if 'search_logq_latent' in r];bytask=[]
  for d in conditions[c]:
   rs=[r for r in d['algorithms']['CEM']['candidates'] if 'search_logq_latent' in r];bytask.append(float(np.mean([(r['search_logq_latent']-r['prior_audit']['parent_logp_latent'])/12 for r in rs])))
  prior[c]['CEM_unselected_generation_MC_KL_per_component']=boot(bytask,[d['task']['protected_group_id'] for d in conditions[c]])
 save('MATCHED_COMPUTE_COMPARISONS.json',dict(config=CFG,source=source(),metrics=metrics,paired=paired,nearest_resource_matched=matched,decisions=decisions,prior=prior,unsolved=unsolved,confidence='20k paired protein-cluster bootstrap; nominal95 and98.333333 family3; conjunction across native/generated for each planning claim; conditional on fixed task/candidate pools'))
 publish(metrics,paired,matched,decisions,prior,unsolved,allrows,files)
def source():return read(R/'PROTOCOL.json')['source']
def publish(metrics,paired,matched,decisions,prior,unsolved,allrows,files):
 protocol=read(R/'PROTOCOL.json');audit=read(R/'INPUT_AUDIT.json');census=read(R/'TASK_CENSUS.json');worker=read(R/'VALIDATION_COMPLETED.json');summary=table(['Condition /64 tasks','Budget','Method','Full success','Valid returns','Abstain','Translation hits','Rotation hits','Mean position A*','Mean orientation deg*'],[[c,K,m,str(v['full_success_count'])+'/64',v['valid_returns'],v['abstentions'],v['translation_hit_count'],v['rotation_hit_count'],f(v['position_A_conditional_mean']),f(v['orientation_deg_conditional_mean'])] for c,kk in metrics.items() for K,mm in kk.items() for m,v in mm.items()])
 costtable=table(['Condition','Cap','Method','Actual terminal evals','Transitions','Actor examples/batches','Planner wall s','CPU process s','Neural CUDA event-span s','Latent gradient steps','Prior audit wall s'],[[c,K,m,v['terminal_evaluations'],v['simulated_transitions'],str(v['actor_examples'])+'/'+str(v['actor_batches']),f(v['wall_seconds']),f(v['cpu_process_seconds']),f(v['neural_GPU_seconds']),v['latent_gradient_steps'],f(v['audit_wall_seconds'])] for c,kk in metrics.items() for K,mm in kk.items() for m,v in mm.items()])
 primary=table(['Planner','Native gain [family98.333%]','Generated gain [family98.333%]','Confirmed under frozen rule','Failures'],[[m,f(paired['NATIVE']['64'][m]['estimate'])+' '+str(paired['NATIVE']['64'][m]['CI98_333333_family3']),f(paired['GENERATED']['64'][m]['estimate'])+' '+str(paired['GENERATED']['64'][m]['CI98_333333_family3']),v['confirmed_planning_improvement'],'; '.join(v['reasons']) or 'none'] for m,v in decisions.items()])
 ptable=table(['Condition','Method','Selected paths','Parent logp/component','Mean max abs standardized latent','Any abs z>3 fraction','Conditional proposal KL'],[[c,m,v['selected_paths'],f(v['mean_parent_logp_latent_per_component']),f(v['mean_max_abs_standardized_latent']),f(v['fraction_selected_with_any_abs_z_over3']),f(v['mean_conditional_proposal_KL'])] for c,mm in prior.items() for m,v in mm.items() if m in ['A','BEAM','CEM','IK']])
 text('PLANNING_METHODS_AND_VALIDITY.md','\n\n'.join(['# E017 V8 planning methods and validity — v1',
 'All methods use the exact V6 parent weights (V5 D+250) with the original H4 native-AA conditioning and unchanged physical transforms/support. A is the V6 joint-goal validity-first selector: discard invalid terminals, minimize max(position/8,orientation/35), then original componentwise Huber(position/3.8)+5sin²(orientation/2), then index. It returns a full success whenever any full success is in its pool. Empty feasible pools abstain and remain failures.',
 'All new methods use FP32 neural inference and FP64 geometry, temperature1, eval/no-grad for every network. This is the same mathematical autoregressive law; historical BF16 samples are not claimed byte-identical or pooled with the new panel. TRAIN preflight checks the serial FP32 reference against batched sampling, exact FK/reward semantics and unchanged parent/J1 fingerprints.',
 'Beam: width2, up to4candidate prefixes per nonterminal stage, K/4full sampled lookaheads per stage; atK8 only2prefixes. Rank current partial states using their actually sampled exact terminal completions, never hidden future coordinates. Retain a diverse second visible prefix when RMSD>=0.1A is available. Retain every scored full completion as an eligible final candidate. Stage3counts every complete extension, including invalid/missing targets, against the terminal cap. Rejected partial proposals and necessary-bound failures are journaled; no free terminal refill.',
 'Safe rejection: already invalid adjacent spacing and degenerate/nonfinite visible geometry cannot be repaired by appending. Existing clash count must be normalized by FINAL length; reject only when count*100/finalN exceeds the original2.0limit. The final Rg lower bound is sqrt(visible scatter sum/finalN)/finalN^(1/3); only an inevitable violation of the original upper bound can reject. A necessary position contour bound uses remaining*original maxbond+8A. Current Rg and low-Rg prefixes are never simply pruned. No orientation heuristic is used as a hard partial feasibility test.',
 'CEM: eight parent trajectories initialize an absolute twelve-dimensional action-latent Gaussian. Each subsequent generation has8draws. Two valid elites update mean/diagonal SD with0.7blend and a covariance floor0.05times the initial mean conditional actorSD. If no valid elite exists, keep the previous search law. All populations and failures count; no action clamp, projection or invalid replacement. This changes the proposal law and is not sampling from the original actor after generation0.',
 'IK: bounded trace version of original V5 equations, two deterministic starts, Adam0.08 and clip10; K/2evaluated iterates per start means K/2−1updates per start. Goal-Huber/orientation/local-clash/distance/native-like bond prior and latent penalty are unchanged. The original parameterization searches the valid distance interior[low+0.01,high−0.01], angle(0.03,pi−0.03), signed-torsion tanh. This is an explicit search subset, not a changed environment rule or a projected actor draw. It does not have an actor-sampling likelihood. Loss FK and exact terminal assessment share the same geometry; all latent gradients are counted. Preflight matches the original IK endpoint after3updates/start.',
 'Candidate caps8/16/32/64mean maximum complete exact-terminal candidates, not equalFLOPs. A/CEM pools are nested; beam and IK are separate bounded searches per cap, so their effort curves need not be monotone. CEM’s first8are reused fromA; standalone CEM costs include their measured generation. Prefix generation is shared across paired goals and separately counted. Prior audits are additional explicitly measured work.',
 'Workers receive only visible prefix, fixed AA identities, target pose and immutable source references. Native/witness future actions are stored in a separate provenance manifest and never parsed by planners. Witness validity is used only in preparation. Dataset hashes are checked without supplying those witness values to algorithms.'
 ]))
 text('IK_AND_CEM_COMPARISON.md','\n\n'.join(['# IK and CEM comparison — V8',summary,costtable,ptable,
 'The original historical IK results used different tasks/horizons and are context only. These IK/CEM measurements are newly matched to the same V8 H4 task/goal/prefix definitions as A and beam. Their optimized actions are not independent samples from the frozen actor. CEM current Gaussian logq and parent autoregressive latent logp yield finite-sample population KL diagnostics before final selection. Those are not selected-search-law KL values. Deterministic IK outputs have no continuous stochastic density of their own; a parent density at the chosen point is not its sampling probability.',
 'Prior diagnostics are in MATCHED_COMPUTE_COMPARISONS.json. High standardized deviations or poor parent density indicate that kinematically valid goal solutions leave the typical frozen proposal region. Unchanged neural weights and original hard C-alpha validity alone do not establish preservation of the learned geometric prior or biochemical plausibility.'
 ]))
 text('NATIVE_VS_GENERATED_PREFIX_RESULTS.md','\n\n'.join(['# Native versus generated prefixes — V8',summary,primary,
 'Each condition contains64tasks from32protected protein groups, one observed-native and one valid synthetic alternative goal per protein. Generated conditions reuse the same groups and targets; they are paired stress conditions, not32more independent biological samples. A frozen-parent H4rollout from16observed residues toward the native20residue pose creates one shared prefix per protein, then the original target at24remains fixed. Failures are retained. The native witness does not prove that the target remains feasible after prefix replacement.',
 'Every abstention remains a full-success failure. Asterisked error means condition on returned valid candidates; denominators and abstention counts remain explicit. Candidate validity/collision rates, goal hits, selected outcomes and diversity are separate JSON fields. A valid return is only C-alpha rule compliance, not certified protein foldability.'
 ]))
 unresolved=[r for r in unsolved if r['no_method_solved']];counts=collections.Counter((r['condition'],r['category']) for r in unsolved);rows=table(['Condition','Task','Variant','Category','Root bounds'],[[r['condition'],r['task_id'],r['variant'],r['category'],','.join(r['root_bounds']) or 'none'] for r in unresolved]);solved_beyond={c:sum(not r['methods']['A'] and any(r['methods'][m] for m in ['BEAM','CEM','IK']) for r in unsolved if r['condition']==c) for c in CFG['conditions']}
 text('UNSOLVED_TASK_ANALYSIS.md','\n\n'.join(['# Unsolved task analysis — V8',str(dict(counts)),rows,
 'Native goals have valid witnesses established before planning: an unsolved native query is finite search/proposal failure under this budget, not proof of kinematic impossibility. Generated conditions can contain irreversible existing spacing/clash/frame violations or fail the necessary position-contour bound. Those explicit bounds prove no valid target completion under the fixed visible prefix and constraints; otherwise finite search failure cannot prove infeasibility.',
 'Tasks solved by another planner after Best-of64misses: '+str(solved_beyond)+'. Such cases show limited coverage/efficiency of this finite frozen proposal pool. They do not prove zero actor probability. Pool-level goal hits that are all invalid distinguish sampled validity conflict from pools containing valid goal misses. JSON retains every query and method outcome so goal variants are not counted as independent proteins.'
 ]))
 confirmed=[m for m,v in decisions.items() if v['confirmed_planning_improvement']];decision='Confirmed finite-planning improvement: '+', '.join(confirmed) if confirmed else 'No planner passed the complete frozen improvement criterion';best={c:max(metrics[c]['64'],key=lambda m:metrics[c]['64'][m]['full_success_count']) for c in CFG['conditions']}
 main=['# E017 V8 final scientific report — v1','**'+decision+'.** No network training or SAC occurred.',
 f"Source `{protocol['source']}`; exact V6 parent `{audit['actor_SHA256']}` (V5 D+250,17,438,114parameters). J1 `{audit['J1_SHA256']}` (12,351,746parameters) remained frozen. Critics and all other networks were not constructed, trained or modified. Strict parent restoration, frozen fingerprints and original IK-equation parity passed.",
 f"Fresh protected VALIDATION: 32 identities/groups, 64 paired native/synthetic goals, native and generated-prefix conditions. Excluded {census['excluded_previous_E017_groups']} previously prepared E017 validation groups. All goals, methods, candidate caps and acceptance rules were sealed before prospective planner evaluation. TRAIN-only development used 4 tasks for bounded execution/profile, with no neural fitting or validation-based retuning. Exact observations, source hashes, witness checks and split exclusions are in TASK_CENSUS/manifests. TEST stayed sealed.",
 '## Success and statistical comparison',summary,primary,
 'Primary confirmation requires at least10ppgain over A atcap64and a paired98.333333%lower bound above0in BOTH conditions, controlling the family of3planner claims by Bonferroni plus an intersection-union transfer requirement. Secondary budget/stratum95%intervals are descriptive.20,000protein-cluster replicates, seed18817, keep both goals per protein. A zero-width conditional interval does not prove population certainty. No training-seed or independent search-pool replication was performed.',
 '## Computation and validity',costtable,
 'A candidate cap is not a FLOP budget. Actor examples and batch forwards, actual simulated transitions, complete exact evaluations, IK gradient work, host process CPU time, total wall time and neural CUDA-event spans are retained. CUDA events bracket neural distribution calls; spans include dispatch gaps, are not exclusive GPU kernel-busy time, and do not separately isolate geometry kernels. Total planner wall time includes geometry, scoring and every discarded proposal. The additional synchronized neural-call wall timer is retained separately. Nearest lower available baseline resource frontiers are provided per task for transition and actor-example counts, without extrapolation below 8 candidates. They are explicitly approximate resource matching, not identical compute. Method setup/shared model loading is excluded from warm planner timers; worker totals include setup, verification, serialization and all diagnostic work.',
 'All search outputs use the original V6 validity-first joint selector. Invalid samples remain in candidate counts but are never returned as successes. Beam pruning can save terminal/transition work only before completion; completed rejected proposals still count. CEM/IK operate outside the actor sampling law. '+str(best)+' are the highest cap64success point estimates by condition; this descriptive choice is not a separately selected scientific model.',
 '## Geometric prior and unresolved cases',ptable,
 'Frozen proposal conditional KL against J1 at selected states is a diagnostic of that proposal, not the selected planner law. CEM adapts a latent Gaussian; IK is deterministic optimization. Neither kinematic success nor low positional error certifies a KL-compliant autoregressive generator or protein folding/stability. Failed original V5/V5.1/V7/V7.1/V7.2model safeguards are not promoted by planning results.',
 '## RL relevance and next decision',
 'This is a fully observed deterministic H4control problem with known transitions and an exactly computable terminal goal/validity predicate. Reinforcement learning is not shown necessary: direct search can use those quantities without a learned value function. Even a bounded planning failure would not establish an RL-specific capability. Reported replanning is model-based computation, not learned causal or biological control.',
 'Meaningful future high-level RL would require a separately specified environment with delayed subgoal choices, sequential constraints, incomplete observations, uncertain effects or corrective interventions learned from feedback. Longer horizons or those features are future options, not V8additions. The next recommendation must use measured success/cost and prior deviations; do not launch more critic updates or another stage automatically.',
 '## Resources and reproducibility',
 f"V8worker seconds {worker['elapsed_seconds']:.3f}; neuralGPUseconds {worker['neural_GPU_seconds']:.3f}; actor examples/batches {worker['actor_examples']}/{worker['actor_batches']}; peakGPU {worker['peak_GPU']:,}bytes; peakRSS {worker['peak_RSS_KiB']:,}KiB. Compact task/condition blocks and SHAcommit records allow deterministic seeded recovery. No optimizer checkpoint is needed for frozen networks; per-task IKsearch is rerunnable and its complete evaluated traces are saved. A dangling task JSON without its SHAcommit fails closed for review. No historical data/checkpoints were deleted. D:headroom remains >=2GiB; source/logs/caches/temp/evidence stay on D:. All workers exit after final publication."
 ]
 text('FINAL_E017_V8_SCIENTIFIC_REPORT.md','\n\n'.join(main))
 text('NEXT_CRISP_RESEARCH_DECISION.md','\n\n'.join(['# Next CRISP research decision — V8','**'+decision+'.**',primary,
 'Compare practical warm runtime and all computation counts, not only terminal cap. Retain the strongest measured validity-aware model-based baseline for this static H4problem, while reporting actor-prior departures of optimized solutions. A planning success is not neural policy promotion.',
 'If direct search solves H4efficiently, there is no demonstrated need for SAC to solve this exact task. If it does not, distinguish finite proposal coverage, irreversible generated-prefix obstructions and search inefficiency before proposing learning. Do not prescribe another critic campaign from RMSE alone.',
 'A future RLstudy must introduce a prospectively justified source of delayed consequential decisions—subgoal selection, sequential goal/constraint revelation, incomplete state information or uncertain feedback-driven corrections—while retaining the model-based baseline. Those environment changes, longer horizons and all learning remain unauthorized here.',
 'Stop after V8. Preserve original J1/actor/checkpoints, TEST boundaries and all evidence; await the next decision.'
 ]))
 plots(metrics);names=['FINAL_E017_V8_SCIENTIFIC_REPORT.md','PLANNING_METHODS_AND_VALIDITY.md','MATCHED_COMPUTE_COMPARISONS.json','SUCCESS_VS_COMPUTATION.png','NATIVE_VS_GENERATED_PREFIX_RESULTS.md','IK_AND_CEM_COMPARISON.md','UNSOLVED_TASK_ANALYSIS.md','NEXT_CRISP_RESEARCH_DECISION.md'];resource=dict(D_free_bytes=shutil.disk_usage(R).free,required_headroom_bytes=2*2**30,validation_worker=worker,TRAIN_worker=read(R/'TRAIN_COMPLETED.json'),preflight=read(R/'PREFLIGHT.json'),actual_task_blocks=128,no_neural_updates=True,all_artifacts_on_D=True,deleted_historical_artifacts=0);save('RESOURCE_AND_RECOVERY_REPORT.json',resource);names.append('RESOURCE_AND_RECOVERY_REPORT.json')
 inputs={**protocol['input_SHA256'],str(R/'PROTOCOL.json'):sha(R/'PROTOCOL.json')}
 for p in files:inputs[str(p)]=sha(p);inputs[str(p.with_suffix('.sha.json'))]=sha(p.with_suffix('.sha.json'))
 for p,h in protocol['input_SHA256'].items():assert sha(p)==h,p
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
 save('PROVENANCE_SHA256.json',dict(schema='E017_V8_FINAL_v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=protocol['source'],input_SHA256=inputs,output_SHA256={n:sha(R/n) for n in names},actor_updates=0,J1_updates=0,critic_updates=0,SAC_updates=0,TEST=False,AlphaFold=False,deleted_historical_artifacts=0,GPU_compute_processes=[],all_artifacts_on_D=True,self_hash_excluded=True))
 print('PUBLICATION_COMPLETE',decision,flush=True)
def plots(metrics):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
 for row,c in enumerate(CFG['conditions']):
  for method in ['A','BEAM','CEM','IK']:
   vals=[metrics[c][str(K)][method] for K in KLIST];y=[z['full_success']['estimate'] for z in vals];axes[row,0].plot(KLIST,y,marker='o',label=method);axes[row,1].plot([z['simulated_transitions']/64 for z in vals],y,marker='o',label=method);axes[row,2].plot([z['wall_seconds']/64 for z in vals],y,marker='o',label=method)
  for col,label in enumerate(['terminal candidate cap','actual simulated transitions/query','warm planner seconds/query']):axes[row,col].set_xlabel(label);axes[row,col].set_ylabel(c+' full valid success');axes[row,col].set_ylim(0,1.02);axes[row,col].legend(fontsize=8);axes[row,col].grid(alpha=.2)
 fig.savefig(R/'SUCCESS_VS_COMPUTATION.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
