"""CPU-only, identity-clustered analysis of sealed V9.2 inference evidence."""
import collections,hashlib,itertools,json,math,runpy,shutil,subprocess,time
from pathlib import Path
import numpy as np
from scipy.stats import beta
R=Path('/mnt/d/Simone/proteinGen_data/e017_v92_prefix_feasibility_20261024_v1');m=runpy.run_path(str(R/'source/scripts/e017_v92/common.py'));globals().update({k:m[k] for k in ['OLD','V9','p','c','AA','SPEC','read','save','sha','prefix_status','motif_info','reconstruct']});START=time.perf_counter()
def cp(k,n):return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]
def boot(values):
 a=np.asarray(values,float);rng=np.random.default_rng(19917);out=a[rng.integers(0,len(a),(10000,len(a)))].mean(1);return dict(mean=float(a.mean()),CI95=list(map(float,np.quantile(out,[.025,.975]))),clusters=len(a),unit='protein/goal identity')
def flags(row,goal):
 v=row['metrics'];mi=motif_info(row['sequence'],goal);f=v['geometry']['failures'];return dict(complete=v['complete'],canonical=not set(row['sequence'])-set(AA),motif_identity_global=mi['identity_anywhere'],motif_identity_region=mi['identity_in_region'],motif_order_region=mi['ordered_subsequence_in_region'],motif_contiguous_region=mi['original_contiguous_in_region'],exposed=v.get('exposed_mean',-1)>=goal['exposure_min']-1e-10,buried=v.get('buried_mean',2)<=goal['burial_max']+1e-10,contrast=v.get('contrast',-1)>=goal['contrast_min']-1e-10,local_geometry='spacing' not in f and 'degenerate_frames' not in f and v['geometry']['finite'],nonlocal_clash='nonlocal_clashes' not in f,global_Rg='Rg' not in f and v['geometry']['finite'],packing=v['packing_valid'],sequence_quality=v['sequence_valid'],backbone=v['backbone_valid'],hard=v['hard_valid'],full=v['success'])
def collect(panel,pol,method):
 if panel=='CONFIRM' and pol=='C' and method=='U':paths=sorted((OLD/'results').glob('CONFIRM_C_FIXED_[0-9][0-9][0-9].json'))
 else:paths=sorted((R/'results').glob(f'{panel}_{pol}_{method}_[0-9][0-9][0-9].json'))
 groups=[]
 for path in paths:
  assert sha(path)==read(path.with_suffix('.sha.json'))['sha256'];obj=read(path);obj['source']=str(path);groups.append(obj)
 return groups

def summarize(groups):
 rows=[(r,g['goal']) for g in groups for r in g['records']];n=len(rows);f=[flags(r,g) for r,g in rows];marg=collections.Counter();ints=collections.Counter();patterns=collections.Counter();coords=[];sims=[];rms=[];first=collections.Counter();causes=collections.Counter();hazard=collections.Counter();stages={str(k):dict(observed=0,valid=0) for k in [16,32,48,64]};proposal_count=0;rejected=0;finite_repairs=0;by_goal={};times=[];first_records=[];terminations=collections.Counter();builder_evaluations=0;strata=collections.defaultdict(lambda:dict(n=0,invalid=0));actor_calls=collections.Counter()
 for group in groups:
  gid=group['goal']['goal_id'];by_goal[gid]=[];times.append(group['records'][0]['elapsed_group_seconds']);actor_calls.update(group.get('calls_delta',{}))
  for row in group['records']:by_goal[gid].append(flags(row,group['goal']))
 for (row,goal),ff in zip(rows,f):
  if row.get('termination'):terminations[row['termination']]+=1
  builder_evaluations+=len(row['geometry'])+sum(not any(z in q['reasons'] for z in ['existing_prefix_already_invalid','invalid_or_nonfinite_action','disallowed_policy_action_support']) for q in row.get('proposals',[]))
  marg.update({k:int(v) for k,v in ff.items()});regional=ff['exposed'] and ff['buried'] and ff['contrast'];ints['regional_properties']+=regional;ints['hard_and_regional']+=ff['hard'] and regional;ints['motif_and_hard']+=ff['motif_contiguous_region'] and ff['hard'];ints['motif_and_hard_and_exposed']+=ff['motif_contiguous_region'] and ff['hard'] and ff['exposed'];ints['motif_and_hard_and_exposed_buried']+=ff['motif_contiguous_region'] and ff['hard'] and ff['exposed'] and ff['buried'];ints['full']+=ff['full'];primary=['complete','motif_contiguous_region','exposed','buried','contrast','local_geometry','nonlocal_clash','global_Rg','packing','sequence_quality'];patterns[' + '.join(k for k in primary if not ff[k]) or 'NONE']+=1
  state=c['BuildState'].empty(64);earliest=None
  for t,(aa,geom) in enumerate(zip(row['sequence'],row['geometry'])):
   state=m['append'](state,aa,geom);ps=prefix_status(state)
   if earliest is None:
    hazard[t+1]+=1
    if not ps['valid']:earliest=t+1;first[earliest]+=1;causes.update(ps['irreversible_failures']);first_records.append(dict(goal_id=goal['goal_id'],candidate_index=row['candidate_index'],residue=earliest,causes=ps['irreversible_failures']))
   if state.t in [16,32,48,64]:stages[str(state.t)]['observed']+=1;stages[str(state.t)]['valid']+=ps['valid']
  calc=c['score'](row['sequence'],state.ca.numpy(),goal,SPEC);assert calc==row['metrics'];coords.append(state.ca.numpy());proposal_count+=len(row.get('proposals',row['geometry']));rejected+=sum(not z['accepted'] for z in row.get('proposals',[]) if row.get('method') in ['V','VM']);finite_repairs+=row.get('repairs',0);kind='exposed_before_buried' if goal['exposed'][0]<goal['buried'][0] else 'buried_before_exposed';strata[kind]['n']+=1;strata[kind]['invalid']+=earliest is not None
 for group in groups:
  ix=[i for i,(_,gg) in enumerate(rows) if gg['goal_id']==group['goal']['goal_id']]
  for i,j in itertools.combinations(ix,2):
   a,b=rows[i][0]['sequence'],rows[j][0]['sequence']
   if len(a)==64 and len(b)==64:sims.append(sum(x==y for x,y in zip(a,b))/64)
   if coords[i].shape==(64,3) and coords[j].shape==(64,3):
    X=coords[i]-coords[i].mean(0);Y=coords[j]-coords[j].mean(0);u,_,vt=np.linalg.svd(X.T@Y);rot=u@np.diag([1,1,np.linalg.det(u@vt)])@vt;rms.append(float(np.sqrt(np.square(X@rot-Y).sum()/64)))
 k1=sum(g['records'][0]['metrics']['success'] for g in groups);k8=sum(any(r['metrics']['success'] for r in g['records']) for g in groups);seqs=[r['sequence'] for r,_ in rows];masklogs=[-math.log(z) for r,_ in rows for z in r.get('AA_mask_masses',[])];calls=dict(actor_calls)
 if not calls:calls=dict(J1_batches=64*len(groups),J1_states=sum(len(r['geometry']) for r,_ in rows),actor_batches=64*len(groups),actor_states=sum(len(r['geometry']) for r,_ in rows),derivation='historical complete fixed8 batches64; not new computation')
 return dict(goals=len(groups),trajectories=n,marginals=dict(marg),intersections=dict(ints),K1=dict(success=k1,total=len(groups),CI95=cp(k1,len(groups))),K8=dict(success=k8,total=len(groups),CI95=cp(k8,len(groups))),hard_rate_cluster_CI=boot([np.mean([a['hard'] for a in v]) for v in by_goal.values()]),full_rate_cluster_CI=boot([np.mean([a['full'] for a in v]) for v in by_goal.values()]),top_violation_patterns=patterns.most_common(10),first_failure_histogram=dict(first),first_failure_causes=dict(causes),first_failure_records=first_records,hazard={str(t):dict(at_risk=hazard[t],failures=first[t],hazard=first[t]/hazard[t] if hazard[t] else None) for t in range(1,65)},prefix_stages=stages,goal_type_failure=strata,proposal_count=proposal_count,actual_builder_evaluations_during_generation=builder_evaluations,terminal_score_evaluations=n,terminations=dict(terminations),diversity_complete_pair_count=len(sims),actually_rejected_proposals=rejected,rejection_fraction=rejected/proposal_count if proposal_count else None,finite_grid_repairs=finite_repairs,all_unique_sequences=len(set(seqs)),unique_sequence_fraction=len(set(seqs))/n,pairwise_AA_identity=float(np.mean(sims)),pairwise_CA_Kabsch_RMSD=float(np.mean(rms)) if rms else None,mean_AA_entropy=float(np.mean([r['metrics']['sequence']['entropy_bits'] for r,_ in rows])),property_mean_defined_trajectories=sum('exposed_mean' in r['metrics'] for r,_ in rows),mean_exposed=float(np.mean([r['metrics']['exposed_mean'] for r,_ in rows if 'exposed_mean' in r['metrics']])),mean_buried=float(np.mean([r['metrics']['buried_mean'] for r,_ in rows if 'buried_mean' in r['metrics']])),mean_contrast=float(np.mean([r['metrics']['contrast'] for r,_ in rows if 'contrast' in r['metrics']])),proposal_AA_KL_to_J1=float(np.mean([z for r,_ in rows for z in r.get('AA_KL',[])])),proposal_geometry_KL_per_active_to_J1=float(np.mean([z for r,_ in rows for z in r.get('geometry_KL_per_active',[])])),AA_mask_exact_conditional_KL_to_actor_mean=float(np.mean(masklogs)) if masklogs else None,accepted_policy_KL=None if any(r.get('method') in ['V','VM'] for r,_ in rows) else 'ordinary or analytically normalized AA mask; see density discussion',generation_seconds=sum(times),seconds_per_candidate=sum(times)/n,seconds_per_successful_candidate=sum(times)/marg['full'] if marg['full'] else None,calls=calls,_goal_flags=by_goal)

def paired(a,b,key):
 assert set(a['_goal_flags'])==set(b['_goal_flags']);vals=[]
 for gid in a['_goal_flags']:vals.append(np.mean([q[key] for q in b['_goal_flags'][gid]])-np.mean([q[key] for q in a['_goal_flags'][gid]]))
 return boot(vals)
def paired_goal_endpoint(ag,bg,K):
 assert [g['goal']['goal_id'] for g in ag]==[g['goal']['goal_id'] for g in bg]
 return boot([int(any(r['metrics']['success'] for r in b['records'][:K]))-int(any(r['metrics']['success'] for r in a['records'][:K])) for a,b in zip(ag,bg)])
def step_analysis():
 out={};raw=[]
 for pol in ['A','B','C','D100']:
  obj=read(R/f'ONE_STEP_{pol}.json');rows=obj['records'];raw+=rows;table={}
  for condition in ['own_generated','common_C_generated','native_TRAIN_control']:
   for t in [8,16,32,48,'ALL']:
    sub=[r for r in rows if r['condition']==condition and (t=='ALL' or r['t']==t)];trials=[q for r in sub for q in r['trials']];valid=sum(q['valid'] for q in trials);reasons=collections.Counter(z for q in trials for z in q['reasons']);ids=sorted({r['identity'] for r in sub});rates=[np.mean([q['valid'] for r in sub if r['identity']==identity for q in r['trials']]) for identity in ids]
    table[condition+'_'+str(t)]=dict(prefixes=len(sub),trials=len(trials),valid=valid,rate=valid/len(trials) if trials else None,identity_CI=boot(rates) if rates else None,causes=dict(reasons))
  generated={};native={}
  for r in rows:
   (native if r['condition']=='native_TRAIN_control' else generated if r['condition']=='own_generated' else {})[(r['identity'],r['t'])]=np.mean([q['valid'] for q in r['trials']])
  diffs=collections.defaultdict(list)
  for key in set(generated)&set(native):diffs[key[0]].append(native[key]-generated[key])
  out[pol]=dict(strata=table,omitted=obj['omitted'],native_minus_generated_clustered=boot([np.mean(x) for x in diffs.values()]) if diffs else None)
 extra={}
 for field in ['goal_type','motif_region']:
  for val in sorted({str(r[field]) for r in raw}):
   sub=[q for r in raw if str(r[field])==val for q in r['trials']];extra[field+'_'+val]=dict(n=len(sub),valid=sum(q['valid'] for q in sub))
 aa=collections.defaultdict(lambda:dict(n=0,valid=0));geo=collections.defaultdict(list)
 for r in raw:
  for q in r['trials']:
   aa[q['aa']]['n']+=1;aa[q['aa']]['valid']+=q['valid'];geo['valid' if q['valid'] else 'invalid'].append(q['geometry'])
 components={k:dict(n=len(v),mean=np.asarray(v).mean(0).tolist(),SD=np.asarray(v).std(0).tolist()) for k,v in geo.items() if v};save('ONE_STEP_SUMMARY.json',dict(policies=out,goal_motif_strata=extra,AA_strata=dict(aa),geometry_components=components,conditional_AA_probes=[dict(policy=r['policy'],identity=r['identity'],condition=r['condition'],t=r['t'],probes=r['forced_motif_AA']) for r in raw]));return out

def future_analysis():
 obj=read(R/'FUTURE_VIABILITY.json');records=obj['records'];table=[];comparisons=[]
 for selection in obj['prefixes']:
  gid=selection['goal_id'];actions=sorted({q['action_index'] for q in records if q['goal_id']==gid});sub=[]
  for a in actions:
   z={}
   for split in ['FIT','CONFIRM']:
    rr=[q for q in records if q['goal_id']==gid and q['action_index']==a and q['split']==split];n=len(rr);hard=sum(q['hard'] for q in rr);full=sum(q['full'] for q in rr);z[split]=dict(n=n,hard=hard,hard_CI95=cp(hard,n),full=full,full_CI95=cp(full,n),reward_mean=float(np.mean([q['reward'] for q in rr])))
   entry=dict(goal_id=gid,action_index=a,**z);table.append(entry);sub.append(entry)
  # Fit-only hard-validity selection; full success is tied at zero, so no
  # learned/MC full-goal ranking benefit is claimed.
  selected=max(sub,key=lambda x:(x['FIT']['hard'],-x['action_index']));comparisons.append(dict(goal_id=gid,selected_using_FIT_only=selected['action_index'],independent_selected_hard=selected['CONFIRM']['hard']/8,independent_random_action_mean_hard=float(np.mean([x['CONFIRM']['hard']/8 for x in sub])),independent_selected_full=selected['CONFIRM']['full']/8,independent_random_action_mean_full=float(np.mean([x['CONFIRM']['full']/8 for x in sub]))))
 result=dict(actions=table,selection_comparison=comparisons,total_continuations=len(records),prefixes=len(obj['prefixes']),prefixes_with_no_observed_valid_continuation=sum(not any(r['hard'] for r in records if r['goal_id']==q['goal_id']) for q in obj['prefixes']),full_success=sum(q['full'] for q in records),lookahead_filter_status='NOT_RUN: no full-success action-ranking signal in128 TRAIN continuation samples; only2 prefix identities,8 independent confirmation replicates/action',interpretation='Monte Carlo under frozen C continuation policy, not reachability proof; CIs per action use independent replicates; actions share CRN');save('FUTURE_VIABILITY_SUMMARY.json',result);return result

def table_line(label,v):
 q=v['marginals'];i=v['intersections'];return f"|{label}|{q['complete']}/{v['trajectories']}|{q['backbone']}|{q['hard']}|{q['motif_contiguous_region']}|{i['motif_and_hard']}|{i['hard_and_regional']}|{i['full']}|{v['K1']['success']}/{v['goals']}|{v['K8']['success']}/{v['goals']}|"
def write(name,text):(R/name).write_text(text.strip()+'\n')
def main():
 history=read(R/'HISTORICAL_CONSTRAINT_DECOMPOSITION.json');results={};groups={}
 for panel in ['TRAIN','CONFIRM']:
  for pol in ['C','D100']:
   for method in ['U','V','M','VM']:
    gg=collect(panel,pol,method)
    if gg:groups[panel+'_'+pol+'_'+method]=gg;results[panel+'_'+pol+'_'+method]=summarize(gg)
 diffs={}
 for pol in ['C','D100']:
  a=results['CONFIRM_'+pol+'_U']
  for method in ['V','M','VM']:
   key='CONFIRM_'+pol+'_'+method
   if key in results:
    diffs[pol+'_'+method]={q:paired(a,results[key],q) for q in ['backbone','hard','full','motif_contiguous_region']}
    diffs[pol+'_'+method].update({f'K{K}_goal':paired_goal_endpoint(groups['CONFIRM_'+pol+'_U'],groups[key],K) for K in [1,8]})
 step=step_analysis();future=future_analysis();source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip();protocol=read(R/'FROZEN_PROTOCOL.json');init=read(R/'INITIAL_RESOURCE_ADMISSION.json');f=shutil.disk_usage(R).free
 # Remove temporary analysis-only mappings; retain compact raw goal identity
 # linkage separately for reproducible paired resampling.
 paired_data={k:v.pop('_goal_flags') for k,v in results.items()};save('PAIRED_GOAL_FLAGS.json',paired_data);save('FILTERING_NUMERICAL_RESULTS.json',dict(results=results,paired_differences=diffs,source_commit=source_commit,protocol_SHA256=sha(R/'FROZEN_PROTOCOL.json')))
 historical_rows=[]
 for label,v in history['panels'].items():
  q=v['marginals'];i=v['intersections'];historical_rows.append(f"|{label}|{v['goals']}|{v['trajectories']}|{q['complete']}|{q['motif_identity_region']}|{q['motif_order_region']}|{q['motif_contiguous_region']}|{q['exposed']}|{q['buried']}|{q['contrast']}|{q['local_geometry']}|{q['nonlocal_clash']}|{q['global_Rg']}|{q['backbone']}|{q['sequence_quality']}|{q['packing']}|{q['hard']}|{i['motif_and_hard']}|{i['hard_and_regional']}|{q['full']}|")
 common=f"\n\nReproducibility: source commit `{source_commit}`; frozen protocol SHA256 `{sha(R/'FROZEN_PROTOCOL.json')}`. Numerical records, goal IDs, seeds and original file hashes are retained in this evidence root. No neural optimizer was constructed. TEST remains sealed. These are deterministic coarse C-alpha property diagnostics, without evidence of folding or biochemical function."
 decomposition='''# Zero-success constraint decomposition

Every historical score was exactly reproduced from compact AA/geometry actions through the unchanged autoregressive builder. All denominators below are trajectories, not independent proteins. CONFIRM panels each contain32 distinct goal/witness identities ×8 candidates; DEV_D100 contains8 distinct goals ×8 and must not be pooled or called the32-goal result.

|Policy/panel|Goals|N|Complete|Motif identity in region|Ordered subsequence|Exact contiguous motif in region|Exposed|Buried|Contrast|Local|Nonlocal|Rg|Backbone|Sequence|Packing|Hard|Motif∩hard|Hard∩all regional|Full|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(historical_rows)+'''

Motif identity, order and adjacency are distinct diagnostic conditions; the original hard motif criterion is the exact ordered contiguous dipeptide somewhere inside the declared8-residue exposed region. No exact motif start or additional spacing was specified, so no new positional requirement is imposed. Global motif identity counts and complete violation patterns are in HISTORICAL_CONSTRAINT_DECOMPOSITION.json. Ordering alone is insufficient.

A. Individual requirements often fail: original motif satisfaction is only10–12/256 and each regional property also has substantial misses. B. Their intersection is still rarer: each policy has4–6 hard-valid candidates meeting all exposure/burial/contrast criteria but none also have the motif. C. Internal inconsistency is not demonstrated: the sealed task manifest records an existing native computationally feasible witness for every confirmation goal. Those witnesses are never supplied to de novo generation. Existing distribution support for the combination is the question, not biological feasibility.

Sequence–structure compatibility is a validated AA/exposure-association *soft reward component*, with no original binary success threshold. Its exact per-candidate scores and summary are retained; inventing a new compatibility pass criterion would change the benchmark. All original geometry, sequence and packing criteria remain unchanged.
'''
 write('ZERO_SUCCESS_CONSTRAINT_DECOMPOSITION.md',decomposition+common)
 prefix='''# Prefix-validity failure audit

Only finite/action support, adjacent-distance bounds, nondegenerate local frames and the original final-length nonlocal-clash budget are treated as irreversible. AtN64 the task permits one clash pair (<2Å, sequence separation≥3); the second pair exceeds2 clashes/100 residues. Applying a current-prefix denominator would wrongly tighten the criterion. Cross-product norm must exceed1e-8. Adjacent distance uses the unchanged calibrated interval [2.8504200470168195,4.052024993913842]Å.

Current radius of gyration and current contact *average* are not absorbing failures. All historical complete chains satisfy the original Rg range. The fixed-final-N contact-edge lower bound and irreversible sequence-quality bounds are recorded separately, never used by the immediate geometry filter. No new angle/torsion cutoffs are invented: original open action support and frame rule are checked.

|Historical panel|Already invalid16|32|48|64|Earliest clash cause|Earliest spacing cause|
|---|---:|---:|---:|---:|---:|---:|
'''
 for label,v in history['panels'].items():prefix+=f"|{label}|"+'|'.join(str(v['stages'][str(t)]['already_invalid']) for t in [16,32,48,64])+f"|{v['first_failure_causes'].get('nonlocal_clash_budget',0)}|{v['first_failure_causes'].get('local_spacing',0)}|\n"
 prefix+='\nHazards use the number still prefix-valid immediately before each residue as the denominator; raw residue-by-residue hazards and first-failure records are retained, rather than treating later consequences of an already invalid prefix as new causes. Goal-type strata and new D100 matched-panel audit are in FILTERING_NUMERICAL_RESULTS.json. First-failure causes can coincide. No nonfinite/support/angle/torsion cause was inferred without observed evidence.\n'
 write('PREFIX_VALIDITY_FAILURE_AUDIT.md',prefix+common)
 one='''# One-step action feasibility

TRAIN-only controls: first6 task identities; stages8,16,32,48;16 independent joint AA/geometry proposals per valid state. Prefix is the *first valid prefix* among8 existing ordinary candidates, selected without looking at its future; omissions are not replaced. Each model is evaluated on own generated prefixes, common C-generated prefixes, and matching native TRAIN control prefixes. Native coordinates never enter the de novo benchmark. Shared common prefixes distinguish distribution differences from occupancy differences. Survivor selection limits claims about all generated states.

|Policy|Condition|Valid proposals/total|Rate|Identity-cluster95% interval|
|---|---|---:|---:|---|
'''
 for pol,data in step.items():
  for condition in ['own_generated','common_C_generated','native_TRAIN_control']:
   v=data['strata'][condition+'_ALL'];one+=f"|{pol}|{condition}|{v['valid']}/{v['trials']}|{v['rate']:.4f}|{v['identity_CI']['CI95']}|\n"
 one+='\nNative minus own-generated differences, paired by identity and stage:\n\n'+ '\n'.join(f"- {pol}: {v['native_minus_generated_clustered']}" for pol,v in step.items())+'\n\nPer-stage, goal-order, motif-position, sampled-AA and geometry-component strata are in ONE_STEP_SUMMARY.json. Rare-AA strata are descriptive and have limited sample sizes. Separate8-draw conditional probes for each requested motif amino acid record its policy probability; these are conditional geometry probes, not joint policy samples. One-step prefix preservation does not establish eventual goal reachability.\n'
 failure_points=read(R/'FAILURE_POINT_ALTERNATIVES.json');fr=failure_points['records'];one+='\nRetrospective TRAIN failure-point probe (separate from the unbiased fixed-stage trials): '+str(len(fr))+' prefixes immediately before a recorded first violation; '+str(sum(sum(q['valid'] for q in z['trials']) for z in fr))+'/'+str(len(fr)*16)+' alternative proposals preserve prefix validity. Selection is conditioned on an observed failure and cannot estimate population hazard. Exact source/candidate/stage and original failure are retained in FAILURE_POINT_ALTERNATIVES.json.\n'
 write('ONE_STEP_ACTION_FEASIBILITY.md',one+common)
 fut='''# Future continuation viability

A very small TRAIN pilot uses2 prefix identities, up to4 immediately valid first actions each, and8 FIT plus8 independent CONFIRM continuations per action. Prefix selection uses only current valid state and still-live known constraints. Frozen C is the identical continuation policy. Action alternatives share matched continuation streams within each split; replicates and splits are independent. Complete metadata and exact seeds are in FUTURE_VIABILITY.json.

|Goal|Action|FIT hard/8|CONFIRM hard/8|CONFIRM hard95% interval|CONFIRM full/8|
|---|---:|---:|---:|---|---:|
'''
 for row in future['actions']:fut+=f"|{row['goal_id']}|{row['action_index']}|{row['FIT']['hard']}|{row['CONFIRM']['hard']}|{row['CONFIRM']['hard_CI95']}|{row['CONFIRM']['full']}|\n"
 fut+=f"\nTotal continuations: {future['total_continuations']}; full successes: {future['full_success']}; prefix states with no observed hard-valid continuation: {future['prefixes_with_no_observed_valid_continuation']}/{future['prefixes']}. Fit-only selection results on independent continuation seeds: `{future['selection_comparison']}`.\n\n{future['lookahead_filter_status']}. Zero/8 has a wide exact95% upper bound; neither zero observed success nor the small hard-validity comparison proves unreachable goals or useful action ranking. No expanded completion-ranking confirmation experiment is admitted.\n"
 write('FUTURE_CONTINUATION_VIABILITY.md',fut+common)
 table='|Panel/policy/method|Complete|Backbone|Hard|Motif|Motif∩hard|Hard∩regional|Full candidates|K1 goals|K8 goals|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(table_line(k,v) for k,v in results.items())
 motif='''# Motif-constraint diagnostic

M masks only AA choices that would make the original contiguous motif impossible to finish anywhere inside its declared exposed region. The deterministic finite-automaton viability mask preserves all legal placements; it does not invent an assigned position or force residues outside the required window. Before the window and after success, non-motif AA choices remain free. At the last viable opportunity it may force the needed token. Geometry is always conditioned on the AA actually selected. All360 exhaustive toy suffix checks and exact original sampler replay passed.

This is known-instruction enforcement, not learned instruction following. It changes the effective AA distribution. It does not enforce exposure, burial, packing, entropy or any native geometry.

'''+table+'\n\nK8 is the pool-existence endpoint, attainable by selecting a full-success candidate using the exact known terminal evaluator. K1 is the first unselected candidate. All aborted candidates remain failures in both endpoints.\n\nC_U vs C_M isolates the sequence control; C_V vs C_VM tests it with the same geometry filter. Matched streams do not imply identical structures after AA changes; conditional geometry must change with AA. Exact positions/spacing absent from the original specification remain absent.\n'
 write('MOTIF_CONSTRAINT_DIAGNOSTIC.md',motif+common)
 filtering='''# Inference-only filtering results

U: original sampling. V: draw up to8 independent joint AA/geometry proposals at the current state; accept the first meeting unchanged irreversible backbone predicates. If none passes, terminate and count an incomplete failure; never replace the failed candidate. M: motif viability mask. VM: both. Distribution parameters are cached once per state so rejected proposals need additional geometric simulations but no extra neural forward pass. Every attempted action and reason is retained. No gradient/optimizer updates occurred.

'''+table+'\n\nPaired goal-cluster bootstrap differences vs ordinary policy (10,000 resamples; seed19917):\n\n'+ '\n'.join(f'- {k}: `{v}`' for k,v in diffs.items())+'''

Density accounting: for motif-only M, q_AA(a|s)=p_AA(a|s)I_allowed/Z_A and geometry retains p(g|s,a). The AA conditional KL to the original actor is exactly −log Z_A. This is a viability mask, not the posterior over whole trajectories conditional on motif success. For V/VM, accepted actions have q(a,g|s)=p_base(a,g|s)I_valid/Z_V, conditional on acceptance. Z_V is a continuous integral, not known exactly; finite8 attempts also create an abort event of mass(1−Z_V)^8. Therefore accepted-policy likelihood/KL is not the original actor likelihood/KL. Original analytical proposal KL at visited states and proposal log-densities are labeled as such; filtered KL is null. No claim of passing a learned-policy KL safeguard or promoting a filtered neural policy is made.

Cost note: historical C_U elapsed times come from the saved V9.1 sampler, whose snapshot bookkeeping differs from this instrumented worker. They are contextual rather than a matched timing speedup. New TRAIN U/V/M/VM and new D100 confirmation use the same instrumented worker; actor-state/proposal/actual-builder/terminal counts are separately retained. Pairwise sequence/structure diversity is conditional on complete chains, with the pair denominator recorded; failed completions remain in every success/validity denominator.\n\nMarginal exact intervals for K1/K8 and goal-cluster paired intervals are in FILTERING_NUMERICAL_RESULTS.json. Histories are one goal per witness identity; within-goal8 candidates do not create8 independent biological identities. The32 validation goals are historically exposed V9.1 tasks; filter algorithms, budgets and thresholds were frozen prospectively using TRAIN only, but no independent biological-validation claim is made. TRAIN and confirmation counts are never pooled.
'''
 write('INFERENCE_ONLY_FILTERING_RESULTS.md',filtering+common)
 # Standalone plots; small PNG only, no redundant coordinate exports.
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 labels=['U','V','M','VM']
 for ax,key,title in [(axes[0],'hard','Hard validity'),(axes[1],'full','Full goal success')]:
  for pol in ['C','D100']:
   ls=[x for x in labels if f'CONFIRM_{pol}_{x}' in results];ys=[results[f'CONFIRM_{pol}_{x}']['marginals'][key]/256 for x in ls];ax.plot(ls,ys,'o-',label=pol)
  ax.set_title(title);ax.set_ylim(0,1 if key=='hard' else max(.1,ax.get_ylim()[1]));ax.set_ylabel('candidate fraction');ax.legend()
 fig.tight_layout();fig.savefig(R/'plots/CONSTRAINT_FILTERING.png',dpi=140);plt.close(fig)
 # Provisional conclusions from predeclared diagnostics, no threshold tuning.
 C=results['CONFIRM_C_U'];V=results['CONFIRM_C_V'];VM=results['CONFIRM_C_VM'];a_gate=(sum(C['first_failure_histogram'].values())/256>=.25 and diffs['C_V']['backbone']['mean']>=.15 and diffs['C_V']['backbone']['CI95'][0]>0);diversity=dict(AA_identity_change=V['pairwise_AA_identity']-C['pairwise_AA_identity'],CA_RMSD_ratio=V['pairwise_CA_Kabsch_RMSD']/C['pairwise_CA_Kabsch_RMSD'],unique_fraction_V=V['unique_sequence_fraction']);c_gate={k:(v['native_minus_generated_clustered']['mean']>=.1 and v['native_minus_generated_clustered']['CI95'][0]>0 and v['native_minus_generated_clustered']['clusters']>=6) for k,v in step.items()};full_total=sum(v['marginals']['full'] for k,v in results.items() if k.startswith('CONFIRM'))
 decision=f'''# Next CRISP research decision

**No RL training is authorized or executed in V9.2. No checkpoint is promoted.**

Outcome A (avoidable action-feasibility bottleneck): predeclared geometry evidence gate={a_gate}. C V−U backbone fraction: {diffs['C_V']['backbone']}. Diversity preservation measurements: {diversity}. Exact validity filtering must be distinguished from learned property steering.

Outcome B (property conjunction bottleneck): ordinary C already has109/256 hard-valid trajectories but only6 hard-valid regional-property matches and6 hard-valid motifs, with zero full intersection. The new sequence-control and VM results are given below; they identify the remaining property bottleneck without redefining tolerances.

Outcome C (generated-prefix shift): predeclared native−generated ≥10pp and positive paired lower bound across≥6 identities flags={c_gate}. Conditional survivor-state measurements and uncertainty remain essential; native-versus-generated state distributions differ in more than just policy error.

Outcome D (no useful feasible support): total observed confirmation full successes across separately reported new/reused method panels={full_total}. Zero observations would not prove no feasible completion. Positive filtered successes demonstrate computational proposal support, not folding, function or a learned instruction-following policy.

{table}

The smallest next experiment should isolate the dominant remaining limitation: preserve deterministic user-specified sequence constraints and exact inference-time irreversible-validity handling as fixed controls; test one TRAIN-only property-steering hypothesis with a prospective disjoint goal panel, hard-valid/full-goal endpoints and original KL controls. Do not train a large hierarchy or SAC critic to rediscover computable validity constraints. If filtered support remains sparse, first calibrate regional property controllability with frozen non-neural controls rather than add RL updates. Current results do not authorize any new run.
'''
 write('NEXT_CRISP_RESEARCH_DECISION.md',decision+common)
 report=f'''# Final E017 V9.2 scientific report

## Scope and state

Completed bounded, inference-only diagnostic. J1, original actor and the quarantined V9.1 update100 actor remain frozen. No actor/critic/planner/auxiliary training, TEST evaluation, native-coordinate inference targets, IK benchmark, folding inference or cleanup occurred.

The V9.1 headline after-RL32-goal comparison was checkpoint50, not the stopped100 checkpoint. Here both historical D50 and original D100 DEV are retained separately; D100 receives a new matched32-goal diagnostic and is never called scientifically selected. C is the repaired pre-V9 initializer/update0. B is the same parent weights with the legacy V9 adapter. A is J1.

## Main measurements

{table}

C V−U backbone change: `{diffs['C_V']['backbone']}`; C VM−U full candidate change: `{diffs['C_VM']['full']}`. Marginal K1/K8 exact intervals and all paired cluster intervals are in the numerical artifact. These are diagnostic comparisons on32 historically exposed VALIDATION goals, not independent biological replication.

## Answers

1. Historical zero full success comes from a rare conjunction: motif identity/order/adjacency, regional exposure/burial/contrast, and hard geometry/sequence/packing validity. Individual constraints and their intersections were measured, not inferred from average reward. All historical action-score replays matched exactly.
2. C ordinary geometry has139/256 irreversible failures, yet109/256 trajectories already satisfy all hard constraints. Only10 have the exact regional motif;6 are motif-and-hard, and none also meet both exposure thresholds. Geometry is a major avoidable problem, but solving it alone is insufficient.
3. Irreversible failures accumulate throughout generation. C already invalid at16/32/48/64:26/66/99/139 of256. First-failure causes:83 clashes and56 spacing; full residue hazards and goal-type strata are retained. Rg/contact averages are not mislabeled prefix-monotone.
4. One-step alternatives and native controls are directly sampled at stages8/16/32/48 for four frozen policies. Their joint and conditional-AA rates/uncertainty are in ONE_STEP_ACTION_FEASIBILITY.md. Future viability has128 continuations over2 TRAIN prefixes; it is not a reachability proof or well-powered value-ranking experiment.
5. Immediate filtering and motif enforcement are evaluated separately and jointly above. Diversity, failed-candidate accounting, attempted actions, cached actor calls and runtime are measured. Accepted-law KL is not claimed equal to proposal KL.
6. Full-goal support without native targets is measured by the M/VM counts above. Native witnesses establish only computational consistency of task specifications, never inference assistance or biological feasibility.
7. No high-level RL/SAC campaign is justified by this diagnostic alone. The next smallest test must isolate regional property steering after fixed hard sequence/validity controls, with independent prospective targets and the same safeguards. See NEXT_CRISP_RESEARCH_DECISION.md.

## Provenance and limits

J1 SHA256 `{c['J1SHA']}`; original independent actor checkpoint SHA256 `{c['PSHA']}`. D100 SHA256 `141fe1de695d2ade17f7453f4155f219d3afadffd324edf3fd84b73e07f1125d`; D50 SHA256 `dd857c1716a3dc19997a10bc3c189a1341f8326e6ca6749684362001ce81eb76`. Full paths and model-state fingerprints are in COMPUTE_AND_STORAGE_ACCOUNTING.json and provenance.

Original reward SHA256 `{sha(OLD/'REWARD_SPECIFICATION.json')}`. Original local geometry and final-N clash allowance remain unchanged. Compatibility has no hard pass threshold. Confirmation goal identity, group and seeds are in immutable source manifests and per-goal records. Confidence intervals cluster by independent witness identity; small TRAIN/native/future panels and exposed validation goals limit generalization. Filtered sampling is a modified inference distribution; deterministic motif control is not learned instruction following. No foldability or experimental-function conclusions follow.

Study source commit `{source_commit}`. All new artifacts, cache and temporary paths are onD:. No new model file is needed for recovery; completed per-goal results are atomically published with hashes and can be verified/reused after interruption.
'''
 write('FINAL_E017_V92_SCIENTIFIC_REPORT.md',report+common)
 counts=dict(new_complete_generation_candidates=sum(v['trajectories'] for k,v in results.items() if not k=='CONFIRM_C_U'),future_continuations=future['total_continuations'],new_one_step_joint_proposals=sum(len(read(R/f'ONE_STEP_{pol}.json')['records'])*16 for pol in ['A','B','C','D100']),retrospective_failure_prefixes=len(fr),retrospective_joint_proposals=len(fr)*16,retrospective_conditional_AA_probes=sum(len(x['forced_motif_AA'])*8 for x in fr),one_step_conditional_AA_probes=sum(sum(len(z['forced_motif_AA'])*8 for z in read(R/f'ONE_STEP_{pol}.json')['records']) for pol in ['A','B','C','D100']),engineering_extra_candidates=16,actor_updates=0,critic_updates=0)
 modelrefs=dict(J1=dict(path=str(c['J1']),SHA256=c['J1SHA']),parent=dict(path=str(c['PARENT']),SHA256=c['PSHA']),V91_recovery=read(OLD/'checkpoints/protein/RECOVERY_INDEX.json'))
 completed={x.name:read(x) for x in R.glob('COMPLETED_*.json')};files=[q for q in R.rglob('*') if q.is_file() and 'source' not in q.relative_to(R).parts];allocated=sum(q.stat().st_blocks*512 for q in files);account=dict(status='COMPLETED_NO_TRAINING',source_commit=source_commit,source_status=subprocess.check_output(['git','status','--porcelain'],cwd=R/'source',text=True),counts=counts,model_references=modelrefs,completed_workers=completed,paired_differences=diffs,diversity_gate_measurements=diversity,initial_admission=init,D_free_bytes=f,D_free_GiB=f/2**30,operational_headroom_bytes=2*2**30,new_evidence_allocated_bytes=allocated,source_worktree_additional=True,zero_new_model_checkpoint_bytes=True,resource_telemetry='logs/RESOURCE_TELEMETRY.jsonl',cache_and_tmp=str(R/'runtime'),failure_evidence_preserved=True,all_raw_scores_replayed_exactly=True,analysis_seconds=time.perf_counter()-START);account['retrospective_failure_probe']=failure_points['completed'];account['generation_workloads']={k:{z:v[z] for z in ['trajectories','proposal_count','actual_builder_evaluations_during_generation','terminal_score_evaluations','actually_rejected_proposals','terminations','calls','generation_seconds','seconds_per_candidate','seconds_per_successful_candidate']} for k,v in results.items()};save('COMPUTE_AND_STORAGE_ACCOUNTING.json',account)
 print(json.dumps(dict(results={k:dict(hard=v['marginals']['hard'],full=v['marginals']['full'],K1=v['K1'],K8=v['K8']) for k,v in results.items()},paired=diffs,free=f,allocated=allocated)),flush=True)
if __name__=='__main__':main()
