"""CPU publication, identity-cluster uncertainty and bounded-study accounting."""
import collections,hashlib,itertools,json,math,runpy,shutil,subprocess,time
from pathlib import Path
import numpy as np
from scipy.stats import beta
R=Path('/mnt/d/Simone/proteinGen_data/e017_v93_regional_controllability_20261024_v1');d=runpy.run_path(str(R/'source/scripts/e017_v93/common.py'));globals().update({k:d[k] for k in ['R','OLD','V91','V9','c','SPEC','read','save','sha','ns','reconstruct']})
def ci(values):
 a=np.asarray(values,float);rng=np.random.default_rng(19917);b=a[rng.integers(0,len(a),(10000,len(a)))].mean(1);return dict(mean=float(a.mean()),CI95=list(map(float,np.quantile(b,[.025,.975]))),identities=len(a))
def binCI(k,n):return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]
def summarize(panel,pol):
 paths=sorted((R/'results').glob(f'{panel}_{pol}_[0-9][0-9][0-9].json'));groups=[];identity=collections.defaultdict(list);m=collections.Counter();cost=collections.Counter();div=[];shape=[];trace=[];by_level=collections.defaultdict(collections.Counter);reward=[];gkl=[]
 for path in paths:
  assert sha(path)==read(path.with_suffix('.sha.json'))['sha256'];o=read(path);groups.append(o);rows=o['records'][:8];level=o['task']['goal']['difficulty'];gid=o['task']['goal']['witness_identity'];stats={};timegroup=rows[0]['seconds'];cost['generation_seconds']+=timegroup;cost.update(o['calls'])
  f=[];coords=[]
  for row in rows:
   q=row['metrics'];goal=o['task']['goal'];flags=dict(complete=q['complete'],motif=q.get('motif_utility',0)==1,exposed=q.get('exposed_mean',-1)>=goal['exposure_min']-1e-10,buried=q.get('buried_mean',2)<=goal['burial_max']+1e-10,contrast=q.get('contrast',-1)>=goal['contrast_min']-1e-10,backbone=q['backbone_valid'],packing=q['packing_valid'],hard=q['hard_valid'],full=q['success']);flags['exposed_buried']=flags['exposed'] and flags['buried'];flags['regional']=flags['exposed_buried'] and flags['contrast'];m.update({k:int(v) for k,v in flags.items()});by_level[level].update({k:int(v) for k,v in flags.items()});by_level[level]['candidates']+=1;f.append(flags);cost['proposals']+=len(row['proposals']);cost['rejected']+=sum(not p[6] for p in row['proposals']) if pol=='C' else 0;cost['terminal_evaluations']+=1;cost['residue_steps']+=len(row['geometry']);cost['aborts']+=row['termination'] is not None;cost['repairs']+=row['repairs'];reward.append(q['reward']);trace.append(row['trace_logprob']-row['reference_trace_logprob']);gkl+=row['geometry_KL']
   if q['complete']:
    assert row['sequence']==o['task']['fixed_sequence'];assert q['motif_utility']==1;coords.append(reconstruct(row).ca.numpy());shape.append(dict(Rg=q['geometry']['Rg_normalized'],contact_degree=q['geometry']['longrange_contact_degree'],exposed=q['exposed_mean'],buried=q['buried_mean'],contrast=q['contrast']))
  for key in ['hard','full','regional','exposed','buried','backbone']:stats[key]=np.mean([z[key] for z in f])
  stats['K1']=int(f[0]['full']);stats['K8']=int(any(z['full'] for z in f));identity[gid].append(stats);by_level[level]['K1']+=stats['K1'];by_level[level]['K8']+=stats['K8'];by_level[level]['goals']+=1
  rms=[]
  for a,b in itertools.combinations(coords,2):
   a=a-a.mean(0);b=b-b.mean(0);u,_,v=np.linalg.svd(a.T@b);rot=u@np.diag([1,1,np.linalg.det(u@v)])@v;rms.append(float(np.sqrt(np.square(a@rot-b).sum()/64)))
  if rms:div.append(float(np.mean(rms)))
 points={key:ci([np.mean([z[key] for z in rows]) for rows in identity.values()]) for key in ['hard','full','regional','exposed','buried','backbone','K1','K8']}
 return dict(goals=len(groups),identities=len(identity),candidates=len(groups)*8,marginals=dict(m),K1=sum(rows[0]['metrics']['success'] for o in groups for rows in [o['records']]),K8=sum(any(r['metrics']['success'] for r in o['records'][:8]) for o in groups),clustered=points,by_difficulty={k:dict(v) for k,v in by_level.items()},cost=dict(cost),mean_structural_pair_RMSD=float(np.mean(div)),complete_shape_n=len(shape),shape_means={k:float(np.mean([s[k] for s in shape])) for k in shape[0]} if shape else {},reward_mean=float(np.mean(reward)),reward_SD=float(np.std(reward)),proposal_geometry_KL_per_active=float(np.mean(gkl)),full_trace_KL_MC_mean=float(np.mean(trace)),full_trace_KL_MC_SE=float(np.std(trace,ddof=1)/math.sqrt(len(trace))),_identities=identity)
def compare(a,b):
 assert set(a['_identities'])==set(b['_identities']);out={}
 for key in ['K1','K8','hard','full','regional','backbone']:
  v=[]
  for ident in a['_identities']:v.append(np.mean([x[key] for x in b['_identities'][ident]])-np.mean([x[key] for x in a['_identities'][ident]]))
  out[key]=ci(v)
 return out
def delay():
 obj=read(R/'DELAYED_ACTION_EFFECTS.json');rows=obj['records'];states=[];stage=collections.defaultdict(lambda:collections.defaultdict(list))
 for selection in obj['selection']:
  gid=selection['goal_id'];t=selection['t'];actions=sorted({r['action'] for r in rows if r['goal_id']==gid and r['t']==t});est={}
  for a in actions:
   est[a]={}
   for split in ['FIT','CONFIRM']:
    rr=sorted([r for r in rows if r['goal_id']==gid and r['t']==t and r['action']==a and r['split']==split],key=lambda z:z['replicate']);est[a][split]=dict(reward=np.array([r['metrics']['reward'] for r in rr]),full=np.array([r['metrics']['success'] for r in rr],float),hard=np.array([r['metrics']['hard_valid'] for r in rr],float))
  if len(actions)<2:continue
  best=max(actions,key=lambda a:(est[a]['FIT']['reward'].mean(),-a));worst=min(actions,key=lambda a:(est[a]['FIT']['reward'].mean(),a));independent={}
  for metric in ['reward','full','hard']:
   selected=est[best]['CONFIRM'][metric];random=np.mean([est[a]['CONFIRM'][metric] for a in actions],axis=0);diff=selected-random;independent[metric]=dict(selected_mean=float(selected.mean()),random_action_mean=float(random.mean()),gain=float(diff.mean()),paired_SE=float(diff.std(ddof=1)/4));stage[t][metric].append(float(diff.mean()))
  gap=est[best]['CONFIRM']['reward']-est[worst]['CONFIRM']['reward'];fit_means=[est[a]['FIT']['reward'].mean() for a in actions];mcnoise=np.mean([est[a]['FIT']['reward'].var(ddof=1)/16 for a in actions]);states.append(dict(goal_id=gid,identity=selection['identity'],t=t,selected_on_FIT=best,worst_on_FIT=worst,independent=independent,FIT_between_action_variance=float(np.var(fit_means,ddof=1)),FIT_MC_mean_variance=float(mcnoise),CONFIRM_best_minus_worst_reward=float(gap.mean()),CONFIRM_paired_SE=float(gap.std(ddof=1)/4),actions={str(a):{split:{k:dict(mean=float(v.mean()),SE=float(v.std(ddof=1)/4),samples=len(v),success_count=int(v.sum()) if k in ['full','hard'] else None) for k,v in vals.items()} for split,vals in est[a].items()} for a in actions}))
 result=dict(states=states,by_stage={str(t):{key:ci(v) for key,v in val.items()} for t,val in stage.items()},continuations=len(rows),identities=4,notes='independent confirmation streams after FIT-only selection; paired CRN within-state; stage CIs cluster4 identities; exploratory, no learned critic or biological causality');save('DELAYED_EFFECT_NUMERICAL_RESULTS.json',result);return result

def main():
 assert read(R/'COMPLETED_train.json')['PASS'] and read(R/'COMPLETED_validation.json')['PASS'];results={f'{panel}_{pol}':summarize(panel,pol) for panel in ['TRAIN','PROSPECTIVE'] for pol in ['A','B','C']};diff={panel:dict(B_minus_A=compare(results[panel+'_A'],results[panel+'_B']),C_minus_B=compare(results[panel+'_B'],results[panel+'_C'])) for panel in ['TRAIN','PROSPECTIVE']};dd=delay();test=read(R/'POLICY_GRADIENT_TESTS.json');back=read(R/'PROTEIN_TRACE_BACKWARD_TEST.json');ready=read(R/'TRAIN_READINESS.json');plan=read(R/'SUBSTANTIAL_CAMPAIGN_SPEC.json');pool=read(R/'manifests/FUTURE_FIXED_TRAIN_POOL.json');publicationsource=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip();iden={k:v.pop('_identities') for k,v in results.items()};save('BASELINE_IDENTITY_STATISTICS.json',iden);save('FIXED_SEQUENCE_NUMERICAL_RESULTS.json',dict(results=results,paired=diff))
 lines=['|Panel / method|Identities / goals|Candidates|Backbone|Hard|Exposed|Buried|E∩B|All regional|Full|K1 goals|K8 goals|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for name,v in results.items():
  m=v['marginals'];lines.append(f"|{name}|{v['identities']}/{v['goals']}|{v['candidates']}|{m['backbone']}|{m['hard']}|{m['exposed']}|{m['buried']}|{m['exposed_buried']}|{m['regional']}|{m['full']}|{v['K1']}/{v['goals']}|{v['K8']}/{v['goals']}|")
 table='\n'.join(lines);foot=f"\n\nSource publication commit `{publicationsource}`. J1 SHA256 `{c['J1SHA']}`, parent actor SHA256 `{c['PSHA']}`. No V91 stopped checkpoint was initialized. All evidence/cache/tmp on D:. No TEST, folding inference, native-coordinate target, protein policy update, critic or planner training."
 def write(name,text):(R/name).write_text(text.strip()+'\n'+foot+'\n')
 controls=read(R/'NATIVE_AND_NEGATIVE_CONTROLS.json');alt=read(R/'SUPPORTED_ALTERNATIVE_GOALS.json');search=read(R/'SUFFIX_SEARCH.json');searchrows=[]
 for i,v in enumerate(search['records']):
  if 'failure' in v:searchrows.append(v);continue
  baseline=read(R/f'results/TRAIN_C_{i:03}.json')['records'];srows=v['records'];allmetrics=[v['original_candidate_metrics']]+[r['metrics'] for r in srows];searchrows.append(dict(goal_id=v['goal_id'],Bestof8_full=any(r['metrics']['success'] for r in baseline[:8]),Bestof16_full=any(r['metrics']['success'] for r in baseline[:16]),rewind16_full=any(m['success'] for m in allmetrics),Bestof16_proposals=sum(len(r['proposals']) for r in baseline[:16]),rewind16_proposals=len(baseline[0]['proposals'])+sum(len(r['proposals']) for r in srows),complete_evaluations_each=16))
 save('SEARCH_COMPARISON.json',dict(records=searchrows,TRAIN_only=True,no_tuning_on_prospective=True))
 altrows=[]
 for v in alt['records']:
  goal=v['original_task']['goal'];task_index=next(i for i,t in enumerate(read(R/'manifests/TRAIN_TASKS.json')['tasks']) if t['goal']['goal_id']==goal['goal_id']);original=read(R/f'results/TRAIN_C_{task_index:03}.json')['records'][:8];new=v['records'];a=[r['metrics'].get('contrast') for r in original if 'contrast' in r['metrics']];b=[r['metrics'].get('contrast') for r in new if 'contrast' in r['metrics']];altrows.append(dict(identity=goal['witness_identity'],original_goal_id=goal['goal_id'],counterfactual_goal=v['task']['goal'],same_fixed_sequence=True,verified_generated_witness=v['witness_candidate'],original_full=sum(r['metrics']['success'] for r in original),reversed_full=sum(r['metrics']['success'] for r in new),signed_goal_steering_effect=float(np.mean(a)+np.mean(b)) if a and b else None,interpretation='positive means changed context moves original-region contrast in desired direction; exploratory TRAIN witness-selected goals'))
 save('SUPPORTED_GOAL_VARIATION_SUMMARY.json',dict(records=altrows))
 write('REGIONAL_PROPERTY_CONTROLLABILITY.md',f'''# Regional-property controllability

Environment decision: **{'GO' if ready['environment_GO'] else 'NOT_ADMITTED'}** under the prospectively frozen TRAIN gate. Selected native TRAIN controls:16/16 complete original-goal successes. Collapsed/extended negatives have reward0; no threshold was relaxed. Fixed AA sequences meet known motifs before generation; this isolates geometry and does not demonstrate AA design or learned instruction following.

{table}

TRAIN filtered backbone rate={ready['filtered_backbone']}; supported identity-goals={ready['positive_TRAIN_identities']}/16; shaped rewardSD={ready['reward_SD']}. Full regional success includes original local/nonlocal/Rg backbone, packing, fixed-sequence quality, both exposure/burial targets and contrast. Compatibility is the original soft reward only, never a new binary threshold. Exposure proxy remains smooth10_sep3, previously checked against atom accessibility; it does not establish folding/function.

All base goals are supported by native computational witnesses. Relaxed thresholds are nested .05 slack inside the same witnessed goal, not an arbitrary property reversal. Empirically reversed same-sequence goals were attempted only after observing a hard-valid generated witness; {len(altrows)} such TRAIN controls were retained. Their effects: `{altrows}`. Failure to find another witness would not prove infeasibility. No native coordinates are provided to the generating controller.

New prospective identity-level results quantify transfer of frozen baselines, not RL learning. Training-readiness is distinct from whether the frozen actor is already strongly goal-controllable.''')
 write('FIXED_SEQUENCE_BASELINES.md',f'''# Fixed-sequence baselines

A: frozen J1 Gaussian geometry conditional on the actual prescribed AA. B: verified preV9 independent actor under repaired V91 goals. C: B with up to8 geometry proposals at a state, first exact-prefix-valid acceptance, deterministic incomplete failure on exhaustion. AA choices are external interventions and carry zero categorical policy-gradient terms. Supports/temperature are unchanged.

D: exact terminal selection from the same C pool. Its K8 success is the pool-existence endpoint in the table; it is attainable because terminal criteria are known, and costs8 complete candidates. It is not a learned policy or an independent sample set.

{table}

Paired prospective differences: `{diff['PROSPECTIVE']}`. Bootstrap resamples16 protein identities with both difficulty levels retained (10,000 resamples,seed19917). Neither32 goal variants nor256 candidates are256 independent proteins. Raw identity contributions and difficulty strata are retained in numerical JSON.

E: bounded TRAIN-only rewind32 suffix search,16 terminal evaluations, compared with whole-chain Bestof8/16: `{searchrows}`. All initial-candidate and discarded-suffix costs are counted;15 suffixes are not15 whole chains. Four identities are contextual evidence, not a matched prospective superiority claim.

Shape/contact means, proposal KL, full-trace log-ratio estimates, complete-chain Kabsch diversity, rejection/abort counts and actor-state costs are in FIXED_SEQUENCE_NUMERICAL_RESULTS.json. Fixed sequences have no meaningful within-task AA diversity endpoint.''')
 write('DELAYED_ACTION_EFFECTS.md',f'''# Delayed-action effects

Four TRAIN identities at prefix lengths8,32,48; up to3 locally valid alternative actions,16 FIT plus16 independent CONFIRM continuations per action. All continuations use the same frozen filtered policy and fixed sequence. First-action rejection attempts are counted. No native future geometry is used. All locally valid actions tie under naive immediate-validity scoring, so uniform selection is its comparison.

Total continuations={dd['continuations']}. FIT-only selected actions were assessed on independent confirmation streams. Stage-specific selected-minus-uniform reward/full/hard differences: `{dd['by_stage']}`.

Per-state Monte Carlo variance, between-action variation and independent best-versus-worst paired errors are retained in DELAYED_EFFECT_NUMERICAL_RESULTS.json. Common random streams couple alternatives; replicates are independent. These four-identity uncertainty intervals are exploratory. Reward differences show computational delayed consequences where supported; noisy or zero estimated full-success differences do not prove action irrelevance. On-policy Monte Carlo RL does not require an admitted learned critic. No claim of biological causality or hierarchical-planning necessity follows.''')
 write('FILTERED_POLICY_DENSITY_AND_GRADIENTS.md',r'''# Filtered-policy stochastic computation

At state s, draw proposals z_j=(AA_j,u_j) until the first exact-valid deterministic extension, or at mostM=8 attempts. If none is valid, emit ABORT and terminal reward0. No retry replaces an aborted protein. For fixed-sequence generation AA=a_t is supplied, and p_theta(z|s)=Normal(u_active;mu_theta(s,a_t),sigma_theta(s,a_t)). There is no categorical factor for a prescribed AA, even if it satisfies a motif. In a later genuinely learned-AA stage, each proposal additionally contributes the sampled categorical density. A normalized allowed-AA mask contributes its actual log normalizer; a forced token contributes no categorical score term.

The full latent proposal trace tau includes every rejected and accepted proposal, stopping branch and current generated prefix. Its density is

    P_theta(tau|goal) = product over every reached proposal node p_theta(z_j|s_j,goal).

Acceptance/rejection/abort indicators are deterministic functions of the sampled trace, known goals and unchanged geometry transition, independent of theta when the trace is held fixed. All Gaussian supports are common; categorical proposal probabilities are positive on the same allowed set. The finite horizon/rejection budget bounds trace length. Under finite score moments, differentiation under this finite stochastic computation gives

    grad E[R] = E[(R - independent baseline) * sum_j grad log p_theta(z_j|s_j)].

This estimator needs no acceptance normalizer. Ignoring rejected proposals is biased; the toy counterexample demonstrates it. A group leave-one-out baseline using other iid trajectories conditional on the same goal has zero expected score contribution. Do not use a baseline containing the same trajectory, same-group estimated whitening, or clipped advantages while claiming exact unbiasedness. Gradient clipping is a separate optimizer stabilization, not an unbiased gradient claim.

Distance=exp(u0), angle=pi*sigmoid(u1), signed torsion=pi*tanh(u2). The physical log density subtracts log-Jacobians u0; log(pi)-softplus(-u1)-softplus(u1); log(pi)+2(log2-u2-softplus(-2u2)). Components activate only at t>=1,2,3. Raw Gaussian latents are detached for score-function replay; otherwise reparameterization cancels the intended score derivative. Geometry conditions on the actual prescribed/sampled AA.

Finite-grid interior representation is the existing explicit inference convention. Store the original u even if its physical representative is repaired. The augmented latent-trace law remains defined without pretending a representative action density is a discrete sampling mass. Inactive Gaussian coordinates are marginalized and contribute zero. Nonfinite latent failures cannot be silently resampled; stop/record numerical failure. Toy repair and Jacobian tests are retained.

KL(P_theta full trace || P_ref full trace)=E_theta[sum(log p_theta-log p_ref)] for a reference with the identical filter/fallback/mask. It equals the expected sum of conditional proposal KLs at all reached nodes, including rejections. By data processing it upper-bounds KL of accepted generated outputs (including abort), but is not that output KL. Original per-active geometry KL is also monitored separately. A differently filtered or unfiltered reference cannot silently replace this trace reference.

For J=E[R]-beta*KL_trace, let L=sum log p_theta-log p_ref and S=sum grad log p_theta. The gradient is E[(R-beta*L-baseline)*S-beta*S]. The explicit final term has expectation zero but is retained in the implementation. Exact enumeration including the group2 leave-one-out baseline verifies the regularized gradient. Differentiating only per-node KL while ignoring theta-dependent visitation is a different, generally biased objective.

Baseline action densities, effective accepted-action densities and complete-trace likelihoods are distinct. Accepted-only q(z|s)=p(z|s)I_valid/Z(s) involves an unknown continuous integral; finite attempts add abort mass(1-Z)^8. No effective-filtered KL is asserted equal to original proposal KL.''')
 write('POLICY_GRADIENT_CORRECTNESS_TESTS.md',f'''# Policy-gradient correctness

CPU finite-trace normalization: {test['exact_traces']} traces, total probability {test['probability_sum']}. Exact expected-return gradient, score gradient and finite differences agree (max finite-difference error {test['finite_difference_max_error']}). Accepted-only gradient demonstrably differs. Independent Monte Carlo gradients and standard errors are saved, including150,000 continuous-normal finite-rejection traces.

KL chain rule={test['trace_KL']} vs {test['chain_rule_KL']}; accepted-output KL={test['accepted_output_KL']} obeys data processing. KL-regularized RLOO gradient passes exact group2 enumeration in KL_GRADIENT_AND_SCHEMA_TESTS.json. Active geometry masks, physical Jacobian max error {test['physical_density_max_error']}, extreme finite-grid representation, forced-AA zero gradients and allowed-AA normalization are tested.

Six-parameter delayed stochastic engineering environment:250 accepted toy updates plus one repeated replay execution; exact expected return {test['toy_initial_return']}→{test['toy_final_return']}. Model/Adam/sampler/RNG replay at update100 is bit-exact. This is not protein RL evidence.

Actual fixed-sequence neural audit: {back['replayed_proposals']} proposals, max replay log-density discrepancy {back['logprob_max_error']}; finite gradient norm {back['total_gradient_norm']}; geometry-head score gradients present, prescribed-AA head score gradients zero; J1 and actor weights unchanged. The first replay improperly selected8 trajectories from a16 behavior batch and failed the unchanged tolerance. Matching the complete active16 batch gave exact agreement. The original failure is preserved. The strict goal-adapter metadata refusal and explicit nonsemantic projection are also documented. No criterion was relaxed.

Gradient/likelihood primitives are GO. Substantial-run release still requires an authorized exact full protein optimizer/scheduler/RNG replay under the campaign's actual execution configuration. V93 creates no protein optimizer update and does not claim the toy alone proves production recovery.''')
 fixed=read(R/'manifests/PROSPECTIVE_FIXED.json');joint=read(R/'manifests/PROSPECTIVE_JOINT.json')
 write('PROSPECTIVE_GOAL_PROTOCOL.md',f'''# Prospective goal protocol

Frozen before additional policy optimization: fixed-sequence {fixed['goals']} goals across16 previously property-policy-unscored VALIDATION identities/protected groups, two witnessed difficulty levels each. Joint stage {joint['goals']} goals across the same16 identities, including ordered motif reversals. Synthetic reversals explicitly change the witness sequence at the native motif occurrence and recheck all computational hard constraints; no physical sequence/structure compatibility is assumed from this construction.

All earlier V9/V91 property task IDs/groups and atom-reference IDs were excluded. Underlying128 VALIDATION reference windows had prior aggregate calibration exposure; these are new policy-evaluation goals, not pristine external biological data. Historical32 V92 goals are development evidence only and never substituted into this prospective panel. TRAIN fitting pool and reserved TRAIN development groups remain separate from VALIDATION. TEST stays sealed.

The fixed panel receives one frozen baseline comparison inV93. Joint panel remains sealed for future policy generation; no joint protein rollout is executed here. For future optimization, freeze the campaign and acceptance criteria now and do not tune repeatedly against fixed-panel outcomes. Final comparison uses K1 primary, K8 secondary, originalhard validity/KL/diversity safeguards and identity-cluster paired intervals. Two difficulty variants and motif reversals sharing a protein remain one uncertainty cluster. Task identities, regions, tokens, witness provenance and hashes are in manifests. No native coordinates or full witness sequence is supplied to a joint de novo controller.''')
 elapsed=sum(read(R/f'COMPLETED_{stage}.json')['seconds'] for stage in ['train','validation']);sampler=results['TRAIN_C']['cost']['generation_seconds']/16;backseconds=back['backward_seconds'];updateseconds=sampler+backseconds;eta5000=5000*updateseconds
 write('SUBSTANTIAL_RL_TRAINING_PLAN.md',f'''# First substantial low-level protein RL protocol

**Not launched; requires future explicit authorization and storage/recovery release.** Initial phase:5,000 optimizer updates ×16 complete-generation attempts=80,000 attempts, with completed proteins, aborts, residue actions and proposal draws counted separately. This is not a100–250-update performance test. The later joint phase is separately admitted, with a candidate additional5,000×16 envelope; never automatically launch it.

Strict initializer: original V3 actor, corrected ordered-goal adapter; no stopped V91 checkpoint. Permanently frozen J1; actor17,438,114 parameters unchanged. Fixed sequence geometry first; no categorical AA score terms. Existing qualified TRAIN pool: {pool['identities']} identities, {pool['protected_groups']} groups, {len(pool['tasks'])} witnessed goal variants. Reserve the16 V93 TRAIN identities/groups for monitoring, not fitting. Identity-uniform deterministic shuffle and balanced difficulty variants; no VALIDATION fitting. Geometry inputs receive actual fixed AA and current prefix, never native coordinates.

Fresh AdamW peakLR1e-6, betas(.9,.999), eps1e-8, weight_decay0; warmup100, flat through4,000, cosine to10% peak at5,000. Physical group16 with matching active replay batch; accumulate gradients across residue/proposal nodes then one optimizer step. FP32 model/FP64 latent density, AMP disabled initially. Terminal original reward, full trace likelihood, independent per-goal RLOO; no clipped advantages or dynamic same-group whitening. Global gradient norm clip5. Fixed trace-KL beta={plan['trace_KL_beta']} from TRAIN rewardSD/(10×MC full-trace KL). TRAIN full-trace KL estimate={plan['TRAIN_traceKL_estimate']}±{plan['TRAIN_traceKL_MC_SE']} SE. This is reference regularization, not a redefinition of original success.

Checkpoint every50 updates; durable named500/1,000/2,000/3,500/5,000; retain latest/best and preserve named files. Checkpoints include actor, Adam, scheduler, AMPNone, all RNG, exact identity/variant cursor, full sampling configuration, update/attempt/completed/residue/proposal counters, dataset/task/reward/source/reference hashes and pending proposal-trace journal. Atomic staged write, hash verification and publication; never modify a linked inode. Keep current-group traces and a bounded post-checkpoint replay journal, durable per-update seed/metric/hash ledger and representative scientific samples rather than every redundant full tensor. Journal replay must be tested before release; no unsafe assumption that discarding committed updates is exact recovery.

Monitoring at250,500,1,000,2,000,3,500,5,000 uses reserved TRAIN development fixtures. These are engineering/safety boundaries; lack of a performance gain at100/250 is not an automatic RL performance verdict. Preserve immediate nonfinite/recovery stops, stepKL>.02 rollback, original per-active geometryKL>1, >5pp hard/backbone-validity degradation and >20% structural-diversity loss. Report stops as training-stability/engineering failures. Final scientific judgement after the declared budget uses sealed prospective targets: paired K1 improvement with positive95% lower interval, original validity/KL/diversity safeguards and K8 not worse. No proxy-reward-only promotion.

Measured TRAIN filtered group timing≈{sampler:.3f}s per8-candidate group (some first4 measurement groups contain16; conservative mixed-size estimate), full16-trajectory gradient replay/counting audit={backseconds:.3f}s. A coarse5,000-update compute envelope from these measured components is{eta5000/3600:.2f}h, excluding validation, I/O, thermal scheduling and optimizer overhead; reprofile actual16 generation + optimizer engineering before release. This is an estimate, not a throughput guarantee.

Later joint AA+geometry learning must sample legitimate goal-conditioned AA actions and include every rejected categorical/conditional-geometry proposal in the trace score. Do not permanently enforce motifs and call that learned instruction following. Any supervised motif warm-start is a separately labeled supervised stage with its own data/provenance. Do not train a critic, reachability model or high-level planner as part of this first campaign.

Executable guarded entry point: `run.sh source/scripts/e017_v93/campaign.py --preflight`. Source contains the trace-gradient training driver, but requires future authorization, full optimizer replay release and a hashed development evaluator; it is a release candidate, not a qualified production launch. Do not bypass these checks. Configuration: SUBSTANTIAL_CAMPAIGN_SPEC.json.''')
 free=shutil.disk_usage(R).free;required=plan['minimum_free_bytes'];deficit=max(0,required-free)
 write('STORAGE_AND_RECOVERY_ADMISSION.md',f'''# Storage and recovery admission

Substantial training **NOT ADMITTED**. Current free={free:,} bytes ({free/2**30:.3f}GiB). Complete recovery measurement={back['complete_recovery_measured_bytes']:,} bytes via Torch serialization counting sink; no duplicated model artifact was written. Actor+Adam+schedule+AMPNone+RNG+sampler/config/hash bindings included; frozen J1/parent and immutable data are referenced by hash.

Conservative peak:8 distinct complete checkpoint images (5 scientifically named, latest, distinct best, staging),2GiB operational reserve,128MiB near growth,128MiB journals/compact telemetry,64MiB cache,32MiB reports. Required free={required:,} bytes ({required/2**30:.3f}GiB); deficit={deficit:,} bytes. Recommend8GiB free before authorization to allow safety margin and later study growth. A current failed/incomplete attempt is never silently promoted into a recovery checkpoint.

Nondestructive measures: reference immutable model/data files; compact JSON and prefix/action reconstruction; bounded transient replay plus seed/metric ledgers; hardlink new run-owned identical named/latest/best images when supported, with atomic replacement and no in-place mutation. Budget does not assume hardlink savings. Changing directories on the same D: volume creates no capacity. No historical checkpoint, dataset or evidence is removed, compressed, relocated or rewritten. Further capacity requires a separately approved storage expansion or review of disposable artifacts; protected data is not a cleanup target.

CPU toy checkpoint replay is bit-exact. Protein trace density/gradient replay is exact at matching batch16, but no protein optimizer update is authorized inV93. The exact future full optimizer/replay/journal configuration must pass its release test before production, after storage and authorization. `CAMPAIGN_PREFLIGHT.json` explicitly denies launch. This storage decision says nothing about RL performance.''')
 write('NEXT_CRISP_RESEARCH_DECISION.md',f'''# Next CRISP research decision

1. Environment: **{'GO' if ready['environment_GO'] else 'NOT_ADMITTED'}** for bounded low-level fixed-sequence property learning under the frozen TRAIN gates. Native controls witness feasibility and generated candidates quantify nonzero support; unseen combinations and biological folding remain unproved.
2. RL likelihood/gradient primitives: **GO** under audited full-proposal-trace semantics, exact CPU gradients/learning/recovery and frozen protein replay. Production campaign release remains conditional on the exact full optimizer/journal recovery test and sufficient storage. No actor parameter was updated inV93.
3. RL performance: **NOT TESTED**. Neither past100-update stops norV93 frozen baselines establish that adequately trained RL can or cannot solve this task.

Recommendation: authorize the first5,000-update,16-attempt/group fixed-sequence geometry-only on-policy campaign only after at least{required/2**30:.3f}GiB free (prefer8GiB), full execution-config recovery qualification and reviewed release callbacks. Use434 eligible fitting identities (exact census in manifest), hold out the16 TRAIN development identities, unchanged reward/J1/actor capacity, and final prospective K1/K8 evaluation. Preserve finite trace likelihoods, zero categorical terms for supplied AA, original hard validity and independent uncertainty.

The specific prerequisites still missing are storage capacity, future authorization and full protein optimizer/journal release—not another small protein performance pilot. No new critic campaign, longer horizon or hierarchy is justified automatically. Later joint AA/geometry learning is a distinct admitted stage; never claim deterministic motif enforcement as learned AA control.''')
 write('FINAL_E017_V93_SCIENTIFIC_REPORT.md',f'''# Final E017 V9.3 scientific report

**Environment {'GO' if ready['environment_GO'] else 'NOT_ADMITTED'}; full-trace likelihood/gradient primitives GO; substantial-training launch NOT ADMITTED (storage/release); RL performance NOT TESTED.** No protein neural optimizer update occurred. CPU toy learning is an engineering test only.

{table}

The fixed-sequence experiment separates sequence instruction following from regional geometric properties. Native motifs are already satisfied; the model receives known canonical AA identities sequentially, not native geometry. Original reward, local/nonlocal/Rg/packing/sequence criteria and exposure proxy remain unchanged. Native controls all succeed and collapsed/extended negatives have zero reward. Candidate failure/abort counts remain in denominators. D uses exact terminal selection from C; E is bounded TRAIN suffix search, with all simulated proposals and terminal costs retained. No learned policy improvement is claimed.

Delayed interventions:4 TRAIN identities, stages8/32/48,3 valid alternative actions and16 FIT+16 independent confirmation continuations. Results and Monte Carlo uncertainty are in DELAYED_ACTION_EFFECTS.md/JSON. Immediate validity cannot rank already-valid alternatives; terminal effects concern computational delayed outcomes, not biochemical causality. Supported same-sequence regional reversal controls={len(altrows)}, admitted only with observed hard-valid generated witnesses.

Complete proposal traces include rejections and finite aborts. Normalized raw-latent likelihood, exact transformed density masks/Jacobians, prescribed-AA zero categorical terms, KL chain rule and unbiased group-RLOO estimator are tested. Exact toy expected return improved {test['toy_initial_return']:.6f}→{test['toy_final_return']:.6f} after250 engineering updates; no protein-learning conclusion follows. Frozen protein replay matched{back['replayed_proposals']} log-densities with maxerror={back['logprob_max_error']}; all actor/J1 weights unchanged. The failed metadata-boundary and8-from16 replay attempts are retained and corrected without relaxed thresholds.

Prospective panels:32 fixed goals/16 fresh property-policy-evaluation identities plus64 future joint goals, difficulty levels and motif order controls. Identity-clustered uncertainty retains shared identities across levels; prior aggregate reference calibration is disclosed. Joint policy generation remains unexecuted. The16 V93 TRAIN development identities are excluded from the future434-identity fitting pool. TEST remains sealed.

First substantial protocol:5,000 updates×16 generation attempts, later separately admitted joint phase; full5000-update final judgement, not a100/250 performance verdict. Measured complete recovery={back['complete_recovery_measured_bytes']:,}bytes; required free≈{required/2**30:.3f}GiB vs current≈{free/2**30:.3f}GiB. No unsafe campaign launched and no protected evidence deleted. Exact scientific, mathematical, storage and recovery details are in the ten companion deliverables and machine-readable JSON. Production driver is an authorization/storage/recovery-gated release candidate; the exact future optimizer/journal replay remains a release prerequisite, not silently claimed by toy results.

Source commit `{publicationsource}`. Checkpoint and source hashes are sealed in PROVENANCE_SHA256.json. All new files/caches/temp stay onD:. Stop all V93 workers after sealing; no further study is authorized automatically.''')
 # Standalone compact scientific plot.
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(6,4))
 for pol in ['A','B','C']:
  v=results['PROSPECTIVE_'+pol];ax.plot([1,8],[v['K1']/v['goals'],v['K8']/v['goals']],'o-',label=pol)
 ax.set(xlabel='Candidate pool K',ylabel='Full regional goal success',ylim=(0,1));ax.legend();fig.tight_layout();fig.savefig(R/'plots/PROSPECTIVE_SUCCESS.png',dpi=140);plt.close(fig)
 save('V93_FINAL_NUMERICAL_SUMMARY.json',dict(environment_GO=ready['environment_GO'],likelihood_gradient_primitives_GO=True,protein_production_release='PENDING future full optimizer/journal test and authorization',RL_performance='NOT_TESTED',storage_GO=False,required_free_bytes=required,current_free_bytes=free,checkpoint_bytes=back['complete_recovery_measured_bytes'],TRAIN_pool_identities=pool['identities'],results=results,paired=diff,delayed=dd['by_stage'],search=searchrows,alternate=altrows,source_commit=publicationsource,protein_updates=0,study_sampling_seconds=elapsed));print(json.dumps(dict(environment_GO=ready['environment_GO'],free=free,checkpoint=back['complete_recovery_measured_bytes'],prospective={p:{k:results['PROSPECTIVE_'+p][k] for k in ['candidates','K1','K8','marginals']} for p in ['A','B','C']})),flush=True)
if __name__=='__main__':main()
