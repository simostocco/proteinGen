"""Compact final synthesis of the bounded property-steering study."""
from pathlib import Path
import collections,json,math,os,runpy,shutil,subprocess,time
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v9_property_steering_20261023_v1');a=runpy.run_path(str(R/'source/scripts/e017_v9/analysis_common.py'));c=a['c'];read=c['read'];save=c['save'];sha=c['sha'];write=a['write'];ci=a['ci'];summarize=a['summarize']

def criteria(groups):
 out=collections.Counter();den=0
 for obj in groups:
  g=obj['goal']
  for row in obj['records']:
   m=row['metrics'];den+=1
   q=dict(exposed=m.get('exposed_mean',-1)>=g['exposure_min']-1e-10,buried=m.get('buried_mean',2)<=g['burial_max']+1e-10,contrast=m.get('contrast',-1)>=g['contrast_min']-1e-10,motif=m.get('motif_utility',0)==1)
   for k,v in q.items():out[k]+=int(v)
   geom=all(q[k] for k in ['exposed','buried','contrast']);out['all_regional_geometry']+=geom;out['hard_valid_and_regional_geometry']+=int(m['hard_valid'] and geom);out['hard_valid_and_motif']+=int(m['hard_valid'] and q['motif'])
 return dict(denominator=den,component_hits=dict(out))
def matched(g1,g2):
 assert [g['goal']['goal_id'] for g in g1]==[g['goal']['goal_id'] for g in g2];out={}
 for k in [1,2,4,8]:out['K'+str(k)]=ci([int(any(r['metrics']['success'] for r in y['records'][:k]))-int(any(r['metrics']['success'] for r in x['records'][:k])) for x,y in zip(g1,g2)])
 for key in ['backbone_valid','hard_valid']:
  out[key]=ci([np.mean([r['metrics'][key] for r in y['records']])-np.mean([r['metrics'][key] for r in x['records']]) for x,y in zip(g1,g2)])
 return out

def final():
 arms={mode:read(R/(mode+'_COMPLETED.json')) for mode in ['JOINT','AA_ONLY','GEOMETRY_ONLY'] if (R/(mode+'_COMPLETED.json')).exists()};assert arms and all(z['status']=='STOPPED' for z in arms.values());assert read(R/'RL_ENGINEERING.json')['PASS'];spec=read(R/'REWARD_SPECIFICATION.json');eng=read(R/'DE_NOVO_ENGINEERING.json');engineering=read(R/'RL_ENGINEERING.json');tests=read(R/'REWARD_TESTS.json');adm=read(R/'LOW_LEVEL_ADMISSION.json');source=subprocess.check_output(['git','-C',str(R/'source'),'rev-parse','HEAD'],text=True).strip();assert not subprocess.check_output(['git','-C',str(R/'source'),'status','--porcelain'],text=True).strip()
 baselines={panel:{policy:summarize(panel,policy) for policy in ['A_J1','B_V3']} for panel in ['TRAIN','DEV','CONFIRM']};milestones={};comparisons={};groups=a['rows_for'];cal=read(R/'SASA_COMPLETED.json')
 for mode,z in arms.items():
  milestones[mode]={}
  for val in z['milestones']:
   label=f"{mode}_U{val['update']:04}";milestones[mode][str(val['update'])]=summarize(label,'DEV');comparisons[label]=matched(groups('DEV','B_V3'),groups(label,'DEV'))
 fullcheck=[]
 for mode in arms:
  for path in sorted((R/'checkpoints'/mode).glob('update_*.pt')):
   state=torch.load(path,map_location='cpu',weights_only=False);digest=c['state_digest'](state);meta=read(R/'checkpoints'/mode/'LATEST.json');latest=str(path)==meta['path']
   if latest:assert sha(path)==meta['sha256'] and digest==meta['state_digest']
   required=['actor','optimizer','scheduler','AMP','RNG','config','deterministic_sampler','update','counters','task_SHA256','parent_SHA256','J1_SHA256','reward_SHA256'];assert all(k in state for k in required);assert state['parent_SHA256']==c['PSHA'] and state['J1_SHA256']==c['J1SHA'] and state['reward_SHA256']==sha(R/'REWARD_SPECIFICATION.json');assert all(sha(R/f'manifests/{p}_GOALS.json')==h for p,h in state['task_SHA256'].items())
   fullcheck.append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,state_digest=digest,update=state['update'],counters=state['counters'],scientific_stop=state['scientific_stop'],all_recovery_fields_verified=True,latest=latest));del state
 assert sha(c['J1'])==c['J1SHA'] and sha(c['PARENT'])==c['PSHA']
 save('FINAL_CHECKPOINT_VERIFICATION.json',dict(checkpoints=fullcheck,engineering_replay=engineering,source_commit=source,optimizer_recovery='bit-exact interrupted engineering update replay verified; each final bundle deserialized and semantically fingerprinted on CPU',GPU_loading_for_verification=False))
 protocols={p.name:sha(p) for p in R.glob('*PROTOCOL*.json')};ckpt=dict(parent_checkpoint=str(c['PARENT']),parent_SHA256=c['PSHA'],J1_checkpoint=str(c['J1']),J1_SHA256=c['J1SHA'],J1_tensor_fingerprint=eng['J1_tensor_fingerprint'],parent_actor_tensor_fingerprint=eng['parent_actor_fingerprint'],architecture_counts=eng['component_counts'],J1_parameters=12351746,source_commit=source,source_files={str(p.relative_to(R)):sha(p) for p in (R/'source/scripts/e017_v9').glob('*.py')},protocols=protocols,arms=arms,complete_checkpoints=fullcheck,replay=engineering['exact_optimizer_scheduler_RNG_replay'],actor_updates=sum(z['updates'] for z in arms.values()),discarded_engineering_updates=engineering['engineering_updates'],J1_updates=0,critic_updates=0,high_level_updates=0,automatic_restart=False,resume='run.sh source/scripts/e017_v9/train.py resume_<ARM> supports genuine interrupted noncompleted runs only. Completed/scientifically stopped arms fail closed and must not automatically resume.')
 save('CHECKPOINT_AND_RECOVERY_MANIFEST.json',ckpt)
 baselinecomponents={p:criteria(groups('TRAIN',p)) for p in ['A_J1','B_V3']};confirmation=read(R/'CONFIRMATION_COMPARISON.json') if (R/'CONFIRMATION_COMPARISON.json').exists() else dict(status='NOT_EXECUTED',reason='No selected checkpoint qualified for prospective confirmation after scientific safety/continuation gates. Panel remains unexposed by V9 policy evaluation.')
 save('K1_AND_BEST_OF_K_COMPARISONS.json',dict(baselines=baselines,milestones=milestones,matched_identity_clustered_comparisons=comparisons,TRAIN_component_hits=baselinecomponents,confirmation=confirmation,seeds='E017V9|9917 hash-derived goal/candidate streams; BASELINE shared across policies/checkpoints; RL_update shared across arms; independent candidates',nested_K=[1,2,4,8],confirmation_primary='independent held-out property goals; no TEST',failed_attempts_retained=True))
 safety=any('VALIDITY' in z['stop_reason'] or 'KL' in z['stop_reason'] or 'NUMERICAL' in z['stop_reason'] or 'RESOURCE' in z['stop_reason'] for z in arms.values());decision='NO-GO' if confirmation.get('promotion_PASS') is not True else 'GO_FOR_SEPARATELY_BOUNDED_NEXT_REVIEW';events='; '.join(f"{m}: {z['updates']} updates, {z['stop_reason']}" for m,z in arms.items());total_sequences=sum(z['counters']['trajectories'] for z in arms.values());total_tokens=sum(z['counters']['residue_decisions'] for z in arms.values());trainidentities=set()
 traces={}
 for mode in arms:
  traces[mode]=[json.loads(line) for line in (R/f'logs/{mode}_TRAIN.jsonl').read_text().splitlines()];trainidentities.update(v['identity'] for v in traces[mode])
 resource_samples=[json.loads(s) for s in (R/'logs/RESOURCE_SAMPLES.jsonl').read_text().splitlines()];runs={p.name:read(p) for p in R.glob('*BASELINES_COMPLETED.json')};free=shutil.disk_usage(R).free;resource=dict(D_free_bytes=free,D_free_GiB=free/2**30,minimum_reserve_bytes=2*2**30,headroom_PASS=free>=2*2**30,resource_samples=resource_samples,baseline_runs=runs,training_runs=arms,training_compute_accounting={m:dict(accepted_updates=z['updates'],generated_trajectories=z['counters']['trajectories'],generated_residue_decisions=z['counters']['residue_decisions'],additional_gradient_pass_J1_batches=64*z['updates'],additional_gradient_pass_J1_states=512*z['updates'],additional_gradient_pass_actor_batches=64*z['updates'],additional_gradient_pass_actor_states=512*z['updates'],inference_and_TR_probe_calls=z['calls'],backward_activation_checkpoint_block_recomputation='not counted as whole actor forward calls; included in measured wall time') for m,z in arms.items()},engineering=dict(discarded_updates=2,trajectories=16,residue_decisions=1024,peak_GPU_bytes=engineering['peak_GPU_bytes']),all_outputs_caches_temporary_files_on_D=True,one_GPU_workload=True,destructive_cleanup=False,CPU_threads=1,Windows_RAM_check=dict(TotalVisibleMemory_KiB=33420356,FreePhysicalMemory_KiB=11887572,matching_native_research_processes=[],point_sample_UTC='2026-10-23T20:23Z'),checkpoint_complete_bytes=engineering['complete_checkpoint_bytes']);save('RESOURCE_AND_RUNTIME_REPORT.json',resource)
 table='| Model/checkpoint | Goals | K1 full success | K8 full success | Backbone-valid candidates | Hard-valid candidates |\n|---|---:|---:|---:|---:|---:|\n'
 for label,z in [('J1 TRAIN',baselines['TRAIN']['A_J1']),('V3 TRAIN',baselines['TRAIN']['B_V3']),('V3 DEV',baselines['DEV']['B_V3'])]+[(m+' DEV update'+u,v) for m,vals in milestones.items() for u,v in vals.items()]:
  table+=f"| {label} | {z['goals']} | {z['success_by_K']['1']['successful_goals']}/{z['goals']} | {z['success_by_K']['8']['successful_goals']}/{z['goals']} | {z['backbone_valid']}/{z['trajectories']} | {z['hard_valid']}/{z['trajectories']} |\n"
 write('PROPERTY_TASK_AND_GOAL_DISTRIBUTION.md',f'''# V9 coordinate-free regional-exposure task
N64,8-residue exposed region,8-residue shielded region separated by>=16 residues, regional contrast>=0.1 and an exact heterodipeptide motif somewhere in the exposed region. Goals are32 TRAIN,8 DEV,16 CONFIRM distinct protein identities/protected groups, with semantic goal signatures distinct between panels. Goal JSON files were sealed before prospective generation. Native witness identities/coordinates are audit/calibration metadata, not neural features. No native AA prefix or endpoint coordinates are supplied at inference. The initial state is empty; first-three-residue geometry is constructed in a canonical gauge.

Goals are computationally feasible because native64-residue controls satisfy them. They are calibrated from TRAIN/VALIDATION witness statistics, rounded permissively on a0.05 exposure grid. This is not evidence of native endpoint matching, biochemical function or physical foldability. TEST coordinates/tasks remain sealed; the original split loader reads partition membership metadata.

Exact distributions, seeds, constraints and source masks are in SCIENTIFIC_DESIGN_INITIAL.json, REWARD_SPECIFICATION.json and manifests. The confirmation panel has status: {confirmation['status']}. No N32 substitute was needed: N64 frozen-proposal admission passed.
''')
 write('TRAIN_REFERENCE_AND_EXPOSURE_CALIBRATION.md',f'''# Reference and proxy calibration
512 TRAIN and128 VALIDATION distinct identities/groups; one hash-indexed continuous64-residue window per identity. Data receipts and exclusions are in REFERENCE_CENSUS.json and manifests. No PDB dataset copy was created.

The TRAIN-fit-selected proxy is{spec['proxy']}.16 TRAIN fit references selected it;16 other TRAIN and16 VALIDATION references independently assessed it. Frozen criterion: mean per-protein Spearman>=0.5 and identity95% lower>0.3. Results:
{json.dumps(spec['proxy_gates'],indent=2)}

Atom reference uses observed heavy-atom Shrake–Rupley960-point SASA,1.4Å probe, element radii and>=90% complete standard residues. Raw mmCIF hashes and every mapped CA were checked; termini/incomplete/ambiguous residue masks retained. Accessibility is SASA in the isolated64-residue fragment divided by the identical residue atoms alone, not DSSP RSA or full assembly accessibility. Analytic sphere tests passed. Full48 mappings are in SASA_COMPLETED.json.

The original attempt omitted legacy mapping and failed coverage; it remains in calibration_attempts/001_legacy_mapping_missing. Source correction833ffca linked canonical mmCIFs without relaxing admission thresholds, before any policy generation. Optional AA/environment compatibility is a held-identity validated association score with TRAIN permutation sensitivity; it is not a folding predictor.

Original V3 backbone limits are unchanged; their scope is spacing, nonlocal clashes, Rg and nondegenerate frames. They do not establish all-atom steric or energetic validity. AA diversity and contact-density ranges use TRAIN quantiles.
''')
 write('REWARD_VALIDITY_AND_EXPLOITATION_TESTS.md',f'''# Deterministic reward and shortcut controls
PASS={tests['PASS']}, {len(tests['checks'])} checks, failures={tests['failures']}. Native witnesses pass; rigid motions preserve results; collapsed/straight/invalid/nonfinite backbones, homopolymers and degenerate motif-repeat sequences receive0 reward. A locally plausible extended helix does not achieve full success. Missing motif never counts as success. Randomized-AA controls and compatibility shuffle intervals are retained in REWARD_TESTS.json, including any coarse metric successes rather than hiding them.

Reward=hard_valid*(0.5*full_success+0.5*mean bounded utilities), range[0,1]. Invalid candidates receive0, valid non-success at most0.5. Exact success additionally requires both exposure inequalities, contrast and motif. All thresholds/scales were TRAIN-calibrated and frozen before candidate outcomes. No biological function or experimental stability follows from satisfying these computational criteria.

TRAIN frozen-proposal admission passed with119/256 hard-valid actor candidates across31/32 goals and reward SD0.13453, despite zero full successes. That zero-positive limitation was explicitly allowed by the predeclared informative-shaping gate, not reclassified as success. Component counts:
{json.dumps(baselinecomponents,indent=2)}
''')
 native_reference={v['sequence'] for v in read(R/'manifests/TRAIN_REFERENCE.json')['records']+read(R/'manifests/VALIDATION_REFERENCE.json')['records']};bounded_novelty={policy:dict(candidates=sum(len(g['records']) for g in groups('TRAIN',policy)),exact_matches_to640_reference_windows=sum(r['sequence'] in native_reference for g in groups('TRAIN',policy) for r in g['records']),comparison_population=640,not_corpus_wide_homology_or_novelty=True) for policy in ['A_J1','B_V3']};save('BOUNDED_SEQUENCE_IDENTITY_AUDIT.json',bounded_novelty)
 write('FROZEN_DE_NOVO_BASELINES.md',f'''# Empty-prefix baselines
{table}

All K1/2/4/8 use nested prefixes of the same candidate pools. Shared candidate-indexed uniform/Gaussian streams couple policies/checkpoints while candidates remain independently seeded. Best-of-K exact validity/property selection can attain the success-existence bound. Full actions, latent/represented densities, failures and seeds are stored compactly; no full repeated backbone tensors were exported.

V6 native-prefix H4 results are not matched baselines for de novo N64. The V3 selected10000 checkpoint is the original joint AA+geometry policy; the specialized V5D250 H2 actor is not substituted. New property contexts were not pretrained for this actor: its baseline is a test of transfer, not proof of an already trained property-controller.
''')
 training_summary={m:dict(updates=z['updates'],stop=z['stop_reason'],distinct_identities=z['counters']['distinct_TRAIN_identities'],trajectories=z['counters']['trajectories'],residue_decisions=z['counters']['residue_decisions'],clipped_updates=sum(v['clipped'] for v in traces[m]),gradient_norm_mean=float(np.mean([v['gradient_norm'] for v in traces[m]])),max_step_KL=max(v['step_joint_KL_max'] for v in traces[m]),max_density_recompute_error=max(v['density_recompute_max_error'] for v in traces[m]),TRAIN_full_successes=sum(v['full_success'] for v in traces[m]),selected=z['selected']) for m,z in arms.items()};save('LOW_LEVEL_RL_RESULTS.json',training_summary)
 write('LOW_LEVEL_RL_PILOT_AND_ABLATIONS.md',f'''# On-policy delayed-reward pilot
{events}

{table}

RLOO groups8, independently sampled complete de novo trajectories, leave-one-out other7 baseline; fixed TRAIN per-goal SD floor0.05, advantage clip[-5,5]. Exact AA categorical plus transformed conditional geometry log densities, early-residue masks, AdamW1e-6, clip5, constant scheduler; reference AA and per-active-geometry KL coefficients0.1 each. Critic updates=0. J1 and immutable parent remain frozen. Source and initial specification are versioned; ABLATION_ADMISSION_RULES.json controls when factor arms can run. Disabled factors are overridden by immutable-parent outputs, so shared-encoder drift cannot silently modify them.

{json.dumps(training_summary,indent=2)}

Total production exposure: {total_sequences} generated protein trajectories, {total_tokens} residue decisions across{len(trainidentities)} distinct TRAIN identities, separately from optimizer updates. Two engineering updates/16 trajectories were discarded and not counted as production. Exact optimizer/scheduler/RNG/trajectory replay was verified; density recomputation error and trust-region probes are recorded. A safe checkpoint is written before scientific-stop worker exit.

Factor-ablation status: {', '.join(m+': executed' if m in arms else m+': blocked by earlier safety/prerequisite gate' for m in ['AA_ONLY','GEOMETRY_ONLY'])}. A stopped safety gate remains a failure even if its small-panel paired interval includes0. No checkpoint is promoted by soft reward or mean geometry improvement alone.
''')
 write('HIGH_LEVEL_MDP_AND_CORRECTION_ENVIRONMENT.md',f'''# Property MDP and true suffix correction interface
Implemented in core.py using the existing typed BuildState/BuildAction/CA builder and nine-option high-level architecture. Public neural inputs use only generated AA/CA prefix, known property goal/motif, horizon, progress and intervention history. Audit witness identity/group metadata are not neural features. No native endpoint/AA trace is an inference target.

Options: continue, focus exposed, focus shielded, focus motif, rewind4/8/16, regenerate, accept/stop. Rewind restores an exact earlier suffix boundary and RNG snapshot; generation of that suffix is discarded from current state, while intervention history is retained. Regeneration creates a distinct reproducible branch stream. Masked actions fail closed. No arbitrary internal inpainting is implemented. Incomplete termination fails. Gamma1; operational limits must be reported as censoring rather than hidden reward penalties. Focus options change context, not frozen network weights, and do not imply the actor has learned their semantics.

CPU tests passed typed64-residue reconstruction, exact state/RNG restore,48-residue retained prefix after rewind16, pending-action masks, distinct regeneration branch and final accept. This prepares executable option semantics, not a fully admitted learned high-level controller. High-level actor/critics/reachability updates are0. Low-level promotion and independently demonstrated correction advantage were not established: high-level optimization was blocked.
''')
 write('CORRECTION_AND_SEARCH_BASELINE_COMPARISON.md',f'''# Correction/search admission and computation
The executed non-RL selection baseline is nested Best-of-K from complete frozen candidate pools, K1/2/4/8. No extra generation was needed for smallerK. Exact full-success existence is attainable by deterministic goal/validity selection. All sampled and failed candidates are counted.

No corrective-search campaign or high-level pilot was admitted after the bounded low-level study: {events}. There is no empirical evidence here that correction outperforms repeated complete sampling. Engineering-level rewind correctness is not measured scientific benefit. An independently authorized comparison should match simulated residue decisions and actor/J1 evaluations, as well as terminal evaluations and wall time. One64-residue candidate plus four16-residue suffix regenerations has128 newly generated residues, matching two64-residue candidates, but five versus two terminal evaluations; both counts must be reported. Native-prefix H4 IK results are contextual, not matched de novo controls.
''')
 write('REACHABILITY_AND_COUNTERFACTUAL_INTERVENTION_DIAGNOSTICS.md','''# Prefix diagnostics and counterfactual semantics
Prefix interfaces expose remaining generation horizon, current region exposure over generated residues, conservative final-exposure upper bounds, motif progress, existing nonlocal clashes and irreversible violation flags, and suffix rewind history. They never use future coordinates. Future residues can shield existing residues, so current exposure is not treated as final observed exposure. No learned probability is fabricated: estimated_reachability is null.

State/RNG snapshots enable kinematic counterfactual suffix execution and matched continuation streams; the mechanics passed tests. Predictive utility for later outcomes and intervention advantage over matched whole-candidate sampling were not established before the scientific stop. No reachability model was trained or introduced into a learned planner. This leaves the desired decomposition—direct RL, RL with corrections, RL with validated reachability information—unresolved. Deterministic simulator effects are not experimentally established biochemical causality.
''')
 limitations='Small DEV panel8 identities; confirmation16. Candidate replicates are not independent biological identities. Calibration references are isolated64-residue fragments with incomplete-atom masks, not biological-assembly SASA. Coarse CA validity and AA/environment association do not imply folding/function. The selected V3 actor was not pretrained for this property context. TRAIN32 goals are a bounded learning pilot rather than a broad task corpus. Finite-grid support repairs are explicitly reported and prohibited for strict-density training. No TEST evaluation, AlphaFold, biological validation or robotics transfer occurred.'
 write('NEXT_HIERARCHICAL_RL_DECISION.md',f'''# V9 decision: {decision}
{events}

The benchmark has measurable computational constraints, native feasible witnesses and an independently assessed exposure proxy. Frozen de novo generation produces a substantial valid population, so the task passed proposal admission. Full conjunction success is rare: frozen TRAIN J1 and actor each had0/256 full successes. Actor component hits show why simultaneous motif, regional structure and hard validity matter; no single low-position-error surrogate can qualify it.

1. Meaningful property task? Yes as a bounded sequence/CA computational benchmark; no biochemical function claim.
2. Plausible initial candidates without native targets? Some satisfy the declared backbone/sequence/packing rules (actor119/256); this is not experimental foldability.
3. Low-level RL improvement? See matched DEV/confirmation results; a soft-reward rise alone is not accepted. No promotion without the frozen primary K1/validity/KL/generalization criteria.
4. Joint sequence–geometry value? Only executed, matched factor arms can answer; blocked arms leave this unresolved.
5. Corrections beyond sampling? Not established; only true suffix mechanics were verified.
6. Long-horizon dependencies? Exposure and nonlocal clashes depend on later residues, but useful learnable intervention credit was not demonstrated.
7. High-level RL justified? No on this evidence. A future separately bounded, matched correction/search feasibility study is the simplest next diagnostic before training a hierarchy. If validity failed, investigate the safety trade-off first; do not relax the hard rules or simply add updates.
8. Robotics transfer? Requires a separate dynamics/observation/action benchmark with delayed intervention consequences, non-RL controls, held-out transfer and physical validation. No current result demonstrates transfer.

{limitations}
No V10 or automatic follow-on study was launched.
''')
 final=f'''# E017 V9 final scientific report
Decision: {decision}. {events}.

{table}

## Executed stages and scientific result
Recovered frozen J1 and original joint V3 selected actor; isolated versioned D: branch/worktree.512 TRAIN/128 VALIDATION reference identities;48 all-atom proxy references;32TRAIN/8DEV/16CONFIRM coordinate-free property specifications;329 reward/recovery checks; exact empty-prefix inference replay; frozen TRAIN and DEV de novo baselines; admitted bounded on-policy delayed-reward learning. Total production{sum(z['updates'] for z in arms.values())} accepted actor updates,{total_sequences} sampled trajectories,{total_tokens} residue decisions,{len(trainidentities)} distinct TRAIN identities. Engineering2 discarded updates counted separately. All critics, pretrained J1 and original parent remain frozen.

The original exposure-reference attempt failed legacy mapping coverage and was archived. Verified raw-source mapping fixed this before generation, without changing scientific gates. The selected smooth10 C-alpha exposure proxy independently achieved heldTRAIN meanSpearman{spec['proxy_gates']['HOLD']['correlation']['estimate']:.6f} and VALIDATION{spec['proxy_gates']['VALIDATION']['correlation']['estimate']:.6f},16 identities each, identity intervals in the calibration report.329 reward/recovery tests passed. Frozen actor admission had119/256 hard-valid samples,31/32 goals represented and reward SD0.13453; full success was0/256, retained honestly as a sparse-conjunction limitation.

## Learning, uncertainty and blocked stages
All K1/2/4/8 are nested candidate pools with common random streams across checkpoints. Development identities were fixed; per-protein paired intervals are in K1_AND_BEST_OF_K_COMPARISONS.json. Small-panel uncertainty does not overturn a prespecified safety stop. TRAIN reward, diversity or positional/proxy changes alone are not promotion. Confirmation status:{confirmation['status']}. No generalization gain is claimed without an independent confirmation result.

{', '.join(m+': '+('executed' if m in arms else 'blocked') for m in ['JOINT','AA_ONLY','GEOMETRY_ONLY'])}. No SAC, critic learning, reachability training, high-level policy optimization or correction/search campaign was launched after failed prerequisites. High-level option/rewind semantics passed engineering checks only. N32 controls were unnecessary because the primary N64 proposal gate passed. No native sequence prefix or native target coordinate trace was supplied during de novo generation.

## Checkpoint/source recovery
J1 SHA256 {c['J1SHA']}; original V3 parent SHA256 {c['PSHA']}. Exact paths, parameter counts, tensor fingerprints and every run-owned checkpoint hash are in CHECKPOINT_AND_RECOVERY_MANIFEST.json and FINAL_CHECKPOINT_VERIFICATION.json. Complete actor/optimizer/scheduler/RNG/sampler/exposure/config/goal/reward provenance was atomically serialized and read back before worker exit. AMP is disabled. No critics/replay buffer are required by this on-policy algorithm. Fixed streams and recorded update cursor reconstruct the next batch exactly. Interrupted engineering update replay was bit exact. Scientifically stopped/completed arms fail closed on automatic resume.

Final source commit {source}; actual production training implementation provenance is versioned separately from the initial frozen scientific protocol. The V6–V8 H2-specialized parent was not silently substituted for the original V3 joint actor. Native-prefix H4 metrics are not matched de novo N64 controls.

## Resources and safeguards
Measured complete recovery size{engineering['complete_checkpoint_bytes']} bytes; dynamic admission preserves2GiB reserve. D: free at analysis{free} bytes ({free/2**30:.6f}GiB). One GPU workload; CPUthreads1; no swap at recorded snapshots. Actor17,438,114/J1 12,351,746 parameters, capacity unchanged. Full calls, checkpoint recomputation caveat, trajectories, wall time and resource snapshots are in RESOURCE_AND_RUNTIME_REPORT.json. All project outputs/cache/temp remain on D:. Historical E016/E017 evidence and datasets were not deleted or modified. Final process/GPU idle evidence is FINAL_WORKER_STATUS.json.

## Limits and next decision
{limitations}
Recommend a separately reviewed correction/search feasibility and safety diagnostic before hierarchical RL. Preserve the frozen property task and safeguards; do not infer success from higher proxy reward, continue training unconditionally or initiate V10.
'''
 write('FINAL_E017_V9_SCIENTIFIC_REPORT.md',final)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(12,3.5))
 for mode,tr in traces.items():
  x=[v['update'] for v in tr];axes[0].plot(x,[np.mean([w['reward_mean'] for w in tr[max(0,i-9):i+1]]) for i in range(len(tr))],label=mode)
  z=arms[mode]['milestones'];axes[1].plot([0]+[v['update'] for v in z],[0]+[v['K1']/8 for v in z],marker='o',label=mode);axes[2].plot([0]+[v['update'] for v in z],[baselines['DEV']['B_V3']['backbone_valid']/64]+[v['backbone_valid'] for v in z],marker='o',label=mode)
 axes[0].set(xlabel='Actor updates',ylabel='TRAIN reward, trailing10groups');axes[1].set(xlabel='Actor updates',ylabel='DEV K1 success',ylim=(-.01,1));axes[2].set(xlabel='Actor updates',ylabel='DEV backbone validity',ylim=(0,1));axes[2].axhline(baselines['DEV']['B_V3']['backbone_valid']/64-.05,color='red',linestyle='--',label='stop boundary');axes[0].legend();axes[2].legend();fig.tight_layout();fig.savefig(R/'plots/RL_LEARNING_AND_SAFETY.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(6,3.5))
 for mode,vals in milestones.items():
  for u,v in vals.items():ax.plot([1,2,4,8],[v['success_by_K'][str(k)]['estimate'] for k in [1,2,4,8]],marker='o',label=mode+'U'+u)
 ax.set(xlabel='K',ylabel='DEV full valid goal success',ylim=(-.01,1));ax.legend(fontsize=7);fig.tight_layout();fig.savefig(R/'plots/SUCCESS_VS_K.png',dpi=160);plt.close(fig)
 # Input report/checkpoint provenance is referenced without copying historical artifacts.
 reports={}
 for relative in ['e017_crisp_v3_throughput_20261021_v1/campaign/FINAL_SUPERVISED_INITIALIZATION_REPORT.md','e017_v4_multistep_20261022_v1/FINAL_E017_V4_MULTISTEP_REPORT.md','e017_v5_goal_observability_20261022_v1/FINAL_E017_V5_SCIENTIFIC_REPORT.md']:
  original=R.parent/relative
  if original.exists():reports[str(original)]=sha(original)
 write('HISTORICAL_SYSTEM_RECOVERY.md',f'''# Recovered V3–V8 components and scope
Frozen J1:12,351,746 parameters,width352,8layers,8heads. Independent V3 actor:17,438,114,width448,7layers,8heads; strict state loading verified. The existing full system is63,389,056 trainable-capacity parameters including twin low-level critics, high-level actor/twin critics and reachability module. V9 trains only the independent actor when admitted, preserving every other network. High-level architecture has9 options and6,378,769 parameters; it is not a qualified pretrained planner.

Original V3 selected10000 checkpoint preserves sequence likelihood and joint categorical/conditional geometry capacity; it is the V9 parent, not a property-qualified model. V4 showed modest geometric-error improvement without full-success improvement. V5 demonstrated H2 goal-relative benefit but H4 safety failure; V5.1 variants failed KL/validity safeguards. V6 finite sampling recovers39/63 H4 tasks atK32, without learning. V7 RWR did not improve K1 and critics failed; V7.1 small MLP beat large critics in RMSE without sufficient action-ranking evidence; V7.2 modest action-value signal did not qualify SAC. V8 exact geometric planning was useful on static deterministic H4 but did not establish an RL-specific capability. None is a matched de novo64-residue property baseline.

Reused modules: typed BuildState/BuildAction, CA builder/early masks, FrozenPrior, LowActor and existing high-level output vocabulary size, ragged prefix batching, Gaussian transforms/Jacobians, analytic categorical/conditional KL and atomic checkpoint/readback/storage utilities. AA action is20-way categorical; conditional geometry has3latent Gaussians, transforms exp,pi*sigmoid,pi*tanh. Undefined early components are masked. Represented finite-grid repairs are measured and forbidden for strict-density learning. No architecture capacity increase, SAC implementation rebuild or J1 change occurred.

The new layer consists of coordinate-free regional property tasks, reward calibration, empty-prefix generation, bounded on-policy RLOO and suffix-correction semantics. Old native endpoint targets are not supplied. Exact source/parent hashes are in the final manifests; historical reports remain in their original locations.
''')
 for root in R.parent.glob('e017_v*'):
  if root==R:continue
  for pth in root.glob('FINAL*SCIENTIFIC_REPORT.md'):reports[str(pth)]=sha(pth)
 artifacts={str(pth.relative_to(R)):sha(pth) for pth in R.rglob('*') if pth.is_file() and pth.name!='PROVENANCE_SHA256.json' and not pth.name.endswith('.part') and 'runtime' not in pth.parts and 'source' not in pth.parts}
 save('PROVENANCE_SHA256.json',dict(schema='V9_final_v1',source_commit=source,source_files=ckpt['source_files'],input_checkpoints={str(c['J1']):c['J1SHA'],str(c['PARENT']):c['PSHA']},historical_reports=reports,artifact_SHA256=artifacts,design_SHA256=sha(R/'SCIENTIFIC_DESIGN_INITIAL.json'),no_self_hash=True));save('FINALIZATION_STATUS.json',dict(status='SCIENTIFIC_ANALYSIS_PUBLISHED',decision=decision,training_workers='must independently verify after analysis exits',automatic_restart=False));print(json.dumps(dict(decision=decision,arms={m:dict(updates=z['updates'],stop=z['stop_reason']) for m,z in arms.items()},D_free_bytes=shutil.disk_usage(R).free)))
if __name__=='__main__':final()
