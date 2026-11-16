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
  a=[int(any(r['metrics']['success'] for r in g['records'][:k])) for g in groups];out['success_by_K'][str(k)]=dict(successful_goals=sum(a),denominator=len(a),**ci(a))
 for key in ['exposed_mean','buried_mean','contrast','motif_utility','compatibility_score']:
  z=[m[key] for m in metrics if m.get(key) is not None];out[key]=dict(mean=float(np.mean(z)),SD=float(np.std(z))) if z else None
 for key in ['AA_KL','geometry_KL_per_active']:
  z=[v for r in rows for v in (r[key][1:] if key.startswith('geometry') else r[key])];out[key]=dict(mean=float(np.mean(z)),max=float(np.max(z))) if z else None
 entropy=[m['sequence']['entropy_bits'] for m in metrics];out['sequence_entropy_bits_mean']=float(np.mean(entropy));pid=[];rms=[]
 for g in groups:
  rs=g['records']
  for i in range(len(rs)):
   for j in range(i):
    a,b=rs[i],rs[j]
    if len(a['sequence'])!=len(b['sequence']):continue
    pid.append(sum(x==y for x,y in zip(a['sequence'],b['sequence']))/len(a['sequence']))
    def xyz(r):
     world=c['PropertyWorld'](g['goal'],read(R/'REWARD_SPECIFICATION.json'),r['seed'])
     for aa,geom in zip(r['sequence'],r['geometry']):world.append(aa,geom)
     return world.state.ca.numpy()
    if len(a['geometry'])==g['goal']['N'] and len(b['geometry'])==g['goal']['N']:
     x,y=xyz(a),xyz(b);x=x-x.mean(0);y=y-y.mean(0);u,_,v=np.linalg.svd(x.T@y);D=np.eye(3);D[2,2]=np.linalg.det(u@v);rms.append(float(np.sqrt(np.square(x@(u@D@v)-y).sum(1).mean())))
 out['within_goal_pairwise_sequence_identity_mean']=float(np.mean(pid)) if pid else None;out['within_goal_Kabsch_CA_RMSD_A_mean']=float(np.mean(rms)) if rms else None
 return out

def publish():
 assert not read(R/'LOW_LEVEL_ADMISSION.json')['PASS'],'This finalizer may only publish the stopped-at-admission study.'
 adm=read(R/'LOW_LEVEL_ADMISSION.json');spec=read(R/'REWARD_SPECIFICATION.json');tests=read(R/'REWARD_TESTS.json');eng=read(R/'DE_NOVO_ENGINEERING.json');census=read(R/'REFERENCE_CENSUS.json');cal=read(R/'SASA_COMPLETED.json');commit=subprocess.check_output(['git','-C',str(R/'source'),'rev-parse','HEAD'],text=True).strip();status=subprocess.check_output(['git','-C',str(R/'source'),'status','--porcelain'],text=True).strip();assert not status
 results={panel:{p:summarize(panel,p) for p in ['A_J1','B_V3']} for panel in ['TRAIN','DEV','CONFIRM','N32']};paired={}
 for panel in ['TRAIN','DEV','CONFIRM']:
  a,b=rows_for(panel,'A_J1'),rows_for(panel,'B_V3')
  if a and b:
   assert [x['goal']['goal_id'] for x in a]==[x['goal']['goal_id'] for x in b]
   paired[panel]={str(k):ci([int(any(r['metrics']['success'] for r in y['records'][:k]))-int(any(r['metrics']['success'] for r in x['records'][:k])) for x,y in zip(a,b)]) for k in [1,2,4,8]}
 save('DE_NOVO_BASELINE_COMPARISONS.json',dict(results=results,paired_actor_minus_J1=paired,selection='exact validity plus joint property/motif scoring; successful candidate existence is attainable because deterministic complete criteria are known',K=[1,2,4,8],seed_namespace='E017V9|9917|BASELINE|goal_id|candidate_index, shared uniform/normal streams across policies; N32_CONTROL separate',sample_failures_retained=True,scientific_status='NO_GO_FOR_RL'))
 a=results['TRAIN']['A_J1'];b=results['TRAIN']['B_V3'];n32=results['N32'];gate_text='; '.join(adm['reasons']);table='| Policy | Complete | Backbone valid | Combined hard valid | Full successes | K1 goals | K8 goals |\n|---|---:|---:|---:|---:|---:|---:|\n'
 for label,z in [('Frozen J1',a),('Frozen V3 actor',b)]:table+=f"| {label} | {z['sequence_complete']}/{z['trajectories']} | {z['backbone_valid']}/{z['trajectories']} | {z['hard_valid']}/{z['trajectories']} | {z['full_success']}/{z['trajectories']} | {z['success_by_K']['1']['successful_goals']}/{z['goals']} | {z['success_by_K']['8']['successful_goals']}/{z['goals']} |\n"
 inputs=read(R/'INITIAL_INPUT_AUDIT.json');sourcehash={str(p.relative_to(R)):sha(p) for p in (R/'source/scripts/e017_v9').glob('*.py')};inputfiles={str(c['J1']):c['J1SHA'],str(c['PARENT']):c['PSHA']};original_reports={}
 for name in ['e017_v6_frozen_parent_candidates_20261023_v1/FINAL_E017_V6_SCIENTIFIC_REPORT.md','e017_v7_amortized_lowlevel_20261023_v1/FINAL_E017_V7_SCIENTIFIC_REPORT.md','e017_v71_expected_return_calibration_20261023_v1/FINAL_E017_V71_SCIENTIFIC_REPORT.md','e017_v72_counterfactual_ranking_20261023_v1/FINAL_E017_V72_SCIENTIFIC_REPORT.md','e017_v8_constrained_planning_20261023_v1/FINAL_E017_V8_SCIENTIFIC_REPORT.md']:
  p=R.parent/name
  if p.exists():original_reports[str(p)]=sha(p)
 checkpoint=dict(status='STOPPED_BEFORE_OPTIMIZATION',actor_updates=0,critic_updates=0,high_level_updates=0,J1_updates=0,inputs=inputfiles,actor_tensor_fingerprint=eng['parent_actor_fingerprint'],J1_tensor_fingerprint=eng['J1_tensor_fingerprint'],source_commit=commit,source_files=sourcehash,source_clean=True,model_counts=eng['component_counts'],no_new_optimizer_checkpoint='No optimizer was instantiated: scientific admission failed before training. Original selected checkpoint remains the only parent and is unchanged.',recovery_verified=dict(typed_state_RNG_export_restore=True,true_suffix_rewind_regeneration=True,deterministic_sample_geometry_replay=True,atomic_candidate_records_with_SHA256=True),sampler='candidate-indexed independent numpy PCG64 streams; shared CRN across policies; seed9917; no TRAIN cursor needed after completed inference',restart='No automatic restart or continuation is authorized after failed scientific admission.',future_training_staging='Not attempted: complete actor/optimizer/scaler/RNG serialization-size admission is mandatory before any future training; this run did not pass its earlier scientific gate.')
 # Verify compact committed records, not a full unrelated historical scan.
 checks=[]
 for p in sorted((R/'results').glob('TRAIN_*.sha.json')):
  candidate=p.with_suffix('').with_suffix('.json');expected=read(p)['sha256'];assert sha(candidate)==expected;checks.append(str(candidate.relative_to(R)))
 checkpoint['verified_committed_candidate_files']=checks;save('CHECKPOINT_AND_RECOVERY_MANIFEST.json',checkpoint)
 runs={p.name:read(p) for p in R.glob('*COMPLETED.json') if p.name in ['TRAIN_BASELINES_COMPLETED.json','N32_CONTROLS_COMPLETED.json']};samples=[json.loads(s) for s in (R/'logs/RESOURCE_SAMPLES.jsonl').read_text().splitlines()] if (R/'logs/RESOURCE_SAMPLES.jsonl').exists() else [];allocated=sum(p.stat().st_blocks*512 for p in R.rglob('*') if p.is_file());resource=dict(D_free_bytes=shutil.disk_usage(R).free,D_free_GiB=shutil.disk_usage(R).free/2**30,operational_reserve_bytes=2*2**30,headroom_PASS=shutil.disk_usage(R).free>=2*2**30,experiment_allocated_bytes_including_source=allocated,runs=runs,resource_samples=samples,one_heavy_GPU_worker=True,all_project_outputs_caches_temp_on_D=True,deleted_files=0,provenance_sampling='resources are point samples, not continuous profiler traces',checkpoint_staging_not_exercised='No training admitted; no claim of optimizer replay verification',GPU_engineering_peak_allocated=eng['peak_GPU']);save('RESOURCE_AND_RUNTIME_REPORT.json',resource)
 write('PROPERTY_TASK_AND_GOAL_DISTRIBUTION.md',f'''# E017 V9 property task v1
N=64, no native coordinate target and no native sequence prefix. The 32 TRAIN, 8 development and 16 confirmation property goals were sealed before policy outcomes. Goals request an exposed eight-residue region, a second shielded region, regional exposure contrast at least0.1 and a heterodipeptide anywhere in the exposed region. Region separation is at least16 residues. Exact properties and seeds are in manifests/*_GOALS.json. Witness coordinates are private calibration controls only; actor input consists of generated prefix, motif, scalar properties and regions, never witness sequence/coordinates. Identity/group metadata are retained for audit and are not neural features.

Goal targets are rounded permissively on a0.05 grid from feasible TRAIN/VALIDATION witnesses, after TRAIN-derived backbone/sequence/packing filters. Feasibility witnesses therefore establish computational consistency; they do not prove de novo proposal feasibility. Proxy choice uses16 TRAIN fit identities;16 disjoint TRAIN hold identities and16 VALIDATION identities assess admission. All goal signatures are unique across panels; DEV/CONFIRM identities exclude atom-reference identities. TRAIN goals come from the512-reference population, not an independent biological validation set. No TEST structures/tasks were read or evaluated; existing partition membership metadata are loaded by the original data loader.

The prospective confirmation panel was not evaluated because the earlier TRAIN admission failed. N32 feasibility controls are separate TRAIN-only measurements and cannot replace the N64 endpoint.

The initial design is SCIENTIFIC_DESIGN_INITIAL.json. Reward scales/thresholds are REWARD_SPECIFICATION.json. No thresholds were changed after prospective generation.
'''.replace("{'", "{'"))
 write('TRAIN_REFERENCE_AND_EXPOSURE_CALIBRATION.md',f'''# TRAIN reference and exposure-proxy calibration
512 TRAIN and128 VALIDATION distinct identities/protected groups, one hash-selected continuous64-residue window per identity; observational-quality filters and masks preserved. Exact coordinate receipts, exclusions, sequences and metadata are in REFERENCE_CENSUS.json and manifests/*_REFERENCE.json. No dataset copy was made.

Selected proxy: {spec['proxy']}. Selection used16 TRAIN fit identities; admission used16 other TRAIN and16 VALIDATION identities. Frozen admission: mean within-protein Spearman>=0.5 and identity-bootstrap95% lower>0.3.

{json.dumps(spec['proxy_gates'],indent=2)}

Independent atom reference: Shrake–Rupley960 deterministic sphere points,1.4Å probe, element-specific heavy-atom radii; each residue accessibility is its observed atom SASA within the same64-residue fragment divided by the SASA of its identical isolated residue atoms. It is not DSSP relative ASA or full-assembly exposure. At least90% complete standard heavy-atom residues required; termini and incomplete/ambiguous residues masked. Analytic single-atom and overlapping-sphere checks passed. Every source compressed-file SHA and CA mapping was checked against the certified coordinate window. SASA_COMPLETED.json contains all48 references and rejected mappings.

The first attempt omitted legacy atom mapping and failed coverage. It is archived unchanged under calibration_attempts/001_legacy_mapping_missing. Commit833ffca added raw-file-hash and residue-label verification without changing the scientific criteria. The corrected full audit replaced no policy outcomes: no generation had yet occurred.

Optional AA/environment compatibility is an association statistic, enabled only after positive held-TRAIN and VALIDATION likelihood gains over the marginal-AA model and TRAIN permutation sensitivity. Details: REWARD_SPECIFICATION.json and REWARD_TESTS.json. It does not establish foldability.
''')
 write('REWARD_VALIDITY_AND_EXPLOITATION_TESTS.md',f'''# Reward validity and exploitation checks
PASS={tests['PASS']}; {len(tests['checks'])} checks, {len(tests['failures'])} failures. All32 native computational witnesses satisfied their declared properties and hard conditions. Rigid motion preserved reward. Collapsed, straight-rod, broken/nonfinite backbone, homopolymer and repeated-motif controls received zero reward. A locally valid extended helix never achieved full success and received at most0.5. Removing the motif never counted as full success. Random AA permutations were explicitly measured and retained in REWARD_TESTS.json; coarse metric satisfaction is not biological validation.

Hard success requires N64 complete canonical sequence, unchanged original V3 backbone limits, TRAIN-calibrated sequence and packing range, both exposure constraints, contrast and motif. The bounded reward is hard-valid*(0.5*success+0.5*mean bounded utilities). It lies in[0,1]; invalid candidates get0, non-successful valid candidates get at most0.5. Compatibility is optional, independently checked, and not a hard-success condition.

The native-witness and adversarial tests validate this declared computational task, not exhaustive resistance to all reward exploitation. Packing lower quantile is{spec['packing_limits']['64'][0]} contacts: this population does not justify asserting compact globular folds from the packing rule. Raw controls are published; no external folding or experimental claims are made.

CPU engineering verifies canonical CA reconstruction, exact state/RNG restore, true48-residue retained prefix after rewind16, independent regeneration branch, masked high actions and final acceptance. Frozen inference independently verifies empty-prefix sampling and exact sequence/geometry replay. No likelihood-bearing action was silently projected; represented numerical repairs are counted and prohibit training admission.
''')
 write('DE_NOVO_BASELINE_COMPARISON.md',f'''# Frozen de novo baseline census
{table}

TRAIN32 independent goal/witness identities,8 candidates each,512 complete-or-failed attempts total. No PDB prefix or native endpoint was supplied. K1/2/4/8 are nested prefixes of identical candidate pools, with coupled candidate random streams across frozen J1 and actor. Best-of-K uses complete deterministic property/validity evaluation; exact success existence is therefore an attainable selector bound. Failed samples are retained.

These are TRAIN admission diagnostics, not an independent confirmation of generalization. Paired identity-bootstrap intervals and full candidate/constraint details are in DE_NOVO_BASELINE_COMPARISONS.json and results/TRAIN_*.json. An all-zero bootstrap interval is not a population guarantee. No held-out gain is claimed.

Frozen V3 actor actor-state fingerprint: {eng['parent_actor_fingerprint']}. J1 tensor fingerprint: {eng['J1_tensor_fingerprint']}. V3 actor was trained for joint sequence/geometry behavior; its new property context was not previously trained. Native-goal H4 V6/V8 results cannot be used as matched de novo N64 baselines.

N32 control, TRAIN-only and K1 only:
{json.dumps(n32,indent=2)}
''')
 write('LOW_LEVEL_RL_PILOT_AND_ABLATIONS.md',f'''# Low-level RL admission: NO-GO
No actor, critic, J1 or high-level updates occurred. No SAC was launched.

Frozen TRAIN admission failed: {gate_text}.

{json.dumps(adm,indent=2)}

The predeclared critic-free episodic RLOO pilot was therefore blocked before optimizer construction. Joint AA+geometry, AA-only and geometry-only arms were not executed; they have no measured learning result. Declining proxy error or prior H4 candidate successes cannot override this gate. Complete training-checkpoint staging and optimizer replay were not tested, because the earlier scientific prerequisite failed. Future training would require those engineering gates as well.

VALIDATION DEV/CONFIRM policy evaluations and trained-policy comparisons remain unexecuted and scientifically unexposed by V9. N32 results are only feasibility controls, not a changed primary task or a training admission pass.
''')
 write('HIGH_LEVEL_MDP_AND_CORRECTION_ENVIRONMENT.md',f'''# Frozen-low-level property MDP and true suffix corrections
Implemented in source/scripts/e017_v9/core.py: variable-length generated AA+CA state, empty initial prefix, known property requirements, generated-prefix observables and intervention history. No native endpoint coordinates enter the transition or reward. Existing high-level architecture has nine option outputs; its parameters were not trained or used for learned decisions.

Options: continue; focus exposed region; focus buried region; focus motif; rewind4,8,16; regenerate; accept/terminate. Rewind restores the typed state and RNG snapshot at an earlier suffix boundary and deletes the generated suffix from the current state; regeneration uses a distinct recorded branch stream. Internal inpainting is unsupported. Terminating an incomplete sequence is failure. Gamma=1; operational intervention caps must be reported as censoring, not hidden reward penalties. Context options are executable semantics, not evidence that the frozen actor understands property steering.

Reward/MDP tests verify exact retained-prefix coordinates, masks, branch changes, state/RNG export/restore and final acceptance. No learned high-level rollout or optimizer was run. High-level training requires a qualified low-level policy, correction advantage over matched Best-of-K and validated future-outcome information; none was established by this stopped study.
''')
 write('CORRECTION_SEARCH_BASELINE_COMPARISON.md',f'''# Corrective search admission
Not executed. The earlier frozen-proposal admission failed ({gate_text}); conducting an optimization/search campaign or high-level learning would bypass the declared stop rule. Best-of-K was computed from the existing TRAIN candidate pools; no extra neural evaluations were needed for K1/2/4/8.

A future admitted comparison should count all actor/J1 state evaluations, simulated residues, completed/failed candidates and wall time. A valid matched suffix comparison could use one64-residue generation plus four16-residue regenerations versus two64-residue independent candidates, both128 residue decisions, while separately reporting five versus two terminal evaluations. This is a specification, not an observed correction benefit. No correction success, causal benefit or RL-specific capability is claimed.
''')
 write('REACHABILITY_AND_COUNTERFACTUAL_DIAGNOSTICS.md','''# Prefix information and long-term diagnostics
Implemented interfaces expose remaining horizon, current region-exposure estimates, conservative exposure bounds, motif progress, irreversible nonlocal-clash violations and rewind history. Only generated coordinates are used. Added future residues can reduce current exposure; missing region residues contribute a conservative upper bound of1. No learned feasibility probability is returned: estimated_reachability is null.

Counterfactual execution is supported through exact typed state/RNG snapshots and true suffix regeneration. These mechanics passed CPU correctness tests; empirical intervention-versus-final-outcome predictive benefit was not assessed because proposal admission failed. No reachability model, high-level policy or critic was trained. The simulator does not establish biochemical causality or transfer to robotics.
''')
 write('NEXT_HIERARCHICAL_RL_DECISION.md',f'''# E017 V9 decision: NO-GO for policy optimization and hierarchical RL
The property benchmark is measurable and has native computational witnesses plus an independently assessed exposure proxy. The frozen de novo proposals do not pass their prospectively frozen admission criteria: {gate_text}.

The next scientific intervention should address the measured de novo validity/proposal deficit before RL. First audit failed chains by first irreversible spacing/clash/sequence degeneration and compare joint-free generation against observed valid fragments without native-coordinate targets. A separately authorized, constrained proposal/bootstrap study could test whether realistic de novo support can be recovered while preserving J1 frozen. Do not relax task validity thresholds or claim that H4 native-prefix successes solve this problem. More critic updates or simultaneous63M-parameter training are not supported.

Answers: (1) meaningful as a coarse computational property task, with proxy/fragment limitations; (2) see exact frozen candidate validity below, insufficient for the RL gate; (3) low-level RL improvement unknown, no updates; (4) joint versus single-factor training value unknown, arms blocked; (5) correction benefit unknown, only executable mechanics verified; (6) exposure and clashes can depend on later residues, but useful learnable long-horizon decision credit was not demonstrated; (7) high-level RL not yet justified; (8) robotics transfer would require an independently defined dynamics/observation/action problem, delayed intervention outcomes, matched non-RL baselines and held-out transfer tests. No such transfer result exists here.

No V10 or further campaign was launched.
''')
 final=f'''# E017 V9 final scientific report
Status: COMPLETED_BOUNDED_STUDY — NO-GO FOR RL. Stopped at the frozen TRAIN proposal/reward-variation admission; no optimization occurred.

{table}

## Scientific decision
{gate_text}. Numerical admission, exact counters and all failure causes are in LOW_LEVEL_ADMISSION.json and DE_NOVO_BASELINE_COMPARISONS.json. No held-out policy improvement, foldability, biochemical function or learned hierarchy is claimed. N64 remains the primary task; N32 controls are separate.

## Executed stages
Recovered original joint AA+geometry V3 selected actor and frozen J1; isolated V9 branch/source on D:;512 TRAIN/128 VALIDATION reference identities;48 independently mapped atom references; TRAIN-only goal/reward calibration;32 TRAIN,8 DEV,16 CONFIRM coordinate-free property specifications sealed;329 reward/recovery tests; deterministic empty-prefix GPU sampling;512 TRAIN candidate attempts and nested K1/2/4/8; bounded separate N32 feasibility controls where recorded. No native sequence prefix or native endpoint was supplied to inference.

Exposure proxy {spec['proxy']} passed original gates: heldTRAIN meanSpearman{spec['proxy_gates']['HOLD']['correlation']['estimate']:.6f}, VALIDATION{spec['proxy_gates']['VALIDATION']['correlation']['estimate']:.6f}, each16 identities; intervals in calibration report. This references isolated64-residue-fragment observed-heavy-atom accessibility, not full-assembly SASA or validated physical folding. The initial legacy-mapping coverage failure and source correction are archived, not erased.

## Blocked stages
RL pilot, joint/AA-only/geometry-only optimization, prospective DEV/CONFIRM policy evaluation, corrective-search campaign, empirical reachability prediction and high-level learning were blocked by the prior scientific gate. No SAC, TEST evaluation, AlphaFold, J1 update, destructive cleanup or V10 occurred. MDP and rewind interfaces passed engineering tests but have no learned control result.

## Models and provenance
Source commit {commit}, isolated branch e017-v9-property-steering-20261023-v1, clean at publication. Frozen J1 SHA256 {c['J1SHA']}; parent V3 selected10000 SHA256 {c['PSHA']}. Exact paths/fingerprints/counts in CHECKPOINT_AND_RECOVERY_MANIFEST.json. Actor17,438,114 and J1 12,351,746 parameters. The specialized V5D250 checkpoint used by V6–V8 was not substituted as an approved joint-property policy. No new trainable model checkpoint was needed because no optimizer was admitted.

## Uncertainty and limitations
All de novo outcomes are TRAIN admission measurements, clustered by32 goal/protein identities; multiple candidates are not independent biological evidence. DEV/CONFIRM policy outcomes were not inspected. Exact per-candidate seeds/actions/densities/constraints and pairwise intervals are retained. No corpus-wide novelty, homology-free generated sequence, experimental stability, function or robotics transfer is demonstrated. The new property context is untrained and may be out of the selected actor's original conditioning distribution; that is a limitation of this frozen baseline, not proof that future property RL cannot work.

## Resources and stop
One GPU workload at a time; compact action records, no repeated dataset/backbone exports. All project outputs/cache/temp are on D:. D: free at report creation{resource['D_free_bytes']} bytes ({resource['D_free_GiB']:.6f}GiB), above2GiB reserve. Final idle verification is in FINAL_WORKER_STATUS.json; resource snapshots and measured call/runtime totals are in RESOURCE_AND_RUNTIME_REPORT.json. No supervisor or automatic restart was created for the baseline commands. Historical evidence remains unchanged.

Recommendation: repair and scientifically validate de novo proposal support under the unchanged property task before authorizing low-level or hierarchical RL. Stop and await a separate research decision.
'''
 write('FINAL_E017_V9_SCIENTIFIC_REPORT.md',final)
 # Reuse compact numerical data for plots; no additional simulation.
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,2,figsize=(9,3.4))
 for label,z in [('J1',a),('V3 actor',b)]:
  ks=[int(k) for k in z['success_by_K']];axs[0].plot(ks,[z['success_by_K'][str(k)]['estimate'] for k in ks],marker='o',label=label)
 axs[0].set(xlabel='K',ylabel='TRAIN goal success',ylim=(-.01,1),title='Nested frozen candidate pools');axs[0].legend();xx=np.arange(3);axs[1].bar(xx-.18,[a['backbone_valid']/256,a['hard_valid']/256,a['full_success']/256],.36,label='J1');axs[1].bar(xx+.18,[b['backbone_valid']/256,b['hard_valid']/256,b['full_success']/256],.36,label='V3 actor');axs[1].set(xticks=xx,xticklabels=['Backbone','All hard','Full goal'],ylim=(0,1),title='N64 TRAIN admission');axs[1].legend();fig.tight_layout();fig.savefig(R/'plots/SUCCESS_AND_VALIDITY.png',dpi=150);plt.close(fig)
 files={str(p.relative_to(R)):sha(p) for p in R.rglob('*') if p.is_file() and '.git' not in p.parts and 'runtime' not in p.parts and 'source' not in p.parts and p.name!='PROVENANCE_SHA256.json' and not p.name.endswith('.part')};save('PROVENANCE_SHA256.json',dict(schema='E017_V9_manifest_v1',source_commit=commit,source_files=sourcehash,input_checkpoints=inputfiles,historical_reports=original_reports,artifacts=files,immutable_scientific_design_SHA256=sha(R/'SCIENTIFIC_DESIGN_INITIAL.json'),no_self_hash=True))
 print(json.dumps(dict(status='NO_GO',reasons=adm['reasons'],TRAIN_J1=a,TRAIN_actor=b,D_free_bytes=shutil.disk_usage(R).free)))
if __name__=='__main__':publish()
