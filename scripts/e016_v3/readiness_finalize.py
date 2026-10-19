"""Cached-only R0/R1 synthesis, reports, protected-input verification and disabled R2 plan."""
import json,hashlib,os,subprocess,statistics,sys,time
from pathlib import Path
import numpy as np
from protein_joint_sac.rl.sequence_geometry_rewards import geometry_features,confidence_aware_agreement
from protein_joint_sac.rl.sequence_rewards import mapped_structure_scores
R=Path(__file__).resolve().parents[3];SRC=R/'source';BASE=R.parent

def read(n):return json.loads((R/n).read_text())
def write(n,x):
 if isinstance(x,str) and n.endswith('.md'):
  replacements={
   'user-selected+7500':'user-selected +7,500','loaded+7500':'loaded +7,500','at43,000historicalupdates,12,351,746productionparameters':'at 43,000 historical updates, with 12,351,746 production parameters',
   'GPU64AA/32geometry forward/backward and40/30/30':'GPU batch of 64 AA and 32 geometry decisions, plus 40/30/30', 'all118,219TRAIN':'all 118,219 TRAIN','repair:13corrected successes,2preserved failed attempts,15/16total':'repair: 13 corrected successes, 2 preserved failed attempts, 15/16 total',
   'cap2700s':'cap 2,700 s','call51–73s':'call 51–73 s','peak5053MiB':'peak 5,053 MiB','0.223639,0/6>=.50 versus required median.60 and4/6>=.50':'0.223639; 0/6 reach .50 versus required median .60 and 4/6 reaching .50',
   'N64episodes':'N=64 episodes','were41.95/.23 and53.69/.25':'were 41.95/.23 and 53.69/.25','>=70pLDDT':'>=70 pLDDT','earlier+500benchmark':'earlier +500 benchmark',
   'exact40/30/30quota':'exact 40/30/30 quota','complete32decisions':'complete 32 decisions','RTX5060':'RTX 5060','everyTRAINnativefailed':'every TRAIN native failed','twoVALIDATION':'two VALIDATION',
   'N64–447calls':'N=64–447 calls','about51–73s':'about 51–73 s','ceiling1e-6,geometryoutputs frozen,strong':'ceiling 1e-6, geometry outputs frozen, strong','future2seeds×100actorupdates,48totalAFcalls':'future 2 seeds × 100 actor updates, 48 total AF calls',
   'AFnative/calibration':'AF native/calibration','startR2':'start R2','+7500 as RL initialization (43,000 historical updates)':' +7,500 as RL initialization (43,000 historical updates)',
   'RMSE<.10 after400':'RMSE < .10 after 400','Freeze6TRAIN controls,2TRAIN variants,1repeat,2VALIDATION controls and2generated':'Freeze 6 TRAIN controls, 2 TRAIN variants, 1 repeat, 2 VALIDATION controls and 2 generated',
   'First2AF':'First 2 AF','All13corrected':'All 13 corrected','0/6at.50':'0/6 at .50','after15attempts/':'after 15 attempts / ','below16/2700caps':'below 16 / 2,700 caps',
   '0.1801 to 1.3695':'0.1801 to 1.3695','27,401-parameter/Q':'27,401 parameters per Q','160,777-parameter/Q':'160,777 parameters per Q','frozen7500':'frozen +7,500','fallback500':'fallback +500',
   '64AA/32G':'64 AA / 32 geometry','32-decision':'32-decision','64-AA':'64-AA','24 development and24':'24 development and 24','each phase8 calls/arm':'each phase 8 calls/arm','current measured51–73s/call':'current measured 51–73 s/call','16-call':'16-call',
   'two seeds, at most 100':'two seeds, at most 100','PDB64 AA at40/30/30 and32':'PDB 64 AA at 40/30/30 and 32','geometryoutputs':'geometry outputs','sixTRAIN':'six TRAIN',
  }
  for a,b in replacements.items():x=x.replace(a,b)
 p=R/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n' if not isinstance(x,str) else x.rstrip()+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=SRC,text=True).strip()
def main():
 complete=read('R1_COMPLETED_v3.json');assert complete['oracle_calls_used']==15 and complete['budget']['charged_seconds']<2700 and complete['J1_optimizer_updates']==0
 raw=read('ALPHAFOLD_NATIVE_CONTROL_RESULTS.json');rows=raw['rows'];good=[r for r in rows if r['status']=='SUCCEEDED'];native=[r for r in good if r['kind']=='native' and r['split']=='TRAIN'];val=[r for r in good if r['kind']=='native' and r['split']=='VALIDATION'];gate=read('AF_REWARD_GO_NO_GO.json');critic=read('TRANSFORMER_CRITIC_DIAGNOSTIC.json');init=read('INITIALIZER_VERIFICATION.json');toy=read('HYBRID_SAC_TOY_RESULTS.json');profile=read('RL_GRADIENT_AND_RESOURCE_PROFILE.json')
 assert len(good)==13 and len(native)==6 and len(val)==2 and gate['GO'] is False
 # Keep original driver output unchanged; add a compact binding index instead.
 bindings=[];peak=0
 for i,row in enumerate(rows):
  out=R/'oracle'/((('v3_' if i>=2 else '')+row['id']));res=json.loads((out/'RESULT.json').read_text());assert res['sequence_sha256']==row['sequence_sha256']
  bindings.append({'id':row['id'],'invocation_version':'v3_cuda_cpu' if i>=2 else 'v2_failed_cuda_only','status':row['status'],'sequence_sha256':res['sequence_sha256'],'protocol_sha256':res['protocol_sha256'],'seed':res['seed'],'RESULT_path':str(out/'RESULT.json'),'RESULT_sha256':sha(out/'RESULT.json'),'artifact_manifest_sha256':sha(out/'ARTIFACT_SHA256.json') if (out/'ARTIFACT_SHA256.json').exists() else None})
  for sample in row.get('GPU_samples',[]):
   fields=sample.split(',')
   if len(fields)>=2:peak=max(peak,int(fields[1]))
 write('ORACLE_RESULT_BINDINGS.json',{'all_attempts':bindings,'no_failed_attempt_removed':True,'wrapper_source_sha256':sha(SRC/'scripts/e016_v3/readiness_oracle.py')})
 # Identity bootstrap is descriptive on this small curated panel, not a population certificate.
 rng=np.random.default_rng(1919);tm=np.array([r['recovery']['robust_fixed_map_TM_similarity_fullN'] for r in native]);boot=np.median(tm[rng.integers(0,len(tm),(10000,len(tm)))],axis=1);interval=np.quantile(boot,[.025,.975]).tolist()
 generated=[]
 for row,filename in [(r,'J1_initial64.json' if r['id']=='J1_generated64' else 'fallback_initial64.json') for r in good if r['kind']=='generated']:
  g=read('generated/'+filename);res=json.loads((R/'oracle'/('v3_'+row['id'])/'RESULT.json').read_text());generated.append({'id':row['id'],'policy_sha256':g['policy_sha256'],'seed':g['seed'],'sequence_sha256':row['sequence_sha256'],'J1_geometry':geometry_features(g['ca']),'AF_predicted_geometry':geometry_features(res['predicted_ca']),'AF_features':row['features'],'agreement':confidence_aware_agreement(g['ca'],res['predicted_ca'],res['plddt']),'scope':'one candidate per model, not a policy-quality comparison; no certified reward'})
 write('GENERATED_EPISODE_COMPONENTS.json',{'rows':generated,'production_reward_attached':False,'native_median_TM_identity_bootstrap95':interval,'bootstrap_replicates':10000,'small_curated_panel_scope':True})
 control=next(c for c in read('configs/NATIVE_CONTROL_PANEL.json')['controls'] if c['id']=='train_64');x=np.asarray(control['native_ca'],float);rot=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]);variants={'native':x,'rigid':x@rot+7,'noisy':x+rng.normal(0,.5,x.shape),'extended':np.column_stack((3.8*np.arange(len(x)),np.zeros((len(x),2)))),'collapsed':np.zeros_like(x),'reflected':x*np.array([-1.,1.,1.])};perturb=[]
 for name,y in variants.items():perturb.append({'name':name,'metrics':geometry_features(y),'agreement':mapped_structure_scores(x,y),'sequence_reward_input_unchanged':True})
 write('GEOMETRY_PERTURBATION_DIAGNOSTICS.json',{'scope':'synthetic coordinate perturbations of one verified TRAIN sequence; no additional oracle calls, no experimental negative labels','rows':perturb,'AF_sequence_features_invariant_by_construction':True})
 audit=read('ALPHAFOLD_INSTALLATION_AUDIT.json');audit['inference_executed']=True;audit['protocol_runtime_validated']=True;audit['successful_v3_calls']=13;audit['reward_certified']=False;audit['runtime_evidence']='ORACLE_RESULT_BINDINGS.json';write('ALPHAFOLD_INSTALLATION_AUDIT.json',audit)
 r2={'schema':'E016_R2_PROPOSAL_v1','production_authorized':False,'coefficients_sealed':False,'AF_reward_certified':False,'real_episode_critic_certified':False,'algorithm':'hybrid_SAC_candidate_NOT_APPROVED','fallback_proposal':'sequence policy gradient only after reward certification and separate authorization','actor_source_sha256':init['source_sha256'],'lengths':[64,128],'initial_length':64,'gamma':1.,'entropy_and_KL_per_length_normalization':True,'actor_lr_planning_ceiling':1e-6,'shared_lr_planning_ceiling':1e-6,'geometry_output_frozen':True,'policy_dropout':0.,'PDB_dropout':.15,'PDB_coordinate_visibility_augmentation':.075,'PDB_AA_batch':64,'PDB_AA_mixture':[.4,.3,.3],'PDB_complete_geometry_batch':32,'geometry_shared_gradient_cap_ratio':.5,'objective':{'beta_G':0.0,'beta_A':None,'alpha_A':None,'alpha_G':None},'geometry_KL_routing':'separately capped retention only, never the primary AA objective','BC_weight':None,'AA_reference_KL_weight':None,'geometry_KL_weight':None,'alpha_A':None,'alpha_G':None,'critic_lr':None,'clip_limits':'reuse verified independent S/A limits; geometry outputs frozen','seed_count_proposed':2,'actor_updates_per_seed_proposed':100,'total_actor_update_ceiling_proposed':200,'RL_states_per_actor_update_ceiling':4,'critic_steps_per_actor_proposed':2,'schedule':'100-step bounded warmup/decay, to seal after corrected reward/gradient calibration','future_oracle_calls_proposed':48,'development_calls':24,'blind_comparison_calls':24,'arms':['new policy','frozen7500 generation-and-AF-selection','fallback500'],'calls_per_arm_per_phase':8,'same_length_seed_protocol_budget_for_all_arms':True,'current_task_oracle_limit':16,'current_task_oracle_calls':15,'pending_timeout_is_unknown_never_zero':True,'stop_rules':['NO_GO until native recovery and confidence calibration pass','NO_GO until intended critics predict held-out observed returns or separately approved PG replaces SAC','stop on nonfinite action or invalid density/support repair entering continuous SAC','stop on positive KL budget violation or PDB geometry deterioration beyond prospective sealed limits','stop on repeated/composition collapse or native-calibrated clash/contact failure','never optimize only critic-predicted quality','preserve 10GiB reserve plus staging; no concurrent AF/policy GPU workloads'],'note':'Executable preflight only. This is a disabled proposal, not a runnable or numerically calibrated production trainer.'};write('configs/R2_PILOT_PROPOSED.json',r2)
 from readiness_r2_preflight import inspect
 blocked=inspect(R);assert not blocked['GO'];write('R2_PREFLIGHT_RESULT.json',blocked)
 natives='\n'.join(f"| {r['id']} | {r['N']} | {r['features']['mean_pLDDT']:.2f} | {r['features']['ptm']:.2f} | {r['recovery']['robust_fixed_map_TM_similarity_fullN']:.3f} | {r['worker_wall_seconds']:.1f} |" for r in native+val)
 source_commit=git('rev-parse','HEAD')
 assert not git('status','--porcelain'),'source worktree must be clean before evidence sealing'
 gpu=subprocess.run(['nvidia-smi'],text=True,capture_output=True);write('FINAL_WORKER_STATUS.json',{'GPU_snapshot':gpu.stdout,'GPU_probe_returncode':gpu.returncode,'J1_optimizer_updates':0,'production_RL_updates':0,'all_owned_GPU_workers_joined':True,'source_worktree_clean':True,'scope':'post-readiness snapshot; desktop GPU use may remain'})
 write('RL_ACTOR_AND_CRITIC_SPEC.md',f'''# E016 R0 actor and critic specification

Source commit: `{source_commit}`. Production J1 remains 12,351,746 parameters. The user-selected +7,500 source represents 43,000 supervised updates. `reference/frozen_J1.pt` is read-only and `actor/initial_J1.pt` is a separate identical initialization. The masked-reconstruction head is not loaded into the generator. Historical qualification is unchanged.

State is exactly `(N,t,AA_prefix,CA_prefix,valid_masks)`; bidirectional attention is restricted to the generated prefix. No future sequence, native geometry or AF prediction enters it. Actions factor into categorical AA and AA-conditioned transformed Gaussian geometry. Distance is exp(u), angle pi*sigmoid(u), signed torsion pi*tanh(u). Physical log densities include their component Jacobians; the t>=1/2/3 bootstrap masks are retained. FP64 action transforms and builder preserve right-handed conventions.

The corrected inference sampler explicitly marks finite-grid support repairs. Those traces remain evidence but cannot enter continuous SAC: a saturated finite-grid action is not silently claimed to have the original continuous density. The differentiable training sampler rejects saturation. `RL_ACTION_LOGPROB_TESTS.json` verifies selected probabilities, physical densities, prefix isolation, extreme torsions and coordinate reconstruction.

Categorical optimization supports exact summation of 20 categories, with a reparameterized conditional geometry draw per category, or an unbiased sampled score-function term plus the direct conditional path. Integer sampling is never differentiated. The initial sequence-priority actor update detaches sampled geometry; geometry-output parameters and selected-AA adapter are frozen, while shared geometry drift has an independent retention gradient. Conditional geometry KL averages with detached categorical weights intentionally to avoid AA-head gradients from the geometry budget. The initial sequence-first SAC interface requires beta_G=0 in its primary value objective; geometry KL is routed only through the separately capped retention callback, preventing an uncapped categorical-gradient bypass.

Independent twin candidate-action Transformer critics receive the prefix plus candidate AA/geometry and length/progress. Target copies use Polyak updates; values use the twin minimum. The 27,401-parameter/Q calibration fixture failed its predeclared RMSE<0.10 gate ({critic['heldout_RMSE_final']}). The separate 160,777-parameter/Q resource fixture was not learning-certified. Neither constitutes a calibrated protein critic.

Finite-horizon gamma=1 preserves terminal reward across lengths; entropy and reference KL are normalized by N. Bootstrap geometry entropy uses only valid components (full-three-component target prorated by valid count). Terminal states never bootstrap. Pending oracle episodes are excluded. See the [SAC primary paper](https://arxiv.org/abs/1812.05905) for the underlying algorithm; these hybrid finite-horizon and sequence-priority controls require their own tests.

Analytic hybrid-bandit twin RMSE was ~7e-12, return improved 0.1801 to 1.3695, and GPU optimizer/RNG restart was bitwise exact. These are synthetic engineering results, not evidence of protein improvement. No real-episode critic calibration or J1 optimizer update occurred. Full production certification remains NO-GO. Production certification must also pin consistent actor computation precision for collection/re-scoring, receipt-to-transition association and complete update/restart behavior; these are not established by finite gradient tests.
''')
 write('PDB_BEHAVIORAL_CLONING_INTEGRATION.md','''# PDB supervision integration

All 118,219 certified TRAIN identities remain available through disjoint complete (21,475), partial-only (29,129), and structurally unqualified AA-only (67,615) populations. Five effective 64-AA batches allocate exactly 128/96/96 targets (40/30/30). Identity-first shuffled position coverage, true lengths, observation ownership and sampler restart are preserved. Availability does not mean all identities were visited in this diagnostic.

Qualified complete/partial prefixes retain their masks; the unused population supplies true AA prefixes with all geometry unavailable, never disqualified coordinates. A separate 32-decision qualified complete geometry stream supports shared geometry retention. No PDB demonstration is assigned a measured thermodynamic reward. Demonstrations and rewarded generated episodes stay distinct.

`sequence_pdb_regularizer.py` streams 16-example microbatches with correct effective-batch normalization. Callbacks perform weighted gradient accumulation and return detached reported losses; callers must not backpropagate the streamed graph a second time. Dropout .15 and visibility augmentation .075 apply during BC, then actor evaluation mode is restored for policy log probabilities.

The actual GPU 64-AA/32-geometry diagnostic passed with finite gradients: AA shared L2 2.449683, raw geometry shared L2 1.703857, AA CE 2.699930. No parameter update occurred. A separate real-J1 composer fixture verified the aggregate geometry-to-sequence shared cap <=.5, disjoint S/A clipping and absent gradients on frozen geometry outputs. CPU streaming tests verified exact gradient/RNG restart and geometry-KL isolation at nonzero drift. These bounded observations do not select production BC/KL coefficients.

AA and geometry gradients are combined with an explicit shared budget; independent scaling/clipping is not advertised as the unmodified gradient of one scalar objective. No whole-model clipping is reintroduced.
''')
 write('ALPHAFOLD_INSTALLATION_AUDIT.md',f'''# Actual local AlphaFold2 installation and invocation

ColabFold 1.6.3, alphafold-colabfold 2.3.20, JAX 0.6.2. Source revision `{audit['source_revision']}`; 105 installed source files checked against the preserved source seal. Existing tools are read from an older project tools directory; no older experiment or RL workflow was executed.

Model 1 pTM parameters: 373,103,340 bytes, SHA256 `{audit['selected_weights_sha256']}`. Full paths and source hashes are in `ALPHAFOLD_INSTALLATION_AUDIT.json`. The installed parameter archive LICENSE is Attribution-NonCommercial-ShareAlike 4.0, SHA256 `{audit['license_sha256']}`. Preserve this local notice; this audit does not assert unrestricted commercial rights or substitute terms from a newer release. Primary sources: [versioned AlphaFold source](https://github.com/google-deepmind/alphafold/tree/v2.0.0), [ColabFold source](https://github.com/sokrypton/ColabFold).

Frozen v3: single sequence, max MSA 1:1, no templates, one model, 3 recycles, tolerance 0, seed 151502 (one native repeat 151503), no relaxation, dropout off, internal BF16/output FP32, fusion on, fast kernels off, backend auto. Actual runtime config was checked per successful job. `cuda,cpu` JAX backends expose CPU feature preprocessing while the wrapper requires the GPU to be the default folding backend. No MSA service calls, downloads or new model databases occurred; network socket connections were denied in the inference wrapper. A configured default service URL does not mean it was contacted.

The first two attempts used cuda-only backend discovery and failed because installed feature RNG explicitly needs a CPU backend. Their logs, protocol and failed receipts remain preserved and count against the hard budget. The corrected invocation has a new wrapper SHA/protocol/cache identity. No failure is treated as a biologically poor protein or as zero reward.

Thirteen corrected calls succeeded with full mapped CA coordinates and genuine pLDDT, pTM and PAE. Total attempts 15/16; charged wall time {complete['budget']['charged_seconds']:.3f}s/2700s. Maximum sampled process/device memory across calls was {peak} MiB; telemetry is sampled, not a guaranteed instantaneous allocator peak. Successful calls took approximately 51–73s. Each subprocess compiled anew, so warm-process throughput may differ.

Operational GO means local execution works. Reward certification is independently NO-GO: native recovery and held-out calibration fail. AF confidence is neither folding stability nor function.
''')
 write('ALPHAFOLD_REWARD_CALIBRATION.md',f'''# AlphaFold sequence reward certification: NO-GO

The native panel and gates were frozen before AF calls. Six verified TRAIN native sequences cover N=64,128,256,331,384,447; historical failures 3fm1_A and 2qmi_A were retained with authoritative mappings. Four TRAIN identities fit one nonnegative confidence scale; two disjoint TRAIN identities test it. Two disjoint VALIDATION identities audit it after the TRAIN decision. No TEST evaluation or VALIDATION fitting occurred.

| Control | N | Mean pLDDT | pTM | Mapped TM-style similarity / full N | Wall s |
|---|---:|---:|---:|---:|---:|
{natives}

Native recovery uses the authoritative residue map, proper rotations and a multistart iteratively reweighted rigid fit. Missing coordinates reduce coverage and the denominator remains full deposited length. This is a fixed-map TM-style similarity, **not US-align optimized TM-score**. 3fm1_A has 319/331 observed positions; native coverage is explicitly recorded. Historical recovery numbers computed differently must not be spliced into this table.

TRAIN median={gate['TRAIN_median_native_TM']:.6f}, identity-bootstrap descriptive 95% interval {interval}; 0/6 reaches .50. Required median>=.60 and at least 4/6>=.50. TRAIN pTM/recovery rank correlation={gate['TRAIN_pTM_recovery_rank_correlation']:.6f}, below .40. VALIDATION audit median is {statistics.median(r['recovery']['robust_fixed_map_TM_similarity_fullN'] for r in val):.6f}, below .50. This small curated panel is not a random population sample or evidence of family independence.

Candidate formula is clip(beta * mean_pLDDT/100 * pTM * fraction(long-range PAE<10A),0,1). TRAIN-only beta={gate['candidate_TRAIN_only_fit']['coefficient']:.6f}; held-out TRAIN MAE={gate['candidate_TRAIN_only_fit']['TRAIN_holdout_MAE']:.6f} exceeds .15, and VALIDATION audit MAE={gate['candidate_TRAIN_only_fit']['VALIDATION_audit_MAE']:.6f} exceeds .20. Failure was recorded before the VALIDATION audit in `TRAIN_ONLY_REWARD_DECISION_BEFORE_VALIDATION.json`. Deployment weights remain null and scoring refuses NO-GO calibration.

The repeated native64 seed changed mean pLDDT from 64.76 to 47.66 and pTM .48 to .34, with both recovery scores poor. The composition-shuffle and AL-repeat diagnostics are not experimentally established nonfolding controls. Their scores cannot supply biological negative labels. Native cofactor/assembly/chain contexts and single-sequence inference are plausible limitations, not isolated causal explanations.

pLDDT, pTM and PAE are genuine and usable as descriptive protocol outputs. None is certified here as a training reward. In particular high local confidence alone is insufficient. Do not blame the native proteins for this protocol's failure. A bounded next corrective study should review MSA-assisted AF2 feasibility and required service/data authorization, reuse this frozen native panel and reprofile cost; no database download or different folding model is authorized by this task.
''')
 write('SEQUENCE_FIRST_REWARD_SPEC.md','''# Sequence-first reward specification

A completed sequence is evaluated once under exact sequence SHA, AF protocol SHA and seed. Local confidence uses mean/q10 pLDDT and coverage; global confidence uses genuine pTM and long-range PAE. Long-range PAE separation is min(24,max(4,N//4)), declared separately from geometry's >=24 contact metric. Global structure/contact/Rg descriptors are monitored without rewarding compactness monotonically.

The tested candidate confidence product is mean_pLDDT/100 * pTM * fraction(PAE<10A over declared long-range pairs). A nonnegative scalar is fitted only on four TRAIN controls and tested on two held-out TRAIN controls, with a disjoint VALIDATION audit. This limited prototype is deliberately not a certified general-purpose protein-quality reward. It failed; deployable coefficients are null.

Primary sequence features depend on completed S and AF(S), never X_J1. Supplementary proposed R_total = R_sequence + lambda_geometry*R_geometry, with R_geometry in [-1,1], lambda<=.1 and actual lambda unsealed. KL and PDB behavioral cloning are separate optimization regularizers. A good sequence reward can remain high despite poor J1–AF agreement; hard nonfinite/failed construction remains generation failure.

Composition entropy, identical runs, tripeptide repetition, exact TRAIN duplication and bounded nearest-sequence similarity are alarms, not unbounded novelty rewards. `GENERATED_SEQUENCE_PLAUSIBILITY.json` scans 4-mer similarities over all TRAIN identities and matching blocks over the top 16; this is not an exhaustive alignment or family novelty test. No reward is awarded for novelty alone.

Potential exploits include confident repetitive helices, composition collapse, easy isolated domains, score saturation, low-confidence global organization, critic overestimation and learning from unavailable predictions. Missing pTM/PAE and uncalibrated coefficients are rejected. Delayed/time-out calls remain unknown. No production return was attached to a generated episode in this task.
''')
 write('GEOMETRY_AUXILIARY_REWARD_SPEC.md','''# Auxiliary geometry specification

Geometry is supplementary to sequence assessment. Hard failures are malformed/nonfinite CA trajectories or failed construction. Separate descriptors include adjacent distances, virtual angles, signed torsions, degenerate frames, nonlocal pairs<2A at separation>=3, clashes/100 residues and per eligible pair, Rg/N^(1/3), and contacts<8A at separation>=24. C-alpha chirality proxies are not atomistic stereochemistry certification.

AF agreement uses exact same-sequence correspondence and full target length normalization, fixed-map robust TM-style rigid alignment, distance errors and contact Jaccard. This is not US-align optimized TM-score. Confidence-qualified agreement requires at least four residues with pLDDT>=70; otherwise it is unavailable, with all-position agreement labeled descriptive only. J1–AF disagreement never multiplies away the sequence reward.

`GENERATED_EPISODE_COMPONENTS.json` reports these components separately for one reference and one fallback complete N64 episode. There are too few samples to compare model quality. `GEOMETRY_PERTURBATION_DIAGNOSTICS.json` checks unchanged sequence assessment under synthetic native/rigid/noisy/extended/collapsed/reflected proposals without extra AF calls. These are controlled geometry probes, not proven nonfolding sequence controls.

No production auxiliary coefficients or native length-specific normalization are certified here. A future calibration must fit qualified TRAIN geometry controls by length, tolerate legitimate elongation/domain variation and guard both collapse and excessive expansion. It must not reward compactness monotonically or indiscriminately repel legitimate contacts. Existing inference, geometry density transforms, masks and input normalization remain unchanged.
''')
 write('RL_REPLAY_AND_ORACLE_QUEUE_SPEC.md','''# Persistent replay and delayed oracle queue

Compact episodes save actual AA/geometry actions, valid-component masks, selected categorical log probability, physical component densities, original latent/noise values, uniform AA draws, prefix hashes, policy SHA and sampler identity. Exact common-noise regeneration is tested on the same pinned GPU; CPU reconstruction tolerance is explicit. No dense attention tensor or repeated PDB archive is stored.

SequenceReplay adds immutable hashed traces around EpisodeReplay. SQLite WAL + synchronous FULL and transactions protect atomic insertion. Stored proposal coordinates are authoritative and are checked against action reconstruction. A separate queue binds exact sequence, protocol and seed; the same sequence can reuse one AF prediction across different proposals while each proposal keeps its own geometry/agreement association. States include PENDING/RUNNING/SUCCEEDED/FAILED/TIMEOUT/UNSUPPORTED as appropriate to the existing adapter. Pending/failed results are never zero biological reward.

Stale RUNNING recovery marks TIMEOUT; automatic scientific retry is not assumed, and budgets retain attempted work. OracleWallBudget records every reservation transactionally; an unfinished call retains its full 360-second reservation. The same global ledger covers failed v2 and corrected v3 calls. All 15 actual attempts remain charged. Existing complete receipts prevent duplicate execution of this readiness campaign.

`REPLAY_QUEUE_CORRECTNESS.json` tests shared prediction caching, different-proposal geometry association, stale-running recovery, persistent budget and process-exit transaction rollback. It does not simulate an entire Windows power failure. `FINAL_DEFENSIVE_TESTS.json` verifies mock/uncalibrated cards cannot enter production replay and update receipt gates reject unknown results before any step.

Continuous SAC additionally rejects any support-repaired trace. Production update requires a successful external calibrated receipt with the sealed calibration identity and explicit authorization. The blocked R2 configuration has neither. Future controller certification still needs exclusive worker ownership, verified durable output recovery on timeout and validation of receipt-to-selected-transition association at the caller boundary; R0 tests do not establish a production-distributed queue.
''')
 write('FIRST_SEQUENCE_RL_PILOT_PLAN.md','''# R2 proposal: blocked pending review and certification

Run the read-only executable gate:

```bash
bash /mnt/d/Simone/proteinGen_data/e016_sequence_first_rl_readiness_20261019_v1/run.sh /mnt/d/Simone/proteinGen_data/e016_sequence_first_rl_readiness_20261019_v1/source/scripts/e016_v3/readiness_r2_preflight.py
```

It returns NO-GO (exit 2), before GPU model construction or optimizer access. `configs/R2_PILOT_PROPOSED.json` is machine-readable and intentionally disabled; it is not an approved production training command.

First fix the AF2 native recovery/calibration gate under a scientifically meaningful feasible protocol. No score optimization is justified with current single-sequence results. Review MSA-assisted AF2 service/data requirements before any access/download, reuse the eight fixed native controls and impose a new small bounded budget. Do not substitute a different folding model automatically.

The Transformer critics also missed their frozen synthetic hold-out threshold. A bounded one-configuration critic correction may be proposed, but no large search is warranted. If critics cannot calibrate on measured real short returns, prefer a separately approved sequence policy-gradient baseline with the same oracle budget and regularizers. This recommendation does not authorize implementation of another production algorithm.

After gates and separate user approval, propose length64 first, length128 only after reliable short episodes; two seeds, at most 100 actor updates each, <=4 resolved states per actor update, two critic steps/actor only if SAC is certified. Proposed shared/AA LR ceiling 1e-6; geometry outputs frozen, shared geometry drift independently controlled. gamma=1, KL/entropy normalized by N, policy dropout off; PDB64 AA at40/30/30 and32 qualified complete geometry, BC dropout .15, visibility augmentation .075, aggregate shared geometry gradient cap .5.

KL, entropy temperatures, PDB AA/geometry coefficients and critic LR are **unsealed**, not evidence-based production recommendations. Measure return/critic gradient scales first; BC diagnostics alone cannot calibrate them. Preserve independent S/A clipping and AdamW state, source/config/replay hashes and RNG in atomic full checkpoints. No mixing later optimizer moments with earlier policy weights.

Future oracle proposal: 48 calls, 24 development and24 blind matched comparisons, each phase8 calls/arm for new policy, frozen+7500 generation-and-AF-selection, and+500 fallback. Same lengths/seeds/protocol and call budget per arm; count failures without replacements. At current measured51–73s/call this is roughly41–58 oracle minutes; any changed MSA protocol must be reprofiled. This is a future budget proposal, **not an extension of this task's16-call allowance**.

Predeclare sequence-confidence improvement and paired uncertainty, reference KL ceilings, PDB perplexity/local component tolerances, length-specific geometry/contact/diversity thresholds and critic calibration before any production checkpoint. Stop on nonfinite updates, unsupported action density, unresolved oracle transitions, budget breach, composition collapse, severe physical degeneration or critic overestimation. Do not equate reward confidence with experimental stability or function.
''')
 write('EXPERIMENT_DECISIONS.md',f'''# Experiment decisions

1. User explicitly selected supervised+7500 as RL initialization (43,000 historical updates), retaining historical research-only qualification. Strict SHA/parameter checks passed. Frozen reference and separate actor initial copy retained. No supervised continuation or J1 optimizer update.
2. Preserve bidirectional attention only within generated prefixes and exact hybrid factorization. Analytic categorical expectation and score/direct estimator tested; FP64 support repairs are recorded and excluded from continuous SAC, not hidden with arbitrary clamps.
3. Separate sequence AF assessment from bounded additive geometry auxiliary. Good sequence confidence is not multiplied by proposal agreement. No production coefficients selected without native calibration.
4. Analytic hybrid toy passed, but intended small Transformer twin critics missed RMSE<.10 after400 bounded updates. Retain failure and return SAC NO-GO; do not search more configurations. Larger critic was resource-only, not an alternative successful arm.
5. Reuse qualified PDB40/30/30 and separate geometry stream with streamed BC. Shared-gradient budgeting and independent head clipping verified; do not let geometry-head spikes force global AA scaling.
6. Freeze6TRAIN controls,2TRAIN variants,1repeat,2VALIDATION controls and2generated sequences before AF execution. Exact authoritative chain mappings required; preparation metadata KeyError repaired through hashed TARGET_AUDIT, not guessed mappings; completed R0 GPU tests not repeated.
7. First2AF invocations failed due disabled CPU preprocessing backend. Preserve attempts/logs and charge same ledger. Version v3 exposes CPU preprocessing while asserting GPU folding, new wrapper-bound protocol. All13corrected attempts succeeded. No data/weight downloads or external MSA.
8. Native median recovery .223639,0/6at.50; held-out TRAIN calibration MAE .240127. Record TRAIN NO-GO before disjoint VALIDATION audit, which also failed. Do not classify native proteins or variants as experimentally bad. Weights remain null.
9. Stop after15attempts/{complete['budget']['charged_seconds']:.3f}s charged time, below16/2700caps. Do not spend the remaining slot without a reason. R2 remains blocked for both reward and critic certification. Source `{source_commit}`; config/source hashes in provenance. No normalization, architecture, split or retired-prefilter changes.
''')
 write('CHANGELOG.md',f'''# Readiness changelog

Source `{source_commit}` adds sequence-first actor action/density interfaces, hybrid expectations, frozen-reference KL, independent Transformer critics via existing modules, gated SAC and geometry-retention gradient composition; persistent action traces/oracle wall ledger; streaming PDB regularization; separate sequence/geometry feature and calibration code; bounded GPU/CPU tests and real AF driver; disabled R2 preflight and reports.

Preserved v2 failed invocation source and receipts; v3 backend correction changes invocation/cache identity while retaining weights, MSA and recycles. Added defensive missing-coefficient, external calibrated receipt and nonfinite checks. Final source commit and exact files are sealed in `PROVENANCE_SHA256.json`. Historical source, checkpoints and qualification records are unchanged. Zero J1 production or supervised updates.
''')
 resources={'D_free_initial_bytes':14323224576,'D_free_final_bytes':os.statvfs(R).f_bavail*os.statvfs(R).f_frsize,'minimum_reserve_bytes':10*2**30,'AF_attempts_total':15,'AF_v2_failed_attempts':2,'AF_v3_succeeded_attempts':13,'AF_charged_seconds':complete['budget']['charged_seconds'],'AF_v3_worker_seconds':sum(r['worker_wall_seconds'] for r in good),'AF_max_attempts':16,'AF_max_charged_seconds':2700,'GPU_sampled_peak_MiB':peak,'real_J1_critic_resource_max_allocated_bytes':max(p['allocated_peak'] for p in profile['profiles']),'real_J1_critic_resource_max_reserved_bytes':max(p['reserved_peak'] for p in profile['profiles']),'runtime_seconds_by_attempt':[{'id':r['id'],'N':r['N'],'status':r['status'],'seconds':r['worker_wall_seconds']} for r in rows],'AF_internal_precision':'BF16, outputsFP32; explicit cuda,cpu v3 backend','J1_geometry_precision':'FP64 transforms/builder','J1_optimizer_updates':0,'production_RL_updates':0,'synthetic_actor_updates':toy['toy_policy_updates'],'synthetic_bandit_critic_updates':toy['toy_critic_updates'],'synthetic_Transformer_critic_updates':critic['updates'],'downloads':0,'additional_deleted_files':0,'source_commit':source_commit,'initialized_RL_recovery':read('J1_INITIAL_STATE_GPU_RECOVERY.json'),'future_R2_not_authorized':True};assert resources['D_free_final_bytes']>=10*2**30;write('RL_RESOURCE_BUDGET.json',resources)
 write('FINAL_E016_SEQUENCE_FIRST_RL_READINESS_REPORT.md',f'''# E016 sequence-first RL readiness: NO-GO for production

R0/R1 executed. No J1 policy or supervised optimizer updates, no R2 production training, no TEST evaluation. Source `{source_commit}`. Results preserve the user-selected+7500 weights and all historical qualification records.

## Main measured results

- Correctly loaded+7500 SHA `{init['source_sha256']}` at43,000historicalupdates,12,351,746productionparameters. Immutable reference and separate actor initialization have identical tensors; saved model-only copies SHA `{init['immutable_reference_copy_sha256']}`. Full original recovery checkpoint remains unchanged.
- Selected AA log probabilities and physical-unit geometry densities recompute correctly; analytic conditional KL, valid bootstrap masks, FP64 builder gradients and deterministic common-draw replay passed. Finite-grid support repairs are explicitly flagged and excluded from continuous SAC.
- Synthetic hybrid SAC passed: return .1801→1.3695, twin RMSE~7e-12, exact/sampled categorical gradient agreement and GPU restart exact. Intended small Transformer critics missed their frozen held-out RMSE<.10 limit: .120089/.111310. Real protein-return calibration remains unestablished. Therefore SAC certification is NO-GO despite toy success.
- An atomic initialized RL recovery checkpoint (`actor/readiness_initial_full.pt`) saves actor, critic/target, optimizer/temperature, sampler, RNG and configuration. Actual GPU reload gives bitwise identical logits/random draws; moments are empty because no J1 update occurred. Nonempty moment recovery was tested separately on the synthetic bandit. See `J1_INITIAL_STATE_GPU_RECOVERY.json` for the checkpoint SHA and nonzero geometry-KL gradient isolation test.
- PDB integration passed an actual GPU64AA/32geometry forward/backward and40/30/30 quota/resume tests, with all118,219TRAIN identities available. No full-dataset training exposure claim. Geometry gradients are separately budgeted; no whole-model clip.
- Actual AF2 GPU invocation succeeded after a versioned CPU-preprocessing backend repair:13corrected successes,2preserved failed attempts,15/16total. Charged worker wall={complete['budget']['charged_seconds']:.3f}s (14.1min), cap2700s. Typical successful call51–73s; sampled peak{peak}MiB. No downloads or external MSA.

## Native controls and reward gate

| Control | N | Mean pLDDT | pTM | Mapped TM-style similarity / full N | Wall s |
|---|---:|---:|---:|---:|---:|
{natives}

TRAIN native median={gate['TRAIN_median_native_TM']:.6f},0/6>=.50 versus required median.60 and4/6>=.50. TRAIN pTM/recovery rank correlation .028989<.40. Held-out TRAIN calibration MAE .240127>.15; disjoint VALIDATION calibration MAE .283947>.20. TRAIN decision preceded VALIDATION audit. **AF sequence reward NO-GO.** Scores describe this protocol, not native biological quality. The recovery metric uses true residue correspondence/fullN and robust rigid fitting, not US-align optimized TM-score. Native64 seed variation was substantial, so single-seed confidence is particularly weak evidence.

Both complete generated N64episodes (new reference and historical fallback) were processed end-to-end. AF pLDDT/pTM were41.95/.23 and53.69/.25, respectively; neither had residues>=70pLDDT. Their AF agreement is descriptive and not a certified reward. One sample/model cannot establish comparative quality or diversity. See [separate generated components](GENERATED_EPISODE_COMPONENTS.json).

## Answers to the readiness questions

1. **Initializer/reference:** verified source SHA, strict architecture load, separate trainable initialization, immutable reference, earlier+500benchmark retained. Historical supervised qualification unchanged.
2. **Prefix attention:** existing bidirectional generated-prefix semantics retained; no future sequence/AF/native coordinates in actor state.
3. **Log probabilities:** categorical and masked physical density/Jacobian tests pass. Support-repaired finite-grid samples are ineligible for continuous SAC; do not advertise exact continuous likelihood at saturation.
4. **SAC critics:** analytic hybrid toy learns; intended Transformer synthetic held-out gate fails, and real-return calibration has not been established. Overall SAC NO-GO.
5. **Reward separation:** sequence confidence uses only S and AF(S); geometry is an independent bounded additive term. No confidence-times-agreement default.
6. **PDB AA BC:** all118,219TRAIN identities and exact40/30/30quota are integrated, streaming GPU gradients and deterministic sampler recovery checked. Only a bounded diagnostic batch was exercised, not all identities.
7. **PDB geometry:** qualified complete32decisions available independently; unqualified observations stay AA-only. Geometry imitation supports constrained shared drift, not fabricated rewards.
8. **AF operation:** installed local AF2 runs on RTX5060 with genuine pLDDT/pTM/PAE under versioned v3. CPU preprocessing remains enabled; GPU folding is asserted.
9. **Native recovery:** not reliable under this single-sequence protocol; everyTRAINnativefailed. The twoVALIDATION controls confirm rather than rescue the gate.
10. **Usable confidence:** outputs are valid descriptors, but no combination is certified as a production reward. High pLDDT alone is insufficient; scalar confidence calibration failed held-out controls.
11. **Exploitation:** repetitive confident helices, composition/duplication collapse, poor global organization, finite-grid density misuse, critic overestimation, invalid/pending reward reuse and low-confidence agreement are explicit risks. Synthetic fixtures detect several mechanisms, not all biological exploits.
12. **Oracle cost:** successful N64–447calls took about51–73s including startup/compilation. Exact per-attempt lengths/times/memory are in [resource report](RL_RESOURCE_BUDGET.json). New MSA protocol cost is unknown.
13. **Recovery:** trace reconstruction, queue cache sharing, stale-RUNNING recovery, budget persistence and process-exit transaction rollback pass. Full distributed controller/whole-PC power-loss certification remains future work. Pending is never zero. No duplicate expensive calls were hidden.
14. **R2 configuration:** disabled executable preflight and JSON proposal provided. Planning LR ceiling1e-6,geometryoutputs frozen,strong reference/PDB controls; production KL/BC/entropy/critic coefficients remain null because reward and critic gradients cannot yet be scientifically calibrated. Proposed future2seeds×100actorupdates,48totalAFcalls split across matched arms/phases requires separate approval and passed gates.
15. **Production readiness:** **NO-GO**, due AFnative/calibration failure and Transformer critic calibration failure. Do not optimize this oracle or startR2. Review a bounded MSA-assisted AF2 calibration subject to explicit service/data approval; if SAC still fails bounded tests, propose a simpler sequence policy-gradient baseline rather than repeated uncontrolled critic searches.

## Evidence and limits

[Actor/critic specification](RL_ACTOR_AND_CRITIC_SPEC.md), [PDB integration](PDB_BEHAVIORAL_CLONING_INTEGRATION.md), [AF installation](ALPHAFOLD_INSTALLATION_AUDIT.md), [calibration](ALPHAFOLD_REWARD_CALIBRATION.md), [sequence reward](SEQUENCE_FIRST_REWARD_SPEC.md), [geometry auxiliary](GEOMETRY_AUXILIARY_REWARD_SPEC.md), [replay/queue](RL_REPLAY_AND_ORACLE_QUEUE_SPEC.md), [disabled R2 plan](FIRST_SEQUENCE_RL_PILOT_PLAN.md), [decisions](EXPERIMENT_DECISIONS.md), [provenance](PROVENANCE_SHA256.json).

No sequence reward is evidence of thermodynamic stability, folding uniqueness, solubility or function. No policy-quality improvement is claimed. All workers are stopped after synthesis; further production learning requires review.
''')
 # Protected input hashes are verified independently of retired prefilter intermediates.
 expected={init['source_path']:init['source_sha256'],str(BASE/'e016_j1_torsion_stabilization_20261018_v1/replay/checkpoint_00500_full.pt'):'9208263300fb9e88519ff0d40e6d9d29a9744b709d9a6181819cd0894b593000',str(BASE/'e016_partial_supervision_20261016_v1/runs/J1-FULL/checkpoint_20000.pt'):'9e9215d3720ffc73edb0c3e382db25332f91cb26b0c8d38d01668e383b1e3810',str(R/'reference/frozen_J1.pt'):init['immutable_reference_copy_sha256'],str(R/'actor/initial_J1.pt'):init['separate_actor_initial_sha256']}
 verified={}
 for p,h in expected.items():actual=sha(p);assert actual==h,(p,actual,h);verified[p]=actual
 weights=Path(audit['backend']['weights_dir'])/'params/params_model_1_ptm.npz';assert sha(weights)==audit['selected_weights_sha256'];verified[str(weights)]=audit['selected_weights_sha256'];assert sha(audit['license_path'])==audit['license_sha256'];verified[audit['license_path']]=audit['license_sha256']
 for c in read('configs/NATIVE_CONTROL_PANEL.json')['controls']:
  if c.get('source_path') and c.get('source_sha256'):assert sha(c['source_path'])==c['source_sha256'];verified[c['source_path']]=c['source_sha256']
 protected=BASE/'pdb_supervised_split_50id80cov_20261016_v1'
 for p in [protected/'PROPOSAL_MANIFEST.json',protected/'SEQUENCE_ASSIGNMENTS.json']:
  if p.exists():verified[str(p)]=sha(p)
 write('PROTECTED_INPUT_VERIFICATION.json',{'status':'PASS','source_commit':source_commit,'verified_files':verified,'no_deletion_or_data_modification':True,'historical_qualification_unchanged':True,'retired_prefilters_not_reconstructed':True})
 # All worker databases are closed. Hash compact evidence, actual oracle artifacts,
 # actor copies and new source files; exclude caches, git control and self-reference.
 excluded={'source','runtime','.git'};manifest={}
 for p in sorted(R.rglob('*')):
  rel=p.relative_to(R)
  if any(v in excluded for v in rel.parts) or not p.is_file() or p.name in {'PROVENANCE_SHA256.json','PROVENANCE_VERIFICATION.json'}:continue
  manifest[str(rel)]=sha(p)
 sourcefiles={str(p.relative_to(SRC)):sha(p) for folder in [SRC/'src/protein_joint_sac/rl',SRC/'scripts/e016_v3'] for p in sorted(folder.glob('sequence_*.py' if folder.name=='rl' else 'readiness_*.py'))}
 write('PROVENANCE_SHA256.json',{'source_commit':source_commit,'source_branch':git('branch','--show-current'),'source_files_sha256':sourcefiles,'evidence_files_sha256':manifest,'scope':'New readiness evidence including failed attempts; excludes runtime caches/source Git control and self-referential verification; source files separately hashed. Historical inputs independently verified.'})
 write('PROVENANCE_VERIFICATION.json',{'status':'PASS','files_hashed':len(manifest),'source_files_hashed':len(sourcefiles),'protected_files_verified':len(verified),'source_commit':source_commit,'production_RL_updates':0,'J1_optimizer_updates':0,'AF_attempts':15,'AF_reward_GO':False,'SAC_critic_GO':False,'D_free_bytes':os.statvfs(R).f_bavail*os.statvfs(R).f_frsize})
 print(json.dumps({'status':'REPORTS_PUBLISHED','source_commit':source_commit,'production_RL_GO':False,'AF_calls':15,'charged_seconds':complete['budget']['charged_seconds'],'evidence_files_hashed':len(manifest),'D_free_GiB':os.statvfs(R).f_bavail*os.statvfs(R).f_frsize/2**30}),flush=True)
if __name__=='__main__':main()
