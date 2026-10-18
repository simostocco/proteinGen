"""Compact final evidence publication; no model loading or optimization."""
import hashlib,json,math,os,shutil,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[3];S=R/'source'
def read(p):return json.loads((R/p).read_text())
def write(p,v):(R/p).write_text(json.dumps(v,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def md(p,x):(R/p).write_text(x+'\n')
def f(x):return f'{x:.6f}'
def number(x,n):return 'UNKNOWN_NO_COMPLETED_TRACE' if x is None else format(x,f'.{n}f')
def main():
 assert read('FINAL_ASSESSMENT_COMPLETE.json')['status']=='COMPLETE'
 done=read('runs/J1/COMPLETED.json');sel=read('FINAL_SELECTION.json');hist=read('TRAIN_VALIDATION_PERPLEXITY.json');cur=read('J1_TRAINING_CURVES.json');cal=read('TORSION_CALIBRATION_HISTORY.json');ex=read('SAMPLING_MIXTURE_AND_EXPOSURES.json');gen=read('CORRECTED_GENERATION_COMPARISON.json');ret=read('APPROVED_RETIREMENT_MANIFEST.json');rep=read('replay/RECONSTRUCTION.json');ver=read('FULL_CHECKPOINT_VERIFICATION.json');cfg=read('configs/FROZEN.json');protected=read('PROTECTED_INTEGRITY_VERIFICATION.json');matched=read('MATCHED_HISTORICAL_BUDGET_COMPARISON.json');n=done['new_optimizer_updates'];base=hist[0];last=hist[-1];best=min(hist[1:],key=lambda x:x['VALIDATION']['AA_NLL']);chosen=next(x for x in hist if x['new_update']==sel['selected_new_update']);ext=read('EXTENSION_DECISION.json') if (R/'EXTENSION_DECISION.json').exists() else {'approved_by_frozen_criteria':False,'failures':[done['stop_reason']]};samples=[json.loads(x) for x in (R/'RESOURCE_SAMPLES.jsonl').read_text().splitlines()];assess=read('FINAL_ASSESSMENT_COMPLETE.json')
 checkpoint_manifest={'selected':sel,'latest':{'path':done['latest_checkpoint'],'sha256':done['latest_sha256'],'new_updates':n,'total_historical_updates':35500+n},'full_checkpoints':ver['checkpoints'],'model_only_parent':{'path':rep['selected_model'],'sha256':rep['selected_sha256'],'preserved':True},'historical_start_parent':{'path':rep['source_full_checkpoint'],'sha256':rep['source_sha256']},'training_commit':'049930ab86e058717aac577380d4033ca9d6c1db','reporting_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip()};write('CHECKPOINT_MANIFEST.json',checkpoint_manifest)
 gpu=[]
 for x in samples:
  try:
   z=x['GPU_csv_timestamp_usedMiB_freeMiB_utilization_temperature_power'].split(',');gpu.append({'used_MiB':float(z[1]),'free_MiB':float(z[2]),'utilization_pct':float(z[3]),'temperature_C':float(z[4]),'power_W':float(z[5])})
  except (ValueError,IndexError):pass
 resource={'retirement_allocated_bytes':9498710016,'retirement_free_before_bytes':10979131392,'retirement_free_after_bytes':20462768128,'retirement_net_capacity_gain_bytes':9483636736,'net_gain_note':'concurrent new source/output allocation makes net lower than removed allocated bytes','D_free_at_publication_bytes':shutil.disk_usage(R).free,'untouched_reserve_bytes':10*2**30,'minimum_observed_D_free_during_monitor_bytes':min(x['D_free_bytes'] for x in samples),'replay_optimizer_seconds':rep['elapsed_seconds'],'main_optimizer_seconds':cur['optimizer_seconds'],'main_update_loop_seconds':cur['optimizer_seconds'],'AA_decisions_per_optimizer_second':cur['AA_decisions_per_optimizer_second'],'main_elapsed_including_evaluation_seconds':done['elapsed_seconds_including_evaluation'],'post_run_assessment_seconds':assess['seconds'],'timing_scope':'update-loop seconds include data preparation, forward/backward, clipping and optimizer calls, not isolated AdamW kernel time; replay excludes data loading; checkpoint-carried main elapsed includes evaluation and recovered worker periods, excludes external +500 recovery generation, data-loading/bootstrap/calibration and discarded restart-reference elapsed; no invented exact end-to-end wall time','main_resources':done['resources'],'post_run_assessment_resources':assess,'observed_device_window':{'samples':len(gpu),'max_VRAM_MiB':max(x['used_MiB'] for x in gpu) if gpu else None,'min_free_VRAM_MiB':min(x['free_MiB'] for x in gpu) if gpu else None,'mean_utilization_pct':sum(x['utilization_pct'] for x in gpu)/len(gpu) if gpu else None,'max_temperature_C':max(x['temperature_C'] for x in gpu) if gpu else None,'limitation':'discrete 30-second samples during part of main worker lifetime; not continuous maxima, evaluation and checkpoint IO lower average utilization'},'host_min_MemAvailable_bytes':min(x['MemAvailable'] for x in samples),'swap_used_max_bytes':max(x['SwapTotal']-x['SwapFree'] for x in samples),'real_PDB_optimizer_calls':cur['total_real_PDB_optimizer_calls_this_task'],'unique_new_updates':n,'historical_replay_calls':500,'restart_reexecuted_calls':2,'no_other_models_TEST_AF_SAC_RL':True};write('RESOURCE_AND_STORAGE_REPORT.json',resource)
 cliprows='\n'.join(f"| {g} | {cur['gradient_norms_before'][g]['median']:.4f} | {cur['gradient_norms_after'][g]['median']:.4f} | {100*cur['clip_fractions'][g]:.2f}% |" for g in cfg['block_limits'])
 table='\n'.join(f"| +{h['new_update']} | {h['TRAIN']['perplexity']:.6f} | {h['VALIDATION']['perplexity']:.6f} | {h['VALIDATION']['AA_NLL']-h['TRAIN']['AA_NLL']:.6f} | {h['VALIDATION']['geometry_NLL']:.6f} | "+' / '.join(f'{x:.6f}' for x in h['VALIDATION']['component_NLL'])+' |' for h in hist)
 cand='Promoted new candidate' if sel['new_candidate_promoted'] else 'No new checkpoint promoted; retain selected historical pilot+500 with its exactly reconstructed full state'
 md('CHECKPOINT_AND_RECOVERY_REPORT.md',f'''# Checkpoint and recovery

{cand}.

Selected full state: `{sel['selected_checkpoint']}`

SHA256 `{sel['selected_sha256']}`.

Latest full state: `{done['latest_checkpoint']}`

SHA256 `{done['latest_sha256']}`; {n} new / {35500+n} historical optimizer updates.

The original model-only +500 and all protected historical checkpoints remain unchanged. Exact historical replay used the archived entrypoint, old clipping, LR and sample/noise ordering for500 updates. Every model tensor is bitwise equal to the selected +500 artifact, and all500 saved historical sample hashes and numerical diagnostics match. Reconstructed AdamW steps35500, scheduler500, sampler5500 and RNG are present and internally consistent. This reconstructs state from the pinned trajectory; there was no historical +500 full-state file for direct optimizer-byte comparison.

A new actual-GPU interrupted test restored after update2 and reproduced updates3/4: model, optimizer, scheduler, sampler, RNG, sample ordering and logged numerical values were exact. Four unique updates consumed six actual optimizer calls; two replayed calls are explicitly counted. The separate historical500 replay is not counted as new training.

At main+500, a missing generation-directory bootstrap caused a process exit after checkpoint and native evaluation publication. No partial checkpoint was loaded. The pending generation was completed read-only, and training resumed from verified+500 with unchanged numerical code and state. Logs, technical recovery receipt and the first failed attempt are preserved.

Atomic publication uses staging, fsync, rename, SHA sidecar and reload checks. Full evaluated milestones and latest recovery are retained. Only verified superseded NEW transient recovery files were retired; NEW_RETENTION.jsonl lists those. Original historical checkpoints were never replaced or deleted. Full-state verification checks parameter shapes, optimizer moment finiteness and step counters, scheduler and sampler counters. CHECKPOINT_MANIFEST.json records all preserved full-state milestone paths and hashes.

Training source is pinned to049930a (full revision in checkpoint lineage). Final reporting source revision is in provenance; do not silently resume under a different code/config hash. COMPLETED.json prevents automatic restart. This run is stopped pending review; no automatic resume is authorized.
''')
 # A compact, explicit global comparison including uncertainty remains separate from optimized likelihood.
 grow=[]
 for name,ss in gen['summaries'].items():
  for N,v in ss.items():
   grow.append(f"| {name} | {N} | {v['complete_valid']} | {number(v['clashes_per100_residues']['mean'],4)} | {number(v['Rg_A']['mean'],3)} | {number(v['long_contact_fraction']['mean'],6)} |")
 # Native TRAIN/VALIDATION references and paired intervals are retained by reference and in JSON.
 md('FREE_GENERATION_COMPARISON.md','''# Corrected free-running geometry

All compared models use corrected FP64 sampler ca_fp64_interior_v1, live batch4, common saved PCG64 categorical uniforms/Gaussian draws,32 attempts per length64/128/256/384/500. All failures remain in denominators and are not replaced. Exact cached historical model/sampler/seed identities were checked before reuse. Native TRAIN/VALIDATION control statistics and original metric definitions are unchanged; TEST is not evaluated.

| Model | Length | Complete | Clashes/100 | Rg Å | Long-contact fraction |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(grow)+'''

CORRECTED_GENERATION_COMPARISON.json gives paired bootstrap95 intervals for clash, native-normalized absolute log-Rg error and contact changes, as well as local histograms, frame degeneracy, sequence uniqueness/Hamming distance and structural-distance diversity. CANDIDATE_REVIEWS.json records every frozen global failure; an improved mean Rg cannot compensate for a failed clash/diversity safeguard. These are32-chain panels, not evidence of folding or thermodynamic stability. The models do not receive native coordinates during complete generation. No long differentiable rollout was trained in this experiment.
''')
 md('FINAL_J1_TORSION_STABILIZATION_REPORT.md',f'''# J1 torsion stabilization — executed results

{cand}. Completed **{n} new optimizer updates** from the selected35500-update model; latest historical total **{35500+n}**. Stop reason: {done['stop_reason']}. Conditional10k extension approved/used: **{ext['approved_by_frozen_criteria']}**. No training remains authorized automatically.

## Scientific comparison

Same frozen complete TRAIN and VALIDATION panels:2947 identities each,8 prefix targets/identity,23576 decisions,20629 valid targets/component. Partial VALIDATION is separately reported (3627 identities,43524 decisions). Dropout/augmentation are off during evaluation. Native teacher-forced likelihoods preserve original density units, transforms and Jacobians. Continuous density NLL can legitimately be negative; local loss is the per-decision mean over valid components, with bootstrap-empty decisions contributing zero. Exact identity and frozen near-duplicate/contained-domain screens are retained; they do not establish universal remote-homology or experimental-context independence.

| New update | TRAIN PPL | VAL PPL | VAL−TRAIN NLL gap | VAL local NLL | VAL distance / angle / torsion NLL |
|---|---:|---:|---:|---:|---|
{table}

Original J1-FULL: TRAIN10.906462, VAL15.865763, local−0.213709. Previous geometrically unqualified pilot+1500: VAL15.490720, local−0.088957. The present best measured sequence point is +{best['new_update']}: **VAL PPL{best['VALIDATION']['perplexity']:.6f}**, TRAIN{best['TRAIN']['perplexity']:.6f}, gap{best['VALIDATION']['AA_NLL']-best['TRAIN']['AA_NLL']:.6f}, local{best['VALIDATION']['geometry_NLL']:.6f}. A best AA point is not automatically a promoted joint model. All unchanged original safeguards and additional conservative gates were declared before new validation and retained.

Final selected full state `{sel['selected_checkpoint']}`

SHA256 `{sel['selected_sha256']}`.

Latest `{done['latest_checkpoint']}`

SHA256 `{done['latest_sha256']}`.

See NATIVE_CANDIDATE_REVIEWS.json, CANDIDATE_REVIEWS.json and EXTENSION_DECISION.json for exact pass/failure reasons, paired identity uncertainty and the5k extension decision. TRAIN_VALIDATION_PERPLEXITY.json contains top1/top3, all length/position strata and separate partial validation metrics. LOCAL_GEOMETRY_COMPONENTS.json contains valid counts, identity/target-weighted distributions and loss tails. No difficult native target was excluded.

## Implemented gradient control

Every one of12,351,746 trainable parameters is accounted for without adding parameters. Final geometry means/logstd output rows are independently clipped: distance[0,3], angle[1,4], torsion[2,5]. Geometry stem remains shared. AA and aggregate auxiliary gradients combine only on shared parameters with auxiliary norm capped at0.5*AA norm; AA head is not rescaled by geometry output spikes. No aggregateG or whole-model clip follows.

| Block | Median norm before clip | Median after | Clip frequency |
|---|---:|---:|---:|
{cliprows}

Observed shared auxiliary/AA ratio max{cur['shared_geometry_ratio']['max']:.6f}; cosine mean{cur['shared_AA_aux_cosine']['mean']:.6f}. These are gradient statistics, not equivalent AdamW update ratios. Per-block actual parameter-update norms and row-specific AdamW moments are independently saved. Training and synthetic extreme-distance tests demonstrate no distance-output-induced torsion-row clipping. Geometry-stem and representation interference remain possible.

Three focused clipping unit tests and37 prior geometry/masking/precision regressions passed. Baseline AA evaluation agrees exactly; local evaluation differs by only floating roundoff. Actual-GPU full-state interrupted restart is exact. Corrected FP64 inference, density/Jacobian, builder, input features and missingness paths are hash-verified unchanged. COMPONENT_GRADIENT_TESTS.json records source locations and results.

## Data exposure and optimization

AA decisions{ex['AA_total']}; partial{ex['partial_AA']} ({100*ex['partial_AA_fraction']:.2f}%), complete{ex['complete_AA']}; partial AA targets at missing Cα positions{ex['partial_AA_at_missing_CA']}. Complete geometry-only decisions{ex['geometry_only_total']}. Distinct AA identities{ex['unique_AA_identities']}, identity-position targets{ex['distinct_AA_identity_position_targets']}, repeated targets within this new lineage{ex['repeated_AA_targets_within_new_lineage']}. Every actual batch hash and final sampler state match metadata-only replay. Geometry-only targets are not double-counted as AA decisions. Nonlocal targets{done['nonlocal_targets']},K4; long differentiable rollout residues0.

Dropout0.15, mask augmentation0.075 with genuine missingness intact,64AA/live16 plus32 complete geometry decisions, direct nonlocal every2 updates. Compatible AdamW moments restored from exact historical replay; new100-update warmup to1e−5, cosine decay to2e−6 at5000, no reheating. Local/nonlocal weights and all input normalization unchanged. Architecture, split, distribution family and chemistry-off behavior unchanged. Calibration used16 TRAIN-only batches; stale inherited descriptive labels are clarified in COMPONENT_CLIPPING_SPEC.md without rewriting frozen settings.

Only one J1 lineage was trained. Historical reconstruction500 calls and restart reexecution2 calls are separately accounted: total real-PDB optimizer calls{cur['total_real_PDB_optimizer_calls_this_task']}, unique new updates{n}. No J2/S1/S2, TEST, AlphaFold, SAC, E015 or RL execution occurred.

## Storage, reproducibility and costs

Exactly ten approved prefilter files retired:9,498,710,016 allocated bytes (8.846GiB). D: free changed10.225→19.058GiB immediately around retirement; net measured gain9,483,636,736 bytes includes new allocation. Current free{resource['D_free_at_publication_bytes']/2**30:.3f}GiB. Ten full pre-unlink hashes/inodes/paths and historical provenance consequences are in APPROVED_RETIREMENT_MANIFEST.json; original manifests were not rewritten. Raw sources, accepted split, final edges and protected checkpoints remain intact. Protected historical verification passed{protected['checked_files']} checks; raw archive contents were not redundantly rehashed in full.

Historical replay phase took{rep['elapsed_seconds']:.1f}s (excluding data loading). Main update-loop time{cur['optimizer_seconds']:.1f}s ({cur['AA_decisions_per_optimizer_second']:.1f}AA decisions/s); checkpoint-carried elapsed including evaluation{done['elapsed_seconds_including_evaluation']:.1f}s; final assessment{assess['seconds']:.1f}s. Additional data-loading/bootstrap/recovery-generation costs are outside those phase clocks. RESOURCE_AND_STORAGE_REPORT.json contains actual Torch peaks, sampled device-wide VRAM/utilization, host RSS and disk headroom; sampled maxima are not continuous hardware maxima.

The+500 bootstrap directory failure was recovered from a verified atomic checkpoint without losing unique updates or changing training mathematics. Attempt logs are preserved. Selected/latest full-state checkpoints include optimizer/scheduler, private sampler/position coverage, CPU/CUDA/global RNG, nonlocal RNG, counters, exact source/config/dataset lineage, dropout/augmentation settings and clipping policy. COMPLETED.json blocks automatic restart.

## Interpretation and remaining work

Full corrected-sampler comparisons are in FREE_GENERATION_COMPARISON.md and CORRECTED_GENERATION_COMPARISON.json. Improvements in teacher-forced likelihood alone do not demonstrate global folding. Torsion coverage, uncertainty tails, loss quantiles and circular-moment numerical limitations remain explicit in TORSION_CALIBRATION_HISTORY.json. Same-sampler/noise matched-budget comparison at1000 new updates shows torsion NLL improved relative to the previous pilot, but both clipping and LR schedule changed; causal attribution to clipping alone is unsupported. Sequential training exposures also differ from immutable historical baselines.

The final VALIDATION torsion increase is tail-driven: on the saved fixed arrays, the source-defined worst207/20629 targets contribute+0.144637nats, while the complement improves−0.015344nats. Native torsions within10degrees of the principal seam contribute+0.012790nats of the total+0.129293. Nominal50/80/95 coverage-error sum slightly decreases0.259918→0.252259, but the worst1% contribution worsens and95% coverage remains below nominal; this is not a satisfactory calibration recovery. No target is excluded in this analysis. TORSION_PAIRED_TAIL_ANALYSIS.json includes TRAIN and every checkpoint. VALIDATION torsion logstd clipping frequency is zero at source and final. These measured5k findings establish that removing output-gradient coupling alone is insufficient evidence of sustained torsion recovery; LR/schedule also changed.

The new complete-panel evaluator does not populate the diagnostic anomaly flag (its anomaly stratum is a placeholder), so its zero flags must not be interpreted as proof of no source anomalies. The original source-quality evidence remains preserved, all observations are retained, and likelihood/target/missingness masks are unchanged. No new anomaly-based exclusion or absence claim is made.

The practical next step is review of the Pareto tradeoff and failed guards before authorizing any further training. Avoid an unchanged long extension if local/torsion calibration or free-running clashes remain disqualifying. Preserve the selected historical+500 and original J1-FULL fallbacks. Any future torsion-distribution change, loss reweighting, normalization or optimizer-lineage reset requires a separately versioned experiment. No protein foldability, stability or function is established by these Cα metrics.
''')
 # Direct answers to all requested final scientific questions.
 questions=[
  f"Recovered9,498,710,016 allocated bytes; net measured D: free-capacity gain9,483,636,736 bytes ({9483636736/2**30:.3f}GiB), before new training artifacts.",
  "Exactly the ten approved pref_0.{0,1,2,3,index} / prefilter.{0,1,2,3,index} files. Exact absolute paths, inodes and pre-unlink SHA256 are in APPROVED_RETIREMENT_MANIFEST.json. No directory was removed.",
  f"Yes: downstream accepted assignment/final-edge hashes match pre-retirement values and {protected['checked_files']} historical protected checks pass. Source/mapping seals and loader evidence retained; the full raw archive was not redundantly rehashed.",
  "Yes:500 pinned historical updates reproduced all selected+500 model tensors exactly, maximum error0, with every recorded historical sample order and numerical diagnostic matching; full optimizer/scheduler/sampler/RNG recovery sealed.",
  "Not applicable: compatible original AdamW moments were recovered; no fresh optimizer lineage or later-checkpoint moment mixing was used. The new schedule/clipping is nevertheless a separately versioned experimental continuation.",
  "Yes for final output rows: disjoint distance/angle/torsion clips prevent distance-specific spikes from rescaling torsion-specific rows. Geometry stem remains genuinely coupled. AA-head clipping is independent; there is no whole-model clip.",
  "Measured per-block norms/activation rates are in the clipping table above and J1_TRAINING_CURVES.json; actual AdamW parameter deltas and row moments are separate from gradients.",
  f"At best-AA checkpoint+{best['new_update']}, VAL torsion NLL change vs start is {best['VALIDATION']['component_NLL'][2]-base['VALIDATION']['component_NLL'][2]:+.6f}; at final+{n} it is {last['VALIDATION']['component_NLL'][2]-base['VALIDATION']['component_NLL'][2]:+.6f}. See TRAIN/VALIDATION component and worst-tail histories; positive changes mean deterioration. Final TRAIN torsion change is {last['TRAIN']['component_NLL'][2]-base['TRAIN']['component_NLL'][2]:+.6f} nats; TRAIN and VALIDATION must be interpreted separately.",
  f"VAL nominal95% interval coverage: {cal['0']['VALIDATION']['calibration']['coverage50_80_95']['95']:.6f} at start to {cal[str(n)]['VALIDATION']['calibration']['coverage50_80_95']['95']:.6f} at final. Coverage alone does not establish calibration; 50/80/95 coverage, PIT and worst1% tails remain explicit in TORSION_CALIBRATION_HISTORY.json.",
  f"{'Yes' if best['VALIDATION']['perplexity']<15.490720 else 'No'}: best observed VAL PPL{best['VALIDATION']['perplexity']:.6f} at+{best['new_update']}. This sequence result does not by itself pass joint-model safeguards.",
  f"NLL gap start{base['VALIDATION']['AA_NLL']-base['TRAIN']['AA_NLL']:.6f}, best-AA{best['VALIDATION']['AA_NLL']-best['TRAIN']['AA_NLL']:.6f}, final{last['VALIDATION']['AA_NLL']-last['TRAIN']['AA_NLL']:.6f}; the final gap {'widened' if last['VALIDATION']['AA_NLL']-last['TRAIN']['AA_NLL']>base['VALIDATION']['AA_NLL']-base['TRAIN']['AA_NLL'] else 'narrowed'} versus the selected source. Compare against the frozen allowance, not stochastic batch loss.",
  f"Final VAL distance/angle changes vs selected start are {last['VALIDATION']['component_NLL'][0]-base['VALIDATION']['component_NLL'][0]:+.6f} / {last['VALIDATION']['component_NLL'][1]-base['VALIDATION']['component_NLL'][1]:+.6f} nats. Negative is better; absolute component values and uncertainty are retained.",
  "Global improvement is not established merely by sequence/local losses. FREE_GENERATION_COMPARISON.md and CANDIDATE_REVIEWS.json report every length-specific Rg/clash/contact/diversity result and unchanged safeguard failure, with paired intervals in the comparison JSON.",
  f"{n} unique new optimizer updates. Historical replay500 and restart reexecutions2 are excluded from that count; total real-PDB optimizer calls{cur['total_real_PDB_optimizer_calls_this_task']}.",
  f"Conditional extension to10000 was {'approved and used' if ext['approved_by_frozen_criteria'] else 'not justified and not used'}; exact reasons in EXTENSION_DECISION.json. No automatic further updates.",
  f"Selected new update{sel['selected_new_update']}; {'new candidate passes all original and conservative gates' if sel['new_candidate_promoted'] else 'no new candidate passes all original plus conservative gates; retain previously accepted source+500'}. Original J1-FULL remains immutable fallback.",
  f"{sel['selected_checkpoint']} — SHA256 {sel['selected_sha256']}.",
  "Review the component/calibration and generated-prefix tradeoff before another run. A bounded TRAIN-only calibration/geometry-stem interference experiment is more defensible than unchanged long training or immediate RL; do not replace the torsion family or normalization without a separate checkpoint-compatible protocol. No additional experiment is started here."
 ]
 with (R/'FINAL_J1_TORSION_STABILIZATION_REPORT.md').open('a') as q:q.write('\n## Direct final answers\n\n'+'\n\n'.join(str(i+1)+'. '+v for i,v in enumerate(questions))+'\n')
 with (R/'EXPERIMENT_DECISIONS.md').open('a') as q:q.write(f'\nFinal assessment: {n} unique new updates, stop={done["stop_reason"]}; extension={ext["approved_by_frozen_criteria"]}; selected_new_update={sel["selected_new_update"]}. All gates unchanged. Separate clipping and LR schedule co-vary, so causal attribution is limited. All GPU training stopped pending review.\n')
 with (R/'CHANGELOG.md').open('a') as q:q.write('\nCompleted source- and state-verified historical500 replay, independent output-row clipping, real J1 continuation, exact restart, fixed bootstrap recovery and final native/corrected-generation comparisons. Post-run analysis code performs no optimizer updates. Final reporting source revision recorded in provenance.\n')
 write('PUBLICATION_READY.json',{'status':'READY_FOR_SEAL','reporting_commit':checkpoint_manifest['reporting_commit'],'selected':sel,'new_updates':n,'all_training_stopped':True})
 print('REPORTS_READY')
if __name__=='__main__':main()
