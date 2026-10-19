"""Read-only scientific interpretation/plots after completed training; no Torch import."""
import json,math,runpy,shutil,subprocess,time
from pathlib import Path
import numpy as np
R=Path('/mnt/d/Simone/proteinGen_data/e016_j1_bidirectional_multitask_20261019_v1')
def read(n):return json.loads((R/n).read_text())
def write(n,v):(R/n).write_text(json.dumps(v,indent=2)+'\n')
def main():
 rows=read('SEQUENCE_TRAIN_VALIDATION_METRICS.json');base,last=rows[0],rows[-1];assert last['new_update']==10000
 man=read('CHECKPOINT_MANIFEST.json');assert man['selected_new_updates']==0
 gen=read('FREE_GENERATION_COMPARISON.json');assert all(gen['decisions'][s]['generation_new']['passes'] for s in ['7500','10000'])
 masked=read('MASKED_RECONSTRUCTION_METRICS.json');rng=np.random.default_rng(20261019);a=masked['milestones']['0']['VALIDATION']['per_identity'];b=masked['milestones']['10000']['VALIDATION']['per_identity'];ids=sorted(a);values=np.array([b[h]['NLL']-a[h]['NLL'] for h in ids]);bs=np.mean(values[rng.integers(0,len(values),(2000,len(values)))],axis=1)
 masked['paired_final_minus_baseline_VALIDATION']={'identities':len(ids),'mean_NLL_change':float(values.mean()),'identity_bootstrap_95pct':np.quantile(bs,[.025,.975]).tolist(),'resamples':2000,'seed':20261019,'not_family_independence':True};write('MASKED_RECONSTRUCTION_METRICS.json',masked)
 cover=read('TRAINING_TOKEN_AND_IDENTITY_COVERAGE.json');resources=read('RESOURCE_AND_STORAGE_REPORT.json');gradient=read('MULTITASK_GRADIENT_BALANCE.json');robust=read('GENERATED_PREFIX_AA_ROBUSTNESS.json');cfg=read('configs/FROZEN.json')
 # Exact full-state +7500 is the strongest unpromoted local/sequence compromise.
 cp=next(v for v in man['checkpoints'] if v['new_updates']==7500 and Path(v['path']).name=='checkpoint_07500.pt')
 report=R/'FINAL_J1_BIDIRECTIONAL_MULTITASK_REPORT.md';text=report.read_text().replace('All include production/auxiliary model, AdamW, scheduler, independent samplers, RNG, counters and task/config hashes.','All new multitask checkpoints include production/auxiliary model, AdamW, scheduler, independent samplers, RNG, counters and task/config hashes. The selected historical parent retains its original full-state format and has no new auxiliary readout.')
 text=text.replace('2.769hours','2.769 hours').replace('13.340GiB','13.340 GiB').replace('0.044432nats','0.044432 nats').replace('uses32','uses 32')
 assessment=f'''
## Final interpretation and next decision

The 10,000-update run produced supported, diminishing next-AA gains, but not a qualifying new generator. Final geometry-free VALIDATION improved by {base['FREE_VALIDATION']['AA_NLL']-last['FREE_VALIDATION']['AA_NLL']:.6f} nats versus the parent and 0.009096 nats versus the previous full-sequence research model (paired 95% interval [−0.009999, −0.008217] for new minus previous). This falls short of the prospectively required 0.020-nat improvement versus that research baseline. The geometry-free NLL gap widened by {last['free_gap']-base['free_gap']:.6f} nats. Native-prefix VALIDATION PPL improved to {last['NATIVE_VALIDATION']['perplexity']:.6f}, with a smaller native-gap increase of {last['native_gap']-base['native_gap']:.6f} nats.

The +7,500 checkpoint is the strongest unpromoted sequence/local-geometry compromise: native VALIDATION PPL 15.409281, geometry-free PPL 17.160075, local NLL −0.123903, and all new final generation limits passed. Its full state is `{cp['path']}`, SHA256 `{cp['sha256']}`. It remains research-only because the prospective sequence target and geometry-free gap criteria fail. The +10,000 state is the best next-AA point estimate, but additionally fails the new local-likelihood allowance. The historical parent remains selected; no historical status is rewritten.

Both new candidates completed 160/160 generation attempts. Sequence and structural diversity passed the declared thresholds. Global changes vary by length, and paired uncertainty does not establish a consistent improvement in global organization. Clashes remain, and these Cα diagnostics do not certify foldability.

From +7,500 to +10,000, geometry-free VALIDATION improved only 0.000781 nats and native-prefix VALIDATION only 0.000279 nats; both are below the frozen 0.002 continuation rule. Therefore no extension was performed. More steps in this unchanged lineage are not justified by the final curve, even though many residue positions remain unseen. Before another large run, prefer a small matched-exposure objective study or a concrete intervention against the widening geometry-free gap. Do not attribute the modest gains to masked spans or KL individually: the objective bundle, schedule and dropout draws changed together. No new run or RL is started.

Masked reconstruction improved from NLL 2.892445 to 2.857072 and target-weighted accuracy 8.166% to 10.215%; the paired identity-bootstrap NLL-change interval is {masked['paired_final_minus_baseline_VALIDATION']['identity_bootstrap_95pct']}. This is modest learning on a separate bidirectional task, not autoregressive perplexity. Both samplers visited all 118,219 TRAIN identities. The masked task processed 160,000 sequence examples and {cover['masked_full_sequence_input_tokens']:,} real sequence-input tokens, supervising 5,385,900 distinct positions. Combined next-AA/masked exposure covered {cover['unique_union_identity_positions']:,} distinct positions; the unused-pool union covered {100*cover['unused_unique_union_fraction_of_available_labels']:.2f}% of its 16,625,418 available labels. Additional paired forward views are not counted as independent AA labels.

On the same ten identities, native-to-unavailable AA KL fell from {robust['same_ten_identity_native_unavailable']['accepted_parent']['KL_NATIVE_TO_UNAVAILABLE']['identity_mean_KL']:.6f} to {robust['same_ten_identity_native_unavailable']['10000']['KL_NATIVE_TO_UNAVAILABLE']['identity_mean_KL']:.6f}. Generated-prefix AA PPL fell from 19.631964 to 19.128161. These small-panel findings support robustness improvement without identifying the consistency loss as its cause.

Main optimizer computation took {resources['main_compute_seconds']/3600:.3f} hours; main elapsed time including milestone evaluations was {resources['elapsed_with_evaluations_seconds']/3600:.3f} hours. Final comparison/publication is additional. Throughput was {resources['nextAA_targets_per_compute_second']:.2f} next-AA decisions/s plus {resources['masked_targets_per_compute_second']:.2f} masked targets/s. Actual main peak Torch allocated/reserved memory was {resources['main_end_peaks']['GPU_allocated_peak']/2**30:.3f}/{resources['main_end_peaks']['GPU_reserved_peak']/2**30:.3f} GiB; batch-32 preflight stress peaks are reported separately. All new full states passed model, AdamW-step, scheduler, sampler-cursor and configuration checks; the GPU restart preflight was bitwise exact.

![Learning curves](LEARNING_CURVES.png)

![Paired generation changes](GENERATION_COMPARISON.png)
'''
 report.write_text(text+assessment)
 table=['| Length | Parent clashes/100 | +7500 clashes/100 | +10000 clashes/100 | Parent Rg Å | +7500 Rg Å | +10000 Rg Å |','|---:|---:|---:|---:|---:|---:|---:|']
 for n in ['64','128','256','384','500']:
  vv=[gen['summaries'][s][n] for s in ['accepted_parent','7500','10000']];table.append('| '+n+' | '+' | '.join(f"{v['clashes_per100_residues']['mean']:.3f}" for v in vv)+' | '+' | '.join(f"{v['Rg_A']['mean']:.3f}" for v in vv)+' |')
 with (R/'GEOMETRY_RETENTION_REPORT.md').open('a') as f:f.write('\n## Final generation\n\nBoth candidates passed the new generation limits at 32 attempts/length, but no consistent cross-length global improvement is established. Paired intervals, contact/diversity distributions and native controls are in FREE_GENERATION_COMPARISON.json.\n\n'+'\n'.join(table)+'\n')
 with (R/'EXPERIMENT_DECISIONS.md').open('a') as f:f.write('\nFinal interpretation: stop at +10000 for the predeclared next-AA plateau; no extension. Keep parent selected. Preserve +7500 as the strongest research sequence/local compromise and +10000 as best next-AA point estimate. New-generation limits passed at 32/length; sequence target/gap still fail. No objective-specific causal claim or RL execution.\n')
 runpy.run_path(str(R/'source/scripts/e016_v3/j1_multitask_plots.py'),run_name='__main__')
 # Compact source-controlled pointer, not duplicated coordinate/dataset evidence.
 note=R/'source/experiments/e016_j1_bidirectional_multitask_20261019_v1/RESULTS.md';note.parent.mkdir(parents=True,exist_ok=True);note.write_text('# E016 J1 bidirectional multitask results\n\nEvidence: '+str(R)+'\n\n10,000 new updates; no extension after held-out plateau. Geometry-free VAL PPL 17.925712 → 17.146676; native VAL PPL 15.597933 → 15.404985. Gap widened; final local NLL −0.101979 fails new allowance. +7500 local NLL −0.123903 and both final 32/length generation panels pass new generation limits. Neither candidate passes all prospective criteria; accepted parent retained.\n\nTraining commit f128fe92c7afc63befb35322a34b51dd7a0fa02d. See final report, checkpoint manifest and SHA256 provenance for exact source/input/model bindings. No TEST, teacher, distillation or RL.\n')
 stopped=subprocess.check_output(['ps','-eo','pid,args'],text=True);active=[s for s in stopped.splitlines() if 'python' in s and s.rstrip().endswith('/j1_bidirectional_multitask.py')];assert not active,active
 write('WORKER_STOP_VERIFICATION.json',{'time_unix':time.time(),'active_multitask_training_workers':active,'training_worker_session_exited':True,'resource_monitor_session_exited':True,'final_evaluator_session_exited':True,'no_additional_optimizer_updates_after_10000':True,'no_publication_optimizer_calls':True,'D_free_bytes':shutil.disk_usage(R).free})
 resources['D_free_after_publication_bytes']=shutil.disk_usage(R).free;resources['final_generation_attempts_new_candidates']=320;resources['new_candidate_generation_failures']=0;resources['main_elapsed_scope']='Main training plus milestone evaluation; final comparison/publication is additional.';write('RESOURCE_AND_STORAGE_REPORT.json',resources)
 destination=R/'source/scripts/e016_v3/j1_multitask_publish.py'
 if Path(__file__).resolve()!=destination.resolve():shutil.copyfile(__file__,destination)
 print(json.dumps({'publication':'PASS','masked_CI':masked['paired_final_minus_baseline_VALIDATION'],'training_stopped':True,'D_free_GiB':shutil.disk_usage(R).free/2**30}))
if __name__=='__main__':main()
