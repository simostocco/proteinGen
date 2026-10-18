"""Seal focused torsion/gradient diagnostics and audited ClassA cleanup. No training."""
import json,math,os,shutil,subprocess,time
from pathlib import Path
import j1_torsion_diagnostic as d
R=d.R

def md(n,s):(R/n).write_text(s)
def usage(p):
 out=[]
 for line in p.read_text().splitlines():
  a,b=line.split('\t',1);out.append((int(a),b))
 return sorted(out,reverse=True)
def main():
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],cwd=R/'source',text=True).strip()
 if subprocess.check_output(['git','status','--porcelain'],cwd=R/'source',text=True).strip():raise ValueError('diagnostic source must be committed clean before sealing')
 cl=json.load(open(R/'SAFE_CLEANUP_MANIFEST.json'));protected=json.load(open(R/'PROTECTED_INTEGRITY_VERIFICATION.json'));cand=json.load(open(R/'APPROVAL_ONLY_STORAGE_CANDIDATES.json'));local=json.load(open(R/'LOCAL_COMPONENT_TRAIN_VALIDATION.json'));resource=json.load(open(R/'RESOURCE_REPORT.json'));free=shutil.disk_usage(R).free;gi=lambda a:f'{a/2**30:.3f}'
 def table(rows):return '| Path | Allocated GiB | Classification |\n|---|---:|---|\n'+''.join(f'| {p} | {gi(s)} | Protected or unassessed scientific data; preserved |\n' for s,p in rows)
 md('D_DISK_USAGE_REPORT.md',f'''# D: usage and executed cleanup

D: volume1,000,202,039,296bytes. Before task10,918,563,840bytes free(10.169GiB); first cache-plan snapshot {cl['free_before_first_cleanup_plan_bytes']}bytes; final snapshot{free}bytes({gi(free)}GiB). Reserve10GiB maintained. Gross allocated ClassA bytes unlinked:{cl['allocated_bytes_removed']}({cl['allocated_bytes_removed']/2**20:.2f}MiB),{cl['ClassA_deleted_files']}files. Net free change since first cache plan:{free-cl['free_before_first_cleanup_plan_bytes']}bytes; this includes new diagnostic/report/worktree/journal allocation and possible unrelated volume activity, so it is not an isolated deletion-only measurement. No files were moved to same-volume trash.

## Largest project roots

{table(usage(R/'REPO_TOP_USAGE.txt')[:14])}

{table(usage(R/'DATA_TOP_USAGE.txt')[:12])}

## Largest data/output/report subdirectories

{table(usage(R/'REPO_LARGE_SUBDIRECTORIES.txt')[:18])}

Measurements use allocated `du -x -B1`, no symlink traversal, hardlink de-duplication within each traversal; overlapping root and child rows must not be summed. Candidate-file ledger records device/inode/nlink/logical/allocated sizes. No multiply linked candidate file was deleted. Volume-wide other user directories were not traversed or cleaned.

## Deleted and preserved

{cl['categories_allocated_bytes']}. Exact6506paths, pre-deletion hashes, allocated/logical sizes, inode and mtime are in SAFE_CLEANUP_MANIFEST.json and durable CLEANUP_DELETION_JOURNAL.jsonl. All6506journal paths reconcile exactly with missing approved cache candidates; zero unjournaled deletions. All115initially preserved cache entries still exist.10additional initial candidates were conservatively preserved after improved relative-path/hash checks. Scope: Python bytecode (including unused project-local ColabFold environment bytecode), pytest/compiler caches; no package source/weights, datasets,scientific fixtures,checkpoints,Git directories or published artifacts were deleted. Historical metadata/Git/open-descriptor/mmap checks preceded deletion. Current cleanup's own receipts intentionally authorize disposal and are not a protection index for their listed cache hashes.

Interrupted inventory attempts are retained. One cache-deletion pass was interrupted during journal publication; reconciliation proves no lost deletion records. Another pass conservatively protected caches by indexing its own cleanup hashes; the final pass excluded these administrative receipts and completed the approved ClassA disposal. No model/training process was interrupted.

Post-cleanup verification PASS:{protected['checked_files']}checks, including all245previous-pilot artifacts,895tracked pilot sources, protected historical checkpoints/input seals,143prior sequence-first artifacts and885prior source files. Old worktree Git status remains clean. No new full raw-archive rehash was needed; source/mapping integrity seals and the real loader's source checks were reused. PROTECTED_INTEGRITY_VERIFICATION.json gives every path/hash.

The15–20GiB goal was NOT achieved with safe automatic cleanup. Large raw/processed datasets, historical models/reports and UniRef evidence dominate; their age or size does not make them disposable.
''')
 bt='| Approval-only group | Exact file inventory | Reclaimable GiB | Consequence |\n|---|---|---:|---|\n'
 for name,v in cand['prefilters'].items():bt+=f"| {name} | APPROVAL_ONLY_STORAGE_CANDIDATES.json → prefilters/{name}/files | {gi(v['allocated_reclaimable_bytes_if_all_unlinked'])} | Historical prefilter candidate evidence/restart files retired; no verified identical copy |\n"
 total=sum(v['allocated_reclaimable_bytes_if_all_unlinked'] for v in cand['prefilters'].values())
 md('STORAGE_RECOMMENDATIONS.md',f'''# Storage actions requiring approval

{bt}
Both groups total{total}bytes({gi(total)}GiB), ten specific shard/index files, each nlink1. Deleting these *if separately approved* would raise current free space to approximately{gi(free+total)}GiB. Retain FASTA/queryDB,INPUT,blastp.sh,logs,receipts,final relation edges and reports. This is not a request to delete entire homology directories or accepted split files. The groups differ in shard sizes; an identical durable copy is UNKNOWN/not demonstrated. Rebuilding derives from preserved sequences/tool commands but consumes compute/scratch and bitwise reproducibility is unproven. These files retain historical audit/recovery value; preserve unless explicit retirement is approved and documented. They are not automatically disposable simply because their directories are called scratch.

Lower-priority ClassB: published model_01500.pt/model_02500.pt in the previous pilot, about{gi(sum(v['allocated_bytes'] for v in cand['model_only_duplicate_snapshots']))}GiB. Each has model tensors verified identical to its protected FULL recovery counterpart, but serialization/history/provenance files differ. Removing snapshots would retire manifest entries; explicit approval/retirement receipt required. The FULL recovery checkpoints,selected500,source5000 and historical7500 remain protected. No ClassB or ClassC file was deleted.

Do not purge the large repository outputs/reports or UniRef/PDB sources as a shortcut: they include historical scientific evidence and unique processed data. Deeper per-artifact retirement review would be a separate task. A future J1 run needs≥10GiB plus~.85GiB estimated two-checkpoint/staging/evidence headroom, preferably15–20GiB. Current free space{gi(free)}GiB does not safely support the proposed retention plan. Approval of specific derived-prefilter retirement, or additional D: capacity, is needed before that run.
''')
 pt='| Checkpoint | TRAIN PPL | VAL PPL | VAL local NLL | Status |\n|---|---:|---:|---:|---|\n'
 for tag,v in local.items():
  status='historical fallback' if tag=='original' else 'selected under existing corrected safeguards; model-only' if tag=='pilot500' else 'not selected'
  pt+=f"| {tag} | {math.exp(v['TRAIN']['AA_NLL']):.6f} | {math.exp(v['VALIDATION']['AA_NLL']):.6f} | {v['VALIDATION']['historic_local_per_decision']:.6f} | {status} |\n"
 md('FINAL_J1_DIAGNOSTIC_AND_STORAGE_REPORT.md',f'''# J1 torsion, gradient and storage results

Completed diagnostic and maintenance task. **Zero optimizer steps, no replay/training, no TEST evaluation.** Production weights,architecture,input scaling,distributions,datasets and optimizer configurations remain unchanged. Source branch `{branch}`, commit `{commit}`; initial diagnostic implementation6844930. Historical pilot and source states preserved.

{pt}

## Required answers

1. **Torsion caused most of the within-pilot local regression.** From500→2500 VAL torsion+0.159958 nats,angle+0.017404,distance−0.025086. Earlier original→source5000 loss deterioration affected all three, with distance the largest contributor then. Compare components using the same units/counts, not aggregate local NLL alone.
2. **Both TRAIN and VALIDATION**, worse on VALIDATION: torsion+0.027410/+0.159958 from500→2500. Identity-bootstrap VAL torsion change95%CI[.097243,.239709]; family correlations remain a limitation.
3. **Uncertainty is not calibrated.** Final VAL nominal50/80/95% intervals cover67.73/85.37/93.09%; broad central predictions coexist with underestimated tails. Mean circular angular error slightly improves despite worse NLL.
4. **Tails matter most; seam and saturation do not explain the regression alone.** Worst1% paired-deterioration targets contribute more than the net mean increase; excluding them is analysis only.10°seam targets contribute~8.98% of final-vs500 VAL increase. Evaluated predicted torsion means/logstd show no saturation/clipping. Known native anomaly observations are absent from VAL panel and unchanged in TRAIN. No inference FP32 failure is conflated with teacher-forced likelihood.
5. **No consistent strong AA–torsion antagonism on this bounded panel.** Cosines change sign and have medians near zero. Detailed pairwise/per-layer/RMS statistics in GRADIENT_DIAGNOSTICS.json;20TRAIN identities are not a population-wide causal study.
6. **Common geometry-head clipping is a measured coupling problem to investigate.** Pilot500 median raw G distance576.981 versus torsion12.511; one G factor scales both output rows. Saved AdamW moments are finite and strongly adapted; insufficient torsion updates or optimizer corruption are not proven. Existing AA-head independence from G is preserved.
7. **Preferred correction: independently control geometric output-row blocks/stem, and apply the AA-first cap to S only.** Calibrate limits on real32-example TRAIN batches, preserve distribution/weights, test before future training. Do not merely increase torsion weight. TRAIN-only circular/calibration prototypes are promising but not production changes; a circular head needs a separate compatibility/reparameterization decision.
8. **Storage measured:** proteinGen source/repository tree≈{gi(usage(R/'REPO_TOP_USAGE.txt')[0][0])}GiB; proteinGen_data≈{gi(usage(R/'DATA_TOP_USAGE.txt')[0][0])}GiB. Largest repository consumers: outputs245.367GiB,data155.615GiB,reports38.735GiB. Detailed ranked allocated paths in D_DISK_USAGE_REPORT.md and inventory text files; nested rows overlap.
9. **Deleted6506verified ClassA cache files only.** Exact paths/hashes/sizes in SAFE_CLEANUP_MANIFEST.json. No scientific dataset/checkpoint/source/report was deleted.
10. **Gross reclaimed allocation148.75MiB**(155,975,680bytes); final D:free{gi(free)}GiB. Net free change since first cache plan is{free-cl['free_before_first_cleanup_plan_bytes']}bytes and includes new evidence allocation; do not confuse gross deallocation with net volume change.
11. **Approval-only additional capacity~{gi(total)}GiB** from ten historical prefilter shard/index files; another~{gi(sum(v['allocated_bytes'] for v in cand['model_only_duplicate_snapshots']))}GiB from two model-only snapshots with equivalent weights in full states. All preserved. No identical copy of the prefilter evidence is established. Exact consequences and paths in STORAGE_RECOMMENDATIONS.md / APPROVAL_ONLY_STORAGE_CANDIDATES.json.
12. **Prefer selected+500**, SHA2569303f0d82a8df044302488340f3ad81535f2fc321360a00bf2892f3a0407aa91, `{d.CP['pilot500']}`. It passes existing quality gates; +1500 has better AA PPL15.490720 but fails local geometry. Original J1-FULL remains fallback.
13. **Full+500 reconstruction remains UNKNOWN.** No optimizer/sampler state exists there. A separately authorized500-step exact historical replay must match all model tensors; no replay occurred here. If impossible,use an explicitly new optimizer lineage. Never mix later optimizer moments with+500weights.
14. **Propose5,000updates, evidence-gated extension to10,000.** At measured98.03AA decisions/s,~54/109optimizer-minutes, estimated~1.2–1.6/2.2–2.8h including evaluation, subject to profiling the proposed changes. Current disk headroom is inadequate for the preferred recoverable retention plan.
15. **Continue only on heldout benefit plus preserved geometry.** Aim VAL PPL<15.490720, controlled TRAIN–VAL gap,torsion recovery,stable distance/angle,local NLL≤−.113709 and unchanged corrected-generation clash/Rg/contact/diversity safeguards. Evaluate500/1000/2500/5000; do not extend after5k without useful evidence. No TEST use or threshold relaxation. Proposed configuration and checkpoint recovery gates are in LONGER_J1_CONTINUATION_PLAN.md / PROPOSED_CONTINUATION_CONFIG.json.

## Evidence and limits

Read [torsion diagnosis](TORSION_LIKELIHOOD_DIAGNOSTIC.md), [gradient analysis](COMPONENT_GRADIENT_INTERACTIONS.md), [correction options](TORSION_CORRECTION_OPTIONS.md), [disk report](D_DISK_USAGE_REPORT.md), [future plan](LONGER_J1_CONTINUATION_PLAN.md). Charts: COMPONENT_NLL_COMPARISON.png, TORSION_PIT_COMPARISON.png. All six models use unchanged frozen complete panels23576targets/2947identities eachTRAIN/VAL,20629valid targets percomponent; target/identity weighting exact here. Partial metrics are compact references to existing verified panels, not a new full partial calibration sweep.

TRAIN-only posthoc fits use1433fit/1514heldout TRAIN identities. They are not trained circular heads or independent validation. No ablation proves clipping,head calibration,representation drift or partial-AA exposure caused the regression. New gradient panel is small,native/eval-mode,no augmentation; actual production clipping/exposure statistics are reused separately. Complete free-running evidence is reused only from the versioned corrected sampler for original/source5000/pilot500. Global comparisons remain mixed and long chains are overly extended; no new generation superiority is claimed for1500/2500. Cα likelihood/proxies do not establish foldability,stability or function.

Diagnostics measured{resource['seconds']:.1f}s(~{resource['seconds']/60:.2f}min),peakTorch allocated{gi(resource['GPU_allocated_peak'])}GiB,reserved{gi(resource['GPU_reserved_peak'])}GiB,hostRSS highwater{gi(resource['RSS_highwater_bytes'])}GiB. Device-wide peak was not sampled. Storage scanning/hash verification adds CPU/I/O elapsed time. All new outputs/cache/scratch remain on D:. Protected verification passes{protected['checked_files']}file checks. Scientific inputs/historical reports unchanged. Work is stopped; next training,replay,model changes and ClassB cleanup require review/approval.
''')
 # Append idempotent sealing receipts, preserving earlier decision content.
 for name in ['EXPERIMENT_DECISIONS.md','CHANGELOG.md']:
  p=R/name;s=p.read_text().split('\n## Final diagnostic seal')[0];p.write_text(s+f'\n## Final diagnostic seal\n\nSource `{commit}`, branch `{branch}`.6506ClassA files/155975680allocated bytes removed; every deletion journaled/reconciled;{protected["checked_files"]}protected checks pass. No production changes or optimizer steps. Inventory failures/interrupted passes retained.0 new training decisions implemented.\n')
 # Fail if required publications absent, before publishing a success seal.
 required=['TORSION_LIKELIHOOD_DIAGNOSTIC.md','TORSION_CALIBRATION.json','LOCAL_COMPONENT_TRAIN_VALIDATION.json','COMPONENT_GRADIENT_INTERACTIONS.md','TORSION_CORRECTION_OPTIONS.md','GRADIENT_DIAGNOSTICS.json','D_DISK_USAGE_REPORT.md','SAFE_CLEANUP_MANIFEST.json','STORAGE_RECOMMENDATIONS.md','EXPERIMENT_DECISIONS.md','CHANGELOG.md','FINAL_J1_DIAGNOSTIC_AND_STORAGE_REPORT.md']
 if any(not (R/n).is_file() for n in required):raise ValueError('missing required report')
 outputs={}
 for p in sorted(R.rglob('*')):
  if not p.is_file() or p.is_symlink() or p.relative_to(R).parts[0] in ['source','runtime'] or p.name in ['PROVENANCE_SHA256.json','PROVENANCE_VERIFICATION.json','FINALIZE.log']:continue
  outputs[str(p.relative_to(R))]={'sha256':d.file_sha(p),'bytes':p.stat().st_size}
 source={n:d.file_sha(R/'source'/n) for n in subprocess.check_output(['git','ls-files'],cwd=R/'source',text=True).splitlines()}
 prov={'version':'E016_J1_TORSION_DIAGNOSTIC_STORAGE_V1','time':time.time(),'branch':branch,'commit':commit,'source_clean':True,'base_reporting_commit':'3e6cbe09d026b17e8e7695c8eed157726391a57d','input_verification_sha256':d.file_sha(R/'INPUT_VERIFICATION.json'),'protected_verification_sha256':d.file_sha(R/'PROTECTED_INTEGRITY_VERIFICATION.json'),'tracked_source_sha256':source,'outputs':outputs,'optimizer_steps':0,'TEST':'NOT_EVALUATED','production_model_optimizer_input_scaling_distribution_changes':False,'excluded':['runtime regenerable caches','provenance self/verification','FINALIZE.log (publication stdout)'],'training_state':'STOPPED_NO_AUTOMATIC_RESUME'}
 d.write('PROVENANCE_SHA256.json',prov)
 errors=[]
 for n,v in outputs.items():
  if d.file_sha(R/n)!=v['sha256']:errors.append(n)
 for n,h in source.items():
  if d.file_sha(R/'source'/n)!=h:errors.append('source/'+n)
 if errors:raise ValueError(errors)
 d.write('PROVENANCE_VERIFICATION.json',{'status':'PASS','outputs_verified':len(outputs),'source_files_verified':len(source),'protected_checks':protected['checked_files'],'provenance_sha256':d.file_sha(R/'PROVENANCE_SHA256.json'),'errors':[],'free_D_bytes':shutil.disk_usage(R).free,'optimizer_steps':0})
 print(json.dumps({'status':'COMPLETE','source_commit':commit,'root':str(R),'D_free_GiB':shutil.disk_usage(R).free/2**30,'outputs_verified':len(outputs),'optimizer_steps':0}),flush=True)
if __name__=='__main__':main()
