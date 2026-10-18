"""Post-pilot corrected-protocol assessment. No optimizer construction/updates."""
import collections,gc,json,math,subprocess,time
from pathlib import Path
import numpy as np
import torch
import j1_clip_pilot as c
from j1_sequence_assess import check_candidate
from j1_global_finalize import metric_summary,paired_ci as _paired_ci
R=c.R
read=c.read;write=c.write

def paired_ci(values):return _paired_ci(values) if len(values) else {'n':0,'mean':None,'bootstrap_95pct':None,'limitation':'No jointly completed samples; failure counts retained'}
def fmt(value,n):return 'UNKNOWN' if value is None else format(value,f'.{n}f')

def text(name,value):
 p=R/name
 if p.exists():raise RuntimeError('preserve completed publication '+str(p))
 p.write_text(value+'\n')
def summary(values):
 x=np.asarray(values,float)
 return {'n':len(x),'mean':float(x.mean()),'median':float(np.median(x)),'p05_p95':np.quantile(x,[.05,.95]).tolist(),'min':float(x.min()),'max':float(x.max())}
def paired(old,new,native):
 out={}
 for N in [64,128,256,384,500]:
  a={r['sample_index']:r for r in old if r['N']==N};b={r['sample_index']:r for r in new if r['N']==N};keys=[k for k in sorted(a.keys()&b.keys()) if 'metrics' in a[k] and 'metrics' in b[k]]
  ref=np.median([r['metrics']['Rg_over_N_cuberoot'] for r in native['TRAIN'][str(N)]['rows']])
  out[str(N)]={'valid_paired':len(keys),'clash_change_per100':paired_ci([b[k]['metrics']['clashes_per100_residues']-a[k]['metrics']['clashes_per100_residues'] for k in keys]),'native_abs_log_Rg_error_change':paired_ci([abs(math.log(b[k]['metrics']['Rg_over_N_cuberoot']/ref))-abs(math.log(a[k]['metrics']['Rg_over_N_cuberoot']/ref)) for k in keys]),'long_contact_fraction_change':paired_ci([b[k]['metrics']['long_contact_fraction']-a[k]['metrics']['long_contact_fraction'] for k in keys])}
 return out

def local_statistics(rows):
 out={}
 for N in [64,128,256,384,500]:
  mm=[v['metrics'] for v in rows if v.get('N')==N and 'metrics' in v]
  if not mm:out[str(N)]={'status':'UNKNOWN_NO_COMPLETED_TRACE'};continue
  out[str(N)]={'complete_traces':len(mm),'angle_edges_rad':np.linspace(0,np.pi,19).tolist(),'torsion_edges_rad':np.linspace(-np.pi,np.pi,37).tolist(),'angle_histogram_identity_mean':np.mean([np.array(m['angle_histogram'])/max(1,sum(m['angle_histogram'])) for m in mm],axis=0).tolist(),'torsion_histogram_identity_mean':np.mean([np.array(m['torsion_histogram'])/max(1,sum(m['torsion_histogram'])) for m in mm],axis=0).tolist(),'adjacent_spacing_A_mean_distribution':summary([m['adjacent_A']['mean'] for m in mm]),'positive_torsion_fraction_distribution':summary([m['torsion_positive_fraction'] for m in mm]),'signed_triple_product_distribution':summary([m['signed_triple_product_mean'] for m in mm]),'degenerate_frame_counts':summary([m['degenerate_frames'] for m in mm]),'weighting':'one completed independent chain, then normalized per-chain histogram; failed attempts reported separately'}
 return out

def main():
 if (R/'FINAL_ASSESSMENT_COMPLETE.json').exists():print('ALREADY_ASSESSED_NO_RERUN');return
 assessment_start=time.monotonic();write(R/'FINAL_ASSESSMENT_STARTED.json',{'time':time.time(),'optimizer_updates':0});done=read(R/'runs/J1/COMPLETED.json');c.s.j.core.setup();torch.use_deterministic_algorithms(False);torch.backends.cudnn.deterministic=False;c.guard(96*2**20)
 history=read(R/'TRAIN_VALIDATION_PERPLEXITY.json');baseline=read(c.OLD/'SEQUENCE_BASELINES.json')['J1'];source=history[0]
 if done['pilot_updates']>2500:raise ValueError('budget exceeded')
 model=c.s.j.core.model('J1');native=read(c.s.j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json')['native'];pop={};raw={}
 for tag,path in [('fallback',c.s.j.PARENT),('source5000',c.SOURCE)]:
  st=c.s.load_checked(path);model.load_state_dict(st['model'],strict=True)
  if tag=='source5000':source_visited=dict(st['sampler']['coverage']['visited'])
  del st;gc.collect()
  rows=c.generate(model,path,tag,32);raw[tag]=rows;pop[tag]={'checkpoint':str(path),'sha256':c.sha(path),'summary':metric_summary(rows)}
 eligible=[h for h in history[1:] if h['VALIDATION']['AA_NLL']<baseline['VALIDATION']['AA_NLL'] and h['VALIDATION']['geometry_NLL']<=baseline['VALIDATION']['geometry_NLL']+.10 and h['VALIDATION']['AA_NLL']-h['TRAIN']['AA_NLL']<=baseline['VALIDATION']['AA_NLL']-baseline['TRAIN']['AA_NLL']]
 eligible.sort(key=lambda h:h['VALIDATION']['AA_NLL']);targets=eligible[:2] if eligible else [min(history[1:],key=lambda h:h['VALIDATION']['AA_NLL'])];review=[]
 for row in targets:
  u=row['pilot_update'];cp=R/'runs/J1'/f'checkpoint_{u:05}.pt'
  if not cp.exists():cp=R/'runs/J1'/f'model_{u:05}.pt'
  st=c.s.load_checked(cp);model.load_state_dict(st['model'],strict=True);full='optimizer' in st;del st;gc.collect()
  cached=R/'generation'/f'pilot{u:05}_{c.PROTOCOL}.json';tag=f'pilot{u:05}' if full and cached.exists() and read(cached)[0]['checkpoint_sha256']==c.sha(cp) else f'final{u:05}'
  rows=c.generate(model,cp,tag,32);ss=metric_summary(rows);fail=check_candidate(ss,pop['fallback']['summary'],native)
  va=row['VALIDATION'];tr=row['TRAIN'];gap=va['AA_NLL']-tr['AA_NLL']
  if va['AA_NLL']>=baseline['VALIDATION']['AA_NLL']:fail.append('AA validation safeguard')
  if va['geometry_NLL']>baseline['VALIDATION']['geometry_NLL']+.10:fail.append('local geometry safeguard')
  if gap>baseline['VALIDATION']['AA_NLL']-baseline['TRAIN']['AA_NLL']:fail.append('TRAIN-VALIDATION gap safeguard')
  ids=sorted(va['per_identity']);aa_change=paired_ci([va['per_identity'][h]['AA_NLL']-source['VALIDATION']['per_identity'][h]['AA_NLL'] for h in ids])
  item={'pilot_update':u,'total_optimizer_updates':35000+u,'checkpoint':str(cp),'sha256':c.sha(cp),'full_recovery_available':full,'passed_all_frozen_safeguards':not fail,'failures':fail,'TRAIN':tr,'VALIDATION':va,'NLL_gap':gap,'partial_VALIDATION':row['PARTIAL_VALIDATION'],'AA_paired_change_vs_source':aa_change,'generation':ss,'paired_generation_vs_source':paired(raw['source5000'],rows,native),'paired_generation_vs_fallback':paired(raw['fallback'],rows,native)}
  review.append(item);write(R/'FINAL_CANDIDATE_REVIEWS.json',review);pop['pilot'+str(u)]={'checkpoint':str(cp),'sha256':c.sha(cp),'summary':ss};raw['pilot'+str(u)]=rows
  if not fail:break
 del model;gc.collect();torch.cuda.empty_cache()
 repairs={}
 for tag,rows in raw.items():
  counts=np.array([v['boundary_repairs'] for v in rows]).sum(0)
  den=[sum(max(0,(v['N'] if v['status']=='COMPLETE' else v['completed_residues']+1)-k) for v in rows) for k in [1,2,3]]
  repairs[tag]={'distance_angle_torsion_adjustments':counts.tolist(),'component_attempt_denominators':den,'fractions':(counts/np.maximum(den,1)).tolist(),'failed_attempts':sum(v['status']!='COMPLETE' for v in rows),'attempts':len(rows),'max_latent_vs_representative_logdensity_difference':max(v['max_latent_representative_logdensity_difference'] for v in rows)}
 write(R/'CORRECTED_SAMPLER_GENERATION_COMPARISON.json',{'protocol':read(R/'configs/FINAL_ASSESSMENT_PROTOCOL.json'),'populations':pop,'reviews':review,'support_representation_counts':repairs,'local_distribution_statistics':{tag:local_statistics(rows) for tag,rows in raw.items()},'native_reference':str(c.s.j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json'),'native_reference_sha256':c.sha(c.s.j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json'),'legacy_results':'Preserved separately; no corrected-versus-legacy statistics pooled','TEST':'NOT_EVALUATED'})
 traces=[]
 for path in sorted((R/'runs/J1').glob('updates_*.jsonl')):
  for line in path.read_text().splitlines():
   if line.strip():traces.append(json.loads(line))
 if len(traces)!=done['pilot_updates'] or sorted(v['pilot_update'] for v in traces)!=list(range(1,done['pilot_updates']+1)):raise ValueError('missing/duplicated actual main updates')
 actual={'updates':len(traces),'group_limits':read(R/'configs/FROZEN.json')['group_limits'],'group_norms_before':{k:summary([v['group_before'][k] for v in traces]) for k in ['S','A','G']},'group_norms_after':{k:summary([v['group_after'][k] for v in traces]) for k in ['S','A','G']},'clip_count':{k:sum(v['group_scale'][k]<1 for v in traces) for k in ['S','A','G']},'clip_fraction':{k:np.mean([v['group_scale'][k]<1 for v in traces]) for k in ['S','A','G']},'clip_scales':{k:summary([v['group_scale'][k] for v in traces]) for k in ['S','A','G']},'AA_shared_L2':summary([v['AA_shared_L2'] for v in traces]),'shared_geometry_ratio':summary([v['geometry_to_AA_shared_ratio'] for v in traces]),'cosine':summary([v['AA_geometry_shared_cosine'] for v in traces]),'old_counterfactual_whole_model_scale':summary([v['old_whole_model_norm1_scale'] for v in traces]),'geometry_shared_cap_count':sum(v['geometry_shared_scale']<1 for v in traces),'parameter_update_norms':{k:summary([v['parameter_update_L2_by_group'][k] for v in traces if 'parameter_update_L2_by_group' in v]) for k in ['S','A','G']},'no_whole_model_clipping':all(not v['whole_model_clip_applied'] for v in traces),'nonfinite_recorded_updates':0}
 old_trace={v['additional_update']:v for v in read(c.OLD/'J1_TRAINING_CURVES.json')}
 actual['prior_target_schedule_comparison']={'matched_update_hashes':sum(v['sample_order_sha256']==old_trace[5000+v['pilot_update']]['sample_order_sha256'] for v in traces),'compared_updates':len(traces),'scope':'same identity/observation/position targets; not identical dropout/noise/optimizer/loss algorithms'}
 write(R/'GROUP_CLIPPING_TRAINING_RESULTS.json',actual)
 calibration_path=R/'TRAIN_CLIP_CALIBRATION.json'
 if not calibration_path.exists():calibration_path.write_bytes((R/'GROUP_GRADIENT_CLIPPING_REPORT.json').read_bytes())
 calibration=read(calibration_path);combined_report=dict(calibration);combined_report.update(training_results=actual,original_calibration_receipt=str(calibration_path),original_calibration_sha256=c.sha(calibration_path));write(R/'GROUP_GRADIENT_CLIPPING_REPORT.json',combined_report)
 latest=Path(done['latest_checkpoint']);st=c.s.load_checked(latest);optimizer_steps={int(float(x['step'])) for x in st['optimizer']['state'].values()};checks={'model_finite':all(bool(torch.isfinite(v).all()) for v in st['model'].values() if torch.is_tensor(v)),'optimizer_finite':all(bool(torch.isfinite(v).all()) for q in st['optimizer']['state'].values() for v in q.values() if torch.is_tensor(v)),'counter_consistency':optimizer_steps=={35000+done['pilot_updates']} and st['scheduler']['last_epoch']==done['pilot_updates'] and st['sampler']['cursor']==5000+done['pilot_updates']}
 if not all(checks.values()):raise ValueError('final recovery checkpoint integrity')
 counts={k:st['sampler']['counts'][k]-done['source_counts'].get(k,0) for k in st['sampler']['counts']};coverage_delta={'new_unique_identity_positions':sum((bits&~source_visited.get(h,0)).bit_count() for h,bits in st['sampler']['coverage']['visited'].items()),'unique_identities_with_new_positions':sum(bool(bits&~source_visited.get(h,0)) for h,bits in st['sampler']['coverage']['visited'].items()),'cumulative_coverage':done['coverage']};write(R/'POSITION_COVERAGE_REPORT.json',coverage_delta);write(R/'FINAL_CHECKPOINT_VERIFICATION.json',{'status':'PASS','checkpoint':str(latest),'sha256':c.sha(latest),'checks':checks,'fields':list(st),'new_exposures':counts,'rollout_residues':0});del st
 write(R/'LOCAL_GEOMETRY_COMPARISON.json',{'original_fallback':baseline,'source5000':source,'pilot_milestones':history[1:],'normalization':'unchanged perdecision valid-component mean, bootstrap contributes0, component means use explicit valid counts','TEST':'NOT_EVALUATED'})
 passing=[v for v in review if v['passed_all_frozen_safeguards']];selected=min(passing,key=lambda v:v['VALIDATION']['AA_NLL']) if passing else None
 selection={'selected_checkpoint':selected['checkpoint'] if selected else str(c.s.j.PARENT),'selected_sha256':selected['sha256'] if selected else c.s.j.PARENT_SHA,'promoted_under_corrected_protocol':selected is not None,'selected_full_optimizer_recovery_available':selected['full_recovery_available'] if selected else True,'legacy_selection_rewritten':False,'candidate_reviews':[{'pilot_update':v['pilot_update'],'passed':v['passed_all_frozen_safeguards'],'failures':v['failures']} for v in review],'reason':'lowest native-qualified AA candidate passing unchanged corrected-protocol global safeguards' if selected else 'no assessed candidate satisfies all frozen safeguards; retain immutable fallback'}
 write(R/'FINAL_SELECTION.json',selection)
 checkpoint_manifest=[]
 for path in sorted((R/'runs/J1').glob('*.pt')):
  state=c.s.load_checked(path)
  if not all(bool(torch.isfinite(t).all()) for t in state['model'].values() if torch.is_tensor(t)):raise ValueError('nonfinite candidate snapshot')
  checkpoint_manifest.append({'path':str(path),'sha256':c.sha(path),'bytes':path.stat().st_size,'pilot_update':state['pilot_update'],'total_optimizer_updates':state['total_optimizer_updates'],'full_optimizer_recovery':'optimizer' in state,'training_source_commit':state['lineage']['source_commit'],'training_entrypoint_sha256':state['lineage']['entrypoint_sha256']});del state
 write(R/'CHECKPOINT_MANIFEST.json',{'retained':checkpoint_manifest,'source_checkpoint':str(c.SOURCE),'source_sha256':c.SOURCE_SHA,'original_fallback':str(c.s.j.PARENT),'original_fallback_sha256':c.s.j.PARENT_SHA,'selection':selection,'retention_receipts':'RETENTION.jsonl','all_candidate_model_weights_preserved':True})
 rr=[json.loads(x) for x in (R/'logs/resources.jsonl').read_text().splitlines() if x.strip()];gpu=[]
 for v in rr:
  try:gpu.append(float(v['gpu_used_free_utilization'].split(',')[0]))
  except ValueError:pass
 resource={'main_updates':len(traces),'effective_AA_batch':64,'local_geometry_batch':32,'live_batch':16,'new_AA_decisions':64*len(traces),'geometry_only_decisions':32*len(traces),'main_optimizer_seconds':sum(v['seconds'] for v in traces),'ordinary_AA_decisions_per_second_excluding_evaluation':64*len(traces)/sum(v['seconds'] for v in traces),'main_elapsed_including_evaluation_seconds':done['elapsed_seconds_including_evaluation'],'final_assessment_elapsed_seconds':time.monotonic()-assessment_start,'technical_storage_amendment':read(R/'STORAGE_STAGING_AMENDMENT.json'),'technical_restore_verification':read(R/'STORAGE_STAGING_RESUME_VERIFICATION.json'),'engineering_optimizer_calls':6,'rollout_residues':0,'resources_at_main_stop':done['resources'],'device_peak_MiB_sampled':max(gpu) if gpu else None,'minimum_D_free_sampled':min(v['D_free_bytes'] for v in rr),'max_swap_used_sampled':max(v['memory']['SwapTotal']-v['memory']['SwapFree'] for v in rr),'final_generation_completed_time':time.time(),'new_exposures':counts,'position_coverage':coverage_delta,'all_main_segments_Torch_allocated_peak_bytes':max(done['resources']['GPU_allocated_peak'],read(R/'STORAGE_PAUSE_COMPLETED.json')['resources']['GPU_allocated_peak']),'all_main_segments_Torch_reserved_peak_bytes':max(done['resources']['GPU_reserved_peak'],read(R/'STORAGE_PAUSE_COMPLETED.json')['resources']['GPU_reserved_peak']),'all_main_segments_RSS_highwater_bytes':max(done['resources']['RSS_highwater_bytes'],read(R/'STORAGE_PAUSE_COMPLETED.json')['resources']['RSS_highwater_bytes']),'TEST_targets':0}
 write(R/'RESOURCE_REPORT.json',resource)
 best=min(history[1:],key=lambda h:h['VALIDATION']['AA_NLL']);final=history[-1]
 lines=['# J1 short precision/clipping pilot','',f"VERIFIED: {done['pilot_updates']} additional updates from previous+5000 ({35000+done['pilot_updates']}historical total). Stop reason: {done['reason']}. No extension, no long rollouts, no TEST. One main J1;12,351,746trainable parameters unchanged.",'','|Model|TRAIN PPL|VALIDATION PPL|NLL gap|VAL local NLL|','|---|---:|---:|---:|---:|']
 for name,h in [('Original fallback',baseline),('Source previous+5000',source)]+[(f"Pilot+{h['pilot_update']}",h) for h in history[1:]]:
  tr=h['TRAIN'];va=h['VALIDATION'];lines.append(f"|{name}|{tr['perplexity']:.6f}|{va['perplexity']:.6f}|{va['AA_NLL']-tr['AA_NLL']:.6f}|{va['geometry_NLL']:.6f}|")
 lines+=['',f"Best pilot AA validation checkpoint+{best['pilot_update']}; source15.602798PPL remains the comparison. Local promotion threshold is original−0.213708968+0.10=−0.113708968; the historical source failure is not relabeled. Final selection: {selection}.",'','## Corrected-protocol free generation','', '32attempts per length and model. Fixed saved common Gaussian/categorical draws, batch4, identical sampling configuration. Failed attempts retained. Historical legacy summaries were not mixed. Identity/sample bootstrap uncertainty and valid-pair denominators are in CORRECTED_SAMPLER_GENERATION_COMPARISON.json.','', '|N|Model|Complete|RgÅ|Clashes/100res|Long contacts|Unique sequences|','|---:|---|---:|---:|---:|---:|---:|']
 for N in ['64','128','256','384','500']:
  for tag,obj in pop.items():
   q=obj['summary'][N];lines.append(f"|{N}|{tag}|{q['complete_valid']}/{q['requested']}|{fmt(q['Rg_A']['mean'],3)}|{fmt(q['clashes_per100_residues']['mean'],4)}|{fmt(q['long_contact_fraction']['mean'],6)}|{q['unique_sequences']}|")
 lines+=['',f"Support-representation counts: {repairs}. Density evaluations at an extreme original latent and its nearest finite-grid action are distinct; no probability-mass claim. Explicit distance overflow/underflow/degenerate-frame failures remain possible.",'','## Optimization and resources','',f"Independent norm limits {actual['group_limits']}; clipping counts {actual['clip_count']}/{len(traces)}updates. Geometry/shared-AA ratio mean{actual['shared_geometry_ratio']['mean']:.4f}, max{actual['shared_geometry_ratio']['max']:.6f}. AA-head clipping depends on its own norm, not geometry-head norms. No whole-model clip remains. Full raw/capped/cosine/group/update statistics: GROUP_CLIPPING_TRAINING_RESULTS.json.", '',f"TRAIN-only calibration12batches;6off-line engineering optimizer calls for the actual12M deterministic restart proof. Restored model/AdamW/scheduler/sampler/RNG and subsequent two updates match exactly in the tested CUDA environment; no arbitrary cross-device/version guarantee. Latest full recovery: {latest}, SHA256{c.sha(latest)}. All evaluated milestone model snapshots retained. Full AA-best/latest retention receipts preserve hashes of retired new-lineage recovery states; historical artifacts unchanged. CHECKPOINT_MANIFEST.json lists every retained full/model-only path and verified SHA256. A selected model-only snapshot lacks immediate AdamW/sampler recovery; no optimizer reset is hidden or authorized.", '',f"Actual new exposures {counts}. Main optimizer time{resource['main_optimizer_seconds']/60:.2f}min; main wall time including evaluation{resource['main_elapsed_including_evaluation_seconds']/60:.2f}min; useful AA decisions/s{resource['ordinary_AA_decisions_per_second_excluding_evaluation']:.2f}. Sampled device peak{resource['device_peak_MiB_sampled']}MiB, no swap observed={resource['max_swap_used_sampled']==0}. All outputs onD:,10GiBreserve protected.", '', 'Dropout.15, augmentation.075,60/40partial/complete AA exposure, identity-first position coverage,32complete geometry decisions and direct nonlocalK4 every2updates retained. Input-feature normalization and architecture unchanged. AdamW moments restored; new100warmup/2500cosine LR2.5e−5→5e−6. Rg/clash/generated-pair rollout losses explicitly disabled. Because clipping, compute objective and schedule changed together, no isolated causal attribution.', '', 'A technical pause at+1000 corrected only payload-specific disk staging:80MiB for model-only snapshots,210MiB for full recoveries,10GiBreserve unchanged. Backward/step/generation/optimizer initialization ASTs were identical. Full model/moments/scheduler/sampler/RNG were restored with zero verification steps; no resets or repeated main updates. Data reload/pause overhead is separate from measured main-worker elapsed. See STORAGE_STAGING_AMENDMENT.json / STORAGE_STAGING_RESUME_VERIFICATION.json.', '', 'Coarse CA metrics do not prove experimental folding, stability or function. The PDB split retains limited near-duplicate/domain guarantees, not all remote homology independence. No further training/RL/oracle execution is authorized by this completion.']
 text('J1_SHORT_PILOT_REPORT.md','\n'.join(lines))
 text('TORSION_PRECISION_REPAIR_REPORT.md','# Inference support repair\n\nca_fp64_interior_v1 preserves the Gaussian exp/sigmoid/tanh reference and checkpoint weights. FP64 transforms precede the FP64 builder. Only representational endpoint saturation is mapped explicitly to an invertible open-support value; every adjustment is counted. Legacy GeometryDistribution/scaled_train code remains byte-identical. Inverse/Jacobian tests and extremes±10,±100,±1e6 pass. Broad-tail paired-noise stress: '+json.dumps(read(R/'TORSION_SUPPORT_TESTS.json')['paired_noise_stress'])+'\n\n'+(R/'configs/INFERENCE_PROTOCOL_SPEC.md').read_text()+'\n\nFinite-grid action density differs from the original latent density in repaired extremes; neither is a discrete probability mass. Training likelihood/Jacobians unchanged. Distances are not physically clamped; overflow/underflow fails. Extremely collinear chains can still fail the established frame check. Saved common noise enables new comparisons; historical latent draws remain unavailable.\n')
 text('CHECKPOINT_AND_RESUME_REPORT.md','# Recovery\n\n'+json.dumps(read(R/'GRADIENT_POLICY_TESTS.json'),indent=2)+'\n\nFinal full-state load/counter/finiteness checks: '+json.dumps(read(R/'FINAL_CHECKPOINT_VERIFICATION.json'),indent=2)+'\n\nArchitecture/config/source/data/geometry/chemistry provenance, AdamW, scheduler, global/dropout/augmentation RNG, sampler/coverage and auxiliary RNG are sealed in the checkpoint lineage. Counters are new pilot updates; starting historical total35000. No automatic resume after completion. CHECKPOINT_MANIFEST.json distinguishes full versus model-only states. Pinned training revisions are2de6e21 and storage-only393aad7, also bound in each checkpoint; reproduce using its pinned implementation, not an arbitrary later reporting HEAD.\n')
 with (R/'EXPERIMENT_DECISIONS.md').open('a') as f:f.write('\nCompleted bounded pilot and corrected-protocol assessment. '+json.dumps({'updates':done['pilot_updates'],'best_AA_update':best['pilot_update'],'best_VALIDATION_PPL':best['VALIDATION']['perplexity'],'selection':selection})+' No acceptance thresholds, input normalization or historical selections changed.\n')
 with (R/'CHANGELOG.md').open('a') as f:f.write('\nCompleted real-data grouped-clipping pilot and corrected-sampler matched comparison. Checkpoint recovery proof exact in tested CUDA setting. No long rollout training or TEST. Source/report provenance to be sealed after writers stop.\n')
 write(R/'FINAL_ASSESSMENT_COMPLETE.json',{'status':'COMPLETE_STOPPED_WAITING_REVIEW','selection':selection,'time':time.time(),'optimizer_updates_during_assessment':0})
 print('CORRECTED_ASSESSMENT_COMPLETE_NO_MORE_TRAINING',flush=True)
if __name__=='__main__':main()
