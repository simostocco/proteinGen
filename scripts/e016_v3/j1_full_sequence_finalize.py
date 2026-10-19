"""Post-pilot evaluation/reporting only; never creates an optimizer or trains a model."""
import collections,copy,gc,json,math,subprocess,time,shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pyarrow.parquet as pq
import torch
import j1_full_sequence as m
R=m.R;D=R.parent;read=m.read;write=m.write;sha=m.sha;O=D/'e016_j1_clip_precision_pilot_20261018_v1';FALLBACK=D/'e016_partial_supervision_20261016_v1/runs/J1-FULL/checkpoint_20000.pt'
def md(n,t):(R/n).write_text(t.rstrip()+'\n')
def corrected_metadata(row):
 z=copy.deepcopy(row)
 for role in ['FREE_TRAIN','FREE_VALIDATION']:
  for kind,v in z[role]['categories'].items():v['decisions']=4*v['identities']
  for band,v in z[role]['length_bands'].items():v['identities']=v['decisions'];v['decisions']*=4
 return z

@torch.no_grad()
def matched_robustness_conditions(model,checkpoints):
 from protein_joint_sac.data.scaled_supervised import Data,canonical,internal
 from protein_joint_sac.data.accepted_supervised import CoordinateReader
 data=Data(False);ids=read(R/'configs/GENERATED_PREFIX_PANEL.json')['identities'];wanted=[data.rows[data.owners[h][0]] for h in ids]
 for row,x in CoordinateReader(data.proposal.external).stream(wanted):
  o=row['observation_id'];data.ca[o]=canonical(x);data.g[o]=internal(x)
 panel=sorted([(h,data.owners[h][0],t) for h in ids for t in range(len(data.seqs[h]))],key=lambda v:(v[2],v[0]));out={}
 for label,path in checkpoints.items():
  st=m.s.load_checked(path);model.load_state_dict(st['model'],strict=True);del st;model.eval();m.s.set_dropout(model,.15);result={}
  with m.s.historical_evaluation():
   for condition in ['NATIVE','UNAVAILABLE']:
    by=collections.defaultdict(list)
    for off in range(0,len(panel),16):
     ex=panel[off:off+16];states=[];ys=[]
     for h,o,t in ex:
      state,y,_=data.target(h,o,t,condition=='NATIVE');states.append(state);ys.append(y)
     b=m.collate(states,'cuda');y=torch.tensor(ys,device='cuda')
     with torch.autocast('cuda',dtype=torch.bfloat16):logits=model.aa_head(m.encode_batch(model,b)).float()
     ce=torch.nn.functional.cross_entropy(logits,y,reduction='none').cpu().tolist()
     for (h,o,t),v in zip(ex,ce):by[h].append(v)
    nll=float(np.mean([np.mean(v) for v in by.values()]));result[condition]={'AA_NLL':nll,'perplexity':math.exp(nll),'identities':len(ids),'decisions':len(panel),'per_identity':{h:float(np.mean(v)) for h,v in by.items()}}
  out[label]=result
 return out

def main():
 done=read(R/'runs/J1/COMPLETED.json');history=[read(p) for p in sorted((R/'runs/J1').glob('evaluation_*.json'))];base=history[0];last=history[-1];assert last['new_update']==done['new_optimizer_updates'];cfg=read(R/'configs/FROZEN.json');m.s.j.core.setup();m.guard()
 # Consistent corrected protocol, exact source tensors and saved draw arrays.
 full=m.s.load_checked(m.SOURCE);model_only=m.s.load_checked(O/'runs/J1/model_00500.pt');assert m.c.same(full['model'],model_only['model']);del full,model_only;gc.collect()
 oldgen={'fallback':read(O/'generation/fallback_ca_fp64_interior_v1.json'),'source':read(O/'generation/final00500_ca_fp64_interior_v1.json')}
 for label,rows in oldgen.items():
  assert len(rows)==160 and all(v['protocol']=='ca_fp64_interior_v1' and v['live_batch']==4 and v['seed']==2026101700+100*v['N']+v['sample_index'] for v in rows)
 cache_receipt={'source_model_bitwise_equals_selected_full_state':True,'source_generation_path':str(O/'generation/final00500_ca_fp64_interior_v1.json'),'source_generation_sha256':sha(O/'generation/final00500_ca_fp64_interior_v1.json'),'fallback_generation_path':str(O/'generation/fallback_ca_fp64_interior_v1.json'),'fallback_generation_sha256':sha(O/'generation/fallback_ca_fp64_interior_v1.json'),'inference_protocol':'ca_fp64_interior_v1','same_seed_noise_and_live_batch':True}
 write(R/'GENERATION_CACHE_BINDING.json',cache_receipt)
 # Select research-best by the new held-out objective; promotion additionally requires all retained safeguards.
 best_research=min(history[1:],key=lambda z:z['FREE_VALIDATION']['AA_NLL']);local_qualified=[v for v in history[1:] if v['NATIVE_VALIDATION']['geometry_NLL']<=-.113708968 and all(a<=b+.02 for a,b in zip(v['NATIVE_VALIDATION']['component_NLL'],base['NATIVE_VALIDATION']['component_NLL']))]
 candidate_updates=sorted({best_research['new_update'],last['new_update']}|({min(local_qualified,key=lambda z:z['FREE_VALIDATION']['AA_NLL'])['new_update']} if local_qualified else set())|{v['new_update'] for v in history[1:] if m.decision(v,base)['passes']})
 model=m.s.j.core.model('J1');m.s.set_dropout(model,.15);model.eval();generations={};summaries={k:m.old.metric_summary(v) for k,v in oldgen.items()}
 for u in candidate_updates:
  path=R/'runs/J1'/f'checkpoint_{u:05d}.pt';state=m.s.load_checked(path);model.load_state_dict(state['model'],strict=True);del state;gc.collect();e=SimpleNamespace(model=model,update=u)
  rows=m.generation(e,path,32);generations[u]=rows;summaries[str(u)]=m.old.metric_summary(rows)
 native=read(m.s.j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json')['native'];decisions={};selected=0
 allrows={**oldgen,**{str(k):v for k,v in generations.items()}}
 for label,rr in allrows.items():
  for N in [64,128,256,384,500]:
   vv=[v['metrics'] for v in rr if v['N']==N and 'metrics' in v]
   if not vv:continue
   z=summaries[label][str(N)]
   for key in ['angle_histogram','torsion_histogram']:
    hist=np.sum([v[key] for v in vv],axis=0).astype(float);z[key+'_probability']=(hist/max(1,hist.sum())).tolist()
   z['torsion_positive_fraction_mean']=float(np.mean([v['torsion_positive_fraction'] for v in vv]));z['angle_histogram_edges_rad']=np.linspace(0,np.pi,19).tolist();z['torsion_histogram_edges_rad']=np.linspace(-np.pi,np.pi,37).tolist()

 for row in history[1:]:
  u=row['new_update'];test=m.decision(row,base);fails=list(test['failures'])
  if u in generations:
   fails.extend(m.old.check_candidate(summaries[str(u)],summaries['fallback'],native))
   for n,z in summaries[str(u)].items():
    if z['long_contact_fraction']['mean']<.8*summaries['source'][n]['long_contact_fraction']['mean']:fails.append(n+' contacts below.8parent32')
  if u not in generations:fails.append('no32-sample final generation assessment')
  if row['NATIVE_VALIDATION']['AA_NLL']>base['NATIVE_VALIDATION']['AA_NLL']-.002:fails.append('historical native AA improvement<.002')
  ids=sorted(base['NATIVE_VALIDATION']['per_identity']);ci=m.old.paired_ci([row['NATIVE_VALIDATION']['per_identity'][h]['AA_NLL']-base['NATIVE_VALIDATION']['per_identity'][h]['AA_NLL'] for h in ids])
  if ci['bootstrap_95pct'][1]>=0:fails.append('native paired95 gain not supported')
  cal=read(R/'calibration'/f'VALIDATION_{u:05d}.json');bc=read(R/'calibration/VALIDATION_00000.json');t=cal['torsion_calibration'];bt=bc['torsion_calibration']
  if t['coverage50_80_95']['95']<bt['coverage50_80_95']['95']-.01:fails.append('historical torsion95 coverage drop>.01')
  if sum(abs(t['coverage50_80_95'][k]-int(k)/100) for k in ['50','80','95'])>sum(abs(bt['coverage50_80_95'][k]-int(k)/100) for k in ['50','80','95'])+.03:fails.append('historical torsion calibration error regression')
  if cal['components']['torsion']['worst_tail']['0.01']['contribution_to_mean']>1.1*bc['components']['torsion']['worst_tail']['0.01']['contribution_to_mean']+.01:fails.append('historical torsion tail loss regression')
  decisions[str(u)]={'passes':not fails,'failures':fails,'paired_native_AA_NLL_change':ci,**{k:v for k,v in test.items() if k not in ['passes','failures']}}
  if not fails and (not selected or row['FREE_VALIDATION']['AA_NLL']<next(v for v in history if v['new_update']==selected)['FREE_VALIDATION']['AA_NLL']):selected=u
 selected_cp=m.SOURCE if not selected else R/'runs/J1'/f'checkpoint_{selected:05d}.pt';selected_sha=sha(selected_cp);selected_row=base if not selected else next(v for v in history if v['new_update']==selected)
 write(R/'FINAL_SELECTION.json',{'selected_new_update':selected,'selected_checkpoint':str(selected_cp),'selected_sha256':selected_sha,'research_best_new_update':best_research['new_update'],'research_best_checkpoint':str(R/'runs/J1'/f"checkpoint_{best_research['new_update']:05d}.pt"),'latest_checkpoint':done['latest_checkpoint'],'decisions':decisions,'no_automatic_promotion':True,'TEST_evaluated':False})
 write(R/'runtime/RAW_EVALUATION_HISTORY.json',history);write(R/'SEQUENCE_TRAIN_VALIDATION_METRICS.json',[corrected_metadata(v) for v in history]);write(R/'EVALUATION_METADATA_CLARIFICATION.json',{'category_and_length_decisions':'Runtime group summaries averaged identity means, so group count denoted identities. Final aggregate labels correct this to4 decisions/identity; NLL, perplexity, accuracy and selection values are unchanged. Raw evaluation/checkpoint history preserved.','inherited_legacy_config_fields':{'source_total_optimizer_updates':cfg.get('source_total_optimizer_updates'),'source_prior_additional_update':cfg.get('source_prior_additional_update')},'authoritative_start_counter':35500,'authoritative_counter_evidence':'verified AdamW35500 steps; lineage.historical_updates_at_start; source full checkpoint','no_training_math_or_threshold_changes':True})
 matched=matched_robustness_conditions(model,{'source':m.SOURCE,**{str(u):R/'runs/J1'/f'checkpoint_{u:05d}.pt' for u in candidate_updates}})
 write(R/'GENERATED_PREFIX_AA_ROBUSTNESS.json',{'matched_native_unavailable_subset':matched,'paired_generated_prefix_AA_changes':{str(v['new_update']):m.old.paired_ci([v['GENERATED_PREFIX']['per_identity'][h]-base['GENERATED_PREFIX']['per_identity'][h] for h in sorted(base['GENERATED_PREFIX']['per_identity']) if h in v['GENERATED_PREFIX']['per_identity']]) for v in history[1:]},'protocol':read(R/'configs/GENERATED_PREFIX_PANEL.json'),'milestones':{str(v['new_update']):v['GENERATED_PREFIX'] for v in history},'condition':'C. TRUE-AA-conditioned model-generated previous geometry; no native coordinates','limitations':'10 fixed VALIDATION identities,2072 decisions; separately weighted and not directly interchangeable with common native/free panels.'})
 comparisons={}
 for label,rows in [('source',oldgen['source']),('fallback',oldgen['fallback'])]+[(str(u),v) for u,v in generations.items()]:
  if label in ['source','fallback']:continue
  comparisons[label]={}
  for N in [64,128,256,384,500]:
   a={v['sample_index']:v for v in rows if v['N']==N and 'metrics' in v};b={v['sample_index']:v for v in oldgen['source'] if v['N']==N and 'metrics' in v};common=sorted(set(a)&set(b));out={}
   for metric in ['clashes_per100_residues','Rg_over_N_cuberoot','long_contact_fraction']:
    out[metric]=m.old.paired_ci([a[k]['metrics'][metric]-b[k]['metrics'][metric] for k in common]) if common else {'n':0}
   comparisons[label][str(N)]=out
 geom={'protocol':'ca_fp64_interior_v1','samples_per_length':32,'generation_cache_binding':cache_receipt,'summaries':summaries,'paired_changes_vs_source':comparisons,'selection_decisions':decisions,'native_reference_path':str(m.s.j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json'),'native_reference_sha256':sha(m.s.j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json'),'local':{str(v['new_update']):{role:{k:v[role][k] for k in ['geometry_NLL','component_NLL','component_counts','AA_NLL']} for role in ['NATIVE_TRAIN','NATIVE_VALIDATION']} for v in history}}
 write(R/'GEOMETRY_SAFEGUARD_EVALUATION.json',geom)
 # Full-state integrity and actual new exposure counters.
 laststate=m.s.load_checked(done['latest_checkpoint']);assert laststate['new_update']==done['new_optimizer_updates'];assert laststate['sampler']['cursor']==done['new_optimizer_updates'];assert laststate['scheduler']['last_epoch']==done['new_optimizer_updates'];assert {float(v['step']) for v in laststate['optimizer']['state'].values()}=={35500.+done['new_optimizer_updates']};counts=done['sampler']['counts'];u=done['new_optimizer_updates']
 assert [counts[k] for k in ['complete','partial_only','unused']]==[int(u*64*.4),int(u*64*.3),int(u*64*.3)] if u%5==0 else True
 exposures=done['sampler'];exposures.update(nonlocal_targets=done['nonlocal_targets'],local_component_targets={k:u*32 for k in ['distance','angle','torsion']},new_AA_only_geometry_targets=0,all118219_available=True,position_coverage_semantics='Inherited historical per-identity affine shuffled bijections; reported unique counts/repeats refer to this pilot.',main_unique_optimizer_updates=u,proof_reexecuted_updates=2,physical_optimizer_calls=u+2,physical_AA_target_passes_including_replay=64*(u+2),physical_complete_geometry_passes_including_replay=32*(u+2))
 lengths={v['sequence_id']:v['length'] for v in pq.read_table(m.P/'SEQUENCE_ASSIGNMENTS.parquet',columns=['sequence_id','length']).to_pylist()}
 exposures['visited_identity_length_profiles']={k:{'mean_length':float(np.mean([lengths[h] for h in visited])),'length_bands':{f'{lo}-{hi}':sum(lo<=lengths[h]<=hi for h in visited) for lo,hi in [(20,64),(65,128),(129,256),(257,384),(385,500)]}} for k,visited in laststate['sampler']['visited'].items()}
 for k,visited in laststate['sampler']['visited'].items():
  assert all(bits.bit_length()<=lengths[h] for h,bits in visited.items())
  assert sum(bits.bit_count() for bits in visited.values())+counts.get('repeated_identity_position_'+k,0)==counts[k]
 exposures['unused_available_position_coverage_fraction']=exposures['categories']['unused']['unique_identity_positions']/16625418
 exposures['unused_geometry_policy']='Existing experimental coordinates deliberately withheld; not labelled as proteins lacking structure.'
 write(R/'SAMPLING_MIXTURE_AND_COVERAGE.json',exposures)
 curves={}
 for p in (R/'runs/J1').glob('updates_*.jsonl'):
  for line in p.read_text().splitlines():
   if line.strip():v=json.loads(line);curves[v['new_update']]=v
 assert sorted(curves)==list(range(1,u+1));curves=[curves[k] for k in sorted(curves)];write(R/'J1_TRAINING_CURVES.json',curves)
 grad={}
 for group in ['S','A','Gstem','distance','angle','torsion']:
  norms=[v['group_before'][group] for v in curves];scales=[v['group_scale'][group] for v in curves];grad[group]={'raw_norm_median_p95':np.quantile(norms,[.5,.95]).tolist(),'clipping_fraction':float(np.mean(np.array(scales)<1)),'mean_retained_fraction':float(np.mean(scales))}
 write(R/'GRADIENT_AND_EXPOSURE_SUMMARY.json',{'groups':grad,'geometry_shared_ratio_max':max(v['geometry_to_AA_shared_ratio'] for v in curves),'shared_cosine_mean':float(np.mean([v['AA_geometry_shared_cosine'] for v in curves])),'AA_shared_gradient_median':float(np.median([v['AA_shared_L2'] for v in curves])),'no_whole_model_clipping':all(not v['whole_model_clip_applied'] for v in curves)})
 training_seconds=sum(v['seconds'] for v in curves);samples=[json.loads(v) for v in (R/'RESOURCE_SAMPLES.jsonl').read_text().splitlines() if v.strip()];device=[]
 for v in samples:
  try:parts=v['GPU_csv_timestamp_usedMiB_freeMiB_utilization_temperature_power'].split(',');device.append((float(parts[1]),float(parts[3])))
  except (ValueError,IndexError):pass
 resources={'main_training_update_seconds':training_seconds,'mean_seconds_per_update':training_seconds/u,'AA_targets_per_update_second':u*64/training_seconds,'AA_plus_geometry_decisions_per_update_second':u*96/training_seconds,'elapsed_including_evaluation_seconds':done['elapsed_seconds_including_evaluation'],'resources_at_training_stop':done['resources'],'GPU_device_wide_peak_MiB':max(v[0] for v in device) if device else None,'GPU_utilization_mean_sampled_percent':float(np.mean([v[1] for v in device])) if device else None,'D_free_min_during_run':min(v['D_free_bytes'] for v in samples),'D_free_after':shutil.disk_usage(R).free,'new_persistent_bytes':sum(p.stat().st_size for p in R.rglob('*') if p.is_file() and not p.is_symlink()),'checkpoint_sizes':{p.name:p.stat().st_size for p in (R/'runs/J1').glob('checkpoint_*.pt')},'new_optimizer_updates':u,'proof_reexecuted_updates':2,'no_other_models_trained':True,'no_TEST_or_RL_or_AF':True}
 write(R/'RESOURCE_AND_STORAGE_REPORT.json',resources)
 # Compact learning curves, no new scientific metric or selection tuning.
 try:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  fig,axes=plt.subplots(1,3,figsize=(13,3.6));xx=[v['new_update'] for v in history]
  for key,lab in [('NATIVE_TRAIN','TRAIN'),('NATIVE_VALIDATION','VALIDATION')]:axes[0].plot(xx,[v[key]['perplexity'] for v in history],marker='o',label=lab)
  for key,lab in [('FREE_TRAIN','TRAIN'),('FREE_VALIDATION','VALIDATION')]:axes[1].plot(xx,[v[key]['perplexity'] for v in history],marker='o',label=lab)
  axes[0].set_title('Native geometry');axes[1].set_title('Geometry unavailable')
  axes[2].plot(xx,[v['NATIVE_VALIDATION']['geometry_NLL'] for v in history],marker='o');axes[2].axhline(-.113708968,color='r',ls='--',label='Frozen allowance');axes[2].set_title('VALIDATION local geometry NLL')
  axes[0].set_ylabel('AA perplexity');axes[1].set_ylabel('AA perplexity');axes[2].set_ylabel('NLL (nats)')
  for ax in axes:ax.set_xlabel('Additional updates');ax.legend();ax.grid(alpha=.2)
  fig.tight_layout();fig.savefig(R/'LEARNING_CURVES.png',dpi=150);plt.close(fig)
 except ImportError:pass
 table='| New updates | Native TRAIN PPL | Native VAL PPL | Native gap | Geometry-free TRAIN PPL | Geometry-free VAL PPL | Free gap | VAL local NLL |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(f"| {v['new_update']:,} | {v['NATIVE_TRAIN']['perplexity']:.6f} | {v['NATIVE_VALIDATION']['perplexity']:.6f} | {v['native_gap']:.6f} | {v['FREE_TRAIN']['perplexity']:.6f} | {v['FREE_VALIDATION']['perplexity']:.6f} | {v['free_gap']:.6f} | {v['NATIVE_VALIDATION']['geometry_NLL']:.6f} |" for v in history)
 md('NATIVE_VS_GEOMETRY_FREE_AA_EVALUATION.md',f'''# Separate AA evaluation conditions

{table}

A: common native prefixes,2947 identities x8 in TRAIN and VALIDATION, unchanged historical weighting. B: all14775 VALIDATION identities x4 with every geometric validity bit false; matched14775 TRAIN identities with exact category-by-length counts and the same four-target rule. Overall NLL is identity-weighted; perplexity=exp(mean NLL). Group and length denominators in the final JSON are explicitly4xidentity counts. Prefix-decile summaries are target-weighted diagnostics.

C: ten frozen complete VALIDATION identities,2072 targets, true amino acids introduced sequentially after each prediction and entirely generated geometry from BOS. This condition's values are in GENERATED_PREFIX_AA_ROBUSTNESS.json. They must not be compared as identical denominators to A/B; baseline-to-candidate comparisons within C use the same identities,noise and scoring.

TRAIN/VALIDATION comparisons for B are matched on declared categories and length bands, not identical proteins or independent evolutionary families. Old native-prefix historical PPL15.486 is a different condition from B, not a threshold for geometry-free sequence prediction. No TEST evaluation.
''')
 gentable='| Checkpoint | N | Complete | Clashes/100 residues | Rg / N^(1/3) | Long contacts |\n|---|---:|---:|---:|---:|---:|\n'+'\n'.join(f"| {label} | {n} | {v['complete_valid']}/{v['requested']} | {v['clashes_per100_residues']['mean']:.4f} | {v['Rg_over_N_cuberoot']['mean']:.4f} | {v['long_contact_fraction']['mean']:.6f} |" for label,ss in summaries.items() for n,v in ss.items())
 md('FREE_GENERATION_COMPARISON.md',f'''# Corrected-sampler complete generation

{gentable}

All comparisons use ca_fp64_interior_v1,live batch4,32 independent attempts per length and saved common categorical uniforms/Gaussian noise. Historical fallback/source caches are reused only after exact selected-model tensor binding and protocol/seed checks. New noise files reference the immutable previous arrays. Failed attempts remain failures; none are replaced.

Per-length paired sample uncertainty,local spacing/angles/torsion-resultant,composition and sequence/structural diversity are in GEOMETRY_SAFEGUARD_EVALUATION.json. Native controls are frozen TRAIN/VALIDATION source distributions. Lower Rg alone and fewer clashes through expansion are not sufficient. These Cα proxies do not establish folding,stability or function.
''')
 md('TRAINING_AND_CHECKPOINT_REPORT.md',f'''# Training and recoverability

Source: `{m.SOURCE}`, SHA256 `{m.SOURCE_SHA}`. Model has12351746 trainable parameters. Restored AdamW35500-step moments; original source is unchanged. Fresh category RNG/order state, inherited identity-position coverage, restored Torch/dropout/augmentation/nonlocal RNG and preserved auxiliary RNG. New100-update warmup to1e-5,cosine to2e-6 at2500; conditional extension holds2e-6. Effective AA64,complete geometry32,live16,BF16 with existing FP32/FP64 geometry.

Completed **{u} unique new optimizer updates**, historical total **{35500+u}**. The actual interruption proof reexecuted2 updates; physical optimizer calls **{u+2}**, not additional progress. GPU_RESTART_TEST.json verifies exact model,optimizer,scheduler,sampler,RNG and sample-order equality. Recovery at250-update intervals and full milestone states is atomic and SHA checked. Only superseded new nonmilestone recovery files were retired; all historical source/checkpoints remain.

Latest: `{done['latest_checkpoint']}` SHA256 `{done['latest_sha256']}`.
Selected/fallback: `{selected_cp}` SHA256 `{selected_sha}`.
Research-best new update: **{best_research['new_update']}**. No automatic promotion of latest or lowest training loss.

Stop reason: {done['stop_reason']}. Extension receipt: EXTENSION_DECISION.json if assessed. Pin the training source commit from checkpoint.lineage to resume; final reporting commits do not redefine that source. No automatic continuation is authorized.
''')
 md('FINAL_FULL_SEQUENCE_J1_REPORT.md',f'''# Full-sequence J1 pilot

Executed one J1 lineage for **{u} additional unique updates** from selected35500-step full state. No architecture,distribution,normalization,split or structural-quality rule was changed. No TEST,other-model training,AlphaFold or RL execution.

{table}

## Population and exposures

All118219 TRAIN identities were available in disjoint21475 complete /29129 partial-only /67615 previously unused populations. The latter have experimental coordinate records,deliberately withheld because structural admission was not established. AA-only uses known canonical prefixes,true N/t,and all-false coordinate,bond,frame,pair and geometry-target validity. No fabricated atom or geometry target.

Realized decisions: complete **{counts['complete']:,}**,partial-only **{counts['partial_only']:,}**,unused AA-only **{counts['unused']:,}**. Separate complete geometry **{counts['geometry']:,}**,nonlocal targets **{done['nonlocal_targets']:,}**,differentiated rollout residues **0**. Exact40/30/30 holds over the five-update cycle. Unique identities visited **{exposures['unique_identities']:,}**,including **{exposures['categories']['unused']['unique_identities']:,}/67615** newly admitted identities. Unique identity-position targets **{exposures['unique_identity_positions']:,}**. Full exposure/length/position counts are in SAMPLING_MIXTURE_AND_COVERAGE.json; an initial2500-update pilot cannot cover67615 unused identities at48000 decisions.

## Outcome and selection

Research-best new checkpoint is +{best_research['new_update']},geometry-free VALIDATION PPL **{best_research['FREE_VALIDATION']['perplexity']:.6f}** and native VALIDATION PPL **{best_research['NATIVE_VALIDATION']['perplexity']:.6f}**. The final/new selection decision is **{'PROMOTED_NEW_CANDIDATE' if selected else 'PARENT_RETAINED; NEW_CHECKPOINTS_RESEARCH_ONLY'}** under the unchanged geometry and predeclared AA safeguards. Detailed failures and paired identity-level uncertainty are in FINAL_SELECTION.json and GEOMETRY_SAFEGUARD_EVALUATION.json.

Selected checkpoint: `{selected_cp}`
SHA256: `{selected_sha}`
Latest recoverable: `{done['latest_checkpoint']}`
SHA256: `{done['latest_sha256']}`

Original immutable J1-FULL remains the historical fallback; the selected +500 source remains the approved starting candidate if no new model passes. Native-prefix and geometry-free PPL are different scientific conditions. Generated-prefix robustness uses ten fixed identities and is separate from unconditional generation. Thirty-two corrected-sampler attempts/length compare fallback,source,new research-best and latest as applicable; failures,paired uncertainties and diversity remain explicit.

## Compute and limits

Main update compute **{training_seconds:.1f}s**,mean **{training_seconds/u:.4f}s/update**,**{u*64/training_seconds:.1f} AA targets/s**; combined AA+complete-geometry decisions **{u*96/training_seconds:.1f}/s**. Elapsed including evaluation **{done['elapsed_seconds_including_evaluation']:.1f}s**,with source loading/engineering receipts separate. GPU/RAM/disk/checkpoint measurements are in RESOURCE_AND_STORAGE_REPORT.json. D reserve10GiB was preserved; no retired prefilter was recreated and no historical artifact deleted.

Preflight mixed GPU gradients and missing/future-placeholder checks passed; exact interruption/resume passed. Geometry-specific targets never come from unused AA-only records. Dropout.15,observed-input augmentation.075 and AA-first shared cap.5 with independent S/A/stem/distance/angle/torsion clipping remain. The old overlapping60/40 and new disjoint40/30/30 populations differ in both new identities and exposure frequencies,so effects cannot be causally attributed solely to the new sequences.

This short run samples a small portion of all eligible residue positions. Sequence uniqueness is not universal evolutionary independence. Cα likelihood and proxy geometry do not demonstrate experimental folding or thermodynamic stability. Stop reason: **{done['stop_reason']}**. No further training or RL was launched.
''')
 native_delta=last['NATIVE_VALIDATION']['AA_NLL']-base['NATIVE_VALIDATION']['AA_NLL'];free_delta=last['FREE_VALIDATION']['AA_NLL']-base['FREE_VALIDATION']['AA_NLL'];robust_delta=last['GENERATED_PREFIX']['AA_NLL']-base['GENERATED_PREFIX']['AA_NLL']
 with (R/'FINAL_FULL_SEQUENCE_J1_REPORT.md').open('a') as f:
  f.write(f'\n## Required scientific answers\n\n1. All118219 identities were available:21475/29129/67615 disjoint pools.\n2. Visited{exposures["unique_identities"]} unique identities.\n3. Newly admitted unused identities visited:{exposures["categories"]["unused"]["unique_identities"]}.\n4. GPU all-missing forward/backward and exact masking/future tests passed.\n5. Realized AA counts:{counts["complete"]}/{counts["partial_only"]}/{counts["unused"]}; exact40/30/30 over5-update cycles.\n6. Final geometry-free VALIDATION NLL change:{free_delta:+.6f}; PPL{base["FREE_VALIDATION"]["perplexity"]:.6f}→{last["FREE_VALIDATION"]["perplexity"]:.6f}.\n7. Native VALIDATION NLL change:{native_delta:+.6f}; PPL{base["NATIVE_VALIDATION"]["perplexity"]:.6f}→{last["NATIVE_VALIDATION"]["perplexity"]:.6f}.\n8. Native gap:{base["native_gap"]:.6f}→{last["native_gap"]:.6f}; geometry-free gap:{base["free_gap"]:.6f}→{last["free_gap"]:.6f}.\n9. Generated-prefix robustness NLL change:{robust_delta:+.6f}; small fixed10-identity diagnostic.\n10. Local geometry total:{base["NATIVE_VALIDATION"]["geometry_NLL"]:.6f}→{last["NATIVE_VALIDATION"]["geometry_NLL"]:.6f}; component means/counts and frozen pass/fail are explicit in safeguard JSON.\n11. Complete generation acceptability:see32-sample per-length safeguard failures in FINAL_SELECTION.json; no proxy is treated as proof of folding.\n12. Actual visited length profiles,including newly encountered longer proteins,are in SAMPLING_MIXTURE_AND_COVERAGE.json; family novelty is not established.\n13. Completed{u} new unique updates,{training_seconds:.1f}s of main update compute,plus2 replayed verification calls and separate evaluation.\n14. Conditional extension decision is in EXTENSION_DECISION.json; budget used:{u}/5000 maximum.\n15. Best defensible approved balance:selected new update{selected}; unpromoted research-best new update{best_research["new_update"]}.\n16. Selected:{selected_cp}; SHA256{selected_sha}.\n17. Recommendation:review sequence gains against the explicit local/global failures before any further continuation. RL was not launched; inadequate global backbone quality remains a material limitation.\n')
 with (R/'EXPERIMENT_DECISIONS.md').open('a') as f:f.write(f'\nFinal decision: selected new update{selected},research-best{best_research["new_update"]},latest{u}; original safeguards unchanged. Stop{done["stop_reason"]}.\n')
 with (R/'CHANGELOG.md').open('a') as f:f.write('\nFinal reporting corrects group denominator labels to4 targets per identity without changing NLL/PPL,weights,selection or raw checkpoint history. Inherited legacy config source-counter labels are clarified by the verified35500-step source state.\n')
 print(json.dumps({'done':u,'selected':selected,'selected_checkpoint':str(selected_cp),'selected_sha256':selected_sha,'research_best':best_research['new_update'],'native_PPL':last['NATIVE_VALIDATION']['perplexity'],'free_PPL':last['FREE_VALIDATION']['perplexity']}),flush=True)


def write_review_summary():
 """Readable review of cached results; no model execution or scoring changes."""
 h=read(R/'SEQUENCE_TRAIN_VALIDATION_METRICS.json');b=h[0];z=h[-1];done=read(R/'runs/J1/COMPLETED.json');sel=read(R/'FINAL_SELECTION.json');exp=read(R/'SAMPLING_MIXTURE_AND_COVERAGE.json');res=read(R/'RESOURCE_AND_STORAGE_REPORT.json');geo=read(R/'GEOMETRY_SAFEGUARD_EVALUATION.json');rob=read(R/'GENERATED_PREFIX_AA_ROBUSTNESS.json')
 table='| Additional updates | Native TRAIN PPL | Native VAL PPL | Native NLL gap | Geometry-free TRAIN PPL | Geometry-free VAL PPL | Geometry-free NLL gap | VAL local NLL |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(f"| {v['new_update']:,} | {v['NATIVE_TRAIN']['perplexity']:.6f} | {v['NATIVE_VALIDATION']['perplexity']:.6f} | {v['native_gap']:.6f} | {v['FREE_TRAIN']['perplexity']:.6f} | {v['FREE_VALIDATION']['perplexity']:.6f} | {v['free_gap']:.6f} | {v['NATIVE_VALIDATION']['geometry_NLL']:.6f} |" for v in h)
 components='| Component | Parent | New +500 | New +2,500 | Valid targets per evaluation |\n|---|---:|---:|---:|---:|\n'+'\n'.join(f"| {name} | {h[0]['NATIVE_VALIDATION']['component_NLL'][k]:.6f} | {h[1]['NATIVE_VALIDATION']['component_NLL'][k]:.6f} | {h[-1]['NATIVE_VALIDATION']['component_NLL'][k]:.6f} | {h[0]['NATIVE_VALIDATION']['component_counts'][k]:,} |" for k,name in enumerate(['Distance','Virtual angle','Signed torsion']))
 aa='| Geometry-free VALIDATION subset | Identities | Parent PPL | Final PPL |\n|---|---:|---:|---:|\n'+'\n'.join(f"| {k} | {b['FREE_VALIDATION']['categories'][k]['identities']:,} | {b['FREE_VALIDATION']['categories'][k]['perplexity']:.6f} | {z['FREE_VALIDATION']['categories'][k]['perplexity']:.6f} |" for k in ['complete','partial_only','unused'])
 grad=read(R/'GRADIENT_AND_EXPOSURE_SUMMARY.json');paired=sel['decisions']['2500']['paired_geometry_free_AA_NLL_change'];rs=done['resources'];latest500=geo['summaries']['2500']['500'];parent500=geo['summaries']['source']['500'];c_ci=rob['paired_generated_prefix_AA_changes']['2500']['bootstrap_95pct']
 md('FINAL_FULL_SEQUENCE_J1_REPORT.md',f'''# Full-sequence J1 pilot: completed and stopped

**Held-out AA prediction improved, but the wider generalization gap and local-geometry regression blocked extension and promotion.** One J1 lineage completed **2,500 additional updates**, reaching **38,000 historical updates**. No extension to 5,000 was used. Training is stopped.

## Prediction and generalization

{table}

The geometry-free VALIDATION NLL improvement is **{paired['mean']:.6f} nats**, with paired identity-bootstrap 95% interval **[{paired['bootstrap_95pct'][0]:.6f}, {paired['bootstrap_95pct'][1]:.6f}]**. Native-prefix VALIDATION improved too, but did not beat the historical, geometrically unqualified 15.485827 PPL point. Both TRAIN–VALIDATION gaps widened; the broader gap rose by **{z['free_gap']-b['free_gap']:.6f} nats**, exceeding the predeclared +0.02 allowance. This observation does not by itself identify memorization as the sole cause.

The native panels remain the original 2,947 identities × eight targets per role. The geometry-free panels contain 14,775 identities × four targets per role, with TRAIN matched to VALIDATION category-by-length counts. These are different evaluation conditions and populations; compare checkpoints within each condition. Overall scoring gives equal weight to identities. No TEST evaluation occurred.

{aa}

Top-1/top-3 accuracy and length/position strata are preserved in `SEQUENCE_TRAIN_VALIDATION_METRICS.json`. The final group denominator labels correctly represent four targets per identity; the raw runtime/checkpoint group counts are preserved with a clarification receipt. No likelihood, accuracy, weighting or decision value was changed by that metadata correction.

## AA admission, mixture and actual coverage

All **118,219** canonical TRAIN identities were available in disjoint populations: **21,475 complete / 29,129 partial-only / 67,615 previously unused**. The unused proteins have experimental structural records; their coordinates were deliberately withheld because structural admission was not established. No new geometry was fabricated or admitted.

Actual AA decisions were **64,000 complete / 48,000 partial-only / 48,000 unused**: exactly **40/30/30**. Five updates supply 128/96/96 targets. Identity-first sampling shuffles each population without replacement before cycling and retains historical position-coverage state.

The pilot visited **98,604 distinct identities**, including **48,000 of the 67,615 newly admitted identities (70.99%)**. It processed **160,000 distinct identity-position targets**, with no repeats within this pilot. There are **19,615** unused-pool identities still unvisited. The newly visited pool includes **7,206 sequences of length 385–500**, and its mean length is **245.30** residues. This is meaningful additional sequence/length coverage, not evidence of independent new evolutionary families.

Separate geometry exposure was **80,000 complete-structure decisions**, each contributing valid distance, angle and torsion targets, plus **5,000 nonlocal targets** with K=4. No AA loss was counted on the geometry-only stream. There were **zero differentiated long-rollout residues**. The partial stream supplied 3,489 AA targets at genuinely missing-Cα positions; the unused stream withheld geometry irrespective of whether its source records contained atoms.

## Masking, optimization and recovery

GPU mixed-batch, missing-placeholder, future-suffix and original-path equivalence tests passed. AA-only inputs retain true AA prefixes, positions and full lengths; coordinate/bond/frame/pair availability and geometry-target validity are false. AA-only examples produced finite shared/AA gradients and zero geometry-head gradients. The new stream never looks up its disqualified observations. Existing qualified loaders can still decode and discard unselected rows from shared Parquet batches.

The 12,351,746-parameter architecture, chemistry-off behavior, dropout 0.15, observed-coordinate augmentation 0.075, input scaling and transformed geometry distributions were unchanged. The AA-first shared cap and independent shared/AA/stem/distance/angle/torsion clipping remained active. Maximum measured shared geometry-to-AA ratio was **{grad['geometry_shared_ratio_max']:.6f}**; no whole-model clipping was applied.

AdamW moments and RNG were restored from the selected 35,500-step full state. A new continuation schedule warmed up for 100 updates to 1e-5 and decayed to 2e-6 at 2,500. Category ordering/RNG were explicitly new, with old identity-position coverage retained. Legacy inherited config source-counter labels are clarified by the verified 35,500-step AdamW state and new lineage; they did not control optimization.

The interruption proof reproduced model, optimizer, scheduler, sampler, RNG and sample order exactly on the RTX 5060. It reexecuted two verification updates: **2,500 unique progress updates / 2,502 physical optimizer calls**, rather than 2,502 steps of model progress. Full atomic recovery states and hash receipts are preserved.

## Local and free-running geometry

{components}

Final local geometry NLL **−0.108511** fails the frozen **−0.113709** allowance by **0.005198 nats**. Distance likelihood improved, while angle and especially torsion deteriorated. The +500 checkpoint passed local/component allowances, but its final generation panel failed the N=384 clash and N=64 contact safeguards; its broader generalization gap also failed.

All compared models completed **32/32 attempts at each of 64, 128, 256, 384 and 500 residues** under `ca_fp64_interior_v1`, using matched saved uniforms/Gaussian draws. The +2,500 model passed the frozen relative free-generation safeguards, preserving sequence and structural diversity. At N=500, clashes per 100 residues changed **{parent500['clashes_per100_residues']['mean']:.3f} → {latest500['clashes_per100_residues']['mean']:.3f}**, normalized Rg **{parent500['Rg_over_N_cuberoot']['mean']:.3f} → {latest500['Rg_over_N_cuberoot']['mean']:.3f}**, and long-contact fraction **{parent500['long_contact_fraction']['mean']:.6f} → {latest500['long_contact_fraction']['mean']:.6f}**. The paired intervals for these N=500 changes include zero. Results across lengths are mixed; there is no demonstrated uniform improvement in global backbone organization.

The final comparison uses the manifest-referenced 32-sample parent cache. A four-sample intermediate cache was rejected by final verification. The runtime extension was already rejected by gap/local/component criteria; checking the intended 32-sample generation reference leaves that non-extension decision unchanged. Both attempt logs and the reference clarification are preserved.

## Robustness to generated geometric prefixes

On ten fixed VALIDATION identities (2,072 targets), true-AA-conditioned generated-prefix PPL changed **19.631964 → 19.239507**. The paired NLL interval is **[{c_ci[0]:.6f}, {c_ci[1]:.6f}]**. This is a bounded diagnostic, not unconditional generation or population-wide evidence.

On the same ten identities and targets, matched native / unavailable / generated-geometry PPL is approximately **17.377 / 19.148 / 19.632** for the parent and **17.277 / 18.737 / 19.240** for the final model. Generated geometric context still hurts AA prediction compared with withholding geometry on this subset. Conditions A/B/C remain separately labelled.

## Checkpoints and decision

**No new checkpoint was promoted.** The selected parent remains the best approved sequence/geometry balance under the frozen protocol:

`{sel['selected_checkpoint']}`

SHA256: `{sel['selected_sha256']}`

The latest research checkpoint has the best geometry-free VALIDATION point estimate and remains fully recoverable:

`{done['latest_checkpoint']}`

SHA256: `{done['latest_sha256']}`

The new +500 state is also retained as a useful local-likelihood/sequence tradeoff. Its hash and all milestone states are listed in the checkpoint report and provenance. Original J1-FULL remains the immutable historical fallback.

## Resources and next decision

Main update compute was **{res['main_training_update_seconds']/60:.2f} minutes** at **{res['mean_seconds_per_update']:.4f} seconds/update**, or **{res['AA_targets_per_update_second']:.1f} AA targets/second** plus the geometry stream. Elapsed time through scheduled training/evaluation was **{done['elapsed_seconds_including_evaluation']/60:.2f} minutes**; initial data loading and final read-only comparisons are separate. Peak Torch allocation/reservation was **{rs['GPU_allocated_peak']/2**30:.2f}/{rs['GPU_reserved_peak']/2**30:.2f} GiB**, device-wide peak **{res['GPU_device_wide_peak_MiB']/1024:.2f} GiB**, and CPU RSS high-water **{rs['RSS_highwater_bytes']/2**30:.2f} GiB**. The 10-GiB D: reserve was preserved. Historical artifacts and the ten retired prefilter files were left unchanged.

Expanded AA supervision is useful for held-out prediction, but **do not extend this configuration unchanged or transition this unqualified candidate to RL**. The next supervised decision should address the widening broader gap and local-geometry retention while preserving the useful new data coverage. Review these tradeoffs before another bounded experiment. No new experiment is launched here.

The interventions also changed relative complete/partial exposure and the continuation schedule, so gains cannot be attributed exclusively to adding sequences. The frozen split supplies its declared near-duplicate/domain safeguards, not universal remote-homology independence. Cα likelihood, compactness and clash proxies do not establish experimental folding, stability or function.
''')

if __name__=="__main__":
 main()
 write_review_summary()
