"""Post-run comparison and provenance preparation. No optimizer or training updates."""
import collections,copy,gc,json,math,platform,subprocess,sys,time,shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pyarrow.parquet as pq
import torch
import j1_bidirectional_multitask as q
import j1_full_sequence_finalize as ff
R=q.R;D=q.D;m=q.m;s=q.s;c=q.c;read=q.read;write=q.write;sha=q.sha

def old_policy(row,base,gen):
 fail=[];v=row['NATIVE_VALIDATION'];tr=row['NATIVE_TRAIN'];b=base['NATIVE_VALIDATION'];keys=sorted(b['per_identity']);ci=m.old.paired_ci([v['per_identity'][h]['AA_NLL']-b['per_identity'][h]['AA_NLL'] for h in keys])
 if v['AA_NLL']>b['AA_NLL']-.002:fail.append('native AA gain<.002')
 if ci['bootstrap_95pct'][1]>=0:fail.append('native paired95 gain unsupported')
 if row['native_gap']>base['native_gap']+.02:fail.append('native gap extra>.02')
 if row['free_gap']>base['free_gap']+.02:fail.append('geometry-free gap extra>.02')
 if v['geometry_NLL']>-.113708968:fail.append('original local threshold')
 if any(a>z+.02 for a,z in zip(v['component_NLL'],b['component_NLL'])):fail.append('original component allowance')
 fallback=read(q.O/'generation/fallback_ca_fp64_interior_v1.json');parent=read(q.O/'generation/final00500_ca_fp64_interior_v1.json');native=read(s.j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json')['native'];ss=m.old.metric_summary(gen);fail.extend(m.old.check_candidate(ss,m.old.metric_summary(fallback),native));ps=m.old.metric_summary(parent)
 for N,z in ss.items():
  if z['long_contact_fraction']['mean']<.8*ps[N]['long_contact_fraction']['mean']:fail.append(N+' original contacts')
 cal=read(R/'calibration'/f"VALIDATION_{row['new_update']:05d}.json");bc=read(R/'calibration/VALIDATION_00000.json');t=cal['torsion_calibration'];bt=bc['torsion_calibration']
 if t['coverage50_80_95']['95']<bt['coverage50_80_95']['95']-.01:fail.append('original torsion95 coverage')
 if sum(abs(t['coverage50_80_95'][k]-int(k)/100) for k in ['50','80','95'])>sum(abs(bt['coverage50_80_95'][k]-int(k)/100) for k in ['50','80','95'])+.03:fail.append('original torsion calibration error')
 if cal['components']['torsion']['worst_tail']['0.01']['contribution_to_mean']>1.1*bc['components']['torsion']['worst_tail']['0.01']['contribution_to_mean']+.01:fail.append('original torsion tail')
 return {'passes':not fail,'failures':fail,'paired_native_change':ci,'historical_policy_not_rewritten':True}

@torch.no_grad()
def matched_conditions(model,checkpoints):
 from protein_joint_sac.data.scaled_supervised import Data,canonical,internal
 from protein_joint_sac.data.accepted_supervised import CoordinateReader
 data=Data(False);ids=read(R/'configs/GENERATED_PREFIX_PANEL.json')['identities'];wanted=[data.rows[data.owners[h][0]] for h in ids]
 for row,x in CoordinateReader(data.proposal.external).stream(wanted):
  o=row['observation_id'];data.ca[o]=canonical(x);data.g[o]=internal(x)
 panel=sorted([(h,data.owners[h][0],t) for h in ids for t in range(len(data.seqs[h]))],key=lambda v:(v[2],v[0]));results={}
 for label,path in checkpoints.items():
  st=s.load_checked(path);model.load_state_dict(st['model'],strict=True);del st;model.eval();by={k:collections.defaultdict(list) for k in ['NATIVE','UNAVAILABLE','KL_NATIVE_TO_UNAVAILABLE']}
  with s.historical_evaluation():
   for off in range(0,len(panel),16):
    ex=panel[off:off+16];probs={};logps={};ys=[]
    for condition in ['NATIVE','UNAVAILABLE']:
     states=[];ys=[]
     for h,o,t in ex:
      state,y,_=data.target(h,o,t,condition=='NATIVE');states.append(state);ys.append(y)
     b=m.collate(states,'cuda');y=torch.tensor(ys,device='cuda')
     with torch.autocast('cuda',dtype=torch.bfloat16):logits=model.aa_head(m.encode_batch(model,b)).float()
     ce=torch.nn.functional.cross_entropy(logits,y,reduction='none').cpu().tolist();logps[condition]=logits.log_softmax(-1);probs[condition]=logps[condition].exp()
     for (h,o,t),loss in zip(ex,ce):by[condition][h].append(loss)
    kl=(probs['NATIVE']*(logps['NATIVE']-logps['UNAVAILABLE'])).sum(-1).cpu().tolist()
    for (h,o,t),value in zip(ex,kl):by['KL_NATIVE_TO_UNAVAILABLE'][h].append(value)
  result={}
  for condition,v in by.items():
   per={h:float(np.mean(values)) for h,values in v.items()};mean=float(np.mean(list(per.values())));result[condition]={'identity_mean_NLL' if condition!='KL_NATIVE_TO_UNAVAILABLE' else 'identity_mean_KL':mean,'identities':len(per),'decisions':len(panel),'per_identity':per}
   if condition!='KL_NATIVE_TO_UNAVAILABLE':result[condition]['perplexity']=math.exp(mean)
  results[label]=result
 return results

def main():
 done=read(R/'runs/J1/COMPLETED.json');u=done['new_optimizer_updates'];hist=[read(p) for p in sorted((R/'runs/J1').glob('evaluation_*.json'))];base=hist[0];assert hist[-1]['new_update']==u,'final evaluation required before publication';cfg=read(R/'configs/FROZEN.json');s.j.core.setup();m.guard();assert sha(m.SOURCE)==m.SOURCE_SHA
 source=s.load_checked(m.SOURCE);original_model_only=s.load_checked(q.O/'runs/J1/model_00500.pt');assert c.same(source['model'],original_model_only['model']);del original_model_only
 parent=read(q.O/'generation/final00500_ca_fp64_interior_v1.json');fallback=read(q.O/'generation/fallback_ca_fp64_interior_v1.json');research=read(q.PREV/'generation/new02500_ca_fp64_interior_v1.json');research_cp=q.PREV/'runs/J1/checkpoint_02500.pt';research_sha=sha(research_cp)
 caches={'accepted_parent':parent,'original_J1_FULL':fallback,'previous_full_sequence_2500':research}
 for tag,rows in caches.items():assert len(rows)==160 and all(v['protocol']=='ca_fp64_interior_v1' and v['live_batch']==4 and v['seed']==2026101700+100*v['N']+v['sample_index'] for v in rows)
 assert all(v['checkpoint_sha256']==research_sha for v in research)
 assert all(v['checkpoint_sha256']==sha(q.O/'runs/J1/model_00500.pt') for v in parent)
 assert all(v['checkpoint_sha256']=='9e9215d3720ffc73edb0c3e382db25332f91cb26b0c8d38d01668e383b1e3810' for v in fallback)
 best=min(hist[1:],key=lambda z:z['FREE_VALIDATION']['AA_NLL']);local=[v for v in hist[1:] if v['NATIVE_VALIDATION']['geometry_NLL']<=cfg['selection_new']['local_NLL_max'] and v['NATIVE_VALIDATION']['AA_NLL']<=base['NATIVE_VALIDATION']['AA_NLL']+.005]
 candidates={best['new_update'],u}|({7500} if any(v['new_update']==7500 for v in hist) else set())|({min(local,key=lambda z:z['FREE_VALIDATION']['AA_NLL'])['new_update']} if local else set())|{v['new_update'] for v in hist[1:] if q.decision(v,base)['failures']==['final32 generation assessment pending']}
 model=s.j.core.model('J1');s.set_dropout(model,.15);model.eval();generations={};summaries={k:m.old.metric_summary(v) for k,v in caches.items()};decisions={};selected=0
 for v in sorted(candidates):
  p=R/'runs/J1'/f'checkpoint_{v:05d}.pt';st=s.load_checked(p);model.load_state_dict(st['model'],strict=True);assert sum(z.numel() for z in st['auxiliary'].values())==7412;del st;gc.collect();rows=q.generation(SimpleNamespace(model=model,update=v),p,32);generations[str(v)]=rows;summaries[str(v)]=m.old.metric_summary(rows);row=next(x for x in hist if x['new_update']==v);gencheck=q.generation_check(rows,32);new=q.decision(row,base,gencheck);old=old_policy(row,base,rows);decisions[str(v)]={'new_sequence_first':new,'original_policy':old,'generation_new':gencheck}
  if new['passes'] and (not selected or row['FREE_VALIDATION']['AA_NLL']<next(x for x in hist if x['new_update']==selected)['FREE_VALIDATION']['AA_NLL']):selected=v
 selected_cp=m.SOURCE if not selected else R/'runs/J1'/f'checkpoint_{selected:05d}.pt';selected_sha=sha(selected_cp)
 paired={}
 for label,rows in generations.items():
  paired[label]={}
  for N in [64,128,256,384,500]:
   a={v['sample_index']:v for v in rows if v['N']==N and 'metrics' in v};b={v['sample_index']:v for v in parent if v['N']==N and 'metrics' in v};ks=sorted(set(a)&set(b));paired[label][str(N)]={key:m.old.paired_ci([a[k]['metrics'][key]-b[k]['metrics'][key] for k in ks]) for key in ['clashes_per100_residues','Rg_over_N_cuberoot','long_contact_fraction']}
 # Additional distribution summaries use identical native/control definitions.
 for label,rows in {**caches,**generations}.items():
  for N in [64,128,256,384,500]:
   vv=[v['metrics'] for v in rows if v['N']==N and 'metrics' in v]
   if not vv:continue
   z=summaries[label][str(N)]
   for key in ['angle_histogram','torsion_histogram']:
    h=np.sum([v[key] for v in vv],axis=0).astype(float);z[key+'_probability']=(h/h.sum()).tolist()
   z['torsion_positive_fraction_mean']=float(np.mean([v['torsion_positive_fraction'] for v in vv]))
 write(R/'FREE_GENERATION_COMPARISON.json',{'protocol':'ca_fp64_interior_v1','attempts_per_length':32,'common_draws_and_live_batch':True,'summaries':summaries,'paired_vs_accepted_parent':paired,'decisions':decisions,'native_controls':str(s.j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json'),'source_cache_binding':'accepted full tensors bitwise equal cached model-only state; previous full-sequence cache bound to exact checkpoint hash','failed_samples_not_replaced':True,'no_TEST':True})
 robust_checkpoints={'accepted_parent':m.SOURCE,**{str(v):R/'runs/J1'/f'checkpoint_{v:05d}.pt' for v in sorted(candidates)}};matched=matched_conditions(model,robust_checkpoints)
 write(R/'GENERATED_PREFIX_AA_ROBUSTNESS.json',{'milestones':{str(v['new_update']):v['GENERATED_PREFIX'] for v in hist},'same_ten_identity_native_unavailable':matched,'limitation':'Ten identities; true-AA-conditioned geometry generation, not standard teacher-forced or unconditional generation.'});del model;torch.cuda.empty_cache()
 state=s.load_checked(done['latest_checkpoint']);assert state['new_update']==u==state['scheduler']['last_epoch']==state['sampler']['cursor'];assert state['masked_sampler']['counts']['examples']==16*u
 cov=done['next_AA_sampler'];ms=done['masked_sampler'];aa={h:bits for cat in state['sampler']['visited'].values() for h,bits in cat.items()};mb=state['masked_sampler']['bits'];intersection=sum((bits&aa.get(h,0)).bit_count() for h,bits in mb.items());unique_next=sum(v.bit_count() for v in aa.values());unique_mask=sum(v.bit_count() for v in mb.values());pops={k:set(state['sampler']['orders'][k]) for k in ['complete','partial_only','unused']};assert sum(len(v) for v in pops.values())==118219 and not any(pops[a]&pops[b] for a in pops for b in pops if a<b)
 coverage={'new_updates':u,'base_nextAA':cov,'masked':ms,'identity_position_intersection_between_tasks':intersection,'unique_nextAA_identity_positions':unique_next,'unique_masked_identity_positions':unique_mask,'unique_union_identity_positions':unique_next+unique_mask-intersection,'distinct_union_identities':len(set(aa)|set(mb)),'paired_prefix_views':done['paired_target_exposures'],'weighted_nextAA_target_decisions':64*u,'additional_paired_forward_views':done['paired_target_exposures'],'complete_geometry_decisions':32*u,'nonlocal_targets':done['nonlocal_targets'],'long_rollout_residues':0,'per_population':{k:{'population':len(ids),'masked_unique_identities':len(set(mb)&ids),'masked_unique_positions':sum(mb[h].bit_count() for h in set(mb)&ids),'nextAA_unique_positions':sum(aa[h].bit_count() for h in set(aa)&ids),'task_intersection_positions':sum((mb[h]&aa[h]).bit_count() for h in set(aa)&set(mb)&ids)} for k,ids in pops.items()},'previous_unused_prefix_only_reference_positions':48000,'unused_masked_unique_positions_vs_previous_48k_ratio':sum(mb[h].bit_count() for h in set(mb)&pops['unused'])/48000,'labels_not_independent_evolutionary_families':True}
 lengths={z['sequence_id']:z['length'] for z in pq.read_table(m.P/'SEQUENCE_ASSIGNMENTS.parquet',columns=['sequence_id','length']).to_pylist()};sm=state['masked_sampler'];full=sm['order'];assert sm['epoch']*len(full)+sm['cursor']==16*u
 coverage['masked_full_sequence_input_tokens']=sm['epoch']*sum(lengths[h] for h in full)+sum(lengths[h] for h in full[:sm['cursor']])
 coverage['masked_input_tokens_are_not_supervised_labels']=True
 coverage['masked_examples_by_population']={k:sm['epoch']*len(ids)+sum(h in ids for h in full[:sm['cursor']]) for k,ids in pops.items()}
 for k,ids in pops.items():
  z=coverage['per_population'][k];z['unique_union_positions']=z['masked_unique_positions']+z['nextAA_unique_positions']-z['task_intersection_positions']
 coverage['unused_unique_union_fraction_of_available_labels']=coverage['per_population']['unused']['unique_union_positions']/16625418
 pairpath=R/'runs/J1/checkpoint_02500.pt'
 if pairpath.exists():
  oldstate=s.load_checked(q.PREV/'runs/J1/checkpoint_02500.pt');paired_state=s.load_checked(pairpath);same=c.same(oldstate['sampler'],paired_state['sampler']);write(R/'BASE_TARGET_SCHEDULE_PAIRING.json',{'compared_update':2500,'base_AA_and_complete_geometry_sampler_state_exact':same,'previous_checkpoint':str(q.PREV/'runs/J1/checkpoint_02500.pt'),'previous_sha256':research_sha,'new_checkpoint':str(pairpath),'new_sha256':sha(pairpath),'interpretation':'Identical sampler states support paired base AA/G exposure schedules; MLM, paired-view objective, dropout draws and LR schedule differ, so this is not a controlled isolated-loss ablation.'});del oldstate,paired_state;gc.collect()
 write(R/'TRAINING_TOKEN_AND_IDENTITY_COVERAGE.json',coverage)
 manifests=[]
 for p in sorted((R/'runs/J1').glob('*.pt')):
  st=s.load_checked(p);v=st['new_update'];assert st['scheduler']['last_epoch']==v and st['sampler']['cursor']==v
  for i,pg in enumerate(st['optimizer']['param_groups']):
   vals={int(st['optimizer']['state'][k]['step']) for k in pg['params'] if k in st['optimizer']['state']};assert vals==({35500+v} if i==0 else ({v} if v else set()))
  assert c.same(st['lineage']['config'],cfg);manifests.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'new_updates':v,'total_updates':35500+v,'full_recoverable':True,'contains':['model','auxiliary','optimizer','scheduler','sampler','masked_sampler','rng','task_config','evaluation_receipts'],'source_commit':st['lineage']['source_commit']});del st;gc.collect()
 delta={}
 for prefix in ['encoder.','aa_head.','geom.','selected_aa.']+[f'encoder.blocks.{i}.' for i in range(8)]:
  differences=[(state['model'][n].double()-v.double()) for n,v in source['model'].items() if n.startswith(prefix)];delta[prefix]={'L2':math.sqrt(sum(float(v.square().sum()) for v in differences)),'max_abs':max(float(v.abs().max()) for v in differences)}
 assert delta['encoder.']['L2']>0 and delta['aa_head.']['L2']>0 and delta['geom.']['L2']>0 and all(delta[f'encoder.blocks.{i}.']['L2']>0 for i in range(8))
 write(R/'CHECKPOINT_MANIFEST.json',{'selected_checkpoint':str(selected_cp),'selected_sha256':selected_sha,'selected_new_updates':selected,'research_best_checkpoint':str(R/'runs/J1'/f"checkpoint_{best['new_update']:05d}.pt"),'research_best_new_updates':best['new_update'],'latest_checkpoint':done['latest_checkpoint'],'latest_sha256':done['latest_sha256'],'checkpoints':manifests,'production_parameters':12351746,'training_only_auxiliary_parameters':7412,'deployment_uses_model_only':True,'source_model_after_sha256':sha(m.SOURCE),'original_policy_vs_new_policy':decisions})
 logs={}
 for p in (R/'runs/J1').glob('updates_from_*.jsonl'):
  for line in p.read_text().splitlines():
   if line.strip():z=json.loads(line);logs[z['new_update']]=z
 assert sorted(logs)==list(range(1,u+1));curves=[logs[k] for k in sorted(logs)];write(R/'MULTITASK_TRAINING_CURVES.json',curves)
 def summary(values):return {'mean':float(np.mean(values)),'median':float(np.median(values)),'p95':float(np.percentile(values,95)),'max':float(max(values))}
 gradient=read(R/'MULTITASK_GRADIENT_BALANCE.json');gradient['main']={key:summary([v[key] for v in curves]) for key in ['primary_AA_shared_L2','sequence_aux_shared_raw_L2','sequence_aux_scale','sequence_aux_to_primary_ratio','primary_aux_shared_cosine','geometry_to_primary_AA_shared_ratio','AA_geometry_shared_cosine','aux_head_gradient_L2_before_clip','MASK_embedding_gradient_L2_before_clip']};gradient['group_clipping']={g:{'activation_count':sum(v['group_scale'][g]<1 for v in curves),'norm_before':summary([v['group_before'][g] for v in curves]),'scale':summary([v['group_scale'][g] for v in curves])} for g in ['S','A','Gstem','distance','angle','torsion']};gradient['production_parameter_changes']=delta;write(R/'MULTITASK_GRADIENT_BALANCE.json',gradient)
 write(R/'MASKED_RECONSTRUCTION_METRICS.json',{'condition':'explicit bidirectional masked tokens, not next-AA perplexity','NLL_weighting':'masked-position mean within identity, then identity mean','accuracy_weighting':'correct/total masked targets; per-identity counts also saved','panels_sha256':sha(R/'configs/MASKED_EVALUATION.json'),'milestones':{str(v['new_update']):{role:v['MASKED_'+role] for role in ['TRAIN','VALIDATION']} for v in hist}})
 samples=[json.loads(x) for x in (R/'RESOURCE_SAMPLES.jsonl').read_text().splitlines() if x.strip()];device=[float(v['GPU_csv_timestamp_usedMiB_freeMiB_utilization_temperature_power'].split(',')[1]) for v in samples if v['GPU_csv_timestamp_usedMiB_freeMiB_utilization_temperature_power']!='UNAVAILABLE'];free=shutil.disk_usage(R).free;compute=sum(v['seconds'] for v in curves)
 resources={'main_unique_updates':u,'discarded_preflight_optimizer_calls':6,'main_compute_seconds':compute,'mean_main_update_seconds':compute/u,'nextAA_targets_per_compute_second':64*u/compute,'masked_targets_per_compute_second':ms['counts']['masked_targets']/compute,'elapsed_with_evaluations_seconds':done['elapsed_with_evaluations'],'main_end_peaks':done['resources'],'sampled_device_peak_MiB_including_stress_profile':max(device),'sampled_GPU_utilization_mean_all_phases':float(np.mean([float(v['GPU_csv_timestamp_usedMiB_freeMiB_utilization_temperature_power'].split(',')[3]) for v in samples if v['GPU_csv_timestamp_usedMiB_freeMiB_utilization_temperature_power']!='UNAVAILABLE'])),'minimum_D_free_bytes_sampled':min(v['D_free_bytes'] for v in samples),'D_free_after_bytes':free,'preflight_gate':read(R/'MAIN_RESOURCE_GATE.json'),'preflight_peak_includes_batch32_stress_test':True,'environment':{'Python':sys.version,'Torch':torch.__version__,'CUDA':torch.version.cuda,'NumPy':np.__version__},'no_historical_artifact_deletion':True};write(R/'RESOURCE_AND_STORAGE_REPORT.json',resources)
 ext=read(R/'EXTENSION_DECISION.json') if (R/'EXTENSION_DECISION.json').exists() else {'passes':False,'not_reached':u<10000,'reason':done['stop_reason'],'assessment':'No extension: the frozen earlier-stop rule applies before any extension assessment.'};write(R/'FINAL_SELECTION.json',{'selected_new_updates':selected,'selected_checkpoint':str(selected_cp),'selected_sha256':selected_sha,'best_research_update':best['new_update'],'decisions':decisions,'extension':ext})
 # Compact prose report with exact answer table.
 table=['| New updates | Native TRAIN PPL | Native VAL PPL | Free TRAIN PPL | Free VAL PPL | Free gap | Local VAL NLL | MLM VAL NLL |','|---:|---:|---:|---:|---:|---:|---:|---:|']
 for z in hist:table.append(f"| {z['new_update']} | {z['NATIVE_TRAIN']['perplexity']:.6f} | {z['NATIVE_VALIDATION']['perplexity']:.6f} | {z['FREE_TRAIN']['perplexity']:.6f} | {z['FREE_VALIDATION']['perplexity']:.6f} | {z['free_gap']:.6f} | {z['NATIVE_VALIDATION']['geometry_NLL']:.6f} | {z['MASKED_VALIDATION']['identity_mean_NLL']:.6f} |")
 last=hist[-1];mlgain=base['MASKED_VALIDATION']['identity_mean_NLL']-last['MASKED_VALIDATION']['identity_mean_NLL'];bestgain=base['FREE_VALIDATION']['AA_NLL']-best['FREE_VALIDATION']['AA_NLL'];decstr='\n'.join(f"- +{v}: new {'PASS' if z['new_sequence_first']['passes'] else ', '.join(z['new_sequence_first']['failures'])}; original {'PASS' if z['original_policy']['passes'] else ', '.join(z['original_policy']['failures'])}." for v,z in decisions.items())
 report=f'''# Final J1 bidirectional multitask experiment\n\nCompleted {u:,} new full-model optimizer updates from accepted 35,500-update parent. Stop reason: {done['stop_reason']}. Six preflight calls were discarded; the accepted parent was restored exactly before baseline/main updates. No teacher, distillation, TEST or RL.\n\n{chr(10).join(table)}\n\nProduction parameters 12,351,746; training-only auxiliary 7,412. Bidirectional attention within original prefixes and both production heads preserved. Full-sequence reconstruction is a separate task with a MASK vector and residue readout; no geometry supplied. No causal conversion or feature/density change.\n\n## Main scientific answers\n\n1. Reconstruction VALIDATION NLL changed by {mlgain:+.6f} improvement; masked accuracy {base['MASKED_VALIDATION']['masked_accuracy']:.6f} → {last['MASKED_VALIDATION']['masked_accuracy']:.6f}. See separate masked metrics; these are not autoregressive perplexities.\n2. Distinct masked identity-position targets: {unique_mask:,}.\n3. Original-prefix unique positions {unique_next:,}; cross-task overlap {intersection:,}; union {coverage['unique_union_identity_positions']:,}. Newly supervised unused-pool masked positions are {coverage['per_population']['unused']['masked_unique_positions']:,}, versus 48,000 positions in the older prefix-only pilot. Actual identity coverage and length/category counts are in TRAINING_TOKEN_AND_IDENTITY_COVERAGE.json.\n4. Geometry-free next-AA change is measured above. Masked reconstruction, paired CE, KL, exposure and schedule changed together; no causal attribution to consistency alone is established.\n5. Best free VALIDATION PPL {best['FREE_VALIDATION']['perplexity']:.6f}, versus parent 17.925712 and previous research 17.303353; gain versus parent {bestgain:.6f}nats. Paired identity intervals versus previous research appear in selection decisions.\n6. Native VALIDATION trajectory appears in the table, scored on the unchanged 2947×8 panel.\n7. Free NLL gap changed {base['free_gap']:.6f} → {last['free_gap']:.6f}; native gap {base['native_gap']:.6f} → {last['native_gap']:.6f}. Matched panels/weights preserved; stochastic batch loss is separate.\n8. Shared encoder parameter-change L2 {delta['encoder.']['L2']:.6f}; AA and geometry branches also changed. Gradient profiles and per-group update logs prove learning beyond the auxiliary readout.\n9. New local allowance is fixed at {cfg['selection_new']['local_NLL_max']:.9f}; original local threshold remains−0.113708968. Separate policy outcomes below.\n10. Complete generation uses 32 attempts per length, corrected FP64 sampler, same saved noise; completion, clashes, Rg, contacts, chirality proxies and diversity appear in FREE_GENERATION_COMPARISON.json. No failures replaced.\n11. Actual main budget used {u:,}; learning curve and frozen progress/early-stop rules determine benefit, rather than optimizer count alone.\n12. Conditional extension assessment: {json.dumps(ext)}.\n13–14. Selected checkpoint `{selected_cp}`, SHA256 `{selected_sha}`. Research-best and latest verified full states are in CHECKPOINT_MANIFEST.json. All include production/auxiliary model, AdamW, scheduler, independent samplers, RNG, counters and task/config hashes. The deployment generator requires only the unchanged production architecture/model state.\n15. Further supervision is justified only by ongoing held-out next-AA gains with geometry/coverage safeguards. The final curve and extension verdict provide the evidence; no automatic further training or RL is authorized.\n\n## Qualification\n\n{decstr}\n\nThe new sequence-first allowance is prospective. Historical rejected checkpoints retain their historical status. A lower MLM loss, lower TRAIN loss or smaller Rg alone is insufficient for promotion.\n\nMain compute {compute/3600:.3f}hours; mean update {compute/u:.4f}s; supervised next-AA targets {64*u:,}; masked target passes {ms['counts']['masked_targets']:,}; separate complete geometry decisions {32*u:,}; nonlocal targets {done['nonlocal_targets']:,}; no long differentiated rollouts. D: free {free/2**30:.3f}GiB, with 10-GiB reserve protected. Stress-profile batch32 peaks are distinct from actual main peaks.\n\nCheckpoint/optimizer recovery passed bitwise on the actual GPU including new samplers, readout and MASK embedding. Source model SHA256 remains {m.SOURCE_SHA}. Training source commit {manifests[-1]['source_commit']}.\n\nSplit safeguards do not certify universal remote-homology independence. Identity/sample intervals are not family-independent. Cα proxy metrics cannot certify folding, experimental function or thermodynamic stability.\n'''
 (R/'FINAL_J1_BIDIRECTIONAL_MULTITASK_REPORT.md').write_text(report)
 gt=['| Updates | Distance NLL | Angle NLL | Torsion NLL | Local NLL |','|---:|---:|---:|---:|---:|']
 for z in hist:gt.append('| '+str(z['new_update'])+' | '+' | '.join(f'{x:.6f}' for x in z['NATIVE_VALIDATION']['component_NLL']+[z['NATIVE_VALIDATION']['geometry_NLL']])+' |')
 (R/'GEOMETRY_RETENTION_REPORT.md').write_text('# Geometry retention\n\n'+ '\n'.join(gt)+'\n\n'+decstr+'\n\nValid component denominators and torsion calibration/tails are preserved in calibration/VALIDATION_*.json and SEQUENCE_TRAIN_VALIDATION_METRICS.json. Continuous densities and Jacobians are unchanged; geometry likelihood can be negative. Geometry remains a constraint, not proof of free-running fold quality.\n')
 with (R/'SEQUENCE_GEOMETRY_CONSISTENCY_REPORT.md').open('a') as f:f.write(f"\nMeasured main mean consistency KL {np.mean([v['consistency_KL'] for v in curves]):.6f}; fixed coefficient {cfg['lambda_consistency']}; paired target exposures {done['paired_target_exposures']}. Next-AA scores are separate from reconstruction. No isolated consistency ablation was trained.\n")
 with (R/'EXPERIMENT_DECISIONS.md').open('a') as f:f.write(f'\nFinal: updates{u}, selected{selected_cp}, SHA256{selected_sha}; researchbest+{best["new_update"]}; stop{done["stop_reason"]}. Original/new-policy assessments preserved separately.\n')
 with (R/'CHANGELOG.md').open('a') as f:f.write('\nFinal: validated independent task/sampler recovery, full-position coverage and two-policy scientific comparisons; all workers stopped without RL execution.\n')
 print(json.dumps({'updates':u,'selected_checkpoint':str(selected_cp),'selected_sha256':selected_sha,'best_free_VAL_PPL':best['FREE_VALIDATION']['perplexity'],'masked_unique_positions':unique_mask,'main_compute_hours':compute/3600,'D_free_GiB':free/2**30}),flush=True)
if __name__=='__main__':main()
