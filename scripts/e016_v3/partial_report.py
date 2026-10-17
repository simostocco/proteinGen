"""Descriptive continuation comparisons; no model selection, fitting, TEST evaluation or new thresholds."""
import collections,json,math
from pathlib import Path
import numpy as np
def scalar_summary(values):
 a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
 return {'n':len(a),'mean':float(a.mean()) if len(a) else None,'median':float(np.median(a)) if len(a) else None,'q05_q95':np.quantile(a,[.05,.95]).tolist() if len(a) else None}
def resource_samples(paths):
 rows=[];gpu=[]
 for p in paths:
  for line in Path(p).read_text().splitlines():
   if not line.strip():continue
   row=json.loads(line);rows.append(row)
   if row.get('GPU'):
    try:gpu.append([float(v.strip()) for v in row['GPU'].split(',')])
    except ValueError:pass
 return {'samples':len(rows),'device_wide_used_MiB_peak':max((v[0] for v in gpu),default=None),'device_wide_free_MiB_min':min((v[1] for v in gpu),default=None),'GPU_utilization_percent_mean':float(np.mean([v[2] for v in gpu])) if gpu else None,'D_free_bytes_min':min((v['D_free_bytes'] for v in rows),default=None),'WSL_available_bytes_min':min((v['memory']['MemAvailable'] for v in rows),default=None),'swap_used_bytes_peak':max((v['memory']['SwapTotal']-v['memory']['SwapFree'] for v in rows),default=None),'scope':'10-second samples; device-wide memory includes desktop/other consumers, not isolated process accounting; brief unsampled peaks remain possible'}
def paired_delta(left,right):
 a=left['per_identity_AA_geometry_mean'];b=right['per_identity_AA_geometry_mean']
 if set(a)!=set(b) or not a:raise ValueError('paired identity panels differ')
 ids=sorted(a);d=np.array([np.asarray(b[h])-a[h] for h in ids]);rng=np.random.default_rng(741);boot=np.array([d[rng.integers(len(d),size=len(d))].mean(0) for _ in range(500)]);boot=np.column_stack([boot,boot.sum(1)]);mean=d.mean(0)
 return {'identities':len(ids),'right_minus_left_AA_geometry_objective':[float(mean[0]),float(mean[1]),float(mean.sum())],'identity_bootstrap_95pct':np.quantile(boot,[.025,.975],axis=0).tolist(),'interpretation':'descriptive paired identity resampling, conditional on one training seed; residual family/source correlations are not removed'}
def sampling_evidence(curves):
 by={arm:{v['update']:v for v in rows} for arm,rows in curves.items()}
 out={'paired_chemistry_schedules':{},'equal_complete_exposure':{}}
 for condition in ['FULL','PARTIAL']:
  left=by['J1-'+condition];right=by['J2-'+condition];common=sorted(set(left)&set(right))
  out['paired_chemistry_schedules'][condition]={'common_logged_updates':len(common),'all_sample_order_hashes_equal':all(left[u]['sample_order_sha256']==right[u]['sample_order_sha256'] for u in common),'all_complete_stream_hashes_equal':all(left[u]['complete_stream_sha256']==right[u]['complete_stream_sha256'] for u in common),'all_partial_stream_hashes_equal':all(left[u]['partial_stream_sha256']==right[u]['partial_stream_sha256'] for u in common)}
 for name in ['J1','J2']:
  full=by[name+'-FULL'];partial=by[name+'-PARTIAL'];out['equal_complete_exposure'][name]=[]
  for u,v in sorted(full.items()):
   if 2*u not in partial:continue
   p=partial[2*u]
   out['equal_complete_exposure'][name].append({'FULL_update':u,'PARTIAL_update':2*u,'complete_decisions':v['branch_decisions']['complete'],'counts_equal':v['branch_decisions']['complete']==p['branch_decisions']['complete'],'complete_stream_hashes_equal':v['complete_stream_sha256']==p['complete_stream_sha256']})
 return out
def selected_update(root,arm,receipt):
 p=Path(receipt['selected_checkpoint'])
 return int(p.stem.split('_')[-1]) if p.parent.resolve()==(root/'runs'/arm).resolve() else 0
def publish(root,done,valid,generated):
 def write(name,x):(root/name).write_text(json.dumps(x,indent=2,allow_nan=False))
 def text(name,x):(root/name).write_text(x)
 selected={a:{'update':selected_update(root,a,r),'checkpoint':r['selected_checkpoint'],'sha256':r['selected_sha256']} for a,r in done.items()}
 for a,r in selected.items():r['validation']=valid[a][str(r['update'])]
 comparisons={}
 for name in ['J1','J2']:
  l=name+'-FULL';r=name+'-PARTIAL';comparisons[name+'_PARTIAL_MINUS_FULL']={pop:paired_delta(selected[l]['validation'][pop],selected[r]['validation'][pop]) for pop in ['complete','partial']}
 for condition in ['FULL','PARTIAL']:
  l='J1-'+condition;r='J2-'+condition;comparisons[condition+'_J2_MINUS_J1']={pop:paired_delta(selected[l]['validation'][pop],selected[r]['validation'][pop]) for pop in ['complete','partial']}
 equal_update={}
 for l,r in [('J1-FULL','J1-PARTIAL'),('J2-FULL','J2-PARTIAL'),('J1-FULL','J2-FULL'),('J1-PARTIAL','J2-PARTIAL')]:
  equal_update[r+'_MINUS_'+l]={u:{pop:{'AA_NLL_delta':valid[r][u][pop]['AA_NLL']-valid[l][u][pop]['AA_NLL'],'geometry_NLL_delta':valid[r][u][pop]['geometry_valid_component_mean_per_decision']-valid[l][u][pop]['geometry_valid_component_mean_per_decision'],'objective_delta':valid[r][u][pop]['selection_objective']-valid[l][u][pop]['selection_objective']} for pop in ['complete','partial']} for u in sorted(set(valid[l])&set(valid[r]),key=int)}
 gen_summary={}
 for arm,items in generated.items():
  gen_summary[arm]={}
  for N in [64,128,256,384,500]:
   attempts=[v for v in items if v['N']==N];complete=[v for v in attempts if v['status']=='COMPLETE'];sequences=[v['sequence'] for v in complete];composition=collections.Counter(''.join(sequences));total=sum(composition.values());p=np.array(list(composition.values()),dtype=float)/max(1,total)
   normalized={aa:n/max(1,total) for aa,n in sorted(composition.items())};geom=[v['independent_CA_metrics'] for v in complete if 'independent_CA_metrics' in v]
   gen_summary[arm][str(N)]={'attempts':len(attempts),'complete':len(complete),'geometry_assessment_failures':sum('assessment_failure' in v for v in complete),'unique_sequences':len(set(sequences)),'duplicate_fraction':1-len(set(sequences))/max(1,len(sequences)) if sequences else None,'AA_composition':normalized,'AA_entropy_bits':float(-(p*np.log2(p)).sum()),'exact_TRAIN_sequence_matches':sum(v.get('exact_TRAIN_sequence_match',False) for v in complete),'seconds':sum(v['seconds'] for v in attempts),'metrics':{key:scalar_summary([g[key] for g in geom]) for key in ['clashes_per100_residues','clashes_per10000_nonlocal_pairs','long_contact_fraction','Rg_A','Rg_over_N_cuberoot','degenerate_frames','torsion_positive_fraction','signed_triple_product_mean']},'adjacent_spacing_mean_A':scalar_summary([g['adjacent_A']['mean'] for g in geom]),'angle_mean_rad':scalar_summary([g['angle_rad']['mean'] for g in geom]),'caveat':'exact overlap only; no new sequence alignments or biological novelty claim'}
 native=json.loads((root/'NATIVE_VS_GENERATED_GEOMETRY.json').read_text())['native'];native_summary={s:{N:{'selected_identities':v['selected_identities'],'available_identities':v['available_identities'],'length_window':v['length_window'],'metrics':{key:scalar_summary([r['metrics'][key] for r in v['rows']]) for key in ['clashes_per100_residues','clashes_per10000_nonlocal_pairs','long_contact_fraction','Rg_A','Rg_over_N_cuberoot','degenerate_frames','torsion_positive_fraction','signed_triple_product_mean']}} for N,v in by.items()} for s,by in native.items()}
 study={'selected':selected,'selected_paired_comparisons':comparisons,'equal_update_comparisons':equal_update,'generation_summary':gen_summary,'native_length_matched_summary':native_summary,'TEST_used':False};curve_path=root/'CONTINUATION_TRAINING_CURVES.json';study['sampling_receipt_checks']=sampling_evidence(json.loads(curve_path.read_text())) if curve_path.exists() else None;write('STUDY_COMPARISON.json',study)
 resource_path=root/'RESOURCE_AND_STORAGE_REPORT.json'
 if resource_path.exists():
  resources=json.loads(resource_path.read_text());resources['sampled_operational_telemetry']=resource_samples(sorted((root/'logs').glob('resource_recovery_*.jsonl')));write(resource_path.name,resources)
 census=json.loads((root/'PARTIAL_PDB_ELIGIBILITY_COUNTS.json').read_text());qa=json.loads((root/'view/PARTIAL_VIEW_QA.json').read_text())
 lines=['# Partial-structure continuation results','','All results are PDB-supervised development results using VALIDATION. TEST was not reevaluated. No AlphaFold, SAC, E015 or editing policy executed.','','| Arm | Additional updates | Selected additional update | AA NLL | Geometry NLL per decision | Common complete objective | New decisions |','|---|---:|---:|---:|---:|---:|---:|']
 for a,r in done.items():
  v=selected[a]['validation']['complete'];lines.append(f"| {a} | {r['additional_updates']} | {selected[a]['update']} | {v['AA_NLL']:.6f} | {v['geometry_valid_component_mean_per_decision']:.6f} | {v['selection_objective']:.6f} | {r['additional_prefix_decisions']} |")
 lines+=['','## Partial-data recovery','',f"TRAIN: {census['splits']['TRAIN']['identities']:,} partial identities, {census['splits']['TRAIN']['additional_to_complete_identities']:,} additional to complete-only, {census['splits']['TRAIN']['observations']:,} partial observations. VALIDATION: {census['splits']['VALIDATION']['identities']:,} partial identities, {census['splits']['VALIDATION']['additional_to_complete_identities']:,} additional, {census['splits']['VALIDATION']['observations']:,} observations.",'',f"TRAIN recovered targets across observations: distance {census['splits']['TRAIN']['distance_targets']:,}, angle {census['splits']['TRAIN']['angle_targets']:,}, torsion {census['splits']['TRAIN']['torsion_targets']:,}; known AA labels at missing CA positions {census['splits']['TRAIN']['AA_at_missing_CA']:,}. Unique identity-position denominators are in view/PARTIAL_VIEW_QA.json.",'','The expanded union has 50,604 TRAIN and 6,125 VALIDATION identities. These are distinct canonical sequences, not proof of evolutionary independence. Source observations remain separate and identity-first weighted.','', '## Paired selected-model comparisons','','Differences below are right minus left, in the order AA / geometry / their fixed selection objective. Negative likelihood differences favor the right-hand model.']
 for name,pops in comparisons.items():
  lines.append(name+': '+json.dumps({pop:{k:v for k,v in x.items() if k in ['right_minus_left_AA_geometry_objective','identity_bootstrap_95pct']} for pop,x in pops.items()}))
 lines+=['','## Checkpoints','']
 for a,r in selected.items():lines.append(a+': '+r['checkpoint']+' SHA256 '+r['sha256'])
 lines+=['','## Interpretation and limits','','Likelihood changes do not establish free-running fold quality. STUDY_COMPARISON.json reports collisions, compactness, contact density, local geometry, signed chirality proxies and diversity separately at every length, against both untouched 10k parents and length-matched experimental TRAIN/VALIDATION traces. Differing last-continuation models are assessed separately with the -LAST label, so a retained original-parent selection cannot conceal the behavior of actually continued weights. These C-alpha measures do not establish atomistic stereochemistry, unique native state, thermodynamic stability or experimental folding.','','FULL/PARTIAL comparisons change both identity coverage and observation availability. They do not isolate missingness augmentation from additional sequence exposure. J1/J2 comparisons share the declared population and schedule but use one training seed. Existing S1/S2 remain frozen historical sequence baselines; they were not expanded or retrained here.','','The accepted split protects the frozen 50% identity/80% bidirectional-coverage criterion and the 70% identity/80% shorter-coverage/50-residue safeguard. It does not eliminate all remote homologues, short shared regions, or experimental-context correlations. Identity bootstrap intervals do not remove those correlations.','','Missing coordinates were never interpolated, borrowed, zero-labeled as physical atoms or supervised. The primary added view requires at least one absent CA. All-CA observations failing earlier full-chain qualification remain outside this view. Missingness is not evidence of experimental disorder; unresolved construct/tag residues and method biases remain.','','Unavailable target coordinates affect only their geometry losses; known canonical AA labels remain valid. Experimental C-N continuity is offline provenance for trustworthy anchors. At generation, continuity masks mean declared construction connectivity, not an atomic C-N measurement inferred from CA. Every online feature uses only the generated prefix.','','Exclusive excluded/unresolved observation counts are reported in view/PARTIAL_VIEW_QA.json. Diagnostic exclusion-reason tallies in the census can overlap and must not be summed as unique exclusions. Observed outliers were retained and flagged; the existing action inverse was unchanged.']
 report='\n'.join(lines)+'\n';text('FINAL_PARTIAL_SUPERVISION_REPORT.md',report);text('FINAL_RESEARCH_ASSESSMENT.md',report)
 vlines=['# Free-running generation comparison','','32 fixed-seed attempts per selected model and length; differing last continuations are labeled -LAST and never change selection. Failures are retained. Identical checkpoint/seed results can be reused. No native coordinates are generation inputs.','', '| Model | N | Complete / attempts | Mean clashes / 100 residues | Mean Rg / N^(1/3) | Mean long-contact fraction | Unique sequences |','|---|---:|---:|---:|---:|---:|---:|']
 for a,by in gen_summary.items():
  for N,v in by.items():
   m=v['metrics'];vlines.append(f"| {a} | {N} | {v['complete']} / {v['attempts']} | {m['clashes_per100_residues']['mean']} | {m['Rg_over_N_cuberoot']['mean']} | {m['long_contact_fraction']['mean']} | {v['unique_sequences']} |")
 vlines+=['','Native reference quantiles and all CA/sequence diagnostics are in STUDY_COMPARISON.json. CA clash cutoff is 2 A at sequence separation >=3; long contacts use 8 A at separation >=24. Rg/N^(1/3) is a descriptive compactness proxy, not a universal target for all native topologies. No foldability or global-fold identity claim follows.'];text('FREE_GENERATION_COMPARISON.md','\n'.join(vlines)+'\n')
 write('SELECTED_VALIDATION_AND_GENERATION_SUMMARY.json',{'comparisons':comparisons,'generation':gen_summary,'native':native_summary})
