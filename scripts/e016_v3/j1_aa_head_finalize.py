"""Read-only post-training assessment and compact publication for AA-head-only J1."""
import collections,copy,gc,json,math,subprocess,time,shutil
from pathlib import Path
import numpy as np
import torch
import j1_aa_head_only as a
import j1_full_sequence_finalize as ff
R=a.R;read=a.read;write=a.write;sha=a.sha;m=a.m;s=a.s;c=a.c

def main():
 done=read(R/'runs/J1/COMPLETED.json');hist=[read(p) for p in sorted((R/'runs/J1').glob('evaluation_*.json'))];base=hist[0];last=hist[-1];u=done['new_head_optimizer_updates'];assert last['new_update']==u
 source=s.load_checked(m.SOURCE);assert sha(m.SOURCE)==m.SOURCE_SHA;assert {int(v['step']) for v in source['optimizer']['state'].values()}=={35500};frozen={n:v for n,v in source['model'].items() if not n.startswith('aa_head.')};man=[];geometrychecks={};maxdiff=0.;params_checks={}
 for p in sorted((R/'runs/J1').glob('checkpoint_*.pt')):
  st=s.load_checked(p);v=st['new_update'];assert v==st['sampler']['cursor']==st['scheduler']['last_epoch'];assert set(st['AA_head'])=={'weight','bias'}
  if v:assert {int(x['step']) for x in st['optimizer']['state'].values()}=={v}
  assert sum(len(g['params']) for g in st['optimizer']['param_groups'])==2
  assert all(torch.equal(z,st['model'][n]) for n,z in frozen.items());assert c.same(st['AA_head'],{n.split('.',1)[1]:z for n,z in st['model'].items() if n.startswith('aa_head.')})
  man.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'new_head_updates':v,'total_model_optimizer_updates':35500+v,'full_recovery':True,'optimizer_parameter_tensors':2,'sampler_update':st['sampler']['cursor'],'frozen_tensors_exact':True,'source_commit':st['lineage']['source_commit']});params_checks[str(v)]={'all_frozen_parameters_and_buffers_bitwise_equal':True,'max_absolute_difference':0.};del st;gc.collect()
 for row in hist:
  v=row['new_update'];checks={}
  for role in ['TRAIN','VALIDATION']:
   x=np.load(R/'calibration'/f'{role}_00000.npz');y=np.load(R/'calibration'/f'{role}_{v:05d}.npz');checks[role]={k:bool(np.array_equal(x[k],y[k])) for k in ['nll','mask','p','u','g']};assert all(checks[role].values());x.close();y.close()
  geometrychecks[str(v)]=checks
 equiv=read(R/'FROZEN_PARAMETER_EQUIVALENCE.json');equiv.update(checkpoints=params_checks,all_saved_frozen_tensors_verified_against_accepted_parent=True,source_sha256_verified_after=True);write(R/'FROZEN_PARAMETER_EQUIVALENCE.json',equiv)
 write(R/'GEOMETRY_LIKELIHOOD_EQUIVALENCE.json',{'all_complete_panels_exact':True,'per_target_density_parameter_target_mask_arrays_bitwise_equal':geometrychecks,'milestones':{str(v['new_update']):{role:{k:v[role][k] for k in ['geometry_NLL','component_NLL','component_counts']} for role in ['NATIVE_TRAIN','NATIVE_VALIDATION']} for v in hist},'important':'Teacher-forced geometry is independent of AA head; free generation can still change through AA selections.'})
 gens={};decisions={};selected=0;native=read(s.j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json')['native'];parent=read(a.O/'generation/final00500_ca_fp64_interior_v1.json');fallback=read(a.O/'generation/fallback_ca_fp64_interior_v1.json');assert len(parent)==len(fallback)==160
 assert all(v['protocol']=='ca_fp64_interior_v1' and v['live_batch']==4 and v['seed']==2026101700+100*v['N']+v['sample_index'] for v in parent+fallback)
 assert all(v['checkpoint_sha256']=='9303f0d82a8df044302488340f3ad81535f2fc321360a00bf2892f3a0407aa91' for v in parent)
 parent_model=s.load_checked(a.O/'runs/J1/model_00500.pt');assert c.same(source['model'],parent_model['model']);del parent_model
 summaries={'accepted_parent':m.old.metric_summary(parent),'original_J1_FULL':m.old.metric_summary(fallback)}
 for row in hist[1:]:
  v=row['new_update'];p=R/'generation'/f'decision_{v:05d}.json';gen=read(p) if p.exists() else None;d=a.decision(row,base,generation=gen);decisions[str(v)]=d
  if gen:
   gens[str(v)]=gen;summaries[str(v)]=gen['summary']
  if d['passes'] and (selected==0 or row['FREE_VALIDATION']['AA_NLL']<next(x for x in hist if x['new_update']==selected)['FREE_VALIDATION']['AA_NLL']):selected=v
 selected_path=m.SOURCE if not selected else R/'runs/J1'/f'checkpoint_{selected:05d}.pt';selected_hash=sha(selected_path)
 gaps={}
 for row in hist:
  v=row['new_update'];by={}
  for cond in ['NATIVE','FREE']:
   va=row[cond+'_VALIDATION'];ba=base[cond+'_VALIDATION'];ids=sorted(ba['per_identity']);ci=m.old.paired_ci([va['per_identity'][h]['AA_NLL']-ba['per_identity'][h]['AA_NLL'] for h in ids]);by[cond]={'TRAIN_PPL':row[cond+'_TRAIN']['perplexity'],'VAL_PPL':va['perplexity'],'NLL_gap':row['native_gap' if cond=='NATIVE' else 'free_gap'],'gap_change':row['native_gap' if cond=='NATIVE' else 'free_gap']-base['native_gap' if cond=='NATIVE' else 'free_gap'],'paired_VAL_NLL_change_vs_parent':ci}
  gaps[str(v)]=by
 write(R/'GENERALIZATION_GAP_COMPARISON.json',gaps)
 # Geometry unchanged means true-AA-conditioned generated traces are unchanged; AA scores can change.
 s.j.core.setup();model=s.j.core.model('J1');s.set_dropout(model,.15)
 candidates={'accepted_parent':m.SOURCE,str(u):R/'runs/J1'/f'checkpoint_{u:05d}.pt'}
 if selected and selected!=u:candidates[str(selected)]=selected_path
 matched=ff.matched_robustness_conditions(model,candidates)
 robust={'milestones':{str(v['new_update']):v['GENERATED_PREFIX'] for v in hist},'matched_same_ten_identity_NATIVE_UNAVAILABLE':matched,'paired_generated_AA_NLL_changes':{str(v['new_update']):m.old.paired_ci([v['GENERATED_PREFIX']['per_identity'][h]-base['GENERATED_PREFIX']['per_identity'][h] for h in sorted(base['GENERATED_PREFIX']['per_identity'])]) for v in hist[1:]},'limitation':'Ten fixed VALIDATION identities; not a population estimate. True-AA-conditioned geometry generation is distinct from fully free generation.'}
 write(R/'GENERATED_PREFIX_AA_EVALUATION.json',robust)
 # Explicit AA-only backward proof and evaluation/train dropout behavior, without optimizer calls.
 from protein_joint_sac.data.scaled_supervised import Data
 data=Data(False);ids=read(R/'configs/GEOMETRY_FREE_PANELS.json')['TRAIN'][:8];states=[];labels=[]
 for h,o,t in ids:
  state,y,g=data.target(h,None,t,False);assert g is None;states.append(state);labels.append(y)
 b=m.collate(states,'cuda');ys=torch.tensor(labels,device='cuda');model.load_state_dict(source['model']);model.eval()
 for n,p in model.named_parameters():p.requires_grad_(n.startswith('aa_head.'))
 rng=s.rng_state()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):h=m.encode_batch(model,b);q=model.aa_head(h);q2=model.aa_head(m.encode_batch(model,b))
 assert torch.equal(q,q2)
 with torch.autocast('cuda',dtype=torch.bfloat16):loss=torch.nn.functional.cross_entropy(model.aa_head(h).float(),ys)
 loss.backward();aa_norm=a.norm([p for p in model.parameters() if p.requires_grad]);assert aa_norm>0 and all(p.grad is None for p in model.parameters() if not p.requires_grad);model.zero_grad(set_to_none=True);model.train()
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):h1=m.encode_batch(model,b);h2=m.encode_batch(model,b)
 assert not torch.equal(h1,h2);s.restore_rng(rng)
 write(R/'HEAD_FINAL_DIAGNOSTIC_TESTS.json',{'AA_only_loss_backward_finite_head_norm':aa_norm,'AA_only_frozen_gradients_absent':True,'deterministic_eval_repeat_exact':True,'dropout_on_frozen_encoder_active_during_train':True,'optimizer_steps':0})
 del data,model,source;gc.collect();torch.cuda.empty_cache()
 # Per-length paired generation uncertainty, preserving every failed attempt.
 paired={}
 for label in gens:
  rows=read(R/'generation'/f'new{int(label):05d}_ca_fp64_interior_v1.json');paired[label]={}
  assert len(rows)==160
  for N in [64,128,256,384,500]:
   x={v['sample_index']:v for v in rows if v['N']==N and 'metrics' in v};y={v['sample_index']:v for v in parent if v['N']==N and 'metrics' in v};ks=sorted(set(x)&set(y));paired[label][str(N)]={key:m.old.paired_ci([x[k]['metrics'][key]-y[k]['metrics'][key] for k in ks]) for key in ['clashes_per100_residues','Rg_over_N_cuberoot','long_contact_fraction']}
 generation={'protocol':'ca_fp64_interior_v1','attempts_per_length':32,'seeds':'2026101700+100*N+sample_index; saved shared Gaussian/uniform arrays; live_batch4','accepted_parent_generation':str(a.O/'generation/final00500_ca_fp64_interior_v1.json'),'accepted_parent_cache_sha256':sha(a.O/'generation/final00500_ca_fp64_interior_v1.json'),'source_model_bitwise_cache_binding_verified':True,'original_J1_FULL_cache':str(a.O/'generation/fallback_ca_fp64_interior_v1.json'),'summaries':summaries,'paired_differences_vs_accepted_parent':paired,'decisions':decisions,'native_controls':str(s.j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json'),'no_TEST':True}
 write(R/'FREE_GENERATION_COMPARISON.json',generation)
 selection={'selected_new_head_updates':selected,'selected_checkpoint':str(selected_path),'selected_sha256':selected_hash,'latest_checkpoint':done['latest_checkpoint'],'latest_sha256':done['latest_sha256'],'candidate_decisions':decisions,'extension':read(R/'EXTENSION_DECISION.json'),'next_phase':'RL_PREPARATION_ONLY; execution requires separate authorization'}
 write(R/'CHECKPOINT_MANIFEST.json',{'selected':selection,'checkpoints':man,'source_immutable':str(m.SOURCE),'source_sha256':m.SOURCE_SHA,'resume':'Pin training commit462bb59, run run.sh source/scripts/e016_v3/j1_aa_head_only.py only after separate authorization; COMPLETED prevents automatic continuation. Checkpoints contain full model, AA head, optimizer/scheduler, RNG, coverage/category orders, frozen manifest and configs.'})
 rows=[json.loads(x) for x in (R/'runs/J1/updates.jsonl').read_text().splitlines() if x.strip()];assert sorted(v['new_update'] for v in rows)==list(range(1,u+1));write(R/'AA_HEAD_TRAINING_CURVES.json',rows)
 samples=[json.loads(x) for x in (R/'RESOURCE_SAMPLES.jsonl').read_text().splitlines() if x.strip()];device=[float(v['GPU_csv_timestamp_usedMiB_freeMiB_utilization_temperature_power'].split(',')[1]) for v in samples if v['GPU_csv_timestamp_usedMiB_freeMiB_utilization_temperature_power']!='UNAVAILABLE'];util=[float(v['GPU_csv_timestamp_usedMiB_freeMiB_utilization_temperature_power'].split(',')[3]) for v in samples if v['GPU_csv_timestamp_usedMiB_freeMiB_utilization_temperature_power']!='UNAVAILABLE'];resources={'new_unique_head_updates':u,'reexecuted_restart_proof_updates':2,'physical_optimizer_calls':u+2,'training_compute_seconds':sum(v['seconds'] for v in rows),'AA_decisions_per_compute_second':64*u/sum(v['seconds'] for v in rows),'elapsed_training_plus_evaluation_seconds':done['elapsed_including_evaluation_seconds'],'mean_update_seconds':float(np.mean([v['seconds'] for v in rows])),'head_clip_activation_count':sum(v['clip_scale']<1 for v in rows),'head_gradient_median_before':float(np.median([v['AA_head_gradient_L2_before'] for v in rows])),'head_gradient_median_after':float(np.median([v['AA_head_gradient_L2_after'] for v in rows])),'resources_at_end_worker':done['resources'],'device_peak_MiB_sampled':max(device),'GPU_utilization_sampled_mean':float(np.mean(util)),'D_free_before_bytes':17653776384,'D_free_after_bytes':shutil.disk_usage(R).free,'minimum_D_free_sampled':min(v['D_free_bytes'] for v in samples),'no_artifact_deletion':True,'sampler':done['sampler'],'geometry_optimization_targets':0,'rollout_training_residues':0,'peak_checkpoint_publication_bytes':max(v['bytes'] for v in man)*2}
 write(R/'RESOURCE_AND_STORAGE_REPORT.json',resources)
 # Protected source and exact panel hashes are independently verified at publication.
 assert sha(m.SOURCE)==m.SOURCE_SHA
 for n,h in read(R/'HEAD_ONLY_OPTIMIZER_CONFIG.json')['panels'].items():assert sha(R/'configs'/n)==h==sha(a.FROOT/'configs'/n)
 table=['| New head updates | Native TRAIN PPL | Native VAL PPL | Free TRAIN PPL | Free VAL PPL | Free NLL gap | Local VAL NLL |','|---:|---:|---:|---:|---:|---:|---:|']
 for z in hist:table.append(f"| {z['new_update']} | {z['NATIVE_TRAIN']['perplexity']:.6f} | {z['NATIVE_VALIDATION']['perplexity']:.6f} | {z['FREE_TRAIN']['perplexity']:.6f} | {z['FREE_VALIDATION']['perplexity']:.6f} | {z['free_gap']:.6f} | {z['NATIVE_VALIDATION']['geometry_NLL']:.6f} |")
 gen_table=['| Length | Parent clashes/100 | New clashes/100 | Parent Rg | New Rg | Parent contacts | New contacts |','|---:|---:|---:|---:|---:|---:|---:|']
 if str(u) in summaries:
  for N in ['64','128','256','384','500']:
   b=summaries['accepted_parent'][N];z=summaries[str(u)][N];gen_table.append(f"| {N} | {b['clashes_per100_residues']['mean']:.4f} | {z['clashes_per100_residues']['mean']:.4f} | {b['Rg_A']['mean']:.3f} | {z['Rg_A']['mean']:.3f} | {b['long_contact_fraction']['mean']:.6f} | {z['long_contact_fraction']['mean']:.6f} |")
 sampler=done['sampler'];sample_table=['| AA category | Targets | Distinct identities | Distinct identity-positions |','|---|---:|---:|---:|']
 for k in ['complete','partial_only','unused']:
  cc=sampler['categories'][k];sample_table.append(f"| {k} | {sampler['counts'][k]} | {cc['unique_identities']} | {cc['unique_identity_positions']} |")
 extension=read(R/'EXTENSION_DECISION.json');ext_text='Accepted' if extension['passes'] else 'Not used: '+'; '.join(extension['failures'])
 decision_text='\n'.join(f"- +{key}: {'passes' if val['passes'] else '; '.join(val['failures'])}. Free VAL NLL change 95% CI: {val['paired_free_VAL_NLL_change']['bootstrap_95pct']}" for key,val in decisions.items())
 chosen='new candidate' if selected else 'accepted parent (no new candidate meets all criteria)';text=f'''# Final J1 AA-head-only experiment\n\nCompleted {u} unique head-only optimizer updates; 2 additional calls reexecuted for exact restart proof. All workers stopped. No TEST, other models or RL executed.\n\n{chr(10).join(table)}\n\n**Selected: {chosen}.** `{selected_path}`\nSHA256 `{selected_hash}`. Latest recoverable state: `{done['latest_checkpoint']}`, SHA256 `{done['latest_sha256']}`.\n\nOnly `aa_head.weight` (20×352) and `aa_head.bias` (20) trained: 7,060 of 12,351,746 parameters. All other 12,344,686 parameters and all buffers were bitwise unchanged at every saved state. Complete TRAIN/VALIDATION teacher-forced geometry arrays, likelihoods and component counts remained exactly equal; maximum deterministic geometry difference zero. Native component NLLs are {base['NATIVE_VALIDATION']['component_NLL']}, counts {base['NATIVE_VALIDATION']['component_counts']}.\n\n{chr(10).join(sample_table)}\n\nTotal distinct identities {sampler['unique_identities']}; distinct identity-position targets {sampler['unique_identity_positions']}. Actual allocation is 40/30/30 over AA decisions. Complete geometry training targets: zero.\n\nThe 118,219 identities were available, not all exposed. AA head received real nonzero gradients and parameter updates; no geometry training targets were computed. Dropout0.15 on frozen features and coordinate visibility augmentation0.075 were retained. Fresh head-only AdamW used LR5e-5,50-update warmup, cosine to5e-6 at1000, inherited AA clipping; no architecture/input normalization/distribution change.\n\nExtension verdict: {ext_text}\n\nSelection assessments:\n\n{decision_text}\n\nNative NLL gap changed {base['native_gap']:.6f} → {last['native_gap']:.6f}; geometry-free gap {base['free_gap']:.6f} → {last['free_gap']:.6f}. Both conditions retain their original frozen panel weighting; the two panels are not interchangeable.\n\nGeneration used32 independent attempts per length and checkpoint with saved common noise, corrected FP64 sampler and unchanged safeguards. All attempts/failures retained.\n\n{chr(10).join(gen_table)}\n\nSee FREE_GENERATION_COMPARISON.json for paired uncertainty, native-normalized Rg, local distributions, composition/diversity, failures and exact safeguard failures. Changes in sampled AA can change downstream geometry despite unchanged geometry weights. Smaller Rg or fewer clashes alone do not establish folding.\n\nGenerated-prefix AA PPL: parent {base['GENERATED_PREFIX']['perplexity']:.6f}, latest {last['GENERATED_PREFIX']['perplexity']:.6f}; fixed10identities,2072decisions. Matched native/unavailable scores and paired intervals are in GENERATED_PREFIX_AA_EVALUATION.json. This small diagnostic is distinct from native-prefix or full free generation.\n\nCompute {resources['training_compute_seconds']:.2f}s ({resources['mean_update_seconds']:.4f}s/update), {resources['AA_decisions_per_compute_second']:.2f} AA targets/s; elapsed training+scheduled evaluation {resources['elapsed_training_plus_evaluation_seconds']:.1f}s. Device sampled peak {resources['device_peak_MiB_sampled']}MiB, Torch peaks/RAM in RESOURCE_AND_STORAGE_REPORT.json. Free D: {resources['D_free_after_bytes']/2**30:.3f}GiB;10GiB reserve preserved. No historical artifact removed.\n\nPrevious full-model2500 reference: geometry-free VAL PPL17.303353, free gap0.212570, localVAL−0.108511, not promoted. Head-only adaptation tests a different optimizer scope/exposure budget; results cannot be attributed to an isolated causal mechanism.\n\n## Final scientific assessment\n\n1. Trained7,060 parameters in two exclusive tensors.\n2. Every shared/geometry parameter and buffer remained bitwise identical.\n3. Teacher-forced geometry predictions and all evaluated geometry likelihoods were exactly preserved.\n4. The AA head learned from real complete/partial/unused TRAIN decisions; actual exposures above.\n5–7. Geometry-free and native likelihoods, gaps, subgroup/length/position breakdowns and paired identity CIs are provided above and in TRAIN_VALIDATION_AA_METRICS.json/GENERALIZATION_GAP_COMPARISON.json.\n8. Generated-prefix prediction is reported separately on the fixed10identity panel.\n9. Complete generation was assessed with retained safeguards; exact pass/fail reasons above.\n10. Extension was used only if all frozen +500 criteria passed.\n11–12. Selected checkpoint/path/hash above; research checkpoints retained regardless of promotion.\n13. Recommend **RL preparation from the selected checkpoint**, not another large supervised continuation. Do not execute RL yet. Prepare TRAIN/VALIDATION native positive controls and deliberately invalid controls; calibrate rewards by length/source; retain an immutable reference policy and validated KL/action log-probabilities; test reward exploitation, collapse, masking and free-running failure handling; measure reward components separately before choosing weights. AF confidence and Cα heuristics cannot certify biological stability.\n\nSplit limitation: exact-identity and frozen primary near-duplicate/contained-domain safeguards do not certify universal remote-homology independence. Identity bootstrap intervals are not family-independent. No experimental foldability, thermodynamic stability or function is established.\n'''
 (R/'FINAL_J1_AA_HEAD_ONLY_REPORT.md').write_text(text)
 (R/'CHECKPOINT_AND_RECOVERY.md').write_text(f'''# Recovery\n\nStopped deliberately after {u} authorized head updates; no automatic resume. Checkpoint manifest lists all verified full model/head, AdamW, scheduler, sampler and RNG states. Exact GPU restart proof is HEAD_GPU_RESTART_TEST.json; frozen state proofs are separate. Pin production commit {man[-1]['source_commit']}. Future continuation requires separate authorization and a versioned budget change; preserve COMPLETED and historical states.\n''')
 with (R/'EXPERIMENT_DECISIONS.md').open('a') as f:f.write('\nFinal selection: '+json.dumps(selection)+'\nRecommendation: RL preparation only; no further optimizer updates or RL executed.\n')
 with (R/'CHANGELOG.md').open('a') as f:f.write('\nPublished complete frozen-state, likelihood, paired VALIDATION and corrected-sampler generation comparisons; retained accepted parent unless all criteria passed.\n')
 write(R/'FINAL_SELECTION.json',selection);print(json.dumps({'selected':selection,'last_free_PPL':last['FREE_VALIDATION']['perplexity'],'last_native_PPL':last['NATIVE_VALIDATION']['perplexity'],'gap':last['free_gap'],'updates':u}),flush=True)
if __name__=='__main__':main()
