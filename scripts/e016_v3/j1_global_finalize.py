"""Final selected-J1 assessment, using native TRAIN/VALIDATION references only."""
import collections,json,math,subprocess,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import j1_global as j
from scaled_train import Engine as Generator
from protein_joint_sac.rl.checkpointing import load_checked
from protein_joint_sac.data.certified_unified import file_sha
from partial_finalize import metrics
j.core.R=j.R
LENGTHS=[64,128,256,384,500]
def read(p):return json.load(open(p))
def paired_ci(differences,seed=742):
 a=np.asarray(differences,float);rng=np.random.default_rng(seed)
 means=[a[rng.integers(len(a),size=len(a))].mean() for _ in range(2000)]
 return {'n':len(a),'mean':float(a.mean()),'bootstrap_95pct':np.quantile(means,[.025,.975]).tolist(),'limitation':'identity/sample resampling; not universal family independence'}
def metric_summary(items):
 out={}
 for N in LENGTHS:
  rows=[v for v in items if v['N']==N];ok=[v for v in rows if v.get('status')=='COMPLETE' and 'metrics' in v]
  mm=[v['metrics'] for v in ok];seq=[v['sequence'] for v in ok]
  result={'requested':len(rows),'complete_valid':len(ok),'failures':len(rows)-len(ok),'unique_sequences':len(set(seq))}
  for key in ['Rg_A','Rg_over_N_cuberoot','clashes_per100_residues','clashes_per10000_nonlocal_pairs','long_contact_fraction','degenerate_frames','torsion_resultant']:
   v=np.array([m[key] for m in mm]);result[key]={'mean':float(v.mean()) if len(v) else None,'median':float(np.median(v)) if len(v) else None,'q05_q95':np.quantile(v,[.05,.95]).tolist() if len(v) else None}
  if mm:
   result['adjacent_A_mean']=float(np.mean([v['adjacent_A']['mean'] for v in mm]));result['virtual_angle_rad_mean']=float(np.mean([v['angle_rad']['mean'] for v in mm]))
   chars=collections.Counter(''.join(seq));result['AA_frequencies']={a:chars[a]/max(1,sum(chars.values())) for a in j.AA}
   result['mean_pair_sequence_Hamming_fraction']=(float(np.mean([sum(a!=b for a,b in zip(seq[i],seq[k]))/N for i in range(len(seq)) for k in range(i)])) if len(seq)>1 else None)
   idx=np.rint(np.linspace(0,N-1,20)).astype(int);fingerprints=[]
   for v in ok:
    x=np.array(v['generated_CA_A'])[idx];fingerprints.append(np.linalg.norm(x[:,None]-x[None,:],axis=-1)[np.triu_indices(20,1)])
   result['mean_pair_distance_fingerprint_RMS_A']=(float(np.mean([np.sqrt(np.mean((fingerprints[i]-fingerprints[k])**2)) for i in range(len(ok)) for k in range(i)])) if len(ok)>1 else None)
   normalized=[f/v['metrics']['Rg_A'] for f,v in zip(fingerprints,ok)]
   result['mean_pair_Rg_normalized_distance_fingerprint_RMS']=(float(np.mean([np.sqrt(np.mean((normalized[i]-normalized[k])**2)) for i in range(len(ok)) for k in range(i)])) if len(ok)>1 else None)
   result['generation_seconds']=sum(v['seconds'] for v in rows)
  out[str(N)]=result
 return out
def assess():
 j.core.setup();j.guard();done=read(j.R/'runs/J1/COMPLETED.json');final_selection=read(j.R/'FINAL_SELECTION.json');done.update({k:final_selection[k] for k in ['selected_update','selected_checkpoint','selected_sha256']});checkpoint=Path(done['selected_checkpoint']);sha=done['selected_sha256']
 if file_sha(checkpoint)!=sha or file_sha(j.PARENT)!=j.PARENT_SHA:raise ValueError('selected/parent integrity')
 baseline=read(j.R/'SEQUENCE_BASELINES.json')['J1']
 all_metrics={};curves={}
 for p in sorted((j.R/'runs/J1').glob('updates_from_*.jsonl')):
  for line in p.read_text().splitlines():
   if not line.strip():continue
   v=json.loads(line);u=v['additional_update']
   if u in curves and curves[u]['sample_order_sha256']!=v['sample_order_sha256']:raise ValueError('repeated update sample ordering differs')
   curves[u]=v
 j.write(j.R/'J1_TRAINING_CURVES.json',[curves[u] for u in sorted(curves)])
 for p in sorted((j.R/'runs/J1').glob('evaluation_*.json')):all_metrics[str(int(p.stem.split('_')[1]))]=read(p)
 selected=baseline if done['selected_update']==0 else all_metrics[str(done['selected_update'])]
 last=all_metrics[str(done['additional_updates'])]
 selected_val=selected['VALIDATION'];parent_val=baseline['VALIDATION']
 ids=sorted(parent_val['per_identity'])
 paired=paired_ci([selected_val['per_identity'][h]['AA_NLL']-parent_val['per_identity'][h]['AA_NLL'] for h in ids])
 j.write(j.R/'LOCAL_GEOMETRY_VALIDATION.json',{'baseline':baseline,'milestones':all_metrics,'selected_update':done['selected_update'],'selected_paired_identity_AA_NLL_change':paired,'TEST':'NOT_EVALUATED'})
 # Cache and reuse original parent outputs under the identical checkpoint/seed/protocol.
 parent=read(j.OLD/'runs/J1-FULL/SELECTED_GENERATION.json')
 baseline_generation=[]
 for v in parent:
  row=dict(v);row['metrics']=row['independent_CA_metrics'];baseline_generation.append(row)
 cache=j.R/'SELECTED_FREE_GENERATION.json';items=read(cache) if cache.exists() else []
 expected={(N,k) for N in LENGTHS for k in range(32)};keys={(v['N'],v['sample_index']) for v in items}
 if len(keys)!=len(items) or not keys<=expected or any(v['checkpoint_sha256']!=sha for v in items):raise ValueError('generation cache identity')
 engine=None
 for N in LENGTHS:
  for k in range(32):
   if (N,k) in keys:continue
   seed=2026101700+100*N+k
   if sha==j.PARENT_SHA:
    v=next(dict(x) for x in baseline_generation if x['N']==N and x['sample_index']==k);v['prediction_cache_reused']=True
   else:
    if engine is None:
     model,state=j.parent_model();del state
     state=load_checked(checkpoint);model.load_state_dict(state['model'],strict=True);del state
     engine=SimpleNamespace(model=model,c={**j.CFG,'name':'J1'},joint=True)
    v=Generator.generate(engine,N,seed);v.update(sample_index=k,checkpoint_sha256=sha)
    if v['status']=='COMPLETE':v['metrics']=metrics(v['generated_CA_A'])
   items.append(v);j.write(cache,items);j.write(j.R/'ASSESSMENT_HEARTBEAT.json',{'N':N,'sample':k,'time':time.time(),'checkpoint_sha256':sha})
 native=read(j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json')['native'];base_summary=metric_summary(baseline_generation);new_summary=metric_summary(items);comparisons={}
 for N in LENGTHS:
  old=sorted([v for v in baseline_generation if v['N']==N],key=lambda v:v['sample_index']);new=sorted([v for v in items if v['N']==N],key=lambda v:v['sample_index'])
  target=np.median([v['metrics']['Rg_over_N_cuberoot'] for v in native['TRAIN'][str(N)]['rows']]);clash=[];rg=[]
  for a,b in zip(old,new):
   if 'metrics' in a and 'metrics' in b:
    clash.append(b['metrics']['clashes_per100_residues']-a['metrics']['clashes_per100_residues'])
    rg.append(abs(math.log(b['metrics']['Rg_over_N_cuberoot']/target))-abs(math.log(a['metrics']['Rg_over_N_cuberoot']/target)))
  comparisons[str(N)]={'paired_clash_change_per100':paired_ci(clash),'paired_absolute_log_Rg_error_change':paired_ci(rg),'native_TRAIN_normalized_Rg_median':float(target)}
 j.write(j.R/'NATIVE_VS_GENERATED_GLOBAL_GEOMETRY.json',{'metric_definitions':'same as prior study; Rg matched by length window, clash<2A/seqsep>=3, long contacts<8A/seqsep>=24; CA proxies only','native_reference_path':str(j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json'),'native_reference_sha256':file_sha(j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json'),'native':native,'baseline':base_summary,'selected':new_summary,'paired_comparison':comparisons,'generation_cache_sha256':file_sha(cache)})
 # Resource totals include failed preflight logs and all retained new artifacts.
 resources=[]
 for line in (j.R/'logs/resources.jsonl').read_text().splitlines():
  if line.strip():
   try:resources.append(json.loads(line))
   except json.JSONDecodeError:pass
 peak_gpu=[];rss=[]
 for v in resources:
  try:peak_gpu.append(float(v['gpu_used_free_MiB_utilization'].split(',')[0]))
  except ValueError:pass
  rss.extend(q['rss_bytes'] for q in v['jobs'])
 report={'main_run':done['resources'],'main_elapsed_seconds_including_milestone_evaluation':done['elapsed_seconds'],'ordinary_update_seconds_sum':sum(v['seconds'] for v in curves.values()),'global_seconds_sum':sum(v['global_seconds'] for v in curves.values()),'device_wide_GPU_sampled_peak_MiB':max(peak_gpu) if peak_gpu else None,'observed_process_RSS_peak_bytes':max(rss) if rss else None,'minimum_observed_D_free_bytes':min(v['D_free_bytes'] for v in resources),'maximum_swap_used_bytes':max(v['memory']['SwapTotal']-v['memory']['SwapFree'] for v in resources),'new_noncache_artifact_bytes':sum(p.stat().st_size for p in j.R.rglob('*') if p.is_file() and 'source' not in p.relative_to(j.R).parts and 'runtime' not in p.relative_to(j.R).parts),'resources_sample_interval_seconds':10,'engineering_preflight_actual_optimizer_updates':34,'architecture_parameters':12351746,'ordinary_decisions':done['ordinary_decisions'],'generated_training_rollout_residues':done['rollout_residues']}
 j.write(j.R/'RESOURCE_REPORT.json',report)
 text='# Free-running J1 assessment\n\n32independent seeds at each of64,128,256,384,500residues. Original parent outputs reused exactly for the original SHA and seed/protocol; selected continuation outputs are separately generated. No native sequence or coordinates condition these complete generations.\n\n'
 for N in LENGTHS:
  a=base_summary[str(N)];b=new_summary[str(N)];text+=f"N={N}: complete{b['complete_valid']}/32; Rg {a['Rg_A']['mean']:.3f}→{b['Rg_A']['mean']:.3f}Angstrom; clashes/100res {a['clashes_per100_residues']['mean']:.4f}→{b['clashes_per100_residues']['mean']:.4f}; long-contact fraction {a['long_contact_fraction']['mean']:.6f}→{b['long_contact_fraction']['mean']:.6f}; unique sequences{b['unique_sequences']}/32.\n\n"
 text+='Paired bootstrap intervals, native TRAIN/VALIDATION reference distributions, local geometry and sequence/distance-fingerprint diversity are in NATIVE_VS_GENERATED_GLOBAL_GEOMETRY.json. Short, true-AA-conditioned TRAIN rollouts are a different task from these complete joint samples. These CA metrics do not establish experimental folding or thermodynamic stability.\n';(j.R/'FREE_GENERATION_ASSESSMENT.md').write_text(text)
 selected_train=selected['TRAIN'];last_train=last['TRAIN'];last_val=last['VALIDATION'];st=read(j.R/'SEQUENCE_BASELINES.json')
 lines=['# J1 global pretraining: measured outcome','',f"Source branch e016-j1-global-pretraining-20261017-v1; training implementation commit2695487. Main J1 additional updates{done['additional_updates']}; total lineage{done['total_lineage_updates']}; parameter count12351746 unchanged. No other model, TEST, AF, SAC or E015 executed.",'',
 f"Selected checkpoint: {checkpoint}; SHA256{sha}. Latest recoverable checkpoint: {done['latest_checkpoint']}; SHA256{done['latest_sha256']}. Stop reason: {done['stop_reason']}.",'',
 '## Required scientific answers','',
 '1. Uniform probabilistic baseline perplexity is exactly20; NLLln20=2.995732273554.',
 f"2. Matched TRAIN identity-first unigram: TRAIN PPL{st['J1']['TRAIN']['probability_baselines']['matched_joint/unigram_identity_first']['perplexity']:.6f}, VALIDATION PPL{st['J1']['VALIDATION']['probability_baselines']['matched_joint/unigram_identity_first']['perplexity']:.6f}. Exact fitted probabilities, full-PDB and residue-weighted alternatives are in SEQUENCE_BASELINES.json.",
 f"3. Matched identity-first Markov: TRAIN PPL{st['J1']['TRAIN']['probability_baselines']['matched_joint/markov_identity_first']['perplexity']:.6f}, VALIDATION PPL{st['J1']['VALIDATION']['probability_baselines']['matched_joint/markov_identity_first']['perplexity']:.6f}; TRAIN-only smoothing0.5/cell including BOS.",
 f"4. Starting J1 TRAIN PPL{baseline['TRAIN']['perplexity']:.6f}; VALIDATION{baseline['VALIDATION']['perplexity']:.6f}; each2947identities/23576fixed decisions.",
 f"5. Starting VALIDATION-minus-TRAIN NLL gap{baseline['VALIDATION']['AA_NLL']-baseline['TRAIN']['AA_NLL']:.6f}: substantial. Contextual TRAIN fitting is stronger than held-out fitting.",
 f"6. Selected TRAIN/VALIDATION PPL{selected_train['perplexity']:.6f}/{selected_val['perplexity']:.6f}; actual last{last_train['perplexity']:.6f}/{last_val['perplexity']:.6f}. Selection is AA-first with declared local/global safeguards, including original parent fallback. Complete history is preserved.",
 f"7. Selected VALIDATION improvement beyond matched identity-first unigram: {st['J1']['VALIDATION']['probability_baselines']['matched_joint/unigram_identity_first']['NLL']-selected_val['AA_NLL']:.6f}nats. Beyond Markov: {st['J1']['VALIDATION']['probability_baselines']['matched_joint/markov_identity_first']['NLL']-selected_val['AA_NLL']:.6f}nats.",
 '8. Rg gradients through reparameterized actions, generated CA, subsequent encoders and geometry head/Transformer are nonzero and verified; actual GPU profiles through horizon128, numerical parity and exact optimizer recovery are in tests/.',
 '9. Compactness outcome is length-specific in FREE_GENERATION_ASSESSMENT.md and paired bootstrap comparisons; reduced radius alone is not success.',
 '10. Clash and long-contact outcomes are reported together by length, per100residues and pereligiblepair; the CA penalty has no all-atom validity claim.',
 f"11. Selected local geometry NLL per decision{selected_val['geometry_NLL']:.6f}, parent{baseline['VALIDATION']['geometry_NLL']:.6f}; per-component likelihoods and target counts remain separately available.",
 f"12. AA improvement relative to parent: {baseline['VALIDATION']['AA_NLL']-selected_val['AA_NLL']:.6f}nats; paired identity-bootstrap change95percent interval {paired['bootstrap_95pct']}. Joint sequence/global-geometry benefit must be assessed with all five length-dependent geometry comparisons; no exclusive causal attribution to Rg is possible because sampling also changed.",
 f"13. Main updates{done['additional_updates']}, ordinary sampled decisions{done['ordinary_decisions']}, differentiable generated rollout residues{done['rollout_residues']}; complete/partial exposures{done['branch_counts']}. Two bounded recovery proofs additionally executed34discarded engineering updates.",
 f"14. Selected checkpoint and verified SHA are above; original J1-FULL remains immutable fallback. Selected additional update{done['selected_update']}.",
 'Final length-specific compactness, clash and diversity safeguards and provisional-versus-final checkpoint selection are documented in FINAL_SELECTION.json; failed candidates remain preserved.',
 '15. Limitations: local32-residue true-AA-conditioned auxiliary rollouts do not directly optimize full500-residue free trajectories; known remote homology/context overlap persists under the accepted practical split; identity-bootstrap intervals do not remove family dependence; TEST remained sealed this stage and was previously inspected; no AF or laboratory evidence exists. RL readiness is not established by lower likelihood or CA proxy changes.','',
 f"Main elapsed{done['elapsed_seconds']/3600:.3f}hours including milestone evaluation; peak allocated/reserved bytes{done['resources']['allocated_bytes']}/{done['resources']['reserved_bytes']}. Full sampled RAM/device/storage measurements in RESOURCE_REPORT.json.",'']
 (j.R/'FINAL_J1_GLOBAL_PRETRAINING_REPORT.md').write_text('\n'.join(lines))
 j.write(j.R/'ASSESSMENT_COMPLETE.json',{'time':time.time(),'status':'ASSESSMENT_AND_REPORTS_WRITTEN_FINAL_MANIFEST_PENDING','selected_sha256':sha})
 print('J1_GLOBAL_ASSESSMENT_COMPLETE',flush=True)
if __name__=='__main__':assess()
