"""Final frozen geometry/diversity safeguard application to retained J1 checkpoints."""
import json,math,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import j1_global as j
from j1_global_finalize import metric_summary,LENGTHS,read
from scaled_train import Engine as Generator
from protein_joint_sac.rl.checkpointing import load_checked
from protein_joint_sac.data.certified_unified import file_sha
from partial_finalize import metrics
j.core.R=j.R

def run():
 j.core.setup();done=read(j.R/'runs/J1/COMPLETED.json');guard=read(j.R/'configs/FINAL_GENERATION_GUARDRAILS_FROZEN.json')
 baseval=read(j.R/'BASELINE_VALIDATION.json');parent=read(j.OLD/'runs/J1-FULL/SELECTED_GENERATION.json')
 for row in parent:row['metrics']=row['independent_CA_metrics']
 bsum=metric_summary(parent);native=read(j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json')['native']['TRAIN'];options=[]
 for cp in (j.R/'runs/J1').glob('checkpoint_*.pt'):
  u=int(cp.stem.split('_')[1]);evaluation=j.R/'runs/J1'/f'evaluation_{u:05}.json'
  if not evaluation.exists():continue
  v=read(evaluation)['VALIDATION']
  if v['AA_NLL']<baseval['AA_NLL'] and v['geometry_NLL']<=baseval['geometry_NLL']+guard['local_geometry_NLL_max_delta_from_parent']:options.append((v['AA_NLL'],u,cp))
 options.sort();checked=[];engine=None;chosen=None;selected_items=None
 for score,u,cp in options:
  sha=file_sha(cp)
  if sha!=cp.with_name(cp.name+'.sha256').read_text().strip():raise ValueError('candidate checksum')
  cache=j.R/f'FINAL_CANDIDATE_{u:05}.json';items=read(cache) if cache.exists() else []
  keys={(v['N'],v['sample_index']) for v in items}
  if len(keys)!=len(items) or not keys<={(N,k) for N in LENGTHS for k in range(32)} or any(v['checkpoint_sha256']!=sha for v in items):raise ValueError('candidate generation association')
  if engine is None:
   model,state=j.parent_model();del state;engine=SimpleNamespace(model=model,c={**j.CFG,'name':'J1'},joint=True)
  state=load_checked(cp);engine.model.load_state_dict(state['model'],strict=True);del state
  for N in LENGTHS:
   for k in range(32):
    if (N,k) in keys:continue
    v=Generator.generate(engine,N,2026101700+100*N+k);v.update(sample_index=k,checkpoint_sha256=sha)
    if v['status']=='COMPLETE':v['metrics']=metrics(v['generated_CA_A'])
    items.append(v);j.write(cache,items);j.write(j.R/'ASSESSMENT_HEARTBEAT.json',{'candidate_update':u,'N':N,'sample':k,'time':time.time()})
  summary=metric_summary(items);reasons=[]
  for N in LENGTHS:
   a=bsum[str(N)];b=summary[str(N)]
   if b['complete_valid']!=32:reasons.append(f'{N}: incomplete generation')
   if b['unique_sequences']<guard['minimum_unique_sequences_per_length']:reasons.append(f'{N}: duplicate diversity')
   if b['complete_valid']<2:continue
   if b['clashes_per100_residues']['mean']>guard['clash_mean_max_relative_to_parent']*a['clashes_per100_residues']['mean']+guard['clash_additive_allowance_per100_residues']:reasons.append(f'{N}: clash deterioration')
   target=np.median([v['metrics']['Rg_over_N_cuberoot'] for v in native[str(N)]['rows']])
   def error(rows):return np.mean([abs(math.log(v['metrics']['Rg_over_N_cuberoot']/target)) for v in rows if v['N']==N and 'metrics' in v])
   if error(items)>guard['absolute_log_Rg_error_max_relative_to_parent']*error(parent)+guard['absolute_log_Rg_error_additive_allowance']:reasons.append(f'{N}: Rg discrepancy deterioration')
   if b['mean_pair_sequence_Hamming_fraction']<guard['minimum_pair_Hamming_fraction_relative_to_parent']*a['mean_pair_sequence_Hamming_fraction']:reasons.append(f'{N}: sequence diversity deterioration')
   if b['mean_pair_Rg_normalized_distance_fingerprint_RMS']<guard['minimum_Rg_normalized_distance_fingerprint_diversity_relative_to_parent']*a['mean_pair_Rg_normalized_distance_fingerprint_RMS']:reasons.append(f'{N}: structural diversity deterioration')
  entry={'update':u,'checkpoint':str(cp),'sha256':sha,'VALIDATION_AA_NLL':score,'passed':not reasons,'reasons':reasons,'summary':summary,'generation_sha256':file_sha(cache)};checked.append(entry);j.write(j.R/'FINAL_CANDIDATE_REVIEW.json',checked)
  if not reasons:chosen=(u,cp,sha);selected_items=items;break
 if chosen is None:
  chosen=(0,j.PARENT,j.PARENT_SHA);selected_items=parent
  for v in selected_items:v['prediction_cache_reused']=True
 u,cp,sha=chosen;j.write(j.R/'SELECTED_FREE_GENERATION.json',selected_items)
 receipt={'selected_update':u,'selected_checkpoint':str(cp),'selected_sha256':sha,'provisional_main_selection':{k:done[k] for k in ['selected_update','selected_checkpoint','selected_sha256']},'guardrails_sha256':file_sha(j.R/'configs/FINAL_GENERATION_GUARDRAILS_FROZEN.json'),'checked_candidates':checked,'interpretation':'original parent fallback' if u==0 else 'AA-improving continuation passed frozen final length-specific geometry/diversity safeguards','TEST':'NOT_EVALUATED'}
 j.write(j.R/'FINAL_SELECTION.json',receipt)
 # Always assess the actual last trained policy, even when the parent is selected.
 latest=Path(done['latest_checkpoint']);lsha=done['latest_sha256'];lu=done['additional_updates']
 if file_sha(latest)!=lsha:raise ValueError('latest checkpoint integrity')
 if latest==cp:latest_items=selected_items
 else:
  cache=j.R/f'FINAL_CANDIDATE_{lu:05}.json';latest_items=read(cache) if cache.exists() else []
  keys={(v['N'],v['sample_index']) for v in latest_items}
  if len(keys)!=len(latest_items) or any(v['checkpoint_sha256']!=lsha for v in latest_items):raise ValueError('latest generation association')
  if engine is None:
   model,state=j.parent_model();del state;engine=SimpleNamespace(model=model,c={**j.CFG,'name':'J1'},joint=True)
  state=load_checked(latest);engine.model.load_state_dict(state['model'],strict=True);del state
  for N in LENGTHS:
   for k in range(32):
    if (N,k) in keys:continue
    v=Generator.generate(engine,N,2026101700+100*N+k);v.update(sample_index=k,checkpoint_sha256=lsha)
    if v['status']=='COMPLETE':v['metrics']=metrics(v['generated_CA_A'])
    latest_items.append(v);j.write(cache,latest_items);j.write(j.R/'ASSESSMENT_HEARTBEAT.json',{'latest_update':lu,'N':N,'sample':k,'time':time.time()})
 j.write(j.R/'LATEST_FREE_GENERATION.json',latest_items)
 print('FINAL_J1_SELECTION '+str(u)+' LATEST_ASSESSED '+str(lu),flush=True)
if __name__=='__main__':run()
