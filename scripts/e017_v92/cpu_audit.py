"""Streaming historical condition intersections and earliest-prefix failures."""
import collections,copy,itertools,json,runpy,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v92_prefix_feasibility_20261024_v1');m=runpy.run_path(str(R/'source/scripts/e017_v92/common.py'));globals().update({k:m[k] for k in ['OLD','V9','p','c','g','AA','SPEC','LIMITS','read','save','sha','prefix_status','extend_checked','motif_info','known_goal_dead','reconstruct']})
def source_sets():
 out={}
 for label in ['A_J1','B_PREV9','C_FIXED','D_AFTER_RL']:out['CONFIRM_'+label]=sorted((OLD/'results').glob('CONFIRM_'+label+'_[0-9][0-9][0-9].json'))
 out['DEV_D100']=sorted((OLD/'results').glob('DEV_D_U0100_[0-9][0-9][0-9].json'));return out

def audit():
 torch.set_num_threads(1);start=time.perf_counter();summary={};all_records=[];input_hashes={};mismatches=[]
 for label,paths in source_sets().items():
  conditions=collections.Counter();intersections=collections.Counter();patterns=collections.Counter();first=collections.Counter();causes=collections.Counter();at_risk=collections.Counter();stage_counts={str(s):collections.Counter() for s in [16,32,48,64]};compat=[];tasks=set();n=0
  for path in paths:
   expected=read(path.with_suffix('.sha.json'))['sha256'];assert sha(path)==expected;input_hashes[str(path)]=expected;obj=read(path);goal=obj['goal'];tasks.add(goal['goal_id'])
   for index,row in enumerate(obj['records']):
    n+=1;seq=row['sequence'];state=c['BuildState'].empty(goal['N']);earliest=None;stages={};seen_clash=0
    for t,(aa,geom) in enumerate(zip(seq,row['geometry'])):
     before=prefix_status(state);after,check=extend_checked(state,aa,geom)
     # Baseline reconstruction continues after invalidity exactly as the saved
     # sampler did. The diagnostic predicate never changes the builder.
     try:state=m['append'](state,aa,geom)
     except (ValueError,RuntimeError) as error:
      if earliest is None:earliest=dict(residue=t+1,action_index=t,causes=['builder_error'],error=repr(error))
      break
     status=prefix_status(state)
     if earliest is None:
      at_risk[t+1]+=1
      bad=list(status['irreversible_failures'])
      if not check['support_valid']:bad+=check['reasons']
      if bad:earliest=dict(residue=t+1,action_index=t,causes=sorted(set(bad)),new_clash_pairs=status['clash_pairs']-before['clash_pairs']);first[t+1]+=1;causes.update(earliest['causes'])
     if state.t in [16,32,48,64]:
      stages[str(state.t)]=dict(**status,goal_dead_reasons=known_goal_dead(state,goal));stage_counts[str(state.t)]['observed']+=1;stage_counts[str(state.t)]['monotone_valid']+=status['valid'];stage_counts[str(state.t)]['already_invalid']+=not status['valid'];stage_counts[str(state.t)]['goal_already_impossible_from_known_prefix']+=bool(stages[str(state.t)]['goal_dead_reasons'])
    metrics=row['metrics'];mi=motif_info(seq,goal);computed=c['score'](seq,state.ca.numpy(),goal,SPEC)
    if computed!=metrics:mismatches.append(dict(source=str(path),candidate=index,reason='reconstructed score differs'))
    if state.t==64:
     status=prefix_status(state);expected_monotone=not any(q in metrics['geometry']['failures'] for q in ['spacing','nonlocal_clashes','degenerate_frames','incomplete_or_nonfinite']);assert status['valid']==expected_monotone
    f=metrics['geometry']['failures'];flags=dict(complete=metrics['complete'],canonical=not set(seq)-set(AA),motif_identity_global=mi['identity_anywhere'],motif_identity_region=mi['identity_in_region'],motif_order_region=mi['ordered_subsequence_in_region'],motif_contiguous_region=mi['original_contiguous_in_region'],exposed=metrics.get('exposed_mean',-1)>=goal['exposure_min']-1e-10,buried=metrics.get('buried_mean',2)<=goal['burial_max']+1e-10,contrast=metrics.get('contrast',-1)>=goal['contrast_min']-1e-10,local_geometry='spacing' not in f and 'degenerate_frames' not in f and metrics['geometry']['finite'],nonlocal_clash='nonlocal_clashes' not in f,global_Rg='Rg' not in f and metrics['geometry']['finite'],packing=metrics['packing_valid'],sequence_quality=metrics['sequence_valid'],backbone=metrics['backbone_valid'],hard=metrics['hard_valid'],full=metrics['success'])
    assert flags['motif_contiguous_region']==(metrics.get('motif_utility',0)==1);assert flags['full']==all(flags[q] for q in ['complete','canonical','motif_contiguous_region','exposed','buried','contrast','local_geometry','nonlocal_clash','global_Rg','packing','sequence_quality'])
    conditions.update({k:int(v) for k,v in flags.items()});regional=flags['exposed'] and flags['buried'] and flags['contrast'];intersections['motif_and_hard']+=flags['motif_contiguous_region'] and flags['hard'];intersections['motif_and_backbone']+=flags['motif_contiguous_region'] and flags['backbone'];intersections['regional_properties']+=regional;intersections['hard_and_regional']+=flags['hard'] and regional;intersections['motif_and_hard_and_exposed']+=flags['motif_contiguous_region'] and flags['hard'] and flags['exposed'];intersections['motif_and_hard_and_exposed_buried']+=flags['motif_contiguous_region'] and flags['hard'] and flags['exposed'] and flags['buried'];intersections['full']+=flags['full']
    primary=['complete','canonical','motif_contiguous_region','exposed','buried','contrast','local_geometry','nonlocal_clash','global_Rg','packing','sequence_quality'];violations=tuple(k for k in primary if not flags[k]);patterns[' + '.join(violations) if violations else 'NONE']+=1
    if metrics.get('compatibility_score') is not None:compat.append(metrics['compatibility_score'])
    all_records.append(dict(policy_panel=label,source=str(path),source_SHA256=expected,candidate_index=index,goal_id=goal['goal_id'],identity=goal['witness_identity'],seed=row['seed'],conditions=flags,first_irreversible_failure=earliest,stages=stages,compatibility_score=metrics.get('compatibility_score'),compatibility_is_soft_not_a_success_condition=True))
  summary[label]=dict(trajectories=n,goals=len(tasks),marginals=dict(conditions),intersections=dict(intersections),top_violation_patterns=patterns.most_common(12),first_failure_residue_histogram=dict(first),first_failure_causes=dict(causes),hazard={str(t):dict(at_risk=at_risk[t],new_failures=first[t],hazard=first[t]/at_risk[t] if at_risk[t] else None) for t in range(1,65)},stages={s:dict(v) for s,v in stage_counts.items()},compatibility=dict(n=len(compat),mean=float(np.mean(compat)) if compat else None,minimum=float(min(compat)) if compat else None,maximum=float(max(compat)) if compat else None,hard_pass_threshold=None,meaning='validated AA/environment association reward component only; not physical foldability'))
 assert not mismatches,mismatches[:2];checkpoint=read(OLD/'checkpoints/protein/RECOVERY_INDEX.json')['owned']['100'];assert sha(Path(checkpoint['path']))==checkpoint['sha256'];assert sha(c['PARENT'])==c['PSHA'] and sha(c['J1'])==c['J1SHA'];save('HISTORICAL_CONSTRAINT_DECOMPOSITION.json',dict(panels=summary,records=all_records,input_SHA256=input_hashes,score_replay_exact=True,thresholds=LIMITS,packing_limits=SPEC['packing_limits']['64'],prefix_filter_semantics='local spacing/nondegenerate frames/finite support and final-N clash budget only; current Rg/contact density never used as absorbing flags',clash_budget_N64=1,compatibility_has_no_hard_success_threshold=True,stopped_checkpoint100=checkpoint,no_training=True,GPU_used=False,seconds=time.perf_counter()-start));print(json.dumps(dict(panels={k:dict(n=v['trajectories'],hard=v['marginals']['hard'],motif=v['marginals']['motif_contiguous_region'],motif_hard=v['intersections']['motif_and_hard'],hard_regional=v['intersections']['hard_and_regional'],full=v['intersections']['full'],irreversible_failures=sum(v['first_failure_residue_histogram'].values())) for k,v in summary.items()},seconds=time.perf_counter()-start)))
if __name__=='__main__':audit()
