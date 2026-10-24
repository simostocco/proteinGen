"""TRAIN witness calibration and prospective sealed VALIDATION goals."""
import runpy,math,copy,collections,json
from pathlib import Path
import numpy as np
R=Path('/mnt/d/Simone/proteinGen_data/e017_v93_regional_controllability_20261024_v1');d=runpy.run_path(str(R/'source/scripts/e017_v93/common.py'));globals().update({k:d[k] for k in ['R','OLD','V91','V9','c','SPEC','read','save','sha','ns','guard']})
def make(row,panel,level='witness',reverse=False):
 E=np.asarray(row['exposure'][SPEC['proxy']]);starts=range(4,53,4);gap,a,b=max((float(E[a:a+8].mean()-E[b:b+8].mean()),a,b) for a in starts for b in starts if abs(a-b)>=16)
 if gap<.15:return None
 motifs=[row['sequence'][i:i+2] for i in range(a,a+7) if row['sequence'][i]!=row['sequence'][i+1]]
 if not motifs:return None
 motif=motifs[ns('motif',row['identity'])%len(motifs)];seq=row['sequence'];synthetic=False
 if reverse:
  i=seq.find(motif,a,a+8);seq=seq[:i]+motif[::-1]+seq[i+2:];motif=motif[::-1];synthetic=True
 slack=.05 if level=='relaxed' else 0.
 goal=dict(goal_id=f'V93_{panel}_{row["identity"][:16]}_{level}'+('_reverse' if reverse else ''),witness_identity=row['identity'],protected_group=row['group'],N=64,exposed=[a,a+8],buried=[b,b+8],exposure_min=max(0.,round(math.floor(E[a:a+8].mean()*20)/20-slack,8)),burial_max=min(1.,round(math.ceil(E[b:b+8].mean()*20)/20+slack,8)),contrast_min=.1,motif=motif,motif_scope='anywhere within exposedregion',native_coordinates_provided=False,difficulty=level,sequence_synthetic_motif_reversal=synthetic)
 metric=c['score'](seq,np.asarray(row['coordinates']),goal,SPEC)
 if not metric['success']:return None
 return dict(goal=goal,fixed_sequence=seq,source_reference_identity=row['identity'],source_reference=str(V9/f'manifests/{row["split"]}_REFERENCE.json'),native_witness_success=True,synthetic_sequence_control=synthetic)
def main():
 guard();assert not (R/'manifests/PROSPECTIVE_JOINT.json').exists();used=set();usedgroups=set()
 for root,panels in [(V9,['TRAIN','DEV','CONFIRM']),(V91,['CONFIRM'])]:
  for panel in panels:
   for row in read(root/f'manifests/{panel}_GOALS.json')['goals']:used.add(row['witness_identity']);usedgroups.add(row['protected_group'])
 for row in read(V9/'SASA_COMPLETED.json')['records']:used.add(row['identity']);usedgroups.add(row['group'])
 selected={};proof={};exclusions={}
 for panel,total in [('TRAIN',16),('VALIDATION',16)]:
  records=read(V9/f'manifests/{panel}_REFERENCE.json')['records'];picked=[];groups=set();exclude=[]
  for row in sorted(records,key=lambda q:ns('panel',q['identity'])):
   if panel=='VALIDATION' and (row['identity'] in used or row['group'] in usedgroups):continue
   if row['group'] in groups:continue
   item=make(row,'TRAIN' if panel=='TRAIN' else 'PROSPECTIVE')
   if item is None:exclude.append(dict(identity=row['identity'],reason='unchanged native hard/regional feasibility gate'));continue
   picked.append(item);groups.add(row['group'])
   if len(picked)==total:break
  assert len(picked)==total,(panel,len(picked));selected[panel]=picked;proof[panel]=[dict(identity=q['identity'],source=q['source'],offset=q['offset']) for q in records if q['identity'] in {x['source_reference_identity'] for x in picked}];exclusions[panel]=exclude
 save('manifests/TRAIN_TASKS.json',dict(tasks=selected['TRAIN'],identities=16,groups=16,source_SHA256=sha(V9/'manifests/TRAIN_REFERENCE.json'),witness_records=proof['TRAIN']))
 val_lookup={q['identity']:q for q in read(V9/'manifests/VALIDATION_REFERENCE.json')['records']};fixed=[];joint=[]
 for task in selected['VALIDATION']:
  row=val_lookup[task['source_reference_identity']]
  for level in ['witness','relaxed']:
   item=make(row,'PROSPECTIVE',level);assert item;fixed.append(item)
   for reverse in [False,True]:
    item=make(row,'PROSPECTIVE',level,reverse)
    if item:joint.append(dict(goal=item['goal'],feasibility_witness_sequence=item['fixed_sequence'],source_reference_identity=item['source_reference_identity'],synthetic_sequence_control=item['synthetic_sequence_control'],native_coordinates_provided=False))
 assert len({z['goal']['motif'] for z in joint})>=8
 save('manifests/PROSPECTIVE_FIXED.json',dict(tasks=fixed,identities=16,groups=16,goals=len(fixed),source_SHA256=sha(V9/'manifests/VALIDATION_REFERENCE.json'),previous_property_policy_evaluation_IDs_excluded=True,aggregate_reference_calibration_previously_exposed=True,exclusions=exclusions['VALIDATION'],no_optimization_in_V93=True))
 save('manifests/PROSPECTIVE_JOINT.json',dict(tasks=joint,identities=16,groups=16,goals=len(joint),purpose='SEALED for future substantial joint policy comparison; no joint generation inV93',sequence_supplied_to_future_joint_controller=False,synthetic_reversal_witness_note='only computational validity is demonstrated; AA changes at native motif occurrence, no biochemical feasibility claim'))
 save('FROZEN_PROTOCOL.json',dict(schema='E017_V93_1',seed=9317,N=64,TRAIN_identities=16,VALIDATION_identities=16,prospective_fixed_goals=32,methods=['A_J1_FIXED','B_ACTOR_FIXED','C_FILTERED_FIXED','D_TERMINAL_SELECT_C_POOL','E_TRAIN_SUFFIX_REWIND_SEARCH'],K=[1,8],TRAIN_C_first4_K=16,validation_levels=['witness','relaxed'],max_attempts_per_step=8,fallback='terminate incomplete candidate; reward0; no replacement',hard_rules='unchanged V9/V92, finalN64 clash budget1, no Rg/contact-average prefix filtering',reward_SHA256=sha(V91/'REWARD_SPECIFICATION.json'),paired_uncertainty='10000 identity-cluster bootstrap seed19917;16 protein identities, not32 independent level-goals',delayed=dict(TRAIN_identities=4,prefix_stages=[8,32,48],valid_actions=3,first_action_attempt_limit=16,FIT_replicates=16,independent_CONFIRM_replicates=16,common_random_numbers=True),search=dict(TRAIN_identities=4,prefix=32,suffix_candidates=15,initial_candidate=1,terminal_budget=16,comparison='C Bestof16 and8; actual simulated transitions reported'),alternative_goals='up to4TRAIN sequences with generated hard-valid witnesses satisfying reversed exposure-region ordering; no unsupported goal swap',max_simulated_transitions=130000,max_GPU_seconds=2400,evidence_budget_bytes=64*2**20,environment_gate='all selected native controls satisfy original hard/regional constraints; negative controls invalidreward0; filtered TRAIN backbone>=.90;>=4of16TRAIN identity-goals with observed full regional support; shaped terminal rewardSD>=.03. Supports a training environment, not policy performance.',implementation_gate='toy normalization/exact gradients/MC checks/learning/recovery and protein frozen trace replay/backward masks all pass; no protein optimizer updates',RL_performance_decision='NOT_TESTED; no substantial protein training authorized',TEST=False,J1_updates=0))
 controls=[]
 train_lookup={q['identity']:q for q in read(V9/'manifests/TRAIN_REFERENCE.json')['records']}
 for item in selected['TRAIN']:
  row=train_lookup[item['source_reference_identity']];goal=item['goal'];seq=item['fixed_sequence'];X=np.asarray(row['coordinates'])
  controls.append(dict(identity=row['identity'],native=c['score'](seq,X,goal,SPEC),collapsed=c['score'](seq,np.zeros((64,3)),goal,SPEC),extended=c['score'](seq,np.column_stack([np.arange(64)*3.8,np.zeros(64),np.zeros(64)]),goal,SPEC)))
 assert all(z['native']['success'] and z['collapsed']['reward']==0 and z['extended']['reward']==0 for z in controls)
 save('NATIVE_AND_NEGATIVE_CONTROLS.json',dict(records=controls,thresholds_unchanged=True));print('SEALED',len(fixed),len(joint),flush=True)
if __name__=='__main__':main()
