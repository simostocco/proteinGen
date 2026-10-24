"""One prospective confirmation; selected-model primary, last-safe descriptive."""
import copy,json,runpy,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v91_ordered_goal_recovery_20261023_v1');t=runpy.run_path(str(R/'source/scripts/e017_v91/train.py'));p=t['p'];c=t['c'];g=p['g'];read=c['read'];save=c['save'];sha=c['sha']

def safe(z,base):return z['backbone_valid']>=max(base['B_DEV']['backbone_valid'],base['C_DEV']['backbone_valid'])-.05 and z['hard_valid']>=max(base['B_DEV']['hard_valid'],base['C_DEV']['hard_valid'])-.05 and z['AA_KL']<=.5 and z['geometry_KL']<=1

def load_actor(e,u,index):
 record=index['owned'][str(u)];assert sha(Path(record['path']))==record['sha256'];state=torch.load(record['path'],map_location='cpu',weights_only=False);assert c['state_digest'](state)==record['state_digest'];e.actor.load_state_dict(state['actor'],strict=True);e.actor.eval();del state;return record

def controls(e,model_label):
 goals=read(R/'manifests/CONFIRM_GOALS.json')['goals'];spec=read(R/'REWARD_SPECIFICATION.json');out={};start=time.perf_counter()
 for variant in ['GOAL_OFF','ORDER_SWAP']:
  records=[]
  for offset in range(0,len(goals),8):
   chunk=goals[offset:offset+8];worlds=[]
   for raw in chunk:
    goal=copy.deepcopy(raw)
    if variant=='ORDER_SWAP':goal['motif']=goal['motif'][::-1];goal['property_signature']=g['semantic_sha'](goal)
    worlds.append(c['PropertyWorld'](goal,spec,c['ns']('BASELINE',raw['goal_id'],0)))
   rows=p['continue_worlds'](e,worlds,'D_GOAL_OFF' if variant=='GOAL_OFF' else 'D_RL')
   for raw,world,row in zip(chunk,worlds,rows):
    row.update(seed=c['ns']('BASELINE',raw['goal_id'],0),candidate_index=0,requested_goal=world.goal,original_goal=raw,variant=variant,original_motif_utility=c['motif_utility'](row['sequence'],raw),counterfactual_motif_utility=c['motif_utility'](row['sequence'],dict(raw,motif=raw['motif'][::-1])),AA_control_info='GOAL_OFF retains only fixed N horizon via state; ORDER_SWAP keeps original paired seed ID',batching='8 independent goal states per batch, no cross-example information')
    records.append(row)
  name=f'results/CONFIRM_{model_label}_{variant}.json';save(name,dict(model=model_label,variant=variant,records=records,goals=32,K=1,no_replacement=True));save(str(Path(name).with_suffix('.sha.json')),dict(sha256=sha(R/name)));out[variant]=dict(records=len(records),full_success=sum(r['metrics']['success'] for r in records),requested_motif_hits=sum(r['metrics'].get('motif_utility',0)==1 for r in records))
 return dict(results=out,elapsed_seconds=time.perf_counter()-start)

def run():
 t['receipt_guard']();assert read(R/'RECOVERY_ENGINEERING.json')['PASS'];finished=read(R/'PRODUCTION_COMPLETED.json');assert finished['status']=='STOPPED';assert not read(R/'checkpoints/protein/RECOVERY_INDEX.json')['automatic_restart'];c['POLICY'].require(R,64*2**20);assert not (R/'FINAL_EVALUATION_COMPLETED.json').exists();start=time.perf_counter();p['evaluate']('CONFIRM',['A_J1','B_PREV9','C_FIXED']);index=read(R/'checkpoints/protein/RECOVERY_INDEX.json');selected=finished['selected'];base=read(R/'INITIAL_QUALITY_ADMISSION.json');eligible=[v for v in finished['milestones'] if safe(v,base)];last=eligible[-1]['update'] if eligible else None;models=[]
 if selected['update']>0:models.append((selected['update'],'D_SELECTED'))
 if last is not None and last!=selected['update']:models.append((last,'D_AFTER_RL'))
 aliases={}
 if selected['update']==0:aliases['D_SELECTED']='C_FIXED'
 if last is None:aliases['D_AFTER_RL']=None
 elif last==selected['update']:aliases['D_AFTER_RL']='D_SELECTED'
 evaluated={}
 for u,label in models:
  e=p['Engine']();record=load_actor(e,u,index);summary=t['evaluate_model'](e,label,'CONFIRM');e.verify(trained=True);evaluated[label]=dict(update=u,checkpoint=record,summary=summary,calls=dict(e.calls));del e;torch.cuda.empty_cache()
 e=p['Engine']();control_results={'C_FIXED':controls(e,'C_FIXED')};e.verify();del e;torch.cuda.empty_cache()
 if last is not None:
  e=p['Engine']();record=load_actor(e,last,index);control_results['D_AFTER_RL']=controls(e,'D_AFTER_RL');e.verify(trained=True);del e;torch.cuda.empty_cache()
 save('FINAL_EVALUATION_COMPLETED.json',dict(status='COMPLETED',goals=32,independent_identity_denominator=32,candidates_per_goal=8,main_K=[1,2,4,8],primary='D_SELECTED minus C_FIXED; D_SELECTED must also improve B; original frozen criteria unchanged',selected=selected,descriptive_after_RL_update=last,evaluated=evaluated,aliases=aliases,controls=control_results,untrained_A_B_C_source='common immutable pre-V9 weights; B legacy input/C canonical input',no_optimization=True,TEST=False,elapsed_seconds=time.perf_counter()-start,primary_status='analysis pending; descriptive after-RL comparison cannot promote a model that failed frozen selection'))
 print(json.dumps(read(R/'FINAL_EVALUATION_COMPLETED.json')))
if __name__=='__main__':run()
