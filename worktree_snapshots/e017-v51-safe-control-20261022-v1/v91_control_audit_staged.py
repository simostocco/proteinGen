"""Frozen CPU diagnosis of the failed engineering learner; no updates."""
import copy,json,runpy
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v91_ordered_goal_recovery_20261023_v1');p=runpy.run_path(str(R/'source/scripts/e017_v91/policy.py'));c=p['c'];g=p['g'];read=c['read'];save=c['save'];AA=c['AA'];OLD=p['OLD']
def run():
 torch.set_num_threads(1);test=read(R/'CONTROLLABILITY_TEST.json');spec=read(R/'REWARD_SPECIFICATION.json');source=read(Path(test['prefix_source']));raw=source['goal'];row=source['records'][test['source_candidate_index']];a=test['first_target_position'];fixtures=[];labels=[]
 for motif in test['motifs']:
  goal=copy.deepcopy(raw);goal['motif']=motif
  for stage in [0,1]:
   w=c['PropertyWorld'](goal,spec,1)
   for t in range(a+stage):w.append(motif[0] if t==a else row['sequence'][t],row['geometry'][t])
   fixtures.append(w);labels.append(AA.index(motif[stage]))
 prior=p['legacy']['FrozenPrior']('cpu');prefix=p['collate']([w.state for w in fixtures],'cpu')
 with torch.no_grad():ref=prior(prefix)
 batch=g['public_batch'](fixtures,prefix,ref);parent=p['legacy']['LowActor'](p['legacy']['PROFILES']['research']['actor']).eval();state=torch.load(c['PARENT'],map_location='cpu',weights_only=False);parent.load_state_dict({k[len('models.actor.'):]:v for k,v in state['system'].items() if k.startswith('models.actor.')},strict=True);del state;actor=copy.deepcopy(parent);record=test['engineering_checkpoint'];assert c['sha'](Path(record['path']))==record['sha256'];state=torch.load(record['path'],map_location='cpu',weights_only=False);actor.load_state_dict(state['actor'],strict=True);del state
 with torch.no_grad():before=parent(batch)[0];after=actor(batch)[0]
 details=[]
 for i,label in enumerate(labels):
  logits=ref['logits'][i].double();others=logits.clone();others[label]=-float('inf');gap=float(others.max()-logits[label]);upper=logits-3;upper[label]+=6;maxp=float(upper.softmax(-1)[label]);delta=after[i]-ref['logits'][i];details.append(dict(motif=fixtures[i].goal['motif'],stage=i%2,target=AA[label],reference_top=AA[int(logits.argmax())],reference_target_probability=float(logits.softmax(-1)[label]),reference_top_minus_target_logit=gap,AA_residual_support=[-3,3],target_can_be_greedy_under_any_actor_weights=gap<6,maximum_possible_target_probability=maxp,parent_target_probability=float(before[i].double().softmax(-1)[label]),learned_target_probability=float(after[i].double().softmax(-1)[label]),learned_argmax=AA[int(after[i].argmax())],fraction_residuals_abs_above2_99=float((delta.abs()>2.99).float().mean())))
 labels=torch.tensor(labels);before_nll=float(torch.nn.functional.cross_entropy(before,labels));after_nll=float(torch.nn.functional.cross_entropy(after,labels));prior.verify();save('CONTROLLABILITY_FAILURE_AUDIT.json',dict(fixtures=details,greedy_unreachable_count=sum(not v['target_can_be_greedy_under_any_actor_weights'] for v in details),NLL_before=before_nll,NLL_after=after_nll,minimum_possible_NLL_under_residual_bound=float(np.mean([-np.log(v['maximum_possible_target_probability']) for v in details])),optimizer_updates=0,GPU_used=False,interpretation='A ref top-minus-target gap>=6 makes the prescribed greedy target impossible under the unchanged +/-3 residual logit support. Probability improvement and greedy success must be distinguished.'));print(json.dumps(read(R/'CONTROLLABILITY_FAILURE_AUDIT.json')))
if __name__=='__main__':run()
