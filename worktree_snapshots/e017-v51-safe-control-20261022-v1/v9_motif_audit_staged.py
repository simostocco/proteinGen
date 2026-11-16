"""CPU-only ordered-goal observability audit; no updates or new rollouts."""
import copy,inspect,json,runpy,sys,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v9_property_steering_20261023_v1');p=runpy.run_path(str(R/'source/scripts/e017_v9/policy.py'));c=p['c'];read=c['read'];save=c['save'];AA=c['AA']
def audit():
 torch.set_num_threads(1);spec=read(R/'REWARD_SPECIFICATION.json');source=read(R/'results/TRAIN_B_V3_000.json');goal=source['goal'];rev=copy.deepcopy(goal);rev['motif']=goal['motif'][::-1];assert rev['motif']!=goal['motif'];prior=p['FrozenPrior']('cpu');latest=read(R/'checkpoints/JOINT/LATEST.json');assert c['sha'](latest['path'])==latest['sha256'];state=torch.load(latest['path'],map_location='cpu',weights_only=False);actor=p['LowActor'](p['PROFILES']['research']['actor']).eval().requires_grad_(False);actor.load_state_dict(state['actor'],strict=True);del state
 original=inspect.getsource(p['public_batch']);adapted=original.replace("device='cuda'","device=prefix['ids'].device");namespace=dict(p);exec(adapted,namespace);batchfn=namespace['public_batch'];row=source['records'][0];tests=[]
 for length in [0,goal['exposed'][0]+1,goal['exposed'][0]+2]:
  worlds=[]
  for g in [goal,rev]:
   world=c['PropertyWorld'](g,spec,row['seed'])
   for i in range(length):
    aa=row['sequence'][i]
    if length==goal['exposed'][0]+2 and i in [goal['exposed'][0],goal['exposed'][0]+1]:aa=goal['motif'][i-goal['exposed'][0]]
    world.append(aa,row['geometry'][i])
   worlds.append(world)
  outputs=[];batches=[]
  for world in worlds:
   prefix=p['collate']([world.state],'cpu')
   with torch.no_grad():ref=prior(prefix);b=batchfn([world],prefix,ref);out=actor(b)
   outputs.append(out);batches.append(b)
  differences=[float((x-y).abs().max()) for x,y in zip(*outputs)];ctxdelta=float((batches[0]['context']-batches[1]['context']).abs().max());perm_equal=torch.equal(batches[0]['goals'][:,[1,0]],batches[1]['goals']);tests.append(dict(prefix_length=length,goal=goal['motif'],reversed_goal=rev['motif'],both_token_RoPE_positions=batches[0]['goals'][0,:,32].tolist(),goals_equal_after_permutation=perm_equal,context_max_difference=ctxdelta,outputs_max_abs_difference=dict(zip(['AA_logits','geometry_mu','geometry_logstd'],differences)),motif_feedback=[float(b['context'][0,15]) for b in batches],interpretation='pre-order-feedback' if length<=goal['exposed'][0]+1 else 'controlled-prefix feedback may reveal order after two region residues'))
 assert all(v['goals_equal_after_permutation'] and v['context_max_difference']==0 and max(v['outputs_max_abs_difference'].values())<2e-5 for v in tests[:2]);prior.verify()
 result=dict(status='CONFIRMED_EARLY_ORDER_INFORMATION_OMISSION',scientific_stop=True,actor_updated=False,new_trajectories_generated=0,GPU_used=False,checkpoint=latest,tests=tests,architecture_proof='Goal tokens have shared linear embedding, identical RoPE position and identical validity masks; attention is permutation-equivariant and query output permutation-invariant. Before motif feedback, AB/BA goals give identical conditional action laws up to FP rounding.',important_limit='The complete task is NOT proven unlearnable: current motif-utility feedback may reveal order after two exposed-region residues. This is an unintended early goal-information omission, not proof that all goal information is absent or that RL cannot learn.',remediation_not_executed='Encode ordered motif roles/relative positions losslessly without changing architecture or task semantics; require forward and gradient goal-sensitivity tests before any new study. No automatic correction/retraining.',J1_tensor_fingerprint=prior.identity,source_SHA256=c['sha'](R/'source/scripts/e017_v9/policy.py'))
 save('ORDERED_MOTIF_OBSERVABILITY_AUDIT.json',result);print(json.dumps(result))
if __name__=='__main__':audit()
