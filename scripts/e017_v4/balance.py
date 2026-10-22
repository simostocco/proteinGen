import pathlib,sys,math,time,json
import torch
ROOT=pathlib.Path('/mnt/d/Simone/proteinGen_data/e017_v4_multistep_20261022_v1');sys.path.insert(0,str(ROOT/'source/scripts/e017_v4'));import experiment as v
v.seed(v.CONFIG['seed']);e=v.Engine();rows=[]
for task in v.dataset('TRAIN')[:4]:
 loss,terms,x,z=e.rollout(task,8104,'C');parts=dict(position=z['goal_parts']['position'],orientation=.5*z['goal_parts']['orientation']);norms={}
 for k,t in parts.items():
  gs=torch.autograd.grad(t,[p for p in e.actor.parameters() if p.requires_grad],retain_graph=True,allow_unused=True);norms[k]=math.sqrt(sum(float(g.float().square().sum()) for g in gs if g is not None))
 rows.append(dict(identity=task['identity'],weighted_gradient_norms=norms,loss_values={k:float(t.detach()) for k,t in parts.items()}))
e.prior.verify();v.save('results/POSITION_ORIENTATION_GRADIENT_SCALES.json',dict(rows=rows,scope='four predetermined TRAIN tasks before validation',weights_unchanged=True));print(rows,flush=True)
