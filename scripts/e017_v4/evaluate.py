"""Inference-only classical, generated-prefix and goal-only intervention controls."""
import pathlib,sys,json,time,copy,math
import numpy as np
import torch
ROOT=pathlib.Path('/mnt/d/Simone/proteinGen_data/e017_v4_multistep_20261022_v1');sys.path.insert(0,str(ROOT/'source/scripts/e017_v4'));import experiment as v
from protein_joint_sac.crisp_v3.environment import assess,frame
from protein_joint_sac.contracts import BuildState

def fk(task,u):
 x=torch.tensor(task['native_prefix'],dtype=torch.float64,device=u.device)
 for z in u:
  g=torch.stack([z[0].exp(),math.pi*z[1].sigmoid(),math.pi*z[2].tanh()]);x=torch.cat([x,v.next_ca(x,g)[None]],0)
 return x

def metric(task,x,ns):
 s=BuildState(tuple(task['sequence'][:len(x)]),x,len(x));g=v.task_goal(task);g['N']=len(x);m=assess(s,g,v.LIMITS);er=m['errors'][0];m.update(identity=task['identity'],seed=ns,position_error_A=er['position_error_A'],orientation_error_deg=er['orientation_error_deg'],finite=bool(torch.isfinite(x).all()),target_hit=er['position_error_A']<=8 and er['orientation_error_deg']<=35,improves_initial=er['position_error_A']<float(torch.linalg.vector_norm(x[task['t']-1]-x.new_tensor(task['target']))));return m

def IK(tasks,baseline,path_label):
 records={k:[] for k in [8,64]};start=time.monotonic()
 for row in baseline:
  task=next(t for t in tasks if t['identity']==row['identity']);x=np.concatenate([np.array(task['native_prefix']),np.array(row['coordinate_tail'])]);g=v.internal(x)[task['t']:];u0=torch.tensor(np.stack([np.log(g[:,0]),np.log(g[:,1]/(math.pi-g[:,1])),np.arctanh(g[:,2]/math.pi)],-1),dtype=torch.float64);delta=torch.zeros_like(u0,requires_grad=True);opt=torch.optim.Adam([delta],lr=.05);t0=time.monotonic()
  for k in range(64):
   opt.zero_grad(set_to_none=True);x=fk(task,u0+delta);guide,_=v.Engine.goal_loss(None,x,task);local=v.Engine.local(None,x,task);loss=guide+.1*local+.01*delta.square().mean();loss.backward();torch.nn.utils.clip_grad_norm_([delta],10.,error_if_nonfinite=True);opt.step()
   if k+1 in records:
    with torch.no_grad():x=fk(task,u0+delta);m=metric(task,x,row['seed']);m.update(iterations=k+1,seconds=time.monotonic()-t0,latent_anchor_cost=float(delta.square().mean()),KL=None,prior_note='Classical point-action optimization has no stochastic policy KL; latent displacement regularizer is not KL-to-J1');records[k+1].append(m)
 for k,rr in records.items():v.save(f'results/{path_label}_{k:02d}.json',dict(iterations_per_attempt=k,attempts=len(rr),records=rr,total_wall_seconds=time.monotonic()-start,hardware='CPU; no neural model in IK loop',no_native_intermediate_coordinates_in_objective=True,initialization='Frozen A sampled actions, recovered bijectively from cached trace',budget_note='8iteration primary approximately512optimizer steps over64queries;64iteration upper bound4096steps. Per-instance optimization, not an amortized actor-training budget.'))

def load_actor(e,arm):
 if arm=='A':e.reset();update=0
 else:
  p=ROOT/'checkpoints'/arm/'INDEX.json';i=json.load(open(p));s=torch.load(i['latest']['path'],map_location='cpu',weights_only=False);assert v.sha(i['latest']['path'])==i['latest']['sha256'];e.actor.load_state_dict(s['actor']);update=s['update']
 e.actor.eval();return update

def extras():
 e=v.Engine();tasks=v.dataset('VALIDATION');e.actor.eval();transfer=[];controls=[]
 # Transfer prefixes generated ONCE by frozen A, identical for every evaluated arm.
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  for task in tasks:
   x=torch.tensor(task['native_prefix'][:-4],dtype=torch.float64,device='cuda');gen=torch.Generator().manual_seed(9101+int(task['identity'][:6],16));eps=torch.randn(4,3,dtype=torch.float64,generator=gen).cuda()
   for k in range(4):
    mu,ls,*_=e.predict(task,x);g,_=v.sample(mu,ls,torch.ones(3,device='cuda',dtype=torch.bool),eps[k]);x=torch.cat([x,v.next_ca(x,g)[None]],0)
   new=copy.deepcopy(task);new['native_prefix']=x.cpu().tolist();d=torch.linalg.vector_norm(x[1:]-x[:-1],dim=-1);new['prefix_adjacent_valid']=bool(d.min()>=v.LIMITS['adjacent_low_A'] and d.max()<=v.LIMITS['adjacent_high_A']);new['transfer_note']='Original native endpoint retained; generated-prefix displacement can change feasibility';new['contour_necessary_bound']=float(torch.linalg.vector_norm(x[-1]-x.new_tensor(task['target'])))<=8*v.LIMITS['adjacent_high_A']+8;transfer.append(new)
 # Counterfactual goals are native-action-derived KINEMATIC controls, not new experimental proteins.
 for task in tasks[:8]:
  pair=[]
  for sign in [-1,1]:
   gs=np.array(task['native_actions']);gs[0,2]=(gs[0,2]+sign*.8+math.pi)%(2*math.pi)-math.pi;x=torch.tensor(task['native_prefix'],dtype=torch.float64)
   for g in gs:x=torch.cat([x,v.next_ca(x,torch.tensor(g))[None]],0)
   new=copy.deepcopy(task);new['target']=x[-1].tolist();new['orientation']=frame(x).tolist();new['goal_variant']=sign;w=metric(new,x,0);new['counterfactual_witness_valid']=w['valid'];new['experimental_target']=False;new['note']='Kinematic intervention on first torsion of a verified native-action witness; fixed prefix/AAs; not experimentally verified altered conformation';pair.append(new)
  controls.append(pair)
 v.save('manifests/GENERATED_PREFIX_CONTROLS.json',dict(tasks=transfer,frozen_A_source=str(v.PARENT),seeds='9101+first6hex(identity)',feasibility_note='Necessary contour bound only; IK success separately establishes known solvability'))
 v.save('manifests/GOAL_INTERVENTION_CONTROLS.json',dict(task_pairs=controls,experimental=False,primary_training_tasks_unchanged=True))
 rows=[]
 for arm in ['A','B','C']:
  update=load_actor(e,arm);e.evaluate(transfer,arm+'_TRANSFER',update)
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   for pair in controls:
    vals=[]
    for task in pair:
     x=v.prefix(task);mu,ls,rm,rl,ref,b=e.predict(task,x);m=e.evaluate_one(task,6101);vals.append(dict(endpoint=m['coordinate_tail'][-1],mu=mu.cpu().tolist(),logstd=ls.cpu().tolist(),position_error_A=m['position_error_A'],orientation_error_deg=m['orientation_error_deg'],valid=m['valid'],target=task['target'],witness_valid=task['counterfactual_witness_valid']))
    delta=np.array(vals[1]['endpoint'])-np.array(vals[0]['endpoint']);wanted=np.array(vals[1]['target'])-np.array(vals[0]['target']);direction=float(delta.dot(wanted)/max(wanted.dot(wanted),1e-12));own=float(np.mean([vals[i]['position_error_A'] for i in [0,1]]));cross=float(np.mean([np.linalg.norm(np.array(vals[i]['endpoint'])-np.array(vals[1-i]['target'])) for i in [0,1]]));rows.append(dict(arm=arm,identity=pair[0]['identity'],variants=vals,goal_direction_alignment=direction,own_error_A=own,cross_goal_error_A=cross,appropriate_direction=direction>0,channel_mu_delta=(np.array(vals[1]['mu'])-np.array(vals[0]['mu'])).tolist(),channel_logstd_delta=(np.array(vals[1]['logstd'])-np.array(vals[0]['logstd'])).tolist(),kinematic_control=True))
 e.prior.verify();v.save('results/GOAL_ONLY_INTERVENTIONS.json',rows)
 # Serialized CPU IK after neural work; no concurrent GPU worker.
 base=json.load(open(ROOT/'results/A_00000_VALIDATION.json'))['records'];IK(tasks,base,'D_IK')
 tr=json.load(open(ROOT/'results/A_TRANSFER_00000_VALIDATION.json'))['records'];IK(transfer,tr,'D_TRANSFER_IK')
 print('EVALUATION_CONTROLS_COMPLETE',flush=True)

if __name__=='__main__':extras()
