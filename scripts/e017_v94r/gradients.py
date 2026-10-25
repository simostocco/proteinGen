import collections,copy,math,runpy,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v94r_packing_safety_20261025_v1')
V=R.parent/'e017_v93_regional_controllability_20261024_v1'
B=runpy.run_path(str(V/'source/scripts/e017_v93/study.py'))
globals().update({k:B[k] for k in ['c','p','T','AA','SPEC','read','sha','ns','world','append','load_engine','reconstruct','prefix_status']})
F=runpy.run_path(str(R/'source/scripts/e017_v94r/packing.py'));extend_checked=F['extend_checked']
def guard():
 assert __import__('shutil').disk_usage(R).free>2*2**30+2*224*2**20+128*2**20
 assert not (R/'control/STOP').exists()
 if torch.cuda.is_initialized():assert torch.cuda.max_memory_allocated()<6*2**30
START=time.perf_counter();COUNTS=collections.Counter()

def gradients(e,task,rows,beta):
 rewards=np.array([r['metrics']['reward'] for r in rows]);lp=np.array([r['trace_logprob'] for r in rows]);rp=np.array([r['reference_trace_logprob'] for r in rows]);aug=rewards-beta*(lp-rp);adv=aug-(aug.sum()-aug)/(len(rows)-1);worlds=[world(task['goal']) for r in rows];maxerr=0.;probes=[];loss_value=0.;entropy_sum=0.;entropy_dims=0
 for t in range(64):
  active=[i for i,r in enumerate(rows) if any(q[0]==t for q in r['proposals'])]
  for start in range(0,len(active),16):
   ids=active[start:start+16];ws=[worlds[i] for i in ids];prefix=p['collate']([w.state for w in ws],'cuda')
   with torch.no_grad():ref=e.prior(prefix);batch=p['public_batch'](ws,prefix,ref)
   logits,mu,ls=e.actor(batch);loss=[]
   for j,i in enumerate(ids):
    a=AA.index(task['fixed_sequence'][t]);terms=[]
    for q in rows[i]['proposals']:
     if q[0]==t:
      value=T['base_logprob'](mu[j,a],ls[j,a],torch.tensor(q[2],dtype=torch.float64,device='cuda'),c['mask_for_step'](t));maxerr=max(maxerr,abs(float(value.detach())-q[4]));terms.append(value)
    loss.append((-adv[i]+beta)*torch.stack(terms).sum()/len(rows))
   node_loss=torch.stack(loss).sum();loss_value+=float(node_loss.detach());node_loss.backward()
   with torch.no_grad():
    for j,i in enumerate(ids):
     mask=torch.tensor(c['mask_for_step'](t),device='cuda');a=AA.index(task['fixed_sequence'][t]);entropy_sum+=float(((ls[j,a].double().clamp(-5,2)+.5*math.log(2*math.pi*math.e))*mask).sum());entropy_dims+=int(mask.sum())
   if t in [1,16,32,48,63]:probes.append((copy.deepcopy(ws),mu.detach().clone(),ls.detach().clone(),AA.index(task['fixed_sequence'][t])))
  for i in active:
   if t<len(rows[i]['geometry']):worlds[i].state=append(worlds[i].state,task['fixed_sequence'][t],rows[i]['geometry'][t])
 assert maxerr<5e-5,('on-policy replay mismatch',maxerr)
 e.last_gradient_diagnostics=dict(policy_gradient_surrogate=loss_value,augmented_advantage_variance=float(adv.var()),reward_variance=float(rewards.var()),latent_Gaussian_entropy_per_active_dimension=entropy_sum/max(1,entropy_dims),entropy_is_not_filtered_output_entropy=True,group_size=len(rows),trajectory_logprob_reduction='sum all sampled raw-latent proposal scores, mean trajectories only',AA_score_terms=0)
 return rewards,maxerr,probes
