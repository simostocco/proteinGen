import collections,copy,math,runpy,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v94r3_one_step_feasibility_20261025_v1')
V=R.parent/'e017_v93_regional_controllability_20261024_v1'
B=runpy.run_path(str(V/'source/scripts/e017_v93/study.py'))
globals().update({k:B[k] for k in ['c','p','T','AA','SPEC','read','sha','ns','world','append','load_engine','reconstruct','prefix_status']})
corrected_action=B['corrected_action']
F=runpy.run_path(str(R/'source/scripts/e017_v94r3/shield.py'));extend_checked=F['extend_checked']
def guard():
 assert __import__('shutil').disk_usage(R).free>2*2**30+2*224*2**20+128*2**20
 assert not (R/'control/STOP').exists()
 if torch.cuda.is_initialized():assert torch.cuda.max_memory_allocated()<6*2**30
START=time.perf_counter();COUNTS=collections.Counter()
@torch.no_grad()
def generate(e,task,policy='C',count=8,state=None,seeds=None):
 guard();assert time.perf_counter()-START<2400;assert COUNTS['simulated_transitions']<130000;goal=task['goal'];seq=task['fixed_sequence'];ws=[world(goal,state) for i in range(count)];seeds=seeds or [ns('CANDIDATE',goal['goal_id'],i) for i in range(count)];records=[dict(sequence='',geometry=[],proposals=[],seed=seeds[i],candidate_index=i,goal_id=goal['goal_id'],initial_prefix_length=ws[i].state.t,termination=None,AA_decisions=0,geometry_KL=[],repairs=0,packing_prefix=[]) for i in range(count)];live=list(range(count));start=time.perf_counter();model='A_J1' if policy=='A' else 'C_FIXED'
 for t in range(ws[0].state.t,64):
  if not live:break
  logits,mu,ls,ref,_=e.distribution([ws[i] for i in live],model);nextlive=[];a=AA.index(seq[t]);mask=c['mask_for_step'](t)
  for j,i in enumerate(live):
   w=ws[i];rec=records[i];accepted=False
   for k in range(8 if policy=='C' else 1):
    rng=np.random.default_rng(ns('STEP',seeds[i],t,k));eps=torch.tensor(rng.standard_normal(3),device='cuda',dtype=torch.float64);u=mu[j,a].double()+ls[j,a].double().clamp(-5,2).exp()*eps;geo,diag=corrected_action(mu[j,a],ls[j,a],mask,eps);assert diag['latent']==u.cpu().tolist();lp=float(T['base_logprob'](mu[j,a],ls[j,a],u,mask));rp=float(T['base_logprob'](ref['mu'][j,a],ref['ls'][j,a],u,mask));geo=geo.cpu().tolist();after,check=extend_checked(w.state,seq[t],geo);COUNTS['simulated_transitions']+=1;repairs=sum(diag['adjusted_components']);rec['repairs']+=repairs;rec['proposals'].append([t,k,u.cpu().tolist(),geo,lp,rp,check['accepted'],repairs,check['reasons'],check.get('one_step_feasibility',{}).get('status','NOT_CHECKED_IMMEDIATE_INVALID')])
    if policy!='C' or check['accepted']:
     w.state=after if after is not None else append(w.state,seq[t],geo);rec['geometry'].append(geo);accepted=True;packing=F['prefix_budget'](w.state);rec['packing_prefix'].append([t,packing['contacts'],packing['slack_pairs']]);break
   if not accepted:rec['termination']='ABORT_8_INVALID_PROPOSALS'
   else:
    ml=ls[j,a].double().clamp(-5,2);rl=ref['ls'][j,a].double().clamp(-5,2);kl=rl-ml+.5*((ml.exp()/rl.exp()).square()+((mu[j,a].double()-ref['mu'][j,a].double())/rl.exp()).square())-.5;rec['geometry_KL'].append(float((kl*torch.tensor(mask,device='cuda')).sum()/max(1,sum(mask))));nextlive.append(i)
  live=nextlive
 for w,rec in zip(ws,records):
  rec['sequence']=''.join(w.state.residues);rec['metrics']=c['score'](rec['sequence'],w.state.ca.numpy(),goal,SPEC);rec['seconds']=time.perf_counter()-start;rec['trace_logprob']=sum(q[4] for q in rec['proposals']);rec['reference_trace_logprob']=sum(q[5] for q in rec['proposals']);COUNTS['terminal_evaluations']+=1
 return records
