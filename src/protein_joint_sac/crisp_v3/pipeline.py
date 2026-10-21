"""Reconstructable episode collection and lazy replay learner batching."""
import copy,random,uuid
import numpy as np
import torch
from protein_joint_sac.contracts import BuildState,AA
from protein_joint_sac.models.batched_bc import collate
from protein_joint_sac.models.geometry_inference import corrected_action
from .networks import prepare
from .environment import World,goal_tokens,assess,reconstruct
from .storage import digest

def observation(prior,world,device,subgoal=0,planner_features=None):
 state=world.state;s=BuildState(state.residues,state.ca.cpu(),state.N,state.valid.cpu());p=collate([s],device);g,v=goal_tokens(world.task,device)
 with torch.no_grad():ref=prior(p)
 # Public intervention statistics, no hidden native suffix. Up to64 descriptors.
 c=torch.zeros(1,64,device=device);c[0,0]=min(len(world.interventions),100)/100;c[0,1]=world.pending;c[0,2]=state.t/state.N;c[0,3]=min(world.context_branch,100)/100
 for i,item in enumerate(world.interventions[-4:]):
  c[0,4+4*i:8+4*i]=torch.tensor([item['kind']=='REWIND',item.get('from_t',state.t)/state.N,item.get('to',state.t)/state.N,item.get('endpoint_displacement_A',0.)/(3.8*state.N)],device=device)
 if planner_features is not None:
  values=torch.as_tensor(planner_features,device=device).flatten()
  if len(values)>44:raise ValueError('planner feature overflow')
  c[0,20:20+len(values)]=values
 return prepare(p,ref,g,v,c,torch.tensor([subgoal],device=device))

def native_goal_batch(pop,goal_sampler,prior,device):
 task=goal_sampler.draw();h=task['source_identity'];o=task['source_observation'];x=pop.coordinates(pop.base.rows[o]);from protein_joint_sac.data.scaled_supervised import canonical,internal
 x=canonical(x);g=internal(x);seq=pop.base.seqs[h];candidates=sorted(set([p['index'] for p in task['targets']]+[int(i) for i in task['motifs']])-{0,1,2});t=goal_sampler.rng.choice(candidates) if candidates and goal_sampler.prefix_cursor%2==0 else goal_sampler.rng.randrange(3,len(seq));goal_sampler.prefix_cursor+=1;s=BuildState(tuple(seq[:t]),torch.tensor(x[:t]),len(seq));world=World(s,task,{});b=observation(prior,world,device);return b,torch.tensor([AA.index(seq[t])],device=device),torch.tensor(g[t:t+1],device=device),torch.ones(1,3,device=device,dtype=torch.bool)

@torch.no_grad()
def collect(system,prior,task,limits,replay,policy,seed=1,max_events=4096,option_steps=4):
 device=next(system.parameters()).device;initial=dict(sequence=task['initial_sequence'],coordinates=task['initial_CA'],valid=[True]*3,N=task['N'],seed=seed,operational_events=max_events);world=World(BuildState(tuple(initial['sequence']),torch.tensor(initial['coordinates'],dtype=torch.float64),initial['N']),task,limits,seed,operational_events=max_events)
 episode=digest(dict(initial=initial,task=task,policy=policy,seed=seed));replay.episode(episode,initial,task,policy,'POLICY_GENERATED');replay.policy_slot(policy);last=-1;low=[];high=[]
 def step(w,subgoal):
  nonlocal last
  b=observation(prior,w,device,subgoal);l,mu,ls=system.models['actor'](b);prob=l.softmax(-1)[0];u=torch.rand((),generator=w.rng);a=int(torch.searchsorted(prob.cpu().cumsum(-1),u).clamp_max(19));mask=(True,True,True);eps=torch.randn(3,dtype=torch.float64,generator=w.rng);g,info=corrected_action(mu[0,a].cpu(),ls[0,a].cpu(),mask,eps);lp=float(l.log_softmax(-1)[0,a]);before=last;w.step(a,g,lp,info['represented_action_log_density']);w.events[-1]['rng_draws']={'uniform':1,'normal3':1};last,_=replay.append(episode,w.events[-1]);done=w.state.t==w.state.N
  reward=w.events[-1]['reward']+(float(assess(w.state,w.task,limits)['success']) if done else 0.)
  low.append(dict(id=f'{episode}:L:{last:08d}',episode=episode,state_event=before,next_event=last,action=dict(level='low',aa=a,geometry=g.tolist(),mask=list(mask),subgoal=subgoal),goal=digest(task),behavior_logp=lp+info['represented_action_log_density'],density=info,policy=policy,reward=reward,duration=1,terminated=done,truncated=w.truncated,status='RESOLVED',source='POLICY_GENERATED'))
 while not(world.terminated or world.truncated):
  b=observation(prior,world,device);m=world.action_mask()[None].to(device);logits=system.models['high_actor'](b,m);lp=logits.log_softmax(-1);u=torch.rand(());a=int(torch.searchsorted(lp.exp()[0].cpu().cumsum(-1),u).clamp_max(8));start=last;event_start=len(world.events);out=world.option(a,step,option_steps)
  # Residue actions persist their random draws. Managerial decisions use a distinct global RNG, saved in full checkpoints.
  if a>=4:
   for e in world.events[event_start:]:last,_=replay.append(episode,e)
  high.append(dict(id=f'{episode}:H:{start:08d}:{last:08d}',episode=episode,state_event=start,next_event=last,action=dict(level='high',option=a),goal=digest(task),behavior_logp=float(lp[0,a]),policy=policy,reward=out['reward'],duration=out['duration'],terminated=out['terminated'],truncated=out['truncated'],status='RESOLVED',source='POLICY_GENERATED'))
 for r in low+high:replay.transition(r)
 outcome=dict(terminated=world.terminated,censored=world.truncated,success=bool(world.terminated and assess(world.state,task,limits)['success']),event_count=len(world.events));replay.finish_episode(episode,outcome)
 return dict(episode=episode,low_transitions=len(low),high_transitions=len(high),success=bool(world.terminated and assess(world.state,task,limits)['success']),terminated=world.terminated,censored=world.truncated,policy=policy,event_count=len(world.events))

def replay_example(prior,replay,record,limits,device):
 ep,events=replay.read(record['episode']);initial=ep['initial'];seed=ep['initial'].get('seed',1)
 worlds=[reconstruct(initial,events[:i+1],ep['goal'],limits,seed) for i in [record['state_event'],record['next_event']]]
 level=record['action']['level'];sg=record['action'].get('subgoal',0);b,nb=[observation(prior,w,device,sg if level=='low' else 0) for w in worlds]
 return b,nb,worlds
