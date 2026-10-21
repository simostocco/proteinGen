"""Final CPU contract checks for late-bound interfaces. No optimizer workload."""
import ast,json,random,time,copy
from pathlib import Path
import torch
from engineering import R,V2,write,fixture
from protein_joint_sac.crisp_v3.networks import System,prepare
from protein_joint_sac.crisp_v3.causal import subgoal_probabilities
from protein_joint_sac.crisp_v3.storage import Replay,digest
from protein_joint_sac.crisp_v3.environment import World,goal_tokens
from protein_joint_sac.models.batched_bc import collate
start=time.monotonic();s=System('development');s.train();assert not s.targets.training and all(not m.training for m in s.targets.values());state,task=fixture();p=collate([state],'cpu');g,v=goal_tokens(task,'cpu');reference=dict(h=torch.zeros(1,352),logits=torch.zeros(1,20),mu=torch.zeros(1,20,3),ls=torch.zeros(1,20,3));b=prepare(p,reference,g,v);probs=subgoal_probabilities(s.models['reach'],b,torch.tensor([0]));assert probs.shape==(1,4) and torch.isfinite(probs).all();assert ((0<probs)&(probs<1)).all()
# Independent component shapes are configurable and still instantiate all seven models.
from dataclasses import asdict
from protein_joint_sac.crisp_v3.networks import PROFILES
c={k:asdict(x) for k,x in PROFILES['development'].items()};c['actor']['layers']=3;custom=System('development',c);assert custom.counts()['trainable']['actor']>s.counts()['trainable']['actor']
r=Replay(R/'data'/f'binding_checks_{time.time_ns()}.sqlite');initial=dict(sequence='ACD',coordinates=[[0,0,0],[3.8,0,0],[4.8,3.6,0]],valid=[True]*3,N=64);r.episode('one',initial,task,'p','POLICY_GENERATED');r.append('one',dict(kind='STOP'));record=dict(id='t',episode='one',state_event=-1,next_event=0,action=dict(level='high',option=8),goal=digest(task),behavior_logp=-1.,policy='p',reward=0.,duration=1,terminated=True,truncated=False,status='RESOLVED',source='POLICY_GENERATED');r.transition(record);assert r.sample(1,random.Random(1),level='high')[0]==record
bad=copy.deepcopy(record);bad['id']='bad';bad['goal']='wrong'
try:r.transition(bad);raise AssertionError('bad goal admitted')
except ValueError:pass
bad=copy.deepcopy(record);bad['id']='pending';bad['status']='PENDING'
try:r.transition(bad);raise AssertionError('pending admitted')
except ValueError:pass
for path in (R/'source/src/protein_joint_sac/crisp_v3').glob('*.py'):ast.parse(path.read_text())
r.close();write('FINAL_BINDING_CHECKS.json',dict(status='PASS',custom_dimensions=True,target_eval_permanent=True,policy_indexed_subgoal_probabilities=True,strict_replay_goal_binding=True,unresolved_replay_rejected=True,source_AST=True,optimizer_steps=0,seconds=time.monotonic()-start));print('binding checks PASS',flush=True)
