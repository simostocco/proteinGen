"""V9.1 actor-only RLOO with complete compact rollout evidence and durable replay."""
import copy,functools,hashlib,json,math,os,random,runpy,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v91_ordered_goal_recovery_20261023_v1');p=runpy.run_path(str(R/'source/scripts/e017_v91/policy.py'));c=p['c'];read=c['read'];save=c['save'];sha=c['sha'];digest=c['state_digest'];ns=c['ns'];AA=c['AA'];original_sample=p['sample']
from protein_joint_sac.crisp_v3.recovery import atomic_checkpoint,serialized_size
from protein_joint_sac.crisp_v3.density import log_prob

def receipt_guard():
 test=read(R/'GOAL_OBSERVABILITY_TESTS.json');assert test['PASS'] and all(sha(R/'source/scripts/e017_v91'/n)==h for n,h in test['source_SHA256'].items());assert read(R/'CONTROLLABILITY_ADMISSION.json')['PASS'];protocol=read(R/'FROZEN_STUDY_PROTOCOL.json');assert sha(R/'source/scripts/e017_v91/goals.py')==protocol['goal_schema_SHA256'];assert sha(R/'REWARD_SPECIFICATION.json')==protocol['task_reward_unchanged_SHA256'];return protocol

def setup():
 torch.manual_seed(9917);np.random.seed(9917);random.seed(9917);e=p['Engine']();e.actor.requires_grad_(True);e.actor.train();opt=torch.optim.AdamW(e.actor.parameters(),lr=1e-6,weight_decay=0.)
 for v in e.actor.parameters():opt.state[v]=dict(step=torch.tensor(0.),exp_avg=torch.zeros_like(v),exp_avg_sq=torch.zeros_like(v))
 return e,e.parent,opt,torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)

@functools.lru_cache(None)
def scales():return {read(path)['goal']['goal_id']:max(.05,float(np.std([r['metrics']['reward'] for r in read(path)['records']]))) for path in (R/'results').glob('TRAIN_C_FIXED_[0-9][0-9][0-9].json')}

def cached_sample(e,goal,spec,policy,count,namespace):
 if not (namespace.startswith('RL_') or namespace.startswith('ENGINEERING_')):return original_sample(e,goal,spec,policy,count,namespace)
 root=R/'rollouts';root.mkdir(exist_ok=True);path=root/(namespace+'.json');fingerprint=digest(e.actor.state_dict());binding=dict(behavior_actor_fingerprint=fingerprint,goal_SHA256=sha(R/'manifests/TRAIN_GOALS.json'),reward_SHA256=sha(R/'REWARD_SPECIFICATION.json'),schema_SHA256=sha(R/'source/scripts/e017_v91/goals.py'),goal_id=goal['goal_id'],namespace=namespace,parent_SHA256=c['PSHA'])
 if path.exists():
  meta=read(path.with_suffix('.sha.json'));assert sha(path)==meta['sha256'];saved=read(path);assert saved['binding']==binding,'cached rollout is not on-policy for recovered weights';return saved['records'],[]
 c['POLICY'].require(R,2*2**20);rows,worlds=original_sample(e,goal,spec,policy,count,namespace)
 for row in rows:row['policy']='D_RL_repaired';row['behavior_actor_fingerprint']=fingerprint
 save(str(path.relative_to(R)),dict(binding=binding,goal=goal,records=rows,source='complete generated trajectories before optimizer update, no native prefix',unique_collected_trajectories=len(rows),residue_decisions=sum(len(r['geometry']) for r in rows)));save(str(path.with_suffix('.sha.json').relative_to(R)),dict(sha256=sha(path),committed=True));return rows,worlds
p['sample']=cached_sample

def outputs(e,parent,batch,mode):
 assert mode=='JOINT';return e.actor(batch)
