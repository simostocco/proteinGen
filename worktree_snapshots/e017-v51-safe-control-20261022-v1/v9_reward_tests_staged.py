"""CPU-only adversarial reward and suffix-recovery correctness tests."""
import copy,json,runpy,sys
from pathlib import Path
import numpy as np
R=Path('/mnt/d/Simone/proteinGen_data/e017_v9_property_steering_20261023_v1');c=runpy.run_path(str(R/'source/scripts/e017_v9/core.py'));read=c['read'];save=c['save']
from protein_joint_sac.data.scaled_supervised import internal

def tests():
 spec=read(R/'REWARD_SPECIFICATION.json');witness=read(R/'manifests/GOAL_WITNESSES_PRIVATE.json');goals=read(R/'manifests/TRAIN_GOALS.json')['goals'];rng=np.random.default_rng(9917);records=[];failures=[];checks=[]
 def check(name,passed,detail=None):
  checks.append(dict(name=name,PASS=bool(passed),detail=detail))
  if not passed:failures.append(name)
 for goal in goals:
  w=witness[goal['goal_id']];seq=w['sequence'];x=np.array(w['coordinates']);native=c['score'](seq,x,goal,spec);check('native_'+goal['goal_id'],native['success']);cases={}
  Q,_=np.linalg.qr(rng.normal(size=(3,3)));Q[:,0]*=np.linalg.det(Q);rot=c['score'](seq,x@Q+np.array([17,-3,29]),goal,spec);check('rigid_'+goal['goal_id'],abs(rot['reward']-native['reward'])<1e-10 and rot['success']==native['success'])
  cases['collapsed']=c['score'](seq,x*.01,goal,spec)
  rod=np.c_[np.arange(64)*3.8,np.zeros((64,2))];cases['straight_rod']=c['score'](seq,rod,goal,spec)
  broken=x.copy();broken[25]+=100;cases['broken_spacing']=c['score'](seq,broken,goal,spec)
  bad=x.copy();bad[25]=np.nan;cases['nonfinite']=c['score'](seq,bad,goal,spec)
  cases['homopolymer']=c['score']('A'*64,x,goal,spec);cases['repeated_motif']=c['score'](goal['motif']*32,x,goal,spec)
  for name,result in cases.items():check('zero_'+name+'_'+goal['goal_id'],result['reward']==0 and not result['success'])
  t=np.arange(64);helix=np.c_[2.3*np.cos(t*np.deg2rad(100)),2.3*np.sin(t*np.deg2rad(100)),t*1.5];cases['locally_valid_extended_helix']=c['score'](seq,helix,goal,spec);check('helix_'+goal['goal_id'],not cases['locally_valid_extended_helix']['success'] and cases['locally_valid_extended_helix']['reward']<=.5)
  absent=next(a for a in c['AA'] if a not in goal['motif']);removed=seq[:goal['exposed'][0]]+absent*8+seq[goal['exposed'][1]:];cases['absent_motif']=c['score'](removed,x,goal,spec);check('absent_motif_'+goal['goal_id'],not cases['absent_motif']['success'])
  for k in range(4):
   shuffled=''.join(rng.permutation(list(seq)));cases['AA_shuffle_'+str(k)]=c['score'](shuffled,x,goal,spec)
  records.append(dict(goal_id=goal['goal_id'],native=native,controls=cases))
 # Typed construction: no future native values are retained in the environment.
 goal=goals[0];w=witness[goal['goal_id']];seq=w['sequence'];x=np.array(w['coordinates']);actions=internal(x);world=c['PropertyWorld'](goal,spec,9917)
 for a,g in zip(seq,actions):world.rng.random();world.rng.standard_normal(3);world.append(a,g.tolist())
 check('builder_reconstruction',np.max(np.abs(world.state.ca.numpy()-x))<1e-8)
 restored=c['PropertyWorld'](goal,spec,1);restored.restore(world.export());check('state_RNG_restore',np.array_equal(restored.state.ca.numpy(),world.state.ca.numpy()) and restored.rng.random()==world.rng.random())
 world.high(6);check('true_suffix_rewind',world.state.t==48 and ''.join(world.state.residues)==seq[:48] and np.max(np.abs(world.state.ca.numpy()-x[:48]))<1e-8);check('pending_regeneration_mask',world.mask()==[False]*7+[True,True]);world.high(7);check('regeneration_branch',world.branch==1 and not world.pending)
 obs=world.observation();altered_future=x.copy();altered_future[48:]+=1000;check('observation_prefix_only',len(obs['coordinates'])==48 and len(obs['sequence'])==48 and obs['remaining']==16 and obs['estimated_reachability'] is None)
 for a,g in zip(seq[48:],actions[48:]):world.append(a,g.tolist())
 check('recovered_suffix_accept',world.high(8)['success']);check('stopped_mask',not any(world.mask()))
 # Optional compatibility must show TRAIN-only permutation sensitivity.
 compat={};refs=read(R/'manifests/TRAIN_REFERENCE.json')['records'][:256]
 if spec['compatibility_enabled']:
  gains=[];model=np.array(spec['compatibility']['probabilities']);marg=np.array(spec['compatibility']['marginal'])
  for r in refs:
   e=np.array(r['exposure'][spec['proxy']]);bins=np.minimum(3,(e*4).astype(int));ids=np.array([c['AA'].index(a) for a in r['sequence']]);native=np.mean(np.log(model[bins,ids])-np.log(marg[ids]));sh=[]
   for k in range(16):
    shuffled=rng.permutation(ids);sh.append(np.mean(np.log(model[bins,shuffled])-np.log(marg[shuffled])))
   gains.append(float(native-np.mean(sh)))
  draw=np.random.default_rng(19917).integers(0,len(gains),(10000,len(gains)));means=np.array(gains)[draw].mean(1);compat=dict(identities=len(gains),mean_native_minus_shuffle=float(np.mean(gains)),CI95=np.quantile(means,[.025,.975]).tolist());check('TRAIN_compatibility_shuffle_sensitivity',compat['CI95'][0]>0,compat)
 save('REWARD_TESTS.json',dict(PASS=not failures,failures=failures,checks=checks,controls=records,compatibility_permutation=compat,actor_updates=0,GPU_used=False,limitations='Random AA assignments satisfying this coarse task remain possible; neither reward nor ASA proxy establishes biological function or foldability.'))
 print(json.dumps(dict(PASS=not failures,failures=failures,checks=len(checks))))
if __name__=='__main__':tests()
