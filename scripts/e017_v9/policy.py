"""Frozen de novo baselines and inference/replay primitives for V9."""
import collections,copy,json,math,os,runpy,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v9_property_steering_20261023_v1');c=runpy.run_path(str(R/'source/scripts/e017_v9/core.py'));save=c['save'];read=c['read'];sha=c['sha'];digest=c['state_digest'];ns=c['ns'];AA=c['AA']
from protein_joint_sac.crisp_v3.prior import FrozenPrior
from protein_joint_sac.crisp_v3.networks import LowActor,PROFILES,prepare
from protein_joint_sac.models.batched_bc import collate
from protein_joint_sac.models.geometry_inference import corrected_action
from protein_joint_sac.crisp_v3.density import log_prob
def public_batch(worlds,prefix,reference):
 B=len(worlds);goals=torch.zeros(B,2,36,device='cuda');valid=torch.ones(B,2,dtype=torch.bool,device='cuda');ctx=torch.zeros(B,64,device='cuda')
 for i,w in enumerate(worlds):
  g=w.goal;o=w.observation();N=g['N'];a,z=g['exposed'];b,d=g['buried'];ctx[i,:11]=torch.tensor([1,g['exposure_min'],g['burial_max'],g['contrast_min'],a/N,z/N,b/N,d/N,w.mode/3,w.state.t/N,1-w.state.t/N],device='cuda')
  for j,name in enumerate(['exposed','buried']):
   q=o['regions'][name];ctx[i,11+2*j]=q['current_known_mean'] if q['current_known_mean'] is not None else 0;ctx[i,12+2*j]=q['generated']/q['total'];ctx[i,26+j]=q['final_exposure_upper_bound']
  ctx[i,15]=o['motif_current_utility'];s=c['sequence_stats'](o['sequence']);ctx[i,16:20]=torch.tensor([s['entropy_bits']/math.log2(20),s['max_fraction'],s['max_run']/N,s['unique']/20],device='cuda');ctx[i,20]=o['clash_count']/max(1,N*.02);ctx[i,21]=o['irreversible_clash_violation'];ctx[i,22]=len(w.interventions)/4;ctx[i,28+w.mode]=1
  for j,aa in enumerate(g['motif']):goals[i,j,12+AA.index(aa)]=1;goals[i,j,32]=(a+z)/2/N;goals[i,j,35]=1
 return prepare(prefix,reference,goals,valid,ctx,torch.tensor([w.mode for w in worlds],device='cuda'))
class Engine:
 def __init__(self):
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);assert sha(c['PARENT'])==c['PSHA'];self.prior=FrozenPrior('cuda');state=torch.load(c['PARENT'],map_location='cpu',weights_only=False);weights={k[len('models.actor.'):]:v for k,v in state['system'].items() if k.startswith('models.actor.')};self.actor=LowActor(PROFILES['research']['actor']).cuda().eval().requires_grad_(False);self.actor.load_state_dict(weights,strict=True);self.fp=digest(self.actor.state_dict());self.jfp=self.prior.identity;self.calls=collections.Counter();self.initial_buffers={n:b.clone() for n,b in self.prior.model.named_buffers()};self.parent_component_counts={}
  for name in ['actor','q1','q2','high_actor','high_q1','high_q2','reach']:self.parent_component_counts[name]=sum(v.numel() for k,v in state['system'].items() if k.startswith('models.'+name+'.'))
  del state,weights
 @torch.no_grad()
 def distribution(self,worlds,policy):
  prefix=collate([w.state for w in worlds],'cuda');reference=self.prior(prefix);self.calls['J1_batches']+=1;self.calls['J1_states']+=len(worlds)
  if policy=='A_J1':return reference['logits'],reference['mu'],reference['ls'],reference,None
  batch=public_batch(worlds,prefix,reference)
  if policy=='B_NEUTRAL':batch['goals']=batch['goals']*0;batch['goal_valid']=batch['goal_valid']&False;batch['context']=batch['context']*0;batch['subgoal']=batch['subgoal']*0
  logits,mu,ls=self.actor(batch);self.calls['actor_batches']+=1;self.calls['actor_states']+=len(worlds);return logits,mu,ls,reference,batch
 def verify(self):
  self.prior.verify();assert self.prior.identity==self.jfp;assert all(torch.equal(v,self.initial_buffers[n]) for n,v in self.prior.model.named_buffers());assert digest(self.actor.state_dict())==self.fp;assert all(p.grad is None for p in self.prior.model.parameters())
def joint_kl(logits,mu,ls,reference,t):
 lp=logits.double().log_softmax(-1);rp=reference['logits'].double().log_softmax(-1);p=lp.exp();aa=(p*(lp-rp)).sum(-1);mask=torch.tensor([t>=1,t>=2,t>=3],device=mu.device);g=reference['ls'].double()-ls.double()+.5*((ls.double().exp()/reference['ls'].double().exp()).square()+((mu.double()-reference['mu'].double())/reference['ls'].double().exp()).square())-.5;geom=(p[:,:,None]*g*mask).sum((-1,-2));active=int(mask.sum());return aa,geom/(active or 1),aa+geom
@torch.no_grad()
def continue_worlds(engine,worlds,policy):
 initial_lengths=[w.state.t for w in worlds];records=[dict(sequence='',geometry=[],latent=[],AA_logprob=[],geometry_logprob=[],latent_action_logdensity=[],AA_KL=[],geometry_KL_per_active=[],joint_KL_raw=[],repairs=0,support_failure=None) for _ in worlds];start=time.perf_counter();N=worlds[0].goal['N'];live=list(range(len(worlds)))
 for step in range(min(w.state.t for w in worlds),N):
  active=[i for i in live if worlds[i].state.t<N]
  if not active:break
  ws=[worlds[i] for i in active];logits,mu,ls,ref,_=engine.distribution(ws,policy);p=logits.double().softmax(-1).cpu().numpy();aaKL,gKL,jKL=joint_kl(logits,mu,ls,ref,ws[0].state.t)
  for j,i in enumerate(active):
   w=worlds[i];rec=records[i]
   try:
    probs=p[j]/p[j].sum();cdf=probs.cumsum();cdf[-1]=1.;a=int(np.searchsorted(cdf,w.rng.random()));eps=torch.tensor(w.rng.standard_normal(3),device='cuda',dtype=torch.float64);mask=c['mask_for_step'](w.state.t);g,diagnostic=corrected_action(mu[j,a],ls[j,a],mask,eps);w.append(AA[a],g.cpu().tolist());rec['geometry'].append(g.cpu().tolist());rec['latent'].append(diagnostic['latent']);rec['AA_logprob'].append(float(logits[j].double().log_softmax(-1)[a]));rec['geometry_logprob'].append(diagnostic['represented_action_log_density']);rec['latent_action_logdensity'].append(diagnostic['latent_action_log_density']);rec['AA_KL'].append(float(aaKL[j]));rec['geometry_KL_per_active'].append(float(gKL[j]));rec['joint_KL_raw'].append(float(jKL[j]));rec['repairs']+=sum(diagnostic['adjusted_components'])
   except (ValueError,FloatingPointError,RuntimeError) as ex:rec['support_failure']=repr(ex);live.remove(i)
 for w,r,initial_length in zip(worlds,records,initial_lengths):
  seq=''.join(w.state.residues);r.update(sequence=seq,sequence_SHA256=__import__('hashlib').sha256(seq.encode()).hexdigest(),metrics=c['score'](seq,w.state.ca.numpy(),w.goal,w.spec),elapsed_group_seconds=time.perf_counter()-start,starting_prefix_length=initial_length,goal_id=w.goal['goal_id'],goal_witness_identity=w.goal['witness_identity'],goal_group=w.goal['protected_group'],policy=policy)
 return records
def sample(engine,goal,spec,policy,count=8,namespace='BASELINE'):
 worlds=[c['PropertyWorld'](goal,spec,ns(namespace,goal['goal_id'],i)) for i in range(count)];rows=continue_worlds(engine,worlds,policy)
 for i,r in enumerate(rows):r.update(candidate_index=i,seed=ns(namespace,goal['goal_id'],i),initial_prefix='EMPTY_CANONICAL_GAUGE',native_coordinates_provided=False)
 return rows,worlds
def engineering():
 spec=read(R/'REWARD_SPECIFICATION.json');goals=read(R/'manifests/TRAIN_GOALS.json')['goals'];assert goals;e=Engine();a,_=sample(e,goals[0],spec,'A_J1',2,'ENGINEERING');again,_=sample(e,goals[0],spec,'A_J1',2,'ENGINEERING');assert a[0]['sequence']==again[0]['sequence'] and a[0]['geometry']==again[0]['geometry'];assert all(max(r['AA_KL']+[0])<1e-6 and max(r['geometry_KL_per_active']+[0])<1e-6 for r in a);assert all(not r['support_failure'] for r in a);e.verify();save('DE_NOVO_ENGINEERING.json',dict(PASS=True,empty_prefix=True,early_masks=[list(c['mask_for_step'](i)) for i in range(4)],deterministic_sequence_geometry_replay=True,J1_reference_KL_zero=True,parent_actor_fingerprint=e.fp,J1_tensor_fingerprint=e.jfp,component_counts=e.parent_component_counts,actor_updates=0,peak_GPU=torch.cuda.max_memory_allocated(),calls=dict(e.calls)))
def baseline(panel):
 spec=read(R/'REWARD_SPECIFICATION.json');assert read(R/'CALIBRATION_ADMISSION.json')['PASS'];assert read(R/'REWARD_TESTS.json')['PASS'];assert read(R/'DE_NOVO_ENGINEERING.json')['PASS'];goals=read(R/f'manifests/{panel}_GOALS.json')['goals'];e=Engine();start=time.perf_counter();count=0
 for policy in ['A_J1','B_V3']:
  for i,goal in enumerate(goals):
   c['POLICY'].require(R,8*2**20);path=R/f'results/{panel}_{policy}_{i:03}.json'
   if path.exists():assert sha(path)==read(path.with_suffix('.sha.json'))['sha256'];continue
   rows,_=sample(e,goal,spec,policy);save(str(path.relative_to(R)),dict(goal=goal,records=rows));save(str(path.with_suffix('.sha.json').relative_to(R)),dict(sha256=sha(path),complete=True));count+=len(rows);print('BASELINE',panel,policy,i+1,len(goals),flush=True)
 e.verify();save(panel+'_BASELINES_COMPLETED.json',dict(complete=True,new_trajectories=count,goals=len(goals),elapsed_seconds=time.perf_counter()-start,calls=dict(e.calls),peak_GPU=torch.cuda.max_memory_allocated(),parent_actor_fingerprint=e.fp,J1_tensor_fingerprint=e.jfp,actor_updates=0))
def admission():
 files=sorted((R/'results').glob('TRAIN_B_V3_[0-9][0-9][0-9].json'));rows=[r for p in files for r in read(p)['records']];assert len(rows)==256;metrics=[r['metrics'] for r in rows];values=np.array([m['reward'] for m in metrics]);aa=[z for r in rows for z in r['AA_KL']];geom=[z for r in rows for z in r['geometry_KL_per_active'][1:]];finite=all(np.isfinite(r['AA_logprob']+r['geometry_logprob']).all() for r in rows);goodgoals={r['goal_id'] for r in rows if r['metrics']['hard_valid']};reasons=[];completion=np.mean([m['complete'] for m in metrics]);valid=np.mean([m['backbone_valid'] for m in metrics]);hard=np.mean([m['hard_valid'] for m in metrics]);means=[float(np.std([m.get(key,0) or 0 for m in metrics])) for key in ['exposed_mean','buried_mean','motif_utility']]
 if completion<.95:reasons.append('complete generation below95%')
 if valid<.2:reasons.append('backbone validity below20%')
 if hard<.1 or len(goodgoals)<16:reasons.append('insufficient sequence/packing-valid proposals across goals')
 if values.std()<.03 or (values>0).mean()<.1 or min(means)<=1e-8:reasons.append('reward/property variation admission fails')
 if not finite or any(r['support_failure'] or r['repairs'] for r in rows):reasons.append('strict-density training numerical admission fails')
 if np.mean(aa)>.5 or np.mean(geom)>1:reasons.append('initial joint policy KL safeguard fails')
 save('LOW_LEVEL_ADMISSION.json',dict(PASS=not reasons,reasons=reasons,trajectories=len(rows),goals=32,completion=float(completion),backbone_valid=float(valid),hard_valid=float(hard),goals_with_hard_valid=len(goodgoals),reward_mean=float(values.mean()),reward_SD=float(values.std()),nonzero_reward=float((values>0).mean()),full_success=sum(m['success'] for m in metrics),AA_KL=float(np.mean(aa)),geometry_KL_per_active=float(np.mean(geom)),property_SD=means,no_actor_updates=True))
if __name__=='__main__':
 command=sys.argv[1]
 if command.startswith('baseline_'):baseline(command.split('_',1)[1])
 else:globals()[command]()
