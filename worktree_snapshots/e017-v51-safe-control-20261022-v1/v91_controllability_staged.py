"""Bounded TRAIN-only instruction/geometry pathway learning; weights never reused."""
import copy,json,runpy,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v91_ordered_goal_recovery_20261023_v1');p=runpy.run_path(str(R/'source/scripts/e017_v91/policy.py'));c=p['c'];g=p['g'];read=c['read'];save=c['save'];AA=c['AA'];OLD=p['OLD']
def run():
 assert read(R/'GOAL_OBSERVABILITY_TESTS.json')['PASS'];cfg=read(R/'FROZEN_STUDY_PROTOCOL.json')['controllability'];torch.manual_seed(9191);e=p['Engine']();e.actor.requires_grad_(True);e.actor.train();spec=read(R/'REWARD_SPECIFICATION.json');source=read(OLD/'results/TRAIN_B_V3_000.json');raw=source['goal'];row=next(r for r in source['records'] if r['metrics']['hard_valid']);a=raw['exposed'][0];first=raw['motif'];others=[x for x in AA if x not in first];second=others[0]+others[1];motifs=[first,first[::-1],second,second[::-1]];fixtures=[];labels=[];signs=[]
 for motif in motifs:
  goal=copy.deepcopy(raw);goal['motif']=motif;sign=1 if AA.index(motif[0])<AA.index(motif[1]) else -1
  for stage in [0,1]:
   world=c['PropertyWorld'](goal,spec,9191)
   for t in range(a+stage):world.append(motif[0] if t==a else row['sequence'][t],row['geometry'][t])
   fixtures.append(world);labels.append(AA.index(motif[stage]));signs.append(sign)
 prefix=p['collate']([w.state for w in fixtures],'cuda')
 with torch.no_grad():ref=e.prior(prefix)
 batch=g['public_batch'](fixtures,prefix,ref);labels=torch.tensor(labels,device='cuda');probe_AA=AA.index('A');target=ref['mu'][:,probe_AA]+torch.tensor(signs,device='cuda')[:,None]*torch.tensor([.01,.08,.08],device='cuda');target=target.detach();opt=torch.optim.AdamW(e.actor.parameters(),lr=cfg['lr'],weight_decay=0.);history=[];start=time.perf_counter();passed=False
 def metrics(out):
  logits,mu,ls=out;accuracy=int((logits.argmax(-1)==labels).sum());rmse=float((((mu[:,probe_AA]-target)/mu.new_tensor([.15,1.,1.])).square().mean().sqrt()).detach());delta=(mu[:,probe_AA]-ref['mu'][:,probe_AA]).detach();contrast=all(float(((delta[j]-delta[j+2])*torch.tensor(signs[j],device='cuda')).mean())>0 for j in [0,4]);return accuracy,rmse,contrast
 for u in range(cfg['max_updates']+1):
  e.actor.eval();out=e.actor(batch);acc,rmse,contrast=metrics(out);history.append(dict(update=u,AA_correct=acc,AA_total=8,geometry_normalized_RMSE=rmse,geometry_order_contrast=contrast))
  if acc==8 and rmse<.1 and contrast:passed=True;break
  if u==cfg['max_updates']:break
  e.actor.train();opt.zero_grad(set_to_none=True);out=e.actor(batch);loss=torch.nn.functional.cross_entropy(out[0],labels)+.5*((out[1][:,probe_AA]-target)/target.new_tensor([.15,1.,1.])).square().mean();assert torch.isfinite(loss);loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(e.actor.parameters(),5));assert np.isfinite(grad);opt.step()
 # Autoregressive two-action execution with the learned actor, fixed valid
 # TRAIN geometry actions, and no hidden future sequence supplied to the actor.
 e.actor.eval();motif_hits=[]
 for motif in motifs:
  goal=copy.deepcopy(raw);goal['motif']=motif;world=c['PropertyWorld'](goal,spec,9191)
  for t in range(a):world.append(row['sequence'][t],row['geometry'][t])
  chosen=''
  for j in [0,1]:
   prefix=p['collate']([world.state],'cuda')
   with torch.no_grad():ref_step=e.prior(prefix);logits,mu,ls=e.actor(g['public_batch']([world],prefix,ref_step));aa=AA[int(logits.argmax(-1))]
   chosen+=aa;world.append(aa,row['geometry'][a+j])
  motif_hits.append(dict(requested=motif,chosen=chosen,matched=chosen==motif,sequence_constraint_satisfied=g['sequence_progress'](''.join(world.state.residues),g['canonical'](goal))['complete']))
 passed=passed and all(v['matched'] and v['sequence_constraint_satisfied'] for v in motif_hits);e.prior.verify();assert c['state_digest'](e.parent.state_dict())==e.fp
 from protein_joint_sac.crisp_v3.recovery import atomic_checkpoint
 root=R/'checkpoints/engineering';root.mkdir(exist_ok=True);record=atomic_checkpoint(root/'CONTROLLABILITY_ACTOR.pt',dict(actor={k:v.detach().cpu() for k,v in e.actor.state_dict().items()},updates=history[-1]['update'],parent_SHA256=c['PSHA'],J1_SHA256=c['J1SHA'],fixtures=motifs,config=cfg,weights_not_for_protein_RL=True),c['POLICY'])
 save('CONTROLLABILITY_TEST.json',dict(PASS=passed,TRAIN_only=True,motifs=motifs,prefix_source=str(OLD/'results/TRAIN_B_V3_000.json'),source_candidate_index=row['candidate_index'],first_target_position=a,history=history,autoregressive_motif_tests=motif_hits,updates=history[-1]['update'],engineering_checkpoint=record,actor_weights_discarded_for_production=True,engineering_not_protein_reward_optimization=True,no_native_future_coordinates_or_sequence=True,geometry_teacher='synthetic reference latent mean +/-[.01,.08,.08] under fixed AA=A, inside unchanged residual/support; not a new production target',J1_fingerprint=e.jfp,parent_fingerprint=e.fp,peak_GPU_bytes=torch.cuda.max_memory_allocated(),seconds=time.perf_counter()-start))
 admission=read(R/'OBSERVABILITY_ADMISSION.json');admission.update(controllability_PASS=passed,controllability_pending=False,RL_authorized=False,RL_pending='storage/replay/initial proposal and validity admission');save('OBSERVABILITY_ADMISSION.json',admission);print(json.dumps(dict(PASS=passed,updates=history[-1]['update'],tests=motif_hits,last=history[-1])))
if __name__=='__main__':run()
