"""One corrected TRAIN engineering attempt with preverified feasible targets."""
import copy,json,runpy,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v91_ordered_goal_recovery_20261023_v1');p=runpy.run_path(str(R/'source/scripts/e017_v91/policy.py'));c=p['c'];g=p['g'];read=c['read'];save=c['save'];AA=c['AA']
def run():
 receipt=read(R/'GOAL_OBSERVABILITY_TESTS.json');assert receipt['PASS'] and all(c['sha'](R/'source/scripts/e017_v91'/n)==h for n,h in receipt['source_SHA256'].items());eligible=read(R/'FEASIBLE_CONTROLLABILITY_FIXTURES.json');assert eligible['PASS'];cfg=read(R/'FROZEN_STUDY_PROTOCOL.json')['controllability'];torch.manual_seed(9191);e=p['Engine']();e.actor.requires_grad_(True);spec=read(R/'REWARD_SPECIFICATION.json');fixtures=[];labels=[];signs=[];sources=[]
 for group in eligible['groups']:
  source=read(Path(group['source']));raw=source['goal'];row=source['records'][group['candidate_index']];a=group['position'];sources.append(dict(raw=raw,row=row,a=a,motifs=group['motifs']))
  for motif in group['motifs']:
   goal=copy.deepcopy(raw);goal['motif']=motif;sign=1 if AA.index(motif[0])<AA.index(motif[1]) else -1
   for stage in [0,1]:
    w=c['PropertyWorld'](goal,spec,9191)
    for t in range(a+stage):w.append(motif[0] if t==a else row['sequence'][t],row['geometry'][t])
    fixtures.append(w);labels.append(AA.index(motif[stage]));signs.append(sign)
 prefix=p['collate']([w.state for w in fixtures],'cuda')
 with torch.no_grad():ref=e.prior(prefix)
 batch=g['public_batch'](fixtures,prefix,ref);labels=torch.tensor(labels,device='cuda');probe_AA=AA.index('A');target=(ref['mu'][:,probe_AA]+torch.tensor(signs,device='cuda')[:,None]*torch.tensor([.01,.08,.08],device='cuda')).detach();opt=torch.optim.AdamW(e.actor.parameters(),lr=cfg['lr'],weight_decay=0.);history=[];start=time.perf_counter();passed=False
 def metrics(out):
  logits,mu,ls=out;accuracy=int((logits.argmax(-1)==labels).sum());rmse=float((((mu[:,probe_AA]-target)/mu.new_tensor([.15,1.,1.])).square().mean().sqrt()).detach());delta=(mu[:,probe_AA]-ref['mu'][:,probe_AA]).detach();contrast=all(float(((delta[j]-delta[j+2])*torch.tensor(signs[j],device='cuda')).mean())>0 for j in [0,4]);return accuracy,rmse,contrast
 for u in range(cfg['max_updates']+1):
  e.actor.eval();out=e.actor(batch);acc,rmse,contrast=metrics(out);history.append(dict(update=u,AA_correct=acc,AA_total=8,geometry_normalized_RMSE=rmse,geometry_order_contrast=contrast))
  if acc==8 and rmse<.1 and contrast:passed=True;break
  if u==cfg['max_updates']:break
  e.actor.train();opt.zero_grad(set_to_none=True);out=e.actor(batch);loss=torch.nn.functional.cross_entropy(out[0],labels)+.5*((out[1][:,probe_AA]-target)/target.new_tensor([.15,1.,1.])).square().mean();assert torch.isfinite(loss);loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(e.actor.parameters(),5));assert np.isfinite(grad);opt.step()
 e.actor.eval();motif_hits=[]
 for source in sources:
  raw,row,a=source['raw'],source['row'],source['a']
  for motif in source['motifs']:
   goal=copy.deepcopy(raw);goal['motif']=motif;world=c['PropertyWorld'](goal,spec,9191)
   for t in range(a):world.append(row['sequence'][t],row['geometry'][t])
   chosen=''
   for j in [0,1]:
    prefix=p['collate']([world.state],'cuda')
    with torch.no_grad():rs=e.prior(prefix);logits,mu,ls=e.actor(g['public_batch']([world],prefix,rs));aa=AA[int(logits.argmax(-1))]
    chosen+=aa;world.append(aa,row['geometry'][a+j])
   motif_hits.append(dict(requested=motif,chosen=chosen,matched=chosen==motif,sequence_constraint_satisfied=g['sequence_progress'](''.join(world.state.residues),g['canonical'](goal))['complete'],source_identity=raw['witness_identity'],position=a))
 passed=passed and all(v['matched'] and v['sequence_constraint_satisfied'] for v in motif_hits);e.prior.verify();assert c['state_digest'](e.parent.state_dict())==e.fp
 from protein_joint_sac.crisp_v3.recovery import atomic_checkpoint
 record=atomic_checkpoint(R/'checkpoints/engineering/CONTROLLABILITY_FEASIBLE_ACTOR.pt',dict(actor={k:v.detach().cpu() for k,v in e.actor.state_dict().items()},updates=history[-1]['update'],parent_SHA256=c['PSHA'],J1_SHA256=c['J1SHA'],fixtures=eligible,config=cfg,weights_not_for_protein_RL=True),c['POLICY'])
 save('CONTROLLABILITY_FEASIBLE_TEST.json',dict(PASS=passed,TRAIN_only=True,history=history,autoregressive_motif_tests=motif_hits,updates=history[-1]['update'],engineering_checkpoint=record,engineering_weights_discarded_for_RL=True,prior_failed_ineligible_test_preserved=True,no_criteria_or_hyperparameter_changes=True,fixture_eligibility_proof='FEASIBLE_CONTROLLABILITY_FIXTURES.json',J1_fingerprint=e.jfp,parent_fingerprint=e.fp,peak_GPU_bytes=torch.cuda.max_memory_allocated(),seconds=time.perf_counter()-start));save('CONTROLLABILITY_ADMISSION.json',dict(PASS=passed,valid_test='CONTROLLABILITY_FEASIBLE_TEST.json',ineligible_test='CONTROLLABILITY_TEST.json',total_engineering_updates=100+history[-1]['update'],maximum_engineering_attempts=2,RL_updates=0,RL_pending='complete checkpoint staging/replay and initial proposal/validity admission'))
 admission=read(R/'OBSERVABILITY_ADMISSION.json');admission.update(controllability_PASS=passed,controllability_pending=False,RL_authorized=False);save('OBSERVABILITY_ADMISSION.json',admission);print(json.dumps(dict(PASS=passed,updates=history[-1]['update'],last=history[-1],motif_tests=motif_hits)))
if __name__=='__main__':run()
