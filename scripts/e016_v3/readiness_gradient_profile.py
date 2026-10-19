"""Bounded real-J1 SAC gradient composition and 64/128 resource profiling; no step."""
import copy,json,math,time
from pathlib import Path
import torch
from torch.nn import functional as F
from protein_joint_sac.contracts import AA,BuildState
from protein_joint_sac.models.build_critic import TwinCritics
from protein_joint_sac.models.dense_geometry_attention import ModelConfig
from protein_joint_sac.geometry.ca_builder import replay,inverse
from protein_joint_sac.rl.sequence_rl import rollout,regularized_value,reference_kl,double_distribution
from protein_joint_sac.rl.sequence_sac import backward_sequence_first,SequenceFirstSAC
from protein_joint_sac.rl.checkpointing import load_checked
from readiness_r0 import R,SOURCE,SHA,actor,write,tensor_same
def main():
 torch.set_num_threads(6);torch.manual_seed(2919);torch.cuda.manual_seed_all(2919);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 source=load_checked(SOURCE);model=actor();model.load_state_dict(source['model']);ref=actor();ref.load_state_dict(source['model']);ref.requires_grad_(False);q=TwinCritics(ModelConfig(width=64,heads=4,layers=2,ffn_mult=2,chemistry=False,backend='sdpa')).cuda().requires_grad_(False)
 _,_,states=rollout(model,128,2026101917,SHA);profiles=[]
 for t in [32,64,127]:
  s=states[t]
  for estimator in ['exact','score_function']:
   model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
   with torch.autocast('cuda',dtype=torch.bfloat16):v,_=regularized_value(model,q,s,ref,estimator=estimator)
   (-v).backward();torch.cuda.synchronize();grad=math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in model.parameters() if p.grad is not None));assert math.isfinite(grad) and grad>0;profiles.append({'prefix':t,'N':128,'estimator':estimator,'seconds':time.monotonic()-start,'allocated_peak':torch.cuda.max_memory_allocated(),'reserved_peak':torch.cuda.max_memory_reserved(),'all_losses_finite':True,'gradient_L2':grad,'full_continuous_gradient':True,'critic_parameters_each':sum(p.numel() for p in q.q1.parameters())});print(json.dumps(profiles[-1]),flush=True)
 limits=json.load(open(R.parent/'e016_j1_torsion_stabilization_20261018_v1/configs/FROZEN.json'))['block_limits']
 for n,p in model.named_parameters():p.requires_grad_(not n.startswith(('geom.','selected_aa.')))
 native=next(j for j in json.load(open(R/'configs/NATIVE_CONTROL_PANEL.json'))['controls'] if j['kind']=='native' and j['split']=='TRAIN' and not j['historical_case']);seq=native['sequence'];xyz=torch.tensor(native['native_ca'],device='cuda',dtype=torch.float64);t=len(seq)//2;s=BuildState(tuple(seq[:t]),xyz[:t],len(seq));label=torch.tensor([AA.index(seq[t])],device='cuda');target,mask=inverse(xyz[:t+1])
 def seq_loss():return F.cross_entropy(model.logits(s)[None],label)
 def geometry_loss():return .05*(-double_distribution(model,s,seq[t]).log_prob(target)/3)+.01*reference_kl(model,ref,s)[1]
 sl,gl,stats=backward_sequence_first(model,seq_loss,geometry_loss,limits);assert stats['geometry_to_sequence_shared_ratio']<=.5+1e-9;assert all(p.grad is None for n,p in model.named_parameters() if n.startswith(('geom.','selected_aa.')))
 # Production gate must reject before any critic, temperature or actor optimizer step.
 config={'actor_lr':1e-6,'critic_lr':1e-4,'alpha_A':.01,'alpha_G':.001,'target_AA_entropy':2.,'target_geometry_entropy':0.,'temperature_lr':1e-4,'production_authorized':False}
 learner=SequenceFirstSAC(model,ref,q,config);blocked=False
 try:learner.update_resolved(s,None,None,s,False,PDB_AA=seq_loss,PDB_geometry=geometry_loss)
 except PermissionError:blocked=True
 assert blocked and not learner.actor_opt.state and not learner.critic_opt.state and not learner.alpha_opt.state;assert tensor_same({k:v.cpu() for k,v in model.state_dict().items()},source['model'])
 write('RL_GRADIENT_AND_RESOURCE_PROFILE.json',{'status':'PASS','profiles':profiles,'PDB_shared_gradient_composer':stats,'frozen_geometry_output_gradients_absent':True,'production_gate_before_any_optimizer_step':True,'J1_optimizer_updates':0,'model_unchanged':True,'R2_numeric_weights_used_here':'synthetic engineering constants, not calibrated production recommendations'})
 print('R0 real J1 gradient composition and production gate PASS; no optimizer step.',flush=True)
if __name__=='__main__':main()
