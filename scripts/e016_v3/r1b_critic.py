"""One controlled critic-only action-conditioning correction versus original and MLP."""
import copy,json,math,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from protein_joint_sac.contracts import AA,BuildAction
from protein_joint_sac.geometry.ca_builder import replay,append,mask_for_step
from protein_joint_sac.models.build_critic import CandidateQ
from protein_joint_sac.models.dense_geometry_attention import ModelConfig
from protein_joint_sac.models.batched_bc import collate,features_batch,encode_batch
from protein_joint_sac.rl.sequence_rl import bellman_target,discounted_nstep
from r1b_common import R,PREV,write,sha,guard

def main():
 if (R/'SAC_CRITIC_COMPARISON.json').exists():print('Critic experiment already complete; no repeat.');return
 guard();torch.set_num_threads(6);torch.manual_seed(191917);torch.cuda.manual_seed_all(191917);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 frozen=json.load(open(PREV/'generated/J1_initial64.json'));actions=[BuildAction(v['residue'],torch.tensor(v['geometry'],dtype=torch.float64),tuple(v['mask'])) for v in frozen['trace'][:8]];state=replay(64,actions);base=state.to('cuda');acts=[];latent=[];ys=[];rng=np.random.default_rng(19019);center=np.array([math.log(3.8),math.log(1.9/(math.pi-1.9)),math.atanh(.7/math.pi)])
 for i in range(128):
  u=center+rng.normal(0,.15,3);g=np.array([np.exp(u[0]),math.pi/(1+np.exp(-u[1])),math.pi*np.tanh(u[2])]);acts.append(BuildAction(AA[i%20],torch.tensor(g),mask_for_step(state.t)));latent.append(u);ys.append(.5*(i%20)/19+.5*np.exp(-np.square(u-center).sum()/.08))
 latent=np.array(latent);mean=latent[:96].mean(0);std=latent[:96].std(0);explicit=torch.cat([torch.nn.functional.one_hot(torch.arange(128)%20,20).float(),torch.tensor((latent-mean)/std).float(),torch.tensor([[64/500,8/64]]).expand(128,-1)],-1).cuda();y=torch.tensor(ys,device='cuda',dtype=torch.float32);candidate=collate([append(state,a) for a in acts],'cuda');prepared=features_batch(candidate,False);train=torch.arange(96,device='cuda');held=torch.arange(96,128,device='cuda');config=ModelConfig(width=32,heads=4,layers=1,ffn_mult=2,chemistry=False,backend='sdpa');original=[CandidateQ(config).cuda() for _ in range(2)];corrected=[]
 for q in original:
  c=copy.deepcopy(q);c.head=nn.Sequential(nn.Linear(64+25,32),nn.SiLU(),nn.Linear(32,1)).cuda()
  with torch.no_grad():c.head[0].weight[:,:64].copy_(q.head[0].weight);c.head[0].weight[:,64:].zero_();c.head[0].bias.copy_(q.head[0].bias);c.head[2].load_state_dict(q.head[2].state_dict())
  corrected.append(c)
 mlps=[nn.Sequential(nn.Linear(25,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,1)).cuda() for _ in range(2)]
 def pred(q,idx,arm):
  if arm=='candidate_MLP':return q(explicit[idx]).squeeze(-1)
  b={k:v[idx] for k,v in candidate.items()};f=tuple(None if v is None else tuple(w[idx] for w in v) if isinstance(v,tuple) else v[idx] for v in prepared);h=q.current(base);z=encode_batch(SimpleNamespace(encoder=q.candidate),b,prepared=f);inputs=[h.expand(len(idx),-1),z]
  if arm=='Transformer_direct_action':inputs.append(explicit[idx])
  return q.head(torch.cat(inputs,-1)).squeeze(-1)
 # Zero extra weights preserve mathematical predictions; changed GEMM width may round differently in FP32.
 with torch.no_grad():
  initial_roundoff=[]
  for a,b in zip(original,corrected):
   pa=pred(a,held,'Transformer_original');pb=pred(b,held,'Transformer_direct_action');initial_roundoff.append(float((pa-pb).abs().max()));torch.testing.assert_close(pa,pb,rtol=1e-6,atol=1e-6)
 aa_means={i:np.mean([ys[k] for k in range(96) if k%20==i]) for i in range(20)};aa_rmse=float(np.sqrt(np.mean([(aa_means[i%20]-ys[i])**2 for i in range(96,128)])));targetstats={'range':[float(min(ys)),float(max(ys))],'TRAIN_mean':float(np.mean(ys[:96])),'TRAIN_std':float(np.std(ys[:96])),'heldout_std':float(np.std(ys[96:])),'AA_only_TRAIN_classmean_heldout_RMSE':aa_rmse,'TRAIN_latent_mean':mean.tolist(),'TRAIN_latent_std':std.tolist(),'shared_prefix_in_benchmark':True,'continuous_action_information_present_but_indirect_in_original':True}
 frozen_config={'seed':191917,'latent_data_seed':19019,'TRAIN_indices':[0,95],'heldout_indices':[96,127],'updates_per_arm':400,'batch':32,'lr':.002,'weight_decay':.01,'gradient_clip':5.,'heldout_RMSE_limit':.1,'arms':['Transformer_original','Transformer_direct_action','candidate_MLP'],'single_correction':'add25 direct action/progress features to existing Q readout; latent scales fitted only on96TRAIN actions; zero-initialized extra readout weights preserve initial predictions','no_hyperparameter_search':True,'J1_optimizer_updates':0,'frozen_benchmark_same_actions_targets_as_R0':True,'historical_initialization_not_reconstructed':True};write('configs/CRITIC_FROZEN.json',frozen_config);write('critic/FROZEN_DATA.json',{'latent':latent.tolist(),'rewards':ys,'AA':[AA[i%20] for i in range(128)],'state_source':str(PREV/'generated/J1_initial64.json'),'state_source_sha256':sha(PREV/'generated/J1_initial64.json'),'prefix_length':8,'N':64,'TRAIN_latent_scales':targetstats});rg=torch.Generator().manual_seed(190193);batches=[torch.randperm(96,generator=rg)[:32].cuda() for _ in range(400)];results=[]
 for arm,qs in [('Transformer_original',original),('Transformer_direct_action',corrected),('candidate_MLP',mlps)]:
  guard();parameters=[p for q in qs for p in q.parameters()];opt=torch.optim.AdamW(parameters,lr=.002,weight_decay=.01);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic();curve=[];clips=0;norms=[]
  for k,idx in enumerate(batches):
   loss=sum((pred(q,idx,arm)-y[idx]).square().mean() for q in qs);opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(parameters,5.,error_if_nonfinite=True));clips+=norm>5;norms.append(norm);opt.step()
   if k in [0,99,199,399]:curve.append({'update':k+1,'TRAIN_minibatch_MSE':float(loss.detach())})
  torch.cuda.synchronize();elapsed=time.monotonic()-start
  with torch.no_grad():p=torch.stack([pred(q,held,arm) for q in qs]);pt=torch.stack([pred(q,train,arm) for q in qs]);errors=(p-y[held]).cpu().numpy();rms=[float(v) for v in (p-y[held]).square().mean(1).sqrt()];trainrms=[float(v) for v in (pt-y[train]).square().mean(1).sqrt()]
  rr={'arm':arm,'status':'PASS' if max(rms)<.1 else 'NO_GO','RMSE_heldout':rms,'RMSE_TRAIN':trainrms,'parameters_per_Q':sum(v.numel() for v in qs[0].parameters()),'seconds':elapsed,'gradient_norm_median':float(np.median(norms)),'clipping_fraction':clips/400,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'training_curve':curve,'heldout_predictions':p.cpu().tolist(),'heldout_targets':y[held].cpu().tolist(),'mean_twin_disagreement':float((p[0]-p[1]).abs().mean()),'per_AA_RMSE':{AA[i]:[float(np.sqrt(np.mean(e[np.array([k%20==i for k in range(96,128)])]**2))) for e in errors] for i in range(20)},'predicted_vs_actual_correlation':[float(np.corrcoef(v,y[held].cpu().numpy())[0,1]) for v in p.cpu().numpy()]};results.append(rr);print(json.dumps({k:rr[k] for k in ['arm','status','RMSE_heldout','seconds']}),flush=True)
 assert bellman_target(1.,True,100.)==1. and discounted_nstep([0.,0.,1.])==1.
 old=json.load(open(PREV/'TRANSFORMER_CRITIC_DIAGNOSTIC.json'));write('SAC_CRITIC_COMPARISON.json',{'historical_benchmark_unchanged':old,'frozen_configuration':frozen_config,'target_diagnostics':targetstats,'initial_same_prediction_FP32_max_abs_difference':initial_roundoff,'rows':results,'corrected_Transformer_pass':next(v for v in results if v['arm']=='Transformer_direct_action')['status']=='PASS','real_protein_return_calibration':'NOT_ESTABLISHED','terminal_gamma1_checks_pass':True,'J1_optimizer_updates':0,'scope':'one fixed generated-prefix synthetic action fixture; not unseen protein states or true AF reward learning'})
if __name__=='__main__':main()
