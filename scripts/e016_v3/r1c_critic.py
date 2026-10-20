"""One sealed full-trajectory hybrid critic comparison. Synthetic actors only; no J1 loads."""
from r1c_common import *
import copy,math,time
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from protein_joint_sac.contracts import AA,BuildAction,BuildState
from protein_joint_sac.geometry.ca_builder import append,replay,mask_for_step
from protein_joint_sac.models.build_critic import CandidateQ
from protein_joint_sac.models.dense_geometry_attention import ModelConfig
from protein_joint_sac.models.batched_bc import collate,features_batch,encode_batch
from protein_joint_sac.models.hybrid_distributions import GeometryDistribution
from protein_joint_sac.rl.sequence_rl import bellman_target,discounted_nstep
from protein_joint_sac.rl.bc_training import rng_state,restore_rng
CENTER=np.array([math.log(3.8),math.log(1.9/(math.pi-1.9)),math.atanh(.7/math.pi)])
SIG=.15;SCALE=.22;EG=(1+SIG**2/SCALE**2)**-1.5;ES=.55*.5+.45*EG

def geo(u):return np.array([np.exp(u[0]),math.pi/(1+np.exp(-u[1])),math.pi*np.tanh(u[2])])
def gscore(u):return np.exp(-np.square((u-CENTER)/SCALE).sum(-1)/2)
def terminal(a,u):
 # All rewards delayed to the last step; early choices create persistent return dependence.
 scores=.55*np.asarray(a)/19+.45*gscore(np.asarray(u));scores[:3]=.55*np.asarray(a[:3])/19+.45*EG
 return .2*a[0]/19+.2*gscore(u[3])+.6*scores.mean()
def montecarlo(prefix_a,prefix_u,N,rng,M=128):
 k=len(prefix_a);remain=N-k;aa=rng.integers(0,20,(M,remain));uu=CENTER+rng.normal(0,SIG,(M,remain,3));a=np.concatenate([np.broadcast_to(prefix_a,(M,k)),aa],1);u=np.concatenate([np.broadcast_to(prefix_u,(M,k,3)),uu],1)
 s=.55*a/19+.45*gscore(u);s[:,:3]=.55*a[:,:3]/19+.45*EG
 return .2*a[:,0]/19+.2*gscore(u[:,3])+.6*s.mean(1)
def exact(prefix_a,prefix_u,N):
 k=len(prefix_a);s=.55*np.array(prefix_a)/19+.45*gscore(np.array(prefix_u));s[:min(k,3)]=.55*np.array(prefix_a[:3])/19+.45*EG
 return .2*prefix_a[0]/19+.2*(gscore(prefix_u[3]) if k>=4 else EG)+.6*(s.sum()+(N-k)*ES)/N

def dataset():
 states=[];candidates=[];vectors=[];targets=[];mc=[];metadata=[];episodes=[];maxlogerr=0.;maxreplay=0.
 for split,count,lengths,seed in [('TRAIN',64,[20,24,32,48],190210),('IID',16,[20,24,32,48],190211),('OOD',16,[64,96],190212)]:
  rng=np.random.default_rng(seed)
  for ep in range(count):
   N=lengths[ep%len(lengths)];a=rng.integers(0,20,N);u=CENTER+rng.normal(0,.28 if split=='OOD' else SIG,(N,3));s=BuildState.empty(N);actions=[];trace=[];selected=set(np.unique(np.linspace(0,N-1,8,dtype=int)).tolist())
   for t in range(N):
    mask=mask_for_step(t);g=torch.tensor(geo(u[t]),dtype=torch.float64)*torch.tensor(mask);action=BuildAction(AA[a[t]],g,mask)
    mu=torch.tensor(CENTER,dtype=torch.float64);ls=torch.full((3,),math.log(.28 if split=='OOD' else SIG),dtype=torch.float64);pd=GeometryDistribution(mu,ls,mask)
    lp=float(pd.log_prob(g));direct=float(torch.where(torch.tensor(mask),pd.base.log_prob(torch.tensor(u[t]))-pd.log_jacobian(torch.tensor(u[t])),0.).sum());maxlogerr=max(maxlogerr,abs(lp-direct));assert math.isfinite(lp)
    nxt=append(s,action);trace.append({'AA':AA[a[t]],'latent':u[t].tolist(),'geometry':g.tolist(),'mask':list(mask),'AA_logp':-math.log(20),'geometry_logp':lp})
    if t in selected:
     hist=np.bincount(a[:t],minlength=20)/max(t,1);ss=.55*a[:t]/19+.45*gscore(u[:t]);ss[:min(t,3)]=.55*a[:min(t,3)]/19+.45*EG
     v=np.r_[np.eye(20)[a[t]],(u[t]-CENTER)/SIG*np.array(mask),N/100,t/N,hist,ss.mean() if t else ES,a[0]/19 if t else .5,gscore(u[3]) if t>=4 else EG]
     future=montecarlo(a[:t+1],u[:t+1],N,rng);q=exact(a[:t+1],u[:t+1],N)
     states.append(s);candidates.append(nxt);vectors.append(v);targets.append(q);mc.append(future.tolist());metadata.append({'split':split,'episode':f'{split}_{ep}','N':N,'t':t,'AA':int(a[t]),'analytic_Q':float(q),'MC_mean':float(future.mean()),'MC_SE':float(future.std()/math.sqrt(len(future)))})
    actions.append(action);s=nxt
   reward=float(terminal(a,u));assert abs(exact(a,u,N)-reward)<1e-12;assert bellman_target(0.,False,reward)==reward and discounted_nstep([0.,0.],reward,False)==reward;rr=replay(N,actions);maxreplay=max(maxreplay,float((rr.ca-s.ca).abs().max()));assert rr.residues==s.residues
   rewards=[0.]*(N-1)+[reward];assert discounted_nstep(rewards)==reward and bellman_target(reward,True,1e9)==reward
   episodes.append({'id':f'{split}_{ep}','split':split,'N':N,'trace':trace,'terminal_reward':reward,'intermediate_rewards_all_zero':True,'coordinates_sha256':hashlib.sha256(s.ca.numpy().tobytes()).hexdigest()})
 write('critic/FULL_EPISODES.json',{'episodes':episodes,'action_logprob_max_inverse_error':maxlogerr,'replay_coordinate_max_error':maxreplay,'behavior':'uniform categorical; Gaussian latent std .15 TRAIN/IID and .28 OOD; OOD Q evaluated under the same .15 continuation policy, an explicit off-policy state/action stress test','return':'gamma1, terminal .2*firstAAindex/19 + .2*first full-geometry exp(-latent deviation²/(2*.22²)) + .6*mean local score; [0,1]; synthetic not protein quality','bootstrap':'geometry contribution of first3 targets is fixed expectation; invalid continuous components masked from log probability','reference_future':'uniform AA and Gaussian .15;128 actual MC branches per state-action; analytic conditional mean used as noise-free training target and separately verified against MC'})
 write('critic/STATE_ACTION_TARGETS.json',{'rows':metadata,'features':np.asarray(vectors).tolist(),'MC_returns':mc,'MC_mean_vs_analytic_RMSE':float(np.sqrt(np.mean([(r['MC_mean']-r['analytic_Q'])**2 for r in metadata]))),'MC_mean_SE_median':float(np.median([r['MC_SE'] for r in metadata]))})
 assert maxlogerr<1e-8 and maxreplay==0
 return states,candidates,np.array(vectors),np.array(targets),np.array(mc),metadata

def sliced(pre,idx):
 return tuple(None if v is None else tuple(w[idx] for w in v) if isinstance(v,tuple) else v[idx] for v in pre)
def main():
 assert not (R/'FULL_TRAJECTORY_CRITIC_COMPARISON.json').exists();guard();torch.set_num_threads(6);torch.manual_seed(190220);torch.cuda.manual_seed_all(190220);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg={'seed':190220,'episodes':{'TRAIN':64,'IID':16,'OOD':16},'TRAIN_lengths':[20,24,32,48],'OOD_lengths':[64,96],'transitions_per_episode':8,'full_episodes_retained':True,'steps_per_arm':800,'batch':32,'AdamW_lr':.002,'weight_decay':.01,'clip_norm':5.,'heldout_RMSE_limit':.1,'model_selection':'one fixed configuration, no sweep','actor':'synthetic random policy; no J1 model imported or loaded','target':'exact conditional expected terminal return checked against128 MC branches; no future suffix in input','OOD':'new lengths and off-policy geometry .28 vs .15; future Monte Carlo continuation always reference .15','features':'same direct candidate AA, latent action, length/progress, observed prefix histogram and causal sufficient-statistic summaries for both arms; Transformer also receives prefix and candidate-prefix encoders','scope':'value regression and MC calibration, not evidence of stable bootstrapped SAC policy optimization','max_critic_seconds_per_arm':600}
 write('configs/FULL_CRITIC_FROZEN.json',cfg);ss,cs,x,y,mc,meta=dataset();splits={k:np.flatnonzero([r['split']==k for r in meta]) for k in ['TRAIN','IID','OOD']};mean=x[splits['TRAIN']].mean(0);std=np.maximum(x[splits['TRAIN']].std(0),.05);x=(x-mean)/std;write('critic/TRAIN_FEATURE_SCALES.json',{'mean':mean.tolist(),'std':std.tolist()})
 X=torch.tensor(x,device='cuda',dtype=torch.float32);Y=torch.tensor(y,device='cuda',dtype=torch.float32);batchS=collate(ss,'cuda');batchC=collate(cs,'cuda');prepS=features_batch(batchS,False);prepC=features_batch(batchC,False)
 conf=ModelConfig(width=32,heads=4,layers=1,ffn_mult=2,chemistry=False,backend='sdpa');transformers=[]
 for _ in range(2):
  q=CandidateQ(conf).cuda();q.head=nn.Sequential(nn.Linear(64+x.shape[1],32),nn.SiLU(),nn.Linear(32,1)).cuda();transformers.append(q)
 mlps=[nn.Sequential(nn.Linear(x.shape[1],128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,1)).cuda() for _ in range(2)]
 def predict(q,idx,arm):
  if arm=='MLP':return q(X[idx]).squeeze(-1)
  bs={k:v[idx] for k,v in batchS.items()};bc={k:v[idx] for k,v in batchC.items()};h=encode_batch(SimpleNamespace(encoder=q.current),bs,prepared=sliced(prepS,idx));z=encode_batch(SimpleNamespace(encoder=q.candidate),bc,prepared=sliced(prepC,idx));return q.head(torch.cat([h,z,X[idx]],-1)).squeeze(-1)
 rg=torch.Generator().manual_seed(190222);batches=[torch.tensor(splits['TRAIN'][torch.randperm(len(splits['TRAIN']),generator=rg)[:32].numpy()],device='cuda') for _ in range(800)];rows=[]
 for arm,qs in [('Transformer',transformers),('MLP',mlps)]:
  opt=torch.optim.AdamW([p for q in qs for p in q.parameters()],lr=.002,weight_decay=.01);targets=copy.deepcopy(qs);start=time.monotonic();torch.cuda.reset_peak_memory_stats();restart=None;curve=[]
  def step(idx):
   opt.zero_grad(set_to_none=True);loss=sum((predict(q,idx,arm)-Y[idx]).square().mean() for q in qs);loss.backward();n=torch.nn.utils.clip_grad_norm_([p for q in qs for p in q.parameters()],5.,error_if_nonfinite=True);opt.step()
   with torch.no_grad():
    for a,b in zip(qs,targets):
     for p,z in zip(a.parameters(),b.parameters()):z.lerp_(p,.01)
   return float(loss.detach()),float(n)
  for k,idx in enumerate(batches):
   if time.monotonic()-start>600:raise RuntimeError('bounded critic budget exceeded')
   if k==100:
    path=R/'critic'/f'{arm}_restart.pt';state={'Q':[q.state_dict() for q in qs],'targets':[q.state_dict() for q in targets],'optimizer':opt.state_dict(),'RNG':rng_state(),'sampler_cursor':k,'configuration':cfg};tmp=path.with_suffix('.part');torch.save(state,tmp);os.replace(tmp,path);probe=step(idx);expected=[copy.deepcopy(q.state_dict()) for q in qs];expected_t=[copy.deepcopy(q.state_dict()) for q in targets];expected_opt=copy.deepcopy(opt.state_dict());next_random=torch.rand(5,device='cuda');loaded=torch.load(path,map_location='cpu',weights_only=True)
    for q,z in zip(qs,loaded['Q']):q.load_state_dict(z)
    for q,z in zip(targets,loaded['targets']):q.load_state_dict(z)
    opt.load_state_dict(loaded['optimizer']);restore_rng(loaded['RNG']);again=step(idx);err=max(float((v-expected[j][n]).abs().max()) for j,q in enumerate(qs) for n,v in q.state_dict().items());assert err==0 and torch.equal(next_random,torch.rand(5,device='cuda')) and again==probe
    targeterr=max(float((v-expected_t[j][n]).abs().max()) for j,q in enumerate(targets) for n,v in q.state_dict().items());assert targeterr==0
    momenterr=max(float((v-expected_opt['state'][j][n]).abs().max()) for j,st in opt.state_dict()['state'].items() for n,v in st.items() if torch.is_tensor(v));assert momenterr==0
    restart={'model_max_abs_difference':err,'target_max_abs_difference':targeterr,'AdamW_state_max_abs_difference':momenterr,'RNG_equal':True,'sampler_cursor_equal':loaded['sampler_cursor']==k,'checkpoint_sha256':sha(path)};loss,n=again
   else:loss,n=step(idx)
   if k in [0,99,399,799]:curve.append({'step':k+1,'batch_MSE':loss})
  torch.cuda.synchronize();elapsed=time.monotonic()-start;results={};preds={}
  with torch.no_grad():
   for sp,ix in splits.items():
    parts=[]
    for sl in range(0,len(ix),32):
     idx=torch.tensor(ix[sl:sl+32],device='cuda');parts.append(torch.stack([predict(q,idx,arm) for q in qs]).cpu().numpy())
    pred=np.concatenate(parts,1);truth=y[ix];actual=mc[ix];results[sp]={'RMSE_expected_return':np.sqrt(np.mean((pred-truth)**2,axis=1)).tolist(),'RMSE_MC_branch_mean':np.sqrt(np.mean((pred-actual.mean(1))**2,axis=1)).tolist(),'RMSE_one_MC_realization':np.sqrt(np.mean((pred-actual[:,0])**2,axis=1)).tolist(),'mean_bias':np.mean(pred-truth,axis=1).tolist(),'max_abs_error':np.max(abs(pred-truth),axis=1).tolist(),'by_length':{str(N):np.sqrt(np.mean((pred[:,[meta[i]['N']==N for i in ix]]-truth[[meta[i]['N']==N for i in ix]])**2,axis=1)).tolist() for N in sorted(set(meta[i]['N'] for i in ix))},'by_progress':{label:np.sqrt(np.mean((pred[:,[lo<=meta[i]['t']/meta[i]['N']<hi for i in ix]]-truth[[lo<=meta[i]['t']/meta[i]['N']<hi for i in ix]])**2,axis=1)).tolist() for label,lo,hi in [('early',0,.34),('middle',.34,.67),('late',.67,1.01)]},'twin_disagreement_mean':float(np.mean(abs(pred[0]-pred[1]))),'calibration_bins':[{'predicted_mean':float(pred[j][(pred[j]>=lo)&(pred[j]<hi)].mean()),'MC_return_mean':float(actual.mean(1)[(pred[j]>=lo)&(pred[j]<hi)].mean()),'count':int(((pred[j]>=lo)&(pred[j]<hi)).sum()),'twin':j+1,'lower':lo,'upper':hi} for j in range(2) for lo,hi in [(-100,.3),(.3,.5),(.5,.7),(.7,100)] if ((pred[j]>=lo)&(pred[j]<hi)).any()],'per_AA_RMSE':{AA[a]:np.sqrt(np.mean((pred[:,[meta[i]['AA']==a for i in ix]]-truth[[meta[i]['AA']==a for i in ix]])**2,axis=1)).tolist() for a in range(20) if any(meta[i]['AA']==a for i in ix)},'MC_noise_floor_RMSE':float(np.sqrt(np.mean(actual.var(1)))),'constant_TRAIN_mean_baseline_RMSE':float(np.sqrt(np.mean((truth-y[splits['TRAIN']].mean())**2))),'sufficient_statistic_analytic_baseline_RMSE':0.,'correlations':[float(np.corrcoef(v,truth)[0,1]) for v in pred]};preds[sp]={'predictions':pred.tolist(),'expected_returns':truth.tolist(),'MC_returns':actual.tolist(),'transition_indices':ix.tolist()}
  row={'arm':arm,'seconds':elapsed,'parameters_per_Q':sum(p.numel() for p in qs[0].parameters()),'metrics':results,'passes_ID_and_OOD_both_twins':max(results['IID']['RMSE_expected_return']+results['OOD']['RMSE_expected_return'])<.1,'restart':restart,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'curve':curve};rows.append(row);write(f'critic/{arm}_PREDICTIONS.json',preds);write('FULL_TRAJECTORY_CRITIC_COMPARISON.json',{'config':cfg,'rows':rows,'J1_optimizer_updates':0,'production_RL_updates':0,'real_protein_return_calibration':'not established','MC_value_regression_not_full_SAC_learning':True});print(json.dumps({'arm':arm,'seconds':elapsed,'IID':results['IID']['RMSE_expected_return'],'OOD':results['OOD']['RMSE_expected_return'],'pass':row['passes_ID_and_OOD_both_twins']}),flush=True)
if __name__=='__main__':main()
