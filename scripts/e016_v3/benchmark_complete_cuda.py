"""One isolated complete exact-hybrid SAC synthetic update at a declared length."""
import argparse,os,time,json,resource,traceback,sys,gc
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--length',type=int,required=True);args=p.parse_args();N=args.length
assert N in [64,128,256,500]
R=Path(os.environ['E016_ARTIFACT_ROOT']);out=R/'benchmarks'/(f'N{N}_run'+os.environ.get('E016_BENCH_RUN_ID','001'));out.mkdir(exist_ok=False)
import torch
from protein_joint_sac.contracts import BuildAction,BuildState,BuildEpisode,GeneratedProtein,RewardCard,AA
from protein_joint_sac.geometry.ca_builder import replay,mask_for_step,append
from protein_joint_sac.geometry.local_frames import features,reuse_static_prefix
from protein_joint_sac.geometry.dense_pair_bias import DensePairBias
from protein_joint_sac.models.dense_geometry_attention import ModelConfig
from protein_joint_sac.models.joint_actor import JointActor
from protein_joint_sac.models.build_critic import TwinCritics,reuse_frozen_current
from protein_joint_sac.rl.hybrid_sac_build import HybridSAC
from protein_joint_sac.rl.delayed_episode_replay import EpisodeReplay
from protein_joint_sac.rl.checkpointing import save_atomic,load_checked
from benchmark_n64_checks import gradient_check

torch.set_num_threads(1);torch.manual_seed(1516);torch.cuda.manual_seed_all(1516)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
(out/'benchmarks').mkdir()
rows=[];started=time.monotonic();stage='initialize'
def timed(name,fn):
 global stage
 stage=name;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.monotonic()
 value=fn();torch.cuda.synchronize()
 row={'operation':name,'seconds':time.monotonic()-t,'CUDA_allocated_peak':torch.cuda.max_memory_allocated(),'CUDA_reserved_peak':torch.cuda.max_memory_reserved(),'CPU_RSS_highwater':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
 rows.append(row)
 with (out/'operations.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
 print(N,json.dumps(row),flush=True);return value

def finite_model(m):
 for name,v in m.named_parameters():
  assert torch.isfinite(v).all(),name
  if v.grad is not None:assert torch.isfinite(v.grad).all(),name+' gradient'
try:
 assert torch.cuda.is_available()
 actions=tuple(BuildAction(AA[i%20],torch.tensor([3.8 if i else 0.,1.9 if i>=2 else 0.,.4 if i>=3 else 0.],device='cuda',dtype=torch.float64),mask_for_step(i)) for i in range(N))
 state=timed('prefix_replay',lambda:replay(N,actions[:-1],device='cuda'));previous=replay(N,actions[:-2],device='cuda')
 actor=JointActor(ModelConfig(width=128,heads=4,layers=4,backend='manual')).cuda();q=TwinCritics(ModelConfig(width=96,heads=4,layers=3,backend='manual')).cuda();alg=HybridSAC(actor,q)
 params={'actor':sum(p.numel() for p in actor.parameters()),'Q_online':sum(p.numel() for p in list(q.q1.parameters())+list(q.q2.parameters())),'Q_targets':sum(p.numel() for p in list(q.target1.parameters())+list(q.target2.parameters())),'temperatures':sum(p.numel() for p in alg.temperatures.parameters())}
 assert next(q.q1.parameters()).data_ptr()!=next(q.q2.parameters()).data_ptr()
 f=timed('geometry_and_14feature_reference_outputs',lambda:features(state))
 assert all(v.device.type=='cuda' for v in f.values() if isinstance(v,torch.Tensor))
 biasnet=DensePairBias(4).cuda()
 with torch.no_grad():timed('dense_geometric_chemical_pair_bias',lambda:biasnet({k:v.float() if isinstance(v,torch.Tensor) and v.is_floating_point() else v for k,v in f.items()}))
 del f,biasnet
 h=timed('actor_forward',lambda:actor.encode(state));logp=torch.log_softmax(actor.aa_head(h),-1);assert torch.isfinite(logp).all() and len(logp)==20
 residue=AA[int(torch.distributions.Categorical(logits=logp).sample())]
 dist=actor.geometry_distribution(state,residue,h);g,lg=timed('conditional_geometry_rsample_logprob',lambda:dist.rsample());assert torch.isfinite(g).all() and torch.isfinite(lg)
 discrepancy=float((dist.log_prob(g)-lg).abs().detach());assert discrepancy<.005,'transformed log-density roundtrip'
 act=BuildAction(residue,g,mask_for_step(state.t));candidate=timed('differentiable_candidate_reconstruction',lambda:append(state,act))
 grad=torch.autograd.grad(candidate.ca[-1].square().sum(),g,retain_graph=True)[0];assert torch.isfinite(grad).all() and grad.abs().sum()>0
 with torch.no_grad():timed('independent_Q_forward',lambda:(q.q1(state,act),q.q2(state,act)))
 del h,logp,g,lg,act,candidate,dist,grad;gc.collect()
 if N==64:
  timed('eager_streamed_exact_gradient_and_FD_equivalence',lambda:gradient_check(state,actor,q,alg,out))
  actor.zero_grad(set_to_none=True);q.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
  # Same parameter shapes and dense bias; explicitly compare SDPA execution, no backend switch.
  import copy
  sdpa=copy.deepcopy(actor)
  for block in sdpa.encoder.blocks if hasattr(sdpa,'encoder') else []:pass
  for module in sdpa.modules():
   if hasattr(module,'c'):
    from dataclasses import replace
    module.c=replace(module.c,backend='sdpa')
  with torch.no_grad():torch.testing.assert_close(sdpa.logits(state),actor.logits(state),atol=1e-5,rtol=1e-5)
  del sdpa
 def exact_target():
  with torch.no_grad(),reuse_static_prefix(state),reuse_frozen_current(state):
   return state.ca.new_tensor(.3)+.99*alg.soft_value(state,targets=True,estimator='exact')
 target=timed('exact20_nonterminal_soft_Bellman_target',exact_target)
 assert torch.isfinite(target)
 def critic_step():
  alg.critic_optimizer.zero_grad(set_to_none=True)
  a=actions[-2]
  with reuse_static_prefix(previous):loss=(q.q1(previous,a)-target).square()+(q.q2(previous,a)-target).square()
  assert torch.isfinite(loss);loss.backward();finite_model(q);alg.critic_optimizer.step();return float(loss.detach())
 critic_loss=timed('critic_loss_backward_optimizer_first_step',critic_step)
 actor.zero_grad(set_to_none=True)
 actor_loss=timed('exact20_streamed_actor_backward',lambda:alg.backward_actor_exact_streamed(state));finite_model(actor)
 timed('actor_optimizer_first_step',alg.actor_optimizer.step)
 def temperature_step():
  _,la,lg=actor.sample(state);loss=alg.temperatures.loss(la,lg,True)
  alg.alpha_optimizer.zero_grad(set_to_none=True);loss.backward();alg.alpha_optimizer.step();return float(loss.detach())
 temp_loss=timed('separate_entropy_temperature_updates',temperature_step)
 old=[p.detach().clone() for p in q.target1.parameters()]
 timed('target_Polyak_update',lambda:q.polyak(.005))
 for before,online,new in zip(old,q.q1.parameters(),q.target1.parameters()):torch.testing.assert_close(new,before.lerp(online.detach(),.005),atol=0,rtol=0)
 del old
 finite_model(actor);finite_model(q);finite_model(alg.temperatures)
 proposal=timed('complete_replay_GPU',lambda:GeneratedProtein(replay(N,actions,device='cuda')))
 cpu=timed('complete_replay_CPU',lambda:replay(N,actions,device='cpu'))
 replay_error=float((proposal.state.ca.cpu()-cpu.ca).abs().max()) if hasattr(proposal,'state') else None
 # GeneratedProtein field is inspected explicitly; do not infer a new coordinate identity from CPU roundoff.
 xgpu=replay(N,actions,device='cuda').ca
 torch.testing.assert_close(xgpu.cpu(),cpu.ca,atol=1e-8,rtol=1e-8)
 replay_error=float((xgpu.cpu()-cpu.ca).abs().max())
 db=EpisodeReplay(out/'replay.sqlite');episode=BuildEpisode(f'synthetic-N{N}',N,actions);db.store(episode,proposal)
 reward=RewardCard('synthetic-only',proposal.sequence_sha256,proposal.geometry_sha256,'a'*64,'MOCK_UNIT_TEST',{'value':1.0})
 db.attach(episode.episode_id,reward);assert db.resolved_ids()==[episode.episode_id];db.close()
 checkpoint=out/'synthetic_step.pt'
 payload={'actor':actor.state_dict(),'critics':q.state_dict(),'actor_optimizer':alg.actor_optimizer.state_dict(),'critic_optimizer':alg.critic_optimizer.state_dict(),'temperature':alg.temperatures.state_dict(),'alpha_optimizer':alg.alpha_optimizer.state_dict(),'torch_rng':torch.get_rng_state(),'CUDA_rng':torch.cuda.get_rng_state(),'N':N,'scope':'SYNTHETIC_BENCHMARK_ONLY'}
 timed('checkpoint_save',lambda:save_atomic(checkpoint,payload));loaded=timed('checkpoint_CPU_reload',lambda:load_checked(checkpoint))
 restored_actor=JointActor(ModelConfig(width=128,heads=4,layers=4)).cuda();restored_q=TwinCritics(ModelConfig(width=96,heads=4,layers=3)).cuda();restored=HybridSAC(restored_actor,restored_q)
 restored_actor.load_state_dict(loaded['actor'],strict=True);restored_q.load_state_dict(loaded['critics'],strict=True);restored.actor_optimizer.load_state_dict(loaded['actor_optimizer']);restored.critic_optimizer.load_state_dict(loaded['critic_optimizer']);restored.temperatures.load_state_dict(loaded['temperature']);restored.alpha_optimizer.load_state_dict(loaded['alpha_optimizer'])
 with torch.no_grad():torch.testing.assert_close(restored_actor.logits(state),actor.logits(state),atol=0,rtol=0)
 for opt in [restored.actor_optimizer,restored.critic_optimizer,restored.alpha_optimizer]:
  for par,values in opt.state.items():
   for k,v in values.items():
    if isinstance(v,torch.Tensor) and k!='step':assert v.device==par.device
 del restored,restored_q,restored_actor,loaded,payload;gc.collect()
 def generation():
  s=BuildState.empty(N,device='cuda');trace=[]
  with torch.no_grad():
   for _ in range(N):
    a,la,lg=actor.sample(s);assert torch.isfinite(la) and torch.isfinite(lg)
    trace.append(a);s=append(s,a)
  assert s.t==N and torch.isfinite(s.ca).all()
  return s
 generated=timed('full_stochastic_generation_20AA_joint_geometry',generation)
 generation_time=rows[-1]['seconds']
 result={'status':'PASS_COMPLETE_SYNTHETIC_SAC_STEP','N':N,'params':params,'model_config':{'actor':[128,4,4],'Q':[96,4,3],'backend':'manual','model_dtype':'FP32','geometry_dtype':'FP64','mixed_precision':False},'optimizer_steps':{'actor':1,'Q_twins':1,'temperatures':1},'actor_loss':actor_loss,'critic_loss':critic_loss,'temperature_loss':temp_loss,'replay_max_error_A':replay_error,'log_density_roundtrip_error':discrepancy,'checkpoint_bytes':checkpoint.stat().st_size,'generation_residues_per_second':N/generation_time,'full_generated_length':generated.t,'full_step_seconds':sum(x['seconds'] for x in rows if x['operation'] in ['exact20_nonterminal_soft_Bellman_target','critic_loss_backward_optimizer_first_step','exact20_streamed_actor_backward','actor_optimizer_first_step','separate_entropy_temperature_updates','target_Polyak_update']),'elapsed_seconds':time.monotonic()-started,'rows':rows,'AF_calls':0,'production_training':False}
 (out/'RESULT.json').write_text(json.dumps(result,indent=2));print('PASS_N',N,flush=True)
except BaseException as exc:
 (out/'FAILURE.json').write_text(json.dumps({'status':'FAILED','stage':stage,'error':repr(exc),'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-started,'RSS_highwater':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'CUDA_allocated_peak':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,'CUDA_reserved_peak':torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None,'rows':rows},indent=2));raise
