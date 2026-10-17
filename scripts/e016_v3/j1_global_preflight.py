"""Bounded real TRAIN gradient/resource preflight; no main training and no TEST."""
import json,math,random,time,resource
import numpy as np
import torch
import j1_global as j
from protein_joint_sac.rl.global_geometry import rollout,VERSION
from protein_joint_sac.models.batched_bc import features_batch
from protein_joint_sac.rl.bc_training import rng_state,restore_rng
from protein_joint_sac.data.certified_unified import file_sha
def gradnorm(loss,model,retain=True):
 gs=torch.autograd.grad(loss,list(model.parameters()),allow_unused=True,retain_graph=retain)
 if any(not torch.isfinite(g).all() for g in gs if g is not None):raise FloatingPointError("nonfinite global gradient")
 total=math.sqrt(sum(float(g.double().square().sum()) for g in gs if g is not None))
 head=math.sqrt(sum(float(g.double().square().sum()) for (n,p),g in zip(model.named_parameters(),gs) if g is not None and n.startswith("geom.")))
 upstream=math.sqrt(sum(float(g.double().square().sum()) for (n,p),g in zip(model.named_parameters(),gs) if g is not None and n.startswith("encoder.blocks.")))
 return {"total":total,"geometry_head":head,"transformer":upstream}
def run():
 j.core.setup();j.guard()
 if not (j.R/'SEQUENCE_BASELINES.json').exists():raise ValueError("baseline first")
 report={};data=j.Data(True,report);model,parent=j.parent_model();del parent;model.train()
 # Freeze TRAIN-only contact scale diagnostics before selecting auxiliary weights.
 rng=random.Random(72042);ids=rng.sample(data.jointids['TRAIN'],512);mins=[];close=0;pairs=0
 for h in ids:
  x=data.ca[data.owners[h][0]];d=np.linalg.norm(x[:,None]-x[None,:],axis=-1);sep=np.arange(len(x))[None]-np.arange(len(x))[:,None];v=d[sep>=3];mins.append(float(v.min()));close+=int((v<2).sum());pairs+=len(v)
 scale={"TRAIN_identities":len(ids),"membership":j.digest(sorted(ids)),"native_nonlocal_min_distance_quantiles":np.quantile(mins,[0,.01,.05,.5,1]).tolist(),"native_pairs_below2A":close,"eligible_pairs":pairs,"chosen_repulsion_radius_A":2.,"reason":"existing severe-CA-overlap metric; TRAIN nonlocal minimum-distance distribution documents native contact separation; no universal atomic validity claim","separation":3}
 j.write(j.R/'configs/TRAIN_CLASH_SCALE.json',scale)
 ex=j.Sampler(data,seed=740).draw(64);ordered=sorted(ex,key=lambda e:e[2]);a=time.monotonic();model.zero_grad(set_to_none=True)
 for i in range(0,64,16):
  b,y,g=j.core.make(data,ordered[i:i+16],True);loss,*_=j.core.loss_call(model,b,y,g,j.CFG);(loss.sum()/64).backward()
 torch.cuda.synchronize();base=math.sqrt(sum(float(p.grad.double().square().sum()) for p in model.parameters() if p.grad is not None));base_seconds=time.monotonic()-a;model.zero_grad(set_to_none=True)
 profiles=[]
 specs=[(16,8),(32,8),(64,16),(128,16),(16,384),(32,256)]
 for index,(H,start) in enumerate(specs):
  candidates=[h for h in data.jointids['TRAIN'] if len(data.seqs[h])>=start+H];h=candidates[index%len(candidates)];o=data.owners[h][0];N=len(data.seqs[h]);reference=torch.tensor(data.ca[o][:start+H],device='cuda');seq=torch.tensor([j.AA.index(v) for v in data.seqs[h][:start+H]],device='cuda')
  torch.manual_seed(742+index);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();a=time.monotonic()
  r=rollout(model,reference[:start],seq,N,H,reference,recompute=True);torch.cuda.synchronize();forward=time.monotonic()-a;a=time.monotonic()
  rg=gradnorm(r['Rg_loss'],model);clash=gradnorm(r['clash_loss'],model,False);torch.cuda.synchronize();back=time.monotonic()-a
  row={"horizon":H,"start_prefix":start,"N":N,"sequence_hash":h,"observation":o,"Rg_loss":float(r['Rg_loss']),"clash_loss":float(r['clash_loss']),"Rg_gradient":rg,"clash_gradient":clash,"forward_seconds":forward,"two_gradient_backwards_seconds":back,"generated_residues_per_second":H/(forward+back),"resources":j.guard(),"status":"PASS"}
  if rg['geometry_head']<=0 or rg['transformer']<=0:raise ValueError("global gradient does not reach geometry head and transformer")
  profiles.append(row);j.write(j.R/'tests/GLOBAL_GPU_PROFILE.json',profiles);print(json.dumps(row),flush=True);del r,reference,seq;model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
 # Actual sampled clashes may be zero in an individual rollout; do not fabricate repulsion gradients.
 positive=[p['clash_gradient']['total'] for p in profiles if p['clash_gradient']['total']>0]
 if not positive:raise ValueError("no actual sampled overlap found for steric gradient calibration; bounded diagnostic needs expansion")
 rgmedian=float(np.median([p['Rg_gradient']['total'] for p in profiles]));clmedian=float(np.median(positive))
 weights={"Rg":min(1.,.10*base/max(rgmedian,1e-20)),"clash":min(1.,.10*base/max(clmedian,1e-20))}
 baseline=json.load(open(j.R/'SEQUENCE_BASELINES.json'))['J1']['VALIDATION']
 config={"version":VERSION,"name":"J1","parameters":12351746,"parent_sha256":j.PARENT_SHA,"base_total_updates":30000,"max_updates":30000,"effective_batch":64,"live_batch":16,"precision":"BF16","geometry_projection_fp32":True,"activation_checkpoint":False,"learning_rate":5e-5,"terminal_learning_rate":5e-6,"warmup_updates":250,"weight_decay":.01,"clip_norm":1.,"geometry_weight":1.,"geometry_normalization":"unchanged per-decision mean of valid distance/angle/torsion NLL; average over64 ordinary decisions","partial_decisions_per_batch":8,"partial_objective":"AA only; input coordinate masks remain active; all partial geometry target masks disabled explicitly","rollout_period":16,"rollout_horizon":32,"rollout_batch":1,"rollout_weight_ramp_updates":1000,"rollout_activation_recompute":True,"lambda_Rg":weights['Rg'],"lambda_clash":weights['clash'],"clash_radius_A":2.,"sampling_seed":42,"rollout_seed":720,"optimizer":"RESTORE_PARENT_ADAMW_MOMENTS_RESET_SCHEDULE_ONLY","milestones":[250,1000,2500,5000,10000,15000,20000,25000,30000],"selection":"lowest VALIDATION AA NLL among geometry-guarded checkpoints; parent remains fallback; report Pareto tradeoffs","guardrails":{"AA_NLL_max":baseline['AA_NLL']+.015,"local_geometry_max":baseline['geometry_NLL']+.10,"emergency_AA_NLL_delta":.05,"emergency_local_geometry_delta":.30,"severe_clash_ratio":2.,"severe_clash_additive_per100":1.,"extension_after20000":"continue only if AA NLL improved >=.001 since10000 or common-seed Rg error or clash rate improved >=10percent without AA/local violations"},"TEST":"SEALED_NOT_EVALUATED","checkpoint_retention":"new-lineage best-eligible AA, best geometry-safe candidate, latest only; parent immutable external fallback"}
 j.write(j.R/'configs/CONTINUATION_FROZEN.json',config)
 receipt={"status":"PASS","base_gradient_L2":base,"base_forward_backward_seconds":base_seconds,"gradient_coefficient_rule":"each auxiliary median L2 at most10percent of ordinary batch L2 at a global update; additionally cap coefficient1; ramp1000; period16 dilutes average contribution","weights":weights,"profiles":profiles,"TRAIN_scale":scale,"config_sha256":file_sha(j.R/'configs/CONTINUATION_FROZEN.json'),"baseline_sha256":file_sha(j.R/'SEQUENCE_BASELINES.json'),"data":report,"main_updates_executed":0}
 j.write(j.R/'GLOBAL_LOSS_GRADIENT_TESTS.json',receipt);print("GLOBAL_GPU_PREFLIGHT_PASS",flush=True)
if __name__=="__main__":run()
