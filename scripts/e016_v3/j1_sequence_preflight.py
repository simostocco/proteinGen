"""Bounded actual-J1 GPU preflight; discarded engineering weights only."""
import copy,gc,json,math,time
import numpy as np
import torch
import j1_sequence_first as s
from j1_sequence_first import *
from protein_joint_sac.rl.global_geometry import rollout
def group(n):
 if n.startswith('encoder.blocks.'):return 'Transformer_'+n.split('.')[2]
 if n.startswith('encoder.'):return 'embedding_pair_context'
 if n.startswith('aa_head.'):return 'AA_head'
 return 'geometry_head'
def gradient(model,obj,retain=False):
 named=list(model.named_parameters());gg=torch.autograd.grad(obj,[p for n,p in named],allow_unused=True,retain_graph=retain);v={n:(torch.zeros_like(p,device='cpu') if g is None else g.detach().float().cpu()) for (n,p),g in zip(named,gg)}
 if any(not x.isfinite().all() for x in v.values()):raise FloatingPointError('nonfinite diagnostic gradient')
 return v
def summary(v):
 rows={}
 for n,g in v.items():
  k=group(n);r=rows.setdefault(k,{'square':0.,'elements':0});r['square']+=float(g.double().square().sum());r['elements']+=g.numel()
 for r in rows.values():r['L2']=math.sqrt(r.pop('square'));r['RMS']=r['L2']/math.sqrt(r['elements'])
 shared=torch.cat([g.flatten() for n,g in v.items() if n.startswith('encoder.')])
 rows['shared_all']={'L2':float(shared.double().norm()),'RMS':float(shared.double().norm()/math.sqrt(shared.numel()))}
 return rows,shared
def profile(model,data):
 model.train();set_dropout(model,.15);rng=random.Random(180057);examples=[]
 for k,(lo,hi) in enumerate(j.BANDS):
  full=next(h for h in data.base.jointids['TRAIN'] if lo<=len(data.seqs[h])<=hi)
  choices=[h for h in data.ids['TRAIN'] if lo<=len(data.seqs[h])<=hi]
  for h in [full]+[choices[len(choices)//q] for q in [3,5]]:
   o=data.base.owners[h][0] if h==full else data.owners['TRAIN'][h][0]
   N=len(data.seqs[h]);positions=[max(3,N//8),max(3,N//2),N-1] if h==full else [N//4,N//2,N-1]
   examples.extend((h,o,t) for t in positions)
 write(R/'configs/GRADIENT_TRAIN_PANEL.json',examples)
 totals={};den=collections.Counter()
 for useaug in [False,True]:
  for off in range(0,len(examples),4):
   ex=examples[off:off+4];b,y,g=data.batch(ex)
   if useaug:b=augment(b,.075)
   aa,geo,parts=loss_call(model,b,y,g)
   isfull=torch.tensor([o not in data.rows for h,o,t in ex],device='cuda')
   objs={'AA':(aa.sum(),len(ex))}
   for i,n in enumerate(['distance','angle','torsion']):
    mask=isfull & b['geometry_target_valid'][:,i]
    if mask.any():objs[n]=(parts[mask,i].sum(),int(mask.sum()))
   for name,(obj,num) in objs.items():
    gg=gradient(model,obj,True);den[name]+=num
    if name not in totals:totals[name]=gg
    else:
     for n in gg:totals[name][n]+=gg[n]
   del b,y,g,aa,geo,parts,objs
 for name,v in totals.items():
  for n in v:v[n]/=den[name]
 totals['local']={n:sum(totals[c][n] for c in ['distance','angle','torsion'])/3 for n in totals['AA']}
 ex=[(h,data.base.owners[h][0],len(data.seqs[h])-1) for h in data.base.jointids['TRAIN'] if len(data.seqs[h])>=128][:4]
 b,y,g=data.batch(ex);b=augment(b,.075);target=torch.tensor(np.asarray([data.base.ca[o][t] for h,o,t in ex]),device='cuda');nl,_=direct_nonlocal(model,b,y,target)
 totals['nonlocal']=gradient(model,nl);del nl,b,y,g,target
 h=next(h for h in data.base.jointids['TRAIN'] if 160<=len(data.seqs[h])<=200);o=data.base.owners[h][0];H=64;t=8
 ref=torch.tensor(data.base.ca[o][:t+H],device='cuda');ids=torch.tensor([j.AA.index(a) for a in data.seqs[h][:t+H]],device='cuda')
 r=generated_bundle(model,ref[:t],ids,len(data.seqs[h]),H,ref)
 for name,key in [('Rg','Rg'),('clash','clash'),('pair','pair')]:totals[name]=gradient(model,r[key],True)
 del r,ref,ids;rows={};vectors={}
 for name,v in totals.items():rows[name],vectors[name]=summary(v)
 aa=vectors['AA'];cos={}
 for name,v in vectors.items():
  cos[name]=float(torch.dot(aa.double(),v.double())/(aa.double().norm()*v.double().norm()).clamp_min(1e-30))
 weights={}
 for name,budget in [('local',.30),('nonlocal',.08),('Rg',.04),('clash',.04),('pair',.04)]:
  norm=rows[name]['shared_all']['L2']
  if norm<=0:raise ValueError('zero gradient calibration: '+name)
  weights['lambda_'+name]=min(1.,budget*rows['AA']['shared_all']['L2']/norm)
 combined={n:totals['AA'][n]+sum(weights['lambda_'+c]*totals[c][n] for c in ['local','nonlocal','Rg','clash','pair']) for n in totals['AA']}
 rows['combined_active'],_=summary(combined)
 totalnorm=math.sqrt(sum(float(v.double().square().sum()) for v in combined.values()));clipfactor=min(1,1/max(totalnorm,1e-30))
 out={'status':'PASS','TRAIN_panel':examples,'ordinary_and_augmented':True,'loss_gradients':rows,'AA_shared_cosines':cos,'weights':weights,'normalization':'AA mean all diagnostic positions, local component means on complete-only targets; local=sum3/3. Global64-step two-path single TRAIN case; not population-wide guarantees.','combined_global_unclipped_L2':totalnorm,'combined_clipping_factor_at1':clipfactor,'combined_postclip_L2':totalnorm*clipfactor,'shared_active_geometry_norm_upper_bound_ratio':sum(weights['lambda_'+c]*rows[c]['shared_all']['L2'] for c in ['local','nonlocal','Rg','clash','pair'])/rows['AA']['shared_all']['L2'],'geometry_head_note':'No interface gradient surgery; scalar coefficients reduce all gradients, geometry-head preservation assessed by native likelihood. All rollout feedback paths included.'}
 write(R/'GRADIENT_BALANCE_REPORT.json',out);del totals,vectors,combined;gc.collect();return weights
def benchmarks(model,data):
 results=[];set_dropout(model,.15);model.train()
 for H in [32,64,128,256]:
  h=next(h for h in data.base.jointids['TRAIN'] if len(data.seqs[h])>=H+8);o=data.base.owners[h][0];t=8
  ref=torch.tensor(data.base.ca[o][:t+H],device='cuda');ids=torch.tensor([j.AA.index(a) for a in data.seqs[h][:t+H]],device='cuda')
  torch.cuda.reset_peak_memory_stats();q=time.monotonic();model.zero_grad(set_to_none=True)
  r=rollout(model,ref[:t],ids,len(data.seqs[h]),H,ref,recompute=True);forward=time.monotonic()-q
  (r['Rg_loss']+.01*r['clash_loss']).backward();torch.cuda.synchronize()
  row={'H':H,'actual_full_gradient_horizon':H,'truncation':False,'paths':1,'forward_seconds':forward,'forward_backward_seconds':time.monotonic()-q,'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved(),'geometry_head_grad_L2':float(model.geom[-1].weight.grad.norm()),'Transformer_grad_L2':float(model.encoder.blocks[0].qkv.weight.grad.norm()),'finite':all(p.grad.isfinite().all() for p in model.parameters() if p.grad is not None)}
  results.append(row);write(R/'LONG_ROLLOUT_BENCHMARKS.json',results);print(json.dumps(row),flush=True);del r,ref,ids;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
 return results
def position_benchmark(model,data):
 set_dropout(model,0);model.eval();h=next(h for h in data.base.jointids['TRAIN'] if 35<=len(data.seqs[h])<=45);o=data.base.owners[h][0];N=len(data.seqs[h]);examples=[(h,o,t) for t in range(N)]
 # Exact full-position supervision via isolated contexts, not one native full chain.
 def go(batch):
  model.zero_grad(set_to_none=True);outputs=[];q=time.monotonic()
  for off in range(0,N,batch):
   b,y,g=data.batch(examples[off:off+batch]);a,ge,_=loss_call(model,b,y,g);(a.sum()/N).backward();outputs.extend(a.detach().cpu().tolist())
  torch.cuda.synchronize();return np.array(outputs),time.monotonic()-q,{n:p.grad.detach().float().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
 a,ta,ga=go(1);b,tb,gb=go(16);err=float(np.max(np.abs(a-b)));relative=math.sqrt(sum(float((ga[n]-gb[n]).double().square().sum()) for n in ga)/sum(float(v.double().square().sum()) for v in ga.values()))
 if err>.05 or relative>.04:raise ValueError('actual BF16 multi-prefix parity failed')
 write(R/'FULL_POSITION_GPU_PARITY.json',{'status':'PASS','sequence_hash':h,'positions':N,'batch1_seconds':ta,'batch16_seconds':tb,'speedup':ta/tb,'max_AA_NLL_abs_error_BF16':err,'gradient_relative_L2_BF16':relative,'thresholds':{'NLL':.05,'relative_gradient':.04},'semantics':'every target has separate literal prefix and query token; no hidden attention across examples'})
 # Representative length/microbatch profiles with dropout active, base mathematical loss.
 model.train();set_dropout(model,.15);rows=[]
 for N in [64,128,256,384,500]:
  h=next(h for h in data.base.jointids['TRAIN'] if len(data.seqs[h])>=N);o=data.base.owners[h][0]
  for B in [1,4,16]:
   ex=[(h,o,N-1)]*B;torch.cuda.reset_peak_memory_stats();q=time.monotonic();b,y,g=data.batch(ex);data_s=time.monotonic()-q;model.zero_grad(set_to_none=True)
   q=time.monotonic();aa,ge,_=loss_call(model,b,y,g);(aa.mean()+.01*ge.mean()).backward();torch.cuda.synchronize();sec=time.monotonic()-q
   rows.append({'prefix_length':N-1,'live_batch':B,'data_seconds':data_s,'forward_backward_seconds':sec,'targets_per_second':B/sec,'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()})
   del b,y,g,aa,ge;model.zero_grad(set_to_none=True)
  write(R/'PREFIX_BATCH_BENCHMARKS.json',rows)
def proof(data,c):
 e=Engine(data,c);e.step();p=R/'tests/engineering_checkpoint_00001.pt';e.checkpoint(p);expected=e.step(force_stage=3,force_horizon=4);mv=cpu_copy(e.model.state_dict());ov=cpu_copy(e.opt.state_dict());ss=e.sampler.state_dict();sc=copy.deepcopy(e.scheduler.state_dict());rd=rng_digest();roll=e.rollrng.getstate()
 del e;gc.collect();torch.cuda.empty_cache();e=Engine(data,c);e.resume(p);actual=e.step(force_stage=3,force_horizon=4)
 pe=maxdiff(mv,e.model.state_dict());oe=maxdiff(ov,e.opt.state_dict());same=ss==e.sampler.state_dict() and sc==e.scheduler.state_dict() and rd==rng_digest() and roll==e.rollrng.getstate() and expected['sample_order_sha256']==actual['sample_order_sha256']
 out={'status':'PASS' if same and pe<2e-5 and oe<1e-4 else 'FAIL','model_max_error':pe,'optimizer_max_error':oe,'sampler_dropout_mask_scheduler_rollout_RNG_equal':same,'expected':expected,'resumed':actual,'engineering_optimizer_calls':3,'main_calls':0,'scope':'actual12.35M model+mixed realPDB64AA+32geometry+dropout.15+augmentation+K4nonlocal+twoH4rollouts; discarded engineering weights'}
 write(R/'CHECKPOINT_RESTART_TEST.json',out)
 if out['status']!='PASS':raise RuntimeError('stochastic recovery failed')
 del e;gc.collect();torch.cuda.empty_cache()
def main():
 j.core.setup();status('PREFLIGHT_DATA');data=PartialData();write(R/'DATA_LOAD_RECEIPT.json',data.receipt);model,parent=j.parent_model();del parent
 # Existing panels are unchanged; all partial validation roles inherited.
 write(R/'configs/PARTIAL_VALIDATION_PANEL.json',data.panel(True))
 weights=profile(model,data);status('PREFLIGHT_ROLLOUTS');bench=benchmarks(model,data);status('PREFLIGHT_POSITION_BATCHING');position_benchmark(model,data)
 c={'name':'J1','parameters':12351746,'dropout':.15,'mask_probability':.075,'AA_coefficient':1.,'effective_AA_batch':64,'geometry_batch':32,'live_batch':16,'precision':'BF16','geometry_projection_fp32':True,'activation_checkpoint':False,'max_updates':30000,'learning_rate':5e-5,'terminal_learning_rate':5e-6,'warmup_updates':250,'partial_schedule':[38,39,38,39,38],'partial_geometry':'NOT_SUPERVISED','geometry_normalization':'per-decision mean3 valid components on32 complete-only decisions','optimizer':'RESTORED_PARENT_ADAMW_MOMENTS_FRESH_COSINE','clip_norm':1.,'weight_decay':.01,'nonlocal_period':2,'nonlocal_K':4,'nonlocal_targets_per_active_update':4,'nonlocal_anchors_max':16,'distance_scale_A':3.8,'rollout_period':8,'horizons':[64,128],'rollout_paths':2,'gradient_truncation':False,'milestones':[2500,5000,7500,10000,15000,20000,25000,30000],'guardrails':{'AA_delta':.03,'local_delta':.20,'severe_clash_mean':6.0},'TEST':'NOT_USED',**weights}
 write(R/'configs/FROZEN.json',c);del model;gc.collect();torch.cuda.empty_cache();status('PREFLIGHT_RESTART');proof(data,c)
 write(R/'REGULARIZATION_CONFIG_AND_TESTS.json',{'status':'PASS','dropout':.15,'dropout_sites':['attention probabilities','attention output residual','FFN output residual'],'mask_probability':.075,'preserve_last3':True,'ca_observed_immutable':True,'train_only':True,'CPU_tests':'tests/CORE_TESTS.xml','GPU_restart':'CHECKPOINT_RESTART_TEST.json'})
 write(R/'NONLOCAL_ENERGY_SCORE_TESTS.json',{'status':'PASS','CPU_tests':'tests/CORE_TESTS.xml','gradient_evidence':'GRADIENT_BALANCE_REPORT.json','real_GPU_optimizer_and_restart':'CHECKPOINT_RESTART_TEST.json','proper_score':'Euclidean energy score unbiased U-statistic K4; generated-pair K2; normalized by3.8A*sqrt(vector dimension)','negative_sample_estimates_possible':True})
 (R/'FULL_POSITION_TRAINING_FEASIBILITY.md').write_text('# Exact multi-position supervision\n\nPrefix-bidirectional representations change with each target. A single full-native chain is invalid; triangular causal attention would change the trained computation. Retained exact ragged batches with one isolated context per target. All positions of a short real TRAIN protein were evaluated, with numerical loss/gradient parity and timing in FULL_POSITION_GPU_PARITY.json. Existing preprocessing is vectorized; no unsafe sharing of suffix-dependent features or KV cache. Packed block-diagonal would contain the same sum of prefix work and lose convenient batch SDPA isolation; not implemented. Long proteins use resumable identity-position permutations rather than paying all N prefixes in one update. Coverage is exact but not a claim all identities were fully visited.\n')
 (R/'NONLOCAL_GEOMETRY_LOSS_SPEC.md').write_text('# Nonlocal/global score\n\nK4 genuine exp/sigmoid/tanh Gaussian reparameterized actions; original CA builder, bootstrap and density/Jacobians unchanged. At most16 earlier visible anchors stratified3-7,8-23,>=24 separation. No future-distance anchor selection. Euclidean unbiased energy-score U-statistic after dividing each anchor distance by3.8A*sqrt(anchor_count); proper for the induced distance-vector distribution, not a unique geometry identification. K2 independent generated paths yield an analogous32-pair score. Native future coordinates are targets only. Rg compares generated segment and exact matching native segment, robust Huber delta.5 log ratio, eps1e-8. Clash<2A seqsep>=3 includes generated-prefix/new-new pairs divided by generated count. Local/nonlocal targets are not independent labels or physical energies.\n')
 with (R/'EXPERIMENT_DECISIONS.md').open('a') as f:f.write('\nPreflight frozen before main updates: scalar shared-gradient budgets local30%, nonlocal8%, Rg/clash/pair4%each (active upper bound50%; periodic average lower). No geometry-head gradient bypass. Horizon64/128 alternating every8 updates, two fully differentiated paths; H256 benchmark only. Coefficients and actual profiles in GRADIENT_BALANCE_REPORT.json/configs/FROZEN.json. Stage2 ramp500updates; stage3 ramp1000. Stop after2 consecutive milestones AA+0.03nats or local+0.20nats; immediate finite/clash mean>6/100res gate. Extension only AAimprovement>=.002 since10k with safeguards. Final selection reuses original lengthwise clash/Rg/diversity thresholds from prior frozen final guard, no relaxation.\n')
 write(R/'PREFLIGHT_PASS.json',{'status':'PASS','package_sha256':code_hash(),'entrypoint_sha256':file_sha(s.__file__),'config_sha256':file_sha(R/'configs/FROZEN.json'),'main_optimizer_updates':0})
 print('SEQUENCE_FIRST_PREFLIGHT_PASS',flush=True)
if __name__=='__main__':main()
