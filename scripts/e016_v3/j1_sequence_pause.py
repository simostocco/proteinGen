"""User-requested early-pause evaluation and read-only scaling audit. NO optimizer steps."""
import argparse,collections,gc,json,math,os,random,shutil,subprocess,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import j1_sequence_first as s
from j1_sequence_first import R,j,write,load_checked,file_sha,PartialData,set_dropout
from j1_sequence_assess import generation_for,partial_eval,check_candidate
from j1_global_finalize import metric_summary,paired_ci
from protein_joint_sac.models.batched_bc import features_batch,rope
def read(p):return json.load(open(p))
def destination(p):
 if p.exists():raise RuntimeError("preserve existing evidence: "+str(p))
 return p
def traces():
 rows={}
 for p in sorted((R/'runs/J1').glob('updates_from*.jsonl')):
  for line in p.read_text().splitlines():
   if line.strip():
    v=json.loads(line);rows[v['additional_update']]=v
 return [rows[u] for u in sorted(rows)]
def describe(x):
 x=np.asarray(x,float);total=len(x);nonfinite=int((~np.isfinite(x)).sum());x=x[np.isfinite(x)]
 if not len(x):return {'n':0}
 q=np.quantile(x,[0,.001,.01,.05,.25,.5,.75,.95,.99,.999,1])
 return {'n':len(x),'n_total':total,'nonfinite_count':nonfinite,'mean':float(x.mean()),'std':float(x.std()),'median':float(q[5]),'IQR':float(q[6]-q[4]),'percentiles':dict(zip(['min','p0.1','p1','p5','p25','p50','p75','p95','p99','p99.9','max'],q.tolist())),'zero_fraction':float(np.mean(x==0))}
def validate_counters(st,done,receipt):
 u=st['additional_update']
 if u!=done['update'] or not (receipt['verified_received_update']<=u<=receipt['maximum_additional_update']):raise ValueError('user ceiling/counter mismatch')
 steps={int(float(v['step'])) for v in st['optimizer']['state'].values()}
 if steps!={30000+u} or st['scheduler']['last_epoch']!=u or st['sampler']['cursor']!=u:raise ValueError('optimizer/scheduler/sampler counter mismatch')
 return u,steps
def verification():
 receipt=read(R/'BUDGET_REDUCTION_RECEIPT.json');done=read(R/'runs/J1/STOPPED.json')
 if (R/'runs/J1/COMPLETED.json').exists():
  end=read(R/'runs/J1/COMPLETED.json');done={'update':end['additional_updates'],'checkpoint':end['latest_checkpoint'],'reason':end['stop_reason']}
 cp=Path(done['checkpoint']);st=load_checked(cp);u,steps=validate_counters(st,done,receipt)
 if u!=done['update'] or not (receipt['verified_received_update']<=u<=receipt['maximum_additional_update']):raise ValueError('user ceiling/counter mismatch')
 required=['model','optimizer','scheduler','sampler','rng','rollout_rng','history','lineage','residues','nonlocal_targets','elapsed','additional_update']
 if any(k not in st for k in required):raise ValueError('incomplete recovery payload')
 steps={int(float(v['step'])) for v in st['optimizer']['state'].values()}
 if steps!={30000+u} or st['scheduler']['last_epoch']!=u or st['sampler']['cursor']!=u:raise ValueError('optimizer/scheduler/sampler counter mismatch')
 finite=all(torch.isfinite(v).all().item() for v in st['model'].values() if torch.is_tensor(v))
 finiteopt=all(torch.isfinite(v).all().item() for q in st['optimizer']['state'].values() for v in q.values() if torch.is_tensor(v))
 if not finite or not finiteopt:raise ValueError('nonfinite saved state')
 if st['lineage']['config']!=read(R/'configs/FROZEN.json'):raise ValueError('configuration changed')
 if st['lineage']['entrypoint_sha256']!=file_sha(s.__file__):raise ValueError('active source changed')
 if st['lineage']['config']['dropout']!=.15:raise ValueError('dropout changed')
 p=R/'PAUSED_CHECKPOINT_VERIFICATION.json'
 result={'status':'PASS_LOAD_CHECKSUM_COUNTERS_FINITE_STATE','checkpoint':str(cp),'sha256':file_sha(cp),'bytes':cp.stat().st_size,'additional_update':u,'total_optimizer_updates':30000+u,'received_verified_update':receipt['verified_received_update'],'further_optimizer_updates':u-receipt['verified_received_update'],'maximum_further_updates':2500,'optimizer_steps':sorted(steps),'scheduler_last_epoch':st['scheduler']['last_epoch'],'sampler_cursor':st['sampler']['cursor'],'recoverable_fields':required,'rng_keys':list(st['rng']),'lineage':st['lineage'],'source_reporting_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip(),'elapsed_main_seconds_including_evaluation':st['elapsed'],'dropout_augmentation_RNG':'global torch CPU/CUDA RNG; sampler Python RNG and independent rollout RNG saved','test':'NO_NEW_OPTIMIZER_UPDATES; prior deterministic interruption proofs retained','reason':'USER_REQUESTED_BUDGET_REDUCTION','timestamp':time.time()}
 result['training_heartbeat_at_pause']=read(R/'HEARTBEAT.json')
 result['active_training_implementation_commit']='df1543e0c5e2c48fa2da9e4796db4b4e46124802'
 return st,result
@torch.no_grad()
def audit(model,data,checkpoint_sha=None):
 rng=np.random.default_rng(180018);chosen=[]
 for k in range(5):
  complete=[h for h in data.base.jointids['TRAIN'] if j.band(len(data.seqs[h]))==k]
  complete=sorted(rng.choice(complete,size=min(12,len(complete)),replace=False).tolist())
  partial=[h for h in data.ids['TRAIN'] if j.band(len(data.seqs[h]))==k]
  # Coverage-stratified selection, using TRAIN observation masks only.
  partial.sort(key=lambda h:(float(data.cache[data.owners['TRAIN'][h][0]]['seen'].mean()),h))
  partial=[partial[i] for i in sorted(set(np.rint(np.linspace(0,len(partial)-1,min(12,len(partial)))).astype(int)))]
  for kind,ids in [('complete',complete),('partial',partial)]:
   for h in ids:
    o=(data.base.owners[h] if kind=='complete' else data.owners['TRAIN'][h])[0];N=len(data.seqs[h])
    for t in sorted(set([max(1,N//8),max(1,N//2),N-1])):chosen.append((h,o,t,kind))
 names=['rbf_'+str(i) for i in range(8)]+['sequence_separation_over500']+['local_direction_'+a for a in 'xyz']+['frame_rotation_'+str(i) for i in range(9)]+['chemical_compatibility_J1_disabled','pair_valid','frame_i_valid','frame_j_valid']
 tokennames=['chemistry_disabled_'+str(i) for i in range(22)]+['bond_A','bend_rad','sin_torsion','cos_torsion','crowding','bond_valid','bend_valid','torsion_valid','frame_valid','crowding_duplicate','exposure_proxy','packing_J1_disabled','N_terminal','C_terminal','open_generated_end','CA_valid']
 ps=[];ts=collections.defaultdict(list);rawdist=[];contexts=[];qkstats=[];diagnostics=[];missing=0;pairtotal=0;validtotal=0
 for index,(h,o,t,kind) in enumerate(chosen):
  if data.proposal.sequences[h]['split']!='TRAIN':raise ValueError('audit membership leakage')
  b,y,g=data.batch([(h,o,t)],device='cpu');feat,pair,present=features_batch(b,chemistry=False);v=feat[0,:t].numpy();pv=pair[1][0,:t,:t].numpy()
  vv=b['valid'][0,:t].numpy();missing+=int((~vv).sum());pairtotal+=t*t;validtotal+=int(pv.sum())
  for c in range(38):
   mask=np.ones(t,bool)
   if c==22:mask=v[:,27]>0
   elif c==23:mask=v[:,28]>0
   elif c in [24,25]:mask=v[:,29]>0
   elif c in [26,31,32]:mask=vv
   ts[tokennames[c]].extend(v[mask,c].tolist())
  count=min(512,t*t);idx=rng.choice(t*t,size=count,replace=False);arr=pair[0][0,:t,:t].reshape(t*t,25)[idx].numpy();pvm=pv.reshape(-1)[idx]
  ps.append(np.concatenate([arr,pvm[:,None]],axis=1));coord=b['x'][0,:t].numpy();ii,jj=np.unravel_index(idx,(t,t));rawdist.extend(np.linalg.norm(coord[ii]-coord[jj],axis=-1)[pvm].tolist())
  contexts.append([len(data.seqs[h])/500,t/len(data.seqs[h])])
  if index%12==0:
   # First-layer content QK and geometry bias in the actual frozen model; no optimizer.
   bg={k:x.cuda() for k,x in b.items()};fg=feat.cuda()
   with torch.autocast('cuda',dtype=torch.bfloat16):
    enc=model.encoder;ct=enc.context(torch.stack([bg['N']/500.,bg['t']/bg['N']],-1).float())
    xx=torch.cat([enc.aa(bg['ids'])+enc.token(fg)+ct[:,None],(enc.tip+ct)[:,None]],1)
    z=enc.blocks[0].qkv(enc.blocks[0].norm(xx)).view(1,t+1,3,8,44);pos=torch.cat([torch.arange(t,device='cuda')[None],bg['t'][:,None]],1)
    q=rope(z[:,:,0].transpose(1,2),pos);k=rope(z[:,:,1].transpose(1,2),pos);qk=(q.float()@k.float().transpose(-1,-2))/math.sqrt(44)
   qkstats.append({'context':index,'content_QK':describe(qk.cpu().numpy().ravel())})
  diagnostics.append({'sequence_hash':h,'observation_id':o,'target':t,'kind':kind,'length':len(data.seqs[h]),'observed_prefix_fraction':float(vv.mean())})
 a=np.concatenate(ps);effective=a[:,25]>0;featurestats={}
 for c,name in enumerate(names):
  eligible=effective.copy()
  if 9<=c<12:eligible &= a[:,23]>0
  if 12<=c<21:eligible &= (a[:,23]>0)&(a[:,24]>0)
  values=a[eligible,c]
  featurestats[name]={'all_sampled_pairs':describe(a[:,c]),'valid_feature_values':describe(values),'quantile_sampling':'up to512 position pairs/context, uniform without replacement','bf16_zero_from_nonzero_fraction':float(np.mean((torch.tensor(values).to(torch.bfloat16).float().numpy()==0)&(values!=0))) if len(values) else None}
  if c<8:
   featurestats[name]['RBF_below_1e-4_fraction_valid_pairs']=float(np.mean(values<1e-4)) if len(values) else None
   featurestats[name]['BF16_rounds_to_one_fraction']=float(np.mean(torch.tensor(values).to(torch.bfloat16).float().numpy()==1)) if len(values) else None
 # Pair-bias attribution is a conditional zero-channel ablation, not an additive decomposition.
 vals=torch.tensor(a[effective,:25],dtype=torch.float32,device='cuda');net=model.encoder.pair.net
 fp32=net(vals).float()
 with torch.autocast('cuda',dtype=torch.bfloat16):bf=net(vals).float()
 contributions={}
 for c,name in enumerate(names):
  ab=vals.clone();ab[:,c]=0
  with torch.autocast('cuda',dtype=torch.bfloat16):change=bf-net(ab).float()
  contributions[name]={'zero_channel_bias_change_RMS':float(change.square().mean().sqrt()),'mean_abs':float(change.abs().mean()),'max_abs':float(change.abs().max()),'RMS_by_head':change.square().mean(0).sqrt().cpu().tolist()}
 tokenstats={k:describe(v) for k,v in ts.items()}
 tokencontrib={name:{'mean_abs_linear_contribution':float(np.mean(np.abs(ts[name]))*model.encoder.token.weight[:,c].abs().mean().item()),'note':'Conditional on each feature valid denominator; linear contribution before AA embedding, shared context and LayerNorm; not population-averaged input importance'} for c,name in enumerate(tokennames)}
 out={'status':'READ_ONLY_TRAIN_DIAGNOSTIC_NO_NORMALIZATION_FITTED_OR_APPLIED','audit_model_sha256':checkpoint_sha,'sampling_seed':180018,'identities':len(set(v[0] for v in chosen)),'prefix_contexts':len(chosen),'pair_sample_count':len(a),'effective_coordinate_pair_sample_count':int(effective.sum()),'all_context_pair_count':pairtotal,'all_context_valid_pair_count':validtotal,'missing_prefix_positions':missing,'source_contexts':diagnostics,'raw_observed_pair_distance_A':describe(rawdist),'context_features':{'N_over500':describe(np.array(contexts)[:,0]),'t_overN':describe(np.array(contexts)[:,1])},'token_features':tokenstats,'pair_features_25':featurestats,'pair_bias_BF16':describe(bf.cpu().numpy().ravel()),'pair_bias_FP32_minus_BF16_abs':describe((fp32-bf).abs().cpu().numpy().ravel()),'pair_feature_conditional_zero_ablation':contributions,'token_linear_contributions':tokencontrib,'first_block_content_QK':qkstats,'no_outlier_filtering':True,'limitations':['Bounded stratified TRAIN sample, not exhaustive archive extrema','Partial coverage strata intentionally broader than population weighting','Pair quantiles sampled; all-context valid/missing pair counts exact','Ablation magnitude is conditional and nonadditive through nonlinear MLP','No TRAIN-fitted scalers exist in active input pathway; none fitted here','No validation/test observations used for audit statistics']}
 stress=[]
 for row in read(j.core.PILOT/'priors/COORDINATE_EXTREMES.json')['examples']:
  h=row['sequence_hash'];o=row['observation_id'];N=len(data.seqs[h]);t=min(N-1,max(row['positions_1based'])+1)
  if data.proposal.sequences[h]['split']!='TRAIN':raise ValueError('stress panel role')
  b,_,_=data.batch([(h,o,t)],device='cpu');f,_,_=features_batch(b,chemistry=False)
  vals=f[0,:,22][f[0,:,27]>0].numpy()
  stress.append({'sequence_hash':h,'observation_id':o,'source_entry':row['entry_id'],'positions_1based':row['positions_1based'],'historically_verified_pair_A':row['distance_A'],'prefix_target':t,'pair_fully_in_prefix':max(row['positions_1based'])<=t,'observed_prefix_bonds_A':describe(vals),'interpretation':'Separate source-anomaly stress panel; not merged into representative quantiles; source unchanged and not masked as missing'})
 out['separate_verified_TRAIN_source_anomaly_stress_panel']=stress
 augrows=[];generator=torch.Generator().manual_seed(180020)
 for h,o,t,kind in chosen[::5]:
  b,_,_=data.batch([(h,o,t)],device='cpu');bb=s.augment(b,.075,generator);f,pair,_=features_batch(bb,chemistry=False)
  original,_,_=features_batch(b,chemistry=False)
  augrows.append({'sequence_hash':h,'observation_id':o,'t':t,'kind':kind,'observed':int(b['valid'].sum()),'input_visible':int(bb['valid'].sum()),'hidden_by_augmentation':int((bb['ca_observed']&~bb['valid']).sum()),'geometry_targets_unchanged':bool(torch.equal(b['geometry_target_valid'],bb['geometry_target_valid'])),'frame_valid_before_after':[int(original[...,30].sum()),int(f[...,30].sum())],'mean_visible_crowding_before_after':[(float(original[...,26][b['valid']].mean()) if b['valid'].any() else None),(float(f[...,26][bb['valid']].mean()) if bb['valid'].any() else None)]})
 out['training_input_augmentation_diagnostic']={'seed':180020,'probability':.075,'preserve_last3':True,'contexts':augrows,'scope':'TRAIN-only read-only masking probe; actual observation masks and target masks unchanged'}
 out['pair_statistics_include_self_diagonal']=True
 write(destination(R/'FEATURE_NORMALIZATION_DISTRIBUTIONS.json'),out)
 return out
def gradient_audit(model,data,checkpoint_sha=None):
 """Derivative-only diagnostic; no optimizer is constructed or stepped."""
 import j1_sequence_preflight as pf
 torch.manual_seed(180019);torch.cuda.manual_seed_all(180019)
 model.train();set_dropout(model,.15);panel=read(R/'configs/GRADIENT_TRAIN_PANEL.json');totals={};den=collections.Counter()
 for aug in [False,True]:
  for off in range(0,len(panel),4):
   ex=panel[off:off+4];b,y,g=data.batch(ex)
   if aug:b=s.augment(b,.075)
   aa,geo,parts=s.loss_call(model,b,y,g);full=torch.tensor([o not in data.rows for h,o,t in ex],device='cuda')
   objs={'AA':(aa.sum(),len(ex))}
   for c,name in enumerate(['distance','angle','torsion']):
    valid=full&b['geometry_target_valid'][:,c]
    if valid.any():objs[name]=(parts[valid,c].sum(),int(valid.sum()))
   for name,(loss,n) in objs.items():
    grad=pf.gradient(model,loss,True);den[name]+=n
    if name not in totals:totals[name]=grad
    else:
     for key,value in grad.items():totals[name][key]+=value
   del aa,geo,parts,objs,b,y,g
 for name,grad in totals.items():
  for key in grad:grad[key]/=den[name]
 totals['local']={key:sum(totals[name][key] for name in ['distance','angle','torsion'])/3 for key in totals['AA']}
 ex=[(h,data.base.owners[h][0],len(data.seqs[h])-1) for h in data.base.jointids['TRAIN'] if len(data.seqs[h])>=128][:4]
 b,y,g=data.batch(ex);b=s.augment(b,.075);target=torch.tensor(np.asarray([data.base.ca[o][t] for h,o,t in ex]),device='cuda')
 nonlocal_loss,anchors=s.direct_nonlocal(model,b,y,target)
 totals['nonlocal']=pf.gradient(model,nonlocal_loss);del nonlocal_loss,b,y,g,target
 h=next(h for h in data.base.jointids['TRAIN'] if 160<=len(data.seqs[h])<=200);o=data.base.owners[h][0];H=64;t=8
 ref=torch.tensor(data.base.ca[o][:t+H],device='cuda');ids=torch.tensor([j.AA.index(a) for a in data.seqs[h][:t+H]],device='cuda')
 rollout=s.generated_bundle(model,ref[:t],ids,len(data.seqs[h]),H,ref)
 for name in ['Rg','clash','pair']:totals[name]=pf.gradient(model,rollout[name],True)
 del rollout,ref,ids
 groups={};vectors={}
 for name,grad in totals.items():groups[name],vectors[name]=pf.summary(grad)
 aa=vectors['AA'];config=read(R/'configs/FROZEN.json');weights={name:config['lambda_'+name] for name in ['local','nonlocal','Rg','clash','pair']}
 cos={name:float(torch.dot(aa.double(),v.double())/(aa.double().norm()*v.double().norm()).clamp_min(1e-30)) for name,v in vectors.items()}
 geometry={key:sum(weights[name]*totals[name][key] for name in weights) for key in totals['AA']}
 geometry_summary,gv=pf.summary(geometry);scale=min(1.,.5*float(aa.norm())/max(float(gv.norm()),1e-30))
 combined={key:totals['AA'][key]+scale*geometry[key] for key in geometry};combined_summary,_=pf.summary(combined)
 norm=math.sqrt(sum(float(v.double().square().sum()) for v in combined.values()));clip=min(1.,1/max(norm,1e-30))
 out={'status':'FINITE_DERIVATIVE_ONLY_DIAGNOSTIC_NO_OPTIMIZER_STEPS','seed':180019,'checkpoint':checkpoint_sha or read(R/'PAUSED_CHECKPOINT_VERIFICATION.json')['sha256'],'panel_sha256':file_sha(R/'configs/GRADIENT_TRAIN_PANEL.json'),'panel_contexts':len(panel),'augmented_and_unaugmented':True,'loss_gradients_by_group':groups,'AA_shared_cosines':cos,'fixed_weights_not_changed':weights,'weighted_shared_norm_ratios_before_aggregate_cap':{name:weights[name]*groups[name]['shared_all']['L2']/groups['AA']['shared_all']['L2'] for name in weights},'all_auxiliary_active_aggregate_raw':geometry_summary,'hypothetical_aggregate_cap_scale':scale,'combined_before_global_norm_clip':combined_summary,'global_norm_clip_at1_factor':clip,'combined_postclip_L2':norm*clip,'limitations':['Same frozen45-context TRAIN panel; its partial contexts mainly high coverage','Global derivatives from one H64 two-path TRAIN example; not population-wide averages','All auxiliary terms simultaneously active for this comparison; production nonlocal every2 and rollout every8 updates','Periodic gradients, dropout and nonlinear cap prevent interpreting this as a continuous production component time series'],'anchors':anchors}
 write(destination(R/'PAUSED_COMPONENT_GRADIENT_AUDIT.json'),out)
 model.zero_grad(set_to_none=True);model.eval();del totals,vectors,geometry,combined;gc.collect();torch.cuda.empty_cache()

def evaluate(n=16):
 j.core.setup();torch.use_deterministic_algorithms(False);torch.backends.cudnn.deterministic=False
 st,verify=verification();u=st['additional_update'];cp=Path(verify['checkpoint']);sampler=st['sampler'];elapsed=st['elapsed'];history=st['history']
 data=PartialData();recovered=s.Engine(data,read(R/'configs/FROZEN.json'));recovered.resume(cp)
 def equal(a,b):
  if torch.is_tensor(a) and torch.is_tensor(b):return torch.equal(a.detach().cpu(),b.detach().cpu())
  if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
  if isinstance(a,(tuple,list)) and isinstance(b,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
  return a==b
 restored={'model':equal(st['model'],recovered.model.state_dict()),'optimizer':equal(st['optimizer'],recovered.opt.state_dict()),'scheduler':equal(st['scheduler'],recovered.scheduler.state_dict()),'sampler':equal(st['sampler'],recovered.sampler.state_dict()),'RNG':equal(st['rng'],s.rng_state()),'rollout_RNG':equal(st['rollout_rng'],recovered.rollrng.getstate())}
 if not all(restored.values()):raise RuntimeError('full no-step restoration mismatch: '+str(restored))
 verify['actual_controller_restore_equal']=restored;verify['optimizer_steps_taken_for_recovery_check']=0;verify['parameters']=sum(p.numel() for p in recovered.model.parameters() if p.requires_grad)
 verify['training_stage']=1 if u<2500 else 2 if u<5000 else 3
 write(destination(R/'PAUSED_CHECKPOINT_VERIFICATION.json'),verify)
 model=recovered.model;del recovered,st;gc.collect();torch.cuda.empty_cache()
 torch.use_deterministic_algorithms(False);torch.backends.cudnn.deterministic=False
 set_dropout(model,.15);model.eval();panels=read(R/'configs/EVALUATION_PANELS.json');result={}
 for role in ['TRAIN','VALIDATION']:result[role]=j.evaluate(model,data.base,panels[role],role)
 from partial_train import Engine as PE
 eng=SimpleNamespace(model=model,data=data,c={**j.CFG,'live_batch':16},update=u)
 j.core.R=R
 result['PARTIAL_VALIDATION']=PE.evaluate(eng,read(R/'configs/PARTIAL_VALIDATION_PANEL.json'),'PARTIAL')
 extra=partial_eval(model,data,read(R/'configs/PARTIAL_VALIDATION_PANEL.json'),'VALIDATION')
 result['PARTIAL_VALIDATION_AA_ACCURACY']=extra
 base=read(R/'SEQUENCE_BASELINES.json')['J1'];va=result['VALIDATION'];tr=result['TRAIN']
 result['gap']=va['AA_NLL']-tr['AA_NLL'];result['AA_validation_paired_change']=paired_ci([va['per_identity'][h]['AA_NLL']-base['VALIDATION']['per_identity'][h]['AA_NLL'] for h in sorted(va['per_identity'])])
 result.update(update=u,checkpoint=str(cp),sha256=verify['sha256'],TEST='NOT_EVALUATED')
 write(destination(R/'PAUSED_TRAIN_VALIDATION_EVALUATION.json'),result)
 gen=generation_for(model,cp,verify['sha256'],n);parentrows=[]
 for v in read(j.OLD/'runs/J1-FULL/SELECTED_GENERATION.json'):
  if v['sample_index']<n:
   v=dict(v);v['metrics']=v['independent_CA_metrics'];parentrows.append(v)
 native=read(j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json')['native'];new=metric_summary(gen);old=metric_summary(parentrows);comp={}
 for N in [64,128,256,384,500]:
  a=sorted([v for v in parentrows if v['N']==N],key=lambda v:v['sample_index']);b=sorted([v for v in gen if v['N']==N],key=lambda v:v['sample_index'])
  ref=np.median([v['metrics']['Rg_over_N_cuberoot'] for v in native['TRAIN'][str(N)]['rows']])
  valid=[(x,y) for x,y in zip(a,b) if 'metrics' in x and 'metrics' in y]
  comp[str(N)]={'clash_delta_per100':paired_ci([y['metrics']['clashes_per100_residues']-x['metrics']['clashes_per100_residues'] for x,y in valid]),'native_abs_log_Rg_error_delta':paired_ci([abs(math.log(y['metrics']['Rg_over_N_cuberoot']/ref))-abs(math.log(x['metrics']['Rg_over_N_cuberoot']/ref)) for x,y in valid]),'native_TRAIN_normalized_Rg_median':float(ref)}
 geometry={'samples_per_length':n,'paused':new,'parent_matched_seed_subset':old,'paired_changes':comp,'native_reference_path':str(j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json'),'native_reference_sha256':file_sha(j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json'),'interpretation':'Fully free-running AA and geometry, not true-AA-conditioned training rollouts; no folding claim'}
 geometry['native_TRAIN_VALIDATION']={}
 for role in ['TRAIN','VALIDATION']:
  geometry['native_TRAIN_VALIDATION'][role]={}
  for N,entry in native[role].items():
   rr=entry['rows'];geometry['native_TRAIN_VALIDATION'][role][N]={'observations':len(rr),'identities':len({r['sequence_hash'] for r in rr}),'selection':'Frozen length-matched native window, not newly selected controls','metrics':{name:describe([r['metrics'][name] for r in rr]) for name in ['Rg_A','Rg_over_N_cuberoot','clashes_per100_residues','long_contact_fraction','degenerate_frames','torsion_resultant']}}
 archive=s.PREVIOUS/'NATIVE_VS_GENERATED_GLOBAL_GEOMETRY.json'
 geometry['historical_60k_reference']={'path':str(archive),'sha256':file_sha(archive),'summary32':read(archive)['latest_actual_policy'],'limitation':'Archived32-sample summary, descriptive comparison; not paired with current16 subset'}
 geometry['pooled_generated_histograms']={}
 for N in [64,128,256,384,500]:
  rr=[v['metrics'] for v in gen if v['N']==N and 'metrics' in v]
  ah=np.array([v['angle_histogram'] for v in rr]).sum(0);th=np.array([v['torsion_histogram'] for v in rr]).sum(0)
  geometry['pooled_generated_histograms'][str(N)]={'virtual_angle_edges_rad':np.linspace(0,np.pi,19).tolist(),'virtual_angle_counts':ah.tolist(),'signed_torsion_edges_rad':np.linspace(-np.pi,np.pi,37).tolist(),'signed_torsion_counts':th.tolist(),'degenerate_frames_per_available_triple':sum(v['degenerate_frames'] for v in rr)/max(1,len(rr)*(N-2))}
 write(destination(R/'PAUSED_FREE_GENERATION_COMPARISON.json'),geometry)
 necessary=[]
 if va['AA_NLL']>=base['VALIDATION']['AA_NLL']:necessary.append('AA did not improve over original parent')
 if va['geometry_NLL']>base['VALIDATION']['geometry_NLL']+.10:necessary.append('frozen local geometry safeguard')
 if result['gap']>base['VALIDATION']['AA_NLL']-base['TRAIN']['AA_NLL']:necessary.append('generalization gap safeguard')
 if n==32:necessary+=check_candidate(new,old,native)
 else:necessary.append('32-sample final promotion protocol not executed; 16-sample diagnostic only')
 write(destination(R/'PAUSED_SELECTION.json'),{'latest_checkpoint':str(cp),'latest_sha256':verify['sha256'],'selected_checkpoint':str(j.PARENT),'selected_sha256':j.PARENT_SHA,'candidate_promoted':False,'reason':'retain immutable fallback pending review of shortened experiment','latest_necessary_or_pending_conditions':necessary,'TEST':'NOT_EVALUATED'})
 audit(model,data)
 gradient_audit(model,data)
 curve=traces();write(R/'J1_TRAINING_CURVES.json',curve)
 stage3=[v for v in curve if 'AA_shared_gradient_L2' in v];gg={}
 for key in ['AA_shared_gradient_L2','geometry_shared_raw_L2','geometry_shared_scaled_L2','geometry_to_AA_shared_ratio','AA_geometry_shared_cosine','geometry_gradient_scale','gradient_norm_preclip']:gg[key]=describe([v[key] for v in stage3])
 gg['auxiliary_gradient_cap_count']=sum(v['geometry_gradient_scale']<1 for v in stage3);gg['stage3_updates']=len(stage3);gg['global_norm_clip_count_all_updates']=sum(v['gradient_norm_preclip']>1 for v in curve);gg['all_updates']=len(curve);gg['AA_shared_after_whole_model_normclip']=describe([v['AA_shared_gradient_L2']*min(1,1/max(v['gradient_norm_preclip'],1e-30)) for v in stage3]);gg['clipping_interpretation']='Raw backward gradients after normclip; not an equivalent AdamW learning-rate estimate';gg['source']='Actual shared encoder parameter gradients, not geometry heads only';gg['individual_auxiliary_loss_gradients']='Preflight GRADIENT_BALANCE_REPORT.json; continuous production logs aggregate all geometry paths, no invented component-specific time series'
 cover=s.Coverage();cover.load_state_dict(sampler['coverage'])
 write(destination(R/'PAUSED_GRADIENT_AND_EXPOSURE_SUMMARY.json'),{'gradients':gg,'sampler_counts':sampler['counts'],'coverage':cover.report(),'rollout_residues':curve[-1]['rollout_residues'],'nonlocal_targets':curve[-1]['nonlocal_targets'],'trained_horizons':dict(collections.Counter(v['rollout_horizon'] for v in curve if 'rollout_horizon' in v)),'ordinary_update_seconds_sum':sum(v['seconds'] for v in curve),'checkpoint_elapsed_seconds':elapsed})
 write(destination(R/'PAUSED_EVALUATION_COMPLETE.json'),{'status':'DONE_NO_OPTIMIZER_STEPS','time':time.time(),'update':u,'samples_per_length':n})
 print('PAUSED_EVALUATION_AND_TRAIN_FEATURE_AUDIT_COMPLETE',flush=True)
def compare_milestones():
 """Only +5000/+7500 enter scientific comparison; +7744 is no-step recovery verification."""
 j.core.setup();st,verify=verification();data=PartialData()
 recovered=s.Engine(data,read(R/'configs/FROZEN.json'));recovered.resume(Path(verify['checkpoint']))
 def same(a,b):
  if torch.is_tensor(a) and torch.is_tensor(b):return torch.equal(a.detach().cpu(),b.detach().cpu())
  if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
  if isinstance(a,(tuple,list)) and isinstance(b,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
  return a==b
 verified={'model':same(st['model'],recovered.model.state_dict()),'optimizer':same(st['optimizer'],recovered.opt.state_dict()),'scheduler':same(st['scheduler'],recovered.scheduler.state_dict()),'sampler':same(st['sampler'],recovered.sampler.state_dict()),'RNG':same(st['rng'],s.rng_state()),'rollout_RNG':same(st['rollout_rng'],recovered.rollrng.getstate())}
 if not all(verified.values()):raise ValueError('no-step recovery state mismatch')
 verify.update(actual_controller_restore_equal=verified,optimizer_steps_taken_for_recovery_check=0,parameters=sum(p.numel() for p in recovered.model.parameters() if p.requires_grad),reason='USER_IMMEDIATE_STOP_REVIEW_5000_VS_7500',comparison_updates=[5000,7500],recovery_updates_after7500=verify['additional_update']-7500,recovery_checkpoint_excluded_from_selection=True,immediate_instruction_receipt=read(R/'IMMEDIATE_USER_STOP_RECEIPT.json'))
 write(destination(R/'PAUSED_CHECKPOINT_VERIFICATION.json'),verify)
 model=recovered.model;coverage=recovered.sampler.report();counts=dict(recovered.sampler.counts);hist=recovered.history;residues=recovered.residues;nonlocal_targets=recovered.nonlocal_targets
 del recovered,st;gc.collect();torch.cuda.empty_cache();torch.use_deterministic_algorithms(False);torch.backends.cudnn.deterministic=False;model.eval()
 base=read(R/'SEQUENCE_BASELINES.json')['J1'];panels=read(R/'configs/EVALUATION_PANELS.json')
 parent=[]
 for row in read(j.OLD/'runs/J1-FULL/SELECTED_GENERATION.json'):
  row=dict(row);row['metrics']=row['independent_CA_metrics'];parent.append(row)
 old=metric_summary(parent);native=read(j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json')['native'];review=[]
 for u in [5000,7500]:
  cp=R/'runs/J1'/f'checkpoint_{u:05}.pt';sha=file_sha(cp);state=load_checked(cp)
  if state['additional_update']!=u or state['lineage']['config']!=read(R/'configs/FROZEN.json'):raise ValueError('comparison checkpoint/config mismatch')
  ev=read(R/'runs/J1'/f'evaluation_{u:05}.json');record=next(v for v in state['history'] if v['update']==u)
  for role in ['TRAIN','VALIDATION']:
   if ev[role]['panel_sha256']!=j.digest(panels[role]) or not same(ev[role],record[role]):raise ValueError('cached evaluation/checkpoint/panel correspondence')
  model.load_state_dict(state['model'],strict=True);del state;gc.collect();set_dropout(model,.15);model.eval()
  # Cached frozen AA/local evaluations are reused exactly, not rerun.
  # Only missing AA accuracy on the partial panel is computed.
  accuracy=partial_eval(model,data,sorted(read(R/'configs/PARTIAL_VALIDATION_PANEL.json'),key=lambda v:(v[2],v[0],v[1])),'VALIDATION')
  rows=generation_for(model,cp,sha,n=32);summary=metric_summary(rows);fail=check_candidate(summary,old,native)
  val=ev['VALIDATION'];tr=ev['TRAIN'];gap=val['AA_NLL']-tr['AA_NLL'];oldgap=base['VALIDATION']['AA_NLL']-base['TRAIN']['AA_NLL']
  if val['AA_NLL']>=base['VALIDATION']['AA_NLL']:fail.append('AA did not improve over original')
  if val['geometry_NLL']>base['VALIDATION']['geometry_NLL']+.10:fail.append('local geometry safeguard')
  if gap>oldgap:fail.append('generalization gap safeguard')
  comparison={}
  for N in [64,128,256,384,500]:
   a=sorted([r for r in parent if r['N']==N],key=lambda r:r['sample_index']);b=sorted([r for r in rows if r['N']==N],key=lambda r:r['sample_index'])
   ref=np.median([r['metrics']['Rg_over_N_cuberoot'] for r in native['TRAIN'][str(N)]['rows']]);valid=[(x,y) for x,y in zip(a,b) if 'metrics' in x and 'metrics' in y]
   comparison[str(N)]={'clash_delta_per100':paired_ci([y['metrics']['clashes_per100_residues']-x['metrics']['clashes_per100_residues'] for x,y in valid]),'native_abs_log_Rg_error_delta':paired_ci([abs(math.log(y['metrics']['Rg_over_N_cuberoot']/ref))-abs(math.log(x['metrics']['Rg_over_N_cuberoot']/ref)) for x,y in valid]),'native_TRAIN_normalized_Rg_median':float(ref)}
  result={'update':u,'checkpoint':str(cp),'sha256':sha,'evaluation_reused_path':str(R/'runs/J1'/f'evaluation_{u:05}.json'),'evaluation_sha256':file_sha(R/'runs/J1'/f'evaluation_{u:05}.json'),'TRAIN':tr,'VALIDATION':val,'PARTIAL_VALIDATION':ev['PARTIAL_VALIDATION'],'PARTIAL_VALIDATION_AA_ACCURACY':accuracy,'NLL_gap':gap,'AA_validation_paired_change':paired_ci([val['per_identity'][h]['AA_NLL']-base['VALIDATION']['per_identity'][h]['AA_NLL'] for h in sorted(val['per_identity'])]),'free_generation':summary,'paired_changes':comparison,'frozen_safeguards_pass':not fail,'failures':fail,'generation_samples_per_length':32}
  review.append(result);write(R/'MILESTONE_5000_7500_COMPARISON.json',review)
  if file_sha(cp)!=sha:raise ValueError('protected comparison checkpoint changed')
  if u==7500:
   audit(model,data,sha);gradient_audit(model,data,sha)
 archive=s.PREVIOUS/'NATIVE_VS_GENERATED_GLOBAL_GEOMETRY.json'
 native_summary={role:{N:{'identities':len({r['sequence_hash'] for r in entry['rows']}),'metrics':{name:describe([r['metrics'][name] for r in entry['rows']]) for name in ['Rg_A','Rg_over_N_cuberoot','clashes_per100_residues','long_contact_fraction','degenerate_frames','torsion_resultant']}} for N,entry in native[role].items()} for role in ['TRAIN','VALIDATION']}
 write(destination(R/'MILESTONE_GEOMETRY_REFERENCES.json'),{'parent':old,'historical_60k':{'source':str(archive),'sha256':file_sha(archive),'summary32':read(archive)['latest_actual_policy']},'native_TRAIN_VALIDATION':native_summary,'native_source_sha256':file_sha(j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json'),'metric_definitions':'Clash<2A seqsep>=3; long contact<8A seqsep>=24; length-matched frozen native windows; CA-only proxies'})
 curve=traces();write(R/'J1_TRAINING_CURVES.json',curve);stage3=[v for v in curve if 'AA_shared_gradient_L2' in v]
 gg={name:describe([v[name] for v in stage3]) for name in ['AA_shared_gradient_L2','geometry_shared_raw_L2','geometry_shared_scaled_L2','geometry_to_AA_shared_ratio','AA_geometry_shared_cosine','geometry_gradient_scale','gradient_norm_preclip']}
 gg.update(auxiliary_cap_count=sum(v['geometry_gradient_scale']<1 for v in stage3),stage3_updates=len(stage3),whole_model_clip_count=sum(v['gradient_norm_preclip']>1 for v in curve),note='Actual shared encoder logs; separate component/head diagnostic is derivative-only, not a missing production time series')
 write(destination(R/'PAUSED_GRADIENT_AND_EXPOSURE_SUMMARY.json'),{'gradients':gg,'sampler_counts':counts,'coverage':coverage,'rollout_residues':residues,'nonlocal_targets':nonlocal_targets,'trained_horizons':dict(collections.Counter(v['rollout_horizon'] for v in curve if 'rollout_horizon' in v)),'main_final_update':verify['additional_update'],'selection_ceiling_update':7500,'post7500_updates_recovery_only':verify['additional_update']-7500})
 passing=[v for v in review if v['frozen_safeguards_pass']];candidate=min(passing,key=lambda v:v['VALIDATION']['AA_NLL']) if passing else None
 write(destination(R/'MILESTONE_SELECTION_RECEIPT.json'),{'considered_updates':[5000,7500],'post7500_excluded':True,'eligible_lowest_AA_candidate':None if candidate is None else {'update':candidate['update'],'checkpoint':candidate['checkpoint'],'sha256':candidate['sha256']},'quality_fallback':str(j.PARENT),'quality_fallback_sha256':j.PARENT_SHA,'automatic_promotion':False,'reason':'User requested review; frozen safeguards applied without modification','best_AA_validation_only':min(review,key=lambda v:v['VALIDATION']['AA_NLL'])['update'],'TEST':'NOT_EVALUATED'})
 write(destination(R/'PAUSED_EVALUATION_COMPLETE.json'),{'status':'COMPARISON_5000_7500_AND_READ_ONLY_AUDIT_COMPLETE','time':time.time(),'recovery_update':verify['additional_update'],'samples_per_length_per_checkpoint':32,'optimizer_updates_after_stop':0})
 print('MILESTONE_COMPARISON_COMPLETE_NO_TRAINING',flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--samples',type=int,default=16,choices=[16,32]);a.add_argument('--compare-milestones',action='store_true');args=a.parse_args();compare_milestones() if args.compare_milestones else evaluate(args.samples)
