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
 write(destination(p),result)
 return st,result
@torch.no_grad()
def audit(model,data):
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
    for t in sorted(set([max(1,int(.25*N)),max(1,int(.8*N))])):chosen.append((h,o,t,kind))
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
  contributions[name]={'zero_channel_bias_change_RMS':float(change.square().mean().sqrt()),'mean_abs':float(change.abs().mean()),'max_abs':float(change.abs().max())}
 tokenstats={k:describe(v) for k,v in ts.items()}
 tokencontrib={name:{'mean_abs_linear_contribution':float(np.mean(np.abs(ts[name]))*model.encoder.token.weight[:,c].abs().mean().item()),'note':'linear contribution before shared context, AA embedding and LayerNorm'} for c,name in enumerate(tokennames)}
 out={'status':'READ_ONLY_TRAIN_DIAGNOSTIC_NO_NORMALIZATION_FITTED_OR_APPLIED','sampling_seed':180018,'identities':len(set(v[0] for v in chosen)),'prefix_contexts':len(chosen),'pair_sample_count':len(a),'effective_coordinate_pair_sample_count':int(effective.sum()),'all_context_pair_count':pairtotal,'all_context_valid_pair_count':validtotal,'missing_prefix_positions':missing,'source_contexts':diagnostics,'raw_observed_pair_distance_A':describe(rawdist),'context_features':{'N_over500':describe(np.array(contexts)[:,0]),'t_overN':describe(np.array(contexts)[:,1])},'token_features':tokenstats,'pair_features_25':featurestats,'pair_bias_BF16':describe(bf.cpu().numpy().ravel()),'pair_bias_FP32_minus_BF16_abs':describe((fp32-bf).abs().cpu().numpy().ravel()),'pair_feature_conditional_zero_ablation':contributions,'token_linear_contributions':tokencontrib,'first_block_content_QK':qkstats,'no_outlier_filtering':True,'limitations':['Bounded stratified TRAIN sample, not exhaustive archive extrema','Partial coverage strata intentionally broader than population weighting','Pair quantiles sampled; all-context valid/missing pair counts exact','Ablation magnitude is conditional and nonadditive through nonlinear MLP','No TRAIN-fitted scalers exist in active input pathway; none fitted here','No validation/test observations used for audit statistics']}
 write(destination(R/'FEATURE_NORMALIZATION_DISTRIBUTIONS.json'),out)
 return out
def gradient_audit(model,data):
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
 out={'status':'FINITE_DERIVATIVE_ONLY_DIAGNOSTIC_NO_OPTIMIZER_STEPS','seed':180019,'checkpoint':read(R/'PAUSED_CHECKPOINT_VERIFICATION.json')['sha256'],'panel_sha256':file_sha(R/'configs/GRADIENT_TRAIN_PANEL.json'),'panel_contexts':len(panel),'augmented_and_unaugmented':True,'loss_gradients_by_group':groups,'AA_shared_cosines':cos,'fixed_weights_not_changed':weights,'weighted_shared_norm_ratios_before_aggregate_cap':{name:weights[name]*groups[name]['shared_all']['L2']/groups['AA']['shared_all']['L2'] for name in weights},'all_auxiliary_active_aggregate_raw':geometry_summary,'hypothetical_aggregate_cap_scale':scale,'combined_before_global_norm_clip':combined_summary,'global_norm_clip_at1_factor':clip,'combined_postclip_L2':norm*clip,'limitations':['Same frozen45-context TRAIN panel; its partial contexts mainly high coverage','Global derivatives from one H64 two-path TRAIN example; not population-wide averages','All auxiliary terms simultaneously active for this comparison; production nonlocal every2 and rollout every8 updates','Periodic gradients, dropout and nonlinear cap prevent interpreting this as a continuous production component time series'],'anchors':anchors}
 write(destination(R/'PAUSED_COMPONENT_GRADIENT_AUDIT.json'),out)
 model.zero_grad(set_to_none=True);model.eval();del totals,vectors,geometry,combined;gc.collect();torch.cuda.empty_cache()

def evaluate(n=16):
 j.core.setup();torch.use_deterministic_algorithms(False);torch.backends.cudnn.deterministic=False
 st,verify=verification();model,parent=j.parent_model();del parent
 model.load_state_dict(st['model'],strict=True);set_dropout(model,.15);model.eval()
 u=st['additional_update'];cp=Path(verify['checkpoint']);sampler=st['sampler'];elapsed=st['elapsed'];history=st['history'];del st;gc.collect()
 data=PartialData();panels=read(R/'configs/EVALUATION_PANELS.json');result={}
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
 gg['auxiliary_gradient_cap_count']=sum(v['geometry_gradient_scale']<1 for v in stage3);gg['stage3_updates']=len(stage3);gg['global_norm_clip_count_all_updates']=sum(v['gradient_norm_preclip']>1 for v in curve);gg['all_updates']=len(curve);gg['source']='Actual shared encoder parameter gradients, not geometry heads only';gg['individual_auxiliary_loss_gradients']='Preflight GRADIENT_BALANCE_REPORT.json; continuous production logs aggregate all geometry paths, no invented component-specific time series'
 cover=s.Coverage();cover.load_state_dict(sampler['coverage'])
 write(destination(R/'PAUSED_GRADIENT_AND_EXPOSURE_SUMMARY.json'),{'gradients':gg,'sampler_counts':sampler['counts'],'coverage':cover.report(),'rollout_residues':curve[-1]['rollout_residues'],'nonlocal_targets':curve[-1]['nonlocal_targets'],'trained_horizons':dict(collections.Counter(v['rollout_horizon'] for v in curve if 'rollout_horizon' in v)),'ordinary_update_seconds_sum':sum(v['seconds'] for v in curve),'checkpoint_elapsed_seconds':elapsed})
 write(destination(R/'PAUSED_EVALUATION_COMPLETE.json'),{'status':'DONE_NO_OPTIMIZER_STEPS','time':time.time(),'update':u,'samples_per_length':n})
 print('PAUSED_EVALUATION_AND_TRAIN_FEATURE_AUDIT_COMPLETE',flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--samples',type=int,default=16,choices=[16,32]);args=a.parse_args();evaluate(args.samples)
