"""Post-run corrected-protocol comparison and immutable evidence. No optimizer construction."""
import collections,gc,json,math,os,resource,time,subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import j1_stabilize_train as t
R=t.R;O=t.O;read=t.read;write=t.write;sha=t.sha

def stats(x):
 a=np.asarray(x,float);return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'p05_p95':np.quantile(a,[.05,.95]).tolist(),'min':float(a.min()),'max':float(a.max())} if len(a) else {'n':0}
def precheck(row,base):
 fails=t.native_checks(row);cfg=read(R/'configs/FROZEN.json')['promotion'];v=row['VALIDATION'];b=base['VALIDATION'];tr=row['TRAIN'];ids=sorted(v['per_identity']);ci=t.paired_ci([v['per_identity'][h]['AA_NLL']-b['per_identity'][h]['AA_NLL'] for h in ids]);cal=read(R/'calibration'/f"VALIDATION_{row['new_update']:05d}.json");bc=read(R/'calibration/VALIDATION_00000.json');a=cal['torsion_calibration'];ba=bc['torsion_calibration']
 def need(ok,why):
  if not ok:fails.append(why)
 need(v['AA_NLL']<b['AA_NLL']-.002,'AA improvement <.002nats');need(ci['bootstrap_95pct'][1]<0,'AA paired95 not significant');need(v['AA_NLL']-tr['AA_NLL']<=b['AA_NLL']-base['TRAIN']['AA_NLL']+.02,'gap growth>.02');need(v['component_NLL'][2]<=b['component_NLL'][2]+.02,'torsion mean deterioration');need(all(x<=y+.02 for x,y in zip(v['component_NLL'][:2],b['component_NLL'][:2])),'distance/angle deterioration');need(a['coverage50_80_95']['95']>=ba['coverage50_80_95']['95']-.01,'95pct tail coverage deterioration');err=lambda x:sum(abs(x[str(k)]-k/100) for k in [50,80,95]);need(err(a['coverage50_80_95'])<=err(ba['coverage50_80_95'])+.03,'torsion calibration error increase');need(cal['components']['torsion']['worst_tail']['0.01']['contribution_to_mean']<=1.10*bc['components']['torsion']['worst_tail']['0.01']['contribution_to_mean']+.01,'torsion likelihood tail deterioration')
 return fails,ci

def main():
 if (R/'FINAL_ASSESSMENT_COMPLETE.json').exists():raise RuntimeError('already assessed; no rerun')
 done=read(R/'runs/J1/COMPLETED.json');history=read(R/'TRAIN_VALIDATION_PERPLEXITY.json');cfg=read(R/'configs/FROZEN.json');base=history[0];start=time.monotonic();t.c.s.j.core.setup();model=t.c.s.j.core.model('J1');native=read(t.c.s.j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json')['native'];source=read(O/'generation/final00500_ca_fp64_interior_v1.json');fallback=read(O/'generation/fallback_ca_fp64_interior_v1.json')
 for name,rows,expected in [('source500',source,sha(O/'runs/J1/model_00500.pt')),('original',fallback,t.c.s.j.PARENT_SHA)]:
  if len(rows)!=160 or any(v['checkpoint_sha256']!=expected or v['protocol']!='ca_fp64_interior_v1' or v['live_batch']!=4 for v in rows):raise ValueError('baseline corrected cache identity mismatch')
 import sys,platform
 write(R/'ENVIRONMENT.json',{'python':sys.version,'platform':platform.platform(),'torch':str(torch.__version__),'torch_CUDA':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'numpy':np.__version__,'GPU':torch.cuda.get_device_name(),'GPU_capability':list(torch.cuda.get_device_capability()),'GPU_total_memory':torch.cuda.get_device_properties(0).total_memory,'training_deterministic_algorithms':'enabled in training Engine; evaluation uses historical_evaluation context','D_only_runtime_environment':{k:os.environ.get(k) for k in ['TMPDIR','XDG_CACHE_HOME','TORCH_HOME','TRITON_CACHE_DIR','CUDA_CACHE_PATH','TORCHINDUCTOR_CACHE_DIR','CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','PYTHONDONTWRITEBYTECODE']}})
 summaries={'source500':t.metric_summary(source),'original':t.metric_summary(fallback)};raw={'source500':source,'original':fallback};reviews=[];byu={v['new_update']:v for v in history};latest=done['new_optimizer_updates'];bestAA=min(history[1:],key=lambda x:x['VALIDATION']['AA_NLL'])['new_update'];eligible=[]
 for row in history[1:]:
  fails,ci=precheck(row,base);reviews.append({'new_update':row['new_update'],'native_calibration_failures':fails,'AA_paired_change_vs_start':ci,'AA_NLL':row['VALIDATION']['AA_NLL'],'VAL_PPL':row['VALIDATION']['perplexity'],'local_NLL':row['VALIDATION']['geometry_NLL']})
  if not fails:eligible.append(row['new_update'])
 write(R/'NATIVE_CANDIDATE_REVIEWS.json',reviews)
 assessed={};selected=None
 for u in list(dict.fromkeys([latest,bestAA]+sorted(eligible,key=lambda x:byu[x]['VALIDATION']['AA_NLL']))):
  cp=R/'runs/J1'/f'checkpoint_{u:05d}.pt';st=t.c.s.load_checked(cp);model.load_state_dict(st['model'],strict=True);del st;gc.collect();rows=t.generation(SimpleNamespace(model=model,update=u),cp,32);ss=t.metric_summary(rows);fail=t.check_candidate(ss,summaries['original'],native)
  for N,v in ss.items():
   if v['long_contact_fraction']['mean'] is None or v['long_contact_fraction']['mean']<.8*summaries['source500'][N]['long_contact_fraction']['mean']:fail.append(N+': >20pct contact loss vs source500')
  nf=next(v['native_calibration_failures'] for v in reviews if v['new_update']==u);assessed[u]={'checkpoint':str(cp),'sha256':sha(cp),'generation':ss,'native_calibration_failures':nf,'generation_failures':fail,'passed_all_frozen_and_predeclared_gates':not nf and not fail};raw['new'+str(u)]=rows;summaries['new'+str(u)]=ss;write(R/'CANDIDATE_REVIEWS.json',assessed)
  # Retain all requested latest/bestAA evaluations; additionally evaluate qualified candidates in NLL order until one passes.
  if not nf and not fail:
   if selected is None or byu[u]['VALIDATION']['AA_NLL']<byu[selected]['VALIDATION']['AA_NLL']:selected=u
  if u in eligible and selected is not None and all(v in assessed for v in [latest,bestAA]):break
 del model;gc.collect();torch.cuda.empty_cache()
 from j1_clip_finalize import paired,local_statistics
 comparison={'protocol':'ca_fp64_interior_v1','live_batch':4,'samples_per_length':32,'lengths':[64,128,256,384,500],'baseline_references':{'source500':str(O/'generation/final00500_ca_fp64_interior_v1.json'),'original':str(O/'generation/fallback_ca_fp64_interior_v1.json')},'summaries':summaries,'reviews':assessed,'paired_vs_source500':{k:paired(source,v,native) for k,v in raw.items() if k.startswith('new')},'local_distribution_statistics':{k:local_statistics(v) for k,v in raw.items()},'boundary_repairs':{k:np.array([v['boundary_repairs'] for v in a]).sum(0).tolist() for k,a in raw.items()},'all_attempts_retained_no_replacement':True,'TEST':'NOT_USED'};comparison['native_reference']=str(t.c.s.j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json');comparison['native_reference_sha256']=sha(t.c.s.j.OLD/'NATIVE_VS_GENERATED_GEOMETRY.json');comparison['native_summary']={role:{N:{'controls':len(v['rows']),'metrics':{key:stats([r['metrics'][key] for r in v['rows']]) for key in ['Rg_A','Rg_over_N_cuberoot','clashes_per100_residues','long_contact_fraction','degenerate_frames']}} for N,v in native[role].items()} for role in ['TRAIN','VALIDATION']};write(R/'CORRECTED_GENERATION_COMPARISON.json',comparison)
 if selected is None:
  cp=Path(read(R/'replay/RECONSTRUCTION.json')['checkpoint']);selection={'new_candidate_promoted':False,'selected_new_update':0,'selected_checkpoint':str(cp),'selected_sha256':sha(cp),'selected_model_weights_match_original_model500_sha256':'9303f0d82a8df044302488340f3ad81535f2fc321360a00bf2892f3a0407aa91','reason':'no new candidate passes all unchanged and predeclared sequence/local/calibration/global gates; retain selected source500 with reconstructed full state'}
 else:
  cp=R/'runs/J1'/f'checkpoint_{selected:05d}.pt';selection={'new_candidate_promoted':True,'selected_new_update':selected,'selected_checkpoint':str(cp),'selected_sha256':sha(cp),'reason':'lowest VAL AA NLL among evaluated candidates passing unchanged and predeclared safeguards'}
 write(R/'FINAL_SELECTION.json',selection)
 traces=[]
 for p in sorted((R/'runs/J1').glob('updates_*.jsonl')):traces.extend(json.loads(x) for x in open(p) if x.strip())
 traces.sort(key=lambda x:x['new_update'])
 if len(traces)!=latest or [v['new_update'] for v in traces]!=list(range(1,latest+1)):raise ValueError('actual unique optimizer traces missing/duplicated')
 groups=list(traces[0]['group_before']);training={'unique_new_updates':len(traces),'historical_replay_updates':500,'restart_reexecuted_updates':2,'total_real_PDB_optimizer_calls_this_task':len(traces)+502,'AA_decisions':len(traces)*64,'complete_geometry_decisions':len(traces)*32,'nonlocal_targets':done['nonlocal_targets'],'differentiated_long_rollout_residues':0,'AA_decisions_per_optimizer_second':len(traces)*64/sum(x['seconds'] for x in traces),'optimizer_seconds':sum(x['seconds'] for x in traces),'group_limits':cfg['block_limits'],'gradient_norms_before':{k:stats([v['group_before'][k] for v in traces]) for k in groups},'gradient_norms_after':{k:stats([v['group_after'][k] for v in traces]) for k in groups},'clip_fractions':{k:float(np.mean([v['group_scale'][k]<1 for v in traces])) for k in groups},'clipping_factors':{k:stats([v['group_scale'][k] for v in traces]) for k in groups},'parameter_update_norms':{k:stats([v['parameter_update_L2_by_block'][k] for v in traces if 'parameter_update_L2_by_block' in v]) for k in groups},'shared_geometry_ratio':stats([v['geometry_to_AA_shared_ratio'] for v in traces]),'shared_AA_aux_cosine':stats([v['AA_geometry_shared_cosine'] for v in traces]),'AA_shared_norm':stats([v['AA_shared_L2'] for v in traces]),'all_G_heads_independent_of_S_cap':all(not v['G_head_scaled_by_shared_cap'] for v in traces),'no_whole_model_clip':all(not v['whole_model_clip_applied'] for v in traces),'source_exposure_counts_delta':{k:v-done['counts_at_start'].get(k,0) for k,v in done['counts_at_end'].items()},'curve_sample_every25':[v for v in traces if v['new_update']%25==0]};write(R/'J1_TRAINING_CURVES.json',training)
 calhistory={};localhistory={}
 for row in history:
  u=row['new_update'];calhistory[str(u)]={};localhistory[str(u)]={}
  for role in ['TRAIN','VALIDATION']:
   cs=read(R/'calibration'/f'{role}_{u:05d}.json');calhistory[str(u)][role]={'calibration':cs['torsion_calibration'],'torsion_tail':cs['components']['torsion']['worst_tail'],'worst_targets':cs['worst_torsion_targets']};localhistory[str(u)][role]={k:cs[k] for k in ['components','decisions','identities','historic_local_per_decision','strata']}
 write(R/'TORSION_CALIBRATION_HISTORY.json',calhistory);write(R/'LOCAL_GEOMETRY_COMPONENTS.json',localhistory)
 # Full checkpoint state/moment checks, including slice-specific AdamW conditioning.
 verify=[];moments={};names=None
 for cp in [Path(selection['selected_checkpoint']),Path(done['latest_checkpoint'])]+[R/'runs/J1'/f"checkpoint_{v['new_update']:05d}.pt" for v in history[1:]]:
  if str(cp) in moments:continue
  st=t.c.s.load_checked(cp);m=t.c.s.j.core.model('J1',device='cpu');m.load_state_dict(st['model']);names=[n for n,p in m.named_parameters()];flat=[i for g in st['optimizer']['param_groups'] for i in g['params']];mm={}
  for n,i in zip(names,flat):
   q=st['optimizer']['state'][i]
   if q['exp_avg'].shape!=st['model'][n].shape:raise ValueError('optimizer parameter mapping')
   if n in ['geom.2.weight','geom.2.bias']:mm[n]=[{'row':j,'m_RMS':float(q['exp_avg'][j].double().square().mean().sqrt()),'v_mean':float(q['exp_avg_sq'][j].double().mean())} for j in range(6)]
  counts={int(float(v['step'])) for v in st['optimizer']['state'].values()};finite=all(bool(torch.isfinite(v).all()) for q in st['optimizer']['state'].values() for v in q.values() if torch.is_tensor(v));moments[str(cp)]=mm;verify.append({'path':str(cp),'sha256':sha(cp),'optimizer_step_counters':sorted(counts),'scheduler_epoch':st['scheduler']['last_epoch'],'sampler_cursor':st['sampler']['cursor'],'finite_optimizer':finite,'full_fields':list(st)})
  if not finite:raise ValueError('nonfinite saved optimizer')
  del st,m;gc.collect()
 write(R/'ADAMW_COMPONENT_MOMENTS.json',moments);write(R/'FULL_CHECKPOINT_VERIFICATION.json',{'status':'PASS','checkpoints':verify,'restart_proof':read(R/'COMPONENT_RESTART_TEST.json')});write(R/'FINAL_ASSESSMENT_COMPLETE.json',{'status':'COMPLETE','seconds':time.monotonic()-start,'selection':selection,'optimizer_updates_in_assessment':0,'allocated_peak':torch.cuda.max_memory_allocated(),'reserved_peak':torch.cuda.max_memory_reserved(),'RSS_highwater_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
 print('ASSESSMENT_COMPLETE',json.dumps(selection),flush=True)
if __name__=='__main__':main()
