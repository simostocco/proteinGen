"""Selected-checkpoint assessment: VALIDATION and native TRAIN only, never TEST."""
import collections,json,math,resource,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
from partial_train import Engine,R,OLD,write,core
from protein_joint_sac.data.partial_continuation import PartialData
from protein_joint_sac.data.certified_unified import file_sha
from protein_joint_sac.rl.checkpointing import load_checked
from protein_joint_sac.data.partial_geometry import derive
ARMS=['J1-FULL','J1-PARTIAL','J2-FULL','J2-PARTIAL'];LENGTHS=[64,128,256,384,500]
def text(name,value):
 p=R/name;q=p.with_name(p.name+'.part');q.write_text(value);q.replace(p)
def summary(values):
 a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
 return {'n':len(a),'mean':float(a.mean()) if len(a) else None,'median':float(np.median(a)) if len(a) else None,'q05_q95':np.quantile(a,[.05,.95]).tolist() if len(a) else None}
def metrics(x):
 x=np.asarray(x,dtype=float);N=len(x)
 if N<4 or not np.isfinite(x).all():raise ValueError('invalid complete trace')
 g,m,seg=derive(x,np.ones(N,bool),np.ones(N-1,bool));d=np.linalg.norm(x[:,None]-x[None,:],axis=-1);sep=np.abs(np.arange(N)[:,None]-np.arange(N)[None,:]);nonlocal_mask=np.triu(sep>=3,1);long=np.triu(sep>=24,1);clashes=int(((d<2)&nonlocal_mask).sum());rg=float(np.sqrt(np.square(x-x.mean(0)).sum(1).mean()));angles=g[m[:,1],1];tau=g[m[:,2],2]
 b=np.diff(x,axis=0);triple=(np.cross(b[:-2],b[1:-1])*b[2:]).sum(-1)/np.maximum(np.linalg.norm(b[:-2],axis=-1)*np.linalg.norm(b[1:-1],axis=-1)*np.linalg.norm(b[2:],axis=-1),1e-300)
 return {'torsion_positive_fraction':float((tau>0).mean()),'signed_triple_product_mean':float(triple.mean()),'N':N,'adjacent_A':summary(g[m[:,0],0]),'angle_rad':summary(angles),'torsion_circular_mean_rad':float(np.arctan2(np.sin(tau).mean(),np.cos(tau).mean())),'torsion_resultant':float(np.hypot(np.sin(tau).mean(),np.cos(tau).mean())),'angle_histogram':np.histogram(angles,bins=np.linspace(0,np.pi,19))[0].tolist(),'torsion_histogram':np.histogram(tau,bins=np.linspace(-np.pi,np.pi,37))[0].tolist(),'nonlocal_clashes_below2A_seqsep_ge3':clashes,'clashes_per100_residues':100*clashes/N,'clashes_per10000_nonlocal_pairs':10000*clashes/max(1,int(nonlocal_mask.sum())),'long_contacts_below8A_seqsep_ge24':int(((d<8)&long).sum()),'long_contact_fraction':float(((d<8)&long).sum()/max(1,long.sum())),'Rg_A':rg,'Rg_over_N_cuberoot':rg/N**(1/3),'degenerate_frames':int((np.linalg.norm(np.cross(np.diff(x,axis=0)[:-1],np.diff(x,axis=0)[1:]),axis=-1)<1e-8).sum())}
def native(data):
 out={}
 for split in ['TRAIN','VALIDATION']:
  out[split]={}
  for N in LENGTHS:
   ids=[h for h in data.base.jointids[split] if max(20,round(.9*N))<=len(data.seqs[h])<=min(500,round(1.1*N))];rng=np.random.default_rng(742+N);ids=sorted(ids);chosen=[ids[i] for i in rng.permutation(len(ids))[:128]];rows=[]
   for h in chosen:
    oid=data.base.owners[h][0];x=data.base.ca[oid]
    rows.append({'sequence_hash':h,'observation_id':oid,'metrics':metrics(x)})
   out[split][str(N)]={'length_window':[max(20,round(.9*N)),min(500,round(1.1*N))],'available_identities':len(ids),'selected_identities':len(rows),'rows':rows}
 return out

def sample_panel(arm,config,checkpoint,expected_sha,cache,data,train_strings,reuse):
 items=json.loads(cache.read_text()) if cache.exists() else []
 expected={(N,j) for N in LENGTHS for j in range(32)}
 keys={(v['N'],v['sample_index']) for v in items}
 if len(keys)!=len(items) or not keys.issubset(expected) or any(v['checkpoint_sha256']!=expected_sha or v['seed']!=2026101700+100*v['N']+v['sample_index'] for v in items):raise ValueError('generation cache lineage/panel')
 eng=None
 for N in LENGTHS:
  for j in range(32):
   if (N,j) in keys:continue
   key=(expected_sha,N,j)
   if key in reuse:result=dict(reuse[key]);result['prediction_cache_reused']=True
   else:
    if eng is None:
     eng=Engine(arm,data,config);saved=load_checked(checkpoint);eng.model.load_state_dict(saved['model'],strict=True);del saved
    result=eng.generate(N,2026101700+100*N+j);result.update(sample_index=j,checkpoint_sha256=expected_sha)
    if result['status']=='COMPLETE':
     try:result['independent_CA_metrics']=metrics(result['generated_CA_A'])
     except ValueError as ex:result['assessment_failure']=repr(ex)
     result['exact_TRAIN_sequence_match']=result['sequence'] in train_strings
   items.append(result);write(cache,items);write(R/'ASSESSMENT_HEARTBEAT.json',{'arm':arm,'N':N,'sample':j,'timestamp':time.time()})
 if eng is not None:del eng;torch.cuda.empty_cache()
 for v in items:reuse[(expected_sha,v['N'],v['sample_index'])]=v
 return items

def main():
 core.setup();start=time.monotonic();cfg=json.load(open(R/'configs/CONTINUATION_FROZEN.json'));done={a:json.load(open(R/'runs'/a/'COMPLETED.json')) for a in ARMS}
 for a,v in done.items():
  if v['config']!=cfg[a] or file_sha(v['selected_checkpoint'])!=v['selected_sha256']:raise ValueError('selected checkpoint SHA mismatch '+a)
 write(R/'SELECTION_FROZEN.json',{a:{'path':v['selected_checkpoint'],'sha256':v['selected_sha256'],'criterion':v['config']['selection']} for a,v in done.items()})
 data=PartialData();curves={};valid={};generated={};reuse={};train_strings={''.join(data.seqs[h]) for h in data.proposal.sequences if data.proposal.sequences[h]['split']=='TRAIN'}
 for arm in ARMS:
  d=R/'runs'/arm;by={}
  for p in sorted(d.glob('updates_from_*.jsonl')):
   for line in p.read_text().splitlines():
    if line.strip():v=json.loads(line);by[v['update']]=v
  curves[arm]=[by[k] for k in sorted(by)];valid[arm]={str(int(p.stem.split('_')[-1])):json.loads(p.read_text()) for p in sorted(d.glob('validation_*.json'))}
  generated[arm]=sample_panel(arm,cfg[arm],done[arm]['selected_checkpoint'],done[arm]['selected_sha256'],d/'SELECTED_GENERATION.json',data,train_strings,reuse)
 for name in ['J1','J2']:
  config=cfg[name+'-FULL'];checkpoint=OLD/'runs'/name/'checkpoint_10000.pt'
  if file_sha(checkpoint)!=config['parent_sha256']:raise ValueError('original generation baseline hash')
  generated['ORIGINAL-'+name]=sample_panel(name+'-FULL',config,checkpoint,config['parent_sha256'],R/('ORIGINAL_'+name+'_GENERATION.json'),data,train_strings,reuse)
 write(R/'CONTINUATION_TRAINING_CURVES.json',curves);write(R/'COMPLETE_VS_PARTIAL_VALIDATION.json',valid)
 native_path=R/'NATIVE_VS_GENERATED_GEOMETRY.json'
 if not native_path.exists():write(native_path,{'metric_definitions':'same CA-only metrics; nonlocal separation>=3/clash<2A; long separation>=24/contact<8A; no atomistic or folding inference','native':native(data),'generated':{a:[{k:v for k,v in x.items() if k not in ['generated_CA_A']} for x in items] for a,items in generated.items()}})
 pairs={}
 for name in ['J1','J2']:
  f=done[name+'-FULL'];p=done[name+'-PARTIAL'];pairs[name]={'FULL_selected_complete_objective':f['best_common_complete_objective'],'PARTIAL_selected_complete_objective':p['best_common_complete_objective'],'PARTIAL_minus_FULL':p['best_common_complete_objective']-f['best_common_complete_objective'],'single_seed_caveat':True,'original_10k_complete_objective':valid[name+'-FULL']['0']['complete']['selection_objective']}
  uf={v['update']:v for v in curves[name+'-FULL']};up={v['update']:v for v in curves[name+'-PARTIAL']}
  if 10000 in uf and 20000 in up:pairs[name]['equal_complete_exposure_stream_verified']=uf[10000]['complete_stream_sha256']==up[20000]['complete_stream_sha256']
 lines=['# Partial-supervision continuation results','', 'Measured results use VALIDATION only; TEST was not reevaluated. No AlphaFold, SAC or E015 executed.','', '| Arm | Additional updates | New decisions | Best common complete VALIDATION objective | Selected checkpoint |','|---|---:|---:|---:|---|']
 for a,v in done.items():lines.append(f"| {a} | {v['additional_updates']} | {v['additional_prefix_decisions']} | {v['best_common_complete_objective']:.6f} | {v['selected_checkpoint']} |")
 lines+=['','## Matched comparisons','',json.dumps(pairs,indent=2),'','Teacher-forced likelihood and CA proxies do not establish native fold quality, experimental foldability or thermodynamic stability. One-seed development comparisons retain residual sequence-relatedness limitations of the accepted split.']
 text('FINAL_PARTIAL_SUPERVISION_REPORT.md','\n'.join(lines)+'\n');text('FINAL_RESEARCH_ASSESSMENT.md','\n'.join(lines)+'\n')
 generation_summary={a:{str(N):{'attempts':32,'complete':sum(v['status']=='COMPLETE' for v in items if v['N']==N),'unique_complete_sequences':len({v['sequence'] for v in items if v['N']==N and v['status']=='COMPLETE'}),'seconds':sum(v['seconds'] for v in items if v['N']==N),'clashes_per100residues':summary([v['independent_CA_metrics']['clashes_per100_residues'] for v in items if v['N']==N and 'independent_CA_metrics' in v])} for N in LENGTHS} for a,items in generated.items()}
 text('FREE_GENERATION_COMPARISON.md','# Free-running assessment\n\n32 fixed-seed attempts per selected model and untouched 10k parent baseline at each length; failures retained. Identical checkpoint/seed predictions are reused with an explicit receipt. Native comparison uses length-windowed TRAIN/VALIDATION, never TEST.\n\n```json\n'+json.dumps(generation_summary,indent=2)+'\n```\nNo experimentally foldable-protein claim is supported.\n')
 text('CHECKPOINT_AND_RESUME_REPORT.md','# Checkpoint recovery\n\nOriginal J1/J2 10k checkpoints were SHA-verified and preserved. Every new branch starts from its corresponding original model/optimizer; only the continuation scheduler restarts. New checkpoints bind source/config/view, RNG and sampler streams. Actual mixed-data restart proof: tests/CONTINUATION_RESTART.json. New-task retention keeps best/latest with checksum-recorded deletion receipts; historical evidence is untouched.\n')
 write(R/'RESOURCE_AND_STORAGE_REPORT.json',{'finalization_seconds':time.monotonic()-start,'D_free_bytes':shutil.disk_usage(R).free,'finalizer_RSS_highwater':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'runs':{a:{'elapsed_seconds':v['elapsed_seconds'],'resources':v['resources']} for a,v in done.items()},'generation_summary':generation_summary,'historical_deletions':0})
 write(R/'MASK_CORRECTNESS_TEST_RESULTS.json',{'CPU':'tests/MASK_CPU_FINAL.xml','recovery_CPU':'tests/MASK_RECOVERY.xml','broader_CPU':'tests/REGRESSION_RECOVERY.xml','historical_fixture_root_rerun':'tests/REAL_SOURCE_RECOVERY.xml','view_QA_unit':'tests/VIEW_QA_UNIT.xml','CUDA':json.load(open(R/'tests/GPU_MASK_PROOF.json')),'actual_mixed_sampler_restart':json.load(open(R/'tests/CONTINUATION_RESTART.json'))})
 text('MISSING_DATA_MASK_SPEC.md','# Missing-data semantics\n\nKnown full-polymer amino-acid identities remain present at missing CA positions. CA, consecutive C-N links, local frames and each geometry target have independent masks. Distance/angle/torsion require 2/3/4 observed contiguous positions. No coordinate interpolation or cross-gap bonds. Prefix input contains only positions before the target; pair geometry requires observed endpoints. Missing-coordinate placeholders have no physical meaning. Original action densities/Jacobians and bootstrap semantics are preserved. BF16 Transformer plus FP32 geometry projection passed the recorded FP32 gradient comparison; FP64 geometry invariants remain.\n')
 text('EXPANDED_TRAIN_VALIDATION_MANIFESTS.md','# Expanded reference view\n\nview/VIEW_FROZEN.json seals the parent split, partial observation references, inherited ownership and census. No raw coordinate archive was duplicated. Complete observations are unchanged; partial observations require exact source/model/author/sequence/CA/conformer correspondence and positively recovered C-N continuity. Shared batches can decode unselected rows; TEST is excluded from admission and evaluation.\n')
 text('MODEL_AND_SAMPLING_CONFIGS.md','# Frozen model and sampling configuration\n\nconfigs/CONTINUATION_FROZEN.json and configs/VALIDATION_PANELS.json contain exact settings. Architecture: 12,351,746 trainable parameters; width352,8blocks,8heads,FFN4. J1/J2 preserve chemistry off/on. Effective64/live16; FULL complete-only; PARTIAL32complete+32partial identity-first decisions. AdamW parent moments restored; new warmup100,peak1e-4,cosine to1e-5. Common complete VALIDATION selection includes initialization.\n')
 (R/'ASSESSMENT_DONE.json').write_text(json.dumps({'status':'COMPLETED','time':time.time()}))
 print('ASSESSMENT_COMPLETE_CHECKSUM_PUBLICATION_PENDING',flush=True);time.sleep(12)
 paths=[p for p in R.rglob('*') if p.is_file() and not any(k in p.parts for k in ['.git','runtime']) and p.name!='PROVENANCE_SHA256.json'];write(R/'PROVENANCE_SHA256.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip(),'artifacts':{str(p.relative_to(R)):file_sha(p) for p in paths if p.suffix not in ['.pt']},'checkpoints':{a:{'path':v['selected_checkpoint'],'sha256':v['selected_sha256']} for a,v in done.items()}})
if __name__=='__main__':main()
