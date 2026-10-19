"""Bounded local MSA-assisted AF2. Uses persisted public-native A3Ms; network denied."""
import dataclasses,threading,subprocess,os,sys,json,time,hashlib
from pathlib import Path
import numpy as np
from r1b_common import R,PREV,write,sha,guard
from r1b_prepare import comparisons
from readiness_oracle import backend
from protein_joint_sac.oracle.adapter import import_saved_result
from protein_joint_sac.rl.sequence_replay import OracleWallBudget
from protein_joint_sac.rl.sequence_rewards import sequence_features,fit_reward_gate

def config():
 a=backend();audit=a.preflight(verify_weights=True);b=a.backend
 return a,{'version':'R1b_MSA_AF2_v1','wrapper_sha256':sha(__file__),'source_revision':b.code_revision,'weights_sha256':b.weights_sha256,'model':1,'num_models':1,'recycles':3,'recycle_tolerance':0.,'templates':False,'relaxation':False,'seed':151502,'max_MSA':'128:256','MSA_mode':'mmseqs2_uniref_env; persisted A3M bypasses network search','internal_BF16':True,'outputs_FP32':True,'fusion':True,'fast_kernels':False,'kernel_backend':'auto','JAX_backends':'cuda,cpu; default gpu required','network':'DENIED_DURING_FOLD','max_calls':10,'max_worker_seconds':2400,'timeout_seconds_per_call':300,'reservation_seconds_per_call':360,'conditional_rule':'all6 principal calls succeed, median paired fixed-map recovery gain>=.10, at least3/6 MSA recoveries>=.50','conditional_TRAIN':['train_r1b_03','train_r1b_05'],'conditional_VALIDATION':['validation_96','validation_256'],'local_calibration_split':'TRAIN sorted canonical SHA; first n-2 fit, last2 heldout; freeze before VALIDATION; no AFDB/local protocol pooling','native_recovery_gates':'TRAIN median>=.60, fraction>=.50 at least2/3; pTM/recovery Spearman>=.40; VALIDATION median>=.50; heldout TRAIN calibration MAE<=.15, VAL audit MAE<=.20','new_reward_gate_not_relaxed':True,'native_mapping_panel_sha256':sha(R/'configs/NATIVE_PANEL_32.json'),'installed_source_preflight':audit}
def fit_control_scope(rows):
 # Preserve same TRAIN-only confidence-product calibration policy; n-2 TRAIN fits
 # after conditional additions, still two predetermined SHA-ranked held-outs.
 native=sorted([r for r in rows if r.get('split')=='TRAIN' and r.get('status')=='SUCCEEDED'],key=lambda r:r['sequence_sha256']);vals=[r for r in rows if r.get('split')=='VALIDATION' and r.get('status')=='SUCCEEDED'];fail=[]
 if len(native)<6:return {'GO':False,'reason':'fewer than6 successful TRAIN controls'}
 from scipy.stats import spearmanr
 tm=np.array([r['recovery']['robust_fixed_map_TM_similarity_fullN'] for r in native]);ptm=[r['features']['ptm'] for r in native];rho=float(spearmanr(tm,ptm).statistic);fitted=native[:-2];hold=native[-2:];x=np.array([r['features']['sequence_confidence_product'] for r in fitted]);y=np.array([r['recovery']['robust_fixed_map_TM_similarity_fullN'] for r in fitted]);beta=float(max(0,x@y/max(x@x,1e-12)));pred=lambda r:float(np.clip(beta*r['features']['sequence_confidence_product'],0,1));mae=float(np.mean([abs(pred(r)-r['recovery']['robust_fixed_map_TM_similarity_fullN']) for r in hold]));valmae=float(np.mean([abs(pred(r)-r['recovery']['robust_fixed_map_TM_similarity_fullN']) for r in vals])) if vals else None
 if np.median(tm)<.60 or np.mean(tm>=.50)<2/3:fail.append('native TRAIN recovery below original gate')
 if not np.isfinite(rho) or rho<.40:fail.append('pTM/recovery correlation below.40')
 if mae>.15:fail.append('TRAIN holdout calibration MAE>.15')
 if len(vals)<2:fail.append('VALIDATION audit not yet complete')
 elif np.median([r['recovery']['robust_fixed_map_TM_similarity_fullN'] for r in vals])<.50:fail.append('VALIDATION recovery median<.50')
 if valmae is not None and valmae>.20:fail.append('VALIDATION calibration MAE>.20')
 return {'GO':not fail,'failures':fail,'beta_TRAIN_only':beta,'TRAIN_fit_ids':[r['sequence_sha256'] for r in fitted],'TRAIN_holdout_ids':[r['sequence_sha256'] for r in hold],'TRAIN_holdout_MAE':mae,'VALIDATION_audit_MAE':valmae,'TRAIN_median_recovery':float(np.median(tm)),'TRAIN_fraction_atleast_.50':float(np.mean(tm>=.50)),'TRAIN_pTM_recovery_Spearman':rho if np.isfinite(rho) else None,'VALIDATION_fit':False,'scope':'native controls under this MSA protocol only; novel-sequence transfer unverified','production_reward_GO':False}

def main():
 phase=sys.argv[1] if len(sys.argv)>1 else 'principal';adapter,c=config();f=R/'configs/MSA_FOLD_FROZEN.json'
 if f.exists():assert json.load(open(f))==c
 else:write(str(f.relative_to(R)),c)
 panel=json.load(open(R/'configs/NATIVE_PANEL_32.json'));ids=panel['MSA_first_six'] if phase=='principal' else c['conditional_TRAIN']+c['conditional_VALIDATION'];assert phase=='principal' or json.load(open(R/'folds/ALLOW_CONDITIONAL.json'))['allowed'];jobs=sorted([v for v in panel['controls'] if v['id'] in ids],key=lambda v:(v['split']=='VALIDATION',v['N']));budget=OracleWallBudget(R/'folds/worker_budget.sqlite',10,2400);oldpath=R/'MSA_ASSISTED_NATIVE_RECOVERY.json';rows=json.load(open(oldpath))['rows'] if oldpath.exists() else [];done={r['id'] for r in rows};failures=0
 for job in jobs:
  if job['id'] in done:continue
  if job['split']=='VALIDATION' and not (R/'folds/TRAIN_CALIBRATION_BEFORE_VALIDATION.json').exists():write('folds/TRAIN_CALIBRATION_BEFORE_VALIDATION.json',fit_control_scope(rows))
  receipt=json.load(open(R/'msa'/job['id']/'RECEIPT.json'))
  if receipt['status']!='SUCCEEDED':continue
  a3m=Path(receipt['a3m_path']);assert sha(a3m)==receipt['a3m_sha256'];guard();out=R/'folds'/job['id']
  if out.exists():raise RuntimeError('Existing unfinished call requires inspection; no duplicate restart')
  budget.reserve(job['id'],360);out.mkdir();proto=dict(c,a3m_sha256=sha(a3m),canonical_sequence_sha256=job['sequence_id']);protocolsha=hashlib.sha256(json.dumps(proto,sort_keys=True).encode()).hexdigest();write(str((out/'PROTOCOL.json').relative_to(R)),proto);b=adapter.backend
  wrapper="import socket; socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('network disabled during local MSA folding')); import colabfold.download; colabfold.download.download_alphafold_params=lambda *a,**k: None; import jax; assert jax.default_backend()=='gpu'; from colabfold.batch import main; main()"
  argv=[b.python,'-B','-c',wrapper,str(a3m),str(out/'prediction'),'--model-type','alphafold2_ptm','--model-order','1','--num-models','1','--num-recycle','3','--recycle-early-stop-tolerance','0','--num-seeds','1','--random-seed','151502','--msa-mode','mmseqs2_uniref_env','--max-msa','128:256','--num-relax','0','--recompile-padding','0','--data',b.weights_dir,'--disable-unified-memory','--no-use-fast-kernels','--kernel-backend','auto'];write(str((out/'ARGV.json').relative_to(R)),argv);env=dict(os.environ,JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',TF_FORCE_UNIFIED_MEMORY='0',XLA_PYTHON_CLIENT_MEM_FRACTION='.75',CUDA_VISIBLE_DEVICES='0');telemetry=[];stop=threading.Event()
  def monitor():
   while not stop.is_set():
    try:telemetry.append(subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],text=True).strip())
    except Exception:pass
    stop.wait(1)
  thread=threading.Thread(target=monitor,daemon=True);thread.start();begin=time.monotonic();row={'id':job['id'],'kind':'native','split':job['split'],'N':job['N'],'sequence_sha256':job['sequence_id'],'protocol_sha256':protocolsha,'a3m_sha256':sha(a3m),'status':'FAILED','MSA_depth':receipt['metrics']}
  try:
   with (out/'worker.log').open('w') as log:subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,cwd=out,env=env,timeout=300,check=True)
   pdbs=list((out/'prediction').glob('*unrelaxed_rank_001*.pdb'));scores=list((out/'prediction').glob('*scores_rank_001*.json'));assert len(pdbs)==len(scores)==1;res=import_saved_result(job['sequence'],pdbs[0],scores[0],protocolsha,151502);row.update(status='SUCCEEDED',features=sequence_features(job['sequence'],res.plddt,res.ptm,res.pae,res.predicted_ca),recovery=comparisons(job,res.predicted_ca,'MSA_'+job['id']),RESULT_path=str(out/'RESULT.json'));write(str((out/'RESULT.json').relative_to(R)),dataclasses.asdict(res));resolved=json.load(open(out/'prediction/config.json'));assert resolved['max_seq']==128 and resolved['max_extra_seq']==256 and not resolved['use_templates'] and resolved['num_recycles']==3 and resolved['random_seed']==151502;row['resolved_config']=resolved;failures=0
  except Exception as e:row['failure_reason']=type(e).__name__+': '+str(e);failures+=1
  finally:stop.set();thread.join(timeout=3);row['worker_wall_seconds']=time.monotonic()-begin;row['GPU_samples']=telemetry;budget.finish(job['id'],row['worker_wall_seconds'])
  rows.append(row);write(str((out/'RECEIPT.json').relative_to(R)),row);write('MSA_ASSISTED_NATIVE_RECOVERY.json',{'rows':rows,'budget':budget.report(),'J1_optimizer_updates':0,'no_TEST':True});print(json.dumps({'id':job['id'],'status':row['status'],'seconds':row['worker_wall_seconds'],'mapped_TM':row.get('recovery',{}).get('robust_fixed_map_TM_similarity_fullN'),'pTM':row.get('features',{}).get('ptm'),'failure':row.get('failure_reason')}),flush=True)
  if failures>=2 or row.get('failure_reason','').startswith('TimeoutExpired'):break
 if phase=='principal':
  old=json.load(open(PREV/'ALPHAFOLD_NATIVE_CONTROL_RESULTS.json'))['rows'];pairs=[]
  for r in rows:
   if r['status']=='SUCCEEDED' and r['id'] in panel['MSA_first_six']:
    baseline=next(v for v in old if v['id']==r['id'] and v['status']=='SUCCEEDED');pairs.append(r['recovery']['robust_fixed_map_TM_similarity_fullN']-baseline['recovery']['robust_fixed_map_TM_similarity_fullN'])
  allowed=(len(pairs)==6 and float(np.median(pairs))>=.10 and sum(r['recovery']['robust_fixed_map_TM_similarity_fullN']>=.5 for r in rows if r['status']=='SUCCEEDED')>=3);write('folds/ALLOW_CONDITIONAL.json',{'allowed':allowed,'principal_successes':len(pairs),'median_paired_recovery_gain':float(np.median(pairs)) if pairs else None,'rule':c['conditional_rule'],'additional_ids':c['conditional_TRAIN']+c['conditional_VALIDATION'] if allowed else [],'budget':budget.report()})
 write('folds/REWARD_REASSESSMENT.json',fit_control_scope(rows))
if __name__=='__main__':main()
