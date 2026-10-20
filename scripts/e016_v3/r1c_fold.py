"""Twelve serial local depth-ablation folds, no network, hard charged-time cap."""
from r1c_common import *
import subprocess,threading,dataclasses,numpy as np
from readiness_oracle import backend
from r1b_prepare import comparisons
from protein_joint_sac.oracle.adapter import import_saved_result
from protein_joint_sac.rl.sequence_replay import OracleWallBudget
from protein_joint_sac.rl.sequence_rewards import sequence_features

def main():
 (R/'alignment').mkdir(exist_ok=True);cfg=json.load(open(R/'configs/DEPTH_STUDY_FROZEN.json'));controls={c['id']:c for c in json.load(open(R/'configs/NATIVE_CONTROLS.json'))['controls']};adapter=backend();audit=adapter.preflight(verify_weights=True);b=adapter.backend;write('folds/INSTALLATION_VERIFICATION.json',audit)
 budget=OracleWallBudget(R/'folds/worker_budget.sqlite',12,1500);rows=json.load(open(R/'MSA_DEPTH_ABLATION.json'))['rows'] if (R/'MSA_DEPTH_ABLATION.json').exists() else []
 for job in cfg['jobs']:
  if job['id'] in {v['id'] for v in rows}:continue
  guard();out=R/'folds'/job['id'];assert not out.exists(),'unfinished call must be audited, no automatic repeat'
  # Every reservation is the enforced per-worker timeout, never merely a nominal estimate.
  report=budget.report();remaining=1500-report.get('charged_seconds',report.get('seconds_charged',0))
  # Database implementation is authoritative; reserve refuses insufficient capacity.
  timeout=115
  try:budget.reserve(job['id'],timeout)
  except Exception as e:write('folds/BUDGET_STOP.json',{'reason':str(e),'report':budget.report()});break
  out.mkdir();c=controls[job['control']];a3m=Path(job['a3m_path']);assert sha(a3m)==job['a3m_sha256']
  proto={'version':'R1c_depth_v1','wrapper_sha256':sha(__file__),'source_revision':b.code_revision,'weights_sha256':b.weights_sha256,'seed':151502,'max_MSA':'128:256','recycles':3,'model':1,'templates':False,'a3m_sha256':sha(a3m),'sequence_sha256':c['sequence_id'],'network':'DENIED','JAX':'cuda,cpu; GPU default; CPU feature preprocessing'};ph=hashlib.sha256(json.dumps(proto,sort_keys=True).encode()).hexdigest();write(f'folds/{job["id"]}/PROTOCOL.json',proto)
  wrapper="import socket; socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('network disabled during local R1c folding')); import colabfold.download; colabfold.download.download_alphafold_params=lambda *a,**k: None; import jax; assert jax.default_backend()=='gpu'; from colabfold.batch import main; main()"
  argv=[b.python,'-B','-c',wrapper,str(a3m),str(out/'prediction'),'--model-type','alphafold2_ptm','--model-order','1','--num-models','1','--num-recycle','3','--recycle-early-stop-tolerance','0','--num-seeds','1','--random-seed','151502','--msa-mode','mmseqs2_uniref_env','--max-msa','128:256','--num-relax','0','--recompile-padding','0','--data',b.weights_dir,'--disable-unified-memory','--no-use-fast-kernels','--kernel-backend','auto'];write(f'folds/{job["id"]}/ARGV.json',argv)
  env=dict(os.environ,JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',TF_FORCE_UNIFIED_MEMORY='0',XLA_PYTHON_CLIENT_MEM_FRACTION='.75',CUDA_VISIBLE_DEVICES='0');samples=[];stop=threading.Event()
  def monitor():
   while not stop.is_set():
    try:samples.append(subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],text=True).strip())
    except Exception:pass
    stop.wait(1)
  t=threading.Thread(target=monitor,daemon=True);t.start();row=dict(job,status='FAILED',sequence_sha256=c['sequence_id'],protocol_sha256=ph);start=time.monotonic()
  try:
   with (out/'worker.log').open('w') as log:subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,cwd=out,env=env,timeout=timeout,check=True)
   seconds=time.monotonic()-start
   pdb=list((out/'prediction').glob('*unrelaxed_rank_001*.pdb'));scores=list((out/'prediction').glob('*scores_rank_001*.json'));assert len(pdb)==len(scores)==1
   res=import_saved_result(c['sequence'],pdb[0],scores[0],ph,151502);write(f'folds/{job["id"]}/RESULT.json',dataclasses.asdict(res));resolved=json.load(open(out/'prediction/config.json'));assert resolved['max_seq']==128 and resolved['max_extra_seq']==256 and resolved['num_recycles']==3 and not resolved['use_templates'] and resolved['random_seed']==151502
   row.update(status='SUCCEEDED',worker_wall_seconds=seconds,features=sequence_features(c['sequence'],res.plddt,res.ptm,res.pae,res.predicted_ca),recovery=comparisons(c,res.predicted_ca,job['id']),resolved_config=resolved)
  except Exception as e:row.update(error=type(e).__name__+': '+str(e),worker_wall_seconds=time.monotonic()-start)
  finally:stop.set();t.join(3);budget.finish(job['id'],row['worker_wall_seconds'])
  row['GPU_samples']=samples;rows.append(row);write(f'folds/{job["id"]}/RECEIPT.json',row);write('MSA_DEPTH_ABLATION.json',{'rows':rows,'budget':budget.report(),'remote_MSA_queries':0,'J1_optimizer_updates':0,'freeze_sha256':sha(R/'configs/DEPTH_STUDY_FROZEN.json')})
  print(json.dumps({'id':row['id'],'status':row['status'],'seconds':row['worker_wall_seconds'],'recovery':row.get('recovery',{}).get('robust_fixed_map_TM_similarity_fullN'),'pTM':row.get('features',{}).get('ptm'),'error':row.get('error')}),flush=True)
  if row['status']!='SUCCEEDED':break
if __name__=='__main__':main()
