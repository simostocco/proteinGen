"""Serial real AF2 variant inference, fixed query-only inputs; no external MSA."""
from r1d_common import *
import subprocess,threading,dataclasses,numpy as np
from readiness_oracle import backend
from protein_joint_sac.oracle.adapter import import_saved_result
from protein_joint_sac.rl.sequence_replay import OracleWallBudget
from protein_joint_sac.rl.sequence_rewards import sequence_features,mapped_structure_scores

def main():
 assert not (R/'folds/BASELINE_ADEQUACY_STOP.json').exists() or (R/'configs/EXPLORATORY_RANKING_AMENDMENT.json').exists(), 'original protocol stopped; explicit exploratory amendment required';cfg=json.load(open(R/'configs/VARIANT_INFERENCE_FROZEN.json'));refs=json.load(open(R/'configs/NATIVE_REFERENCES.json'));adapter=backend();write('folds/INSTALLATION_VERIFICATION.json',adapter.preflight(verify_weights=True));b=adapter.backend;budget=OracleWallBudget(R/'folds/worker_budget.sqlite',12,1500);rows=json.load(open(R/'ALPHAFOLD_VARIANT_INFERENCE_MANIFEST.json'))['rows'] if (R/'ALPHAFOLD_VARIANT_INFERENCE_MANIFEST.json').exists() else []
 for job in cfg['jobs']:
  if job['id'] in {r['id'] for r in rows}:continue
  guard();out=R/'folds'/job['id'];assert not out.exists(),'unfinished job requires manual recovery; no repeat'
  if job['role']=='AUDIT' and not (R/'analysis/BEFORE_AUDIT_SCORING_LOCK.json').exists():write('analysis/BEFORE_AUDIT_SCORING_LOCK.json',{'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'development_prediction_ids':[r['id'] for r in rows],'no_audit_labels_opened':True,'primary':'pTM, fixed a priori','candidate_formulas_and_gates_sha256':sha(R/'configs/SELECTION_AND_ANALYSIS_PREDECLARED_v2.json'),'inference_protocol_sha256':sha(R/'configs/VARIANT_INFERENCE_FROZEN.json'),'development_labels_sha256':sha(R/'analysis/DEVELOPMENT_LABELS.json'),'audit_labels_sha256':cfg['audit_labels_sha256'],'fitted_coefficients':0,'metric_or_threshold_changes':0})
  try:budget.reserve(job['id'],115)
  except Exception as e:write('folds/BUDGET_STOP.json',{'reason':str(e),'budget':budget.report()});break
  out.mkdir();a3m=Path(job['a3m_path']);assert sha(a3m)==job['a3m_sha256'];proto={'version':'R1d_query_only_variants_v1','wrapper_sha256':sha(__file__),'weights_sha256':b.weights_sha256,'code_revision':b.code_revision,'sequence_sha256':job['sequence_sha256'],'a3m_sha256':sha(a3m),'model':1,'recycles':3,'seed':151502,'templates':False,'relaxation':False,'max_MSA':'1:1','mode':'single_sequence','network':'DENIED','JAX_backends':'cuda,cpu;GPU default and CPU preprocessing'};ph=hashlib.sha256(json.dumps(proto,sort_keys=True).encode()).hexdigest();write(f'folds/{job["id"]}/PROTOCOL.json',proto)
  wrapper="import socket; socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('network disabled during R1d variant folding')); import colabfold.download; colabfold.download.download_alphafold_params=lambda *a,**k: None; import jax; assert jax.default_backend()=='gpu'; from colabfold.batch import main; main()"
  argv=[b.python,'-B','-c',wrapper,str(a3m),str(out/'prediction'),'--model-type','alphafold2_ptm','--model-order','1','--num-models','1','--num-recycle','3','--recycle-early-stop-tolerance','0','--num-seeds','1','--random-seed','151502','--msa-mode','single_sequence','--max-msa','1:1','--num-relax','0','--recompile-padding','0','--data',b.weights_dir,'--disable-unified-memory','--no-use-fast-kernels','--kernel-backend','auto'];write(f'folds/{job["id"]}/ARGV.json',argv);env=dict(os.environ,JAX_PLATFORMS='cuda,cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',TF_FORCE_UNIFIED_MEMORY='0',XLA_PYTHON_CLIENT_MEM_FRACTION='.75',CUDA_VISIBLE_DEVICES='0');samples=[];stop=threading.Event()
  def monitor():
   while not stop.is_set():
    try:samples.append(subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],text=True).strip())
    except Exception:pass
    stop.wait(1)
  t=threading.Thread(target=monitor,daemon=True);t.start();row=dict(job,status='FAILED',protocol_sha256=ph);start=time.monotonic()
  try:
   with (out/'worker.log').open('w') as log:subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,cwd=out,env=env,timeout=115,check=True)
   seconds=time.monotonic()-start;pdb=list((out/'prediction').glob('*unrelaxed_rank_001*.pdb'));scores=list((out/'prediction').glob('*scores_rank_001*.json'));assert len(pdb)==len(scores)==1;res=import_saved_result(job['sequence'],pdb[0],scores[0],ph,151502);write(f'folds/{job["id"]}/RESULT.json',dataclasses.asdict(res));resolved=json.load(open(out/'prediction/config.json'));assert resolved['max_seq']==1 and resolved['max_extra_seq']==1 and resolved['num_recycles']==3 and not resolved['use_templates'] and resolved['random_seed']==151502 and resolved['msa_mode']=='single_sequence';native=refs[job['parent_key']]['native_ca'];row.update(status='SUCCEEDED',worker_wall_seconds=seconds,features=sequence_features(job['sequence'],res.plddt,res.ptm,res.pae,res.predicted_ca),WT_native_fold_compatibility=mapped_structure_scores(native,res.predicted_ca),compatibility_interpretation='WT experimental native conformation reference; not a measured mutant structure or stability label',resolved_config=resolved,predicted_PDB=str(pdb[0]),predicted_PDB_sha256=sha(pdb[0]),scores_path=str(scores[0]),scores_sha256=sha(scores[0]))
  except Exception as e:row.update(error=type(e).__name__+': '+str(e),worker_wall_seconds=time.monotonic()-start)
  finally:stop.set();t.join(3);budget.finish(job['id'],row['worker_wall_seconds'])
  row['GPU_samples']=samples;rows.append(row);write(f'folds/{job["id"]}/RECEIPT.json',row);write('ALPHAFOLD_VARIANT_INFERENCE_MANIFEST.json',{'rows':rows,'budget':budget.report(),'J1_updates':0,'remote_MSA_queries':0,'new_model_downloads':0,'experimental_labels_input_to_oracle':False,'frozen_protocol_sha256':sha(R/'configs/VARIANT_INFERENCE_FROZEN.json')});print(json.dumps({'id':job['id'],'parent':job['parent'],'status':row['status'],'seconds':row['worker_wall_seconds'],'pTM':row.get('features',{}).get('ptm'),'pLDDT':row.get('features',{}).get('mean_pLDDT'),'WT_fold_compatibility':row.get('WT_native_fold_compatibility',{}).get('robust_fixed_map_TM_similarity_fullN'),'error':row.get('error')}),flush=True)
  if row['status']!='SUCCEEDED':break
  if len(rows)==2 and all(r['role']=='DEVELOPMENT' and r['WT'] for r in rows) and all(r['features']['ptm']<.5 and r['WT_native_fold_compatibility']['robust_fixed_map_TM_similarity_fullN']<.5 for r in rows):write('folds/BASELINE_ADEQUACY_STOP.json',{'reason':cfg['baseline_adequacy_rule'],'replacements':0,'remaining_variants_not_run':True});break
if __name__=='__main__':main()
