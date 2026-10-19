"""R1: bounded real local AF2 calls; no policy optimization, network or downloads."""
import dataclasses,hashlib,json,math,os,shutil,subprocess,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from protein_joint_sac.contracts import AA,OracleProtocol,OracleStatus,digest
from protein_joint_sac.oracle.adapter import AF2Backend,RealAlphaFoldAdapter,file_hash
from protein_joint_sac.oracle.queue import EvaluationQueue
from protein_joint_sac.rl.sequence_replay import OracleWallBudget
from protein_joint_sac.rl.sequence_rewards import sequence_features,mapped_structure_scores,fit_reward_gate
R=Path(__file__).resolve().parents[3];D=R.parent;TOOLS=Path('/mnt/d/Simone/proteinGen/e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1/external_designability_loop_v1/tools');HIST=D/'e016_v3_blocker_resolution_20261014_v1'
def write(name,value):
 p=R/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.part');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(tmp,p)
class SequenceQueue(EvaluationQueue):
 def enqueue_sequence(self,identity,sequence,association,protocol,seed):
  if not 20<=len(sequence)<=500 or set(sequence)-set(AA) or seed not in protocol.seeds:raise ValueError('complete known sequence/seed')
  seq=hashlib.sha256(sequence.encode()).hexdigest();keys=(seq,association,protocol.protocol_sha256,seed);self.db.execute('BEGIN IMMEDIATE')
  try:
   old=self.db.execute('SELECT sequence_sha,geometry_sha,protocol_sha,seed FROM requests WHERE episode=?',(identity,)).fetchone()
   if old and old!=keys:raise ValueError('immutable oracle job mismatch')
   if not old:
    if self.db.execute("SELECT count(*) FROM requests WHERE status IN ('PENDING','RUNNING')").fetchone()[0]>=self.max_requests:raise ValueError('bounded queue')
    cached=self.db.execute('SELECT result FROM predictions WHERE cache_key=?',(self.key(seq,protocol.protocol_sha256,seed),)).fetchone();self.db.execute('INSERT INTO requests(episode,sequence_sha,geometry_sha,protocol_sha,seed,status,result,evidence) VALUES(?,?,?,?,?,?,?,?)',(identity,*keys,'SUCCEEDED' if cached else 'PENDING',cached[0] if cached else None,'EXTERNAL_PREDICTION'))
   self.db.execute('COMMIT')
  except Exception:self.db.execute('ROLLBACK');raise
class ReadinessAdapter(RealAlphaFoldAdapter):
 def prepare(self,sequence,seed,output):
  p=super().prepare(sequence,seed,output);p['argv']+=['--disable-unified-memory','--no-use-fast-kernels','--kernel-backend','auto'];return p
def backend():
 protocol=OracleProtocol('LOCAL_COLABFOLD_AF2_PTM','efbf31c37cedb38cd09c69c1b991910a9866480e',('5e564f79af5bcd54ccef6e2a6bb0ff01015d01650ebc41d4575e35f0de9ecc84',),'single_sequence','none',(151502,151503),3,'PLANNED_UNCALIBRATED')
 source=TOOLS/'ColabFold/source';seal=HIST/'ORACLE_SOURCE_SEAL.json';b=AF2Backend(str(TOOLS/'colabfold_env/bin/python'),str(source),protocol.code_revision,file_hash(source/'colabfold/batch.py'),str(TOOLS/'alphafold_weights'),protocol.weights_sha256[0],str(seal),file_hash(seal),precision_contract='R1_v2: internalBF16 true; outputFP32; fuse true; fastkernels false; kernelauto; unified memory disabled; JAX preallocation false; maxMSA1:1; no templates or remote MSA')
 return ReadinessAdapter(protocol,b)
def score(job,result):
 z={'id':job['id'],'kind':job['kind'],'split':job['split'],'N':len(job['sequence']),'seed':job['seed'],'status':result.status.value,'sequence_sha256':result.sequence_sha256,'output_sha256':result.predictor_output_sha256,'failure_reason':result.failure_reason}
 if result.status==OracleStatus.SUCCEEDED:
  z['features']=sequence_features(job['sequence'],result.plddt,result.ptm,result.pae,result.predicted_ca)
  if job.get('native_ca') is not None:z['recovery']=mapped_structure_scores(np.array([v if v is not None else [np.nan]*3 for v in job['native_ca']],float),result.predicted_ca)
  if job.get('proposed_ca') is not None:z['J1_AF_agreement']=mapped_structure_scores(job['proposed_ca'],result.predicted_ca)
 return z
def main():
 if (R/'R1_COMPLETED.json').exists():print('R1 already completed; no duplicate calls.');return
 adapter=backend();audit=adapter.preflight(verify_weights=True);b=adapter.backend
 expected=json.load(open(HIST/'ORACLE_SOURCE_SEAL.json'));assert file_hash(expected['license_path'])==expected['license_sha256'];audit.update({'backend':dataclasses.asdict(b),'declared_protocol':dataclasses.asdict(adapter.protocol),'complete_protocol_sha256':adapter.bound_protocol.protocol_sha256,'selected_weights_sha256':b.weights_sha256,'license_path':expected['license_path'],'license_sha256':expected['license_sha256'],'license_notice':expected['license_first_line'],'scope':'local research with recorded NC-SA parameter terms; not assertion of unrestricted use','external_MSA_calls':0,'downloads':0,'historical_caution':'331/447 single-sequence controls failed structural recovery; no implication native proteins are bad'})
 write('ALPHAFOLD_INSTALLATION_AUDIT.json',audit)
 panel=json.load(open(R/'configs/NATIVE_CONTROL_PANEL.json'));jobs=panel['controls'];train=[j for j in jobs if j['split']=='TRAIN' and j['kind']=='native'];variants=[j for j in jobs if j['kind']=='variant'];repeats=[j for j in jobs if j['kind']=='repeat'];val=[j for j in jobs if j['split']=='VALIDATION']
 for label in ['J1','fallback']:
  p=json.load(open(R/f'generated/{label}_initial64.json'));jobs.append({'id':label+'_generated64','kind':'generated','split':'GENERATED_NOT_PDB','sequence':p['sequence'],'N':64,'proposed_ca':p['ca'],'native_ca':None,'seed':151502,'policy_sha256':p['policy_sha256']})
 generated=[j for j in jobs if j['kind']=='generated'];order=sorted(train,key=lambda j:j['N'])+variants+repeats+val+generated
 config={'backend':dataclasses.asdict(b),'protocol':dataclasses.asdict(adapter.protocol),'bound_protocol_sha256':adapter.bound_protocol.protocol_sha256,'controls_panel_sha256':file_hash(R/'configs/NATIVE_CONTROL_PANEL.json'),'max_calls':16,'max_aggregate_charged_seconds':2700,'timeout_per_call':240,'reserved_per_call':360,'native_GO_rules':panel['GO_rules'],'oracle_GPU_serialized':True,'sequence_reward_input':'complete sequence only, no J1 scaffold/template','VALIDATION_fit':False,'TRAIN_reward_fit':{'fit_native_identities':4,'held_out_TRAIN_native_identities':2,'confidence_product_beta':'nonnegative least squares scalar on TRAIN fit identities only','held_out_TRAIN_MAE_max':.15,'VALIDATION_audit_MAE_max':.20},'candidate_sequence_score':'local confidence and pTM/long-range PAE calibrated jointly, weights unsealed until native gates','geometry_reward':'separate additive bounded auxiliary; never multiply sequence confidence by J1 agreement'}
 if (R/'configs/R1_FROZEN.json').exists():assert json.load(open(R/'configs/R1_FROZEN.json'))==config
 else:write('configs/R1_FROZEN.json',config)
 queue=SequenceQueue(R/'replay/oracle.sqlite',max_requests=32,max_calls=16);recovered=queue.recover_running();budget=OracleWallBudget(R/'replay/oracle_wall.sqlite',16,2700);rows=[];consecutive_fail=0;decision_frozen=False;begin=time.monotonic();resources=[]
 env=os.environ;env['JAX_PLATFORMS']='cuda';env['XLA_PYTHON_CLIENT_PREALLOCATE']='false';env['TF_FORCE_UNIFIED_MEMORY']='0';env['XLA_PYTHON_CLIENT_MEM_FRACTION']='.75';env['CUDA_VISIBLE_DEVICES']='0'
 for job in order:
  if job in val and not decision_frozen:
   pre=fit_reward_gate([v for v in rows if v['split']!='VALIDATION'],[]);pre['VALIDATION_not_yet_evaluated']=True;pre['fit_weights']=pre['candidate_TRAIN_only_fit'];write('TRAIN_ONLY_REWARD_DECISION_BEFORE_VALIDATION.json',pre);decision_frozen=True
  if shutil.disk_usage(R).free<10*2**30+2*2**30:write('R1_RESOURCE_STOP.json',{'reason':'10GiB reserve plus2GiB oracle scratch gate','budget':budget.report()});break
  if (R/'STOP_REQUESTED').exists():break
  ident=job['id'];association=digest(['sequence_control_no_J1_proposal',ident]) if job.get('proposed_ca') is None else digest(job['proposed_ca']);queue.enqueue_sequence(ident,job['sequence'],association,adapter.bound_protocol,job['seed']);request=queue.row(ident)
  # Never rerun failed/time-out calls automatically after a crash.
  if request['status'] not in ['PENDING','SUCCEEDED']:
   data=json.loads(request['result']);data['status']=OracleStatus(data['status']);from protein_joint_sac.contracts import OracleResult;result=OracleResult(**data);z=score(job,result);rows.append(z);continue
  holder=SimpleNamespace(sequence=job['sequence'],sequence_sha256=hashlib.sha256(job['sequence'].encode()).hexdigest(),geometry_sha256=association);out=R/'oracle'/ident;old_cached=request['status']=='SUCCEEDED'
  if not old_cached:budget.reserve(ident,360)
  t=time.monotonic();telemetry=[]
  # Background telemetry is CPU-only and has no additional GPU workload.
  import threading
  stop=threading.Event()
  def monitor():
   while not stop.is_set():
    try:telemetry.append(subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],text=True).strip())
    except Exception:pass
    stop.wait(1)
  mon=threading.Thread(target=monitor,daemon=True);mon.start()
  approval={'protocol_sha256':adapter.bound_protocol.protocol_sha256,'resource_preflight_approved':True,'precision_reviewed':True,'approved_calls':16,'length_ceiling':500,'timeout_seconds':240,'exclusive_gpu_lock':str(R/'runtime/GPU_EXCLUSIVE.lock')}
  try:result=adapter.evaluate(holder,job['seed'],out,approval,queue=queue,episode=ident)
  finally:
   stop.set();mon.join(timeout=3)
   if not old_cached:budget.finish(ident,time.monotonic()-t)
  z=score(job,result);z['worker_wall_seconds']=time.monotonic()-t;z['cache_reused']=old_cached
  if result.status==OracleStatus.SUCCEEDED:
   config_files=list((out/'prediction').glob('config.json'))
   if not old_cached:
    assert len(config_files)==1;resolved=json.load(open(config_files[0]));z['resolved_config']=resolved
    assert resolved.get('model_type')=='alphafold2_ptm' and resolved.get('msa_mode')=='single_sequence' and not resolved.get('use_templates')
   outputs={str(p.relative_to(R)):file_hash(p) for p in out.rglob('*') if p.is_file() and 'runtime' not in p.parts};write('oracle/'+ident+'/ARTIFACT_SHA256.json',outputs);z['artifact_manifest_sha256']=file_hash(out/'ARTIFACT_SHA256.json')
  z['GPU_samples']=telemetry;rows.append(z);write('ALPHAFOLD_NATIVE_CONTROL_RESULTS.json',{'scope':'real AF2, unoptimized J1 and certified native controls','rows':rows,'budget':budget.report(),'oracle_calls_used':queue.calls_used,'recovered_RUNNING_calls':recovered,'no_TEST':True});print(json.dumps({'phase':'R1_REAL_AF','id':ident,'N':job['N'],'status':result.status.value,'seconds':z['worker_wall_seconds'],'pLDDT':z.get('features',{}).get('mean_pLDDT'),'pTM':z.get('features',{}).get('ptm'),'native_TM':z.get('recovery',{}).get('robust_fixed_map_TM_similarity_fullN'),'calls':queue.calls_used}),flush=True)
  consecutive_fail=consecutive_fail+1 if result.status!=OracleStatus.SUCCEEDED else 0
  if consecutive_fail>=2:write('R1_RESOURCE_STOP.json',{'reason':'two consecutive oracle failures; no unbounded retries','budget':budget.report()});break
 final=fit_reward_gate([v for v in rows if v['split']!='VALIDATION'],[v for v in rows if v['split']=='VALIDATION']);write('AF_REWARD_GO_NO_GO.json',final);write('R1_COMPLETED.json',{'oracle_calls_used':queue.calls_used,'budget':budget.report(),'rows':len(rows),'seconds':time.monotonic()-begin,'reward_GO':final['GO'],'production_RL_updates':0,'J1_optimizer_updates':0});queue.close();print(json.dumps({'phase':'R1_COMPLETE','reward_gate':final,'budget':budget.report()}),flush=True)
if __name__=='__main__':main()
