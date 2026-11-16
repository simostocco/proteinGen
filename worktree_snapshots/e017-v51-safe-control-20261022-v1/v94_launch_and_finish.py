"""One bounded D-only production worker, no automatic retries or next study."""
import fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
R=Path('/mnt/d/Simone/proteinGen_data/e017_v94_fixed_sequence_geometry_rl_20261024_v1');Q=R/'recovery_qualification_20261025_v1'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
 return h.hexdigest()
def save(name,obj):
 p=R/name;t=p.with_name(p.name+'.part')
 with t.open('w') as f:json.dump(obj,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def main():
 lock=(R/'PRODUCTION_WORKER_LOCK').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (R/'LAUNCH_RECEIPT.json').exists(),'refuse duplicate production launch'
 rel=read(Q/'PRODUCTION_RECOVERY_RELEASE.json');core=read(Q/'RELEASE.json');assert rel['PASS'] and core['PASS']
 assert sha(Q/'RELEASE.json')==rel['optimizer_journal_release_SHA256']
 for n,h in rel['binding']['source'].items():assert sha(R/'source'/n)==h
 assert sha(R/'TRAINING_CONFIGURATION_AND_SEEDS.json')==rel['binding']['configuration_SHA256']
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip()==rel['source_commit']
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=R/'source',text=True).strip()
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip(),'CUDA owner blocks launch'
 cfg=read(R/'TRAINING_CONFIGURATION_AND_SEEDS.json')
 images=[p for p in Q.glob('*/checkpoints/recovery_*.pt')]
 assert len(images)==8
 already=sum(p.stat().st_blocks*512 for p in images)
 # Count logical bytes if DrvFS allocation accounting is unavailable. Existing
 # qualification images already occupy part of the admitted peak budget.
 already=min(already,sum(p.stat().st_size for p in images))
 remaining=cfg['minimum_free_bytes']-already;free=shutil.disk_usage(R).free
 assert free>=remaining and free>=8*2**30,(free,remaining)
 save('PRODUCTION_LAUNCH_ADMISSION.json',dict(PASS=True,D_free_bytes=free,peak_requirement_including_reserve=cfg['minimum_free_bytes'],
  qualification_images_already_allocated=already,remaining_requirement=remaining,operational_reserve=2*2**30,
  recovery_release_SHA256=sha(Q/'PRODUCTION_RECOVERY_RELEASE.json'),all_artifacts_on_D=True))
 cmd=[str(R/'run.sh'),str(R/'source/scripts/e017_v94/production.py'),'production']
 log=(R/'production.log').open('xb')
 def lower_priority():os.nice(10)
 child=subprocess.Popen(cmd,cwd=R/'source',stdout=log,stderr=subprocess.STDOUT,preexec_fn=lower_priority)
 save('LAUNCH_RECEIPT.json',dict(UTC=time.time(),PID=child.pid,controller_PID=os.getpid(),command=cmd,
  source_commit=rel['source_commit'],configuration_SHA256=rel['binding']['configuration_SHA256'],
  target_optimizer_updates=5000,target_generation_attempts=80000,automatic_restarts=False,
  controller_source_SHA256=sha(Path(__file__)),postbudget_evaluator_SHA256=sha(R/'audit_source/v94_final_evaluate.py'),
  reporting_source_SHA256=sha(R/'audit_source/v94_update_handoff.py')))
 subprocess.run([sys.executable,'-B',str(R/'audit_source/v94_update_handoff.py')],stdout=subprocess.DEVNULL,check=True)
 rc=child.wait();log.close();ev=None
 if rc==0 and (R/'production/TRAINING_COMPLETED.json').exists():
  with (R/'final_evaluation.log').open('xb') as f:
   ev=subprocess.call([str(R/'run.sh'),str(R/'audit_source/v94_final_evaluate.py')],cwd=R/'source',stdout=f,stderr=subprocess.STDOUT)
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip().splitlines()
 save('WORKER_EXIT.json',dict(UTC=time.time(),trainer_exitcode=rc,final_evaluation_exitcode=ev,
  CUDA_compute_processes_after=gpu,no_automatic_restart=True,controller_PID=os.getpid()))
 subprocess.run([sys.executable,'-B',str(R/'audit_source/v94_update_handoff.py')],stdout=subprocess.DEVNULL,check=True)
 print('BOUNDED_CAMPAIGN_EXIT',rc,'EVALUATION',ev,flush=True)
if __name__=='__main__':main()
