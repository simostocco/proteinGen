"""Single bounded trainer followed by frozen final assessment; no retries."""
import fcntl,os,runpy,subprocess,sys,time
from pathlib import Path
R=Path('/mnt/d/Simone/proteinGen_data/e017_v94r3_substantial_geometry_rl_20261026_v1');C=runpy.run_path(str(R/'source/scripts/e017_v94r2/common.py'))
read=C['read'];atomic=C['atomic']
def main():
 lock=(R/'WORKER_LOCK').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (R/'LAUNCH_RECEIPT.json').exists(),'refuse duplicate launch'
 q=read(R/'qualification/RELEASE.json');auth=read(R/'REVIEW_AUTHORIZATION.json');assert q['PASS'] and auth['explicit_review_authorized']
 assert C['sha'](R/'TRAINING_CONFIGURATION_AND_SEEDS.json')==q['binding']['configuration_SHA256']
 for n,h in q['binding']['source'].items():assert C['sha'](R/'source'/n)==h
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip()==q['binding']['V94_source_commit']
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=R/'source',text=True).strip()
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'competing CUDA process'
 projection=C['projection']();assert projection['admitted'],projection
 atomic(R/'FINAL_LAUNCH_STORAGE_ADMISSION.json',projection)
 atomic(R/'WORKER_PHASE.json',dict(phase='INITIALIZATION_AND_TRAINING',UTC=C['utc']()))
 cmd=[str(R/'run.sh'),str(R/'source/scripts/e017_v94r/production.py'),'production','--authorization',str(R/'REVIEW_AUTHORIZATION.json')]
 with (R/'production.log').open('xb') as log:
  trainer=subprocess.Popen(cmd,cwd=R/'source',stdout=log,stderr=subprocess.STDOUT,preexec_fn=lambda:os.nice(10))
  atomic(R/'LAUNCH_RECEIPT.json',dict(UTC=C['utc'](),PID=trainer.pid,controller_PID=os.getpid(),command=cmd,
   source_commit=q['binding']['V94_source_commit'],configuration_SHA256=q['binding']['configuration_SHA256'],
   automatic_restarts=False,target_updates=5000,target_attempts=80000,all_artifacts_on_D=True))
  monlog=(R/'monitor.log').open('xb')
  monitor=subprocess.Popen([sys.executable,'-B',str(R/'source/scripts/e017_v94r2/monitor.py')],stdout=monlog,stderr=subprocess.STDOUT)
  while trainer.poll() is None:
   if monitor.poll() is not None:
    atomic(R/'MONITOR_FAILURE.json',dict(UTC=C['utc'](),exitcode=monitor.returncode,action='cooperative pause; no restart'))
    if not (R/'control/STOP').exists():(R/'control/STOP').write_text('Resource/metric monitor exited; pause safely\n')
   time.sleep(5)
  rc=trainer.wait()
 ev=None
 if rc==0 and (R/'production/TRAINING_COMPLETED.json').exists():
  atomic(R/'WORKER_PHASE.json',dict(phase='FINAL_EVALUATION',UTC=C['utc']()))
  with (R/'final_evaluation.log').open('xb') as log:
   ev=subprocess.call([str(R/'run.sh'),str(R/'source/scripts/e017_v94r2/final_evaluate.py')],cwd=R/'source',stdout=log,stderr=subprocess.STDOUT)
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip().splitlines()
 atomic(R/'WORKER_EXIT.json',dict(UTC=C['utc'](),trainer_exitcode=rc,final_evaluation_exitcode=ev,CUDA_compute_processes_after=gpu,no_automatic_restart=True))
 if rc!=0 and not (R/'production/STOP_RECORD.json').exists():
  # Status publication only: never serialize unsafe/uncommitted actor state.
  old=read(R/'production/LIVE_STATUS.json') if (R/'production/LIVE_STATUS.json').exists() else {}
  atomic(R/'production/LIVE_STATUS.json',dict(old,state='FAILED',worker_exitcode=rc,reason='see production.log and durable checkpoint/journal',UTC=C['utc']()))
 monitor.wait(timeout=90);monlog.close()
 subprocess.run([sys.executable,'-B',str(R/'source/scripts/e017_v94r2/publish.py')],check=True)
 print('BOUNDED_R2_EXIT',rc,ev,flush=True)
if __name__=='__main__':main()
