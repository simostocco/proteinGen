"""Sequential final-only assessments and publication; never launches training."""
from pathlib import Path
import fcntl
import json
import os
import shutil
import subprocess
import time

R=Path('/mnt/d/Simone/proteinGen_data/e017_v71_expected_return_calibration_20261023_v1')
def save(name,data):
 p=R/name;q=p.with_name(p.name+'.part')
 with q.open('w') as f:json.dump(data,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def gpu_idle():
 out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
 assert not out.strip(),out
def main():
 lock=(R/'runtime/finish.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (R/'PUBLICATION_COMPLETED.json').exists()
 deadline=time.monotonic()+3600
 while not (R/'CAMPAIGN_COMPLETED.json').exists():
  state=json.loads((R/'EXECUTION.json').read_text())
  assert state['status']=='RUNNING',state
  assert time.monotonic()<deadline,'campaign wait deadline'
  time.sleep(10)
 # The telemetry watcher exits after the final campaign status; allow it to flush.
 time.sleep(25)
 jobs=[('realized_confirmation','v71_realized_eval.py','REALIZED_CRITIC_CONFIRMATION.json'),('rwr_goal','v71_rwr_goal_eval.py','RWR_GOAL_SENSITIVITY.json'),('publication','v71_analyze.py','PROVENANCE_SHA256.json')]
 for stage,script,result in jobs:
  if (R/result).exists():continue
  gpu_idle();assert shutil.disk_usage(R).free>2*2**30+64*2**20
  with (R/f'logs/final_{stage}_{time.time_ns()}.log').open('x') as f:
   p=subprocess.Popen([str(R/'run.sh'),str(R/'analysis_source'/script)],stdout=f,stderr=subprocess.STDOUT)
   save('FINALIZATION_STATUS.json',dict(status='RUNNING',stage=stage,pid=p.pid,supervisor_pid=os.getpid(),automatic_restart=False,UTC=time.time()))
   rc=p.wait()
  assert rc==0,(stage,rc)
  assert (R/result).exists(),result
 gpu_idle()
 save('FINALIZATION_STATUS.json',dict(status='COMPLETED',automatic_restart=False,UTC=time.time()))
 save('PUBLICATION_COMPLETED.json',dict(complete=True,actor_updates=0,SAC_actor_updates=0,automatic_restart=False,UTC=time.time()))
 print('V7.1 final-only assessments and publication complete',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as ex:
  save('FINALIZATION_STATUS.json',dict(status='FAILED',error=repr(ex),automatic_restart=False,UTC=time.time()));raise
