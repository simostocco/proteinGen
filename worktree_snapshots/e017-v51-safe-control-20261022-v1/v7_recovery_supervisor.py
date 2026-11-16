"""Continue only missing stages of the sealed V7 campaign. Never restart stops."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

R=Path('/mnt/d/Simone/proteinGen_data/e017_v7_amortized_lowlevel_20261023_v1')
CHILD=None
STOP=False

def read(p):return json.loads(Path(p).read_text())
def save(obj):
 p=R/'RECOVERY_EXECUTION.json';q=p.with_suffix('.part')
 with q.open('w') as f:json.dump(obj,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(q,p)

def handler(*args):
 global STOP
 STOP=True
 if CHILD is not None and CHILD.poll() is None:CHILD.send_signal(signal.SIGTERM)

def call(stage,*args):
 global CHILD
 if STOP:return False
 owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
 if owners.strip():raise RuntimeError('unexpected CUDA owner before stage: '+owners)
 name='_'.join([stage,*args]).replace('--','');log=R/f'logs/recovery_{name}_{time.time_ns()}.log'
 with log.open('x') as f:
  CHILD=subprocess.Popen([str(R/'run.sh'),str(R/'source/scripts/e017_v7/study.py'),stage,*args],stdout=f,stderr=subprocess.STDOUT)
  save(dict(status='RUNNING',supervisor_pid=os.getpid(),child_pid=CHILD.pid,stage=name,log=str(log),UTC=time.time(),automatic_restart=False))
  rc=CHILD.wait()
 print('STAGE',name,rc,flush=True)
 if rc:
  save(dict(status='STOPPED',stage=name,returncode=rc,log=str(log),UTC=time.time(),automatic_restart=False));return False
 return not STOP

def main():
 lock=(R/'recovery_supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 signal.signal(signal.SIGTERM,handler);signal.signal(signal.SIGINT,handler)
 if (R/'CAMPAIGN_COMPLETED.json').exists():save(dict(status='ALREADY_COMPLETED',UTC=time.time(),automatic_restart=False));return
 if not (R/'CANDIDATES_COMPLETED.json').exists() and not call('collect'):return
 if not (R/'ENGINEERING_TESTS.json').exists() and not call('engineering'):return
 for arm in ['B','C']:
  if (R/f'results/{arm}_FINISHED.json').exists():continue
  if (R/f'results/{arm}_STOP.json').exists():raise RuntimeError('scientific stop exists without final receipt; do not resume '+arm)
  if (R/f'checkpoints/{arm}_INDEX.json').exists():raise RuntimeError('interrupted training requires explicit exact recovery audit before '+arm)
  if not call('train','--arm',arm):return
 B=read(R/'results/B_FINISHED.json');C=read(R/'results/C_FINISHED.json');admit=False
 if not B['reason'] and not C['reason'] and B['actor_updates']>=250 and C['actor_updates']>=250:
  b=read(R/f'results/B{B["actor_updates"]}_DEV_K2.json');c=read(R/f'results/C{C["actor_updates"]}_DEV_K2.json');a=read(R/'results/A_DEV_K2.json');admit=b['K1']['success']>a['K1']['success'] and c['K1']['success']>=a['K1']['success']
 p=R/'D_ADMISSION.json'
 if not p.exists():p.write_text(json.dumps(dict(admitted=admit,reason='sealed B/C eligibility,250update,and K1 gates; no threshold changes'),indent=2))
 if admit and not (R/'results/D_FINISHED.json').exists() and not call('train','--arm','D'):return
 if not (R/'FINAL_EVALUATION_COMPLETED.json').exists() and not call('final_evaluate'):return
 (R/'CAMPAIGN_COMPLETED.json').write_text(json.dumps(dict(complete=True,UTC=time.time(),no_TEST=True,no_high_level=True,recovered_session=True)))
 save(dict(status='COMPLETED',UTC=time.time(),automatic_restart=False))

if __name__=='__main__':main()
