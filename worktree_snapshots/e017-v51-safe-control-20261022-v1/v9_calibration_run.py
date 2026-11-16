from pathlib import Path
import json,os,subprocess,time
R=Path('/mnt/d/Simone/proteinGen_data/e017_v9_property_steering_20261023_v1')
def save(n,x):
 p=R/n;q=p.with_name(p.name+'.part')
 with q.open('w') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
assert (R/'INITIAL_INPUT_AUDIT.json').exists();assert not (R/'CALIBRATION_ADMISSION.json').exists()
if '--worker' not in __import__('sys').argv:
 import shutil
 shutil.copyfile(__file__,R/'control/v9_calibration_run.py');log=R/f'logs/calibration_controller_{time.time_ns()}.log'
 with log.open('x') as f:p=subprocess.Popen([str(R/'run.sh'),str(R/'control/v9_calibration_run.py'),'--worker'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 print(json.dumps(dict(pid=p.pid,log=str(log))))
else:
 for stage in ['sasa','specification']:
  marker='SASA_COMPLETED.json' if stage=='sasa' else 'CALIBRATION_ADMISSION.json'
  if (R/marker).exists():continue
  with (R/f'logs/{stage}_{time.time_ns()}.log').open('x') as f:
   p=subprocess.Popen([str(R/'run.sh'),str(R/'source/scripts/e017_v9/calibrate.py'),stage],stdout=f,stderr=subprocess.STDOUT);save('CALIBRATION_EXECUTION.json',dict(status='RUNNING',stage=stage,pid=p.pid,supervisor_pid=os.getpid(),automatic_restart=False));rc=p.wait()
  if rc:save('CALIBRATION_EXECUTION.json',dict(status='FAILED',stage=stage,exit_code=rc,automatic_restart=False));raise RuntimeError(stage)
 save('CALIBRATION_EXECUTION.json',dict(status='COMPLETED',automatic_restart=False))
