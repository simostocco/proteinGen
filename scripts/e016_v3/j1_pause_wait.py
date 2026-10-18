"""CPU wait/dispatch only; never launches training or a second GPU job."""
import json,os,subprocess,time
from pathlib import Path
R=Path('/mnt/d/Simone/proteinGen_data/e016_j1_sequence_first_20261018_v1')
PID=1524156
expected=str(R/'source/scripts/e016_v3/j1_sequence_first.py')
def active():
 try:return expected in Path(f'/proc/{PID}/cmdline').read_bytes().replace(b'\0',b' ').decode()
 except FileNotFoundError:return False
while active():
 time.sleep(5)
# Existing worker must have published a new stop/completion state.
path=R/'runs/J1/STOPPED.json'
if (R/'runs/J1/COMPLETED.json').exists():path=R/'runs/J1/COMPLETED.json'
end=json.loads(path.read_text());u=end.get('update',end.get('additional_updates',0))
if not 6000<=u<=8500:raise RuntimeError('No valid new pause receipt; preserve evidence for manual recovery')
(R/'ARCHIVER_STOP').write_text('User-requested training pause; no more milestone work\n')
log=R/'logs/pause_evaluation.log'
with log.open('x') as f:
 p=subprocess.Popen(['bash',str(R/'run.sh'),str(R/'source/scripts/e016_v3/j1_sequence_pause.py'),'--samples','16'],stdout=f,stderr=subprocess.STDOUT)
 (R/'PAUSE_EVALUATION_PROCESS.json').write_text(json.dumps({'pid':p.pid,'training_stopped_update':u,'time':time.time(),'no_optimizer_steps':True},indent=2))
 rc=p.wait()
(R/'PAUSE_EVALUATION_EXIT.json').write_text(json.dumps({'exit_code':rc,'update':u,'time':time.time()},indent=2))
if rc:raise SystemExit(rc)
print('PAUSE_EVALUATION_DISPATCH_FINISHED',flush=True)
