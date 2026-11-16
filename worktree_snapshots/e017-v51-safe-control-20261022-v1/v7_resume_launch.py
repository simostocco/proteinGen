"""Start one detached recovery supervisor, immune to conversation transport loss."""
from pathlib import Path
import json
import subprocess
import time

R=Path('/mnt/d/Simone/proteinGen_data/e017_v7_amortized_lowlevel_20261023_v1')
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 if ('scripts/e017_v7/study.py' in cmd or 'v7_recovery_supervisor.py' in cmd) and p.name!=str(__import__('os').getpid()):
  if (p/'comm').read_text().startswith('python'):raise RuntimeError('V7 process already active '+p.name)
script=Path(__file__).with_name('v7_recovery_supervisor.py');log=R/f'logs/recovery_supervisor_{time.time_ns()}.log'
with log.open('x') as f:pr=subprocess.Popen(['python3','-B',str(script)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(supervisor_pid=pr.pid,log=str(log))))
