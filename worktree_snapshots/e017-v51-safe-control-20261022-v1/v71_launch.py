from pathlib import Path
import json
import subprocess
import time
import os
R=Path('/mnt/d/Simone/proteinGen_data/e017_v71_expected_return_calibration_20261023_v1')
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 if 'scripts/e017_v71/study.py' in cmd and (p/'comm').read_text().startswith('python'):raise RuntimeError('V7.1 already active '+p.name)
log=R/f'logs/campaign_{time.time_ns()}.log'
with log.open('x') as f:p=subprocess.Popen([str(R/'run.sh'),str(R/'source/scripts/e017_v71/study.py'),'campaign'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,log=str(log))))
