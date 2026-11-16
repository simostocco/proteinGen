"""Low-impact telemetry and calibration-evidence preservation; no model calls."""
import json
import os
from pathlib import Path
import shutil
import signal
import time

R=Path('/mnt/d/Simone/proteinGen_data/e017_v7_amortized_lowlevel_20261023_v1')
with (R/'results/resource_watch.jsonl').open('a') as f:
 while True:
  state=json.loads((R/'RECOVERY_EXECUTION.json').read_text())
  if state['status']!='RUNNING':break
  pid=state['child_pid'];stage=state['stage'];m={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())};free=shutil.disk_usage(R).free
  for arm in ['C','D']:
   p=R/'CRITIC_CALIBRATION.json';target=R/f'results/{arm}_CRITIC_CALIBRATION.json'
   if stage==f'train_arm_{arm}' and p.exists() and not target.exists():shutil.copyfile(p,target)
  reasons=[]
  if free<2*2**30+64*2**20:reasons.append('D operating reserve')
  if m['MemAvailable']<2**30:reasons.append('WSL available memory below1GiB')
  row=dict(UTC=time.time(),stage=stage,child_pid=pid,D_free_bytes=free,WSL_available_RAM_bytes=m['MemAvailable'],swap_used_bytes=m['SwapTotal']-m['SwapFree'],loadavg=Path('/proc/loadavg').read_text().split()[:3],stop_reasons=reasons)
  f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
  if reasons:
   cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
   if 'scripts/e017_v7/study.py' in cmd:os.kill(pid,signal.SIGTERM)
   break
  time.sleep(20)
print('WATCH_ENDED',flush=True)
