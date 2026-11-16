"""Lightweight resource guard for the already running, verified V6 worker."""
import json
import os
from pathlib import Path
import signal
import time

R=Path('/mnt/d/Simone/proteinGen_data/e017_v6_frozen_parent_candidates_20261023_v1')
r=json.loads((R/'RECOVERY.json').read_text());pid=r['pid']
cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
assert str(R/'source/scripts/e017_v6/study.py')+' worker' in cmd
with (R/'results/resource_guard.jsonl').open('x') as f:
 while Path(f'/proc/{pid}').exists():
  m={a:int(b.split()[0])*1024 for a,b in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
  r=json.loads((R/'RECOVERY.json').read_text());st=os.statvfs(R)
  data=dict(UTC=time.time(),worker_pid=pid,completed_candidates=r['completed_candidates'],D_free_bytes=st.f_bavail*st.f_frsize,
    WSL_MemAvailable_bytes=m['MemAvailable'],swap_used_bytes=m['SwapTotal']-m['SwapFree'],
    last_GPU_peak_allocated_bytes=r['max_GPU_allocated_bytes'])
  reasons=[]
  if data['D_free_bytes']<2*2**30+64*2**20:reasons.append('D reserve')
  if data['last_GPU_peak_allocated_bytes']>6*2**30:reasons.append('CUDA allocation6GiB')
  if data['WSL_MemAvailable_bytes']<2**30:reasons.append('WSL available RAM1GiB')
  data['stop_reasons']=reasons
  f.write(json.dumps(data)+'\n');f.flush();os.fsync(f.fileno())
  if reasons:
   os.kill(pid,signal.SIGTERM);print('COOPERATIVE_STOP_REQUESTED',reasons,flush=True);break
  time.sleep(10)
print('GUARD_FINISHED',flush=True)
