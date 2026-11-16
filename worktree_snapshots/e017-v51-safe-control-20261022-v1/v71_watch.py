"""Lightweight host-side V7.1 telemetry and cooperative resource protection."""
from pathlib import Path
import json
import os
import shutil
import signal
import time
R=Path('/mnt/d/Simone/proteinGen_data/e017_v71_expected_return_calibration_20261023_v1')
with (R/'results/resource_watch.jsonl').open('a') as f:
 while True:
  p=R/'EXECUTION.json'
  if not p.exists():time.sleep(5);continue
  s=json.loads(p.read_text())
  if s['status']!='RUNNING':break
  mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())};free=shutil.disk_usage(R).free;bad=free<2*2**30+64*2**20 or mem['MemAvailable']<2**30
  f.write(json.dumps(dict(UTC=time.time(),stage=s['stage'],pid=s['pid'],D_free_bytes=free,available_RAM_bytes=mem['MemAvailable'],swap_used_bytes=mem['SwapTotal']-mem['SwapFree'],loadavg=Path('/proc/loadavg').read_text().split()[:3],resource_stop=bad))+'\n');f.flush();os.fsync(f.fileno())
  if bad:
   cmd=Path(f'/proc/{s["pid"]}/cmdline').read_bytes().replace(b'\0',b' ').decode()
   if 'scripts/e017_v71/study.py' in cmd:os.kill(s['pid'],signal.SIGTERM)
   break
  time.sleep(20)
