from pathlib import Path
import json,os,shutil,signal,time
R=Path('/mnt/d/Simone/proteinGen_data/e017_v8_constrained_planning_20261023_v1');start=time.monotonic();seen=False
with (R/'results/resource_watch.jsonl').open('a') as f:
 while True:
  p=R/'EXECUTION.json'
  if not p.exists():time.sleep(5);continue
  s=json.loads(p.read_text())
  if s['status']!='RUNNING':
   if seen:break
   if time.monotonic()-start>60:break
   time.sleep(2);continue
  seen=True;mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())};free=shutil.disk_usage(R).free;bad=free<2*2**30+128*2**20 or mem['MemAvailable']<2**30 or time.monotonic()-start>10800
  f.write(json.dumps(dict(UTC=time.time(),stage=s['stage'],pid=s['pid'],D_free_bytes=free,available_RAM_bytes=mem['MemAvailable'],swap_used_bytes=mem['SwapTotal']-mem['SwapFree'],resource_stop=bad))+'\n');f.flush();os.fsync(f.fileno())
  if bad:
   try:cmd=Path(f'/proc/{s["pid"]}/cmdline').read_bytes().replace(b'\0',b' ').decode()
   except FileNotFoundError:break
   if 'scripts/e017_v8/study.py' in cmd:os.kill(s['pid'],signal.SIGTERM)
   break
  time.sleep(20)
