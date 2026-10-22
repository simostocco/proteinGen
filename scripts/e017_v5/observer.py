"""Passive resource observer for sealed V5; exits with parent or terminal status."""
import os,pathlib,json,time,subprocess,sys
R=pathlib.Path('/mnt/d/Simone/proteinGen_data/e017_v5_goal_observability_20261022_v1');parent=int(sys.argv[1]);previous={}
with (R/'results/RESOURCE_TIMELINE.jsonl').open('a',buffering=1) as out:
 while pathlib.Path(f'/proc/{parent}').exists():
  status=json.load(open(R/'LIVE_STATUS.json'))
  if status.get('phase') in ['STUDY_COMPLETE','FAILED']:break
  now=time.time();workers=[]
  for p in pathlib.Path('/proc').glob('[0-9]*'):
   try:
    a=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();name=(p/'comm').read_text().strip()
    if 'python' not in name or str(R) not in a or 'observer.py' in a:continue
    st=(p/'stat').read_text().split();ticks=int(st[13])+int(st[14]);old=previous.get(p.name);cpu=None if old is None else 100*(ticks-old[0])/os.sysconf('SC_CLK_TCK')/(now-old[1]);previous[p.name]=(ticks,now);stats=(p/'status').read_text().splitlines();rss=next(int(x.split()[1])*1024 for x in stats if x.startswith('VmRSS:'));workers.append(dict(pid=int(p.name),RAM=rss,CPU_percent_one_core=cpu))
   except (OSError,StopIteration,ValueError):continue
  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip();out.write(json.dumps(dict(UTC_seconds=now,phase=status,workers=workers,GPU=gpu))+'\n');time.sleep(15)
