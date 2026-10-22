import pathlib,json,time,os,shutil
R=pathlib.Path('/mnt/d/Simone/proteinGen_data/e017_v5_goal_observability_20261022_v1');p=pathlib.Path('/proc/1529334');rows=[]
while p.exists():
 try:
  rows.append(dict(UTC_seconds=time.time(),io={k:int(v) for k,v in (x.split(':') for x in (p/'io').read_text().splitlines())},ticks=(p/'stat').read_text().split()[13:17],free_D_bytes=shutil.disk_usage(R).free));q=R/'results/PREPARATION_RESOURCE_TIMELINE.json';tmp=q.with_suffix('.tmp');tmp.write_text(json.dumps(rows,indent=2));os.replace(tmp,q)
 except OSError:break
 time.sleep(30)
