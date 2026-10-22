import pathlib,time,json,sys,subprocess,os
import psutil
R=pathlib.Path('/mnt/d/Simone/proteinGen_data/e017_v51_safe_control_20261022_v1');parent=psutil.Process(int(sys.argv[1]));log=open(R/'results/RESOURCE_OBSERVER.jsonl','a',buffering=1)
while parent.is_running():
 row=dict(UTC=time.time(),free_D_bytes=psutil.disk_usage(str(R)).free,available_RAM=psutil.virtual_memory().available,CPU_percent=psutil.cpu_percent(),processes=[])
 for p in parent.children(recursive=True):
  try:row['processes'].append(dict(pid=p.pid,rss=p.memory_info().rss,cpu_seconds=sum(p.cpu_times()[:2])))
  except psutil.Error:pass
 try:row['GPU']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip()
 except Exception as e:row['GPU_error']=repr(e)
 log.write(json.dumps(row)+'\n');time.sleep(10)
