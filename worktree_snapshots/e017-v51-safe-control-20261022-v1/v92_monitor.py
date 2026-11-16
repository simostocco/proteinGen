from pathlib import Path
import datetime,json,shutil,subprocess,time
r=Path('/mnt/d/Simone/proteinGen_data/e017_v92_prefix_feasibility_20261024_v1')
for i in range(180):
 if (r/'control/MONITOR_DONE').exists():break
 gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip();processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip();mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines());rec=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),D_free_bytes=shutil.disk_usage(r).free,GPU=gpu,CUDA_processes=processes,MemAvailable=mem['MemAvailable'].strip(),SwapFree=mem['SwapFree'].strip())
 with (r/'logs/RESOURCE_TELEMETRY.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
 if rec['D_free_bytes']<2*2**30+32*2**20:(r/'control/STOP').write_text('storage guard\n');break
 time.sleep(10)
