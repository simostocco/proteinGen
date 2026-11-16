from pathlib import Path
import json,subprocess,time,shutil
R=Path('/mnt/d/Simone/proteinGen_data/e017_v91_ordered_goal_recovery_20261023_v1')
def sample():
 mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if k in ['MemTotal','MemAvailable','SwapTotal','SwapFree']};gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True).strip();apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip();rows=subprocess.check_output(['ps','-eo','pid,ppid,etime,pcpu,rss,comm,args','--sort=pid'],text=True).splitlines();workers=[row for row in rows if any(x in row for x in ['e017_v91','e007','e015','e012','ProteinMPNN','colabfold']) and 'v91_resource_sample' not in row and 'bash -' not in row];obj=dict(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),D_free_bytes=shutil.disk_usage(R).free,WSL_memory=mem,GPU=gpu,CUDA_processes=apps,research_workers=workers)
 with (R/'logs/RESOURCE_SAMPLES.jsonl').open('a') as f:f.write(json.dumps(obj)+'\n')
 print(json.dumps(obj))
if __name__=='__main__':sample()
