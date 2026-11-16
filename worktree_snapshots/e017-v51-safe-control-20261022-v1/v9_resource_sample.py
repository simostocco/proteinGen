from pathlib import Path
import json,subprocess,time,shutil,os
R=Path('/mnt/d/Simone/proteinGen_data/e017_v9_property_steering_20261023_v1')
def snap():
 mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if k in ['MemTotal','MemAvailable','SwapTotal','SwapFree']}
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True).strip();apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip();rows=subprocess.check_output(['ps','-eo','pid,ppid,etime,pcpu,rss,comm,args','--sort=pid'],text=True).splitlines();workers=[s for s in rows if ('e017_v9' in s or 'e017_v8' in s or 'e007' in s or 'e015' in s or 'e012' in s or 'ProteinMPNN' in s or 'colabfold' in s) and 'v9_resource_sample' not in s and 'bash -' not in s];x=dict(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),D_free_bytes=shutil.disk_usage(R).free,WSL_memory=mem,GPU=gpu,CUDA_processes=apps,research_workers=workers)
 with (R/'logs/RESOURCE_SAMPLES.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
 print(json.dumps(x))
if __name__=='__main__':snap()
