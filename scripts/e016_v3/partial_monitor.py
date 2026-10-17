"""Non-CUDA-context operational telemetry, bounded frequency and D:-only output."""
import json,os,shutil,subprocess,time
from pathlib import Path
R=Path('/mnt/d/Simone/proteinGen_data/e016_partial_supervision_20261016_v1')
def main():
 with (R/'logs'/('resource_recovery_'+str(int(time.time()))+'.jsonl')).open('x',buffering=1) as f:
  while not (R/'ASSESSMENT_DONE.json').exists():
   m={v.split(':')[0]:int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith(('MemTotal:','MemAvailable:','SwapFree:','SwapTotal:'))};v={'time':time.time(),'D_free_bytes':shutil.disk_usage(R).free,'memory':m,'memory_pressure':Path('/proc/pressure/memory').read_text().strip()}
   try:v['GPU']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
   except (OSError,subprocess.SubprocessError) as e:v['GPU_error']=repr(e)
   f.write(json.dumps(v)+'\n');time.sleep(10)
if __name__=='__main__':main()
