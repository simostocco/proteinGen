"""Read-only resource sampler; no training, no model loading."""
import json,shutil,subprocess,time
from pathlib import Path
R=Path('/mnt/d/Simone/proteinGen_data/e016_j1_clip_precision_pilot_20261018_v1')
with (R/'logs/resources.jsonl').open('a') as f:
 while not (R/'MONITOR_STOP').exists():
  mem={v.split(':')[0]:int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith(('MemAvailable:','MemTotal:','SwapFree:','SwapTotal:'))}
  try:g=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
  except Exception as e:g=repr(e)
  f.write(json.dumps({'time':time.time(),'D_free_bytes':shutil.disk_usage(R).free,'memory':mem,'gpu_used_free_utilization':g})+'\n');f.flush()
  for _ in range(20):
   if (R/'MONITOR_STOP').exists():break
   time.sleep(1)
