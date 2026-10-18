"""Low-cost resource samples for this one authorized worker; no GPU computation."""
import json,time,shutil,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[3]
with (R/'RESOURCE_SAMPLES.jsonl').open('a') as log:
 while not (R/'runs/J1/COMPLETED.json').exists():
  mem={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith(('MemAvailable:','MemTotal:','SwapFree:','SwapTotal:'))};free=shutil.disk_usage(R).free
  try:gpu=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip()
  except subprocess.CalledProcessError:gpu='UNAVAILABLE'
  row={'time':time.time(),'D_free_bytes':free,'GPU_csv_timestamp_usedMiB_freeMiB_utilization_temperature_power':gpu,**mem};log.write(json.dumps(row)+'\n');log.flush()
  if free<10*2**30+700*2**20 or mem['MemAvailable']<2*2**30:(R/'STOP_REQUESTED').write_text('resource headroom safeguard; preserve checkpoint')
  time.sleep(30)
print('RESOURCE_MONITOR_STOPPED')
