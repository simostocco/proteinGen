"""Read-only host telemetry; never creates a CUDA context or launches a workload."""
import json,subprocess,time
from pathlib import Path
R=Path('/mnt/d/Simone/proteinGen_data/e016_scaled10m_training_20261016_v1')
with (R/'RESOURCE_TELEMETRY.jsonl').open('x') as f:
 while not (R/'CHECKPOINT_SELECTIONS_FROZEN.json').exists() and not (R/'BLOCKED.json').exists():
  try:
   hb=json.load(open(R/'HEARTBEAT.json'));row={'timestamp':time.time(),'name':hb.get('name'),'update':hb.get('update'),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip(),'meminfo':{line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','SwapFree:'))},'allocator':'expandable_segments:True,garbage_collection_threshold:0.8'};f.write(json.dumps(row)+'\n');f.flush()
  except Exception as e:f.write(json.dumps({'timestamp':time.time(),'telemetry_error':str(e)})+'\n');f.flush()
  time.sleep(10)
