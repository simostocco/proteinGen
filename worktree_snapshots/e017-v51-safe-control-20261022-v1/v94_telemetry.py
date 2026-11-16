"""Low-impact external telemetry; no training, evaluation, or restart actions."""
import datetime,json,os,shutil,subprocess,time
from pathlib import Path
R=Path('/mnt/d/Simone/proteinGen_data/e017_v94_fixed_sequence_geometry_rl_20261024_v1')
path=R/'RESOURCE_TELEMETRY.jsonl'
while not (R/'WORKER_EXIT.json').exists():
    mem={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        k,v=line.split(':',1)
        if k in ['MemTotal','MemAvailable','SwapTotal','SwapFree']:mem[k]=int(v.strip().split()[0])*1024
    def call(args):
        try:return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT,timeout=15).strip()
        except Exception as e:return repr(e)
    state=R/'production/LIVE_STATUS.json'
    row=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),D_free_bytes=shutil.disk_usage(R).free,
        WSL_RAM=mem,loadavg=os.getloadavg(),GPU=call(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader']),
        CUDA_owners=call(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']),
        counters=json.loads(state.read_text())['counters'] if state.exists() else None)
    with path.open('a') as f:f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
    for _ in range(12):
        if (R/'WORKER_EXIT.json').exists():break
        time.sleep(5)
