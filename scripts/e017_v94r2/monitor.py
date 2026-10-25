"""Compact sidecar metrics/resource guard; never launches a CUDA workload."""
import collections,os,runpy,subprocess,time
from pathlib import Path
R=Path('/mnt/d/Simone/proteinGen_data/e017_v94r2_corrected_geometry_rl_20261025_v1')
C=runpy.run_path(str(R/'source/scripts/e017_v94r2/common.py'));read=C['read'];atomic=C['atomic'];P=R/'production'
def call(args):
 try:return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT,timeout=15).strip()
 except Exception as e:return repr(e)
def main():
 last=0;tracked={};sizes=[];tot=collections.Counter();secs=[];identities=set();next_resource=0
 log=R/'AUGMENTED_TRAINING_METRICS.jsonl';resource=R/'RESOURCE_TELEMETRY.jsonl'
 # Rebuild an interrupted sidecar deterministically without optimizer work.
 if log.exists():
  for line in log.read_text().splitlines():
   x=__import__('json').loads(line);last=x['update'];secs.append(x['seconds']);identities.add(x['identity'])
   for k in ['N','completed','aborts','hard_valid','full_success','proposals','rejected','completed_overpacking']:tot[k]+=x[k]
  for f in (P/'rollouts').glob('u*.json'):tracked[f.name]=f.stat().st_blocks*512;sizes.append(f.stat().st_size)
 while True:
  while (P/f'commits/u{last+1:06}.json').exists():
   cp=P/f'commits/u{last+1:06}.json';commit=read(cp);rp=P/f'rollouts/u{last+1:06}.json'
   assert C['sha'](rp)==commit['logical']['rollout_SHA256'];packet=read(rp);x=C['extra_metrics'](packet,commit)
   with log.open('a') as f:f.write(__import__('json').dumps(x,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
   last=x['update'];secs.append(x['seconds']);identities.add(x['identity']);sizes.append(rp.stat().st_size);tracked[rp.name]=rp.stat().st_blocks*512
   for k in ['N','completed','aborts','hard_valid','full_success','proposals','rejected','completed_overpacking']:tot[k]+=x[k]
  pending=P/f'rollouts/u{last+1:06}.json'
  if pending.exists():tracked[pending.name]=pending.stat().st_blocks*512
  projected=C['projection'](sum(tracked.values()),last,sizes[-50:])
  atomic(R/'LATEST_STORAGE_PROJECTION.json',projected)
  if not projected['admitted'] and not (R/'WORKER_EXIT.json').exists() and not (P/'TRAINING_COMPLETED.json').exists():
   # Storage-only cooperative pause, under the frozen remaining-work budget.
   if not (R/'control/STOP').exists():
    atomic(R/'STORAGE_STOP_RECORD.json',projected)
    with (R/'control/STOP').open('x') as f:f.write('Frozen remaining-work storage admission failed\n');f.flush();os.fsync(f.fileno())
  core=read(P/'LIVE_STATUS.json') if (P/'LIVE_STATUS.json').exists() else None
  terminal=read(R/'WORKER_EXIT.json') if (R/'WORKER_EXIT.json').exists() else None
  stop=read(P/'STOP_RECORD.json') if (P/'STOP_RECORD.json').exists() else None
  phase=read(R/'WORKER_PHASE.json') if (R/'WORKER_PHASE.json').exists() else {}
  status='RUNNING' if last else 'INITIALIZING'
  if phase.get('phase')=='FINAL_EVALUATION':status='COMPLETED_FINAL_EVALUATION_PENDING'
  if stop:status='STOPPED_PREDEFINED_SAFETY_GATE'
  if terminal:status=C['live']()['V94R2_LIVE_STATUS']
  atomic(R/'V94R2_LIVE_STATUS.json',dict(V94R2_LIVE_STATUS=status,UTC=C['utc'](),TRAINER_PID=read(R/'LAUNCH_RECEIPT.json')['PID'] if not terminal else None,
   COMMITTED_UPDATES=last,TARGET_UPDATES=5000,TRAIN_IDENTITIES_ENCOUNTERED=len(identities),GENERATION_ATTEMPTS=tot['N'],TARGET_ATTEMPTS=80000,COMPLETED_TRAJECTORIES=tot['completed'],
   RECENT_SECONDS_PER_UPDATE=sum(secs[-50:])/len(secs[-50:]) if secs else None,D_FREE_GIB=projected['D_free_bytes']/2**30,
   VALIDITY_STATUS='BREACH' if stop else 'PASS_AT_LAST_FROZEN_MONITOR' if (R/'FROZEN_INITIALIZATION_BASELINE.json').exists() else 'BASELINE_PENDING',
   PACKING_FAILURE_RATE=tot['completed_overpacking']/max(1,tot['N']),PROPOSAL_ABORT_RATE=tot['aborts']/max(1,tot['N']),KL_STATUS='PASS_COMMITTED' if last else 'PENDING',
   RECOVERY_STATUS='QUALIFIED_SOURCE_EQUIVALENCE_AND_INITIAL_ROUNDTRIP' if (R/'ACTUAL_INITIAL_RECOVERY_ROUNDTRIP.json').exists() else 'INITIAL_ROUNDTRIP_PENDING',
   LATEST_CHECKPOINT=core.get('latest_checkpoint') if core else None,core_status=core,projection=projected))
  if time.monotonic()>=next_resource:
   mem={}
   for line in Path('/proc/meminfo').read_text().splitlines():
    k,v=line.split(':',1)
    if k in ['MemTotal','MemAvailable','SwapTotal','SwapFree']:mem[k]=int(v.strip().split()[0])*1024
   x=dict(UTC=C['utc'](),update=last,D_free_bytes=projected['D_free_bytes'],WSL_RAM=mem,CPU_load=os.getloadavg(),
    GPU=call(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free,memory.total','--format=csv,noheader']),
    CUDA_owners=call(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']))
   with resource.open('a') as f:f.write(__import__('json').dumps(x,separators=(',',':'))+'\n');f.flush()
   next_resource=time.monotonic()+60
  if terminal:return
  time.sleep(5)
if __name__=='__main__':main()
