"""Authorized sequential GPU ladder with system-pressure monitoring, no RSS ceiling."""
import os,sys,json,time,subprocess,signal,shutil,statistics
from pathlib import Path
R=Path(os.environ['E016_ARTIFACT_ROOT']);lock=R/'scratch/complete_cuda.lock'
def meminfo():
 d={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if len(x.split())>=3}
 return {k:d[k] for k in ['MemTotal','MemAvailable','SwapTotal','SwapFree']}
def rss(pid):
 try:return int(next(x.split()[1] for x in Path(f'/proc/{pid}/status').read_text().splitlines() if x.startswith('VmRSS:')))*1024
 except (OSError,StopIteration):return None
baseline=meminfo();headroom=baseline['MemTotal']*.10
# Capacity-derived kernel/OS headroom, not an arbitrary process-memory cap.
if baseline['MemAvailable']<=headroom:raise RuntimeError('HOST_ALREADY_PRESSURED')
if shutil.disk_usage(R).free<11*2**30:raise RuntimeError('DISK_RESERVE')
fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.write(fd,str(os.getpid()).encode());os.close(fd)
run_id=os.environ.get('E016_BENCH_RUN_ID','001')
results=[]
try:
 for N in [64,128,256,500]:
  folder=R/'benchmarks'/f'N{N}_run{run_id}'
  if folder.exists():raise RuntimeError('IMMUTABLE_BENCHMARK_COLLISION')
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
  cmd=[sys.executable,'-B',str(R/'launch.py'),str(R/'source/scripts/e016_v3/benchmark_complete_cuda.py'),'--length',str(N)]
  t=time.monotonic();pressure_samples=0;peakrss=0;peakdevice=None;previous_swap=baseline['SwapTotal']-baseline['SwapFree']
  with (R/'logs'/f'CUDA_N{N}_run{run_id}.log').open('w') as output,(R/'benchmarks'/f'N{N}_run{run_id}_system_telemetry.jsonl').open('w') as telemetry:
   child=subprocess.Popen(cmd,stdout=output,stderr=subprocess.STDOUT,env=env,start_new_session=True)
   while child.poll() is None:
    mi=meminfo();mi.update(pid=child.pid,RSS_bytes=rss(child.pid),elapsed_seconds=time.monotonic()-t,free_disk_bytes=shutil.disk_usage(R).free)
    peakrss=max(peakrss,mi['RSS_bytes'] or 0)
    psi=Path('/proc/pressure/memory').read_text();mi['memory_PSI']=psi
    full=float(psi.split('full avg10=')[1].split()[0]) if 'full avg10=' in psi else 0
    swap=mi['SwapTotal']-mi['SwapFree'];pressure=(mi['MemAvailable']<headroom or (full>1 and swap-previous_swap>baseline['MemTotal']*.01))
    pressure_samples=pressure_samples+1 if pressure else 0
    query=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
    mi['GPU_device_sample']=query.stdout.strip()
    try:peakdevice=max(peakdevice or 0,int(query.stdout.split(',')[0]))
    except ValueError:pass
    proc=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,timeout=5);mi['GPU_process_sample']=proc.stdout.strip()
    telemetry.write(json.dumps(mi)+'\n');telemetry.flush()
    if pressure_samples>=3 or mi['free_disk_bytes']<10*2**30:
     os.killpg(child.pid,signal.SIGINT);child.wait(timeout=15);raise RuntimeError('STOP_REAL_SYSTEM_PRESSURE_OR_DISK')
    time.sleep(.5)
   code=child.wait()
  result={'N':N,'exit_code':code,'RSS_sampled_peak_bytes':peakrss,'total_GPU_device_sampled_peak_MiB':peakdevice,'wall_seconds':time.monotonic()-t,'baseline':baseline,'derived_OS_headroom_bytes':headroom,'fixed_process_RSS_cap':None,'fixed_allocator_fraction_cap':None}
  (R/'benchmarks'/f'N{N}_run{run_id}_HOST_RECEIPT.json').write_text(json.dumps(result,indent=2));results.append(result)
  print(json.dumps(result),flush=True)
  if code or not (folder/'RESULT.json').exists():raise RuntimeError(f'LENGTH_FAILED_N{N}; preserve diagnostics before optimization')
 (R/f'benchmarks/LADDER_COMPLETE_run{run_id}.json').write_text(json.dumps(results,indent=2))
finally:lock.unlink(missing_ok=True)
