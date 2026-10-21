from pathlib import Path
import json,os,time,subprocess,statistics,shutil
R=Path('/mnt/d/Simone/proteinGen_data/e017_crisp_v3_microbatch_20261021_v1');A=R.parent/'e017_crisp_v3_training_admission_20261021_v1';PID=178643;run=A/'training_runs/run_bf2bbb285d0b4d929eb8cef5215063f3';duration=180;hz=os.sysconf('SC_CLK_TCK')
def stat():
 raw=Path(f'/proc/{PID}/stat').read_text();v=raw.rsplit(') ',1)[1].split();status={}
 for line in Path(f'/proc/{PID}/status').read_text().splitlines():
  if ':' in line:
   k,val=line.split(':',1);status[k]=val.strip()
 io={}
 for line in Path(f'/proc/{PID}/io').read_text().splitlines():k,val=line.split(':');io[k]=int(val)
 return dict(ticks=int(v[11])+int(v[12]),state=v[0],start_ticks=v[19],RSS_bytes=int(status['VmRSS'].split()[0])*1024,peak_RSS_bytes=int(status['VmHWM'].split()[0])*1024,threads=int(status['Threads']),io=io,wchan=Path(f'/proc/{PID}/wchan').read_text().strip())
def updates():
 rows=[]
 for line in (run/'updates.jsonl').read_text().splitlines():
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:pass
 return rows
before=updates();prev=stat();then=time.monotonic();start=then;data=[]
with open(R/'results/PASSIVE_RESOURCE_SAMPLES.jsonl','x') as f:
 while time.monotonic()-start<duration:
  now=time.monotonic();s=stat();g=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=10).strip().split(',');gg=[float(v.strip()) for v in g];row=dict(elapsed_seconds=now-start,UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),cpu_percent=100*(s['ticks']-prev['ticks'])/hz/max(1e-6,now-then),GPU_utilization_percent=gg[0],GPU_memory_used_MiB=gg[1],GPU_memory_total_MiB=gg[2],GPU_temperature_C=gg[3],GPU_power_W=gg[4],process=s,free_bytes=shutil.disk_usage(R).free);f.write(json.dumps(row)+'\n');f.flush();data.append(row);prev=s;then=now;time.sleep(max(0,1-(time.monotonic()-now)))
after=updates();new=after[len(before):]
def summary(values):
 a=sorted(values);return dict(n=len(a),mean=statistics.mean(a),min=a[0],p10=a[int(.1*(len(a)-1))],median=statistics.median(a),p90=a[int(.9*(len(a)-1))],max=a[-1])
result=dict(pid=PID,process_start_ticks=prev['start_ticks'],duration_seconds=time.monotonic()-start,initial_update=before[-1]['update'] if before else 0,final_update=after[-1]['update'],GPU_utilization_percent=summary([r['GPU_utilization_percent'] for r in data]),GPU_process_plus_desktop_memory_MiB=summary([r['GPU_memory_used_MiB'] for r in data]),CPU_process_percent=summary([r['cpu_percent'] for r in data[1:]]),process_RSS_bytes=summary([r['process']['RSS_bytes'] for r in data]),complete_update_seconds=summary([r['seconds'] for r in new]),primary_AA_decisions_per_second=64/statistics.mean([r['seconds'] for r in new]),all_stream_decisions_per_second=104/statistics.mean([r['seconds'] for r in new]),microbatch=1,effective_AA_batch=64,additional_geometry_decisions=32,goal_decisions=8,interpretation='Passive samples only; nvidia-smi resident allocation includes desktop and CUDA allocator. Live torch allocated/reserved and per-stage timing are not exposed by the sealed process and are measured in an isolated clone after safe pause.',GPU_workers_launched=0)
with open(R/'results/PASSIVE_RESOURCE_PROFILE.json','x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
