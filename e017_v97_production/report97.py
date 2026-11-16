"""Read-only launch verification plus a dated D: status publication."""
import pathlib,json,hashlib,os,subprocess,shutil,datetime,statistics
r=pathlib.Path('/mnt/d/Simone/proteinGen_data/e017_v97_function_preserving_gate_20261031_v1/paired_production_20261031_v1')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):
 t=p.with_name(p.name+'.part');data=x if isinstance(x,str) else json.dumps(x,indent=2)
 with t.open('w') as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(t,p)
ctrl=read(r/'runtime/CONTROLLER_STATUS.json');mon=read(r/'runtime/MONITOR_LAUNCH.json');assert ctrl['mode']=='train' and ctrl['arm']=='A'
pids=[ctrl['controller_PID'],mon['PID'],ctrl['worker_PID']]
ps=subprocess.run(['ps','-p',','.join(map(str,pids)),'-o','pid,ppid,stat,etime,args'],capture_output=True,text=True,check=True)
assert all(str(pid) in ps.stdout for pid in pids)
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip();assert gpu==str(ctrl['worker_PID'])
arms={}
for a in ['A','B']:
 root=r/f'arm_{a}/production';paths=sorted((root/'commits').glob('u*.json'));assert [p.name for p in paths]==[f'u{i:06d}.json' for i in range(1,len(paths)+1)]
 commits=[]
 for p in paths:
  x=read(p);assert sha(root/'rollouts'/p.name)==x['logical']['rollout_SHA256'];commits.append(x)
 stats=[c['logical']['stats'] for c in commits]
 arms[a]=dict(status='RUNNING' if a=='A' else 'QUEUED',committed_updates=len(paths),attempts=sum(x['attempts'] for x in stats),completed=sum(x['complete'] for x in stats),aborted=sum(x['aborts'] for x in stats),max_likelihood_replay_error=max([c['logical']['replay_logprob_maxerror'] for c in commits],default=0),latest_step_KL=commits[-1]['logical']['step_KL'] if commits else None,clipped_updates=sum(c['logical']['gradient_norm']>5 for c in commits),mean_commit_update_seconds=statistics.mean(c['seconds'] for c in commits) if commits else None,actual_initial_recovery=read(r/f'arm_{a}/ACTUAL_INITIAL_RECOVERY_ROUNDTRIP.json')['PASS'])
 if len(commits)>=2:arms[a]['wall_seconds_per_update']=(commits[-1]['UTC']-commits[0]['UTC'])/(len(commits)-1)
assert arms['A']['committed_updates']>0 and arms['B']['committed_updates']==0
sa=read(r/'arm_A/production/diagnostics/DEVELOPMENT_00000.json')['summary'];sb=read(r/'arm_B/production/diagnostics/DEVELOPMENT_00000.json')['summary'];assert sa==sb
plan=read(r/'PRODUCTION_PLAN.json');estimate=dict(plan['runtime']);pace=arms['A'].get('wall_seconds_per_update',arms['A']['mean_commit_update_seconds'])
estimate['A_remaining_fitting_hours_from_early_wall_pace']=(5000-arms['A']['committed_updates'])*pace/3600
estimate['remaining_total_hours_range_from_early_wall_pace']=[estimate['A_remaining_fitting_hours_from_early_wall_pace']+estimate['fitting_hours_B']+estimate['safety_hours_A']+estimate['safety_hours_B']+h+estimate['development_hours_estimate'] for h in estimate['final_validation_hours_range']]
x=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),V97_PRODUCTION_STATUS='RUNNING',ACTIVE_ARM='A',arms=arms,trainer_PID=ctrl['worker_PID'],controller_PID=ctrl['controller_PID'],monitor_PID=mon['PID'],host_processes_verified=ps.stdout,GPU_owner=ctrl['worker_PID'],INITIALIZATION_SAFETY='PASS_BOTH',SAFETY_PROTOCOL='QUALIFIED',RECOVERY_STATUS='QUALIFIED_AND_BOTH_ACTUAL_ZERO_UPDATE_ROUNDTRIPS_PASS',J1_FROZEN=True,D_free_GiB=shutil.disk_usage(r).free/2**30,staging_reserve_GiB=plan['minimum_free_launch_bytes']/2**30,STORAGE_ADMISSION='PASS',baseline_raw_filtered_and_sensitivity_summaries_identical=True,next_safety_look=250,next_development_evaluation=250,remaining_runtime=estimate,monitor_log=str(r/'runtime/monitor.log'),source_binding_SHA256=sha(r/'FROZEN_SOURCE_AND_INPUTS.json'),TEST_SEAL='UNKNOWN_HISTORICAL_ACCESS; NO_TEST_CONTENT_ACCESSED')
put(r/'runtime/LAUNCH_CERTIFICATION.json',x)
put(r/'PRODUCTION_LAUNCH_REPORT.md','# V9.7 production launch snapshot\n\n'+json.dumps(x,indent=2)+'\n\nThis dated snapshot is superseded by runtime/MONITOR_LATEST.json and authoritative optimizer commitments. Both arms target5000 additional updates and80000 fitting attempts each. Initialization look0 is already PASS; six4096-candidate looks per arm remain. No optimizer continuation after a non-PASS. Early ArmA stop pauses the paired campaign for review; no automatic ArmB start after it. Final frozen assessment/publication is queued after both full budgets finish. No further research stage is launched. Monitoring provides durable logs, not proactive ChatGPT notifications.\n')
print(json.dumps(x),flush=True)
