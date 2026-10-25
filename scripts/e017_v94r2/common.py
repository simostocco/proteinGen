"""D-only telemetry/publication helpers; no neural imports."""
import collections,csv,datetime,hashlib,json,math,os,shutil,statistics,time
from pathlib import Path
ROOT=Path('/mnt/d/Simone/proteinGen_data/e017_v94r2_corrected_geometry_rl_20261025_v2')
V93=ROOT.parent/'e017_v93_regional_controllability_20261024_v1'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
 return h.hexdigest()
def atomic(path,obj):
 path=Path(path);tmp=path.with_name(path.name+'.part');assert not path.is_symlink() and not tmp.is_symlink()
 with tmp.open('w') as f:json.dump(obj,f,separators=(',',':'),allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);fd=os.open(path.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def projection(trace_bytes=0,committed=0,recent_sizes=()):
 cfg=read(ROOT/'TRAINING_CONFIGURATION_AND_SEEDS.json');seen=set();allocated=0
 for p in (ROOT/'production/checkpoints').glob('*'):
  if not (p.name.endswith('.pt') and p.name.startswith(('recovery_','milestone_','development_best_','incoming_')) or p.name.startswith('incoming_') and p.name.endswith('.pt.part')):continue
  assert not p.is_symlink();s=p.stat();key=(s.st_dev,s.st_ino)
  if key not in seen:seen.add(key);allocated+=s.st_blocks*512
 cp_remaining=max(0,10*cfg['checkpoint_hint_bytes']-allocated)
 observed=math.ceil(statistics.mean(recent_sizes)*max(0,5000-committed)*1.15) if recent_sizes else 0
 traces_remaining=max(0,cfg['proposal_trace_storage_bytes']-trace_bytes,observed)
 required=cp_remaining+traces_remaining+2*2**30+512*2**20;free=shutil.disk_usage(ROOT).free
 return dict(UTC=utc(),admitted=free>=required,D_free_bytes=free,required_remaining_bytes=required,
  verified_current_checkpoint_allocation=allocated,future_checkpoint_bound=cp_remaining,
  future_trace_bound=traces_remaining,trace_allocated_bytes=trace_bytes,operational_reserve_bytes=2*2**30,
  misc_future_bound=512*2**20,margin_bytes=free-required,observed_trace_forecast_15percent_margin=observed)
def extra_metrics(packet,commit):
 rows=packet['records'];goal=packet['task']['goal'];m=[r['metrics'] for r in rows];n=len(rows)
 rewards=[x['reward'] for x in m];llr=[r['trace_logprob']-r['reference_trace_logprob'] for r in rows]
 beta=read(ROOT/'TRAINING_CONFIGURATION_AND_SEEDS.json')['trace_KL_beta'];a=[v-beta*k for v,k in zip(rewards,llr)];adv=[v-(sum(a)-v)/(n-1) for v in a]
 exposed=buried=0;errors=[];degrees=[];collisions=0;bypos=collections.defaultdict(lambda:[0,0,0])
 for r in rows:
  v=r['metrics'];e=v.get('exposed_mean');b=v.get('buried_mean')
  if e is not None and b is not None:
   exposed+=e>=goal['exposure_min'];buried+=b<=goal['burial_max'];errors.append(max(0,goal['exposure_min']-e)+max(0,b-goal['burial_max'])+max(0,goal['contrast_min']-e+b))
  if v['complete']:degrees.append(v['geometry']['longrange_contact_degree'])
  collisions+=v['geometry'].get('clash_count',0)
  for q in r['proposals']:
   d=bypos[q[0]];d[0]+=1;d[1]+=not q[6];d[2]+='provably_exceeds_final_contact_budget' in q[8]
 logical=commit['logical'];s=logical['stats']
 assert all(not v['success'] or v['hard_valid'] for v in m)
 assert all(v['hard_valid'] or v['reward']==0 for v in m)
 return dict(update=packet['update'],identity=logical['identity'],goal_id=goal['goal_id'],difficulty=goal['difficulty'],N=n,
  exposure_success=exposed,burial_success=buried,target_error_mean=statistics.mean(errors) if errors else None,
  full_success=s['full'],hard_valid=s['hard'],backbone_valid=s['backbone'],completed=s['complete'],aborts=s['aborts'],
  completed_overpacking=s['global_contact_violations'],contact_degree_mean=statistics.mean(degrees) if degrees else None,
  contact_degree_max=max(degrees) if degrees else None,clash_pairs=collisions,
  reward_min=min(rewards),reward_max=max(rewards),reward_mean=statistics.mean(rewards),reward_SD=statistics.pstdev(rewards),
  advantage_mean=statistics.mean(adv),advantage_variance=statistics.pvariance(adv),advantage_min=min(adv),advantage_max=max(adv),
  invalid_positive_advantage=sum(not v['hard_valid'] and d>0 for v,d in zip(m,adv)),
  proposals=s['proposals'],accepted_actions=s['accepted_actions'],rejected=s['rejected'],packing_rejected=s['packing_proposals_rejected'],
  proposals_per_accepted=s['proposals']/max(1,s['accepted_actions']),rejection_rate=s['rejected']/max(1,s['proposals']),abort_rate=s['aborts']/n,
  gradient_norm=logical['gradient_norm'],clipped=logical['gradient_norm']>5,learning_rate=logical['lr'],
  reference_KL_per_active_dimension=s['geometry_KL'],step_KL_max_active_sum=logical['step_KL'],full_trace_KL_MC=s['trace_log_ratio'],
  diagnostics=logical['diagnostics'],proposal_counts_by_residue=dict(bypos),seconds=commit['seconds'],UTC=commit['UTC'])
def live():
 p=ROOT/'production';commits=sorted((p/'commits').glob('u*.json'));tot=collections.Counter();secs=[];last=None
 for f in commits:
  x=read(f);last=x;tot.update({k:x['logical']['stats'][k] for k in ['attempts','complete','aborts','hard','full','proposals','rejected','global_contact_violations']});secs.append(x['seconds'])
 idx=read(p/'RECOVERY_INDEX.json') if (p/'RECOVERY_INDEX.json').exists() else None
 stop=read(p/'STOP_RECORD.json') if (p/'STOP_RECORD.json').exists() else None
 failure=read(p/'EXECUTION_FAILURE.json') if (p/'EXECUTION_FAILURE.json').exists() else None
 launch=read(ROOT/'LAUNCH_RECEIPT.json') if (ROOT/'LAUNCH_RECEIPT.json').exists() else None
 terminal=read(ROOT/'WORKER_EXIT.json') if (ROOT/'WORKER_EXIT.json').exists() else None
 result=read(ROOT/'FINAL_PROSPECTIVE_COMPARISON.json') if (ROOT/'FINAL_PROSPECTIVE_COMPARISON.json').exists() else None
 status='INITIALIZING' if launch else 'ADMISSION'
 if commits:status='RUNNING'
 if (p/'TRAINING_COMPLETED.json').exists():status='COMPLETED_FINAL_EVALUATION_PENDING'
 if stop:status='STOPPED_PREDEFINED_SAFETY_GATE'
 if failure:status='FAILED_RECOVERY_REQUIRED'
 if terminal and not stop and not failure and not (p/'TRAINING_COMPLETED.json').exists():status='PAUSED_OR_INFRASTRUCTURE_STOP'
 if terminal and (p/'TRAINING_COMPLETED.json').exists() and terminal.get('final_evaluation_exitcode') not in [None,0]:status='COMPLETED_FINAL_EVALUATION_FAILED'
 if result:status='COMPLETED_SCIENTIFIC_RESULTS_AVAILABLE'
 baseline=read(p/'diagnostics/DEVELOPMENT_00000.json')['summary'] if (p/'diagnostics/DEVELOPMENT_00000.json').exists() else None
 return dict(UTC=utc(),V94R2_LIVE_STATUS=status,TRAINER_PID=launch['PID'] if launch and not terminal else None,
  COMMITTED_UPDATES=len(commits),TARGET_UPDATES=5000,GENERATION_ATTEMPTS=tot['attempts'],TARGET_ATTEMPTS=80000,
  COMPLETED_TRAJECTORIES=tot['complete'],ABORTED_ATTEMPTS=tot['aborts'],RECENT_SECONDS_PER_UPDATE=statistics.mean(secs[-50:]) if secs else None,
  D_FREE_BYTES=shutil.disk_usage(ROOT).free,D_FREE_GIB=shutil.disk_usage(ROOT).free/2**30,
  VALIDITY_STATUS='BREACH' if stop else 'PASS_AT_LAST_MONITOR' if baseline else 'INITIAL_BASELINE_PENDING',
  PACKING_FAILURE_RATE=tot['global_contact_violations']/max(1,tot['attempts']),PROPOSAL_ABORT_RATE=tot['aborts']/max(1,tot['attempts']),
  KL_STATUS='PASS_COMMITTED_UPDATES' if last else 'PENDING',RECOVERY_STATUS='ATOMIC_CHECKPOINT_PLUS_DURABLE_JOURNAL' if idx else 'PENDING',
  LATEST_CHECKPOINT=idx['latest'] if idx else None,counters=dict(tot),baseline=baseline,stop=stop,failure=failure,
  launch=launch,terminal=terminal,scientific_result=result)
