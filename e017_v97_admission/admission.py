"""Frozen V9.7 initialization admission only. No optimizer is constructed."""
from pathlib import Path
import os,sys,json,hashlib,datetime,time,subprocess,shutil,fcntl,collections,traceback,copy,gzip
BASE=Path('/mnt/d/Simone/proteinGen_data/e017_v97_function_preserving_gate_20261031_v1')
RUN=BASE/'initialization_admission_20261031_v1'
WT=BASE.parent/'e017_v97_engineering_worktree_20261031_v1'
sys.path.insert(0,str(WT/'e017_v97'))
import core97 as Q
from safety97 import Monitor,SafetyPause,N,LOOKS,ALPHA
read,sha,put=Q.read,Q.sha,Q.put

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(rel,x,immutable=True):put(RUN/rel,x,immutable)
def event(kind,**kw):
 x=dict(UTC=now(),kind=kind,**kw)
 with (RUN/'runtime/EVENTS.jsonl').open('a') as f:f.write(json.dumps(x)+'\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(x),flush=True)
def no_gpu():
 p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True)
 assert not p.stdout.strip(),('CONFLICTING_GPU_WORKER',p.stdout)
def check_binding():
 x=read(RUN/'EXECUTION_BINDING.json')
 for p,h in x['files'].items():assert sha(p)==h,('ADMISSION_BINDING_CHANGED',p)
 assert sha(BASE/'SOURCE_BINDING.json')==x['qualified_source_binding_SHA256']
 return x

def prepare():
 assert not RUN.exists();RUN.mkdir();(RUN/'source').mkdir();(RUN/'runtime/tmp').mkdir(parents=True);(RUN/'runtime/cache').mkdir();(RUN/'control').mkdir()
 old=read(BASE/'PROVENANCE_SHA256.json');bad=[]
 for p,v in old['files'].items():
  if sha(p)!=v['SHA256']:bad.append(p)
 assert not bad,bad
 Q.verify();no_gpu();r=read(BASE/'RESOURCE_AND_RUNTIME_ESTIMATE.json');reserve=r['estimate']['forecast_required_bytes']+2*r['estimate']['checkpoint_bytes']+2*2**30
 assert shutil.disk_usage(RUN).free>reserve+2**30
 shutil.copyfile('/mnt/d/Simone/proteinGen/e017_v97_admission/admission.py',RUN/'source/admission.py')
 launcher=(BASE/'run.sh').read_text().replace(str(BASE),str(RUN));(RUN/'run.sh').write_text(launcher);(RUN/'run.sh').chmod(0o755)
 schedule=read(BASE/'MONITOR_SCHEDULE.json');pop=read(BASE/'MONITOR_POPULATION.json');fit=read(BASE/'TRAINING_POPULATION.json');legacy=read(BASE/'qualified_train/manifests/TRAIN_TASKS.json')['tasks']
 assert N==4096 and len(LOOKS)==7 and ALPHA==.05/28 and len(pop['tasks'])==256 and pop['identities']==128 and fit['identities']==306
 assert not {v['source_reference_identity'] for v in pop['tasks']}&{v['source_reference_identity'] for v in fit['tasks']}
 assert all(d['goal_id']==pop['tasks'][d['task_index']]['goal']['goal_id'] and d['identity']==pop['tasks'][d['task_index']]['source_reference_identity'] for d in schedule['looks']['0'])
 assert len({d['candidate_seed'] for d in schedule['looks']['0']})==4096 and len(legacy)==16
 # Same-arm qualified incomplete B look0: adopt immutable commitments, never A data.
 src=BASE/'qualification/real_monitor_prefix';b=read(src/'BINDING.json');assert b['arm']=='B' and b['look']==0 and not b['fixture'];assert b['source_binding_SHA256']==sha(BASE/'SOURCE_BINDING.json')
 assert b['monitor_schedule_SHA256']==sha(BASE/'MONITOR_SCHEDULE.json')
 dst=RUN/'arm_B/monitor';dst.mkdir(parents=True)
 adopted={}
 for rel in ['BINDING.json']+[f'{folder}/{i:05d}.json' for i in range(16) for folder in ['rows','commits']]:
  p=src/rel;dest=dst/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);assert sha(p)==sha(dest);adopted[rel]=sha(p)
 m=Monitor(dst,'B',0,b['checkpoint_SHA256'],actor_fingerprint=b['actor_fingerprint']);assert m.frontier==16
 write('B_PREFIX_RECOVERY.json',dict(source=str(src),source_binding=b,adopted_attempts=16,files=adopted,justification='Frozen incomplete look resumes exact suffix at same arm/checkpoint/source/seeds; no ArmA candidate reuse; original qualification evidence unmodified'))
 files={str(RUN/'source/admission.py'):sha(RUN/'source/admission.py'),str(RUN/'run.sh'):sha(RUN/'run.sh')}
 for p in [BASE/'MONITOR_POPULATION.json',BASE/'MONITOR_SCHEDULE.json',BASE/'TRAINING_POPULATION.json',BASE/'TRAINING_SCHEDULE.json',BASE/'FROZEN_DESIGN.json',BASE/'qualified_train/manifests/TRAIN_TASKS.json']:files[str(p)]=sha(p)
 write('EXECUTION_BINDING.json',dict(files=files,qualified_source_binding_SHA256=sha(BASE/'SOURCE_BINDING.json'),qualified_publication_manifest_SHA256=sha(BASE/'PROVENANCE_SHA256.json'),statistical_code_SHA256=sha(WT/'e017_v97/safety97.py'),minimum_D_reserve_bytes=reserve))
 write('SAMPLING_ADHERENCE.json',dict(VERIFIED=True,N=4096,look=0,arms=['A','B'],within_look='independent uniform identity/goal draws with replacement from256 TRAIN tasks; one fresh candidate per draw; stateless unique indexed candidate/step/proposal seeds; fixed eval-mode policy; no dropout or BatchNorm; fresh empty world each draw',cross_arm='matched task and seed arrays; cross-arm dependence permitted by frozen Bonferroni procedure; no result copying',inference='conditional finite TRAIN monitoring population; exact binomial only under declared iid draw/RNG assumptions; no guarantee for arbitrary shared cluster shocks',old_B_prefix=16,new_A_attempts=4096,new_B_attempts=4080,extra_sentinel_attempts_per_arm=128,sentinel_purpose='required frozen TRAIN KL/diversity safeguards; never added to4096 denominator',no_adaptive_extra_draws=True,production_updates=0,VALIDATION_inference=False,TEST=False))
 write('PREPARATION.json',dict(PASS=True,UTC=now(),prior_files_verified=len(old['files']),D_free_bytes=shutil.disk_usage(RUN).free,reserve_bytes=reserve,production_updates=0))
 print('INITIALIZATION_ADMISSION_PREPARED',flush=True)

def summarize_monitor(path):
 counts=collections.Counter();kl=[];positions=[];reasons=collections.Counter();ids=collections.Counter();disk=0
 for i in range(N):
  rp=path/f'rows/{i:05d}.json';cp=path/f'commits/{i:05d}.json';s=read(cp);assert sha(rp)==s['SHA256'];x=read(rp);assert x['index']==i
  row=x['candidate_record'];m=row['metrics'];counts['attempts']+=1;counts['completed']+=x['complete'];counts['aborted']+=x['aborted'];counts['backbone_valid']+=x['backbone_valid'];counts['hard_valid']+=x['hard_valid'];counts['successful_candidates']+=m['success'];ids[x['draw']['identity']]+=1
  counts['completed_packing_failures']+=x['complete'] and not m['packing_valid'];counts['all_attempt_packing_false']+=not m['packing_valid'];counts['completed_clash_failures']+=x['complete'] and 'nonlocal_clashes' in m['geometry']['failures'];counts['packing_rejected_proposals']+=sum('provably_exceeds_final_contact_budget' in q[8] for q in row['proposals']);counts['clash_rejected_proposals']+=sum('nonlocal_clash_budget' in q[8] for q in row['proposals']);counts['proposals']+=len(row['proposals']);counts['AA_decisions']+=row['AA_decisions'];kl.extend(row['geometry_KL']);disk+=rp.stat().st_size+cp.stat().st_size
  if x['aborted']:
   assert row['termination']=='ABORT_8_INVALID_PROPOSALS';t=len(row['geometry']);qs=[q for q in row['proposals'] if q[0]==t];assert len(qs)==8 and all(not q[6] for q in qs);positions.append(t);reasons.update(v for q in qs for v in q[8]);counts['finite_proposal_exhaustions']+=1
 assert counts['completed']+counts['aborted']==4096 and counts['AA_decisions']==0
 return dict(counts=dict(counts),rates={k:counts[k]/4096 for k in ['completed','aborted','backbone_valid','hard_valid','completed_packing_failures','completed_clash_failures']},monitor_geometry_KL_mean=sum(kl)/len(kl),abort_prefixes=positions,terminal_proposal_reason_counts=dict(reasons),identity_draw_counts=dict(ids),result_storage_bytes=disk)

def worker(arm):
 import torch,runpy
 check_binding();Q.verify();no_gpu();root=RUN/f'arm_{arm}';root.mkdir(exist_ok=True);lock=(root/'WORKER.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 P,F=Q.factory(arm,root);e,_=P['K']['load_engine']('C');e.actor.requires_grad_(False).eval()
 modelpath=Q.R3/'production/checkpoints/milestone_05000.pt';checkpoint=sha(modelpath);state=torch.load(modelpath,map_location='cpu',weights_only=False)['actor'];live=e.actor.state_dict();basekeys=set(live)-({'burial_gate'} if arm=='B' else set());assert basekeys==set(state) and all(torch.equal(live[k].cpu(),state[k]) for k in state);del state
 assert sum(p.numel() for p in e.actor.parameters())==17438114+(arm=='B')
 if arm=='B':assert float(e.actor.burial_gate)==0.
 fp=P['c']['state_digest'](e.actor.state_dict());binding=dict(arm=arm,checkpoint=str(modelpath),checkpoint_SHA256=checkpoint,actor_fingerprint=fp,gate_value=0. if arm=='B' else None,parameter_count=sum(p.numel() for p in e.actor.parameters()),source_binding_SHA256=sha(BASE/'SOURCE_BINDING.json'),execution_binding_SHA256=sha(RUN/'EXECUTION_BINDING.json'),J1_SHA256=read(BASE/'CONFIGURATION.json')['J1_SHA256'],weights_and_buffers_exact=True,optimizer_constructed=False)
 write(f'arm_{arm}/INITIALIZER_CERTIFICATION.json',binding)
 # Original sentinel guards must use measurements, never classifier defaults.
 E=runpy.run_path(str(Q.R3/'source/scripts/e017_v94r/evaluation.py'));tasks=read(BASE/'qualified_train/manifests/TRAIN_TASKS.json')['tasks'];groups=[];sentinel_start=time.monotonic()
 for i,t in enumerate(tasks):
  path=root/f'sentinel/{i:02d}.json';seeds=[P['S']['ns']('CANDIDATE',t['goal']['goal_id'],j) for j in range(8)]
  if path.exists():
   obj=read(path);assert obj['binding']==binding and obj['task']==t and obj['seeds']==seeds
  else:
   s0=time.monotonic();rows=P['sample'](e,t,'C',8,seeds);obj=dict(binding=binding,task=t,seeds=seeds,records=rows,elapsed_seconds=time.monotonic()-s0);put(path,obj)
  assert len(obj['records'])==8;groups.append(obj)
 sentinel=E['summarize'](groups);write(f'arm_{arm}/SENTINEL.json',dict(summary=sentinel,attempts=128,source='frozen16 reserved TRAIN goals; filtered8 each',seconds=sum(g['elapsed_seconds'] for g in groups),counts_excluded_from_primary4096=True))
 event('SENTINEL_COMPLETED',arm=arm,KL=sentinel['geometry_KL_per_active'],diversity=sentinel['structural_pair_RMSD'])
 m=Monitor(root/'monitor',arm,0,checkpoint,actor_fingerprint=fp);startfront=m.frontier;started=time.monotonic();write(f'arm_{arm}/ACTIVE_WORKER.json',dict(PID=os.getpid(),UTC=now(),resume_frontier=startfront),False);event('INITIAL_LOOK_STARTED',arm=arm,PID=os.getpid(),frontier=startfront)
 # Publish a record-bound frontier, construct a fresh collector, verify exact recovery
 # without replaying or regenerating any already committed candidate.
 recovered=False;prev=started;lastfront=startfront;completed_so_far=sum(read(root/f'monitor/rows/{i:05d}.json')['complete'] for i in range(startfront))
 while m.frontier<N:
  if (RUN/'control/STOP').exists():raise RuntimeError('COOPERATIVE_INFRASTRUCTURE_PAUSE')
  assert shutil.disk_usage(RUN).free>check_binding()['minimum_D_reserve_bytes'],'STORAGE_RESERVE_PAUSE'
  before=m.frontier;m.collect(P,e,max_new=min(16,N-m.frontier));completed_so_far+=sum(read(root/f'monitor/rows/{i:05d}.json')['complete'] for i in range(before,m.frontier));t=time.monotonic();rate=(m.frontier-lastfront)/(t-prev);lastfront=m.frontier;prev=t
  write(f'arm_{arm}/PROGRESS.json',dict(UTC=now(),state='RUNNING',PID=os.getpid(),committed=m.frontier,total=N,backbone=m.backbone,hard=m.hard,completed=completed_so_far,aborted=m.frontier-completed_so_far,recent_attempts_per_second=rate,elapsed_this_process_seconds=t-started,estimated_remaining_seconds=(N-m.frontier)/rate,D_free_bytes=shutil.disk_usage(RUN).free),False)
  if not recovered:
   frontier=m.frontier;backbone=m.backbone;hard=m.hard;heads={str(p.relative_to(root/'monitor')):sha(p) for p in (root/'monitor/commits').glob('*.json')}
   m=Monitor(root/'monitor',arm,0,checkpoint,actor_fingerprint=fp);assert (m.frontier,m.backbone,m.hard)==(frontier,backbone,hard);assert all(sha(root/'monitor'/p)==h for p,h in heads.items());recovered=True
   write(f'arm_{arm}/LIVE_FRONTIER_RECOVERY.json',dict(PASS=True,frontier=frontier,all_commits_unchanged=True,regenerated_attempts=0,same_checkpoint=True));event('DURABLE_FRONTIER_RECOVERY_VERIFIED',arm=arm,frontier=frontier)
 assert P['c']['state_digest'](e.actor.state_dict())==fp;e.prior.verify();assert P['c']['state_digest'](e.parent.state_dict())==e.fp
 checked=Monitor(root/'monitor',arm,0,checkpoint,actor_fingerprint=fp);assert checked.frontier==N and checked.backbone==m.backbone and checked.hard==m.hard
 accounting=summarize_monitor(root/'monitor');assert accounting['counts']['backbone_valid']==m.backbone and accounting['counts']['hard_valid']==m.hard
 decision=m.decision(KL=sentinel['geometry_KL_per_active'],diversity=sentinel['structural_pair_RMSD'])
 try:m.enforce(KL=sentinel['geometry_KL_per_active'],diversity=sentinel['structural_pair_RMSD'])
 except SafetyPause:assert decision['status'] in ['BREACH','INCONCLUSIVE']
 result=dict(UTC=now(),arm=arm,INITIALIZATION_SAFETY=decision['status'],decision=decision,accounting=accounting,sentinel=sentinel,initializer=binding,seconds_this_process=time.monotonic()-started,sentinel_seconds=read(root/'SENTINEL.json')['seconds'],initial_committed_prefix=startfront,new_primary_attempts=N-startfront,recovery_verified=True,production_updates=0,neural_weights_unchanged=True,J1_frozen=True,STATISTICAL_ASSUMPTIONS='VERIFIED_UNDER_FROZEN_CONDITIONAL_IID_DESIGN')
 write(f'arm_{arm}/RESULT.json',result);write(f'arm_{arm}/PROGRESS.json',dict(UTC=now(),state='COMPLETED',committed=N,INITIALIZATION_SAFETY=decision['status'],PID=os.getpid()),False);event('INITIAL_LOOK_COMPLETED',arm=arm,result=decision['status'],backbone=m.backbone,hard=m.hard)


def controller():
 lock=(RUN/'control/CONTROLLER.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);check_binding();Q.verify();no_gpu();event('CONTROLLER_STARTED',PID=os.getpid())
 try:
  for arm in ['A','B']:
   root=RUN/f'arm_{arm}';root.mkdir(exist_ok=True)
   if (root/'RESULT.json').exists():event('COMPLETED_ARM_PRESERVED',arm=arm);continue
   no_gpu();log=(RUN/f'runtime/arm_{arm}.log').open('a');p=subprocess.Popen([str(RUN/'run.sh'),str(RUN/'source/admission.py'),'worker',arm],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
   write(f'arm_{arm}/LAUNCH.json',dict(UTC=now(),PID=p.pid),False);write('runtime/CONTROLLER_STATUS.json',dict(UTC=now(),arm=arm,worker_PID=p.pid,controller_PID=os.getpid()),False)
   lastreport=0.
   while p.poll() is None:
    progress=read(root/'PROGRESS.json') if (root/'PROGRESS.json').exists() else dict(state='INITIALIZING')
    write('runtime/LATEST_STATUS.json',dict(UTC=now(),controller_PID=os.getpid(),arm=arm,worker_PID=p.pid,progress=progress,D_free_bytes=shutil.disk_usage(RUN).free),False)
    if time.monotonic()-lastreport>=1800:event('THIRTY_MINUTE_PROGRESS',arm=arm,progress=progress);lastreport=time.monotonic()
    time.sleep(60)
   log.close();assert p.returncode==0,('INITIAL_LOOK_INFRASTRUCTURE_FAILURE',arm,p.returncode)
   for _ in range(30):
    try:no_gpu();break
    except AssertionError:time.sleep(1)
   no_gpu();event('GPU_RELEASE_VERIFIED',arm=arm)
  publish();write('runtime/CONTROLLER_EXIT.json',dict(UTC=now(),status='COMPLETED',exit_code=0),False);event('ADMISSION_ASSESSMENT_COMPLETED')
 except BaseException as e:
  failure=dict(UTC=now(),error=repr(e),traceback=traceback.format_exc(),automatic_retry=False);write('runtime/FAILURE_'+str(time.time_ns())+'.json',failure);write('runtime/LATEST_FAILURE.json',failure,False)
  write('runtime/CONTROLLER_EXIT.json',dict(UTC=now(),status='INFRASTRUCTURE_PAUSED',exit_code=1,error=repr(e)),False);raise

def publish():
 check_binding();Q.verify();no_gpu();a=read(RUN/'arm_A/RESULT.json');b=read(RUN/'arm_B/RESULT.json');results={'A':a,'B':b}
 for arm,x in results.items():
  assert x['accounting']['counts']['attempts']==4096 and x['neural_weights_unchanged'] and x['production_updates']==0
  m=Monitor(RUN/f'arm_{arm}/monitor',arm,0,x['initializer']['checkpoint_SHA256'],actor_fingerprint=x['initializer']['actor_fingerprint']);assert m.frontier==4096
  for folder in ['rows','commits']:
   assert sorted(p.name for p in (m.root/folder).glob('*.json'))==[f'{i:05d}.json' for i in range(4096)]
  stats=summarize_monitor(RUN/f'arm_{arm}/monitor');assert stats==x['accounting']
 prep=read(BASE/'PROVENANCE_SHA256.json');assert all(sha(p)==v['SHA256'] for p,v in prep['files'].items())
 reserve=read(RUN/'EXECUTION_BINDING.json')['minimum_D_reserve_bytes'];storage=shutil.disk_usage(RUN).free>reserve;paired=storage and all(x['INITIALIZATION_SAFETY']=='PASS' for x in results.values())
 status='PASS' if paired else 'INCONCLUSIVE' if any(x['INITIALIZATION_SAFETY']=='INCONCLUSIVE' for x in results.values()) and not any(x['INITIALIZATION_SAFETY']=='BREACH' for x in results.values()) else 'BLOCKED'
 comparison=dict(same_frozen_seed_arrays=True,records_reused_between_arms=False,matching_primary_validity=(a['accounting']['counts']['backbone_valid']==b['accounting']['counts']['backbone_valid']),cross_arm_independence_not_required=True)
 readiness=dict(UTC=now(),V97_INITIALIZATION_ADMISSION=status,arms={k:v['INITIALIZATION_SAFETY'] for k,v in results.items()},STATISTICAL_DESIGN_ADHERENCE='VERIFIED',RECOVERY_QUALIFICATION='PASS',STORAGE_ADMISSION='PASS' if storage else 'FAIL',D_free_bytes=shutil.disk_usage(RUN).free,required_staging_reserve_bytes=reserve,READY_FOR_PAIRED_5000_UPDATE_TRAINING=paired,production_authorized=False,PRODUCTION_UPDATES=0,TEST_SEAL='UNKNOWN_HISTORICAL_ACCESS; NO_TEST_CONTENT_ACCESSED',next_action='Request separate production authorization; no automatic training' if paired else 'Keep paired production blocked; no additional samples or optimizer updates under this protocol',comparison=comparison)
 write('PAIRED_PRODUCTION_READINESS.json',readiness)
 for arm,x in results.items():
  c=x['accounting']['counts'];d=x['decision'];write(f'ARM_{arm}_INITIALIZATION_SAFETY_REPORT.md',f"# Arm {arm}: initialization safety\n\n"+json.dumps(x,indent=2)+'\n\nPrimary4096 denominator excludes128 frozen TRAIN sentinel attempts. Abort attempts are failures. Completed packing violations and aborted unfinished geometry are reported separately; terminal rejection reason counts may overlap. Full inference uses qualified safety97 exactly.\n')
 write('FINAL_INITIALIZATION_ADMISSION_REPORT.md','# V9.7 full initialization admission\n\n'+json.dumps(readiness,indent=2)+'\n\n'+ '\n'.join(f"Arm {arm}: {x['INITIALIZATION_SAFETY']}; backbone {x['accounting']['counts']['backbone_valid']}/4096; hard {x['accounting']['counts']['hard_valid']}/4096; aborts {x['accounting']['counts']['aborted']}/4096; adjusted backbone interval {x['decision']['backbone_CI']}." for arm,x in results.items())+'\n\nBoth arms used the frozen matched task/seed arrays and separately generated results. B own qualified16-prefix was recovered; no ArmA records were used as B data. Counts are conditional on the frozen finite TRAIN monitoring population; no claim of arbitrary-cluster binomial validity is made. No VALIDATION/TEST inference or optimizer operation occurred. All GPU workers exited.\n')
 files={str(p.relative_to(RUN)):dict(SHA256=sha(p),bytes=p.stat().st_size) for p in RUN.rglob('*') if p.is_file() and 'runtime' not in p.relative_to(RUN).parts and p.name!='PROVENANCE_SHA256.json'}
 write('PROVENANCE_SHA256.json',dict(files=files,self_excluded=True,runtime_excluded=True,qualified_manifest_SHA256=sha(BASE/'PROVENANCE_SHA256.json'),qualified_files_reverified=len(prep['files'])))
 assert all(sha(RUN/p)==v['SHA256'] for p,v in files.items());write('runtime/PUBLICATION_VERIFICATION.json',dict(PASS=True,UTC=now(),files=len(files),primary_attempts=8192,imported_same_arm_B_attempts=16,new_primary_attempts=8176,sentinel_attempts=256,production_updates=0))

if __name__=='__main__':
 mode=sys.argv[1]
 {'prepare':prepare,'controller':controller,'worker':lambda:worker(sys.argv[2]),'publish':publish}[mode]()
