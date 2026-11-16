"""Controlled V9.6 paired continuation: original scientific runtime, isolated arms."""
import argparse, collections, copy, datetime, fcntl, gzip, hashlib, json, math, os
from pathlib import Path
import runpy, shutil, signal, subprocess, sys, time, traceback

PAIR=Path('/mnt/d/Simone/proteinGen_data/e017_v96_paired_burial_control_20261029_v1')
ENG=PAIR.parent/'e017_v96_burial_conditioning_20261029_v1'
R3=PAIR.parent/'e017_v94r3_substantial_geometry_rl_20261026_v1'
ARMS={'A':PAIR.parent/'e017_v96_arm_a_original_20261029_v1','B':PAIR.parent/'e017_v96_arm_b_burial_token_20261029_v1'}
sys.path.insert(0,str(ENG/'source96'))
import common96 as old
import algorithm
import numpy as np
import torch

def read(path):
    path=Path(path)
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:return json.load(f)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,obj,immutable=False):
    path=Path(path);assert path.resolve().is_relative_to(PAIR.parent.resolve()) and not path.is_symlink()
    path.parent.mkdir(parents=True,exist_ok=True);payload=json.dumps(obj,separators=(',',':'),allow_nan=False).encode()
    if path.suffix=='.gz':payload=gzip.compress(payload,mtime=0)
    if immutable and path.exists():assert path.read_bytes()==payload,('IMMUTABLE_MISMATCH',str(path));return
    temp=path.with_name(path.name+'.part')
    with temp.open('wb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    os.replace(temp,path);fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def event(kind,**kwargs):
    row={'UTC':now(),'kind':kind,**kwargs}
    with (PAIR/'runtime/EVENTS.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(row),flush=True)
def execution():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def no_gpu_worker():
    x=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True)
    assert not x.stdout.strip(),('CONFLICTING_GPU_WORKLOAD',x.stdout)
def verify():
    frozen=read(PAIR/'FROZEN_SOURCE_AND_INPUTS.json')
    for path,h in frozen['files'].items():assert sha(path)==h,('SOURCE_OR_INPUT_CHANGED',path)
    for path,h in frozen['historical_models'].items():assert sha(path)==h,('CHECKPOINT_CHANGED',path)
    return frozen
def factory(arm):
    root=ARMS[arm];P,F,compat=old.load();loader=P['K']['load_engine']
    def engine(policy):
        e,meta=loader(policy);F['checkpoint'](e,5000)
        assert not getattr(e.actor,'_v96_burial_token_installed',False)
        if arm=='B':algorithm.install(e.actor)
        assert bool(getattr(e.actor,'_v96_burial_token_installed',False))==(arm=='B')
        assert not getattr(e.parent,'_v96_burial_token_installed',False)
        return e,meta
    P['K']['load_engine']=engine
    config=read(root/'TRAINING_CONFIGURATION_AND_SEEDS.json')
    def binding():return dict(schema='V96_PAIRED_FULL_TRACE_1',arm=arm,burial_token=arm=='B',
        paired_source_and_inputs_SHA256=sha(PAIR/'FROZEN_SOURCE_AND_INPUTS.json'),
        configuration_SHA256=sha(root/'TRAINING_CONFIGURATION_AND_SEEDS.json'),
        initializer_SHA256=read(ENG/'HISTORICAL_BINDINGS.json')['models']['milestone_05000']['SHA256'],
        original_scientific_binding=read(ENG/'HISTORICAL_BINDINGS.json')['original_scientific_binding'])
    g=P['Runtime'].__init__.__globals__;g['ROOT']=root;g['SPEC']=config;g['bindings']=binding
    P['ROOT']=root;P['SPEC']=config;P['bindings']=binding
    return P,F
def check_initializer(P,rt,arm):
    state=torch.load(R3/'production/checkpoints/milestone_05000.pt',map_location='cpu',weights_only=False)
    assert rt.e.actor.state_dict().keys()==state['actor'].keys()
    assert all(torch.equal(v.cpu(),state['actor'][k]) for k,v in rt.e.actor.state_dict().items())
    assert sum(p.numel() for p in rt.e.actor.parameters())==17438114
    assert bool(getattr(rt.e.actor,'_v96_burial_token_installed',False))==(arm=='B')
    assert all(float(s['step'])==0 and not s['exp_avg'].count_nonzero() and not s['exp_avg_sq'].count_nonzero() for s in rt.opt.state.values())
    assert rt.update==0 and not rt.counters
    rt.e.prior.verify();assert P['c']['state_digest'](rt.e.parent.state_dict())==rt.e.fp
def safe_initial(s):return s['hard_valid']>=.91875 and s['backbone_valid']>=.95 and s['geometry_KL_per_active']<=1 and s['structural_pair_RMSD']>=12.449118749897715
def safe_monitor(s,b):return s['hard_valid']>=max(b['hard_valid'],.96875)-.05 and s['backbone_valid']>=max(b['backbone_valid'],1.)-.05 and s['geometry_KL_per_active']<=1 and s['structural_pair_RMSD']>=.8*max(b['structural_pair_RMSD'],15.561398437372143)
def goal_score(base,alternate,goal,altgoal,region):
    a,b=goal[region];c,d=altgoal[region];values=[];valid=0
    for x,y in zip(base,alternate):
        assert x['seed']==y['seed'];v=0.
        if x['metrics']['hard_valid'] and y['metrics']['hard_valid']:
            ex=np.asarray(x['metrics']['exposure']);ey=np.asarray(y['metrics']['exposure'])
            v=.5*((ex[a:b].mean()-ey[a:b].mean())+(ey[c:d].mean()-ex[c:d].mean()))
            if region=='buried':v=-v
            valid+=1
        values.append(float(v))
    return dict(S=float(np.mean(values)),valid_pairs=valid,pairs=len(values))
def baseline(P,rt,arm):
    path=rt.root/f'diagnostics/DEVELOPMENT_{rt.update:05d}.json';fp=P['c']['state_digest'](rt.e.actor.state_dict())
    if path.exists():
        x=read(path);assert x['actor_fingerprint']==fp and x['arm']==arm;return x['summary']
    event('SCHEDULED_TRAIN_EVALUATION',arm=arm,update=rt.update)
    tasks=read(P['V93']/'manifests/TRAIN_TASKS.json')['tasks'];assert len(tasks)==16
    E=runpy.run_path(str(R3/'source/scripts/e017_v94r/evaluation.py'));out={};cost=0.;counts=0
    for policy,label in [('C','filtered'),('B','underlying_unfiltered')]:
        groups=[]
        for i,task in enumerate(tasks):
            seeds=[P['S']['ns']('CANDIDATE',task['goal']['goal_id'],j) for j in range(8)]
            gp=rt.root/f'diagnostics/DEV_{rt.update:05d}_{label}_{i:02d}.json'
            if gp.exists():
                obj=read(gp);assert obj['arm']==arm and obj['actor_fingerprint']==fp and obj['task']==task and obj['seeds']==seeds
            else:
                t=time.monotonic();rows=P['sample'](rt.e,task,policy,8,seeds)
                obj=dict(arm=arm,actor_fingerprint=fp,task=task,seeds=seeds,records=rows,seconds=time.monotonic()-t)
                atomic(gp,obj,True)
            groups.append(obj);cost+=obj['seconds'];counts+=8
        out[label]=E['summarize'](groups)
        if policy=='B':out[label]['invalid_proposal_checks']=out[label]['rejections'];out[label]['rejections']=0
    controls=read(ENG/'TRAIN_DIAGNOSTIC_MANIFEST.json');response=[]
    for gi,group in enumerate(controls['groups']):
        for policy in ['B','C']:
            results={}
            for tag,task in [('base',group['base'])]+list(group['alternatives'].items()):
                gp=rt.root/f'diagnostics/SENS_{rt.update:05d}_{gi}_{policy}_{tag}.json'
                seeds=controls['seeds'][gi]
                if gp.exists():
                    x=read(gp);assert x['arm']==arm and x['actor_fingerprint']==fp and x['task']==task and x['seeds']==seeds
                else:
                    t=time.monotonic();rows=P['sample'](rt.e,task,policy,8,seeds)
                    x=dict(arm=arm,actor_fingerprint=fp,task=task,seeds=seeds,records=rows,seconds=time.monotonic()-t);atomic(gp,x,True)
                results[tag]=x['records'];cost+=x['seconds'];counts+=8
            for tag,region in [('burial','buried'),('exposure','exposed')]:
                response.append(dict(group=gi,policy=policy,target=tag,**goal_score(results['base'],results[tag],group['base']['goal'],group['alternatives'][tag]['goal'],region)))
    assert counts==352
    out['TRAIN_sensitivity']=response
    atomic(path,dict(arm=arm,actor_fingerprint=fp,summary=out,attempts=counts,seconds=cost,TRAIN_only=True),True)
    event('TRAIN_EVALUATION_FINISHED',arm=arm,update=rt.update,filtered=out['filtered'],seconds=cost)
    return out
def draws():return dict(torch=torch.rand(8).tolist(),cuda=torch.rand(8,device='cuda').cpu().tolist(),numpy=np.random.random(8).tolist(),python=__import__('random').random())
def qualify(arm):
    verify();execution();no_gpu_worker();root=ARMS[arm]
    lock=(root/'control/WORKER.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    P,F=factory(arm);rt=P['Runtime'](root/'qualification/engineering');check_initializer(P,rt,arm)
    original=P['state_digest'](rt.state());d=draws();rt.restore();assert P['state_digest'](rt.state())==original and draws()==d;rt.restore()
    report=baseline(P,rt,arm)
    if not safe_initial(report['filtered']):
        atomic(root/'qualification/ADMISSION.json',dict(PASS=False,UTC=now(),arm=arm,reason='INITIAL_SAFETY_ADMISSION_FAILED',baseline=report['filtered'],production_updates=0,engineering_unique_updates=0),True)
        event('ARM_ADMISSION',arm=arm,PASS=False,baseline=report['filtered']);raise SystemExit(30)
    # Probe actual stochastic prefix8, not the masked gauge-fixed first step.
    first=read(rt.root/'diagnostics/DEV_00000_filtered_00.json');row=next(r for r in first['records'] if len(r['sequence'])>=8)
    state=P['S']['reconstruct'](row,64,8);world=P['S']['world'](first['task']['goal'],state)
    prefix=P['p']['collate']([state],'cuda')
    with torch.no_grad():ref=rt.e.prior(prefix);batch=P['p']['public_batch']([world],prefix,ref)
    batch['context'].requires_grad_(True);captured=[]
    def capture(module,args):args[0]['goals'].retain_grad();captured.append(args[0]['goals'])
    # For A the unaugmented goal tensor must be made differentiable explicitly.
    batch['goals'].requires_grad_(True)
    h=rt.e.actor.register_forward_pre_hook(capture)
    _,mu,ls=rt.e.actor(batch);aa=P['AA'].index(first['task']['fixed_sequence'][8]);(mu[0,aa].sum()+ls[0,aa].sum()).backward();h.remove()
    assert captured[0].shape[1]==len(first['task']['goal']['motif'])+(arm=='B')
    if arm=='B':assert torch.equal(captured[0][:,-1],algorithm.augment(batch)['goals'][:,-1])
    token_gradient=float(captured[0].grad[0,-1].norm());assert token_gradient>0
    rt.e.actor.zero_grad(set_to_none=True)
    expected=read(PAIR/'TRAINING_SCHEDULE.json')['updates'][0]
    assert rt.task(1)['goal']['goal_id']==expected['goal_id']
    first_step=rt.step(1);post=P['state_digest'](rt.state());rt.restore();again=rt.step(1)
    assert again==first_step and P['state_digest'](rt.state())==post
    rt.publish('ENGINEERING_RECOVERY_QUALIFIED');post=P['state_digest'](rt.state());d=draws();rt.restore();assert P['state_digest'](rt.state())==post and draws()==d;rt.restore()
    binding=rt.binding;rt.binding=dict(binding,arm='INCORRECT')
    try:rt.restore()
    except AssertionError:pass
    else:raise AssertionError('WRONG_ARM_BINDING_ACCEPTED')
    rt.binding=binding;rt.restore()
    old_index=sha(rt.root/'RECOVERY_INDEX.json')
    child=subprocess.run([str(root/'run.sh'),str(PAIR/'source/paired.py'),'partial',arm])
    assert child.returncode==75 and sha(rt.root/'RECOVERY_INDEX.json')==old_index
    rt.restore();assert P['state_digest'](rt.state())==post
    rt.e.prior.verify();assert P['c']['state_digest'](rt.e.parent.state_dict())==rt.e.fp
    admitted=safe_initial(report['filtered'])
    atomic(root/'qualification/ADMISSION.json',dict(PASS=admitted,UTC=now(),arm=arm,original_weights_exact=True,optimizer_fresh=True,parameter_count=17438114,burial_token=(arm=='B'),gradient_probe_prefix=8,goal_gradient_norm=token_gradient,full_state_RNG_roundtrip=True,journal_replay_exact=True,partial_full_checkpoint_recovery=True,wrong_arm_binding_rejected=True,replay_maxerror=first_step['replay_logprob_maxerror'],engineering_unique_updates=1,engineering_optimizer_calls=2,new_training_attempts=16,development_attempts=352,baseline=report['filtered'],production_updates=0),True)
    event('ARM_ADMISSION',arm=arm,PASS=admitted)
    if not admitted:raise SystemExit(30)
def partial(arm):
    from protein_joint_sac.crisp_v3.recovery import atomic_checkpoint,StoragePolicy
    import builtins
    root=ARMS[arm];idx=read(root/'qualification/engineering/RECOVERY_INDEX.json')
    state=torch.load(root/'qualification/engineering'/idx['latest']['relative'],map_location='cpu',weights_only=False)
    target=root/'qualification/INTERRUPTED_STATE.pt';original=builtins.open
    class Fault:
        def __init__(self,f):self.f=f;self.n=0
        def __getattr__(self,n):return getattr(self.f,n)
        def __enter__(self):return self
        def __exit__(self,*a):return self.f.__exit__(*a)
        def write(self,b):
            n=self.f.write(b);self.n+=n
            if self.n>=8*2**20:self.f.flush();os.fsync(self.f.fileno());os._exit(75)
            return n
    def opened(path,*a,**kw):
        f=original(path,*a,**kw);return Fault(f) if str(path)==str(target)+'.part' and a and a[0]=='xb' else f
    builtins.open=opened;atomic_checkpoint(target,state,StoragePolicy(checkpoint_hint_bytes=224*2**20));raise AssertionError('FAULT_NOT_TRIGGERED')
def metrics(rt,P,u):
    path=rt.root/f'metrics/u{u:06d}.json'
    packet=read(rt.root/f'rollouts/u{u:06d}.json');rows=packet['records'];g=packet['task']['goal'];m=[r['metrics'] for r in rows];logical=read(rt.root/f'commits/u{u:06d}.json')['logical']
    ex=sum(x.get('exposed_mean',-1)>=g['exposure_min'] for x in m);bu=sum(x.get('buried_mean',2)<=g['burial_max'] for x in m)
    value=dict(update=u,identity=packet['task']['source_reference_identity'],difficulty=g['difficulty'],reward=logical['stats']['reward_mean'],exposure=ex,burial=bu,success=sum(x['success'] for x in m),packing=sum(x['packing_valid'] for x in m),clash_failures=sum(x['complete'] and 'nonlocal_clashes' in x['geometry']['failures'] for x in m),grad_norm=logical['gradient_norm'],clipped=logical['gradient_norm']>5,entropy=logical['diagnostics']['latent_Gaussian_entropy_per_active_dimension'],stats=logical['stats'],seconds=read(rt.root/f'commits/u{u:06d}.json')['seconds'])
    atomic(path,value,True)
def training(arm):
    verify();execution();no_gpu_worker();root=ARMS[arm]
    assert read(PAIR/'PRODUCTION_ADMISSION.json')['PASS'];assert read(root/'qualification/ADMISSION.json')['PASS']
    receipt=read(root/'PRODUCTION_AUTHORIZATION.json');assert receipt['explicit_review_authorized'] and receipt['configuration_SHA256']==sha(root/'TRAINING_CONFIGURATION_AND_SEEDS.json') and receipt['frozen_inputs_SHA256']==sha(PAIR/'FROZEN_SOURCE_AND_INPUTS.json')
    lock=(root/'control/WORKER.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'production/STOP_RECORD.json').exists(),'REGISTERED_STOP_MUST_NOT_BE_RESTARTED'
    P,F=factory(arm);rt=P['Runtime'](root/'production')
    if rt.update==0:check_initializer(P,rt,arm)
    if rt.baseline is None:
        rt.baseline=baseline(P,rt,arm);rt.best=dict(update=0,**rt.baseline['filtered']);rt.publish('DEVELOPMENT_BEST')
    assert safe_initial(rt.baseline['filtered'])
    (rt.root/'metrics').mkdir(exist_ok=True)
    plan=read(PAIR/'TRAINING_SCHEDULE.json')['updates'];milestones=set(P['SPEC']['development_updates'])
    def callbacks(u):
        metrics(rt,P,u)
        if u in milestones:
            report=baseline(P,rt,arm);s=report['filtered']
            if not safe_monitor(s,rt.baseline['filtered']):
                atomic(root/'control/STOP',dict(reason='REGISTERED_VALIDITY_KL_DIVERSITY_STOP',update=u))
                rt.publish('REGISTERED_SAFETY_STOP')
                atomic(rt.root/'STOP_RECORD.json',dict(UTC=now(),arm=arm,update=u,kind='REGISTERED_SAFETY_STOP',report=report,baseline=rt.baseline,thresholds_unchanged=True),True)
                event('SAFETY_STOP',arm=arm,update=u,report=s);raise RegisteredStop()
            if (s['K1'],s['K8'],s['hard_valid'])>(rt.best['K1'],rt.best['K8'],rt.best['hard_valid']):rt.best=dict(update=u,**s);rt.publish('DEVELOPMENT_BEST')
        if u%50==0:rt.publish('ROLLING')
    def step(u):
        task=rt.task(u);schedule=plan[u-1]
        assert task['goal']['goal_id']==schedule['goal_id'] and task['source_reference_identity']==schedule['identity']
        assert [P['S']['ns']('SUBSTANTIAL',P['SPEC']['seed'],u,i) for i in range(16)]==schedule['seeds']
        return rt.step(u)
    started=time.monotonic();event('TRAINING_STARTED',arm=arm,PID=os.getpid(),committed_update=rt.update)
    status='COMPLETED';exitcode=0
    try:
        commits=sorted((rt.root/'commits').glob('u*.json'));end=int(commits[-1].stem[1:]) if commits else 0;assert len(commits)==end
        if rt.update:callbacks(rt.update)
        for u in range(rt.update+1,end+1):step(u);callbacks(u)
        for u in range(rt.update+1,5001):
            if (root/'control/STOP').exists() or (PAIR/'control/STOP').exists():
                rt.publish('COOPERATIVE_PAUSE');status='COOPERATIVE_STOP';exitcode=21;break
            step(u);callbacks(u)
        if status=='COMPLETED':rt.publish('COMPLETED_DECLARED_BUDGET');atomic(rt.root/'TRAINING_COMPLETED.json',dict(update=rt.update,counters=dict(rt.counters)),True)
    except RegisteredStop:status='REGISTERED_SAFETY_STOP';exitcode=20
    except BaseException as error:
        text=repr(error)
        if 'STEP_KL_STABILITY_STOP' in text or 'GEOMETRY_KL_STABILITY_STOP' in text or 'nonfinite gradient' in text:
            status='REGISTERED_STEP_SAFETY_STOP';exitcode=20
            # Restore and replay only committed safe work, never retry the unsafe update.
            rt.restore();commits=sorted((rt.root/'commits').glob('u*.json'));end=int(commits[-1].stem[1:]) if commits else 0
            for u in range(rt.update+1,end+1):step(u);callbacks(u)
            rt.publish('REGISTERED_STEP_SAFETY_STOP')
            atomic(root/'control/STOP',dict(reason=text,update=rt.update));atomic(rt.root/'STOP_RECORD.json',dict(UTC=now(),arm=arm,update=rt.update,kind=status,reason=text),True)
            event('SAFETY_STOP',arm=arm,update=rt.update,reason=text)
        else:
            atomic(root/'runtime/WORKER_FAILURE.json',dict(UTC=now(),error=text,traceback=traceback.format_exc(),no_automatic_restart=True));raise
    final=read(rt.root/'RECOVERY_INDEX.json')['latest'];assert sha(rt.root/final['relative'])==final['SHA256']
    rollouts=list((rt.root/'rollouts').glob('u*.json'))
    outcome=dict(UTC=now(),arm=arm,status=status,exit_code=exitcode,PID=os.getpid(),updates=rt.update,committed_attempts=rt.counters['attempts'],all_generated_training_attempts=16*len(rollouts),seconds=time.monotonic()-started,checkpoint=final,training_seed=P['SPEC']['seed'],J1_frozen=True)
    atomic(root/'TRAINING_OUTCOME.json',outcome,True);event('TRAINING_EXIT',**outcome)
    raise SystemExit(exitcode)
class RegisteredStop(Exception):pass

def monitor():
    last_report=0.;last_event=None
    while True:
        rows=[]
        for arm,root in ARMS.items():
            live=root/'production/LIVE_STATUS.json';x=read(live) if live.exists() else dict(state='NOT_STARTED',update=0)
            metric=root/f"production/metrics/u{x.get('update',0):06d}.json"
            if metric.exists():x['latest_metrics']=read(metric)
            dev=sorted((root/'production/diagnostics').glob('DEVELOPMENT_*.json'))
            if dev:x['fixed_development']=read(dev[-1])
            rows.append(dict(arm=arm,**x))
        free=shutil.disk_usage(PAIR).free
        atomic(PAIR/'runtime/MONITOR_LATEST.json',dict(UTC=now(),D_free_bytes=free,arms=rows))
        with (PAIR/'runtime/MONITOR.jsonl').open('a') as f:f.write(json.dumps(dict(UTC=now(),D_free_bytes=free,arms=rows))+'\n')
        for row in rows:
            root=ARMS[row['arm']]
            with (root/'runtime/MONITOR.jsonl').open('a') as f:f.write(json.dumps(dict(UTC=now(),D_free_bytes=free,**row))+'\n')
        if free<2*2**30+2*224*2**20+128*2**20:
            atomic(PAIR/'control/STOP',dict(reason='STORAGE_RESERVE'));event('STORAGE_WARNING',D_free_bytes=free)
        if time.monotonic()-last_report>=1800:
            last_report=time.monotonic();event('THIRTY_MINUTE_PROGRESS',arms=[dict(arm=x['arm'],update=x.get('update',0),state=x.get('state')) for x in rows])
        if (PAIR/'runtime/CONTROLLER_EXIT.json').exists():return
        time.sleep(60)

def run_child(mode,arm):
    assert not (PAIR/'control/STOP').exists(),'GLOBAL_COOPERATIVE_STOP'
    root=ARMS[arm];log=root/'runtime'/f'{mode}.log'
    with log.open('x') as f:
        p=subprocess.Popen([str(root/'run.sh'),str(PAIR/'source/paired.py'),mode,arm],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        atomic(root/'runtime'/f'{mode.upper()}_LAUNCH.json',dict(UTC=now(),PID=p.pid,command=mode))
        atomic(PAIR/'runtime/CONTROLLER_STATUS.json',dict(UTC=now(),mode=mode,arm=arm,worker_PID=p.pid))
        while True:
            try:code=p.wait(timeout=60);break
            except subprocess.TimeoutExpired:
                if (PAIR/'runtime/MONITOR_FAILURE.json').exists():atomic(PAIR/'control/STOP',dict(reason='MONITOR_FAILURE'))
    for i in range(30):
        x=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
        if not x.stdout.strip():break
        time.sleep(1)
    no_gpu_worker();return code
def controller():
    lock=(PAIR/'control/CONTROLLER.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify();assert shutil.disk_usage(PAIR).free>=read(PAIR/'PAIRED_PROTOCOL.json')['minimum_free_launch_bytes'];no_gpu_worker()
    monlog=(PAIR/'runtime/monitor.log').open('x');mon=subprocess.Popen([str(PAIR/'run.sh'),str(PAIR/'source/paired.py'),'monitor'],stdout=monlog,stderr=subprocess.STDOUT)
    atomic(PAIR/'runtime/MONITOR_LAUNCH.json',dict(UTC=now(),PID=mon.pid,poll_seconds=60,report_seconds=1800))
    try:
        for arm in ['A','B']:
            code=run_child('qualify',arm)
            if code!=0:
                atomic(PAIR/'PRODUCTION_ADMISSION.json',dict(PASS=False,UTC=now(),blocked_arm=arm,exit_code=code,reason='See arm qualification/ADMISSION.json or runtime/qualify.log',production_updates=0),True)
                raise RuntimeError('PRODUCTION_ADMISSION_FAILED')
        for arm in ['A','B']:assert read(ARMS[arm]/'qualification/ADMISSION.json')['PASS']
        # Baselines differ only by encoding; all numeric criteria remain frozen.
        a=read(ARMS['A']/'qualification/ADMISSION.json');b=read(ARMS['B']/'qualification/ADMISSION.json')
        atomic(PAIR/'BASELINE_COMPARISON.json',dict(A=a['baseline'],B=b['baseline'],weights_identical=True,parameters_identical=True,token_can_change_zero_update_predictions=True),True)
        atomic(PAIR/'PRODUCTION_ADMISSION.json',dict(PASS=True,UTC=now(),arms={'A':a,'B':b},D_free_bytes=shutil.disk_usage(PAIR).free,all_before_either_arm_training=True),True)
        for arm,root in ARMS.items():
            atomic(root/'PRODUCTION_AUTHORIZATION.json',dict(explicit_review_authorized=True,authorization='User authorized matched substantial RL after final admission',configuration_SHA256=sha(root/'TRAINING_CONFIGURATION_AND_SEEDS.json'),frozen_inputs_SHA256=sha(PAIR/'FROZEN_SOURCE_AND_INPUTS.json')),True)
            # Copy only immutable zero-update evaluation data, never engineering weights.
            (root/'production/diagnostics').mkdir(parents=True,exist_ok=True)
            for p in (root/'qualification/engineering/diagnostics').glob('*.json'):shutil.copyfile(p,root/'production/diagnostics'/p.name)
        for arm in ['A','B']:
            code=run_child('train',arm)
            if code not in [0,20]:raise RuntimeError(f'ARM_{arm}_INFRASTRUCTURE_OR_COOPERATIVE_STOP_{code}')
            event('GPU_RELEASE_VERIFIED',arm=arm)
        # Final evaluation is the same frozen panel for both arms, even after a
        # safety stop; unequal exposures are explicitly partial comparisons.
        with (PAIR/'runtime/evaluation.log').open('x') as f:
            code=subprocess.run([str(PAIR/'run.sh'),str(PAIR/'source/evaluate.py')],stdout=f,stderr=subprocess.STDOUT).returncode
        assert code==0,'FINAL_EVALUATION_FAILED';no_gpu_worker()
        with (PAIR/'runtime/analysis.log').open('x') as f:
            code=subprocess.run([str(PAIR/'run.sh'),str(PAIR/'source/analyze.py')],stdout=f,stderr=subprocess.STDOUT).returncode
        assert code==0,'FINAL_ANALYSIS_FAILED'
        atomic(PAIR/'runtime/CONTROLLER_EXIT.json',dict(UTC=now(),exit_code=0,status='COMPLETED'),True)
    except BaseException as error:
        atomic(PAIR/'runtime/CONTROLLER_FAILURE.json',dict(UTC=now(),error=repr(error),traceback=traceback.format_exc(),automatic_retry=False))
        atomic(PAIR/'runtime/CONTROLLER_EXIT.json',dict(UTC=now(),exit_code=1,status='STOPPED',error=repr(error)),True)
        admissions={a:read(root/'qualification/ADMISSION.json') if (root/'qualification/ADMISSION.json').exists() else {'status':'NOT_FINISHED'} for a,root in ARMS.items()}
        (PAIR/'FINAL_SCIENTIFIC_RECOMMENDATION.md').write_text('# Paired study stopped before complete assessment\n\n'+repr(error)+'\n\n'+json.dumps(admissions,indent=2)+'\n\nNo safeguard was relaxed. Scientific effects are not established; see durable worker logs and controller failure record. No automatic retry.\n')
        raise
    finally:mon.wait(timeout=70);monlog.close()

if __name__=='__main__':
    mode=sys.argv[1];arm=sys.argv[2] if len(sys.argv)>2 else None
    try:{'qualify':lambda:qualify(arm),'partial':lambda:partial(arm),'train':lambda:training(arm),'controller':controller,'monitor':monitor}[mode]()
    except BaseException as error:
        if mode=='monitor':
            atomic(PAIR/'runtime/MONITOR_FAILURE.json',dict(UTC=now(),error=repr(error),traceback=traceback.format_exc()))
            atomic(PAIR/'control/STOP',dict(reason='MONITOR_FAILURE'))
        raise
