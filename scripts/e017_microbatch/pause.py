"""Interrupt only a verified post-publication checkpoint readback, never an optimizer update.
If an unambiguous safe point cannot be proved, resume immediately and decline.
"""
from pathlib import Path
import hashlib,json,os,signal,time,subprocess,sys
R=Path('/mnt/d/Simone/proteinGen_data/e017_crisp_v3_microbatch_20261021_v1')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def identity(pid):
 p=Path(f'/proc/{pid}');v=(p/'stat').read_text().rsplit(') ',1)[1].split();return dict(start_ticks=v[19],state=v[0],command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
def descriptors(pid):
 out={}
 for p in Path(f'/proc/{pid}/fd').iterdir():
  try:out[p.name]=os.readlink(p)
  except FileNotFoundError:pass
 return out
def stop_at_publication(pid,run,target,expected_command,fixture=False,timeout=3600):
 ident=identity(pid)
 if expected_command not in ident['command']:raise ValueError('wrong process identity')
 start=time.monotonic();index=run/'RECOVERY_INDEX.json';attempts=0
 while time.monotonic()-start<timeout:
  try:
   idx=json.load(open(index));bundle=idx['latest']
   if not bundle or bundle['update']<target:time.sleep(.05);continue
   if bundle['update']>target:return dict(status='DECLINED_TARGET_PASSED',update=bundle['update'])
   if not fixture:
    score=json.load(open(run/f'validation_{target:05d}.json'))
    if score['stop']['stop']:return dict(status='SCIENTIFIC_STOP_LEAVE_UNCHANGED',decision=score['stop'])
   checkpoint=bundle['checkpoint'];fds=descriptors(pid);matching={fd:path for fd,path in fds.items() if path==checkpoint}
   if not matching:time.sleep(.001);continue
   if identity(pid)['start_ticks']!=ident['start_ticks']:raise ValueError('PID reused')
   os.kill(pid,signal.SIGSTOP);attempts+=1
   for _ in range(1000):
    if identity(pid)['state'] in ['T','t']:break
    time.sleep(.001)
   after=descriptors(pid);same=json.load(open(index))['latest'];safe=identity(pid)['state'] in ['T','t'] and same['sha256']==bundle['sha256'] and any(path==checkpoint for path in after.values())
   if not safe:os.kill(pid,signal.SIGCONT);time.sleep(.001);continue
   # In the audited sealed source, an open NEW checkpoint after index publication
   # can occur only in post-publication collect_superseded/_bundle_valid. Before
   # loop re-entry all such descriptors close. Recheck while stopped, then verify
   # the entire recovery before delivering the cooperative Python interrupt.
   try:
    if sha(checkpoint)!=bundle['sha256']:raise ValueError('checkpoint SHA')
    import torch
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if not fixture:
     assert state['schema']=='crisp_v3_full_v1' and state['counters']['updates']==target
     assert state['steps']=={'actor':target} and set(state['optimizers'])=={'actor'}
     assert set(state['samplers'])=={'AA','geometry','goals'}
     assert all(k in state for k in ['system','schedulers','rng','config','replay_snapshot'])
     assert state['config']['source_commit']=='b62978364dc054850f6b37e888baaa9d1638d4f6'
     assert sha(state['replay_snapshot']['path'])==state['replay_snapshot']['sha256']
     assert state['j1_sha256']=='881457ff0396975c1b20dd0a8e2caac8521e65b7e1cec76a7bb531dbc437b412'
     lines=(run/'updates.jsonl').read_text().splitlines();assert json.loads(lines[-1])['update']==target
     # Stop only the old read-only observer; it has no scientific state and would
     # otherwise mislabel this intentional operator pause as an unexpected exit.
     observer=183419
     try:
      oi=identity(observer)
      if 'CAMPAIGN_OBSERVER.py' in oi['command']:
       for _ in range(100):
        w=Path(f'/proc/{observer}/wchan').read_text().strip()
        if 'sleep' in w:os.kill(observer,signal.SIGINT);break
        time.sleep(.05)
     except FileNotFoundError:pass
    record=dict(status='VERIFIED_POST_PUBLICATION_PAUSE',pid=pid,process_identity=ident,target_update=target,checkpoint=checkpoint,checkpoint_sha256=bundle['sha256'],bundle=bundle,stopped_open_checkpoint_descriptors={fd:p for fd,p in after.items() if p==checkpoint},attempts=attempts,all_recovery_fields_verified=True,signal='SIGINT delivered only after verified SIGSTOP at readback; SIGCONT permits Python finally cleanup',fixture=fixture,UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    receipt=R/'results'/('PAUSE_FIXTURE_RESULT.json' if fixture else 'VERIFIED_OPERATOR_PAUSE.json')
    with open(receipt,'x') as f:json.dump(record,f,indent=2);f.flush();os.fsync(f.fileno())
    os.kill(pid,signal.SIGINT);os.kill(pid,signal.SIGCONT)
    for _ in range(600):
     try:
      if identity(pid)['state']=='Z':break
     except FileNotFoundError:break
     time.sleep(.1)
    else:raise RuntimeError('graceful interrupt did not exit; do not force kill')
    record['process_exited']=True
    with open(receipt.with_name(receipt.stem+'_EXIT.json'),'x') as f:json.dump(record,f,indent=2)
    return record
   except BaseException:
    try:os.kill(pid,signal.SIGCONT)
    except ProcessLookupError:pass
    raise
  except (FileNotFoundError,ProcessLookupError):
   try:identity(pid)
   except FileNotFoundError:return dict(status='PROCESS_ALREADY_EXITED')
   time.sleep(.005)
 return dict(status='DECLINED_TIMEOUT_NO_UNAMBIGUOUS_SAFEPOINT')
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--fixture',action='store_true');a.add_argument('--pid',type=int);a.add_argument('--run');a.add_argument('--target',type=int,default=250);args=a.parse_args()
 if args.fixture:
  d=R/'benchmarks/pause_fixture';d.mkdir(exist_ok=False);log=open(d/'stdout.log','w');p=subprocess.Popen([sys.executable,str(Path(__file__).with_name('pause_fixture.py')),str(d/'run')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  result=stop_at_publication(p.pid,d/'run',1,'pause_fixture.py',True,timeout=60);log.close();assert result['status']=='VERIFIED_POST_PUBLICATION_PAUSE';assert (d/'run/FINALLY_CLEANUP.json').exists();assert not (d/'run/NEXT_OPTIMIZER_UPDATE_STARTED').exists();print(json.dumps(result,indent=2))
 else:
  assert json.load(open(R/'results/PAUSE_FIXTURE_RESULT_EXIT.json'))['process_exited'];print(json.dumps(stop_at_publication(args.pid,Path(args.run),args.target,'scripts/e017_v3/train.py'),indent=2))
