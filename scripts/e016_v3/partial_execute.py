"""Serial authorized continuations; explicit preflight, crash recovery and final evidence seal."""
import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
R=Path('/mnt/d/Simone/proteinGen_data/e016_partial_supervision_20261016_v1')
ARMS=['J1-FULL','J1-PARTIAL','J2-FULL','J2-PARTIAL']
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,value):
 q=p.with_name(p.name+'.part')
 with q.open('w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
 q.replace(p)
def event(value):
 with (R/'EXECUTION_STAGES.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),**value})+'\n');f.flush();os.fsync(f.fileno())
def stage(script,args):
 label=script.replace('.py','')+'_'+('_'.join(args) or 'assessment');log=R/'logs'/(label+'_'+str(time.time_ns())+'.log');start=time.monotonic();command=[sys.executable,'-B','-u',str(R/'source/scripts/e016_v3'/script),*args];event({'stage':label,'status':'START','command':command,'log':str(log)});write(R/'ACTIVE_CONTROLLER.json',{'pid':os.getpid(),'phase':label,'started':time.time(),'log':str(log)})
 print('START '+label,flush=True)
 with log.open('xb') as f:result=subprocess.run(command,cwd=R/'source',stdout=f,stderr=subprocess.STDOUT)
 event({'stage':label,'status':'END','returncode':result.returncode,'seconds':time.monotonic()-start})
 if result.returncode:raise RuntimeError('stage failed; preserved log: '+str(log))
def main():
 lock=(R/'CONTROLLER.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if not (R/'PREFLIGHT_PASS.json').exists():raise RuntimeError('real mixed-data preflight must pass before training launch')
 for arm in ARMS:
  stage('partial_train.py',['--arm',arm])
  if not (R/'runs'/arm/'COMPLETED.json').exists():raise RuntimeError('no verified completion receipt: '+arm)
 stage('partial_finalize.py',[])
 # Finalizer has stopped resource telemetry. Seal controller termination as well.
 import torch
 checkpoints={}
 for p in sorted((R/'runs').glob('*/checkpoint_*.pt')):
  h=sha(p)
  if h!=p.with_name(p.name+'.sha256').read_text().strip():raise ValueError('durable checkpoint checksum')
  state=torch.load(p,map_location='cpu',weights_only=True);u=state['additional_update'];arm=p.parent.name;optimizer_steps=sorted({int(v['step']) for v in state['optimizer']['state'].values()})
  if u!=int(p.stem.split('_')[-1]) or state['sampler']['cursor']!=64*u or state['scheduler']['last_epoch']!=u or max(optimizer_steps)!=10000+u or state['lineage']['arm']!=arm:raise ValueError('durable checkpoint counter consistency')
  if not all(torch.isfinite(v).all() for v in state['model'].values() if torch.is_tensor(v)):raise ValueError('nonfinite durable model')
  checkpoints[str(p.relative_to(R))]={'sha256':h,'additional_update':u,'optimizer_steps':optimizer_steps,'sampler_cursor':state['sampler']['cursor'],'scheduler_epoch':state['scheduler']['last_epoch']};del state
 write(R/'FINAL_CHECKPOINT_CONTENT_VERIFICATION.json',checkpoints);write(R/'ACTIVE_CONTROLLER.json',{'pid':os.getpid(),'phase':'COMPLETED','time':time.time()});event({'stage':'all','status':'COMPLETED'})
 old=R/'PROVENANCE_SHA256.json';earlier=json.loads(old.read_text());snapshot=R/'FINALIZER_PROVENANCE_SHA256.json'
 if snapshot.exists():raise RuntimeError('final evidence publication already exists; preserve prior version')
 old.replace(snapshot)
 artifacts={str(p.relative_to(R)):sha(p) for p in R.rglob('*') if p.is_file() and not any(k in p.parts for k in ['.git','runtime']) and p.name!='PROVENANCE_SHA256.json'}
 write(R/'PROVENANCE_SHA256.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip(),'source_status':subprocess.check_output(['git','status','--short'],cwd=R/'source',text=True),'artifact_sha256':artifacts,'selected_checkpoints':earlier['checkpoints'],'durable_checkpoint_contents':checkpoints})
 print('ALL_AUTHORIZED_STAGES_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as ex:
  write(R/('CONTROLLER_BLOCKED_'+str(time.time_ns())+'.json'),{'error':repr(ex),'time':time.time()});raise
