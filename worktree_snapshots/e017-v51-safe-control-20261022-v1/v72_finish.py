from pathlib import Path
import fcntl,json,os,subprocess,time
R=Path('/mnt/d/Simone/proteinGen_data/e017_v72_counterfactual_ranking_20261023_v1')
def save(name,x):
 p=R/name;q=p.with_name(p.name+'.part')
 with q.open('w') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def main():
 lock=(R/'runtime/final_review.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (R/'FINAL_VERIFICATION.json').exists();deadline=time.monotonic()+1800
 while not (R/'CAMPAIGN_COMPLETED.json').exists():
  state=json.loads((R/'EXECUTION.json').read_text());assert state['status']=='RUNNING',state;assert time.monotonic()<deadline,'final review wait deadline';time.sleep(10)
 time.sleep(25)
 with (R/f'logs/final_review_{time.time_ns()}.log').open('x') as f:
  p=subprocess.Popen([str(R/'run.sh'),str(R/'control/v72_final_audit.py')],stdout=f,stderr=subprocess.STDOUT);save('FINALIZATION_STATUS.json',dict(status='RUNNING',pid=p.pid,supervisor_pid=os.getpid(),automatic_restart=False));rc=p.wait()
 assert rc==0,rc;save('FINALIZATION_STATUS.json',dict(status='COMPLETED',automatic_restart=False,UTC=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException as ex:save('FINALIZATION_STATUS.json',dict(status='FAILED',error=repr(ex),automatic_restart=False));raise
