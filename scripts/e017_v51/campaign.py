"""Sequential bounded V5.1; scientific stops are arm-local, never auto-resumed."""
import pathlib,subprocess,sys,os,json,time,signal,shutil
R=pathlib.Path('/mnt/d/Simone/proteinGen_data/e017_v51_safe_control_20261022_v1');P=R/'source/scripts/e017_v51'
def call(command,*args):
 name='_'.join([command,*args]);log=R/'logs'/f'campaign_{name}.log'
 with log.open('x') as f:pr=subprocess.run([sys.executable,'-B',str(P/'study.py'),command,*args],stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
 print(name,pr.returncode,flush=True)
 if pr.returncode:raise RuntimeError(f'{name}: see {log}')
def main():
 obs=subprocess.Popen([sys.executable,'-B',str(P/'observer.py'),str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 try:
  call('baseline')
  for arm in ['A','B','C','D']:call('train','--arm',arm)
  call('final')
  (R/'CAMPAIGN_COMPLETED.json').write_text(json.dumps(dict(complete=True,UTC=time.time(),no_RL=True)))
 finally:
  obs.terminate();obs.wait(timeout=15)
 print('V51_BOUNDED_CAMPAIGN_COMPLETE',flush=True)
if __name__=='__main__':main()
