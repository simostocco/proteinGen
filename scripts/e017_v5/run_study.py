"""Sealed sequential V5 campaign. No SAC. All escalation gates are prospective."""
import pathlib,json,sys,subprocess,os,time,hashlib,shutil
R=pathlib.Path('/mnt/d/Simone/proteinGen_data/e017_v5_goal_observability_20261022_v1');P=R/'source/scripts/e017_v5';sys.path.insert(0,str(P));import study as s

def run(script,*args):
 name='_'.join([pathlib.Path(script).stem,*map(str,args)]).replace('/','_');log=R/'logs'/f'{name}.log'
 with log.open('a') as f:
  subprocess.run([sys.executable,'-B',str(script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT,check=True,env=os.environ.copy())

def main():
 seal=json.load(open(R/'SEALED_CONFIG.json'));assert s.confighash()==seal['config_hash'];s.save('LIVE_STATUS.json',dict(phase='BASELINE',UTC=time.time()));observer=subprocess.Popen([sys.executable,'-B',str(P/'observer.py'),str(os.getpid())]);run(P/'study.py','baseline');active=['B','C','D','E'];last={a:0 for a in active};gates={}
 for H in [2,4,8]:
  if not active:break
  for arm in list(active):
   s.save('LIVE_STATUS.json',dict(phase='SUPERVISED',arm=arm,H=H,UTC=time.time()));run(P/'study.py','train','--arm',arm,'--H',H);report=json.load(open(R/f'results/{arm}_H{H}_FINISHED.json'));last[arm]=report['update'];gates[f'{arm}_H{H}']=report['gate']
  following=[a for a in active if gates[f'{a}_H{H}']['pass_gate']]
  # The primary C/D routing comparison must remain matched in update budget.
  if not {'C','D'}<=set(following):following=[a for a in following if a not in ['C','D']]
  s.save('results/CURRICULUM_GATES.json',dict(completed_horizon=H,gates=gates,next_arms=following,latest=last));active=following
 s.save('LIVE_STATUS.json',dict(phase='FINAL_EVALUATION',UTC=time.time(),latest=last));run(P/'final_evaluate.py');s.save('LIVE_STATUS.json',dict(phase='STUDY_COMPLETE',UTC=time.time(),latest=last,no_RL=True));print('SEALED_STUDY_COMPLETE',last,flush=True)

if __name__=='__main__':
 try:main()
 except BaseException as er:
  s.save('LIVE_STATUS.json',dict(phase='FAILED',error=repr(er),UTC=time.time()));raise
