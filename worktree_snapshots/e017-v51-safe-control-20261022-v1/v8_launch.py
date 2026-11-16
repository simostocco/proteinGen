from pathlib import Path
import json,os,shutil,subprocess,time
R=Path('/mnt/d/Simone/proteinGen_data/e017_v8_constrained_planning_20261023_v1')
assert json.loads((R/'PREFLIGHT.json').read_text())['PASS'];assert not (R/'EXECUTION.json').exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
shutil.copyfile(Path(__file__).with_name('v8_watch.py'),R/'control/v8_watch.py');procs={}
for label,args in [('campaign',[str(R/'source/scripts/e017_v8/study.py'),'campaign']),('resource_watch',[str(R/'control/v8_watch.py')])]:
 log=R/f'logs/{label}_{time.time_ns()}.log'
 with log.open('x') as f:p=subprocess.Popen([str(R/'run.sh'),*args],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 procs[label]=dict(pid=p.pid,log=str(log))
print(json.dumps(procs))
