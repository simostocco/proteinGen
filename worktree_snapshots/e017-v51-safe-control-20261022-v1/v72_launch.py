from pathlib import Path
import json,os,shutil,subprocess,time
R=Path('/mnt/d/Simone/proteinGen_data/e017_v72_counterfactual_ranking_20261023_v1')
assert json.loads((R/'PREFLIGHT.json').read_text())['PASS'];assert not (R/'EXECUTION.json').exists()
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 if 'scripts/e017_v72/study.py' in cmd and (p/'comm').read_text().startswith('python'):raise RuntimeError('existingV72process '+p.name)
control=R/'control';control.mkdir(exist_ok=True);shutil.copyfile(Path(__file__).with_name('v72_watch.py'),control/'v72_watch.py')
procs={}
for label,args in [('campaign',[str(R/'source/scripts/e017_v72/study.py'),'campaign']),('resource_watch',[str(control/'v72_watch.py')])]:
 log=R/f'logs/{label}_{time.time_ns()}.log'
 with log.open('x') as f:p=subprocess.Popen([str(R/'run.sh'),*args],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 procs[label]=dict(pid=p.pid,log=str(log))
print(json.dumps(procs))
