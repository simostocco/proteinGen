from pathlib import Path
import ast,json,shutil,subprocess,time
R=Path('/mnt/d/Simone/proteinGen_data/e017_v8_constrained_planning_20261023_v1')
assert not (R/'FINAL_VERIFICATION.json').exists()
for name in ['v8_final_audit.py','v8_finish.py']:
 p=Path(__file__).with_name(name);ast.parse(p.read_text());shutil.copyfile(p,R/'control'/name)
log=R/f'logs/final_review_controller_{time.time_ns()}.log'
with log.open('x') as f:p=subprocess.Popen([str(R/'run.sh'),str(R/'control/v8_finish.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,log=str(log),scope='wait for bounded V8 then CPU-only final verification; no research launch or model evaluation')))
