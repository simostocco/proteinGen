from pathlib import Path
import ast,json,shutil,subprocess,time
R=Path('/mnt/d/Simone/proteinGen_data/e017_v72_counterfactual_ranking_20261023_v1')
assert not (R/'FINAL_VERIFICATION.json').exists()
for name in ['v72_final_audit.py','v72_finish.py']:
 p=Path(__file__).with_name(name);ast.parse(p.read_text());shutil.copyfile(p,R/'control'/name)
log=R/f'logs/final_review_controller_{time.time_ns()}.log'
with log.open('x') as f:p=subprocess.Popen([str(R/'run.sh'),str(R/'control/v72_finish.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,log=str(log),scope='wait for sealed study then CPU-only descriptive audit and integrity publication; no fitting or inference')))
