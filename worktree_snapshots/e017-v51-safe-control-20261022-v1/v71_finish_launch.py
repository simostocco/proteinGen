from pathlib import Path
import ast
import json
import shutil
import subprocess
import time
R=Path('/mnt/d/Simone/proteinGen_data/e017_v71_expected_return_calibration_20261023_v1')
assert not (R/'PUBLICATION_COMPLETED.json').exists()
for name in ['v71_finish.py','v71_analyze.py']:
 p=Path(__file__).with_name(name);ast.parse(p.read_text());shutil.copyfile(p,R/'analysis_source'/name)
log=R/f'logs/finalization_{time.time_ns()}.log'
with log.open('x') as f:p=subprocess.Popen([str(R/'run.sh'),str(R/'analysis_source/v71_finish.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,log=str(log),scope='wait for active campaign; read-only final assessments; CPU publication; no training')))
