from pathlib import Path
import subprocess
import time
R=Path('/mnt/d/Simone/proteinGen_data/e017_v71_expected_return_calibration_20261023_v1')
with (R/f'logs/watch_{time.time_ns()}.log').open('x') as f:p=subprocess.Popen(['python3','-B',str(Path(__file__).with_name('v71_watch.py'))],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid)
