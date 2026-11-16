from pathlib import Path
import subprocess
import time
R=Path('/mnt/d/Simone/proteinGen_data/e017_v7_amortized_lowlevel_20261023_v1')
assert not (R/'V7_STUDY_COMPLETE.json').exists()
with (R/f'logs/final_pipeline_{time.time_ns()}.log').open('x') as f:p=subprocess.Popen(['python3','-B',str(Path(__file__).with_name('v7_finish_pipeline.py'))],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(p.pid)
