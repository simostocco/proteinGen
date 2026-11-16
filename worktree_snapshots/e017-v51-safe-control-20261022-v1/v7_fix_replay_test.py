"""Correct only reference precision after the documented engineering stop."""
from pathlib import Path
import json
import shutil
import subprocess

R=Path('/mnt/d/Simone/proteinGen_data/e017_v7_amortized_lowlevel_20261023_v1')
assert json.loads((R/'REPLAY_PRECISION_DIAGNOSIS.json').read_text())['PASS']
assert not list((R/'results').glob('B*_updates.jsonl')) and not list((R/'checkpoints').glob('B*'))
p=R/'source/scripts/e017_v7/study.py';old=p.read_text();before="torch.tensor(ep['tail'])";after="torch.tensor(ep['tail'],dtype=torch.float64)"
assert old.count(before)==1
archive=R/'protocol_engineering_fix/v1';archive.mkdir(parents=True)
shutil.copyfile(p,archive/'study_before_FP64_test_fix.py')
shutil.move(R/'SEALED_CONFIG.json',archive/'SEALED_CONFIG.json');shutil.move(R/'CONFIG.json',archive/'CONFIG.json')
p.write_text(old.replace(before,after))
subprocess.run(['git','-C',str(R/'source'),'add','scripts/e017_v7/study.py'],check=True)
subprocess.run(['git','-C',str(R/'source'),'commit','-m','Fix FP64 replay-test reference without changing tolerance or scientific protocol'],check=True)
code='import runpy;runpy.run_path('+repr(str(p))+')["seal"]()'
subprocess.run([str(R/'run.sh'),'-c',code],check=True)
print('Only engineering reference dtype changed; original1e-8tolerance, all data and science unchanged',flush=True)
