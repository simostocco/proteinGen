"""Install V7 source under its own worktree and run the requested stage."""
from pathlib import Path
import subprocess
import shutil
import sys

BASE=Path('/mnt/d/Simone/proteinGen_data/e017_v51_safe_control_20261022_v1/source')
R=BASE.parent.parent/'e017_v7_amortized_lowlevel_20261023_v1';p=R/'source/scripts/e017_v7/study.py';p.parent.mkdir(exist_ok=True)
if sys.argv[1]=='precomparison_reseal':
 assert not (R/'PARENT_ARCHITECTURE_AUDIT.json').exists() and not (R/'results/TRAIN_candidates.jsonl').exists()
 archive=R/'protocol_precomparison/v1';archive.mkdir(parents=True)
 shutil.move(R/'SEALED_CONFIG.json',archive/'SEALED_CONFIG.json');shutil.move(R/'CONFIG.json',archive/'CONFIG.json')
 sys.argv[1]='seal'
assert not (R/'SEALED_CONFIG.json').exists()
shutil.copyfile(BASE/'v7_study_staged.py',p)
if sys.argv[1]=='seal':
 subprocess.run(['git','-C',str(R/'source'),'add','scripts/e017_v7/study.py'],check=True)
 subprocess.run(['git','-C',str(R/'source'),'commit','-m','Seal bounded fixed-AA V7 RWR and gated low-level SAC pilot'],check=True)
code='import runpy; scope=runpy.run_path('+repr(str(p))+'); scope['+repr(sys.argv[1])+']()'
subprocess.run([str(R/'run.sh'),'-c',code],check=True)
