from pathlib import Path
import shutil
import subprocess
import sys
R=Path('/mnt/d/Simone/proteinGen_data/e017_v71_expected_return_calibration_20261023_v1');BASE=Path(__file__).resolve().parent
assert not (R/'PROTOCOL.json').exists()
p=R/'source/scripts/e017_v71/study.py';p.parent.mkdir(exist_ok=True);shutil.copyfile(BASE/'v71_study_staged.py',p)
if sys.argv[1]=='seal':
 subprocess.run(['git','-C',str(R/'source'),'add','scripts/e017_v71/study.py'],check=True);subprocess.run(['git','-C',str(R/'source'),'commit','-m','Seal V7.1 stage-stratified expected-return and fixed-policy RWR study'],check=True)
code='import runpy;runpy.run_path('+repr(str(p))+')['+repr(sys.argv[1])+']()'
subprocess.run([str(R/'run.sh'),'-c',code],check=True)
