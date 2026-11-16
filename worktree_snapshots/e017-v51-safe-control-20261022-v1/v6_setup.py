"""Create only the newly authorized isolated V6 worktree and evidence directory."""
from pathlib import Path
import os
import shutil
import subprocess

BASE=Path('/mnt/d/Simone/proteinGen_data/e017_v51_safe_control_20261022_v1/source')
R=BASE.parent.parent/'e017_v6_frozen_parent_candidates_20261023_v1'
assert not R.exists(), 'Do not overwrite an earlier V6'
R.mkdir()
for d in ['results','logs','runtime/tmp','runtime/cache']:(R/d).mkdir(parents=True)
subprocess.run(['git','-C',str(BASE),'worktree','add','-b','e017-v6-frozen-candidates-20261023-v1',str(R/'source'),'732134d9aa2b3b30102d3a9e009b4665127f9506'],check=True)
p=R/'source/scripts/e017_v6';p.mkdir()
shutil.copyfile(BASE/'v6_study_staged.py',p/'study.py')
subprocess.run(['git','-C',str(R/'source'),'add','scripts/e017_v6/study.py'],check=True)
subprocess.run(['git','-C',str(R/'source'),'commit','-m','E017 V6 frozen-parent nested finite-candidate diagnostic'],check=True)
wrapper=(BASE.parent/'run.sh').read_text().replace(str(BASE.parent),str(R)).replace('OMP_NUM_THREADS=2 MKL_NUM_THREADS=2','OMP_NUM_THREADS=1 MKL_NUM_THREADS=1')
(R/'run.sh').write_text(wrapper);(R/'run.sh').chmod(0o755)
print(R,flush=True)
subprocess.run([str(R/'run.sh'),str(p/'study.py'),'initialize'],check=True)
