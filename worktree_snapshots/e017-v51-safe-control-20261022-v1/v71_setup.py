"""Create V7.1 source/evidence without modifying V7 artifacts."""
from pathlib import Path
import subprocess
R=Path('/mnt/d/Simone/proteinGen_data/e017_v71_expected_return_calibration_20261023_v1')
BASE=R.parent/'e017_v7_amortized_lowlevel_20261023_v1'
assert not R.exists();R.mkdir()
for n in ['results','manifests','checkpoints','logs','plots','runtime/tmp','runtime/cache']:(R/n).mkdir(parents=True)
subprocess.run(['git','-C',str(BASE/'source'),'worktree','add','-b','e017-v71-expected-return-20261023-v1',str(R/'source'),'e6ad343c10b5836415df18c4bf098ab5c38dfead'],check=True)
(R/'run.sh').write_text((BASE/'run.sh').read_text().replace(str(BASE),str(R)));(R/'run.sh').chmod(0o755)
print(R)
