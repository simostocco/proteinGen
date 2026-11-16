"""Create the authorized isolated V7 source/evidence root, without training."""
from pathlib import Path
import subprocess

BASE=Path('/mnt/d/Simone/proteinGen_data/e017_v6_frozen_parent_candidates_20261023_v1/source')
R=BASE.parent.parent/'e017_v7_amortized_lowlevel_20261023_v1'
assert not R.exists()
R.mkdir()
for d in ['manifests','results','checkpoints','logs','runtime/tmp','runtime/cache','plots']:(R/d).mkdir(parents=True)
subprocess.run(['git','-C',str(BASE),'worktree','add','-b','e017-v7-amortized-lowlevel-20261023-v1',str(R/'source'),'04ea6394381203f88dfc180972ccf512ee18f3e5'],check=True)
wrapper=(BASE.parent/'run.sh').read_text().replace(str(BASE.parent),str(R))
(R/'run.sh').write_text(wrapper);(R/'run.sh').chmod(0o755)
print(R,flush=True)
