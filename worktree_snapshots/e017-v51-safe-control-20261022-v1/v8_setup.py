from pathlib import Path
import subprocess
R=Path('/mnt/d/Simone/proteinGen_data/e017_v8_constrained_planning_20261023_v1');B=R.parent/'e017_v72_counterfactual_ranking_20261023_v1'
assert not R.exists();R.mkdir()
for n in ['manifests','results','logs','plots','control','runtime/tmp','runtime/cache']:(R/n).mkdir(parents=True)
subprocess.run(['git','-C',str(B/'source'),'worktree','add','-b','e017-v8-constrained-planning-20261023-v1',str(R/'source'),'5e9c0f54e539c992220a8e2cdc292bf7aa5bce32'],check=True)
(R/'run.sh').write_text((B/'run.sh').read_text().replace(str(B),str(R)));(R/'run.sh').chmod(0o755);print(R)
