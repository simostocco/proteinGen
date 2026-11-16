from pathlib import Path
import subprocess
R=Path('/mnt/d/Simone/proteinGen_data/e017_v9_property_steering_20261023_v1');B=R.parent/'e017_v8_constrained_planning_20261023_v1'
assert not R.exists();R.mkdir()
for n in ['manifests','results','reports','checkpoints','logs','plots','control','runtime/tmp','runtime/cache']:(R/n).mkdir(parents=True)
subprocess.run(['git','-C',str(B/'source'),'worktree','add','-b','e017-v9-property-steering-20261023-v1',str(R/'source'),'d3eb8b7d0ef361f7224c495abc1b71860b1f1bac'],check=True)
(R/'run.sh').write_text((B/'run.sh').read_text().replace(str(B),str(R)));(R/'run.sh').chmod(0o755);print(R)
