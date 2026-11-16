from pathlib import Path
import subprocess
R=Path('/mnt/d/Simone/proteinGen_data/e017_v72_counterfactual_ranking_20261023_v1')
B=R.parent/'e017_v71_expected_return_calibration_20261023_v1'
assert not R.exists();R.mkdir()
for name in ['manifests','results','checkpoints','logs','plots','runtime/tmp','runtime/cache']:(R/name).mkdir(parents=True)
subprocess.run(['git','-C',str(B/'source'),'worktree','add','-b','e017-v72-counterfactual-ranking-20261023-v1',str(R/'source'),'1f9a91b226460f163e13c501b9d2e1c524d84a42'],check=True)
(R/'run.sh').write_text((B/'run.sh').read_text().replace(str(B),str(R)));(R/'run.sh').chmod(0o755)
print(R)
