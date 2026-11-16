from pathlib import Path
import json,shutil,subprocess
base=Path('/mnt/d/Simone/proteinGen_data');old=base/'e017_v92_prefix_feasibility_20261024_v1';r=base/'e017_v93_regional_controllability_20261024_v1';assert not r.exists();free=shutil.disk_usage(base).free;assert free>2*2**30+192*2**20;r.mkdir()
for name in ['manifests','results','logs','reports','plots','control','runtime/tmp','runtime/cache']:(r/name).mkdir(parents=True)
subprocess.run(['git','-C',str(old/'source'),'worktree','add','-b','e017-v93-regional-controllability-20261024-v1',str(r/'source'),'95b27b0834fb36b0373bc89e7fdaec6773a85fb3'],check=True)
(r/'source/scripts/e017_v93').mkdir();(r/'run.sh').write_text((old/'run.sh').read_text().replace(str(old),str(r)));(r/'run.sh').chmod(0o755)
(r/'INITIAL_ADMISSION.json').write_text(json.dumps(dict(D_free_before_bytes=free,reserve_bytes=2*2**30,evidence_budget_bytes=64*2**20,no_protein_optimizer_updates=True,source_base='95b27b0834fb36b0373bc89e7fdaec6773a85fb3'),indent=2)+'\n')
print(str(r),free)
