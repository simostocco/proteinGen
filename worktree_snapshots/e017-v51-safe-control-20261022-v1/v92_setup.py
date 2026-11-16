from pathlib import Path
import hashlib,json,shutil,subprocess
OLD=Path('/mnt/d/Simone/proteinGen_data/e017_v91_ordered_goal_recovery_20261023_v1');R=OLD.parent/'e017_v92_prefix_feasibility_20261024_v1';assert not R.exists();R.mkdir()
for name in ['manifests','results','reports','logs','plots','control','runtime/tmp','runtime/cache']:(R/name).mkdir(parents=True)
free=shutil.disk_usage(R).free;required=2*2**30+128*2**20+128*2**20;assert free>required
subprocess.run(['git','-C',str(OLD/'source'),'worktree','add','-b','e017-v92-prefix-feasibility-20261024-v1',str(R/'source'),'82d23e47d1d33a0a6b2d2ff542aef3e8d86d0bfe'],check=True)
(R/'run.sh').write_text((OLD/'run.sh').read_text().replace(str(OLD),str(R)));(R/'run.sh').chmod(0o755);(R/'source/scripts/e017_v92').mkdir()
record=dict(schema='E017_V92_DIAGNOSTIC_1',root=str(R),source_base='82d23e47d1d33a0a6b2d2ff542aef3e8d86d0bfe',D_free_bytes_before_worktree=free,admission_required_bytes=required,operational_reserve_bytes=2*2**30,near_growth_bytes=128*2**20,evidence_growth_cap_bytes=128*2**20,no_neural_training=True,no_new_model_checkpoints=True,no_historical_deletions=True,TEST=False,AlphaFold=False)
(R/'INITIAL_RESOURCE_ADMISSION.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
