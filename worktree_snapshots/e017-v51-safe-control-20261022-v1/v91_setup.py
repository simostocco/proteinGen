from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
OLD=Path('/mnt/d/Simone/proteinGen_data/e017_v9_property_steering_20261023_v1');R=OLD.parent/'e017_v91_ordered_goal_recovery_20261023_v1';assert not R.exists();R.mkdir()
for name in ['manifests','results','reports','checkpoints','logs','plots','control','runtime/tmp','runtime/cache']:(R/name).mkdir(parents=True)
# The reused observer appended one new resource sample to V9. Preserve it here,
# restore the exact sealed historical bytes, and verify the prior seal.
p=OLD/'logs/RESOURCE_SAMPLES.jsonl';data=p.read_bytes();lines=data.splitlines(keepends=True);expected=json.loads((OLD/'PROVENANCE_SHA256.json').read_text())['artifact_SHA256']['logs/RESOURCE_SAMPLES.jsonl'];assert hashlib.sha256(b''.join(lines[:-1])).hexdigest()==expected
new=json.loads(lines[-1]);assert new['UTC']=='2026-10-23T21:33:18Z';(R/'INITIAL_RESOURCE_AUDIT.json').write_text(json.dumps(new,indent=2));q=p.with_name(p.name+'.own_append_undo.part')
with q.open('xb') as f:f.write(b''.join(lines[:-1]));f.flush();os.fsync(f.fileno())
os.replace(q,p);assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
subprocess.run(['git','-C',str(OLD/'source'),'worktree','add','-b','e017-v91-ordered-goal-recovery-20261023-v1',str(R/'source'),'2ba4ccfc31b194323f7d1227da5a724bc260c72a'],check=True)
(R/'run.sh').write_text((OLD/'run.sh').read_text().replace(str(OLD),str(R)));(R/'run.sh').chmod(0o755)
for name in ['REWARD_SPECIFICATION.json','manifests/TRAIN_GOALS.json','manifests/DEV_GOALS.json']:
 shutil.copyfile(OLD/name,R/name)
(R/'source/scripts/e017_v91').mkdir()
core=(OLD/'source/scripts/e017_v9/core.py').read_text().replace(str(OLD),str(R));(R/'source/scripts/e017_v91/core.py').write_text(core)
(R/'STUDY_INTENT.json').write_text(json.dumps(dict(schema='E017_V91_v1',parent_experiment=str(OLD),pre_V9_initializer='V3 selected10000, strict unchanged architecture; no V9 learned weights/moments/cursor',maximum_RL_updates=250,engineering_weights_discarded=True,task_reward_unchanged=True,TEST=False,high_level_training=False,critics=False,operational_reserve_bytes=2*2**30,known_training_checkpoint_bytes=209388115,initial_free_bytes=shutil.disk_usage(R).free,initial_source='2ba4ccfc31b194323f7d1227da5a724bc260c72a',historical_observer_append_exactly_undone_and_SHA_verified=True),indent=2))
print(json.dumps(dict(root=str(R),free_bytes=shutil.disk_usage(R).free,historical_log_restored_to_sealed_SHA256=expected)))
