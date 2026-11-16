from pathlib import Path
import json,hashlib,shutil,subprocess,time,os
r=Path('/mnt/d/Simone/proteinGen_data/e017_v92_prefix_feasibility_20261024_v1');read=lambda p:json.loads(p.read_text())
assert read(r/'CORRECTNESS_TESTS.json')['PASS']
for pol in ['C','D100','A','B']:assert read(r/f'COMPLETED_train_{pol}.json')['complete']
rows=[z for p in (r/'results').glob('TRAIN_*_[UVM]*_[0-9][0-9][0-9].json') for z in read(p)['records']];assert len(rows)==512;assert all(z['repairs']==0 and z['termination'] is None for z in rows)
sizes=[p.stat().st_size for p in (r/'results').glob('TRAIN_D100_*.json') if '.sha.' not in p.name];next_bytes=int(max(sizes)*192*1.5);free=shutil.disk_usage(r).free;assert free>2*2**30+128*2**20+next_bytes
v=read(r/'FUTURE_VIABILITY.json');assert not any(z['full'] for z in v['records'])
(r/'TRAIN_ADMISSION.json').write_text(json.dumps(dict(PASS=True,TRAIN_candidates=512,policies_frozen=True,no_repairs=True,no_aborts=True,free_bytes=free,conservative_remaining_evidence_bytes=next_bytes,required_headroom_bytes=2*2**30,additional_growth_guard_bytes=128*2**20,lookahead_status='NOT_RUN:128 continuation samples show no full-goal ranking signal',methods_frozen_unchanged=True,admission='engineering/resource admission only, not scientific promotion'),indent=2)+'\n')
# One sequential process: no restart policy or background supervisor.
for pol in ['C','D100']:
 with (r/f'logs/confirm_{pol}.log').open('a') as log:
  proc=subprocess.Popen([str(r/'run.sh'),str(r/'source/scripts/e017_v92/study.py'),'confirm',pol],stdout=log,stderr=subprocess.STDOUT)
  (r/'LIVE_STATUS.json').write_text(json.dumps(dict(stage='CONFIRM',policy=pol,pid=proc.pid,launcher_pid=os.getpid(),restarts=False)))
  code=proc.wait()
  if code:raise SystemExit(f'{pol} failed with{code}; no subsequent worker launched')
(r/'LIVE_STATUS.json').write_text(json.dumps(dict(stage='COMPLETE',pid=None,restarts=False,training_updates=0)))
