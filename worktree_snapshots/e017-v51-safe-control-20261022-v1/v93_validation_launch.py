from pathlib import Path
import json,math,shutil,subprocess,time,os
r=Path('/mnt/d/Simone/proteinGen_data/e017_v93_regional_controllability_20261024_v1');read=lambda p:json.loads(p.read_text())
assert read(r/'COMPLETED_train.json')['PASS'];assert read(r/'POLICY_GRADIENT_TESTS.json')['PASS'];assert read(r/'KL_GRADIENT_AND_SCHEMA_TESTS.json')['PASS'];assert read(r/'PROTEIN_TRACE_BACKWARD_TEST.json')['PASS']
groups=[read(r/f'results/TRAIN_C_{i:03}.json') for i in range(16)];rows=[z for g in groups for z in g['records'][:8]];hard=sum(z['metrics']['backbone_valid'] for z in rows)/len(rows);positive=sum(any(z['metrics']['success'] for z in g['records'][:8]) for g in groups);rewards=[z['metrics']['reward'] for z in rows];mean=sum(rewards)/len(rewards);sd=math.sqrt(sum((v-mean)**2 for v in rewards)/len(rewards));env=hard>=.9 and positive>=4 and sd>=.03
size=read(r/'PROTEIN_TRACE_BACKWARD_TEST.json')['complete_recovery_measured_bytes'];required=8*size+2*2**30+128*2**20+128*2**20+64*2**20+32*2**20;free=shutil.disk_usage(r).free
(r/'TRAIN_READINESS.json').write_text(json.dumps(dict(environment_GO=env,filtered_backbone=hard,positive_TRAIN_identities=positive,total_TRAIN_identities=16,primary_TRAIN_candidates=len(rows),reward_SD=sd,likelihood_and_gradient_primitives_GO=True,full_protein_optimizer_release='PENDING future authorization; no protein optimizer updates inV93',substantial_storage_GO=free>=required,measured_full_checkpoint_bytes=size,conservative_campaign_required_bytes=required,free_bytes=free,baseline_evaluation_admitted=True,validation='frozen baseline-only comparison; no algorithm tuning or model selection'),indent=2)+'\n')
assert free>2*2**30+96*2**20
with (r/'logs/validation.log').open('w') as log:
 p=subprocess.Popen([str(r/'run.sh'),str(r/'source/scripts/e017_v93/study.py'),'validation'],stdout=log,stderr=subprocess.STDOUT);(r/'LIVE_STATUS.json').write_text(json.dumps(dict(stage='PROSPECTIVE_FIXED_BASELINES',pid=p.pid,restarts=False)));code=p.wait()
 if code:raise SystemExit(code)
(r/'LIVE_STATUS.json').write_text(json.dumps(dict(stage='SAMPLING_COMPLETE',pid=None,restarts=False,protein_updates=0)))
