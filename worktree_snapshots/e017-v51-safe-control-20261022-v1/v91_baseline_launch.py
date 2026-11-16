from pathlib import Path
import hashlib,json,subprocess,os,sys
R=Path('/mnt/d/Simone/proteinGen_data/e017_v91_ordered_goal_recovery_20261023_v1');OLD=R.parent/'e017_v9_property_steering_20261023_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
test=json.loads((R/'GOAL_OBSERVABILITY_TESTS.json').read_text());assert test['PASS'];assert all(sha(R/'source/scripts/e017_v91'/n)==h for n,h in test['source_SHA256'].items());assert json.loads((R/'CONTROLLABILITY_ADMISSION.json').read_text())['PASS'];protocol=json.loads((R/'FROZEN_STUDY_PROTOCOL.json').read_text());assert sha(R/'source/scripts/e017_v91/goals.py')==protocol['goal_schema_SHA256']
seal=json.loads((OLD/'PROVENANCE_SHA256.json').read_text());aliases={}
for panel in ['TRAIN','DEV']:
 assert sha(R/f'manifests/{panel}_GOALS.json')==sha(OLD/f'manifests/{panel}_GOALS.json')
 for policy in ['A_J1','B_V3']:
  files=sorted((OLD/'results').glob(f'{panel}_{policy}_[0-9][0-9][0-9].json'));assert len(files)==(32 if panel=='TRAIN' else 8);records=[]
  for path in files:
   expected=seal['artifact_SHA256'][str(path.relative_to(OLD))];assert sha(path)==expected;records.append(dict(path=str(path),sha256=expected))
  aliases[panel+'_'+('B_PREV9' if policy=='B_V3' else policy)]=records
(R/'REUSED_FROZEN_BASELINES.json').write_text(json.dumps(dict(aliases=aliases,matching='unchanged parent/J1, legacy B adapter, task/reward, original BASELINE goal/candidate seeds, temperature1 and precision; no large record copies or reruns',historical_files_read_only=True),indent=2))
for panel in ['TRAIN','DEV']:
 log=R/f'logs/{panel}_C_INITIAL.log'
 with log.open('x') as f:
  child=subprocess.Popen([str(R/'run.sh'),str(R/'source/scripts/e017_v91/policy.py'),panel,'C_FIXED'],stdout=f,stderr=subprocess.STDOUT);print(json.dumps(dict(stage=panel,pid=child.pid,log=str(log))),flush=True);code=child.wait()
 if code:raise SystemExit(code)
print('V91_C_INITIAL_BASELINES_COMPLETE',flush=True)
