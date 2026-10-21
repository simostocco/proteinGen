"""Inject pre-update forecast pressure; save current full state, no optimizer calls."""
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import json,copy,sqlite3,time,subprocess
import torch
from protein_joint_sac.crisp_v3 import training
from protein_joint_sac.crisp_v3.recovery import RollingRecovery,StoragePolicy,StoragePause,state_digest
from protein_joint_sac.crisp_v3.storage import Replay,digest
from protein_joint_sac.crisp_v3.optimization import Learner
from protein_joint_sac.crisp_v3.prior import SHA
E=Path(__file__).resolve().parents[3];R=E.parent;start=time.monotonic();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=E/'source',text=True).strip();d=json.load(open(R/'results/ENTRYPOINT_ENGINEERING_SMOKE.json'))['runs']['reachability']['checkpoint'];s=torch.load(d['path'],map_location='cpu',weights_only=False);c={k:v for k,v in s['config'].items() if k not in ['data_manifest','data_input_hashes','native_limits_sha256','validation_panel_sha256','source_commit']};c['updates']=2;c['storage_policy']=dict(min_headroom_bytes=2*2**30,warning_free_bytes=4*2**30,near_term_growth_bytes=32*2**20,checkpoint_hint_bytes=64*2**20,expected_live_replay_bytes=0,expected_log_bytes=32*2**20)
s['config']={**s['config'],**c,'source_commit':commit};s['config_sha256']=digest(s['config']);run=E/'safe_pause_fixture';m=RollingRecovery(run,StoragePolicy(**c['storage_policy']),create=True);src=sqlite3.connect('file:'+s['replay_snapshot']['path']+'?mode=ro',uri=True);dst=sqlite3.connect(run/'replay.sqlite');src.backup(dst);src.close();dst.close();replay=Replay(run/'replay.sqlite')
def f(snap):a=copy.copy(s);a['replay_snapshot']=snap;return a
b=m.publish(replay,f,1)['bundle'];replay.close();(run/'CONFIG.json').write_text(json.dumps(s['config'],indent=2));g=E/'results/PAUSE_FIXTURE_ONLY.json';g.write_text(json.dumps(dict(approved=True,stage=c['stage'],max_updates=2,reference_sha256=SHA,config_sha256=training.seal_config(c),source_commit=commit,scope='Storage pressure injected before any update; optimizer entrypoints disabled')))
with ExitStack() as stack:
 for method in ['critic','low_actor','high_actor','reach','supervised']:stack.enter_context(patch.object(Learner,method,side_effect=AssertionError('optimizer work forbidden')))
 stack.enter_context(patch.object(training,'supervised_update',side_effect=AssertionError('optimizer work forbidden')))
 stack.enter_context(patch.object(RollingRecovery,'runway',side_effect=StoragePause('ENGINEERING simulated forecast pressure before update')))
 returned=training.execute(c,E/'unused_output_parent',g,b['checkpoint'])
assert returned==str(run);status=json.load(open(run/'RUN_STATUS.json'));pause=json.load(open(run/'PAUSED_STORAGE.json'));assert status['status']=='PAUSED_STORAGE' and pause['exact_current_state_saved'] and pause['current_completed_updates']==1 and not pause['current_update_incomplete'];latest=json.load(open(run/'RECOVERY_INDEX.json'))['latest'];restored=torch.load(latest['checkpoint'],map_location='cpu',weights_only=False);expected=f(latest['replay']);assert state_digest(restored)==state_digest(expected)
with open(E/'results/SAFE_ENGINE_PAUSE_TEST.json','x') as out:json.dump(dict(status='PASS',pre_update_pause=True,coherent_current_full_state_saved=True,optimizer_calls=0,J1_updates=0,resume_update=latest['update'],seconds=time.monotonic()-start),out,indent=2)
print('safe engine pause PASS',flush=True)
