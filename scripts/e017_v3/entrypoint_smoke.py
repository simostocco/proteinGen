"""One-update development fixtures through real entrypoints; no scientific training campaign."""
import json,time,copy,subprocess
from engineering import R,write
from protein_joint_sac.crisp_v3.training import execute,seal_config
from protein_joint_sac.crisp_v3.prior import SHA,file_sha
start=time.monotonic();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip();base=json.load(open(R/'configs/development_supervised.json'));base.update(updates=1,effective_AA_batch=3,complete_geometry_decisions=1,goal_demonstrations=1,checkpoint_every=1,operational_events=8,collect_every=1,actor_every=1,critic_every=1,validation_panel=None);runs={};parent=None
for stage in ['supervised','low_sac','high_sac','reachability']:
 c={**base,'stage':stage,'seed':1703,'engineering_fixture':True}
 if parent:c['initial_checkpoint']=parent
 grant=R/'configs'/f'ENGINEERING_ONLY_{stage}_ONE_UPDATE.json';grant.write_text(json.dumps(dict(approved=True,scope='Only the user-authorized small-fixture correctness test; not a production training approval',source_commit=commit,config_sha256=seal_config(c),reference_sha256=SHA,stage=stage,max_updates=1),indent=2))
 out=execute(c,R/'engineering_runs',grant);manifest=json.load(open(__import__('pathlib').Path(out)/'LATEST_RECOVERY.json'));parent=dict(path=manifest['checkpoint'],sha256=manifest['sha256']);runs[stage]=dict(path=out,checkpoint=parent,updates=[json.loads(x) for x in (__import__('pathlib').Path(out)/'updates.jsonl').read_text().splitlines()])
 # Resume a completed one-update fixture: zero extra updates, full reconstruction gate.
 resumed=execute(c,R/'engineering_runs',grant,parent['path']);assert (__import__('pathlib').Path(resumed)/'updates.jsonl').read_text()==''
 runs[stage]['completed_resume_zero_updates']=True
write('ENTRYPOINT_ENGINEERING_SMOKE.json',dict(status='PASS',scope='one development-profile update slot per stage only; skipped/censored rows retained',runs=runs,J1_updates=0,production_training_updates=0,seconds=time.monotonic()-start));print('entrypoint fixtures PASS',flush=True)
