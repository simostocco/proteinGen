"""Detach exactly one authorized V9.7 controller; no training logic here."""
import json,os,pathlib,subprocess,datetime
r=pathlib.Path('/mnt/d/Simone/proteinGen_data/e017_v97_function_preserving_gate_20261031_v1/paired_production_20261031_v1')
assert json.loads((r/'ORCHESTRATION_QUALIFICATION.json').read_text())['PASS']
receipt=r/'runtime/DETACHED_CONTROLLER.json'
assert not receipt.exists(),'Use exact-frontier reviewed recovery, never duplicate launch'
with (r/'runtime/controller.log').open('x') as log:
 p=subprocess.Popen([str(r/'run.sh'),str(r/'source/production97.py'),'controller'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=str(r),close_fds=True)
 x=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller_PID=p.pid,detached_session=True,log=str(r/'runtime/controller.log'))
 with receipt.open('x') as f:json.dump(x,f);f.flush();os.fsync(f.fileno())
print(json.dumps(x),flush=True)
