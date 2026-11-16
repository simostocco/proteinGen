from pathlib import Path
import json,time,datetime
r=Path('/mnt/d/Simone/proteinGen_data/e017_v97_function_preserving_gate_20261031_v1/initialization_admission_20261031_v1')
def read(p):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return None
while True:
 out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 for a in ['A','B']:
  x=read(r/f'arm_{a}/PROGRESS.json');out[a]=x or read(r/f'arm_{a}/LAUNCH.json') or {'state':'QUEUED'}
 end=read(r/'runtime/CONTROLLER_EXIT.json');out['controller_exit']=end
 print(json.dumps(out),flush=True)
 if end:break
 time.sleep(55)
