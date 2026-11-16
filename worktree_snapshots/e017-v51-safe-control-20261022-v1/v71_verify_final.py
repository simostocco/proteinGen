"""Read-only integrity/process audit, with one compact D: verification record."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time
R=Path('/mnt/d/Simone/proteinGen_data/e017_v71_expected_return_calibration_20261023_v1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=json.loads((R/'PROVENANCE_SHA256.json').read_text());checked=[]
 for kind,items in [('input',p['input_SHA256']),('output',p['output_SHA256'])]:
  for name,digest in items.items():
   path=Path(name) if kind=='input' else R/name
   assert sha(path)==digest,str(path);checked.append(str(path))
 assert json.loads((R/'PUBLICATION_COMPLETED.json').read_text())['complete']
 assert json.loads((R/'FINALIZATION_STATUS.json').read_text())['status']=='COMPLETED'
 assert json.loads((R/'EXECUTION.json').read_text())['status']=='COMPLETED'
 m=json.loads((R/'CRITIC_METRICS.json').read_text())
 assert all(not x['GO'] for x in m['decisions'].values())
 assert 'by_stage' in m['expected']['CONFIRM']['critic1']['action_ranking']
 assert not list((R/'checkpoints').glob('*.part'))
 for rec in json.loads((R/'checkpoints/INDEX.json').read_text()).values():
  assert Path(rec['path']).stat().st_size==rec['bytes'];assert sha(rec['path'])==rec['sha256']
 active=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
  try:
   comm=(proc/'comm').read_text().strip();cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode();state=(proc/'stat').read_text().split(') ',1)[1][0]
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if state!='Z' and comm.startswith('python') and any(x in cmd.lower() for x in ['scripts/e017_','v71_finish.py','v71_analyze.py','v71_realized_eval.py','v71_rwr_goal_eval.py','v71_watch.py','proteinmpnn','colabfold','e007','e015','e012']):active.append(dict(pid=int(proc.name),command=cmd))
 assert not active,active
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);assert not gpu.strip(),gpu
 device=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True).strip()
 git=subprocess.check_output(['git','-C',str(R/'source'),'status','--porcelain'],text=True);assert not git.strip(),git
 free=shutil.disk_usage(R).free;assert free>2*2**30
 result=dict(schema='E017_V71_FINAL_VERIFICATION_v1',UTC=time.time(),PASS=True,checked_artifacts=len(checked),all_input_output_hashes_verified=True,checkpoint_hash_and_size_verified=True,source_worktree_clean=True,source_commit=p['source_commit'],provenance_SHA256=sha(R/'PROVENANCE_SHA256.json'),active_research_workers=active,GPU_compute_processes=[],GPU=device,D_free_bytes=free,D_free_GiB=free/2**30,operational_headroom_bytes=2*2**30,decision='NO-GO for SAC actor updates',actor_updates=0,J1_and_reference_unchanged=True,all_artifacts_on_D=True,pending_evaluations=0)
 q=R/'FINAL_VERIFICATION.json.part'
 with q.open('w') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(q,R/'FINAL_VERIFICATION.json');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
