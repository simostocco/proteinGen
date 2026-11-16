"""Wait for existing evaluation, execute sealed final controls, publish once."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

R=Path('/mnt/d/Simone/proteinGen_data/e017_v7_amortized_lowlevel_20261023_v1')
BASE=Path(__file__).resolve().parent

def main():
 if (R/'V7_STUDY_COMPLETE.json').exists():print('ALREADY_FINALIZED');return
 assert not (R/'PROVENANCE_SHA256.json').exists()
 deadline=time.monotonic()+7200
 while not (R/'CAMPAIGN_COMPLETED.json').exists():
  status=json.loads((R/'RECOVERY_EXECUTION.json').read_text())
  if status['status']=='STOPPED':raise RuntimeError('campaign stopped: '+json.dumps(status))
  if time.monotonic()>deadline:raise TimeoutError('existing final-evaluation wait limit')
  time.sleep(15)
 for _ in range(15):
  gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
  if not gpu.strip():break
  time.sleep(1)
 else:raise RuntimeError('GPU still owned after campaign completion; do not launch another process')
 aux=json.loads((R/'AUXILIARY_CONTROL_PROTOCOL.json').read_text());causal=R/'analysis_source/causal.py'
 assert hashlib.sha256(causal.read_bytes()).hexdigest()==aux['script_SHA256']
 if not (R/'CAUSAL_DIAGNOSTICS.json').exists():
  with (R/'logs/final_causal_controls.log').open('x') as f:subprocess.run([str(R/'run.sh'),str(causal)],stdout=f,stderr=subprocess.STDOUT,check=True)
 # Final-only helpers, copied for self-contained D: evidence; never alter training source.
 files=['v7_finalize.py','v7_numeric_audit.py','v7_replay_precision_audit.py','v7_recovery_supervisor.py','v7_watch.py']
 for name in files:shutil.copyfile(BASE/name,R/'analysis_source'/name)
 with (R/'logs/final_publication.log').open('x') as f:subprocess.run([str(R/'run.sh'),str(R/'analysis_source/v7_finalize.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
 provenance=json.loads((R/'PROVENANCE_SHA256.json').read_text())
 for name,digest in provenance['output_SHA256'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
 marker=R/'V7_STUDY_COMPLETE.json'
 with marker.open('x') as f:json.dump(dict(complete=True,UTC=time.time(),no_more_experiments=True,provenance_SHA256=hashlib.sha256((R/'PROVENANCE_SHA256.json').read_bytes()).hexdigest()),f,indent=2);f.flush();os.fsync(f.fileno())
 print('V7_FINALIZED',flush=True)

if __name__=='__main__':main()
