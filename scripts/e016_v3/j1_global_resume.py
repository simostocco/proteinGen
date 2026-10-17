"""Checked J1 recovery entry point: refuses duplicate or completed training."""
import argparse,hashlib,json,os,signal,subprocess,sys
from pathlib import Path
R=Path("/mnt/d/Simone/proteinGen_data/e016_j1_global_pretraining_20261017_v1")
def sha(path):
 with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def inspect(root=R,processes=None):
 env=json.load(open(root/"ENVIRONMENT_AND_SOURCE_RECEIPT.json"))
 for name,expected in env["critical_source_sha256"].items():
  if sha(root/"source"/name)!=expected:raise ValueError("source differs from admitted execution: "+name)
 if processes is None:
  if "sandbox" in Path("/proc/1/comm").read_text().lower():raise RuntimeError("host process inspection unavailable in sandbox; use approved host execution")
  processes=[]
  for p in Path("/proc").iterdir():
   if not p.name.isdigit():continue
   try:
    args=(p/"cmdline").read_bytes().split(b"\0")
    if args and b"python" in args[0] and any(a.endswith(b"/j1_global_train.py") for a in args[1:]):processes.append(int(p.name))
   except OSError:continue
 complete=root/"runs/J1/COMPLETED.json"
 if complete.exists():
  record=json.load(open(complete))
  for p,k in [("selected_checkpoint","selected_sha256"),("latest_checkpoint","latest_sha256")]:
   if sha(record[p])!=record[k]:raise ValueError("completed checkpoint integrity mismatch")
 return {"source_integrity":"PASS","active_training_pids":processes,"already_completed":complete.exists(),"resume_allowed":not processes and not complete.exists()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--check-only",action="store_true");a=ap.parse_args();state=inspect();print(json.dumps(state),flush=True)
 if a.check_only or state["already_completed"]:return
 if state["active_training_pids"]:raise RuntimeError("existing J1 training process; no duplicate launched")
 child=subprocess.Popen([sys.executable,"-B",str(R/"source/scripts/e016_v3/j1_global_train.py"),"train"])
 def forward(sig,frame):child.send_signal(sig)
 signal.signal(signal.SIGTERM,forward);signal.signal(signal.SIGINT,forward)
 code=child.wait()
 if code:raise SystemExit(code)
if __name__=="__main__":main()
