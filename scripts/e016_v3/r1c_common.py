"""R1c local-only evidence helpers; no network client."""
from pathlib import Path
import os,json,hashlib,shutil,time
R=Path(__file__).resolve().parents[3]
B=R.parent/'e016_r1b_afdb_msa_critic_20261019_v1'
P=R.parent/'e016_sequence_first_rl_readiness_20261019_v1'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(4<<20),b''):h.update(x)
 return h.hexdigest()
def write(p,x):
 p=R/p;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.part')
 q.write_text(x if isinstance(x,str) else json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(q,p)
def guard():
 n=shutil.disk_usage(R).free
 if n<11*2**30:raise RuntimeError('10 GiB reserve plus 1 GiB scratch/publication required')
 return n
