"""D-only R1d compact evidence helpers."""
from pathlib import Path
import hashlib,json,os,shutil,time
R=Path(__file__).resolve().parents[3];D=R.parent;C=D/'e016_r1c_reward_reliability_20261019_v1';B=D/'e016_r1b_afdb_msa_critic_20261019_v1'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(4<<20),b''):h.update(x)
 return h.hexdigest()
def write(p,x):
 p=R/p;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.part');q.write_text(x if isinstance(x,str) else json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(q,p)
def guard(extra=512*2**20):
 n=shutil.disk_usage(R).free
 if n<10*2**30+extra:raise RuntimeError('10 GiB reserve plus staging headroom required')
 return n
