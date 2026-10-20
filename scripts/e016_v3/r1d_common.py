"""D-only R1d compact evidence helpers."""
from pathlib import Path
import hashlib,json,os,shutil,time,re
R=Path(__file__).resolve().parents[3];D=R.parent;C=D/'e016_r1c_reward_reliability_20261019_v1';B=D/'e016_r1b_afdb_msa_critic_20261019_v1'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(4<<20),b''):h.update(x)
 return h.hexdigest()
def write(p,x):
 if isinstance(x,str) and str(p).endswith('.md'):
  parts=re.split(r'(`[^`]*`|https?://[^\s)]+)',x)
  for k in range(0,len(parts),2):
   v=parts[k]
   v=re.sub(r'\b(Obtained|Completed|Reproduced|all|All|of|in|by|for|only|below|under|above|after|before|both|Both|exactly|selected|the|The|uses|use|with|and|across|within|from|source|Source|release|Dataset|CSV|member|numeric|original|parents|families|groups|rows|Mutation|pTM|pLDDT|confidence|replicate|intervals|calls|model|recycles|seed|depth|into|during|score|scores|Nature)(?=[−-]?\d)',r'\1 ',v)
   v=re.sub(r'(?<=\d)(?=(MiB|GiB|kcal|Å|s\b|aa\b|rows?\b|parents?\b|families\b|natural\b|designed\b|calls?\b))',' ',v)
   parts[k]=v
  x=''.join(parts)
 p=R/p;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.part');q.write_text(x if isinstance(x,str) else json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(q,p)
def guard(extra=512*2**20):
 n=shutil.disk_usage(R).free
 if n<10*2**30+extra:raise RuntimeError('10 GiB reserve plus staging headroom required')
 return n
