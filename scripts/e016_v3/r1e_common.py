from pathlib import Path
import json,hashlib,shutil
import numpy as np
R=Path(__file__).resolve().parents[3];D=R.parent;AA='ACDEFGHIKLMNPQRSTVWY'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(n,x):
 p=R/n;p.parent.mkdir(exist_ok=True,parents=True);q=p.with_suffix(p.suffix+'.part');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def read(n):return json.loads((R/n).read_text())
def desc(x):
 a=np.array(x,dtype=float);a=a[np.isfinite(a)]
 if not len(a):return dict(n=0)
 rng=np.random.default_rng(221);means=np.mean(rng.choice(a,(2000,len(a))),axis=1)
 return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),q05_q95=np.quantile(a,[.05,.95]).tolist(),mean_bootstrap95=np.quantile(means,[.025,.975]).tolist())
def guard(extra=256*2**20):
 if shutil.disk_usage(R).free<10*2**30+extra:raise RuntimeError('D reserve and staging gate')
