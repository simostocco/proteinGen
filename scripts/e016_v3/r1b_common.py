"""R1b D-only bounded HTTP cache and resource checks."""
import json,hashlib,os,time,shutil,urllib.parse
try:import requests
except ImportError:requests=None
from pathlib import Path
R=Path(__file__).resolve().parents[3];PREV=R.parent/'e016_sequence_first_rl_readiness_20261019_v1';SESSION=requests.Session() if requests else None
if SESSION is not None:SESSION.headers['User-Agent']='proteinGen-E016-R1b-bounded-public-PDB-controls/1.0'
def write(name,value):
 if isinstance(value,str) and name.endswith('.md'):
  import re
  chunks=re.split(r'(`[^`]*`|https?://[^\s)]+)',value)
  for i in range(0,len(chunks),2):
   text=chunks[i]
   text=re.sub(r'\b(all|All|of|in|In|by|for|only|Only|below|under|above|to|after|before|both|exactly|selected|the|The|uses|use|with|With|made|totals|contains|model|recycles|seed|depth|budget|score|scores|ColabFold|JAX|alphafold-colabfold)(?=\d)',r'\1 ',text)
   text=re.sub(r'(\d)(?=[A-Za-z]{4,}\b)',r'\1 ',text)
   text=re.sub(r'(\d)(?=(and|is|has|on|of|to|in|as|can|are|for|its|AA|CA|MSA|CPU|GPU|RMSE|MiB|GiB|nats|pTM|pLDDT|rows)\b)',r'\1 ',text)
   text=text.replace('pTM(single','pTM (single').replace('regionalpLDDT','regional pLDDT').replace('freeD','free D:').replace('(n12)','(n=12)').replace('(n3)','(n=3)').replace('sigma1','sigma 1')
   chunks[i]=text
  value=''.join(chunks)
 p=R/name;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.part');q.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n' if not isinstance(value,str) else value);os.replace(q,p)
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def guard(extra=2**30):
 free=shutil.disk_usage(R).free
 if free<10*2**30+extra:raise RuntimeError('10GiB reserve plus publication/scratch gate')
 return free
def events():
 p=R/'network/HTTP_EVENTS.jsonl';return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
def log_event(x):
 p=R/'network/HTTP_EVENTS.jsonl'
 with p.open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def get(url,relative,max_bytes=32*2**20):
 p=R/relative;receipt=p.with_suffix(p.suffix+'.http.json')
 if receipt.exists():
  meta=json.load(open(receipt));assert p.exists() and sha(p)==meta['sha256'];return p,meta
 guard();used=sum(e.get('response_bytes',0) for e in events());remaining=750*2**20-used
 if remaining<=0:raise RuntimeError('750MiB aggregate download cap')
 begin=time.monotonic();record={'method':'GET','url':url,'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'scope':'R1b targeted public docs/native controls'}
 try:
  response=SESSION.get(url,timeout=(15,60),stream=True);parts=[];n=0
  for b in response.iter_content(1<<16):
   n+=len(b)
   if n>min(max_bytes,remaining):raise RuntimeError('bounded HTTP response/download cap exceeded')
   parts.append(b)
  body=b''.join(parts);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.part');q.write_bytes(body);os.replace(q,p);record.update(status=response.status_code,final_url=response.url,response_bytes=n,sha256=sha(p),headers={k:v for k,v in response.headers.items() if k.lower() in ['content-type','content-length','date','etag','last-modified','retry-after']},redirects=[{'url':v.url,'status':v.status_code} for v in response.history],seconds=time.monotonic()-begin);write(str(receipt.relative_to(R)),record);log_event(record);time.sleep(.3);return p,record
 except Exception as exc:record.update(error=type(exc).__name__+': '+str(exc),seconds=time.monotonic()-begin);log_event(record);raise

def getjson(url,relative,max_bytes=16*2**20):
 p,m=get(url,relative,max_bytes)
 if m['status']!=200:return None,m
 try:return json.load(open(p)),m
 except ValueError:return None,dict(m,error='not JSON')
