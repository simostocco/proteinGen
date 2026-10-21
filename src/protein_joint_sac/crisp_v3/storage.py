"""Compact event-sourced replay and atomic complete recovery. No J1 duplication."""
import copy,hashlib,json,os,random,shutil,sqlite3,tempfile
from pathlib import Path
import numpy as np
import torch
from .prior import SHA

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
class Replay:
 def __init__(self,path):
  self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);self.db=sqlite3.connect(self.path);self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
  self.db.executescript("""CREATE TABLE IF NOT EXISTS episodes(id TEXT PRIMARY KEY, data TEXT); CREATE TABLE IF NOT EXISTS events(episode TEXT,n INTEGER,data TEXT,sha TEXT, PRIMARY KEY(episode,n)); CREATE TABLE IF NOT EXISTS transitions(id TEXT PRIMARY KEY,data TEXT); CREATE TABLE IF NOT EXISTS policies(id TEXT PRIMARY KEY,slot INTEGER UNIQUE); CREATE TABLE IF NOT EXISTS outcomes(episode TEXT PRIMARY KEY,data TEXT); CREATE INDEX IF NOT EXISTS transition_level_source ON transitions(json_extract(data,'$.source'),json_extract(data,'$.action.level'),id);""")
 def episode(self,identity,initial,goal,policy,source):
  if source not in ['POLICY_GENERATED','DEMONSTRATION','COUNTERFACTUAL','SYNTHETIC_ENGINEERING']:raise ValueError('source')
  data=dict(initial=initial,goal=goal,policy=policy,source=source,j1=SHA);data['identity_sha256']=digest(data)
  with self.db:self.db.execute('INSERT INTO episodes VALUES (?,?)',(identity,canonical(data)))
 def policy_slot(self,identity):
  row=self.db.execute('SELECT slot FROM policies WHERE id=?',(identity,)).fetchone()
  if row:return row[0]
  slot=self.db.execute('SELECT COUNT(*) FROM policies').fetchone()[0]
  if slot>=256:raise RuntimeError('policy vocabulary full; explicit schema migration required')
  with self.db:self.db.execute('INSERT INTO policies VALUES (?,?)',(identity,slot))
  return slot
 def append(self,episode,event):
  if not self.db.execute('SELECT 1 FROM episodes WHERE id=?',(episode,)).fetchone():raise KeyError(episode)
  prev=self.db.execute('SELECT n,sha FROM events WHERE episode=? ORDER BY n DESC LIMIT 1',(episode,)).fetchone();n=prev[0]+1 if prev else 0;event=copy.deepcopy(event);event['previous_sha']=prev[1] if prev else 'GENESIS';h=digest(event)
  with self.db:self.db.execute('INSERT INTO events VALUES (?,?,?,?)',(episode,n,canonical(event),h))
  return n,h
 def read(self,episode,until=None):
  row=self.db.execute('SELECT data FROM episodes WHERE id=?',(episode,)).fetchone()
  if not row:raise KeyError(episode)
  events=[];previous='GENESIS'
  for n,s,h in self.db.execute('SELECT n,data,sha FROM events WHERE episode=? ORDER BY n',(episode,)):
   if until is not None and n>until:break
   e=json.loads(s)
   if digest(e)!=h or e['previous_sha']!=previous:raise ValueError('replay chain corrupted')
   events.append(e);previous=h
  ep=json.loads(row[0]);expected=ep.pop('identity_sha256')
  if digest(ep)!=expected:raise ValueError('episode header corrupted')
  ep['identity_sha256']=expected;return ep,events
 def transition(self,record):
  required={'id','episode','state_event','next_event','action','goal','behavior_logp','policy','reward','duration','terminated','truncated','status','source'}
  if not required<=record.keys():raise ValueError('transition fields missing')
  if record['duration']<1 or record['status']!='RESOLVED':raise ValueError('unresolved/zero-duration transition')
  if record['source']=='POLICY_GENERATED' and (record['behavior_logp'] is None or not np.isfinite(record['behavior_logp'])):raise ValueError('actual behavior density required')
  ep,_=self.read(record['episode'],record['next_event'])
  if record['action'].get('level') not in ['low','high']:raise ValueError('transition hierarchy')
  if record['goal']!=digest(ep['goal']):raise ValueError('goal reference mismatch')
  if ep['source']!=record['source'] or ep['policy']!=record['policy']:raise ValueError('replay ownership')
  if record['source']=='DEMONSTRATION':raise ValueError('native demonstrations use BC, not actor RL replay')
  if not -1<=record['state_event']<=record['next_event']<self.db.execute('SELECT COUNT(*) FROM events WHERE episode=?',(record['episode'],)).fetchone()[0]:raise ValueError('transition references')
  with self.db:self.db.execute('INSERT INTO transitions VALUES (?,?)',(record['id'],canonical(record)))
 def finish_episode(self,episode,outcome):
  if outcome['terminated'] and outcome['censored']:raise ValueError('inconsistent end status')
  with self.db:self.db.execute('INSERT INTO outcomes VALUES (?,?)',(episode,canonical(outcome)))
 def sample(self,n,rng,source='POLICY_GENERATED',level=None):
  where="json_extract(data,'$.source')=?";args=[source]
  if level is not None:where+=" AND json_extract(data,'$.action.level')=?";args.append(level)
  count=self.db.execute('SELECT COUNT(*) FROM transitions WHERE '+where,args).fetchone()[0]
  if count<n:raise ValueError('insufficient admitted replay')
  return [json.loads(self.db.execute('SELECT data FROM transitions WHERE '+where+' ORDER BY id LIMIT 1 OFFSET ?',args+[i]).fetchone()[0]) for i in rng.sample(range(count),n)]
 def snapshot(self,path):
  dst=sqlite3.connect(path);self.db.backup(dst);dst.close();return str(path)
 def close(self):self.db.close()

def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def set_rng(s):
 random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'])
 if s['cuda']:torch.cuda.set_rng_state_all(s['cuda'])

def payload(system,learner,samplers,config,replay_snapshot,counters):
 return dict(schema='crisp_v3_full_v1',j1_sha256=SHA,profile=system.profile,system=system.state_dict(),optimizers={k:o.state_dict() for k,o in learner.optimizers.items()},schedulers={k:o.state_dict() for k,o in learner.schedulers.items()},steps=learner.steps,samplers={k:v.state_dict() for k,v in samplers.items()},config=config,config_sha256=digest(config),replay_snapshot=replay_snapshot,counters=counters,rng=rng_state())
def atomic_save(path,state,estimated_bytes):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():raise FileExistsError('checkpoint publication is immutable')
 if shutil.disk_usage(path.parent).free<10*2**30+estimated_bytes+64*2**20:raise OSError('10 GiB reserve plus atomic staging gate')
 tmp=path.with_suffix(path.suffix+'.part')
 try:
  with open(tmp,'wb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
  os.replace(tmp,path);fd=os.open(path.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 finally:
  if tmp.exists():tmp.unlink() # Only this task's failed transient publication.

def restore(path,system,learner,samplers,config):
 s=torch.load(path,map_location='cpu',weights_only=False)
 if s['schema']!='crisp_v3_full_v1' or s['j1_sha256']!=SHA or s['config_sha256']!=digest(config):raise ValueError('checkpoint/config/reference mismatch')
 system.load_state_dict(s['system'],strict=True)
 for k,o in learner.optimizers.items():o.load_state_dict(s['optimizers'][k]);learner.schedulers[k].load_state_dict(s['schedulers'][k])
 for k,v in samplers.items():v.load_state_dict(s['samplers'][k])
 learner.steps=s['steps'];set_rng({**s['rng'],'torch':s['rng']['torch'].cpu(),'cuda':[v.cpu() for v in s['rng']['cuda']]});return s
