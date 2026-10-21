"""Dynamic admission and run-owned rolling latest/best recovery bundles.

All sizes are bytes. A single authoritative index commits BOTH pointers. No file
outside a freshly registered run can become an automatic retirement candidate.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json,math,os,shutil,sqlite3,uuid,warnings,errno
import numpy as np
import torch
GIB=2**30;MIB=2**20
class StoragePause(OSError):pass

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*MIB),b''):h.update(b)
 return h.hexdigest()
def fsync_dir(path):
 fd=os.open(path,os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def atomic_json(path,value):
 path=Path(path);tmp=path.with_name(path.name+'.part')
 if tmp.is_symlink():raise ValueError('symlink metadata staging')
 with open(tmp,'w') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);fsync_dir(path.parent)

class Counter:
 def __init__(self):self.n=0
 def write(self,b):self.n+=len(b);return len(b)
 def flush(self):pass
 def tell(self):return self.n

def serialized_size(state):
 f=Counter();torch.save(state,f);return f.n

def state_digest(value):
 """Canonical semantic fingerprint including EVERY tensor and recovery field."""
 h=hashlib.sha256()
 def visit(v):
  h.update(type(v).__name__.encode()+b':')
  if isinstance(v,torch.Tensor):
   a=v.detach().cpu().contiguous();h.update(str((a.dtype,tuple(a.shape))).encode());h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
  elif isinstance(v,np.ndarray):h.update(str((v.dtype.str,v.shape)).encode());h.update(v.tobytes())
  elif isinstance(v,dict):
   for k in sorted(v,key=lambda x:(type(x).__name__,repr(x))):visit(k);visit(v[k])
  elif isinstance(v,(list,tuple)):
   for x in v:visit(x)
  elif isinstance(v,(set,frozenset)):
   for x in sorted(v,key=repr):visit(x)
  elif v is None or isinstance(v,(str,bool,int,float,np.generic)):h.update(repr(v).encode())
  else:raise TypeError('unfingerprinted recovery type '+str(type(v)))
  h.update(b';')
 visit(value);return h.hexdigest()

@dataclass
class StoragePolicy:
 min_headroom_bytes:int=2*GIB
 warning_free_bytes:int=4*GIB
 near_term_growth_bytes:int=128*MIB
 checkpoint_hint_bytes:int=1024*MIB
 expected_live_replay_bytes:int=0
 expected_log_bytes:int=32*MIB
 def __post_init__(self):
  if self.min_headroom_bytes<0 or self.warning_free_bytes<self.min_headroom_bytes or min(self.near_term_growth_bytes,self.checkpoint_hint_bytes,self.expected_live_replay_bytes,self.expected_log_bytes)<0:raise ValueError('storage policy bounds')
 def admission(self,path,next_write,near_growth=None,free=None):
  if next_write<0:raise ValueError('negative next write')
  growth=self.near_term_growth_bytes if near_growth is None else near_growth
  available=shutil.disk_usage(path).free if free is None else free
  required=int(next_write)+int(growth)+self.min_headroom_bytes
  return dict(admitted=available>=required,free_bytes=available,next_write_bytes=int(next_write),near_growth_bytes=int(growth),minimum_headroom_bytes=self.min_headroom_bytes,required_bytes=required,post_write_free_bytes=available-int(next_write),warning=available-int(next_write)-growth<self.warning_free_bytes)
 def require(self,path,next_write,near_growth=None):
  d=self.admission(path,next_write,near_growth)
  if not d['admitted']:raise StoragePause(json.dumps(d,sort_keys=True))
  if d['warning'] and not getattr(self,'_warned',False):self._warned=True;warnings.warn('D: storage approaching 4 GiB operating warning: '+json.dumps(d),RuntimeWarning)
  if not d['warning']:self._warned=False
  return d
 def startup(self,path):
  # Fresh-run worst rolling window: latest + best + incoming checkpoint; one
  # live replay plus at most three distinct snapshots. Actual writes rechecked.
  return self.require(path,3*self.checkpoint_hint_bytes+4*self.expected_live_replay_bytes+self.expected_log_bytes,near_growth=0)

def atomic_checkpoint(path,state,policy=None,near_growth=None):
 policy=policy or StoragePolicy();path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists() or path.is_symlink():raise FileExistsError('immutable checkpoint exists')
 n=serialized_size(state);policy.require(path.parent,n+MIB,near_growth);tmp=path.with_name(path.name+'.part')
 if tmp.exists() or tmp.is_symlink():raise FileExistsError('staging collision')
 expected=state_digest(state)
 try:
  with open(tmp,'xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
  # Recheck before opening a deserialized copy; verify all networks/targets,
  # moments/schedulers/samplers/RNG/config/replay binding, not just file size.
  loaded=torch.load(tmp,map_location='cpu',weights_only=False)
  if state_digest(loaded)!=expected:raise ValueError('complete checkpoint readback mismatch')
  del loaded
  actual=tmp.stat().st_size
  if actual!=n:raise ValueError('serialization count changed between passes')
  record=dict(path=str(path),sha256=sha(tmp),bytes=actual,state_digest=expected)
  os.replace(tmp,path);fsync_dir(path.parent);return record
 except OSError as error:
  if error.errno in [errno.ENOSPC,errno.EDQUOT]:raise StoragePause('capacity changed during checkpoint write; previous recovery remains authoritative') from error
  raise
 finally:
  if tmp.exists():tmp.unlink() # This invocation's own incomplete transient only.

class RollingRecovery:
 def __init__(self,run,policy=None,create=False):
  self.run=Path(run).absolute();self.policy=policy or StoragePolicy();self.index_path=self.run/'RECOVERY_INDEX.json';self.owner_path=self.run/'STORAGE_OWNER.json'
  if self.run.is_symlink() or self.run.resolve()!=self.run:raise ValueError('run must not traverse symlinks')
  self.run.mkdir(parents=True,exist_ok=True)
  if create:
   if any(self.run.iterdir()):raise ValueError('cannot adopt historical/nonempty run')
   atomic_json(self.owner_path,dict(schema='crisp_storage_owner_v2',run_id=uuid.uuid4().hex,root=str(self.run)))
   atomic_json(self.index_path,dict(schema='crisp_rolling_recovery_v2',run_id=json.load(open(self.owner_path))['run_id'],latest=None,best=None,owned={}))
  self.owner=json.load(open(self.owner_path));self.index=json.load(open(self.index_path))
  if self.owner['schema']!='crisp_storage_owner_v2' or self.owner['root']!=str(self.run) or self.index['run_id']!=self.owner['run_id']:raise ValueError('run ownership mismatch')
  for n in ['checkpoints','replay_snapshots']:(self.run/n).mkdir(exist_ok=True)
 def safe(self,relative):
  p=self.run/relative
  if Path(relative).is_absolute() or '..' in Path(relative).parts or p.parent not in [self.run/'checkpoints',self.run/'replay_snapshots'] or p.is_symlink() or p.parent.is_symlink() or p.resolve().parent!=p.parent:raise ValueError('retirement outside owned artifact directories')
  return p
 def _bundle_valid(self,bundle):
  if bundle is None:return
  p=self.safe(bundle['checkpoint_relative']);r=self.safe(bundle['replay_relative'])
  if sha(p)!=bundle['sha256'] or sha(r)!=bundle['replay']['sha256']:raise ValueError('recovery bundle checksum mismatch')
  state=torch.load(p,map_location='cpu',weights_only=False)
  if state_digest(state)!=bundle['state_digest'] or state['replay_snapshot']['sha256']!=bundle['replay']['sha256'] or state['replay_snapshot']['path']!=str(r):raise ValueError('recovery/replay binding mismatch')
  del state
 def current(self):
  self.index=json.load(open(self.index_path));return self.index
 def live_snapshot_bytes(self,replay):
  return int(replay.db.execute('PRAGMA page_count').fetchone()[0])*int(replay.db.execute('PRAGMA page_size').fetchone()[0])
 def runway(self,replay,extra_write=0,near_growth=None):
  return self.policy.require(self.run,self.policy.checkpoint_hint_bytes+self.live_snapshot_bytes(replay)+extra_write+MIB,near_growth)
 def _refs(self,index=None):
  out=set();idx=index or self.index
  for key in ['latest','best']:
   b=idx[key]
   if b:out.update([b['checkpoint_relative'],b['replay_relative']])
  return out
 def collect_superseded(self):
  idx=self.current();self._bundle_valid(idx['latest']);self._bundle_valid(idx['best']);refs=self._refs(idx);retired=[];dirty=False
  if idx['latest'] is None:return retired
  for relative,meta in list(idx['owned'].items()):
   if relative in refs:continue
   p=self.safe(relative)
   if p.exists():
    if p.stat().st_nlink!=1 or sha(p)!=meta['sha256']:raise ValueError('refuse to retire modified/shared artifact '+relative)
    with open(self.run/'RETIREMENTS.jsonl','a') as journal:
     journal.write(json.dumps(dict(status='verified_superseded_retirement',relative=relative,sha256=meta['sha256'],bytes=meta['bytes'],replacement=idx['latest']['checkpoint_relative']))+'\n');journal.flush();os.fsync(journal.fileno())
    p.unlink();retired.append(dict(relative=relative,bytes=meta['bytes'],sha256=meta['sha256']))
   del idx['owned'][relative];dirty=True
  if dirty:
   fsync_dir(self.run/'checkpoints');fsync_dir(self.run/'replay_snapshots');atomic_json(self.index_path,idx)
  self.index=idx;return retired
 def _recover_pending(self):
  pending=self.run/'PENDING_PUBLICATION.json'
  if not pending.exists():return
  d=json.load(open(pending));idx=self.current()
  if d['run_id']!=self.owner['run_id']:raise ValueError('pending ownership')
  self._bundle_valid(idx['latest']);refs=self._refs(idx)
  # A crash before first publication has no prior recovery. Only intent-listed
  # own staging files are removed; no historical directory scan is performed.
  for relative in d['paths']:
   if relative in refs:continue
   p=self.safe(relative)
   if p.exists():
    if p.stat().st_nlink!=1:raise ValueError('shared pending artifact')
    p.unlink()
  pending.unlink();fsync_dir(self.run)
 def publish(self,replay,state_factory,update,validation=None,near_growth=None):
  self._recover_pending();idx=self.current();self.collect_superseded();idx=self.current()
  validation=validation or None
  if validation:
   if not math.isfinite(validation['value']) or validation['mode'] not in ['min','max'] or not validation.get('panel_sha256'):raise ValueError('invalid validation selection')
   old=idx['best']
   if old and any(old['validation'][k]!=validation[k] for k in ['metric','mode','panel_sha256']):raise ValueError('best metric/panel changed; separate lineage required')
  revision=replay.revision();old=idx['latest'];reuse=old and old['replay']['revision']==revision
  cp_rel=f'checkpoints/checkpoint_{update:08d}_{uuid.uuid4().hex[:8]}.pt';rp_rel=old['replay_relative'] if reuse else f'replay_snapshots/replay_{update:08d}_{uuid.uuid4().hex[:8]}.sqlite';cp=self.safe(cp_rel);rp=self.safe(rp_rel)
  estimated_replay=0 if reuse else self.live_snapshot_bytes(replay)
  projected_snap=dict(old['replay']) if reuse else dict(path=str(rp),sha256='0'*64,bytes=estimated_replay,revision=revision)
  state=state_factory(projected_snap);next_checkpoint=serialized_size(state)
  self.policy.require(self.run,next_checkpoint+estimated_replay+MIB,near_growth)
  paths=[cp_rel,cp_rel+'.part']+([] if reuse else [rp_rel,rp_rel+'.part'])
  for relative in paths:
   if self.safe(relative).exists():raise FileExistsError('publication intent collision')
  atomic_json(self.run/'PENDING_PUBLICATION.json',dict(run_id=self.owner['run_id'],paths=paths,update=update))
  if reuse:
   if sha(rp)!=old['replay']['sha256']:raise ValueError('reused replay snapshot changed')
   snap=dict(old['replay'])
  else:
   tmp=self.safe(rp_rel+'.part');replay.snapshot(tmp)
   db=sqlite3.connect('file:'+str(tmp)+'?mode=ro',uri=True)
   try:
    if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('replay backup integrity')
   finally:db.close()
   with open(tmp,'rb') as f:os.fsync(f.fileno())
   snap=dict(path=str(rp),sha256=sha(tmp),bytes=tmp.stat().st_size,revision=revision);os.replace(tmp,rp);fsync_dir(rp.parent)
  state['replay_snapshot']=snap;record=atomic_checkpoint(cp,state,self.policy,near_growth)
  # No old checkpoint/snapshot is retired until both durable files read back.
  bundle=dict(checkpoint=str(cp),checkpoint_relative=cp_rel,sha256=record['sha256'],bytes=record['bytes'],state_digest=record['state_digest'],replay_relative=rp_rel,replay=snap,update=update,validation=validation,config_sha256=state['config_sha256'],source_commit=state['config'].get('source_commit'))
  self._bundle_valid(bundle);new=json.loads(json.dumps(idx));new['latest']=bundle
  if validation and (new['best'] is None or (validation['value']<new['best']['validation']['value'] if validation['mode']=='min' else validation['value']>new['best']['validation']['value'])):new['best']=bundle
  new['owned'][cp_rel]=dict(sha256=record['sha256'],bytes=record['bytes'],kind='checkpoint');new['owned'][rp_rel]=dict(sha256=snap['sha256'],bytes=snap['bytes'],kind='replay_snapshot')
  atomic_json(self.index_path,new);self.index=new;self.policy.checkpoint_hint_bytes=max(self.policy.checkpoint_hint_bytes,record['bytes'])
  (self.run/'PENDING_PUBLICATION.json').unlink();fsync_dir(self.run);retired=self.collect_superseded();return dict(bundle=bundle,retired=retired,snapshot_reused=bool(reuse))
