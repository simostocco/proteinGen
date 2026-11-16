from pathlib import Path
import os,json,hashlib,datetime,uuid,gzip,sys
ROOT=Path('/mnt/d/Simone/proteinGen_data/e017_v97_function_preserving_gate_20261031_v1')
WT=ROOT.parent/'e017_v97_engineering_worktree_20261031_v1'
PAIR=ROOT.parent/'e017_v96_paired_burial_control_20261029_v1'
ENG=ROOT.parent/'e017_v96_burial_conditioning_20261029_v1'
R3=ROOT.parent/'e017_v94r3_substantial_geometry_rl_20261026_v1'
def read(p):
 p=Path(p)
 with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def put(p,x,immutable=True):
 p=Path(p);assert p.resolve().is_relative_to(ROOT.resolve());p.parent.mkdir(parents=True,exist_ok=True);data=x if isinstance(x,str) else json.dumps(x,separators=(',',':'),allow_nan=False)
 if immutable and p.exists():assert p.read_text()==data,('IMMUTABLE_CHANGED',str(p));return
 t=p.with_name(p.name+'.part_'+uuid.uuid4().hex)
 with t.open('x') as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(t,p);fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def save(name,x):put(ROOT/name,x)
def seed(*items):return int(hashlib.sha256(('E017_V97|'+ '|'.join(map(str,items))).encode()).hexdigest()[:15],16)
def binding():return dict(source=read(ROOT/'SOURCE_BINDING.json'),design_SHA256=sha(ROOT/'FROZEN_DESIGN.json'),train_SHA256=sha(ROOT/'TRAINING_POPULATION.json'),monitor_SHA256=sha(ROOT/'MONITOR_POPULATION.json'),gate='TANH_SHARED_GEOMETRY_DELTA_2')
def verify():
 for p,h in read(ROOT/'SOURCE_BINDING.json')['files'].items():assert sha(p)==h,('SOURCE_BINDING',p)
 for p,h in read(ROOT/'SOURCE_BINDING.json')['models'].items():assert sha(p)==h,('MODEL_BINDING',p)
def factory(arm,runroot):
 sys.path.insert(0,str(ENG/'source96'));import common96
 import gate97
 P,F,_=common96.load();loader=P['K']['load_engine']
 def load(policy):
  e,m=loader(policy);F['checkpoint'](e,5000)
  if arm=='B':gate97.install(e.actor)
  assert not hasattr(e.parent,'burial_gate')
  return e,m
 P['K']['load_engine']=load
 arm_binding=lambda:dict(binding(),arm=arm)
 spec=read(ROOT/'CONFIGURATION.json');g=P['Runtime'].__init__.__globals__;g.update(ROOT=ROOT,SPEC=spec,V93=ROOT/'qualified_train',bindings=arm_binding)
 P.update(ROOT=ROOT,SPEC=spec,V93=ROOT/'qualified_train',bindings=arm_binding)
 return P,F
