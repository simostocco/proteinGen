from pathlib import Path
import json,gzip,hashlib,datetime,os,sys
ROOT=Path('/mnt/d/Simone/proteinGen_data/e017_v96_stability_review_20261031_v1')
PAIR=ROOT.parent/'e017_v96_paired_burial_control_20261029_v1'
ENG=ROOT.parent/'e017_v96_burial_conditioning_20261029_v1'
R3=ROOT.parent/'e017_v94r3_substantial_geometry_rl_20261026_v1'
ARMS={k:ROOT.parent/f'e017_v96_arm_{k.lower()}_{name}_20261029_v1' for k,name in [('A','original'),('B','burial_token')]}
def read(p):
 p=Path(p)
 with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,x):
 p=ROOT/name;assert p.resolve().is_relative_to(ROOT.resolve());p.parent.mkdir(parents=True,exist_ok=True)
 data=json.dumps(x,indent=2,allow_nan=False) if not isinstance(x,str) else x
 assert not p.exists(),str(p)
 with p.open('x') as f:f.write(data);f.flush();os.fsync(f.fileno())
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
