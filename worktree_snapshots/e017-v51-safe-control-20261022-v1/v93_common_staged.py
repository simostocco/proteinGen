import collections,copy,hashlib,json,math,os,runpy,shutil,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v93_regional_controllability_20261024_v1');OLD=R.parent/'e017_v92_prefix_feasibility_20261024_v1';V91=R.parent/'e017_v91_ordered_goal_recovery_20261023_v1';V9=R.parent/'e017_v9_property_steering_20261023_v1';base=runpy.run_path(str(OLD/'source/scripts/e017_v92/common.py'));p=base['p'];c=base['c'];g=base['g'];read=base['read'];sha=base['sha'];SPEC=base['SPEC'];AA=base['AA'];append=base['append'];prefix_status=base['prefix_status'];extend_checked=base['extend_checked'];load_engine=base['load_engine'];verify_engine=base['verify_engine'];reconstruct=base['reconstruct']
def ns(*x):return int(hashlib.sha256(('V93|9317|'+'|'.join(map(str,x))).encode()).hexdigest()[:15],16)
def save(name,obj):
 path=R/name;tmp=path.with_name(path.name+'.part');assert not tmp.is_symlink()
 with tmp.open('w') as f:json.dump(obj,f,separators=(',',':'),allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);fd=os.open(path.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
def guard():
 assert shutil.disk_usage(R).free>2*2**30+32*2**20,'storage guard'
 assert not (R/'control/STOP').exists(),'cooperative STOP'
 if torch.cuda.is_initialized():assert torch.cuda.max_memory_allocated()<6*2**30,'VRAM guard'
def world(goal,state=None):
 w=c['PropertyWorld'](goal,SPEC,0)
 if state is not None:w.state=copy.deepcopy(state);w.snapshots={}
 return w
