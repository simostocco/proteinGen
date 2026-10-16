"""One-time trusted SHA-verified NumPy metadata normalization; original checkpoint preserved."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
import torch
from protein_joint_sac.rl.checkpointing import load_checked,save_atomic
from protein_joint_sac.data.certified_unified import file_sha
R=Path('/mnt/d/Simone/proteinGen_data/e016_scaled10m_training_20261016_v1');old=R/'runs/S1/checkpoint_00250.pt';new=R/'runs/S1/recovery_metadata_v2/checkpoint_00250.pt';new.parent.mkdir(parents=True,exist_ok=False)
assert file_sha(old)==old.with_name(old.name+'.sha256').read_text().strip()
# Only permit NumPy scalar/dtype reconstruction in this audited local checkpoint.
from numpy._core.multiarray import scalar
with torch.serialization.safe_globals([scalar,np.dtype,type(np.dtype(np.float64)),type(np.dtype(np.float32))]):state=load_checked(old)
oldentry=state['lineage']['entrypoint_sha256'];saved=R/'attempts/checkpoint_metadata_v1/scaled_train.py';assert file_sha(saved)==oldentry
state['validation']=json.loads(json.dumps(state['validation'],allow_nan=False));state['lineage']['entrypoint_sha256']=file_sha(R/'source/scripts/e016_v3/scaled_train.py')
sha=save_atomic(new,state);reloaded=load_checked(new)
for name in ['model','optimizer','scheduler','rng','sampler']:
 def equal(a,b):
  if isinstance(a,torch.Tensor):return torch.equal(a.cpu(),b.cpu())
  if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
  if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
  return a==b
 assert equal(state[name],reloaded[name]),name
receipt={'status':'VERIFIED_METADATA_ONLY_RECOVERY','path':str(new),'sha256':sha,'update':250,'old_checkpoint':str(old),'old_sha256':file_sha(old),'old_entrypoint_sha256':oldentry,'new_entrypoint_sha256':state['lineage']['entrypoint_sha256'],'changed':'validation NumPy scalars normalized to built-in JSON numeric types; runner metadata-only change recorded','unchanged':['model','optimizer','scheduler','sampler','RNG','update','exposures','objective','data','precision','architecture']}
(R/'runs/S1/RECOVERY_START.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
