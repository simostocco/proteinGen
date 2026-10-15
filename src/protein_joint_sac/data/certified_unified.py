"""Hash-gated PDB-only data boundary; native arrays are offline targets only."""
import hashlib,json,gzip
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from ..contracts import DatasetRef,BuildAction,BCPrefixTarget
from ..geometry.ca_builder import replay,action_from_positions

def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def verify_release(reference):
 if type(reference)!=DatasetRef or reference.status!='CERTIFIED':raise ValueError('certified TRAIN membership required')
 root=Path(reference.root);mp=root/'RELEASE_MANIFEST.json'
 if file_sha(mp)!=reference.manifest_sha256 or file_sha(root/'split_assignments.parquet')!=reference.split_sha256:raise ValueError('dataset provenance mismatch')
 m=json.load(open(mp))
 if m['status']!='CERTIFIED':raise ValueError('provisional source population cannot train')
 for name,expected in m['artifact_sha256'].items():
  path=(root/name).resolve()
  if not str(path).startswith(str(root.resolve())+'/') or file_sha(path)!=expected:raise ValueError('release artifact changed: '+name)
 external=m.get('external_artifact_sha256')
 if not external:raise ValueError('external coordinate/mapping source integrity manifest required')
 for name,expected in external.items():
  path=Path(name).resolve()
  if not str(path).startswith('/mnt/d/') or file_sha(path)!=expected:raise ValueError('external source changed: '+name)
 audit=json.load(open(root/'INDEPENDENT_LEAKAGE_AUDIT.json'))
 if audit['status']!='PASS' or not audit['independent_screen_complete']:raise ValueError('independent homology screening incomplete')
 return m

def native_coordinates(row):
 """The caller is an offline supervised-target reader, never an actor observation."""
 path=Path(row['coordinate_path']);i=int(row['coordinate_row']);kind=row['coordinate_kind']
 if not str(path.resolve()).startswith('/mnt/d/'):raise ValueError('D-only immutable source reference')
 if kind=='legacy_cache':
  pf=pq.ParquetFile(path);offset=0;source=None
  for batch in pf.iter_batches(batch_size=64,columns=['selected_calpha_coordinates'],use_threads=False):
   if offset+len(batch)>i:source=batch.to_pylist()[i-offset]['selected_calpha_coordinates'];break
   offset+=len(batch)
  if source is None:raise ValueError('missing cached coordinate reference')
  coords=json.loads(source)
 elif kind in ['priority_v3','regression_v3']:
  data=json.load(gzip.open(path,'rt'));coords=[r['ca'] for r in data['observations'][i]['residue_mapping']]
 else:raise ValueError('unknown coordinate source policy')
 if any(x is None for x in coords):raise ValueError('missing CA in full-chain BC source')
 x=np.asarray(coords,dtype=np.float64)
 if x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all():raise ValueError('invalid native coordinates')
 return x

def canonical_native_prefix(sequence,coordinates,t):
 """Read only prefix residues/coordinates, with requested length known."""
 N=len(sequence)
 if not 20<=N<=500 or not 0<=t<=N:raise ValueError('fixed length or prefix boundary')
 x=torch.as_tensor(coordinates[:t],dtype=torch.float64).reshape(t,3)
 if not torch.isfinite(x).all():raise ValueError('missing native prefix CA')
 actions=tuple(action_from_positions(sequence[i],x[:i+1]) for i in range(t))
 return replay(N,actions)

def canonical_teacher_prefix(sequence,coordinates,t,source_sha256):
 """State is prefix-only; the next action is a separately typed offline target."""
 N=len(sequence)
 if not 0<=t<N:raise ValueError('unfinished prefix')
 state=canonical_native_prefix(sequence,coordinates,t)
 x=torch.as_tensor(coordinates[:t+1],dtype=torch.float64)
 action=action_from_positions(sequence[t],x)
 return BCPrefixTarget(state,action,source_sha256,t+1)

def iter_joint_rows(reference,split='TRAIN'):
 verify_release(reference)
 if split not in ['TRAIN','VALIDATION','TEST']:raise ValueError('explicit split')
 for batch in pq.ParquetFile(Path(reference.root)/'joint_bc_eligibility.parquet').iter_batches(batch_size=128,use_threads=False):
  for row in batch.to_pylist():
   if row['split']==split and row['eligible'] and row['role']!='NATIVE_AF_CALIBRATION':yield row
