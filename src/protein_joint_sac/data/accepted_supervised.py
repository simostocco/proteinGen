"""Explicit limited-study admission. Original certification remains unchanged."""
import json,gzip
from pathlib import Path
from functools import lru_cache
import numpy as np
import pyarrow.parquet as pq
from .pdb_split_proposal import PDBSplitProposal
from .certified_unified import file_sha
from ..rl.bc_training import Population,_CERTIFIED_ADMISSION
CLAIM="The initial supervised study uses full canonical PDB polymers with exact-identity separation and no detected cross-split violations of its frozen primary near-duplicate and strong-contained-domain criteria. It does not claim removal of every remote homologue, shared short biological region, or experimental context relationship."
def admit(root,output):
 root=Path(root);output=Path(output)
 if output.exists():raise ValueError('immutable admission exists')
 p=PDBSplitProposal(root)
 for n in ['SOURCE_INTEGRITY_CHECK.json','CROSS_SPLIT_AUDIT.json','LOADER_INTEGRATION_TEST.json','DETERMINISM_CHECK.json']:
  if json.load(open(root/n))['status']!='PASS':raise ValueError('receipt failed '+n)
 for key in ['input_table_manifest','coordinate_provenance_manifest']:
  ref=p.manifest[key]
  if file_sha(ref['path'])!=ref['sha256']:raise ValueError('source manifest mismatch')
 for split in ['TRAIN','VALIDATION']:
  for row in p.iter_joint(split):pass
 rec={'version':'limited-supervised-admission-v1','status':'ACCEPTED_LIMITED_SUPERVISED_PILOT','claim':CLAIM,'proposal_root':str(root),'proposal_manifest_sha256':file_sha(root/'PROPOSAL_MANIFEST.json'),'policy_sha256':file_sha(root/'POLICY_FROZEN.json'),'artifacts':p.manifest['artifacts'],'scope':'LIMITED_PDB_PILOT','TEST':'SEALED_NO_METRIC_EVALUATION','max_updates_per_mode':100,'max_seconds_per_mode':1800,'max_persistent_bytes':2*2**30,'counts':json.load(open(root/'SPLIT_COUNTS.json'))}
 output.write_text(json.dumps(rec,indent=2));return rec
class CoordinateReader:
 def __init__(self,external):self.external=external;self.checked=set()
 def check(self,path):
  if path in self.checked:return path
  path=str(Path(path).resolve())
  if path not in self.external:raise ValueError('untracked coordinate source')
  if path not in self.checked:
   if file_sha(path)!=self.external[path]:raise ValueError('coordinate source hash mismatch')
   self.checked.add(path)
  return path
 @lru_cache(maxsize=4)
 def group(self,path,index):return pq.ParquetFile(path).read_row_group(index,columns=['selected_calpha_coordinates'],use_threads=False).column(0).to_pylist()
 @lru_cache(maxsize=4)
 def small(self,path):return json.load(gzip.open(path,'rt'))
 def read(self,row):
  path=self.check(row['coordinate_path']);i=int(row['coordinate_row'])
  if row['coordinate_kind']=='legacy_cache':
   f=pq.ParquetFile(path);offset=0
   for k in range(f.num_row_groups):
    n=f.metadata.row_group(k).num_rows
    if i<offset+n:return np.asarray(json.loads(self.group(path,k)[i-offset]),dtype=np.float64)
    offset+=n
   raise ValueError('coordinate row missing')
  if row['coordinate_kind'] not in ['priority_v3','regression_v3']:raise ValueError('coordinate kind')
  return np.asarray([x['ca'] for x in self.small(path)['observations'][i]['residue_mapping']],dtype=np.float64)
 def stream(self,rows):
  by={}
  for row in rows:by.setdefault(self.check(row['coordinate_path']),[]).append(row)
  for path,items in sorted(by.items()):
   if items[0]['coordinate_kind']=='legacy_cache':
    wanted={int(x['coordinate_row']):[] for x in items}
    for x in items:wanted[int(x['coordinate_row'])].append(x)
    offset=0
    for b in pq.ParquetFile(path).iter_batches(batch_size=64,columns=['selected_calpha_coordinates'],use_threads=False):
     for j,v in enumerate(b.column(0)):
      if offset+j in wanted:
       x=np.asarray(json.loads(v.as_py()),dtype=np.float64)
       for row in wanted[offset+j]:yield row,x
     offset+=len(b)
   else:
    for row in items:yield row,self.read(row)
class AcceptedPopulation(Population):
 def __init__(self,receipt_path,mode,matched=True):
  rec=json.load(open(receipt_path));root=Path(rec['proposal_root'])
  if rec['status']!='ACCEPTED_LIMITED_SUPERVISED_PILOT' or file_sha(root/'PROPOSAL_MANIFEST.json')!=rec['proposal_manifest_sha256']:raise ValueError('admission mismatch')
  p=PDBSplitProposal(root);rows=[]
  for split in ['TRAIN','VALIDATION']:rows.extend(p.iter_bc_rows(split,mode,matched))
  self.reader=CoordinateReader(p.external)
  super().__init__(rows,'LIMITED_PDB_PILOT',{'admission_sha256':file_sha(receipt_path),'mode':mode,'matched':matched},_admission=_CERTIFIED_ADMISSION)
 @lru_cache(maxsize=16)
 def coordinates(self,i):
  import torch
  x=torch.from_numpy(self.reader.read(self.rows[i]))
  if x.shape!=(len(self.rows[i]['sequence']),3) or not torch.isfinite(x).all():raise ValueError('complete finite mapped CA')
  return x
