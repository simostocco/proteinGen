"""Read-only candidate-split adapter. Does not grant BC optimizer admission.
Uses the existing canonical-gauge prefix/coordinate implementation unchanged.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from .certified_unified import file_sha,native_coordinates,canonical_teacher_prefix

class PDBSplitProposal:
 training_authorized=False
 def __init__(self,root):
  self.root=Path(root).resolve()
  if not str(self.root).startswith('/mnt/d/'):raise ValueError('D-only proposal')
  self.manifest=json.load(open(self.root/'PROPOSAL_MANIFEST.json'))
  if self.manifest['status']!='PASS_PRIMARY_SCREEN / SCIENTIFIC_REVIEW_PENDING' or self.manifest['training_authorized'] is not False:raise ValueError('review-pending proposal only')
  for n,h in self.manifest['artifacts'].items():
   p=(self.root/n).resolve()
   if not p.is_relative_to(self.root) or file_sha(p)!=h:raise ValueError('proposal artifact changed: '+n)
  self.sequences={}
  strings=set();group_split={}
  for row in pq.read_table(self.root/'SEQUENCE_ASSIGNMENTS.parquet').to_pylist():
   h=row['sequence_id'];s=row['canonical_full_sequence'];g=row['protected_group_id'];split=row['split']
   if h in self.sequences or s in strings or not 20<=len(s)<=500 or set(s)-set('ACDEFGHIKLMNPQRSTVWY') or hashlib.sha256(s.encode()).hexdigest()!=h:raise ValueError('canonical identity contradiction')
   if split not in ('TRAIN','VALIDATION','TEST') or (g in group_split and group_split[g]!=split):raise ValueError('invalid split/group ownership')
   self.sequences[h]=row;strings.add(s);group_split[g]=split
  ref=self.manifest['coordinate_provenance_manifest']
  if file_sha(ref['path'])!=ref['sha256']:raise ValueError('parent coordinate provenance changed')
  self.external=json.load(open(ref['path']))['external_artifact_sha256'];self.checked=set()
 def iter_sequences(self,split,matched_joint=False):
  if split not in ('TRAIN','VALIDATION','TEST'):raise ValueError('explicit split required')
  for r in self.sequences.values():
   if r['split']==split and (not matched_joint or r['joint_eligible']):yield r
 def iter_joint(self,split):
  if split not in ('TRAIN','VALIDATION','TEST'):raise ValueError('explicit split required')
  for batch in pq.ParquetFile(self.root/'JOINT_OBSERVATION_ASSIGNMENTS.parquet').iter_batches(batch_size=128,use_threads=False):
   for row in batch.to_pylist():
    parent=self.sequences.get(row['sequence_hash'])
    if parent is None or row['split']!=parent['split'] or not row['complete_CA'] or not row['eligible'] or row['fragment']:raise ValueError('joint ownership/eligibility contradiction')
    if row['split']==split:yield row
 def iter_bc_rows(self,split,mode='joint',matched_joint=True,evaluation_only=False):
  """Existing BC row schema, without constructing/admitting a training Population.
  homology_group is a legacy software field alias for the declared protected group;
  it does not establish universal evolutionary homology independence.
  """
  if split=='TEST' and not evaluation_only:raise ValueError('final TEST is evaluation-only')
  if mode not in ('sequence','joint') or (mode=='joint' and not matched_joint):raise ValueError('explicit matched joint population')
  if mode=='sequence':
   source_hash=file_sha(self.root/'POLICY_FROZEN.json')
   for p in self.iter_sequences(split,matched_joint):
    yield {'observation_id':'sequence:'+p['sequence_id'],'sequence':p['canonical_full_sequence'],'sequence_hash':p['sequence_id'],'homology_group':p['protected_group_id'],'split':split,'role':{'TRAIN':'TRAINING','VALIDATION':'DEVELOPMENT','TEST':'FINAL_TEST'}[split],'source_sha':source_hash,'fragment':False,'label_positions':list(range(1,p['length']+1))}
  else:
   for o in self.iter_joint(split):
    p=self.sequences[o['sequence_hash']]
    yield {**o,'sequence':p['canonical_full_sequence'],'homology_group':p['protected_group_id'],'label_positions':list(range(1,p['length']+1))}
 def teacher_prefix(self,row,t):
  parent=self.sequences[row['sequence_hash']]
  if parent['split']!=row['split']:raise ValueError('parent partition mismatch')
  path=str(Path(row['coordinate_path']).resolve())
  if path not in self.external:raise ValueError('untracked coordinate source')
  if path not in self.checked:
   if file_sha(path)!=self.external[path]:raise ValueError('coordinate source changed')
   self.checked.add(path)
  x=native_coordinates(row)
  if x.shape!=(parent['length'],3) or not np.isfinite(x).all():raise ValueError('full polymer/CA correspondence mismatch')
  return canonical_teacher_prefix(parent['canonical_full_sequence'],x,t,row['source_sha'])
 def training_population(self,*args,**kwargs):
  raise ValueError('SCIENTIFIC_REVIEW_PENDING: proposal cannot authorize production training')
