"""No implicit certification or prior fitting from incomplete acquisition."""
import json,hashlib
from pathlib import Path
from ..contracts import DatasetRef

def inspect(root):
 p=Path(root);required=['RELEASE_MANIFEST.json','unique_sequences.parquet','split_assignments.parquet','homology_groups.parquet']
 return {'root':str(p),'required_files':{n:(p/n).is_file() for n in required},'training_certified':False if not all((p/n).is_file() for n in required) else None}
def require_certified(reference):
 if type(reference)!=DatasetRef or reference.status!='CERTIFIED':raise ValueError('BLOCKED: certified dataset and splits required')
 p=Path(reference.root);m=p/'RELEASE_MANIFEST.json';s=p/'split_assignments.parquet'
 if hashlib.sha256(m.read_bytes()).hexdigest()!=reference.manifest_sha256 or hashlib.sha256(s.read_bytes()).hexdigest()!=reference.split_sha256:raise ValueError('certification hash mismatch')
 if json.loads(m.read_text()).get('status')!='CERTIFIED':raise ValueError('scientific certification not established')
 raise NotImplementedError('G4 independent split and geometry eligibility verification required before production training')
