"""No implicit certification or prior fitting from incomplete acquisition."""
import json,hashlib
from pathlib import Path
from ..contracts import DatasetRef

def inspect(root):
 p=Path(root);required=['RELEASE_MANIFEST.json','unique_sequences.parquet','split_assignments.parquet','homology_groups.parquet']
 return {'root':str(p),'required_files':{n:(p/n).is_file() for n in required},'training_certified':False if not all((p/n).is_file() for n in required) else None}
def require_certified(reference):
 if type(reference)!=DatasetRef or reference.status!='CERTIFIED':raise ValueError('BLOCKED: certified dataset and splits required')
 from .certified_unified import verify_release
 return verify_release(reference)
