"""Versioned, fail-closed infrastructure binding compatibility; no neural changes.

The original scientific release remains the serialized checkpoint identity.
Only the enumerated infrastructure revisions normalize to that identity.
Both direct mapping lookups and production function globals use one callable.
"""
import copy,hashlib,json,subprocess
from pathlib import Path
ROOT=Path('/mnt/d/Simone/proteinGen_data/e017_v94r3_substantial_geometry_rl_20261026_v1')
CONTRACT=ROOT/'evaluation_repair_20261028_v1/COMPATIBILITY_CONTRACT.json'
ORIGINAL_RELEASE_SHA256='0b09f974525894af196d9f8d74d94913c7987d567ce96aeeaf6d21275943a94c'
QUALIFIED_INFRA_BINDING_SHA256='fa027262f4cfce12b1b76b0d2b84a89a1e3cc148e7fc2db321a86df387e049d5'
SCHEMA='E017_R3_STRICT_INFRA_BINDING_2'
NEW_MODULE='scripts/e017_v94r3/binding_compatibility.py'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def validate_contract(m):
 assert m['schema']==SCHEMA and m['root']==str(ROOT)
 old=m['original_binding'];infra=m['qualified_infrastructure_binding'];new=m['repaired_infrastructure_binding']
 assert m['original_release_SHA256']==ORIGINAL_RELEASE_SHA256
 assert digest(infra)==QUALIFIED_INFRA_BINDING_SHA256
 assert set(old)==set(infra)==set(new)
 assert old['source'].keys()==infra['source'].keys()
 assert set(new['source'])==set(old['source'])|{NEW_MODULE}
 for k in old:
  if k not in ['source','V94_source_commit']:assert old[k]==infra[k]==new[k],('scientific binding changed',k)
 for name,h in old['source'].items():
  if name!='scripts/e017_v94r2/common.py':assert infra['source'][name]==new['source'][name]==h,('scientific source changed',name)
  else:assert infra['source'][name]==new['source'][name]
 assert len(m['artifacts'])>=10 and m['final_update']==5000 and m['final_attempts']==80000
 assert m['final_actor_fingerprint'] and m['checkpoint_binding']==old
 return m
def normalize(actual,m):
 validate_contract(m)
 assert actual in [m['original_binding'],m['qualified_infrastructure_binding'],m['repaired_infrastructure_binding']],'UNRECOGNIZED_OR_SCIENTIFICALLY_CHANGED_BINDING'
 return copy.deepcopy(m['original_binding'])
def verify_artifact(path,expected_sha,expected_bytes=None):
 p=Path(path);assert not p.is_symlink(),'SYMLINK_ARTIFACT'
 assert p.is_file(),'REQUIRED_ARTIFACT_MISSING'
 if expected_bytes is not None:assert p.stat().st_size==expected_bytes,'ARTIFACT_SIZE_MISMATCH'
 assert sha(p)==expected_sha,'ARTIFACT_HASH_MISMATCH'
 return True
def verify_checkpoint_lineage(binding,m):
 assert binding==m['original_binding'],'INCOMPATIBLE_CHECKPOINT_LINEAGE'
 return True
def verify_live_contract(m):
 validate_contract(m)
 verify_artifact(ROOT/'qualification/RELEASE.json',ORIGINAL_RELEASE_SHA256)
 assert read(ROOT/'qualification/RELEASE.json')['binding']==m['original_binding']
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'source',text=True).strip()==m['repaired_infrastructure_binding']['V94_source_commit']
 assert sha(Path(__file__))==m['repaired_infrastructure_binding']['source'][NEW_MODULE]
 for name,h in m['repaired_infrastructure_binding']['source'].items():verify_artifact(ROOT/'source'/name,h)
 for a in m['artifacts']:verify_artifact(a['path'],a['SHA256'],a['bytes'])
 done=read(ROOT/'production/TRAINING_COMPLETED.json');assert done['update']==5000 and done['counters']['attempts']==80000
 assert done['best']['update']==m['selected_update']==5000
 assert read(ROOT/'production/RECOVERY_INDEX.json')['latest']==m['final_recovery_checkpoint']
 assert not (ROOT/'production/STOP_RECORD.json').exists()
 return True
def install(P,m=None):
 m=read(CONTRACT) if m is None else m;validate_contract(m)
 lookup=P['bindings'];lookup=getattr(lookup,'original_lookup',lookup)
 def canonical_binding():return normalize(lookup(),m)
 canonical_binding.original_lookup=lookup
 # runpy returns an export mapping separate from function __globals__.
 # Updating BOTH is necessary: the evaluator uses P['bindings'] directly.
 P['bindings']=canonical_binding
 P['production'].__globals__['bindings']=canonical_binding
 assert P['bindings'] is P['production'].__globals__['bindings']
 assert P['Runtime'].__init__.__globals__['bindings'] is canonical_binding
 assert canonical_binding()==m['original_binding']
 return P
