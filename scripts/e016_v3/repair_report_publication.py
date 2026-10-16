"""Metadata publication correction only; never executes model or TEST evaluation."""
import hashlib,json,shutil,subprocess
from pathlib import Path
R=Path('/mnt/d/Simone/proteinGen_data/e016_scaled10m_training_20261016_v1')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic(p,value):
 part=p.with_name(p.name+'.part');part.write_text(json.dumps(value,indent=2,allow_nan=False));part.replace(p)
def main():
 a=R/'attempts/json_encoded_reports_v1';old=json.load(open(R/'PROVENANCE_SHA256.json'))
 if (a/'PROVENANCE_SHA256.json').exists():raise RuntimeError('publication correction already attempted; inspect receipt')
 shutil.copy2(R/'PROVENANCE_SHA256.json',a/'PROVENANCE_SHA256.json');changed={}
 for p in sorted(R.glob('*.md')):
  before=sha(p)
  if old['artifact_sha256'].get(p.name)!=before:raise RuntimeError('original report checksum mismatch')
  value=json.loads(p.read_text())
  if not isinstance(value,str):raise TypeError('unexpected report format')
  shutil.copy2(p,a/p.name);part=p.with_name(p.name+'.part');part.write_text(value);part.replace(p)
  changed[p.name]={'original_sha256':before,'plain_markdown_sha256':sha(p)}
 for item in old['selected_checkpoints'].values():
  if sha(item['path'])!=item['sha256']:raise RuntimeError('model checkpoint changed')
 commit=subprocess.check_output(['git','-C',str(R/'source'),'rev-parse','HEAD'],text=True).strip()
 receipt={'scope':'PUBLICATION_ONLY_NO_TRAINING_NO_TEST_REEVALUATION','previous_manifest_sha256':sha(a/'PROVENANCE_SHA256.json'),'training_and_evaluation_source_commit':old['source_commit'],'publication_fix_commit':commit,'changed_reports':changed,'original_bytes_preserved':str(a),'weights_metrics_selections_unchanged':True}
 atomic(R/'PUBLICATION_CORRECTION.json',receipt)
 artifacts={str(p.relative_to(R)):sha(p) for p in sorted(R.rglob('*')) if p.is_file() and '.git' not in p.parts and p.name not in ['PROVENANCE_SHA256.json','FINALIZE.log']}
 old.update({'source_commit':commit,'training_and_evaluation_source_commit':receipt['training_and_evaluation_source_commit'],'artifact_sha256':artifacts,'publication_correction':receipt})
 atomic(R/'PROVENANCE_SHA256.json',old)
 print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
