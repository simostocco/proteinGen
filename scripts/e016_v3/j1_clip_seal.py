"""Verify protected history and seal stopped short-pilot artifacts; no model execution."""
import hashlib,json,os,subprocess,time
from pathlib import Path
R=Path('/mnt/d/Simone/proteinGen_data/e016_j1_clip_precision_pilot_20261018_v1')
OLD=Path('/mnt/d/Simone/proteinGen_data/e016_j1_sequence_first_20261018_v1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 if not (R/'FINAL_ASSESSMENT_COMPLETE.json').exists() or not (R/'MONITOR_STOP').exists():raise RuntimeError('all writers must stop before sealing')
 for pp in Path('/proc').glob('[0-9]*/cmdline'):
  try:cmd=pp.read_bytes().replace(b'\0',b' ').decode()
  except OSError:continue
  if str(R/'source/scripts/e016_v3/j1_clip_pilot.py') in cmd and 'python' in cmd:raise RuntimeError('training controller still active')
 if subprocess.check_output(['git','status','--porcelain'],cwd=R/'source',text=True):raise RuntimeError('uncommitted source')
 old=json.load(open(OLD/'PROVENANCE_SHA256.json'));ver=json.load(open(OLD/'PROVENANCE_VERIFICATION.json'))
 if sha(OLD/'PROVENANCE_SHA256.json')!=ver['manifest_sha256'] or ver['manifest_sha256']!='041b9f9fe6048efcfd6ce9cb0a15db6d30a81ff23d9bb4e2badbbc2e7d9477a3':raise RuntimeError('historical provenance changed')
 for p,h in old['protected_checkpoint_sha256'].items():
  if sha(p)!=h:raise RuntimeError('protected checkpoint changed '+p)
 inputs=old['input_manifests_and_reference_evidence']
 for p,entry in inputs.items():
  if sha(p)!=entry['sha256']:raise RuntimeError('source/data/reference manifest changed '+p)
 # Verify all sealed prior reports, checkpoints and code without writing that lineage.
 errors=[]
 for rel,v in old['outputs'].items():
  if sha(OLD/rel)!=v['sha256']:errors.append(rel)
 for rel,h in old['tracked_source_sha256'].items():
  if sha(OLD/'source'/rel)!=h:errors.append('source/'+rel)
 if errors:raise RuntimeError('historical integrity contradictions '+str(errors))
 outputs={}
 for root,dirs,files in os.walk(R):
  if Path(root)==R:dirs[:]=[d for d in dirs if d not in ['runtime','source']]
  for name in files:
   p=Path(root)/name;rel=str(p.relative_to(R))
   if rel in ['PROVENANCE_SHA256.json','PROVENANCE_VERIFICATION.json']:continue
   if p.is_symlink():raise RuntimeError('unexpected output symlink '+rel)
   outputs[rel]={'sha256':sha(p),'bytes':p.stat().st_size}
 tracked=subprocess.check_output(['git','ls-files','-z'],cwd=R/'source').decode().split('\0');sources={n:sha(R/'source'/n) for n in tracked if n and (R/'source'/n).is_file()}
 result={'version':'J1_SHORT_CLIP_PRECISION_V1','time':time.time(),'branch':subprocess.check_output(['git','branch','--show-current'],cwd=R/'source',text=True).strip(),'reporting_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip(),'training_implementation_commits':[json.load(open(R/'STORAGE_STAGING_AMENDMENT.json'))['old_source_commit'],json.load(open(R/'STORAGE_STAGING_AMENDMENT.json'))['successor_source_commit']],'source_clean':True,'frozen_config_sha256':sha(R/'configs/FROZEN.json'),'corrected_generation_protocol_sha256':sha(R/'configs/FINAL_ASSESSMENT_PROTOCOL.json'),'historical_manifest_sha256':sha(OLD/'PROVENANCE_SHA256.json'),'historical_output_files_verified':len(old['outputs']),'historical_source_files_verified':len(old['tracked_source_sha256']),'protected_checkpoint_sha256':old['protected_checkpoint_sha256'],'input_source_data_and_reference_manifest_sha256':inputs,'tracked_source_sha256':sources,'outputs':outputs,'excluded':['runtime caches','self manifest and final verification receipt'],'TEST':'NOT_EVALUATED','normalization_changes':False,'training_stopped':True,'automatic_extension':False}
 path=R/'PROVENANCE_SHA256.json'
 if path.exists():raise RuntimeError('preserve existing final provenance')
 path.write_text(json.dumps(result,indent=2)+'\n');errors=[]
 for rel,v in outputs.items():
  if sha(R/rel)!=v['sha256']:errors.append(rel)
 final={'status':'PASS' if not errors else 'FAIL','files_verified':len(outputs),'source_files_verified':len(sources),'protected_checkpoints_verified':5,'prior_sealed_artifacts_and_source_verified':len(old['outputs'])+len(old['tracked_source_sha256']),'errors':errors,'manifest_sha256':sha(path),'time':time.time()}
 (R/'PROVENANCE_VERIFICATION.json').write_text(json.dumps(final,indent=2)+'\n')
 if errors:raise RuntimeError(str(errors))
 print(json.dumps(final),flush=True)
if __name__=='__main__':main()
