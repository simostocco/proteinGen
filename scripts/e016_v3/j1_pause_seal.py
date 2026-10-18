"""Seal this user-paused evidence root after all writers stop. No scientific processing."""
import hashlib,json,os,subprocess,time
from pathlib import Path
R=Path('/mnt/d/Simone/proteinGen_data/e016_j1_sequence_first_20261018_v1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.load(open(p))
def main():
 if not (R/'PAUSED_PUBLICATION_COMPLETE.json').exists() or not (R/'MONITOR_STOP').exists():raise RuntimeError('publication/monitor must finish before sealing')
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:cmd=p.read_bytes().replace(b'\0',b' ').decode()
  except OSError:continue
  if str(R/'source/scripts/e016_v3/j1_sequence_first.py') in cmd and 'python' in cmd and ' -c ' not in cmd:raise RuntimeError('training worker still active')
 dirty=subprocess.check_output(['git','status','--porcelain'],cwd=R/'source',text=True)
 if dirty:raise RuntimeError('source worktree not clean')
 version=read(R/'PAUSED_CHECKPOINT_VERIFICATION.json');reviews=read(R/'MILESTONE_5000_7500_COMPARISON.json')
 protected={v['checkpoint']:v['sha256'] for v in reviews};protected[version['checkpoint']]=version['sha256']
 protected.update({
 '/mnt/d/Simone/proteinGen_data/e016_partial_supervision_20261016_v1/runs/J1-FULL/checkpoint_20000.pt':'9e9215d3720ffc73edb0c3e382db25332f91cb26b0c8d38d01668e383b1e3810',
 '/mnt/d/Simone/proteinGen_data/e016_j1_global_pretraining_20261017_v1/runs/J1/checkpoint_30000.pt':'c37d2bd7eb9136f12bc4d9ea4128403fc72a6a323937bccfbd68a582c1fd78b1'})
 for p,h in protected.items():
  if sha(p)!=h:raise RuntimeError('protected checkpoint changed '+p)
 sources={}
 tracked=subprocess.check_output(['git','ls-files','-z'],cwd=R/'source').decode().split('\0')
 for rel in tracked:
  if rel and (R/'source'/rel).is_file():sources[rel]=sha(R/'source'/rel)
 outputs={}
 excluded={'PROVENANCE_SHA256.json','PROVENANCE_VERIFICATION.json'}
 for root,dirs,files in os.walk(R):
  if Path(root)==R:dirs[:]=[d for d in dirs if d not in ['runtime','source']]
  for name in files:
   p=Path(root)/name;rel=str(p.relative_to(R))
   if rel in excluded:continue
   if p.is_symlink():raise RuntimeError('unexpected scientific output symlink '+rel)
   outputs[rel]={'sha256':sha(p),'bytes':p.stat().st_size}
 inputs={}
 candidates=[
 Path('/mnt/d/Simone/proteinGen_data/pdb_supervised_split_50id80cov_20261016_v1/PROPOSAL_MANIFEST.json'),
 Path('/mnt/d/Simone/proteinGen_data/e016_partial_supervision_20261016_v1/view/VIEW_FROZEN.json'),
 Path('/mnt/d/Simone/proteinGen_data/e016_partial_supervision_20261016_v1/view/SOURCE_HASHES.json'),
 Path('/mnt/d/Simone/proteinGen_data/e016_real_pdb_pilots_20261016_v1/priors/GEOMETRY_PRIORS.json'),
 Path('/mnt/d/Simone/proteinGen_data/e016_real_pdb_pilots_20261016_v1/priors/SEQUENCE_STATS.json'),
 Path('/mnt/d/Simone/proteinGen_data/e016_real_pdb_pilots_20261016_v1/PRIOR_MEMBERSHIP_VERIFICATION.json'),
 Path('/mnt/d/Simone/proteinGen_data/e016_partial_supervision_20261016_v1/runs/J1-FULL/SELECTED_GENERATION.json'),
 Path('/mnt/d/Simone/proteinGen_data/e016_partial_supervision_20261016_v1/NATIVE_VS_GENERATED_GEOMETRY.json'),
 Path('/mnt/d/Simone/proteinGen_data/e016_j1_global_pretraining_20261017_v1/LOCAL_GEOMETRY_VALIDATION.json'),
 Path('/mnt/d/Simone/proteinGen_data/e016_j1_global_pretraining_20261017_v1/NATIVE_VS_GENERATED_GLOBAL_GEOMETRY.json')]
 for p in candidates:inputs[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
 manifest={'version':'E016_IMMEDIATE_USER_PAUSE_5000_7500_V1','timestamp':time.time(),'branch':subprocess.check_output(['git','branch','--show-current'],cwd=R/'source',text=True).strip(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip(),'source_clean':True,'active_training_implementation_commit':'df1543e0c5e2c48fa2da9e4796db4b4e46124802','active_training_entrypoint_sha256':version['lineage']['entrypoint_sha256'],'frozen_config_sha256':sha(R/'configs/FROZEN.json'),'package_sha256_at_training':version['lineage']['package_sha256'],'comparison_updates':[5000,7500],'recovery_only_update':7744,'protected_checkpoint_sha256':protected,'input_manifests_and_reference_evidence':inputs,'tracked_source_sha256':sources,'outputs':outputs,'excluded':['runtime caches','self-referential manifest','post-manifest verification receipt'],'coordinate_sources':'Existing read-only mappings/sources; loader verified hashes. Source manifest links retained, no copied archive.','TEST':'NOT_EVALUATED','post_stop_optimizer_updates':0,'normalization_changes':False}
 dest=R/'PROVENANCE_SHA256.json'
 if dest.exists():raise RuntimeError('immutable provenance collision')
 dest.write_text(json.dumps(manifest,indent=2)+'\n')
 errors=[]
 for rel,v in outputs.items():
  if sha(R/rel)!=v['sha256']:errors.append(rel)
 for p,h in protected.items():
  if sha(p)!=h:errors.append(p)
 for rel,h in sources.items():
  if sha(R/'source'/rel)!=h:errors.append('source/'+rel)
 result={'status':'PASS' if not errors else 'FAIL','output_files_verified':len(outputs),'source_files_verified':len(sources),'protected_checkpoints_verified':len(protected),'errors':errors,'manifest_sha256':sha(dest),'timestamp':time.time()}
 (R/'PROVENANCE_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
 if errors:raise RuntimeError(str(errors))
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
