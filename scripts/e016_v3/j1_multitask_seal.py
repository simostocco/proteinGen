import hashlib,json,subprocess,time
from pathlib import Path
R=Path('/mnt/d/Simone/proteinGen_data/e016_j1_bidirectional_multitask_20261019_v1');D=R.parent
names=['FINAL_J1_BIDIRECTIONAL_MULTITASK_REPORT.md','MASKED_SPAN_OBJECTIVE_SPEC.md','MASKED_SPAN_LEAKAGE_AND_GRADIENT_TESTS.json','SEQUENCE_GEOMETRY_CONSISTENCY_REPORT.md','MULTITASK_GRADIENT_BALANCE.json','TRAINING_TOKEN_AND_IDENTITY_COVERAGE.json','SEQUENCE_TRAIN_VALIDATION_METRICS.json','MASKED_RECONSTRUCTION_METRICS.json','GEOMETRY_RETENTION_REPORT.md','FREE_GENERATION_COMPARISON.json','CHECKPOINT_MANIFEST.json','RESOURCE_AND_STORAGE_REPORT.json','EXPERIMENT_DECISIONS.md','CHANGELOG.md']
for n in names:assert (R/n).is_file(),n
restart=json.loads((R/'MULTITASK_GPU_RESTART_TEST.json').read_text());assert restart['status']=='PASS' and all(restart['checks'].values())
manifest=json.loads((R/'CHECKPOINT_MANIFEST.json').read_text());assert len(manifest['checkpoints'])>=3
frozen=json.loads((R/'DECISIONS_FROZEN_BEFORE_BASELINE.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
protected={D/'e016_j1_torsion_stabilization_20261018_v1/replay/checkpoint_00500_full.pt':'9208263300fb9e88519ff0d40e6d9d29a9744b709d9a6181819cd0894b593000',D/'e016_j1_clip_precision_pilot_20261018_v1/runs/J1/model_00500.pt':'9303f0d82a8df044302488340f3ad81535f2fc321360a00bf2892f3a0407aa91',D/'e016_partial_supervision_20261016_v1/runs/J1-FULL/checkpoint_20000.pt':'9e9215d3720ffc73edb0c3e382db25332f91cb26b0c8d38d01668e383b1e3810',D/'e016_j1_full_sequence_20261018_v1/runs/J1/checkpoint_02500.pt':'d4d6351bc42b85655786a984b11b7a6c5e7f253396e815663bd771f7f8f20f70'}
old=json.loads((D/'e016_j1_full_sequence_20261018_v1/ALL_TRAIN_IDENTITY_ADMISSION.json').read_text())
for p,key in [(D/'pdb_supervised_split_50id80cov_20261016_v1/PROPOSAL_MANIFEST.json','split_manifest_sha256'),(D/'pdb_supervised_split_50id80cov_20261016_v1/SEQUENCE_ASSIGNMENTS.parquet','sequence_assignments_sha256'),(D/'e016_partial_supervision_20261016_v1/view/VIEW_FROZEN.json','partial_view_sha256')]:protected[p]=old[key]
protected[D/'e016_j1_aa_head_only_20261019_v1/runs/J1/checkpoint_00500.pt']='4115eff6f1207046c9769cbb04e6036bc61a0a789b6491ccb8a9943702cf0275'
assert sha(R/'configs/FROZEN.json')==frozen['config_sha256']
assert sha(R/'configs/MASKED_EVALUATION.json')==frozen['MLM_validation_panel_sha256']
for p,expected in protected.items():assert sha(p)==expected,('protected mismatch',p)
for c in manifest['checkpoints']:assert sha(c['path'])==c['sha256']
commit_note='Training implementation commit '+manifest['checkpoints'][-1]['source_commit']+'; final assessment/reporting commit '+subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip()+'.'
for doc in ['EXPERIMENT_DECISIONS.md','CHANGELOG.md']:
 p=R/doc;content=p.read_text()
 if commit_note not in content:p.write_text(content+'\n'+commit_note+'\n')
files={};links={}
for p in sorted(R.rglob('*')):
 rel=p.relative_to(R)
 if rel.parts[0] in ['source','runtime'] or p.name in ['PROVENANCE_SHA256.json','PROVENANCE_VERIFICATION.json']:continue
 if p.is_file():
  assert not p.name.endswith('.part'),('unfinished publication',str(p));files[str(rel)]=sha(p)
  if p.is_symlink():links[str(rel)]=str(p.resolve())
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=R/'source',text=True).strip()
z={'version':'e016_j1_bidirectional_multitask_v1','created_time_unix':time.time(),'source_reporting_commit':commit,'source_training_commit':manifest['checkpoints'][-1]['source_commit'],'protected_external_SHA256_verified':{str(k):v for k,v in protected.items()},'evidence_files':files,'external_saved_draw_links':links,'no_TEST_or_RL':True}
(R/'PROVENANCE_SHA256.json').write_text(json.dumps(z,indent=2)+'\n')
assert all(sha(R/k)==h for k,h in files.items())
(R/'PROVENANCE_VERIFICATION.json').write_text(json.dumps({'status':'PASS','evidence_files_verified':len(files),'protected_inputs_verified':len(protected),'checkpoint_states_verified':len(manifest['checkpoints']),'source_worktree_clean':True,'reporting_commit':commit,'PROVENANCE_SHA256':sha(R/'PROVENANCE_SHA256.json'),'no_optimizer_updates_in_publication':True},indent=2)+'\n')
print(json.dumps({'status':'PASS','files':len(files),'protected':len(protected),'reporting_commit':commit}))
