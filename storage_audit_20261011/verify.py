import json,hashlib,time
from pathlib import Path
O=Path(__file__).resolve().parent
R=O.parent
root=R/'e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024**2):h.update(b)
 return h.hexdigest()
before=json.loads((root/'production/checkpoints/LATEST.json').read_text());p=Path(before['path']);digest=sha(p)
after=json.loads((root/'production/checkpoints/LATEST.json').read_text())
assert before==after,'Checkpoint pointer changed during audit hash; repeat verification'
assert digest==before['sha256']
e12=Path('/mnt/d/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization/scientific_corrected_protein_v1')
cert=json.loads((e12/'reports/certification.json').read_text())
assert cert['certified'] and cert['sequences']==20_000_000
shards=list((e12/'manifests').glob('final-shard-*.complete.json'));assert len(shards)==512
fix=json.loads((O/'source_fix.json').read_text());assert sha(Path(fix['path']))==fix['after_sha256']
required=['SUMMARY.md','filesystem_snapshot.json','largest_directories.tsv','largest_files.tsv','growth_snapshot_before.json','growth_snapshot_after.json','active_writers.json','checkpoint_inventory.json','e012_mmseqs_inventory.json','reclaim_candidates.json','protected_artifacts.json','proposed_deletions.json','executed_deletions.json']
assert all((O/n).exists() for n in required)
assert not json.loads((O/'executed_deletions.json').read_text())['deletions']
for n in required:
 if n.endswith('.json'):json.loads((O/n).read_text())
result={'timestamp':time.time(),'e015_latest_sha_verified':digest,'e015_latest_position':before['position'],'e012_sequences_from_certification':cert['sequences'],'shard_certificate_count':len(shards),'source_fix_matches_tested_commit':True,'audit_required_files_present':True,'deletions':0,'e007_pause_sha_previously_verified':'47d58d914400c08f2d90af8d19a7f2751f7575efd4e63a8f137818fb9357693c','corpus_hashes_not_recomputed':'Preserved certification evidence; no unnecessary 20M-corpus rehash'}
(O/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
