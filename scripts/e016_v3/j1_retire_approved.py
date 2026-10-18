import json,os,stat,hashlib,shutil,time
from pathlib import Path
R=Path(__file__).resolve().parents[3];D=R.parent;OLD=D/'e016_j1_torsion_diagnostics_storage_20261018_v1';M=OLD/'APPROVAL_ONLY_STORAGE_CANDIDATES.json'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
 return h.hexdigest()
def write(v):
 p=R/'APPROVED_RETIREMENT_MANIFEST.json';q=p.with_suffix('.part');q.write_text(json.dumps(v,indent=2));os.replace(q,p)
source=json.load(open(M));files=[v for g in source['prefilters'].values() for v in g['files']]
assert len(files)==10 and sum(v['allocated_bytes'] for v in files)==9498710016
# Mandatory downstream scientific inputs are independent of these old intermediates.
protected=[D/'pdb_supervised_split_50id80cov_20261016_v1'/n for n in ['PROPOSAL_MANIFEST.json','SEQUENCE_ASSIGNMENTS.parquet','JOINT_OBSERVATION_ASSIGNMENTS.parquet']]+[D/'e016_gpu_unified_pdb_20261015_v1/homology/full_edges.tsv',D/'e016_gpu_unified_pdb_20261015_v1/homology/exact_source_union_forest.jsonl',D/'e016_j1_clip_precision_pilot_20261018_v1/runs/J1/model_00500.pt',D/'e016_j1_sequence_first_20261018_v1/runs/J1/checkpoint_05000.pt']
assert all(p.is_file() for p in protected)
protect={str(p):sha(p) for p in protected};paths={v['path'] for v in files};openpaths=set()
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  for p in (proc/'fd').iterdir():
   try:openpaths.add(os.readlink(p))
   except OSError:pass
  for line in (proc/'maps').read_text(errors='replace').splitlines():
   if '/mnt/d/' in line:openpaths.add(line.split(maxsplit=5)[-1])
 except (OSError,PermissionError):pass
# Accepted proposal manifest and J1 data modules cannot reference the retired inputs.
for p in [protected[0],*list((R/'source/src/protein_joint_sac/data').glob('*.py'))]:
 text=p.read_text()
 if any(x in text for x in paths):raise RuntimeError('active dataset dependency '+str(p))
receipt={'authorization':'user explicit retirement of exactly10 candidates, 2026-10-18','candidate_manifest':str(M),'candidate_manifest_sha256':sha(M),'before_free_bytes':shutil.disk_usage(R).free,'protected_sha256_before':protect,'files':[],'retired':[],'preserved':[],'consequence':'historical candidate-prefilter audit/restart material retired; identical durable copies not established; old manifests remain unchanged and may reference now-retired files; recreation from retained sequences/tool commands requires compute/scratch and bitwise reproducibility is not established'}
refs=[]
H=D/'e016_gpu_unified_pdb_20261015_v1/homology'
for name in ['full_search.log','independent_all_prefilter.log','full_search.RECEIPT.json','full_search.DONE.json','independent_all_pool_v1/INPUT.json','independent_all_pool_v1/prefilter.RECEIPT.json']:
 p=H/name
 if p.is_file():refs.append({'path':str(p),'sha256':sha(p)})
receipt['retained_original_provenance_references']=refs
for v in files:
 p=Path(v['path']);s=p.lstat();reason=None
 if p.is_symlink() or p.resolve()!=p or not stat.S_ISREG(s.st_mode):reason='real location/type mismatch'
 if s.st_ino!=v['inode'] or s.st_nlink!=v['nlink'] or s.st_size!=v['logical_bytes'] or s.st_blocks*512!=v['allocated_bytes'] or s.st_uid!=os.getuid():reason='size/inode/owner/link changed'
 if str(p) in openpaths:reason='active open/mapped dependency'
 row={**v,'realpath':str(p.resolve()),'device':s.st_dev,'uid':s.st_uid,'gid':s.st_gid,'mtime_ns':s.st_mtime_ns,'sha256':sha(p),'approved_reason':'explicitly approved retirement of ten historical prefilter files only','dependency_check':reason or 'PASS'};receipt['files'].append(row);write(receipt)
 if reason:receipt['preserved'].append(row);write(receipt);continue
 # Record full hash and identity durably BEFORE unlink.
 with (R/'RETIREMENT_JOURNAL.jsonl').open('a') as f:f.write(json.dumps({'before_unlink':row})+'\n');f.flush();os.fsync(f.fileno())
 now=p.lstat()
 if (now.st_ino,now.st_size,now.st_mtime_ns)!=(s.st_ino,s.st_size,s.st_mtime_ns):raise RuntimeError('retirement candidate changed')
 p.unlink();receipt['retired'].append(row);receipt['after_free_bytes']=shutil.disk_usage(R).free;write(receipt);print(json.dumps({'retired':str(p),'free_bytes':receipt['after_free_bytes']}),flush=True)
receipt['protected_sha256_after']={str(p):sha(p) for p in protected}
assert receipt['protected_sha256_before']==receipt['protected_sha256_after']
receipt['allocated_bytes_retired']=sum(v['allocated_bytes'] for v in receipt['retired']);receipt['after_free_bytes']=shutil.disk_usage(R).free;receipt['net_recovered_bytes']=receipt['after_free_bytes']-receipt['before_free_bytes'];receipt['status']='COMPLETE';write(receipt)
(R/'APPROVED_PREFILTER_RETIREMENT_REPORT.md').write_text(f"# Approved retirement\n\nRetired {len(receipt['retired'])}/10 specifically approved files; allocated {receipt['allocated_bytes_retired']} bytes. Before free {receipt['before_free_bytes']}, after {receipt['after_free_bytes']}, net {receipt['net_recovered_bytes']} bytes. Every exact path, SHA256, size, inode, ownership, reference and consequence is recorded in APPROVED_RETIREMENT_MANIFEST.json. No directories or other artifacts deleted. Protected downstream assignments, model inputs/checkpoints and final edges retain identical hashes. Old historical manifests have not been rewritten: their ten entries now refer to explicitly retired intermediates. No identical durable copy or bitwise re-creation guarantee is claimed.\n")
print('RETIREMENT_COMPLETE',receipt['allocated_bytes_retired'],flush=True)
