"""Recheck only the existing sealed sidecars actually supporting primary BC links."""
import os,json,hashlib,time
from pathlib import Path
import pyarrow.parquet as pq
R=Path(os.environ['E016_ARTIFACT_ROOT']);S=Path('/mnt/d/Users/Simone Stocco/proteinGen_audits/e006_rich_geometry_sidecars_v2');protocol=json.load(open(S/'protocol.json'));expected={str(S/r['path']):r['sha256'] for r in protocol['shards']};paths=set();out=R/'mapping/SEALED_CONNECTIVITY_DEPENDENCIES.json';assert not out.exists();start=time.monotonic()
for b in pq.ParquetFile(R/'release/intake_v1/chain_observations.parquet').iter_batches(batch_size=512,columns=['BC_eligible','metadata'],use_threads=False):
 for r in b.to_pylist():
  if not r['BC_eligible']:continue
  m=json.loads(r['metadata'])
  if m.get('connectivity_ref'):paths.add(m['connectivity_ref'].split('#')[0])
rows=[]
for name in sorted(paths):
 assert name in expected;path=Path(name);before=path.stat();h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 after=path.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns);assert h.hexdigest()==expected[name],'SEALED_SIDECAR_HASH_DIVERGED'
 rows.append({'path':name,'sha256':h.hexdigest(),'bytes':before.st_size,'source_seal':'E006 protocol.shards + prior v1 audit; freshly rechecked selected connectivity dependencies'})
out.write_text(json.dumps({'status':'PASS','files':rows,'files_count':len(rows),'bytes_sequentially_checked':sum(r['bytes'] for r in rows),'elapsed_seconds':time.monotonic()-start,'reuse_scope':'C-N connectivity and atom-mask evidence only; historical sampling weights and fitted normalizers not reused'},indent=2));print('PASS sidecar dependencies',len(rows),time.monotonic()-start,flush=True)
