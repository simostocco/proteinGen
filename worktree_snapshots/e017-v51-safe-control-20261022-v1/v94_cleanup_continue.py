from pathlib import Path
import json,hashlib,os,struct,subprocess,time
C=Path('/mnt/d/Simone/proteinGen_data/e017_v94_fixed_sequence_geometry_rl_20261024_v1/uniref_cleanup_20261024_v1');read=lambda p:json.loads(p.read_text());a=read(C/'DELETE_ALLOWLIST.json');e=a['entries'][0];O=Path(a['root'])/'v2c_optimization';inventory=read(C/'NATIVE_INVENTORY.json');rows={q['path']:q for q in inventory['large_files']}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
 return h.hexdigest()
log=(C/'PREFLIGHT.log').read_text();assert all('SOURCE_CERTIFIED '+n in log for n in ['reservoir.sqlite\n','reservoir.sqlite-wal\n','reservoir.sqlite-shm\n']);assert 'assert fields[6]' in log
protected=[]
# Reuse the completed two sequential SHA passes, with native identity, size
# and unchanged last-write continuity; do not rescan21GiB for a wrong SHM-copy
# assumption. Every SOURCE_CERTIFIED line occurs only after SHA==certificate.
for ref in e['retained_source_refs']:
 p=Path(ref['path']);s=p.stat();row=rows[str(p)];assert s.st_size==ref['bytes']==row['logical'];assert int(s.st_mtime)==row['mtime_unix'];assert s.st_nlink==row['nlink'];protected.append(dict(path=str(p),SHA256=ref['expected_SHA256'],bytes=s.st_size,mtime_ns=s.st_mtime_ns,WSL_inode=s.st_ino,native_id128=row['id'],nlink=s.st_nlink,hash_evidence='two previous completed SHA checks, retained logs/source, unchanged native metadata continuity'))
with (O/'preservation/reservoir.sqlite-shm').open('rb') as f:hdr=f.read(96)
assert hdr[:48]==hdr[48:];fields=struct.unpack('<IIIBBHIIIIIIII',hdr[:48]);export=read(O/'exports/export.complete.json');assert fields[0]==3007000 and fields[3]==1 and fields[5]==4096 and fields[6]==export['statistics']['sqlite_validated_wal_maxframe'] and fields[7]==export['statistics']['logical_page_count']
with (O/'audit/reservoir.sqlite-shm').open('rb') as f:aud=f.read(96)
assert aud[:48]==aud[48:];af=struct.unpack('<IIIBBHIIIIIIII',aud[:48]);source=Path(e['generator']);recipe=source.read_text().replace("(ROOT/'audit/reservoir.sqlite-shm')","(ROOT/'preservation/reservoir.sqlite-shm')");assert recipe!=source.read_text();(C/'source/REGENERATION_RECIPE_NOT_EXECUTED.py').write_text(recipe)
(C/'SHM_SOURCE_RECONCILIATION.json').write_text(json.dumps(dict(resolved=True,canonical_snapshot_header=dict(mxFrame=fields[6],pages=fields[7]),mutable_audit_copy_header=dict(mxFrame=af[6],pages=af[7]),canonical_snapshot_SHA256=protected[2]['SHA256'],audit_copy_SHA256=sha(O/'audit/reservoir.sqlite-shm'),all_copies_preserved=True,recipe_change='Read the certified preservation SHM alongside the preservation DB/WAL, rather than the distinct mutable audit SHM whose header is now zero. Source code copied for provenance only; no export regeneration, screening or corpus materialization executed.',new_bitwise_regeneration_claim=False),indent=2)+'\n')
# Correct classification: this is an authorized obsolete UniRef dataset payload.
# Preserve the complete metadata and regeneration recipe; exact replay loss is
# explicitly accepted by the user and recorded, not hidden behind cache wording.
(C/'DELETE_ALLOWLIST_BEFORE_SOURCE_RECONCILIATION.json').write_text(json.dumps(a,indent=2)+'\n');e['classification']='DISPOSABLE_UNIREF_DATASET';e['role']='obsolete24M-UniRef candidate sequence payload export, derived from certified preserved reservoir; dataset bytes, not scientific certificates or results';e['regeneration_note']='Canonical source bytes verified; patched read-only source selection recipe retained. Fresh bitwise regeneration has NOT been performed; dataset replay availability is lost until separately reconstructed.';(C/'DELETE_ALLOWLIST.json').write_text(json.dumps(a,indent=2)+'\n')
E=Path(a['root']).parent/'e012_large_corpus_warmstart_20m_20261013_v1';evidence=[]
for p in sorted(E.iterdir()):
 if not p.is_file():continue
 s=p.stat();q=dict(path=str(p),device=s.st_dev,inode=s.st_ino,nlink=s.st_nlink,bytes=s.st_size,mtime_ns=s.st_mtime_ns)
 if p.suffix=='.pt':
  q['SHA256']=sha(p);mp=p.with_suffix(p.suffix+'.json');expected=read(mp).get('sha256') if mp.exists() else None
  if expected:assert q['SHA256']==expected
 elif s.st_size<10*2**20:q['SHA256']=sha(p)
 evidence.append(q)
assert any(Path(x['path']).name=='milestone_1000000.pt' for x in evidence)
(C/'PROTECTED_CHECKPOINTS_VERIFIED.json').write_text(json.dumps(evidence,indent=2)+'\n')
V=E.parent.parent/'e017_v93_regional_controllability_20261024_v1';tasks=read(V/'manifests/FUTURE_FIXED_TRAIN_POOL.json');assert tasks['identities']==434 and all('uniref' not in q['source_reference'].lower() for q in tasks['tasks']);deps=subprocess.run(['rg','-l','-F','candidate_payload.bin',str(V/'source/src'),str(V/'source/scripts/e017_v93')],capture_output=True,text=True);assert deps.returncode==1
uses=[];errors=[];needle=str(Path(a['root']).resolve())
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  for fd in (p/'fd').iterdir():
   try:
    path=os.readlink(fd)
    if path.startswith(needle):uses.append(dict(pid=int(p.name),kind='fd',path=path))
   except FileNotFoundError:pass
  for line in (p/'maps').read_text().splitlines():
   if needle in line:uses.append(dict(pid=int(p.name),kind='mapped_file',entry=line))
 except (FileNotFoundError,ProcessLookupError):pass
 except PermissionError as ex:errors.append(str(ex))
(C/'WSL_HANDLE_AUDIT.json').write_text(json.dumps(dict(uses=uses,errors=errors),indent=2)+'\n');assert not uses and not errors,(uses,errors)
(C/'PROTECTED_AND_DEPENDENCY_VERIFICATION.json').write_text(json.dumps(dict(PASS=True,protected_sources=protected,protected_E012_files=evidence,generator=str(source),generator_SHA256=sha(source),manifest_SHA256=sha(O/'exports/export.complete.json'),WSL_open_handles=[],WSL_scan_errors=[],E016_E017_dependency_check='434 PDB-only task sources and current source do not refer to this UniRef export',native_id_reconciliation='FILE_ID_RECONCILIATION.json',SHM_source_reconciliation='SHM_SOURCE_RECONCILIATION.json',no_SQLite_connections=True,no_payload_deleted=True),indent=2)+'\n');print('PREFLIGHT_PASS_CANONICAL_SNAPSHOT',flush=True)
