from pathlib import Path
import os,json,hashlib,stat,time,subprocess
R=Path('/mnt/d/Simone/proteinGen_data/e017_v94_fixed_sequence_geometry_rl_20261024_v1/uniref_cleanup_20261024_v1');A=json.loads((R/'DELETE_ALLOWLIST.json').read_text());entry=A['entries'][0]
def sha(p):
 h=hashlib.sha256();n=0;last=time.monotonic()
 with Path(p).open('rb') as f:
  while True:
   b=f.read(4*2**20)
   if not b:break
   h.update(b);n+=len(b)
   if time.monotonic()-last>20:print('VERIFY_PROTECTED',Path(p).name,n,flush=True);last=time.monotonic()
 return h.hexdigest()
protected=[]
for ref in entry['retained_source_refs']:
 p=Path(ref['path']);before=p.stat();assert before.st_size==ref['bytes'];actual=sha(p);assert actual==ref['expected_SHA256'],('source mismatch',str(p));after=p.stat();assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_nlink)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_nlink);protected.append(dict(path=str(p),SHA256=actual,device=after.st_dev,inode=after.st_ino,nlink=after.st_nlink,bytes=after.st_size,mtime_ns=after.st_mtime_ns));print('SOURCE_CERTIFIED',p.name,flush=True)
# Verify exact generator and immutable source bindings; no regeneration run.
O=Path(A['root'])/'v2c_optimization';script=Path(entry['generator']);text=script.read_text();assert "Pager()" in text and "candidate_payload.bin.partial" in text and "ROOT/'audit" in text;assert all((O/'audit'/name).stat().st_ino==(O/'preservation'/name).stat().st_ino for name in ['reservoir.sqlite','reservoir.sqlite-wal']);assert json.loads((R/'FILE_ID_RECONCILIATION.json').read_text())['resolved']
import struct
with (O/'audit/reservoir.sqlite-shm').open('rb') as f:header=f.read(96)
assert header[:48]==header[48:];fields=struct.unpack('<IIIBBHIIIIIIII',header[:48]);export=json.loads((O/'exports/export.complete.json').read_text());assert fields[6]==export['statistics']['sqlite_validated_wal_maxframe'] and fields[7]==export['statistics']['logical_page_count'];protected.append(dict(path=str(O/'audit/reservoir.sqlite-shm'),SHA256=sha(O/'audit/reservoir.sqlite-shm'),role='distinct protected committed-header copy',mxFrame=fields[6],pages=fields[7]))
E=Path(A['root']).parent/'e012_large_corpus_warmstart_20m_20261013_v1';evidence=[]
for p in sorted(E.iterdir()):
 if p.is_file():
  s=p.stat();item=dict(path=str(p),device=s.st_dev,inode=s.st_ino,nlink=s.st_nlink,bytes=s.st_size,mtime_ns=s.st_mtime_ns)
  if p.suffix=='.pt':
   actual=sha(p);meta=p.with_suffix(p.suffix+'.json');expected=json.loads(meta.read_text()).get('sha256') if meta.exists() else None
   if expected:assert actual==expected
   item['SHA256']=actual
  elif p.stat().st_size<10*2**20:item['SHA256']=sha(p)
  evidence.append(item)
assert any(Path(q['path']).name=='milestone_1000000.pt' for q in evidence)
# Current E016/E017 manifests/sources have no reference to this obsolete export.
V=E.parent.parent/'e017_v93_regional_controllability_20261024_v1';task=json.loads((V/'manifests/FUTURE_FIXED_TRAIN_POOL.json').read_text());assert task['identities']==434 and all('uniref' not in q['source_reference'].lower() for q in task['tasks'])
refs=subprocess.run(['rg','-l','-F','candidate_payload.bin',str(V/'source/src'),str(V/'source/scripts/e017_v93')],capture_output=True,text=True);assert refs.returncode==1,refs.stdout
# Actual host /proc scan: include fd and mapped files, stop on access errors.
uses=[];errors=[];pidself=os.getpid();needle=str(Path(A['root']).resolve());target=Path(entry['path']);s=target.stat();identity=(s.st_dev,s.st_ino)
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==pidself:continue
 try:
  for fd in (proc/'fd').iterdir():
   try:
    linked=os.readlink(fd)
    if linked.startswith(needle):uses.append(dict(pid=int(proc.name),kind='fd',fd=fd.name,path=linked))
   except FileNotFoundError:pass
  maps=proc/'maps'
  for line in maps.read_text().splitlines():
   if needle in line:uses.append(dict(pid=int(proc.name),kind='mmap',entry=line))
 except (FileNotFoundError,ProcessLookupError):pass
 except PermissionError as e:errors.append(str(e))
assert not errors,errors;assert not uses,uses
(R/'PROTECTED_AND_DEPENDENCY_VERIFICATION.json').write_text(json.dumps(dict(PASS=True,protected_sources=protected,protected_E012_files=evidence,generator_SHA256=sha(script),generator=str(script),manifest_SHA256=sha(O/'exports/export.complete.json'),source_state_SHA256=sha(O/'audit/authoritative_state.json'),WSL_open_handles=[],WSL_scan_errors=[],E016_E017_dependency_check='PDB-only434-task manifest and current source do not refer to obsolete UniRef export',no_SQLite_connections=True,no_payload_deleted=True,corpus_and_source_already_absent=A['source_and_corpus_absent']),indent=2)+'\n');print('PREFLIGHT_PASS',flush=True)
