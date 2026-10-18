"""Project-only cache cleanup with per-file receipts and conservative reference checks."""
import os,json,stat,time,shutil,subprocess,hashlib,re
from pathlib import Path
R=Path(__file__).resolve().parents[3];D=R.parent;REPO=Path('/mnt/d/Simone/proteinGen')
def write(n,x):
 p=R/n;p.write_text(json.dumps(x,indent=2));return x
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def walkfiles(root):
 for base,ds,fs in os.walk(root,followlinks=False):
  ds[:]=[d for d in ds if not (Path(base)/d).is_symlink()]
  for f in fs:
   p=Path(base)/f
   if p.is_file() and not p.is_symlink():yield p

def main(delete=False):
 before=shutil.disk_usage(D).free
 trees=[Path(s.split('worktree ',1)[1]) for s in subprocess.check_output(['git','worktree','list','--porcelain'],cwd=REPO,text=True).splitlines() if s.startswith('worktree ')]
 # Do not touch /home worktrees. Only proteinGen D: scopes explicitly requested.
 trees=[p for p in trees if str(p).startswith('/mnt/d/Simone/proteinGen') and p.exists()]
 roots=[]
 for tree in trees:
  for base,ds,fs in os.walk(tree,followlinks=False):
   kept=[]
   for d in ds:
    p=Path(base)/d
    if p.is_symlink() or d=='.git':continue
    if d in ['__pycache__','.pytest_cache']:roots.append(p);continue
    if d in ['runs','data','datasets','checkpoints','artifacts','qualification_artifacts','node_modules','.venv']:continue
    kept.append(d)
   ds[:]=kept
 for exp in D.iterdir():
  if not exp.is_dir() or exp.is_symlink() or exp==R:continue
  cache=exp/'runtime/cache'
  if cache.exists():
   for p in cache.iterdir():
    if p.is_dir() and not p.is_symlink() and (p.name in ['pyc','cuda','inductor','triton','jax'] or p.name.startswith('pytest')):roots.append(p)
  for p in [exp/'runtime/pytest',exp/'runtime/.pytest_cache',exp/'runtime/pytest_cache']:
   # Only pytest cache metadata, not fixture/restart outputs.
   if p.exists() and (p/'v/cache').exists():roots.append(p)
 # Report/manifest text index: shallow metadata directories, avoiding raw datasets/coordinate dumps.
 texts=[];scan_roots=[D,REPO]+[p for p in D.iterdir() if p.is_dir() and not p.is_symlink() and p!=R]
 for root in scan_roots:
  for base,ds,fs in os.walk(root,followlinks=False):
   rel=Path(base).relative_to(root);depth=len(rel.parts)
   ds[:]=[d for d in ds if not (Path(base)/d).is_symlink() and d in ['configs','reports','priors','view','manifests','mapping','homology','qualification_artifacts'] and depth<3]
   for f in fs:
    p=Path(base)/f
    if p.suffix.lower() in ['.json','.jsonl','.md'] and p.stat().st_size<30*2**20:
     try:texts.append((str(p),p.read_text(errors='replace')))
     except OSError:pass
 # Individual file paths checked against open descriptors/mmap and exact manifest/report strings.
 active=[];openpaths=set()
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   for f in (proc/'fd').iterdir():
    try:openpaths.add(os.readlink(f))
    except OSError:pass
   for line in (proc/'maps').read_text(errors='replace').splitlines():
    if '/mnt/d/' in line:openpaths.add(line.split(maxsplit=5)[-1])
   if 'python' in cmd:active.append({'pid':int(proc.name),'cmd':cmd[:500]})
  except (OSError,PermissionError):pass
 print('METADATA_DOCUMENTS',len(texts),'CACHE_ROOTS',len(roots),flush=True)
 tokens=[];published_hashes=set();document_count=len(texts)
 for report,txt in texts:
  published_hashes.update(re.findall(r'\b[0-9a-f]{64}\b',txt))
  tokens.extend((report,t) for t in re.findall(r'[^\s"`<>]*?(?:__pycache__|cache/|\.pytest_cache)[^\s"`<>]*',txt))
 del texts
 references_by_token={}
 for report,tok in tokens:
  tok=tok.strip("'(),[]{};:")
  references_by_token.setdefault(tok,[]).append(report)
 del tokens
 tracked_paths=set()
 for tree in trees:
  tracked_paths.update(str(tree/p) for p in subprocess.check_output(['git','ls-files'],cwd=tree,text=True).splitlines())
 candidates=[];seen=set();protected=[]
 for root in sorted(set(roots)):
  for p in walkfiles(root):
   if str(p) in seen:continue
   seen.add(str(p));s=p.lstat();rel=str(p.relative_to(D)) if p.is_relative_to(D) else str(p.relative_to(REPO));references=[]
   # A full pathname or relative cache pathname is evidence linkage. Generic cache-directory mentions are not artifact membership.
   variants={str(p),rel}
   for ancestor in p.parents:
    if ancestor==D.parent:break
    variants.add(str(p.relative_to(ancestor)))
   for token in variants:references.extend(references_by_token.get(token,[]))
   digest=sha(p)
   if digest in published_hashes:references.append('content SHA256 occurs in published metadata')
   tracked=str(p) in tracked_paths
   entry={'path':str(p),'allocated_bytes':s.st_blocks*512,'logical_bytes':s.st_size,'device':s.st_dev,'inode':s.st_ino,'nlink':s.st_nlink,'mtime_ns':s.st_mtime_ns,'sha256':digest,'cache_root':str(root),'referenced_by':references,'tracked':tracked,'open':str(p) in openpaths,'reason':'regenerable compiler/Python/pytest cache; not scientific source or weights'}
   if references or tracked or entry['open'] or s.st_nlink>1:entry['classification']='PRESERVE_REFERENCE_OR_ACTIVE_OR_HARDLINK';protected.append(entry)
   else:entry['classification']='A';candidates.append(entry)
 receipt={'time':time.time(),'free_before_bytes':before,'candidate_files':candidates,'preserved_cache_files':protected,'metadata_documents_checked':document_count,'metadata_check_note':'historical report/manifest paths and SHA256 indexed; current cleanup receipts excluded because they intentionally authorize disposal, not protect caches; metadata_documents_count in log','active_python_processes':active,'scope':'D: proteinGen worktrees and experiment runtime caches only; no symlink traversal; no scientific fixture temp dirs, no HF/model/data caches; no ClassB deletions','deletion_authorized_by_user':True,'deleted':[]}
 write('SAFE_CLEANUP_PLAN.json',receipt)
 if delete:
  for e in candidates:
   p=Path(e['path']);s=p.lstat()
   if not stat.S_ISREG(s.st_mode) or s.st_ino!=e['inode'] or s.st_mtime_ns!=e['mtime_ns'] or sha(p)!=e['sha256']:raise RuntimeError('candidate changed before deletion '+str(p))
   p.unlink();receipt['deleted'].append(e)
   with (R/'CLEANUP_DELETION_JOURNAL.jsonl').open('a') as journal:journal.write(json.dumps(e)+'\n');journal.flush();os.fsync(journal.fileno())
  receipt['free_after_bytes']=shutil.disk_usage(D).free;receipt['sum_unlinked_allocated_bytes']=sum(e['allocated_bytes'] for e in receipt['deleted']);receipt['measured_volume_free_change_bytes']=receipt['free_after_bytes']-before
  write('SAFE_CLEANUP_MANIFEST.json',receipt)
 print(json.dumps({k:v for k,v in receipt.items() if k not in ['candidate_files','preserved_cache_files','deleted','active_python_processes']}),flush=True)
 # Allocated sizes, hardlink dedup within one du traversal; no symlinks followed.
 for name,cmd in [('DATA_TOP_USAGE.txt',['du','-x','-B1','--max-depth=1',str(D)]),('REPO_TOP_USAGE.txt',['du','-x','-B1','--max-depth=1',str(REPO)])]:
  p=subprocess.run(cmd,capture_output=True,text=True);(R/name).write_text(p.stdout);(R/(name+'.stderr')).write_text(p.stderr)
 if receipt.get('deleted'):print('DELETED_FILES',len(receipt['deleted']),flush=True)
if __name__=='__main__':
 import sys
 main('--delete-class-a' in sys.argv)
