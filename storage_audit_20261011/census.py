import os, sys, time, json, stat, heapq
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOTS=[Path('/mnt/d/Simone/proteinGen'),Path('/mnt/d/Simone/proteinGen_data')]
def owner(p):
 s=str(p).lower()
 for a in ['e012','e015','e007','e014']:
  if a in s or (a=='e012' and 'sequence_foundation' in s): return a.upper()
 return 'shared dataset' if 'proteinGen_data' in str(p) else 'source/worktree' if any(x in s for x in ['worktree','/.git','/src/']) else 'unknown'
dirs={}; files=[]; errors=[]; checkpoints=[]; special=[]
start=time.time()
for root in ROOTS:
 dev=root.stat().st_dev
 for base, children, names in os.walk(root,topdown=False,followlinks=False):
  p=Path(base)
  if p==OUT: continue
  try:
   if p.stat().st_dev!=dev: continue
  except OSError: continue
  size=allocated=count=direct=0
  for n in names:
   f=p/n
   try:
    s=f.lstat()
    if not stat.S_ISREG(s.st_mode): continue
   except OSError as e: errors.append(str(e));continue
   size+=s.st_size;allocated+=s.st_blocks*512;count+=1;direct+=1
   r={'path':str(f),'bytes':s.st_size,'mtime':s.st_mtime,'owner':owner(f)}
   if s.st_size>=250*1024**2: files.append(r)
   if f.suffix in ['.pt','.pth','.ckpt','.safetensors']: checkpoints.append(r)
   if any(x in n.lower() for x in ['sqlite','-wal','-shm','.tmp']) or f.suffix=='.log': special.append(r)
  for c in children:
   r=dirs.get(str(p/c),{})
   size+=r.get('bytes',0);allocated+=r.get('allocated',0);count+=r.get('files',0)
  dirs[str(p)]={'bytes':size,'allocated':allocated,'files':count,'direct_files':direct,'owner':owner(p)}
end=time.time();v=os.statvfs('/mnt/d')
data={'start':start,'end':end,'free':v.f_bavail*v.f_frsize,'capacity':v.f_blocks*v.f_frsize,'directories':dirs,'large_files':files,'checkpoints':checkpoints,'special':special,'errors':errors}
target=sys.argv[1]
(OUT/target).write_text(json.dumps(data,separators=(',',':')))
if target=='growth_snapshot_before.json':
 for name,rows in [('largest_directories.tsv',[(p,r) for p,r in dirs.items()]),('largest_files.tsv',[(r['path'],r) for r in files])]:
  rows.sort(key=lambda x:x[1]['bytes'],reverse=True)
  with (OUT/name).open('w') as h:
   h.write('bytes\towner\tfiles\tpath\n')
   for p,r in rows:h.write(f"{r['bytes']}\t{r['owner']}\t{r.get('files',1)}\t{p}\n")
print(json.dumps({'target':target,'seconds':end-start,'roots':{str(r):dirs.get(str(r)) for r in ROOTS},'free':data['free'],'top_dirs':sorted(dirs.items(),key=lambda x:x[1]['bytes'],reverse=True)[:20],'top_files':sorted(files,key=lambda x:x['bytes'],reverse=True)[:15]},indent=2))
