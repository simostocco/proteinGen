import json,os,hashlib,collections
from pathlib import Path
O=Path(__file__).resolve().parent
d=json.loads((O/'growth_snapshot_before.json').read_text())
groups=collections.defaultdict(list)
seen=set()
for r in d['large_files']+d['checkpoints']:
 if r['bytes']<100*1024**2 or r['path'] in seen:continue
 seen.add(r['path']);groups[r['bytes']].append(r['path'])
result=[];budget=2*1024**3
for size,paths in groups.items():
 if len(paths)<2:continue
 by_inode=collections.defaultdict(list)
 for p in paths:
  s=os.stat(p);by_inode[(s.st_dev,s.st_ino)].append(p)
 representatives=[]
 for inode,ps in by_inode.items():
  if len(ps)>1:result.append({'kind':'same physical file via hardlinks','bytes_per_path':size,'paths':ps,'reclaim_by_unlinking_one_copy':0,'classification':'PROTECTED' if any('reservoir' in x for x in ps) else 'KEEP FOR NOW'})
  representatives.append(ps[0])
 samples=collections.defaultdict(list)
 for p in representatives:
  with open(p,'rb') as f:
   head=f.read(65536);f.seek(max(0,size-65536));tail=f.read(65536)
  samples[hashlib.sha256(head+tail).hexdigest()].append(p)
 for ps in samples.values():
  if len(ps)<2:continue
  if size*len(ps)>budget or any('/corpus/' in p or 'reservoir.sqlite' in p for p in ps):
   result.append({'kind':'same size and first/last 64 KiB; full equality unproven','bytes_per_path':size,'paths':ps,'classification':'PROTECTED' if any('reservoir' in p or '/corpus/' in p for p in ps) else 'UNKNOWN'});continue
  budget-=size*len(ps);hashes=collections.defaultdict(list)
  for p in ps:
   h=hashlib.sha256()
   with open(p,'rb') as f:
    while b:=f.read(8*1024**2):h.update(b)
   hashes[h.hexdigest()].append(p)
  for digest,ps in hashes.items():
   if len(ps)>1:result.append({'kind':'byte-identical SHA256','sha256':digest,'bytes_per_path':size,'paths':ps,'classification':'KEEP FOR NOW','reason':'Equality does not establish disposability; checkpoints and scientific materializations retained'})
(O/'duplicate_inventory.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
