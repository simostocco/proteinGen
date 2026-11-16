import os, stat, json, heapq, time, shutil
from pathlib import Path

OUT=Path('/mnt/d/Simone/proteinGen_data/thesis_storage_inventory_v1')
ROOTS=['/mnt/d/Simone/proteinGen','/mnt/d/Users/Simone Stocco/proteinGen_audits','/mnt/d/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2','/mnt/d/Simone/proteinGen_data/sequence_foundation/e012_large_corpus_warmstart_20m_20261013_v1','/mnt/d/Simone/proteinGen_data/thesis_pdb_sequence_audit_v1','/mnt/d/Simone/proteinGen_data/thesis_pdb_sequence_audit_v2']
dirs={}; top=[]; links={}; errors=[]; roots=[]; count=0; started=time.time(); last=started
before=shutil.disk_usage('/mnt/d')._asdict()
for root in ROOTS:
    print('SCAN '+root,flush=True)
    if not os.path.isdir(root):
        errors.append({'path':root,'error':'missing'}); continue
    stack=[root]; order=[]
    while stack:
        directory=stack.pop(); order.append(directory)
        row={'bytes':0,'allocated_bytes':0,'files':0,'symlinks':0}
        dirs[directory]=row
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        s=entry.stat(follow_symlinks=False)
                        if stat.S_ISLNK(s.st_mode): row['symlinks']+=1; continue
                        if stat.S_ISDIR(s.st_mode): stack.append(entry.path); continue
                        if not stat.S_ISREG(s.st_mode): continue
                        size=s.st_size; allocated=s.st_blocks*512
                        row['bytes']+=size; row['allocated_bytes']+=allocated; row['files']+=1; count+=1
                        item=(size,entry.path,allocated,s.st_nlink,s.st_ino)
                        if len(top)<200: heapq.heappush(top,item)
                        elif size>top[0][0]: heapq.heapreplace(top,item)
                        if s.st_nlink>1:
                            key=str(s.st_dev)+':'+str(s.st_ino)
                            group=links.setdefault(key,{'bytes':size,'allocated_bytes':allocated,'nlink':s.st_nlink,'paths':[]})
                            group['paths'].append(entry.path)
                        if time.time()-last>25:
                            print('PROGRESS files='+str(count)+' seconds='+str(round(time.time()-started)),flush=True); last=time.time()
                    except OSError as e: errors.append({'path':entry.path,'error':str(e)})
        except OSError as e: errors.append({'path':directory,'error':str(e)})
    for directory in reversed(order):
        if directory!=root:
            parent=dirs[os.path.dirname(directory)]
            for k,v in dirs[directory].items(): parent[k]+=v
    roots.append({'path':root,**dirs[root]})
overcount=sum((len(g['paths'])-1)*g['bytes'] for g in links.values())
result={'started_unix':started,'finished_unix':time.time(),'method':'single sequential scandir/lstat; no file payload reads, no hashing; symlinks not followed','disk_before':before,'disk_after':shutil.disk_usage('/mnt/d')._asdict(),'roots':roots,'directories':dirs,'largest_files':[{'bytes':v[0],'path':v[1],'allocated_bytes':v[2],'nlink':v[3],'inode':v[4]} for v in sorted(top,reverse=True)],'hardlinks':[g for g in links.values() if len(g['paths'])>1],'known_hardlink_path_overcount_bytes':overcount,'unique_apparent_bytes_in_scope':sum(r['bytes'] for r in roots)-overcount,'errors':errors,'regular_files':count}
(OUT/'INVENTORY.json').write_text(json.dumps(result,indent=2))
with (OUT/'LARGEST_FILES.tsv').open('w') as f:
    f.write('bytes\tallocated_bytes\tnlink\tpath\n')
    for r in result['largest_files']: f.write(f"{r['bytes']}\t{r['allocated_bytes']}\t{r['nlink']}\t{r['path']}\n")
with (OUT/'DIRECTORIES.tsv').open('w') as f:
    f.write('bytes\tallocated_bytes\tfiles\tpath\n')
    for p,r in sorted(dirs.items(),key=lambda x:x[1]['bytes'],reverse=True): f.write(f"{r['bytes']}\t{r['allocated_bytes']}\t{r['files']}\t{p}\n")
print(json.dumps({'roots':roots,'files':count,'hardlink_overcount':overcount,'errors':len(errors),'seconds':time.time()-started}),flush=True)
