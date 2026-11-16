from pathlib import Path
import os,json
C=Path('/mnt/d/Simone/proteinGen_data/e017_v94_fixed_sequence_geometry_rl_20261024_v1/uniref_cleanup_20261024_v1');a=json.loads((C/'DELETE_ALLOWLIST.json').read_text());root=str(Path(a['root']).resolve());uses=[];errors=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  for fd in (p/'fd').iterdir():
   try:
    path=os.readlink(fd)
    if path.startswith(root):uses.append(dict(pid=int(p.name),kind='fd',path=path))
   except (FileNotFoundError,ProcessLookupError):pass
  for line in (p/'maps').read_text().splitlines():
   if root in line:uses.append(dict(pid=int(p.name),kind='mmap',entry=line))
 except (FileNotFoundError,ProcessLookupError):pass
 except PermissionError as e:errors.append(str(e))
(C/'WSL_ROOT_HANDLE_AUDIT.json').write_text(json.dumps(dict(PASS=not uses and not errors,uid=os.geteuid(),uses=uses,errors=errors,read_only_scan=True),indent=2)+'\n');assert not uses and not errors,(uses,errors);print('ROOT_HANDLE_SCAN_PASS',flush=True)
