import os,json,subprocess,time
from pathlib import Path
O=Path(__file__).resolve().parent
data={'timestamp':time.time(),'linux':[],'deleted_open':[],'errors':[]}
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  parts=(p/'cmdline').read_bytes().split(b'\0');cmd=' '.join(x.decode(errors='replace') for x in parts if x)
  if not parts or not any(x in Path(parts[0].decode(errors='replace')).name.lower() for x in ['python','mmseqs']):continue
  row={'pid':int(p.name),'command':cmd,'cwd':os.readlink(p/'cwd'),'open_project_files':[]}
  for f in (p/'fd').iterdir():
   try:
    link=os.readlink(f)
    if '/mnt/d/' in link:row['open_project_files'].append({'fd':f.name,'path':link,'fdinfo':(p/'fdinfo'/f.name).read_text()})
   except OSError:pass
  for line in (p/'maps').read_text().splitlines():
   if '/mnt/d/' in line:row['open_project_files'].append({'mapped':line.split(maxsplit=5)[-1]})
  row['io']=(p/'io').read_text()
  allowed={'TMPDIR','TMP','TEMP','TORCH_HOME','HF_HOME','TRITON_CACHE_DIR','CUDA_CACHE_PATH','MPLCONFIGDIR','XDG_CACHE_HOME','NUMBA_CACHE_DIR','PYTHONDONTWRITEBYTECODE','PYTHONPYCACHEPREFIX'}
  row['artifact_environment']={}
  for item in (p/'environ').read_bytes().split(b'\0'):
   key,sep,value=item.partition(b'=')
   if key.decode(errors='replace') in allowed:row['artifact_environment'][key.decode()]=value.decode(errors='replace')
  data['linux'].append(row)
 except OSError as e:data['errors'].append(str(e))
r=subprocess.run(['lsof','-nP','+L1'],capture_output=True,text=True)
data['deleted_open']={'stdout':r.stdout,'stderr':r.stderr,'returncode':r.returncode,'limitation':'Unprivileged Linux visibility; native Windows handles not covered'}
q='Get-CimInstance Win32_Process | Where-Object { $_.Name -match "python|mmseqs|linclust" } | Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Depth 3'
r=subprocess.run(['powershell.exe','-NoProfile','-Command',q],capture_output=True,text=True)
data['windows']={'stdout':r.stdout,'stderr':r.stderr,'returncode':r.returncode,'no_matching_processes':r.returncode==0 and not r.stdout.strip()}
data['windows_handle_limitation']='No native worker found; Windows-wide deleted-open NTFS handles not proven absent. No Sysinternals handle utility invoked/installed.'
(O/'active_writers.json').write_text(json.dumps(data,indent=2))
print(json.dumps(data,indent=2))
