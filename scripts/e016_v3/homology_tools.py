import os,sys,json,time,subprocess,resource,shutil,signal
from pathlib import Path
R=Path(os.environ['E016_ARTIFACT_ROOT']);D=R/'homology';RESERVE=10*2**30
def mem():return {x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if len(x.split())>=3}
def diskbytes(root):return sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
def run(name,cmd):
 receipt=D/(name+'.DONE.json')
 if receipt.exists():
  previous=json.load(open(receipt))
  if previous['command']!=cmd:raise RuntimeError('resume command changed:'+name)
  return previous
 guard=mem();assert guard['MemAvailable']>.1*guard['MemTotal']
 if shutil.disk_usage(R).free<RESERVE+4*2**30:raise RuntimeError('BLOCKED_DISK_BEFORE_SEARCH')
 log=D/(name+'.log');start=time.monotonic();history=[]
 with log.open('a') as out:
  child=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
  while child.poll() is None:
   mm=mem();free=shutil.disk_usage(R).free;psi=Path('/proc/pressure/memory').read_text()
   history.append({'elapsed':time.monotonic()-start,'MemAvailable':mm['MemAvailable'],'SwapFree':mm['SwapFree'],'free_disk':free,'PSI':psi})
   if free<RESERVE+2**30 or mm['MemAvailable']<.1*mm['MemTotal']:
    os.killpg(child.pid,signal.SIGINT);child.wait(timeout=20);raise RuntimeError('BLOCKED_ACTUAL_RESOURCE_PRESSURE:'+name)
   time.sleep(.2)
  code=child.wait()
 payload={'command':cmd,'exit_code':code,'seconds':time.monotonic()-start,'max_child_RSS_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,'minimum_MemAvailable':min((x['MemAvailable'] for x in history),default=None),'minimum_free_disk':min((x['free_disk'] for x in history),default=None),'workspace_bytes':None,'workspace_measurement_scope':'filesystem free sampled; avoid repeated recursive traversal per batch','threads':int(cmd[cmd.index('--threads')+1]) if '--threads' in cmd else None}
 (D/(name+'.RECEIPT.json')).write_text(json.dumps(payload,indent=2))
 if code:raise RuntimeError('MMseqs failure '+name+'; retain logs/partial outputs')
 receipt.write_text(json.dumps(payload,indent=2));print(name,json.dumps(payload),flush=True);return payload
