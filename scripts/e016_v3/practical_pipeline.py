"""Sequential bounded continuation. No training, downloads or policy selection."""
import collections,json,os,shutil,subprocess,sys,time
from pathlib import Path
from practical_split import sha,save,GIB

def stage(root,name,extra=()):
 cmd=[sys.executable,'-B',str(Path(__file__).with_name('practical_split.py')),name,'--root',str(root),*extra]
 label=name+('_pilot' if extra else '')
 if (root/(label+'_RUN_RESULT.json')).exists():
  prior=json.load(open(root/(label+'_RUN_RESULT.json')))
  if prior['exit_code']==0:return
 start=time.monotonic();peak=0;minimum=shutil.disk_usage(root).free;minmem=2**64
 logfile=root/(label+'_RUN.log');attempt=1
 while logfile.exists():attempt+=1;logfile=root/(label+f'_RUN_attempt{attempt}.log')
 save(root/'PIPELINE_STATE.json',{'stage':label,'status':'RUNNING','command':cmd,'source_sha256':sha(Path(__file__).with_name('practical_split.py'))})
 with logfile.open('x') as log:
  proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
  while proc.poll() is None:
   ids=[proc.pid];rss=0
   for i in ids:
    try:
     d=Path(f'/proc/{i}/status').read_text();rss+=int(d.split('VmRSS:')[1].split()[0])*1024
     ids.extend(map(int,Path(f'/proc/{i}/task/{i}/children').read_text().split()))
    except (OSError,IndexError):pass
   peak=max(peak,rss);free=shutil.disk_usage(root).free;minimum=min(minimum,free)
   mem=int(Path('/proc/meminfo').read_text().split('MemAvailable:')[1].split()[0])*1024;minmem=min(minmem,mem)
   if free<10*GIB+256*2**20:
    proc.terminate();proc.wait();raise RuntimeError('disk reserve pressure; preserve logs')
   time.sleep(.5)
 result={'command':cmd,'exit_code':proc.returncode,'wall_seconds':time.monotonic()-start,'observed_parent_and_descendants_RSS_peak_bytes':peak,'minimum_free_bytes':minimum,'minimum_MemAvailable_bytes':minmem,'log':str(logfile)}
 save(root/(label+'_RUN_RESULT.json'),result)
 if proc.returncode:raise RuntimeError(f'{label} failed; see {logfile}')
 print('COMPLETE',label,result['wall_seconds'],flush=True)

def main(root):
 start=time.monotonic()
 while not (root/'alignment_coverage_v1/COMPLETE.json').exists():
  save(root/'PIPELINE_STATE.json',{'stage':'WAITING_FOR_GROUPING','status':'WAITING','wait_seconds':time.monotonic()-start})
  if time.monotonic()-start>3*3600:raise RuntimeError('bounded wait expired; inspect grouping journal')
  time.sleep(10)
 stage(root,'allocate')
 stage(root,'cross',('--limit','2'))
 batches=[json.load(open(p)) for p in (root/'cross_screen_coverage_v1').glob('batch-*/DONE.json')];n=sum(len(x['queries']) for x in batches);used=sum(x['bytes'] for x in batches);projection=used/n*147769;free=shutil.disk_usage(root).free
 budget={'pilot_queries':n,'pilot_bytes':used,'projected_cross_screen_bytes':projection,'conservative_projected_remaining_bytes':2*projection-used+512*2**20,'free_bytes':free,'reserve_bytes':10*GIB,'status':'GO' if free-(2*projection-used+512*2**20)>10*GIB else 'BLOCKED_RESOURCE_ESTIMATE'};save(root/'CROSS_SCREEN_BUDGET.json',budget)
 if budget['status']!='GO':raise RuntimeError('cross-screen storage forecast does not preserve reserve; optimize engineering before resume')
 stage(root,'cross')
 stage(root,'audit')
 stage(root,'independent')
 save(root/'PIPELINE_STATE.json',{'stage':'READY_FOR_LOADER_AND_FINAL_REPORT','status':'CHECKS_COMPLETE_NO_TRAINING','wall_seconds':time.monotonic()-start})

if __name__=='__main__':
 root=Path(sys.argv[1])
 try:main(root)
 except Exception as exc:
  save(root/'PIPELINE_BLOCKER.json',{'error':str(exc),'status':'BLOCKED_NO_PUBLICATION','time_utc_epoch':time.time()});raise
