"""Resume the approved2440 queue; one child/100attempt checkpoint, no outside IDs."""
import os,sys,json,subprocess,time,shutil
from pathlib import Path
R=Path(os.environ['E016_ARTIFACT_ROOT']);P=Path('/mnt/d/Simone/proteinGen_data/pdb_only_v3_20261013')
queue=json.load(open(P/'catalogs/download_queue.json'));allowed={x['entry_id'] for x in queue};assert len(allowed)==2440
if (R/'acquisition/COMPLETE.json').exists():
 previous=json.load(open(R/'acquisition/COMPLETE.json'));latest={x['entry_id']:x for x in map(json.loads,(P/'DOWNLOAD_JOURNAL.jsonl').open())};assert allowed<=latest.keys();actual={status:sum(latest[k]['status']==status for k in allowed) for status in {latest[k]['status'] for k in allowed}};assert actual==previous['statuses']
 print(json.dumps({'status':'ALREADY_COMPLETE_NO_WRITES_NO_HTTP','priority_entries':len(allowed),'terminal_states':actual}),flush=True);sys.exit(0)
start=time.monotonic();batch=0
while True:
 latest={x['entry_id']:x for x in map(json.loads,(P/'DOWNLOAD_JOURNAL.jsonl').open())}
 pending=sorted(allowed-latest.keys())
 if not pending:break
 free=shutil.disk_usage(R).free
 if free<30*2**30:raise RuntimeError('BLOCKED_RESOURCE acquisition20GiBforecast+10GiBreserve')
 batch+=1;evidence=R/'acquisition'/f'batch-{batch:04d}'
 if evidence.exists():continue
 evidence.mkdir();env=os.environ.copy();env['E016_BATCH_EVIDENCE_ROOT']=str(evidence)
 cmd=[sys.executable,'-B',str(R/'source/scripts/e016_v3/resume_priority_batch.py'),'--phase','incremental','--limit','100']
 with (evidence/'output.log').open('w') as log:
  child=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1850)
 after={x['entry_id']:x for x in map(json.loads,(P/'DOWNLOAD_JOURNAL.jsonl').open())}
 row={'batch':batch,'exit_code':child.returncode,'remaining_unattempted':len(allowed-after.keys()),'DONE':sum(x['status']=='DONE' for x in after.values()),'free_bytes':shutil.disk_usage(R).free,'elapsed_seconds':time.monotonic()-start}
 with (R/'acquisition/CONTINUATION.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
 if child.returncode or len(after)<=len(latest):raise RuntimeError('BLOCKED_NO_PROGRESS_OR_CHILD_FAILURE')
 if (evidence/'ACQUISITION_STOP.json').exists():raise RuntimeError('BLOCKED_CHILD_RESOURCE_OR_SOURCE')
(R/'acquisition/COMPLETE.json').write_text(json.dumps({'priority_entries':2440,'terminal_entries':len(allowed&latest.keys()),'DONE':sum(x['status']=='DONE' for k,x in latest.items() if k in allowed),'statuses':{s:sum(x['status']==s for k,x in latest.items() if k in allowed) for s in {x['status'] for x in latest.values()}},'elapsed_seconds':time.monotonic()-start},indent=2))
