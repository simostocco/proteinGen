import sys
from pathlib import Path
sys.path.insert(0,'/mnt/d/Simone/proteinGen_data/pdb_only_v3_20261013')
from v3_common import *
# Same acquisition/parser contract; new diagnostics preserve historical summaries.
def write(name,obj,mutable=False):
 p=Path(os.environ.get('E016_BATCH_EVIDENCE_ROOT','/mnt/d/Simone/proteinGen_data/e016_cuda_n64_pdb_certification_20261014_v1'))/name
 if p.exists():raise RuntimeError('immutable continuation evidence collision '+str(p))
 p.write_text(json.dumps(obj,indent=2)+'\n')

import argparse,subprocess,gzip,collections
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['pilot','incremental'],required=True);p.add_argument('--limit',type=int,default=None);args=p.parse_args()
EVIDENCE_ROOT=Path(os.environ.get('E016_BATCH_EVIDENCE_ROOT','/mnt/d/Simone/proteinGen_data/e016_cuda_n64_pdb_certification_20261014_v1')).resolve()
if not str(EVIDENCE_ROOT).startswith('/mnt/d/') or not EVIDENCE_ROOT.is_dir():raise SystemExit('D_ONLY_BATCH_EVIDENCE_ROOT_REQUIRED')
if (EVIDENCE_ROOT/('ACQUISITION_'+args.phase.upper()+'_SUMMARY.json')).exists():raise SystemExit('NEW_BATCH_EVIDENCE_ROOT_REQUIRED_BEFORE_ACQUISITION')
queue=json.loads((ROOT/'catalogs/download_queue.json').read_text())
db=connection('catalogs/discovery.sqlite')
existing={}
journal=ROOT/'DOWNLOAD_JOURNAL.jsonl'
if journal.exists():
 for line in journal.open():
  r=json.loads(line)
  if r.get('status') in ['DONE','UPDATED','FAILED','DEFERRED']:existing[r['entry_id']]=r
phase_start=time.time();attempted=0;good=0;counts=collections.Counter();head_records=[]
for candidate in queue:
 pid=candidate['entry_id']
 if pid in existing:continue
 if args.limit is not None and attempted>=args.limit:break
 if time.time()-phase_start>=LIMIT['per_phase_wall_seconds']:
  write('ACQUISITION_STOP.json',{'status':'BLOCKED_RESOURCE','reason':'frozen per-phase1800-second wall budget','remaining_queue_entries':len(queue)-len(existing)-attempted,'phase':args.phase,'resume':'python -B '+str(ROOT/'acquire.py')+' --phase incremental'},mutable=True);break
 if traffic()>=LIMIT['initial_download_cap_bytes']-LIMIT['max_single_mmcif_compressed_bytes']:
  write('ACQUISITION_STOP.json',{'status':'BLOCKED_RESOURCE','reason':'initial2GiB HTTP cap; no expansion authorized automatically','phase':args.phase,'remaining_queue_entries':len(queue)-len(existing)-attempted,'resume':'python -B '+str(ROOT/'acquire.py')+' --phase incremental (requires explicit new cap contract after telemetry)'},mutable=True);break
 attempted+=1;started=time.time()
 raw_path=ROOT/'scratch'/(pid.lower()+'.cif.gz')
 parsed=ROOT/'scratch'/(pid.lower()+'.parsed.json')
 status='FAILED';reason=None
 try:
  guard('ACQUIRE_ENTRY')
  url=CFG['sources']['mmcif'].format(id=pid.lower())
  request=urllib.request.Request(url,method='HEAD',headers={'User-Agent':'proteinGen-PDB-only-v3/1.0'})
  try:
   with urllib.request.urlopen(request,timeout=LIMIT['http_timeout_seconds']) as response:
    size=response.headers.get('Content-Length');head={'entry_id':pid,'url':url,'status':response.status,'content_length':int(size) if size else None,'headers':dict(response.headers),'utc':now()}
  except Exception as ex:head={'entry_id':pid,'status':'UNKNOWN','reason':str(ex),'utc':now()}
  head_records.append(head);log('HEAD_JOURNAL.jsonl',head)
  if head.get('content_length') and head['content_length']>LIMIT['max_single_mmcif_compressed_bytes']:
   status='DEFERRED';raise OSError('DEFERRED_SIZE compressed-size cap')
  meta=fetch(CFG['sources']['entry_metadata'].format(id=pid),ROOT/'catalogs/entries'/(pid+'.json'),catalog=True)
  api=json.loads(Path(meta['path']).read_text())
  if normalize(api.get('rcsb_id',''))!=pid:raise ValueError('API_ENTRY_ID_MISMATCH')
  downloaded=fetch(url,raw_path,commit=False)
  child=subprocess.run([sys.executable,'-B',str(ROOT/'parse_entry.py'),downloaded['path'],pid,str(parsed)],capture_output=True,text=True,timeout=180,env=os.environ)
  (ROOT/'logs'/(pid+'_parser.log')).write_text(child.stdout+child.stderr)
  if child.returncode:
   fail=parsed.with_suffix('.failure.json');info=json.loads(fail.read_text()) if fail.exists() else {'reason':child.stderr[-1000:]}
   if 'BLOCKED_RESOURCE' in info['reason']:
    write('ACQUISITION_STOP.json',{'status':'BLOCKED_RESOURCE','entry_id':pid,'reason':info['reason'],'resume':'inspect parser budget before resuming'},mutable=True);reason=info['reason'];status='DEFERRED';raise MemoryError(info['reason'])
   status='DEFERRED' if 'DEFERRED_RESOURCE' in info['reason'] else 'FAILED';raise ValueError(info['reason'])
  result=json.loads(parsed.read_text())
  if result['source_sha256']!=downloaded['sha256']:raise ValueError('DOWNLOADED_SOURCE_SHA_MISMATCH')
  accession=api.get('rcsb_accession_info',{});revision_ok='UNKNOWN';revs=result['revision_history']
  if revs and accession.get('major_revision') is not None and accession.get('minor_revision') is not None:
   last=max(revs,key=lambda r:int(r.get('ordinal') or 0))
   revision_ok=int(last['major_revision'])==accession['major_revision'] and int(last['minor_revision'])==accession['minor_revision']
  seqres_hashes={r[0] for r in db.execute("SELECT sha256 FROM seqres WHERE entry_id=? AND molecule='protein'",(pid,))}
  for entity in result['entities']:
   if entity['status']=='VERIFIED' and entity['full_sha256'] not in seqres_hashes:
    entity.update(status='QUARANTINE',lm_eligible=False,reason='SEQRES_MMCIF_FULL_SEQUENCE_DISCORDANCE');result['quarantine'].append({'entity_id':entity['entity_id'],'reason':entity['reason']})
   if revision_ok is False:entity.update(status='QUARANTINE',lm_eligible=False,reason='API_MMCIF_REVISION_DISCORDANCE')
  allowed={r['entity_id'] for r in result['entities'] if r['status']=='VERIFIED'}
  for ob in result['observations']:
   if ob['entity_id'] not in allowed:ob['status']='QUARANTINE';ob['observed_order_proxy']=False
  result['segments']=[s for s in result['segments'] if any(o['observation_id']==s['observation_id'] and o['status']=='VERIFIED' for o in result['observations'])]
  destination=ROOT/'raw'/pid/downloaded['sha256']/(pid.lower()+'.cif.gz');destination.parent.mkdir(parents=True,exist_ok=True)
  if destination.exists():
   if sha(destination)!=downloaded['sha256']:raise ValueError('EXISTING_CONTENT_HASH_PATH_MISMATCH')
   raw_committed=False
  else:os.replace(downloaded['path'],destination);raw_committed=True
  result.update(source_path=str(destination),api_metadata_path=meta['path'],api_revision_check=revision_ok,download_headers=downloaded['headers'],source_sha256=downloaded['sha256'],source_revision_status='source CIF revision recorded; historical source never overwritten')
  output=ROOT/'tables/parsed'/pid/(downloaded['sha256']+'.json.gz');output.parent.mkdir(parents=True,exist_ok=True)
  if not output.exists():
   with gzip.GzipFile(filename=str(output)+'.part',mode='wb',mtime=0) as f:f.write(json.dumps(result,separators=(',',':')).encode())
   os.replace(str(output)+'.part',output)
  status='DONE';good+=1
  record={'entry_id':pid,'status':status,'phase':args.phase,'source_sha256':downloaded['sha256'],'source_path':str(destination),'source_bytes':downloaded['bytes'],'parsed_path':str(output),'parsed_sha256':sha(output),'api_metadata_sha256':meta['sha256'],'revision_check':revision_ok,'methods':result['methods'],'resolutions_angstrom':result['resolutions_angstrom'],'entities':len(result['entities']),'eligible_lm_entities':sum(e['lm_eligible'] for e in result['entities']),'observations':len(result['observations']),'verified_observations':sum(o['status']=='VERIFIED' for o in result['observations']),'geometry_segments':len(result['segments']),'quarantine_entities':len(result['quarantine']),'parser_peak_rss_bytes':result['peak_rss_bytes'],'novelty_scope':candidate['novel_vs_v2_only'],'length_stratum':candidate['length_stratum'],'elapsed_seconds':time.time()-started}
  log('DOWNLOAD_JOURNAL.jsonl',record);counts[status]+=1
 except Exception as ex:
  reason=str(ex)
  if reason.startswith('DEFERRED') or 'BLOCKED_RESOURCE' in reason:status='DEFERRED'
  log('DOWNLOAD_JOURNAL.jsonl',{'entry_id':pid,'status':status,'phase':args.phase,'reason':reason,'part_source':str(raw_path)+'.part','length_stratum':candidate['length_stratum'],'elapsed_seconds':time.time()-started});counts[status]+=1
  if 'SHA_MISMATCH' in reason or 'HASH_PATH_MISMATCH' in reason or 'BLOCKED_RESOURCE' in reason:
   write('ACQUISITION_STOP.json',{'status':'BLOCKED_RESOURCE' if 'BLOCKED_RESOURCE' in reason else 'BLOCKED_SOURCE','entry_id':pid,'reason':reason,'resume':'inspect source/resource contradiction before resuming'},mutable=True);break
 if attempted%10==0:
  telemetry=guard('ACQUISITION_PROGRESS');log('RESOURCE_TELEMETRY.jsonl',{**telemetry,'attempted_in_phase':attempted,'validated_in_phase':good,'transferred_bytes':traffic()});print('acquired',args.phase,attempted,'valid',good,'counts',dict(counts),'free GiB',round(telemetry['free_bytes']/2**30,3),flush=True)
db.close()
write('ACQUISITION_'+args.phase.upper()+'_SUMMARY.json',{'phase':args.phase,'utc':now(),'attempted':attempted,'validated':good,'status_counts':dict(counts),'head_sample_count':len(head_records),'head_known_bytes':sum(r.get('content_length') or 0 for r in head_records),'head_unknown_count':sum(r.get('content_length') is None for r in head_records),'transferred_bytes_total':traffic(),'free_bytes':shutil.disk_usage('/mnt/d').free,'wall_seconds':time.time()-phase_start,'remaining_entries':'computed from immutable queue and journal at publication','pilot_is_full_PDB':False,'stop_file':str(ROOT/'ACQUISITION_STOP.json') if (ROOT/'ACQUISITION_STOP.json').exists() else None})
print(json.dumps({'phase':args.phase,'attempted':attempted,'validated':good,'counts':dict(counts),'wall_seconds':time.time()-phase_start}),flush=True)
