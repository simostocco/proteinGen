import os,json,time,sys
from pathlib import Path
O=Path(__file__).resolve().parent
E15=Path('/mnt/d/Simone/proteinGen/e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1')
E12=Path('/mnt/d/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization/scientific_corrected_protein_v1')
targets=[E15/'production',E15/'production/checkpoints',E15/'logs',E15/'production/logs',E15/'production/evaluation',E12/'clusters',E12/'mmseqs',E12/'tmp',E12/'production',E12/'production/reports',E12/'production/logs']
data={'timestamp':time.time(),'free':os.statvfs('/mnt/d').f_bavail*os.statvfs('/mnt/d').f_frsize,'directories':{},'files':{},'states':{}}
for target in targets:
 total=0;count=0
 for base,dirs,names in os.walk(target,followlinks=False):
  for name in names:
   p=Path(base)/name
   if p.is_symlink():continue
   try:s=p.stat()
   except OSError:continue
   total+=s.st_size;count+=1;data['files'][str(p)]={'bytes':s.st_size,'mtime':s.st_mtime}
 data['directories'][str(target)]={'bytes':total,'files':count}
for p in [E15/'production/HEARTBEAT.json',E15/'production/COMMITTED.json',E12/'production/STATE.json']:
 try:
  r=json.loads(p.read_text());data['states'][str(p)]={k:r[k] for k in ['phase','pid','timestamp_unix','update','exposures','path','sha256','position','error','stage_at_failure','total_identities','proteins_per_second'] if k in r}
 except Exception as e:data['states'][str(p)]=str(e)
(O/sys.argv[1]).write_text(json.dumps(data,separators=(',',':')))
print(json.dumps({k:v for k,v in data.items() if k!='files'},indent=2))
