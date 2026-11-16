import json,zipfile,pickle,io,collections,os
from pathlib import Path
O=Path(__file__).resolve().parent
roots=[Path('/mnt/d/Simone/proteinGen/e015_artifacts'),Path('/mnt/d/Simone/proteinGen/.e015_worktree'),Path('/mnt/d/Simone/proteinGen/.e014_worktree/e007_matched_fullscale_v1')]
class Dummy:
 def __new__(cls,*a,**k):return super().__new__(cls)
 def __setstate__(self,state):pass
class Metadata(pickle.Unpickler):
 def find_class(self,module,name):
  if module=='collections' and name=='OrderedDict':return collections.OrderedDict
  return Dummy
 def persistent_load(self,pid):return Dummy()
records=[];metadata=[]
for root in roots:
 for base,ds,fs in os.walk(root):
  for n in fs:
   p=Path(base)/n
   if p.suffix=='.json' and (n in ['COMMITTED.json','LATEST.json','STATE.json'] or 'replay' in n.lower() or 'resume' in n.lower() or 'pause' in n.lower()):
    try:
     s=p.read_text()
     if len(s)<2_000_000:metadata.append((str(p),s))
    except Exception:pass
   if p.suffix not in ['.pt','.pth','.ckpt','.safetensors'] and '.pt.partial' not in n:continue
   st=p.stat();row={'path':str(p),'bytes':st.st_size,'mtime_unix':st.st_mtime,'classification':'PROTECTED','scientific_comparison':'boundary/preflight/replay evidence retained; no checkpoint deletion authorized','apparently_redundant':False}
   try:
    with zipfile.ZipFile(p) as z:
     member=next(x for x in z.namelist() if x.endswith('/data.pkl'))
     d=Metadata(io.BytesIO(z.read(member))).load()
    row['position']=d.get('position',{k:d.get(k) for k in ['update','exposures','epoch','schedule_cursor','total_exposures']})
    row['payload_keys']=list(d)
   except Exception as e:row['metadata_error']=str(e)
   records.append(row)
for r in records:
 references=[];digest=None
 for p,s in metadata:
  if r['path'] in s:
   references.append(p)
   try:
    d=json.loads(s)
    if d.get('path')==r['path']:digest=d.get('sha256')
   except Exception:pass
 r['metadata_references']=references;r['recorded_sha256']=digest;r['recovery_required']=bool(references) or 'recovery_' in r['path'] or 'sequence_pause_3580.pt' in r['path']
 if isinstance(r.get('position'),dict):
  r['update']=r['position'].get('update');r['exposures']=r['position'].get('exposures')
 elif 'sequence_pause_3580.pt' in r['path']:
  r['update']=3580;r['exposures']=4549;r['metadata_source']='Control COMMITTED.json'
 else:
  r['update']=r.get('position');r['exposures']=None;r['exposures_note']='No scalar recorded; exposure array was intentionally not materialized.'
(O/'checkpoint_inventory.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))
