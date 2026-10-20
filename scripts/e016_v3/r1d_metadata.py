from pathlib import Path
import urllib.request,json,time,hashlib,shutil
R=Path('/mnt/d/Simone/proteinGen_data/e016_r1d_experimental_stability_20261020_v1');(R/'network').mkdir(exist_ok=True)
events=[]
def get(url,name,cap=8<<20):
 req=urllib.request.Request(url,headers={'User-Agent':'proteinGen-E016-R1d-bounded-public-stability-audit/1.0'})
 with urllib.request.urlopen(req,timeout=60) as res:
  data=res.read(cap+1);assert len(data)<=cap;meta={'url':url,'status':res.status,'bytes':len(data),'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'headers':dict(res.headers),'sha256':hashlib.sha256(data).hexdigest()}
 (R/'network'/name).write_bytes(data);events.append(meta);(R/'network/METADATA_HTTP.json').write_text(json.dumps(events,indent=2));return json.loads(data)
z=get('https://zenodo.org/api/records/7992926','ZENODO_RECORD.json')
print('LICENSE',z['metadata'].get('license'))
for f in z['files']:print(f['key'],f['size'],f['checksum'],f['links'])
g=get('https://api.github.com/repos/Rocklin-Lab/cdna-display-proteolysis-pipeline/commits/main','PIPELINE_COMMIT.json');rev=g['sha'];tree=get(f'https://api.github.com/repos/Rocklin-Lab/cdna-display-proteolysis-pipeline/git/trees/{rev}?recursive=1','PIPELINE_TREE.json')
for t in tree['tree']:
 if 'quality' in t['path'].lower() or 'STEP5' in t['path']:
  url=f'https://raw.githubusercontent.com/Rocklin-Lab/cdna-display-proteolysis-pipeline/{rev}/{t["path"]}'
  get(url,t['path'].split('/')[-1],16<<20)
print('pipeline',rev,'free_GiB',shutil.disk_usage(R).free/2**30)
