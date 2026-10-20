from r1e_fetch import R,get,log
import json,zipfile,io,shutil
# Official supplementary alias is PRO-35-e70453-s002.csv in PMC XML.
for url in ['https://pmc.ncbi.nlm.nih.gov/articles/instance/12817478/bin/PRO-35-e70453-s002.csv','https://onlinelibrary.wiley.com/doi/suppl/10.1002/pro.70453/supinfo/pro70453-sup-0001-DataS1.csv','https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12817478/supplementaryFiles']:
 b=get(url,'data/supplement_download_'+str(len(log))+'.dat',20*1024**2)
 if b and b[:2]==b'PK':
  z=zipfile.ZipFile(io.BytesIO(b));print(z.namelist())
  for n in z.namelist():
   if n.lower().endswith('.csv'):(R/'data/Garcia_DataS1.csv').write_bytes(z.read(n))
 elif b and (b',' in b[:500]) and not b[:100].lower().startswith(b'<!doctype') and b'html' not in b[:100].lower():(R/'data/Garcia_DataS1.csv').write_bytes(b)
 if (R/'data/Garcia_DataS1.csv').exists():break
commit=json.loads((R/'data/esm_source_commit.json').read_text())['sha']
b=get(f'https://codeload.github.com/facebookresearch/esm/zip/{commit}','vendor/esm_source.zip',8*1024**2)
if b:
 z=zipfile.ZipFile(io.BytesIO(b))
 for n in z.namelist():
  q=n.split('/',1)[-1]
  if q.startswith('esm/') or q in ['LICENSE','README.md']:
   if n.endswith('/'):continue
   p=R/'vendor/fair-esm'/q;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(n))
assert shutil.disk_usage(R).free>10*2**30+500*2**20
get('https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t12_35M_UR50D.pt','weights/esm2_t12_35M_UR50D.pt',250*2**20)
(R/'data/DOWNLOAD_RECEIPTS.json').write_text(json.dumps(log,indent=2));print(log)
