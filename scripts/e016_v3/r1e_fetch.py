from pathlib import Path
import urllib.request,json,hashlib,zipfile,io,shutil,time
R=Path(__file__).resolve().parents[3]; log=[]
def get(url,name,cap=5*1024**2):
 p=R/name
 if p.exists():return p.read_bytes()
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'E016-academic-evaluation/1.0'})
  with urllib.request.urlopen(req,timeout=90) as f:
   b=f.read(cap+1)
   if len(b)>cap:raise ValueError('download cap')
  p.write_bytes(b);log.append(dict(url=url,path=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()));return b
 except Exception as e:log.append(dict(url=url,error=str(e)));return b''
if __name__=='__main__':
 xml=get('https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12817478/fullTextXML','data/garcia_article.xml')
 if xml:
  import xml.etree.ElementTree as E
  root=E.fromstring(xml)
  for x in root.iter():
   if x.tag.endswith('supplementary-material'): print(E.tostring(x,encoding='unicode')[:3000])
 get('https://huggingface.co/api/models/facebook/esm2_t12_35M_UR50D','data/esm_hf_metadata.json')
 get('https://api.github.com/repos/facebookresearch/esm/commits/main','data/esm_source_commit.json')
 (R/'data/NETWORK_RECEIPTS.json').write_text(json.dumps(log,indent=2))
 print(log)
