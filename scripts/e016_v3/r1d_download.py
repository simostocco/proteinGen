"""Official bounded download; stream ZIP later, never extract large CSVs."""
from r1d_common import *
import urllib.request,zipfile

def main():
 record=json.load(open(R/'network/ZENODO_RECORD.json'));entry=next(f for f in record['files'] if f['key']=='Processed_K50_dG_datasets.zip');p=R/'data'/entry['key'];assert not p.exists();guard(entry['size']+512*2**20);temp=p.with_suffix('.zip.part');assert not temp.exists()
 url=entry['links']['self'];req=urllib.request.Request(url,headers={'User-Agent':'proteinGen-E016-R1d-bounded-stability-audit/1.0'});n=0;md=hashlib.md5();hh=hashlib.sha256();start=time.monotonic()
 with urllib.request.urlopen(req,timeout=60) as resp,temp.open('xb') as f:
  headers=dict(resp.headers);assert resp.status==200
  while True:
   block=resp.read(1<<20)
   if not block:break
   n+=len(block);assert n<=entry['size'];guard();assert time.monotonic()-start<900
   f.write(block);md.update(block);hh.update(block)
   if n%(64<<20)==0:print(json.dumps({'download_MiB':n/2**20,'seconds':time.monotonic()-start}),flush=True)
  f.flush();os.fsync(f.fileno())
 assert n==entry['size'] and 'md5:'+md.hexdigest()==entry['checksum'];os.replace(temp,p)
 z=zipfile.ZipFile(p);members=[{'path':i.filename,'bytes_uncompressed':i.file_size,'bytes_compressed':i.compress_size,'CRC32':f'{i.CRC:08x}'} for i in z.infolist()];write('data/ZIP_INVENTORY.json',members)
 receipt={'url':url,'official_record':'https://zenodo.org/records/7992926','record_version':record['metadata'].get('version'),'license':record['metadata'].get('license'),'bytes':n,'md5':md.hexdigest(),'official_md5_verified':True,'sha256':hh.hexdigest(),'seconds':time.monotonic()-start,'headers':headers,'archive_path':str(p),'no_full_CSV_extraction':True,'free_bytes_after':guard(),'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};write('TSUBOYAMA_DATASET_PROVENANCE.json',receipt);print(json.dumps({'receipt':receipt,'members':members}),flush=True)
if __name__=='__main__':main()
