from r1e_common import *
import urllib.request,zipfile,io
url='https://pypi.org/pypi/fair-esm/2.0.0/json'
x=json.load(urllib.request.urlopen(url,timeout=30));u=next(v for v in x['urls'] if v['filename'].endswith('.whl'))
b=urllib.request.urlopen(u['url'],timeout=60).read(1024**2);assert hashlib.sha256(b).hexdigest()==u['digests']['sha256']
p=R/'vendor'/u['filename'];p.write_bytes(b);z=zipfile.ZipFile(io.BytesIO(b));z.extractall(R/'vendor/fair-esm');write('data/ESM_IMPLEMENTATION_PROVENANCE.json',dict(metadata_url=url,wheel_url=u['url'],sha256=sha(p),bytes=len(b),version='fair-esm2.0.0',license=x['info']['license'],note='official PyPI inference source; no environment installation or dependency expansion'))
print('esm source',len(b))
