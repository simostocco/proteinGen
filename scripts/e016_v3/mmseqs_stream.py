"""Version-scoped MMseqs15 native-result I/O, validated against native createtsv.
Not a biological-source parser. Zstd frames and index sizes follow DBReader/Writer.
"""
import struct,bisect
from pathlib import Path
import pyarrow as pa
class Reader:
 def __init__(self,base):
  self.base=Path(base);self.kind=struct.unpack('<I',Path(str(base)+'.dbtype').read_bytes())[0];assert self.kind&0xffff in [5,7],'only alignment/prefilter result types'
  self.compressed=bool(self.kind&0x80000000);self.index=[tuple(map(int,line.split())) for line in Path(str(base)+'.index').read_text().splitlines()];assert all(len(x)==3 and x[2]>=1 for x in self.index)
  assert len({x[0] for x in self.index})==len(self.index),'duplicate native query key'
  files=[self.base] if self.base.is_file() else sorted((p for p in self.base.parent.glob(self.base.name+'.*') if p.name.rsplit('.',1)[-1].isdigit()),key=lambda p:int(p.name.rsplit('.',1)[-1]))
  assert files;self.files=[p.open('rb') for p in files];self.offsets=[0]
  for p in files:self.offsets.append(self.offsets[-1]+p.stat().st_size)
 def entries(self,skip_keys=()):
  skip_keys=set(skip_keys)
  for key,offset,length in sorted(self.index,key=lambda x:x[1]):
   if key in skip_keys:continue
   i=bisect.bisect_right(self.offsets,offset)-1;assert 0<=i<len(self.files);f=self.files[i];local=offset-self.offsets[i];f.seek(local)
   if self.compressed:
    header=f.read(4);assert len(header)==4;csize=struct.unpack('<I',header)[0];assert local+4+csize+1<=self.offsets[i+1]-self.offsets[i]
    payload=f.read(csize);marker=f.read(1);assert marker in [b'\0',b'\xff']
    body=pa.Codec('zstd').decompress(payload,decompressed_size=length-1,asbytes=True) if marker==b'\0' else payload
   else:
    raw=f.read(length);assert raw.endswith(b'\0');body=raw[:-1]
   assert len(body)==length-1 and b'\0' not in body
   yield key,body
 def close(self):
  for f in self.files:f.close()
class Writer:
 def __init__(self,base,kind=7):
  self.base=Path(base);self.f=self.base.open('xb');self.index=[];self.kind=kind
 def add(self,key,body):
  assert not body or body.endswith(b'\n');assert b'\0' not in body
  offset=self.f.tell();payload=pa.compress(body,codec='zstd',asbytes=True);self.f.write(struct.pack('<I',len(payload)));self.f.write(payload);self.f.write(b'\0');self.index.append((key,offset,len(body)+1))
 def close(self):
  self.f.flush();self.f.close()
  with Path(str(self.base)+'.index').open('x') as f:
   for row in sorted(self.index):f.write('%d\t%d\t%d\n'%row)
  with Path(str(self.base)+'.dbtype').open('xb') as f:f.write(struct.pack('<I',self.kind|0x80000000))
class Components:
 def __init__(self,parent):self.parent=list(parent);self.rank=[0]*len(parent)
 def find(self,i):
  p=self.parent
  while p[i]!=i:p[i]=p[p[i]];i=p[i]
  return i
 def link(self,a,b):
  a=self.find(a);b=self.find(b)
  if a==b:return False
  if self.rank[a]<self.rank[b]:a,b=b,a
  self.parent[b]=a
  if self.rank[a]==self.rank[b]:self.rank[a]+=1
  return True
