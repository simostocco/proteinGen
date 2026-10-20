"""Stream the official processed CSV into one compact provenance-preserving table."""
from r1d_common import *
import io,zipfile,collections,numpy as np,pandas as pd,pyarrow as pa,pyarrow.parquet as pq
class HashReader(io.RawIOBase):
 def __init__(self,f):self.f=f;self.h=hashlib.sha256();self.n=0
 def readable(self):return True
 def readinto(self,b):
  x=self.f.read(len(b));b[:len(x)]=x;self.h.update(x);self.n+=len(x);return len(x)
def main():
 guard();z=zipfile.ZipFile(R/'data/Processed_K50_dG_datasets.zip');wanted=[i for i in z.infolist() if i.filename=='Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv'];assert len(wanted)==1;entry=wanted[0]
 for i in z.infolist():
  if i.filename=='Processed_K50_dG_datasets/Single_DMS_list.csv':
   data=z.read(i);(R/'data/Single_DMS_list.csv').write_bytes(data);print('DMS_LIST',data[:2000].decode(),flush=True)
 with z.open(entry) as ff:
  header=ff.readline().decode();write('data/CSV_HEADER.txt',header);print('HEADER',header,flush=True)
 columns=next(__import__('csv').reader([header]));keep=[c for c in columns if c in ['name','aa_seq','aa_seq_full','mut_type','WT_name','WT_cluster','dG_ML','ddG_ML','Stabilizing_mut','deltaG','deltaG_95CI','deltaG_95CI_high','deltaG_95CI_low','deltaG_t','deltaG_t_95CI','deltaG_t_95CI_low','deltaG_t_95CI_high','deltaG_c','deltaG_c_95CI','deltaG_c_95CI_low','deltaG_c_95CI_high','log10_K50_trypsin_ML','log10_K50_chymotrypsin_ML','fitting_error_t','fitting_error_c']]
 assert {'name','aa_seq','mut_type','WT_name','WT_cluster','dG_ML','ddG_ML','deltaG_95CI'}<=set(keep)
 h=HashReader(z.open(entry));stream=io.BufferedReader(h);temp=R/'data/STABILITY_SOURCE_ROWS.parquet.part';assert not temp.exists();writer=None;counts=collections.Counter();N=0;started=time.monotonic();parents={}
 for chunk in pd.read_csv(stream,usecols=keep,dtype=str,keep_default_na=False,chunksize=25000):
  chunk.insert(0,'source_row',np.arange(N,N+len(chunk),dtype=np.int64));N+=len(chunk)
  for n in ['dG_ML','ddG_ML']:counts[n+'_numeric']+=int(pd.to_numeric(chunk[n],errors='coerce').notna().sum());counts[n+'_censored']+=int(chunk[n].str.contains('[<>]',regex=True).sum());counts[n+'_dash']+=int(chunk[n].eq('-').sum())
  for name,g in chunk.groupby('WT_name',sort=False):
   a=parents.setdefault(name,{'rows':0,'numeric_dG':0,'numeric_ddG':0,'WT_cluster':str(g['WT_cluster'].iloc[0])});a['rows']+=len(g);a['numeric_dG']+=int(pd.to_numeric(g['dG_ML'],errors='coerce').notna().sum());a['numeric_ddG']+=int(pd.to_numeric(g['ddG_ML'],errors='coerce').notna().sum())
  table=pa.Table.from_pandas(chunk,preserve_index=False)
  if writer is None:writer=pq.ParquetWriter(temp,table.schema,compression='zstd')
  writer.write_table(table);guard()
  if N%100000==0:print(json.dumps({'rows':N,'seconds':time.monotonic()-started}),flush=True)
 writer.close();assert not stream.read(1);stream.close();path=R/'data/STABILITY_SOURCE_ROWS.parquet';os.replace(temp,path)
 write('data/CSV_MEMBER_PROVENANCE.json',{'archive_member':entry.filename,'uncompressed_bytes':h.n,'expected_uncompressed_bytes':entry.file_size,'member_SHA256':h.h.hexdigest(),'ZIP_CRC32':f'{entry.CRC:08x}','ZIP_CRC_verified_by_full_stream_read':True,'rows':N,'columns_preserved':keep,'source_row_semantics':'zero-based data record after header; original selected values remain strings including censor and missing markers','parquet_SHA256':sha(path),'seconds':time.monotonic()-started,'counts':dict(counts),'parents':parents,'no_full_CSV_extracted':True})
 print(json.dumps({'N':N,'parents':len(parents),'counts':counts,'parquet_MiB':path.stat().st_size/2**20,'seconds':time.monotonic()-started}),flush=True)
if __name__=='__main__':main()
