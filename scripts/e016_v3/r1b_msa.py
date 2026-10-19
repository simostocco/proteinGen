"""Serial official ColabFold/MMseqs2 searches for the six frozen public TRAIN natives only."""
from r1b_common import *
import sys,re,numpy as np
class HardStop(BaseException):pass
CURRENT={};ORIGINAL=requests.sessions.Session.request

def audited_request(self,method,url,**kw):
 guard();ev=events();used=sum(v.get('response_bytes',0) for v in ev);reserved=sum(v.get('event')=='MSA_HTTP_RESERVED' and v['method']=='POST' for v in ev)
 if time.monotonic()-CURRENT['start']>600:raise HardStop('600-second per-search bound')
 if urllib.parse.urlparse(url).hostname!='api.colabfold.com':raise HardStop('unapproved MSA endpoint')
 if method.upper()=='POST':
  if reserved>=12:raise HardStop('12 submission hard cap, including retries')
  text=kw.get('data',{}).get('q','');sq=''.join(v.strip() for v in text.splitlines() if not v.startswith('>'))
  if sq!=CURRENT['sequence']:raise HardStop('only the active verified public PDB sequence may be submitted')
 record={'event':'MSA_HTTP_RESERVED','method':method.upper(),'url':url,'native_id':CURRENT['id'],'sequence_sha256':hashlib.sha256(CURRENT['sequence'].encode()).hexdigest(),'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};log_event(record);started=time.monotonic();kw['stream']=True
 try:
  response=ORIGINAL(self,method,url,**kw);body=[];size=0
  for b in response.iter_content(1<<16):
   size+=len(b)
   if size>min(128*2**20,750*2**20-used):raise HardStop('MSA response exceeds download cap')
   body.append(b)
  content=b''.join(body);response._content=content;response._content_consumed=True;name=f"network/MSA_HTTP_{sum(v.get('event')=='MSA_HTTP_RESERVED' for v in ev):04d}.body";p=R/name;p.write_bytes(content)
  row=dict(record,event='MSA_HTTP_COMPLETED',status=response.status_code,response_bytes=size,body_path=name,sha256=sha(p),seconds=time.monotonic()-started,headers={k:v for k,v in response.headers.items() if k.lower() in ['date','retry-after','content-type','content-length']});log_event(row)
  if response.status_code in [429,403]:raise HardStop(f'server HTTP{response.status_code}; stop instead of bypassing limits')
  return response
 except HardStop:raise
 except Exception as e:log_event(dict(record,event='MSA_HTTP_ERROR',error=type(e).__name__+': '+str(e),seconds=time.monotonic()-started));raise

def msa_metrics(text,query):
 lines=[];seq=''
 for ln in text.splitlines():
  if ln.startswith('#'):continue
  if ln.startswith('>'):
   if seq:lines.append(seq);seq=''
  else:seq+=ln.strip()
 if seq:lines.append(seq)
 aligned=[re.sub('[a-z.]','',s).upper() for s in lines];assert aligned and aligned[0].replace('-','')==query and all(len(v)==len(query) for v in aligned)
 unique=list(dict.fromkeys(aligned));array=np.array([list(v) for v in unique],dtype='U1');coverage=(array!='-').mean(1);eligible=np.flatnonzero(coverage>=.5);chosen=eligible[np.linspace(0,len(eligible)-1,min(512,len(eligible)),dtype=int)];a=array[chosen];counts=[]
 for v in a:
  union=((a!='-')|(v!='-')).sum(1);identity=((a==v)&(a!='-')&(v!='-')).sum(1)/np.maximum(union,1);counts.append(int((identity>=.8).sum()))
 return {'raw_depth':len(aligned),'exact_unique_depth':len(unique),'query_coverage':1.,'homologs_coverage_atleast50pct':int((coverage[1:]>=.5).sum()),'unique_row_query_coverage_mean':float(coverage.mean()),'gap_fraction_mean':float(1-coverage.mean()),'row_coverage_quantiles':np.quantile(coverage,[0,.1,.5,.9,1]).tolist(),'column_gap_fraction_quantiles':np.quantile((array=='-').mean(0),[0,.1,.5,.9,1]).tolist(),'effective_depth80_on_max512_coverage50_rows':float(sum(1/max(v,1) for v in counts)),'effective_depth_sample_rows':len(a),'effective_depth_definition':'sum reciprocal >=80% identity neighbors; identity counts shared non-gap matches divided by non-gap union; deterministic evenly spaced sample up to512 eligible unique rows, not extrapolated full-MSA Neff','length':len(query)}

def main():
 import tarfile
 original_extract=tarfile.TarFile.extractall
 def checked_extract(self,path='.',members=None,**kw):
  target=Path(path).resolve();items=self.getmembers() if members is None else list(members)
  if sum(v.size for v in items)>128*2**20:raise HardStop('MSA extracted-size cap')
  for v in items:
   if not (target/v.name).resolve().is_relative_to(target) or v.issym() or v.islnk():raise HardStop('unsafe MSA tar member')
  return original_extract(self,path,items,**kw)
 tarfile.TarFile.extractall=checked_extract;requests.sessions.Session.request=audited_request
 from colabfold.colabfold import run_mmseqs2
 panel=json.load(open(R/'configs/NATIVE_PANEL_32.json'));phase=sys.argv[1] if len(sys.argv)>1 else 'principal';frozen=json.load(open(R/'configs/MSA_FOLD_FROZEN.json')) if phase=='conditional' else None
 if phase=='conditional':assert json.load(open(R/'folds/ALLOW_CONDITIONAL.json'))['allowed']
 ids=panel['MSA_first_six'] if phase=='principal' else frozen['conditional_TRAIN']+frozen['conditional_VALIDATION'];controls=sorted((c for c in panel['controls'] if c['id'] in ids),key=lambda c:c['N']);rows=json.load(open(R/'COLABFOLD_MSA_QUERY_MANIFEST.json'))['rows'] if (R/'COLABFOLD_MSA_QUERY_MANIFEST.json').exists() else [];metrics=[r['metrics'] for r in rows if r['status']=='SUCCEEDED']
 for c in controls:
  if (R/'msa'/c['id']/'RECEIPT.json').exists():
   row=json.load(open(R/'msa'/c['id']/'RECEIPT.json'))
   if row['id'] not in {v['id'] for v in rows}:
    rows.append(row)
    if row['status']=='SUCCEEDED':metrics.append(row['metrics'])
   continue
  out=R/'msa'/c['id'];out.mkdir(parents=True,exist_ok=True);CURRENT.update(id=c['id'],sequence=c['sequence'],start=time.monotonic());row={'id':c['id'],'sequence_sha256':c['sequence_id'],'entry_id':c['entry_id'],'public_PDB_accepted_split_verified':True,'split':c['split'],'endpoint':'https://api.colabfold.com','official_workflow':'colabfold.colabfold.run_mmseqs2 use_env=True,use_filter=True,use_templates=False','status':'RUNNING'};write(str((out/'RECEIPT.json').relative_to(R)),row)
  try:
   result=run_mmseqs2(c['sequence'],str(out/'search'),use_env=True,use_filter=True,use_templates=False,host_url='https://api.colabfold.com',user_agent='proteinGen-E016-R1b-bounded-public-native-controls/1.0');text=result[0];p=out/'query.a3m';p.write_text(text);metric=msa_metrics(text,c['sequence']);metric['id']=c['id'];row.update(status='SUCCEEDED',a3m_path=str(p),a3m_sha256=sha(p),metrics=metric);metrics.append(metric)
  except BaseException as e:row.update(status='FAILED',error=type(e).__name__+': '+str(e))
  row['wall_seconds']=time.monotonic()-CURRENT['start'];write(str((out/'RECEIPT.json').relative_to(R)),row);rows.append(row);write('COLABFOLD_MSA_QUERY_MANIFEST.json',{'rows':rows,'submissions_including_retries':sum(v.get('event')=='MSA_HTTP_RESERVED' and v['method']=='POST' for v in events()),'max_submissions':12,'serial_client':True,'new_generated_sequences_submitted':0});write('MSA_DEPTH_AND_COVERAGE.json',{'rows':metrics});print(json.dumps({'id':c['id'],'status':row['status'],'seconds':row['wall_seconds'],'depth':row.get('metrics',{}).get('raw_depth'),'error':row.get('error')}),flush=True)
  if row['status']!='SUCCEEDED':break
 if not (R/'COLABFOLD_MSA_QUERY_MANIFEST.json').exists():write('COLABFOLD_MSA_QUERY_MANIFEST.json',{'rows':rows,'max_submissions':12,'serial_client':True,'new_generated_sequences_submitted':0})
if __name__=='__main__':main()
