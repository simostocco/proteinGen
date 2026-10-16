"""PDB proposal only. Approved 50/80 full + 70/80 shorter >=50 policy.
No context/shadow/control closure. No training admission or raw coordinate writes.
"""
import argparse,collections,hashlib,json,os,pickle,resource,shutil,subprocess,time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from mmseqs_stream import Reader,Writer,Components

OLD=Path('/mnt/d/Simone/proteinGen_data/e016_gpu_unified_pdb_20261015_v1')
INTAKE=OLD/'release/intake_v1'; H=OLD/'homology'
FMT='query,target,nident,alnlen,qstart,qend,tstart,tend,qlen,tlen,evalue,bits'
SPLITS=('TRAIN','VALIDATION','TEST'); BANDS=('20-64','65-128','129-256','257-384','385-500')
GIB=2**30

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 p=Path(p); tmp=p.with_name(p.name+'.part');tmp.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');os.replace(tmp,p)
def band(n):return next(i for i,x in enumerate((64,128,256,384,500)) if n<=x)
def match(nident,alnlen,qs,qe,ts,te,ql,tl):
 # Exact integer tests. Covered residues are inclusive full-sequence coordinate spans.
 qspan=abs(qe-qs)+1;tspan=abs(te-ts)+1
 full=2*nident>=alnlen and 5*qspan>=4*ql and 5*tspan>=4*tl
 shorter_span=qspan if ql<=tl else tspan
 domain=10*nident>=7*alnlen and 5*shorter_span>=4*min(ql,tl) and min(qspan,tspan)>=50
 return full,domain

def load():
 p=pickle.load(open(H/'initial_graph.pkl','rb'))
 rows=pq.read_table(INTAKE/'unique_sequences.parquet').to_pylist(); rows=[r for r in rows if r['lm_eligible']]
 by={r['sequence_id']:r for r in rows};assert len(by)==len(rows)==147769
 strings={}
 for r in rows:
  s=r['canonical_full_sequence'];assert 20<=len(s)<=500 and set(s)<=set('ACDEFGHIKLMNPQRSTVWY') and hashlib.sha256(s.encode()).hexdigest()==r['sequence_id']
  assert s not in strings;strings[s]=r['sequence_id']
 active=np.zeros(len(p['hashes']),dtype=bool)
 for i,(h,s) in enumerate(zip(p['hashes'],p['sequences'])):
  if h in by:assert s==by[h]['canonical_full_sequence'];active[i]=True
 assert int(active.sum())==len(by)
 return p,by,active

def run(root,name,cmd):
 done=root/(name+'.DONE.json')
 if done.exists():
  d=json.load(open(done));assert d['command']==cmd
  return d
 start=time.monotonic();rss=0;minfree=shutil.disk_usage(root).free;minmem=2**64
 with (root/(name+'.log')).open('x') as log:
  proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
  while proc.poll() is None:
   ids=[proc.pid];tree_rss=0
   for child in ids:
    try:
     tree_rss+=int(Path(f'/proc/{child}/status').read_text().split('VmRSS:')[1].split()[0])*1024
     ids.extend(map(int,Path(f'/proc/{child}/task/{child}/children').read_text().split()))
    except (OSError,IndexError):pass
   rss=max(rss,tree_rss)
   mem={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.split(':')[0] in ('MemAvailable','MemTotal')}
   free=shutil.disk_usage(root).free;minfree=min(minfree,free);minmem=min(minmem,mem['MemAvailable'])
   if free<10*GIB+512*2**20 or mem['MemAvailable']<max(512*2**20,.05*mem['MemTotal']):
    proc.terminate();proc.wait();save(root/(name+'.BLOCKED.json'),{'reason':'RESOURCE_PRESSURE','free':free,'mem':mem});raise RuntimeError('resource stop')
   time.sleep(.25)
  if proc.returncode:raise RuntimeError(f'{name} exit {proc.returncode}: see log')
 d={'command':cmd,'seconds':time.monotonic()-start,'child_peak_RSS_bytes':rss,'RSS_scope':'sampled native process tree, interval250ms','minimum_free_bytes':minfree,'minimum_MemAvailable_bytes':minmem,'exit_code':0};save(done,d);return d

def freeze(root):
 manifest=json.load(open(INTAKE/'INTAKE_MANIFEST.json')); hashes={}
 for n in ('unique_sequences.parquet','chain_observations.parquet','polymer_entities.parquet','raw_sources_manifest.parquet','exclusion_events.parquet'):
  actual=sha(INTAKE/n);assert actual==manifest['hashes'][n];hashes[str(INTAKE/n)]=actual
 pf=json.load(open(H/'POOL_FREEZE.json'));assert sha(H/'initial_graph.pkl')==pf['graph_sha256'];assert sha(H/'all_full_sequences.fasta')==pf['FASTA_sha256']
 p,by,active=load()
 policy={'version':'pdb-practical-50-80-domain70-v1','seed':42,'full':{'identity':.5,'query_coverage':.8,'target_coverage':.8},'contained':{'identity':.7,'shorter_coverage':.8,'minimum_aligned_residue_span_on_both':50},'identity_definition':'identical aligned residues / alignment columns including gaps; integer predicates','coverage_definition':'inclusive aligned coordinate span / complete polymer length; gaps disclosed','tool':'MMseqs2 15-6f452+ds-2','retrieval':'reuse complete sensitivity7.5 masked k6 candidate index; all eligible query/target sequences; no candidate cap truncation; heuristic recall not proved','alignment':{'min_seq_id':.3,'min_aln_len':1,'Evalue_ceiling':1e9,'seq_id_mode':0,'alignment_mode':3,'backtrace':True,'threads':4},'no_context_closure':True,'no_shadow_nodes':True,'no_calibration_reservation':True,'target_proportions':[.8,.1,.1],'allocation':'whole sequence groups; deterministic largest normalized group first; minimize squared normalized deficits across sequence and joint identity/observation length + method bins; seed42 tie breaks','training_authorized':False,'input_hashes':hashes,'old_pool_pickle_sha256':pf['graph_sha256'],'eligible_unique_sequences':len(by),'scope':'existing unified intake only; no new biological sources'}
 if (root/'POLICY_FROZEN.json').exists():assert json.load(open(root/'POLICY_FROZEN.json'))==policy
 else:save(root/'POLICY_FROZEN.json',policy)
 (root/'SPLIT_POLICY.md').write_text('# Frozen supervised split policy\n\n'+json.dumps(policy,indent=2)+'\n\nFrozen before new alignment results or split sizes. No universal homology independence claim. Native controls are metadata, without component reservation. Primary joint examples retain existing complete full-deposited-construct eligibility. Natural gene termini are UNKNOWN. No fitting or training.\n')
 print('FROZEN',len(by),'sequences',int(sum(len(s) for s,a in zip(p['sequences'],active) if a)), 'residues',flush=True)

def align(root,limit=None,cross=False):
 assert (root/'POLICY_FROZEN.json').exists();p,by,active=load()
 directory=root/('cross_screen_coverage_v1' if cross else 'alignment_coverage_v1');directory.mkdir(exist_ok=True)
 pref=next((H/'full_scratch').glob('*/pref_0.index')); pref=Path(str(pref)[:-6]);querydb=H/'all_db';qmap=np.arange(len(active))
 if cross:
  base=H/'independent_all_pool_v1';pref=base/'prefilter';querydb=base/'query_db'
  evidence=json.load(open(base/'INPUT.json'));assert sha(base/'queries.fasta')==evidence['query_FASTA_sha256']
  lookup=[tuple(map(int,line.split())) for line in (base/'query_db.lookup').read_text().splitlines()]
  assert len(lookup)==len(active) and {k for k,n,_ in lookup}==set(range(len(active))) and {n for k,n,_ in lookup}==set(range(len(active)))
  qmap=np.empty(len(active),dtype=np.int64)
  for k,n,_ in lookup:qmap[k]=n
  with querydb.open('rb') as dbfile:
   for line in Path(str(querydb)+'.index').open():
    key,offset,length=map(int,line.split());dbfile.seek(offset);assert dbfile.read(length)==p['sequences'][qmap[key]].encode()+b'\n\0','independent query DB identity mismatch'
 reader=Reader(pref)
 assignments=None
 if cross:
  assignments=np.full(len(active),-1,dtype=np.int8);lookup={r['sequence_id']:SPLITS.index(r['split']) for r in pq.read_table(root/'SEQUENCE_ASSIGNMENTS.parquet',columns=['sequence_id','split']).to_pylist()}
  for i,h in enumerate(p['hashes']):
   if active[i]:assignments[i]=lookup[h]
 completed={q for f in directory.glob('batch-*/DONE.json') for q in json.load(open(f))['queries']}
 pending=[];batches=0
 components=Components(range(len(active)))
 if not cross:
  for a,b,ni,al,qs,qe,ts,te,ql,tl,ev,bits in edge_rows(root):
   if any(match(ni,al,qs,qe,ts,te,ql,tl)):components.link(a,b)
 roots=np.array([components.find(i) for i in range(len(active))])
 def flush(items):
  nonlocal batches,roots
  batches+=1; base=directory/f'batch-{items[0][0]:06d}';dest=base;attempt=1
  while dest.exists():attempt+=1;dest=directory/(base.name+f'-attempt{attempt:03d}')
  dest.mkdir(exist_ok=False);w=Writer(dest/'filtered');pairs=0
  for q,body in items:
   values=np.fromstring(body.decode(),sep=' ',dtype=np.int64).reshape(-1,3);node=int(qmap[q]);keep=active[values[:,0]] & (values[:,0]!=node)
   if cross:keep &=assignments[values[:,0]]!=assignments[node]
   else:keep &= roots[values[:,0]]!=roots[node]
   lines=body.splitlines(keepends=True);w.add(q,b''.join(lines[i] for i in np.flatnonzero(keep)));pairs+=int(keep.sum())
  w.close()
  common=['--min-seq-id','0.3' if cross else '0.5','--seq-id-mode','0','--alignment-mode','3','-a','1','--min-aln-len','1','-c','0.75','-e','1000000000','--max-accept','2147483647','--max-rejected','2147483647','--threads','12','--compressed','1']
  count=0;secs=0
  for cov in (1,2):
   a=run(dest,f'align{cov}',['/usr/bin/mmseqs','align',str(querydb),str(H/'all_db'),str(dest/'filtered'),str(dest/f'result{cov}'),*common,'--cov-mode',str(cov)])
   run(dest,f'convert{cov}',['/usr/bin/mmseqs','convertalis',str(querydb),str(H/'all_db'),str(dest/f'result{cov}'),str(dest/f'EDGES{cov}.tsv'),'--format-output',FMT,'--threads','12'])
   count+=sum(1 for _ in (dest/f'EDGES{cov}.tsv').open());secs+=a['seconds']
  size=sum(f.stat().st_size for f in dest.iterdir() if f.is_file())
  save(dest/'DONE.json',{'queries':[q for q,_ in items],'candidate_pairs':pairs,'alignments_in_two_coverage_passes':count,'bytes':size,'alignment_seconds':secs,'tsv_sha256':{f'EDGES{cov}.tsv':sha(dest/f'EDGES{cov}.tsv') for cov in (1,2)}})
  if not cross:
   for cov in (1,2):
    for line in (dest/f'EDGES{cov}.tsv').open():
     v=line.split('\t');aa,bb,ni,al,qs,qe,ts,te,ql,tl=map(int,v[:10])
     if any(match(ni,al,qs,qe,ts,te,ql,tl)):components.link(aa,bb)
   roots=np.array([components.find(i) for i in range(len(active))])
  print('BATCH',directory.name,len(completed)+len(items), 'pairs',pairs,'alignments',count,'bytes',size,'seconds',secs,flush=True);completed.update(q for q,_ in items)
 for q,body in reader.entries(completed):
  if not active[qmap[q]]:continue
  pending.append((q,body))
  if len(pending)==(4096 if cross else 256):
   flush(pending);pending=[]
   if limit and batches>=limit:break
 else:
  if pending:flush(pending)
 reader.close()
 if len(completed)==int(active.sum()):save(directory/'COMPLETE.json',{'queries':len(completed),'batches':len(list(directory.glob('batch-*/DONE.json'))),'all_eligible_queries':True,'cross_only':cross})
 print('PROGRESS',len(completed),'/',int(active.sum()),flush=True)

def edge_rows(root,which='alignment_coverage_v1'):
 for f in sorted((root/which).glob('batch-*/DONE.json')):
  for name,expected in json.load(open(f))['tsv_sha256'].items():
   assert sha(f.parent/name)==expected
   for line in (f.parent/name).open():
    v=line.rstrip().split('\t');a,b,ni,al,qs,qe,ts,te,ql,tl=map(int,v[:10]);yield a,b,ni,al,qs,qe,ts,te,ql,tl,float(v[10]),float(v[11])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','align','cross','allocate','audit','independent','loader','traceback']);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--limit',type=int);args=ap.parse_args()
 if args.stage=='traceback':
  from practical_checks import traceback_check
  traceback_check(args.root)
 elif args.stage in ('independent','loader'):
  from practical_checks import independent_check,loader_check
  {'independent':independent_check,'loader':loader_check}[args.stage](args.root)
 elif args.stage in ('allocate','audit'):
  from practical_allocate import allocate,audit
  {'allocate':allocate,'audit':audit}[args.stage](args.root)
 elif args.stage=='freeze':freeze(args.root)
 else:align(args.root,args.limit,args.stage=='cross')
if __name__=='__main__':main()
