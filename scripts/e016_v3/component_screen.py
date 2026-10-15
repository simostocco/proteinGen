"""All-pool component-equivalent alignment with bounded resumable native batches.
Only pairs already connected by exact/source or accepted edges are skipped.
Same global target, scoring and thresholds: no representative-only approximation.
"""
import os,json,pickle,hashlib,time,collections
from pathlib import Path
import numpy as np
from mmseqs_stream import Reader,Writer,Components
from homology_tools import run,mem
R=Path(os.environ['E016_ARTIFACT_ROOT']);H=R/'homology';FMT='query,target,fident,alnlen,qcov,tcov,evalue,qstart,qend,tstart,tend,qlen,tlen,bits'
def hash_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def screen(pref,querydb,targetdb,parent,out,query_to_node,skip_mode='CONNECTED',batch_size=512):
 assert json.load(open(R/'tests/component_alignment_equivalence/REPORT.json'))['status']=='PASS'
 out=Path(out);out.mkdir(exist_ok=True);c=Components(parent);processed=set();counts=collections.Counter();forests=[]
 # Completed batch records are the journal. No incomplete batch becomes evidence.
 for done in sorted(out.glob('batch-*/DONE.json')):
  d=json.load(open(done));assert hash_file(done.parent/'FOREST.tsv')==d['forest_sha256'];assert not processed.intersection(d['query_keys']);processed.update(d['query_keys']);counts.update(d['counts'])
  for line in (done.parent/'FOREST.tsv').open():
   x=line.split('\t');c.link(int(x[0]),int(x[1]))
  forests.append(done.parent/'FOREST.tsv')
 reader=Reader(pref);assert set(query_to_node)=={x[0] for x in reader.index},'all native prefilter query keys required';pending=[];roots=np.asarray([c.find(i) for i in range(len(parent))],dtype=np.int64);batchnum=len(forests);start=time.monotonic()
 def flush(items,roots):
  nonlocal batchnum
  batchnum+=1;base=out/f'batch-{batchnum:05d}';dest=base;attempt=1
  while dest.exists():attempt+=1;dest=out/(base.name+f'-attempt{attempt:03d}')
  dest.mkdir();writer=Writer(dest/'filtered');local=collections.Counter();keys=[]
  for q,body in items:
   keys.append(q);node=query_to_node[q];values=np.fromstring(body.decode('ascii'),dtype=np.int64,sep=' ');assert values.size%3==0;values=values.reshape(-1,3);assert np.all(values[:,0]>=0) and np.all(values[:,0]<len(parent));lines=body.splitlines(keepends=True);assert len(lines)==len(values)
   keep=roots[values[:,0]]!=roots[node];local['prefilter_pairs_seen']+=len(values);local['already_connected_pairs_skipped']+=int((~keep).sum());local['cross_component_candidates_aligned']+=int(keep.sum());writer.add(q,b''.join(lines[i] for i in np.flatnonzero(keep)))
  writer.close();local['query_records']=len(items);forest=dest/'FOREST.tsv'
  if local['cross_component_candidates_aligned']:
   common=['--min-seq-id','0.3','--seq-id-mode','0','--alignment-mode','3','--min-aln-len','20','-c','0','--cov-mode','0','-e','0.001','--max-accept','2147483647','--max-rejected','2147483647','--threads','4','--compressed','1','--max-seq-len','65535']
   run(out.name+'_'+dest.name+'_align',['/usr/bin/mmseqs','align',str(querydb),str(targetdb),str(dest/'filtered'),str(dest/'result'),*common])
   run(out.name+'_'+dest.name+'_convert',['/usr/bin/mmseqs','convertalis',str(querydb),str(targetdb),str(dest/'result'),str(dest/'EDGES.tsv'),'--format-output',FMT,'--threads','4'])
   with forest.open('x') as f:
    for line in (dest/'EDGES.tsv').open():
     x=line.rstrip('\n').split('\t');assert len(x)==14;a,b=int(x[0]),int(x[1]);assert float(x[2])>=.2995 and int(x[3])>=20 and float(x[6])<=.00100001
     local['accepted_cross_previous_component_alignments']+=1
     if c.link(a,b):f.write(line);local['new_component_union_edges']+=1
  else:forest.write_text('')
  payload={'status':'DONE','query_keys':keys,'counts':dict(local),'forest_sha256':hash_file(forest),'input_native_prefilter':str(pref),'global_target_database':str(targetdb),'source_sha256':hash_file(__file__)}
  temp=dest/'DONE.json.part'
  with temp.open('x') as f:f.write(json.dumps(payload,indent=2));f.flush();os.fsync(f.fileno())
  os.replace(temp,dest/'DONE.json');processed.update(keys);counts.update(local);forests.append(forest)
  if batchnum%10==0:print('component batches',batchnum,'queries',len(processed),'/',len(reader.index),'counts',dict(counts),flush=True)
 for key,body in reader.entries(processed):
  if key in processed:continue
  pending.append((key,body))
  if len(pending)==batch_size:flush(pending,roots);pending=[];roots=np.asarray([c.find(i) for i in range(len(parent))],dtype=np.int64)
 if pending:flush(pending,roots)
 reader.close();assert processed==set(query_to_node)
 report={'status':'COMPLETE_COMPONENT_EQUIVALENT_SCREEN','counts':dict(counts),'query_records':len(processed),'elapsed_seconds_this_attempt':time.monotonic()-start,'scientific_thresholds':'identity30% seqidmode0, minlocal20AA, Evalue1e-3; coverageboth80/shorter80/localdomain annotations; same full target background','skip_proof':'adding an edge whose endpoints are already connected cannot alter connected components; all other native prefilter candidate pairs aligned under unchanged thresholds; no AA-gradient or homology approximation introduced','limitations':'heuristic k-mer search, not all remote homology; skipped within-component pairwise alignment counts/metrics are not measured','all_query_keys_processed':True}
 (out/'COMPLETE.json').write_text(json.dumps(report,indent=2));return report,forests,c

def main():
 pool=pickle.load(open(H/'initial_graph.pkl','rb'));freeze=json.load(open(H/'POOL_FREEZE.json'));pref=Path(str(next((H/'full_scratch').glob('*/pref_0.index')))[:-6]);assert len(Reader(pref).index)==freeze['node_sequences'];N=freeze['node_sequences']
 lookup={int(x.split()[0]):int(x.split()[1]) for x in (H/'all_db.lookup').read_text().splitlines()};assert lookup=={i:i for i in range(N)}
 report,forests,c=screen(pref,H/'all_db',H/'all_db',pool['parent'],H/'component_stream_v1',lookup)
 with (H/'full_edges.tsv').open('x') as out:
  for path in forests:
   with path.open() as f:
    for line in f:out.write(line)
 report.update(pool_sha256=freeze['FASTA_sha256'],tool_version=freeze['tool_version'],edge_format=FMT,full_edges_sha256=hash_file(H/'full_edges.tsv'),edge_storage='accepted union forest plus immutable per-batch native alignment results; not all within-component pairwise alignments',parent_prefilter=str(pref))
 (H/'GLOBAL_SCREEN_COMPLETE.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
