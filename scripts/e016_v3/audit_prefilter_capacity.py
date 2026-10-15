"""Verify actual result-list capacity and uniqueness, rather than assume --max-seqs."""
import os,json,time,collections,resource
from pathlib import Path
import numpy as np
from mmseqs_stream import Reader
R=Path(os.environ['E016_ARTIFACT_ROOT']);H=R/'homology';N=json.load(open(H/'POOL_FREEZE.json'))['node_sequences'];pref=Path(str(next((H/'full_scratch').glob('*/pref_0.index')))[:-6]);reader=Reader(pref);counts=collections.Counter();bad=[];start=time.monotonic();maxhits=0
for q,body in reader.entries():
 x=np.fromstring(body.decode('ascii'),dtype=np.int64,sep=' ');assert x.size%3==0;x=x.reshape(-1,3);unique=np.unique(x[:,0]);counts['queries']+=1;counts['candidate_pairs']+=len(x);maxhits=max(maxhits,len(x));assert np.all(unique>=0) and np.all(unique<N)
 if len(unique)!=len(x):bad.append(q);counts['duplicate_target_queries']+=1
 if len(x)==N:counts['queries_at_maximum_with_entire_target_pool_present']+=int(len(unique)==N)
 assert len(x)<=N
reader.close();assert counts['queries']==N
report={'status':'PASS_NO_RESULT_COUNT_TRUNCATION' if not bad else 'UNKNOWN_DUPLICATE_TARGET_CAPACITY','counts':dict(counts),'max_candidates_per_query':maxhits,'max_seqs_parameter':N,'target_count':N,'duplicate_target_query_ids':bad,'all_target_ids_unique_per_query':not bad,'scope':'complete actual native prefilter output; normal k-mer heuristic misses are not assessed by this capacity audit','elapsed_seconds':time.monotonic()-start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
(H/'PREFILTER_CAPACITY_AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if bad:raise RuntimeError('cannot claim no candidate count truncation')
