"""Freeze the complete canonical pool and conservative exact/source/fragment graph."""
import os,json,sqlite3,hashlib,time,collections,pickle,resource,shutil
from pathlib import Path
R=Path(os.environ['E016_ARTIFACT_ROOT']);D=R/'homology';D.mkdir(exist_ok=True)
db=sqlite3.connect(R/'tables/unified_v2.sqlite');db.execute('pragma temp_store=FILE');db.execute('pragma cache_size=-262144')
assert db.execute("SELECT 1 FROM stages WHERE name='intake_complete'").fetchone(),'complete intake prerequisite'
if (D/'POOL_FREEZE.json').exists():raise RuntimeError('frozen pool already exists; do not overwrite')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
start=time.monotonic();grouped={}
for seqhash,seq,status,lm,metadata in db.execute('SELECT sequence_hash,sequence,status,lm_eligible,metadata FROM entities WHERE sequence_hash IS NOT NULL'):
 if not seq or len(seq)<20 or set(seq)-set('ACDEFGHIKLMNPQRSTVWY'):continue
 m=json.loads(metadata)
 polymer=(m.get('polymer_type') or '').strip("'\"")
 if polymer and 'polypeptide' not in polymer:
  assert not lm,'non-protein admitted as LM';continue
 g=grouped.setdefault(seqhash,[seq,False,set(),0]);assert g[0]==seq,'canonical hash collision'
 g[1]|=bool(lm and status=='VERIFIED');g[3]+=1
 if lm and status=='VERIFIED':
  m=json.loads(metadata);methods=m.get('methods') or [m.get('method')]
  g[2].update(x for x in methods if x)
hashes=sorted(grouped);idx={h:i for i,h in enumerate(hashes)};seqs=[grouped[h][0] for h in hashes];target=[grouped[h][1] for h in hashes];N=len(hashes)
parent=list(range(N));rank=[0]*N;edgecounts=collections.Counter();forest=[]
def find(i):
 while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
 return i
def link(a,b,kind,detail=None):
 edgecounts[kind]+=1;a0,b0=a,b;a=find(a);b=find(b)
 if a==b:return
 if rank[a]<rank[b] or rank[a]==rank[b] and a>b:a,b=b,a
 parent[b]=a
 if rank[a]==rank[b]:rank[a]+=1
 forest.append([a0,b0,kind,detail])
previous=None;anchor=None
for entry,seqhash in db.execute('SELECT entry_id,sequence_hash FROM entities WHERE sequence_hash IS NOT NULL ORDER BY entry_id,sequence_hash'):
 if seqhash not in idx:continue
 if entry!=previous:anchor=idx[seqhash];previous=entry
 else:link(anchor,idx[seqhash],'SAME_PDB_ENTRY',entry)
# Exact full and observed contiguous-fragment containment, complete prefix-index algorithm.
# Index only20mer prefixes of actual queries; scan every20mer of each full sequence.
queries={}
for i,s in enumerate(seqs):queries.setdefault(s,[]).append(i)
fragment_queries=0
for frag,owner in db.execute('SELECT sequence,owner_hash FROM fragments WHERE fragment_hash!=owner_hash'):
 if owner in idx and len(frag)>=20:
  owners=queries.setdefault(frag,[]);owner_id=idx[owner]
  if owner_id not in owners:owners.append(owner_id)
  fragment_queries+=1
prefixes=collections.defaultdict(list)
for s,owners in queries.items():prefixes[s[:20]].append((s,owners))
for i,s in enumerate(seqs):
 seen=set()
 for pos in range(len(s)-19):
  for q,owners in prefixes.get(s[pos:pos+20],[]):
   if len(q)>len(s) or q in seen:continue
   seen.add(q)
   if s.find(q,pos)>=0:
    for owner in owners:
     if owner!=i:link(owner,i,'EXACT_FULL_OR_CONTIGUOUS_FRAGMENT_CONTAINMENT',hashlib.sha256(q.encode()).hexdigest())
 if i%10000==0:print('exact-containment scanned',i,'/',N,'components',len({find(k) for k in range(N)}),flush=True)
 if shutil.disk_usage(R).free<14*2**30:raise RuntimeError('disk reserve before homology')
components=collections.defaultdict(list)
for i in range(N):components[find(i)].append(i)
weighted=sorted([(sum(target[i] for i in group),len(group)) for group in components.values()],reverse=True)
controls=json.load(open('/mnt/d/Simone/proteinGen_data/e016_v3_blocker_resolution_20261014_v1/NATIVE_CONTROL_PANEL.json'))
# Source entry reservation is authoritative even when a cached author-chain mapping is unavailable.
control_entries=['3FM1','2QMI','6ADZ','4UDM','4O1N'];reserved=set()
for entry in control_entries:
 found=[idx[h] for h, in db.execute('SELECT DISTINCT sequence_hash FROM entities WHERE entry_id=?',(entry,)) if h in idx]
 if not found:raise RuntimeError('native control entry absent from complete sequence pool:'+entry)
 reserved.update(find(i) for i in found)
reserved_targets=sum(target[i] for i in range(N) if find(i) in reserved)
with (D/'all_full_sequences.fasta').open('w') as f:
 for i,s in enumerate(seqs):f.write('>'+str(i)+'\n'+s+'\n')
# No learned statistics are fitted: this is the frozen identity/split graph only.
state={'hashes':hashes,'sequences':seqs,'target':target,'methods':[sorted(grouped[h][2]) for h in hashes],'multiplicity':[grouped[h][3] for h in hashes],'parent':parent,'rank':rank,'control_entries':control_entries}
with (D/'initial_graph.pkl').open('wb') as f:pickle.dump(state,f,protocol=5)
with (D/'exact_source_union_forest.jsonl').open('w') as f:
 for edge in forest:f.write(json.dumps(edge)+'\n')
summary={'status':'FROZEN_COMPLETE_CANONICAL_GRAPH_POOL','node_sequences':N,'LM_target_sequences':sum(target),'shadow_metadata_sequences':N-sum(target),'residues':sum(map(len,seqs)),'max_length':max(map(len,seqs)),'component_count_before_MMseqs':len(components),'largest_components_target_and_all':weighted[:20],'native_control_reserved_target_sequences_before_MMseqs':reserved_targets,'fragment_queries_with_owner':fragment_queries,'exact_source_edge_counts':dict(edgecounts),'FASTA_sha256':sha(D/'all_full_sequences.fasta'),'graph_sha256':sha(D/'initial_graph.pkl'),'full_source_population_complete':True,'canonical_nodes_scope':'all standard20canonical>=20 including quarantined/out-of-range shadows; LM target onlyverified20–500 eligibleentities','fragment_policy':'only actual contiguous full-label observedCA intervals; never concatenated missing internals','domain_screen_policy':{'identity':.3,'Evalue':.001,'min_alignment_residues':20,'both80coverage_class':True,'shorter80_class':True,'additional_significant_local_domain_edges':True,'seq_id_mode':0,'alignment_mode':3,'sensitivity':7.5,'mask':1,'max_seqs':'entiretargetcount; no300candidate truncation'},'tool_version':os.popen('/usr/bin/mmseqs version').read().strip(),'elapsed_seconds':time.monotonic()-start,'RSS_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'free_bytes':shutil.disk_usage(R).free}
(D/'POOL_FREEZE.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True);db.close()
