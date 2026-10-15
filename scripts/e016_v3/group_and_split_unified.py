"""Frozen conservative graph grouping and deterministic group-preserving split proposal.
A manifest is not certified by this script: independent cross-split screening follows.
"""
import os,json,pickle,hashlib,collections,sqlite3,time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
R=Path(os.environ['E016_ARTIFACT_ROOT']);D=R/'homology';release=R/'release';release.mkdir(exist_ok=True)
assert (D/'GLOBAL_SCREEN_COMPLETE.json').exists(),'full approved pool must be screened'
freeze=json.load(open(D/'POOL_FREEZE.json'))
with (D/'initial_graph.pkl').open('rb') as f:s=pickle.load(f)
assert hashlib.sha256((D/'initial_graph.pkl').read_bytes()).hexdigest()==freeze['graph_sha256']
hashes=s['hashes'];seqs=s['sequences'];target=s['target'];methods=s['methods'];N=len(hashes);ids={h:i for i,h in enumerate(hashes)};parent=s['parent'];rank=s['rank'];start=time.monotonic();counts=collections.Counter()
def find(i):
 while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
 return i
def union(a,b):
 a=find(a);b=find(b)
 if a==b:return
 if rank[a]<rank[b] or rank[a]==rank[b] and a>b:a,b=b,a
 parent[b]=a
 if rank[a]==rank[b]:rank[a]+=1
for line in (D/'full_edges.tsv').open():
 x=line.rstrip('\n').split('\t');assert len(x)==14
 a,b=int(x[0]),int(x[1]);assert 0<=a<N and 0<=b<N
 if a==b:continue
 ident=float(x[2]);length=int(x[3]);e=float(x[6]);assert ident>=.2995 and length>=20 and e<=.00100001
 qc=(int(x[8])-int(x[7])+1)/int(x[11]);tc=(int(x[10])-int(x[9])+1)/int(x[12])
 kind='BIDIR80' if min(qc,tc)>=.8 else 'SHORTER80' if max(qc,tc)>=.8 else 'SIGNIFICANT_LOCAL_DOMAIN'
 counts[kind]+=1;union(a,b)
# Explicit repair edges from independently screened previous proposals, when present.
if (D/'independent_repair_edges.tsv').exists():
 for line in (D/'independent_repair_edges.tsv').open():
  a,b=map(int,line.split()[:2]);union(a,b);counts['INDEPENDENT_REPAIR']+=1
members=collections.defaultdict(list)
for i in range(N):members[find(i)].append(i)
db=sqlite3.connect('file:'+str(R/'tables/unified_v2.sqlite')+'?mode=ro',uri=True);reserved=set()
for entry in s['control_entries']:
 for (h,) in db.execute('SELECT DISTINCT sequence_hash FROM entities WHERE entry_id=?',(entry,)):
  if h in ids:reserved.add(find(ids[h]))
features=np.zeros((N,9),dtype=float)
for i,(seq,lm) in enumerate(zip(seqs,target)):
 if not lm:continue
 features[i,0]=1
 for j,(a,b) in enumerate([(20,64),(65,128),(129,256),(257,384),(385,500)]):
  if a<=len(seq)<=b:features[i,1+j]=1
 m=[x for x in ['X-RAY DIFFRACTION','ELECTRON MICROSCOPY','SOLUTION NMR'] if x in methods[i]]
 assert m,'eligible sequence without allowed method'
 for j,x in enumerate(['X-RAY DIFFRACTION','ELECTRON MICROSCOPY','SOLUTION NMR']):features[i,6+j]=int(x in m)/len(m)
groups={};vec={};gid={}
for root,items in members.items():
 h=hashlib.sha256()
 for i in sorted(items):h.update(hashes[i].encode()+b'\n')
 gid[root]=h.hexdigest();vec[root]=features[items].sum(0);groups[root]=items
ratios=np.asarray([.8,.1,.1]);globalvec=features.sum(0);goals=ratios[:,None]*globalvec[None];totals=np.zeros((3,9));assignment={}
for root in sorted(reserved,key=lambda r:gid[r]):assignment[root]=1;totals[1]+=vec[root]
order=sorted((r for r in groups if r not in reserved),key=lambda r:(-vec[r][0],hashlib.sha256(('42|'+gid[r]).encode()).hexdigest()))
for root in order:
 if vec[root][0]==0:assignment[root]=-1;continue
 costs=[]
 for split in range(3):
  before=((totals[split]-goals[split])**2/(globalvec+1)).sum();after=((totals[split]+vec[root]-goals[split])**2/(globalvec+1)).sum();costs.append(after-before)
 chosen=min(range(3),key=lambda k:(costs[k],k));assignment[root]=chosen;totals[chosen]+=vec[root]
# Repair a greedy empty-split artifact by moving whole unreserved components only.
# No sequence is separated from its component and calibration groups never move.
nonempty_repairs=[]
for empty in range(3):
 if totals[empty,0]>0:continue
 candidates=[]
 for root,origin in assignment.items():
  if root in reserved or origin<0 or origin==empty or vec[root][0]<=0:continue
  if totals[origin,0]-vec[root][0]<=0:continue
  delta=(((totals[origin]-vec[root]-goals[origin])**2-(totals[origin]-goals[origin])**2)/(globalvec+1)).sum()+(((totals[empty]+vec[root]-goals[empty])**2-(totals[empty]-goals[empty])**2)/(globalvec+1)).sum()
  candidates.append((float(delta),gid[root],root,origin))
 if candidates:
  _,_,root,origin=min(candidates);assignment[root]=empty;totals[origin]-=vec[root];totals[empty]+=vec[root];nonempty_repairs.append({'group':gid[root],'from':origin,'to':empty})
labels=['TRAIN','VALIDATION','TEST'];rows=[];group_rows=[]
for root in sorted(groups,key=lambda r:gid[r]):
 split=labels[assignment[root]] if assignment[root]>=0 else 'METADATA_ONLY';role='NATIVE_AF_CALIBRATION' if root in reserved else 'FROZEN_FINAL_TEST' if split=='TEST' else 'DEVELOPMENT' if split=='VALIDATION' else 'TRAINING' if split=='TRAIN' else 'METADATA_ONLY'
 group_rows.append({'homology_group':gid[root],'split':split,'role':role,'eligible_unique_sequences':int(vec[root][0]),'all_node_sequences':len(groups[root])})
 for i in groups[root]:rows.append({'sequence_id':hashes[i],'node_id':i,'homology_group':gid[root],'split':split,'role':role,'lm_eligible':bool(target[i]),'length':len(seqs[i])})
rows.sort(key=lambda r:r['sequence_id'])
# Immutable proposal artifacts are versioned by the exact complete edge file hash.
edgehasher=hashlib.sha256()
with (D/'full_edges.tsv').open('rb') as f:
 for b in iter(lambda:f.read(2**20),b''):edgehasher.update(b)
version=edgehasher.hexdigest()[:12]
if (D/'independent_repair_edges.tsv').exists():version+='_'+hashlib.sha256((D/'independent_repair_edges.tsv').read_bytes()).hexdigest()[:8]
proposal=release/('split_proposal_'+version)
if proposal.exists():raise RuntimeError('new immutable proposal required after repair')
proposal.mkdir();pq.write_table(pa.Table.from_pylist(rows),proposal/'split_assignments.parquet');pq.write_table(pa.Table.from_pylist(group_rows),proposal/'homology_groups.parquet')
control_groups=[g for g in group_rows if g['role']=='NATIVE_AF_CALIBRATION'];counts_by_split={label:int(totals[k,0]) for k,label in enumerate(labels)}
report={'status':'PROPOSED_NOT_CERTIFIED','counts_unique_LM':counts_by_split,'eligible_total':sum(target),'target_ratios':[.8,.1,.1],'actual_ratios':{k:v/max(1,sum(target)) for k,v in counts_by_split.items()},'length_and_method_stratum_counts':totals.tolist(),'group_count':len(groups),'largest_groups':sorted(group_rows,key=lambda g:-g['eligible_unique_sequences'])[:20],'native_control_groups':control_groups,'MMseqs_union_forest_edge_counts':dict(counts),'all_source_and_contiguous_fragment_links':True,'independent_cross_split_screen':'NOT_YET_RUN','whole_component_nonempty_repairs':nonempty_repairs,'nonempty_splits':all(v>0 for v in counts_by_split.values()),'source_entry_control_policy':'all sequence-connected control groups stay in VALIDATION/calibration role; no TRAIN or finalTEST','limitations':'components are conservative screening/source/context groups, not proof all members share a biological fold; heuristic missed remote homology remains possible','elapsed_seconds':time.monotonic()-start,'proposal_path':str(proposal)}
(proposal/'SPLIT_PROPOSAL_REPORT.json').write_text(json.dumps(report,indent=2));(D/'LATEST_PROPOSAL.json').write_text(json.dumps({'path':str(proposal)},indent=2));print(json.dumps(report),flush=True)
if not report['nonempty_splits']:raise RuntimeError('BLOCKED_SCIENTIFIC_POLICY_CONNECTED_COMPONENTS_LEAVE_EMPTY_SPLIT; do not break groups or move controls into TRAIN')
