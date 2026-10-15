"""Fresh ALL-node query prefilter, global target, component-complete independent audit."""
import os,json,pickle,hashlib,sqlite3,collections
from pathlib import Path
import pyarrow.parquet as pq
from homology_tools import run,mem
from component_screen import screen,hash_file
R=Path(os.environ['E016_ARTIFACT_ROOT']);D=R/'homology';proposal=Path(json.load(open(D/'LATEST_PROPOSAL.json'))['path']);out=proposal/'independent';out.mkdir(exist_ok=True);assert not (out/'REPORT.json').exists(),'independent report already sealed'
pool=pickle.load(open(D/'initial_graph.pkl','rb'));freeze=json.load(open(D/'POOL_FREEZE.json'));rows=pq.read_table(proposal/'split_assignments.parquet').to_pylist();byid={r['node_id']:r for r in rows};byhash={r['sequence_id']:r for r in rows};N=len(rows);assert N==freeze['node_sequences']
fresh=D/'independent_all_pool_v1';fresh.mkdir(exist_ok=True)
if not (fresh/'INPUT.json').exists():
 with (fresh/'queries.fasta').open('x') as f:
  for i in reversed(range(N)):f.write('>'+str(i)+'\n'+pool['sequences'][i]+'\n')
 command=['-k','6','-s','7.5','--mask','1','--max-seqs',str(N),'--threads','4','--split-memory-limit',str(int(mem()['MemAvailable']/2)//2**20)+'M','--compressed','1','--max-seq-len','65535']
 (fresh/'INPUT.json').write_text(json.dumps({'pool_graph_sha256':freeze['graph_sha256'],'query_FASTA_sha256':hash_file(fresh/'queries.fasta'),'query_count':N,'parameters':command,'target':'same full global all_db; no E-value background change','query_order':'reverse canonical SHA/node order, all metadata shadows included'},indent=2))
contract=json.load(open(fresh/'INPUT.json'));assert contract['pool_graph_sha256']==freeze['graph_sha256'] and hash_file(fresh/'queries.fasta')==contract['query_FASTA_sha256']
run('independent_all_createdb',['/usr/bin/mmseqs','createdb',str(fresh/'queries.fasta'),str(fresh/'query_db'),'--dbtype','1','--shuffle','0'])
run('independent_all_prefilter',['/usr/bin/mmseqs','prefilter',str(fresh/'query_db'),str(D/'all_db'),str(fresh/'prefilter'),*contract['parameters']])
lookup={int(x.split()[0]):int(x.split()[1]) for x in (fresh/'query_db.lookup').read_text().splitlines()};assert sorted(lookup.values())==list(range(N));assert all(v==N-1-k for k,v in lookup.items())
anchors={};parent=list(range(N))
for row in rows:parent[row['node_id']]=anchors.setdefault(row['homology_group'],row['node_id'])
summary,forests,c=screen(fresh/'prefilter',fresh/'query_db',D/'all_db',parent,out/'candidate_screen',lookup)
violations=[];new_component_relations=[]
def relation(a,b,kind):
 if byid[a]['homology_group']!=byid[b]['homology_group']:new_component_relations.append([a,b,kind])
 if byid[a]['split']!=byid[b]['split'] and 'METADATA_ONLY' not in [byid[a]['split'],byid[b]['split']]:violations.append([a,b,kind])
for path in forests:
 for line in path.open():
  x=line.split('\t');relation(int(x[0]),int(x[1]),'INDEPENDENT_LOCAL_HOMOLOGY_FOREST_EDGE')
queries={}
for r in rows:
 seq=pool['sequences'][r['node_id']];queries.setdefault(seq,set()).add(r['node_id'])
db=sqlite3.connect('file:'+str(R/'tables/unified_v2.sqlite')+'?mode=ro',uri=True)
for seq,owner in db.execute('SELECT sequence,owner_hash FROM fragments WHERE fragment_hash!=owner_hash'):
 if owner in byhash:queries.setdefault(seq,set()).add(byhash[owner]['node_id'])
prefix=collections.defaultdict(list)
for seq,owners in queries.items():prefix[seq[:20]].append((seq,owners))
for targetrow in rows:
 t=targetrow['node_id'];seq=pool['sequences'][t];seen=set()
 for pos in range(len(seq)-19):
  for query,owners in prefix.get(seq[pos:pos+20],[]):
   if query in seen or len(query)>len(seq):continue
   seen.add(query)
   if seq.find(query,pos)>=0:
    for q in owners:relation(q,t,'INDEPENDENT_EXACT_FULL_OR_CONTIGUOUS_FRAGMENT')
entrygroup={};source_violations=0
for entry,seqhash in db.execute('SELECT entry_id,sequence_hash FROM entities WHERE sequence_hash IS NOT NULL'):
 if seqhash not in byhash:continue
 r=byhash[seqhash]
 if entry in entrygroup and entrygroup[entry]!=r['homology_group']:source_violations+=1
 entrygroup[entry]=r['homology_group']
for r in rows:
 if r['role']=='NATIVE_AF_CALIBRATION':assert r['split']=='VALIDATION'
report={'status':'PASS' if not violations and not source_violations and not new_component_relations else 'REPAIR_REQUIRED','independent_screen_complete':True,'query_scope_all_canonical_nodes':N,'eligible_unique_LM_nodes':sum(r['lm_eligible'] for r in rows),'detected_cross_split_forest_or_exact_relations':len(violations),'new_component_relations':len(new_component_relations),'source_entry_cross_group_violations':source_violations,'exact_canonical_overlap':len(rows)-len(byhash),'all_exact_full_and_contiguous_fragment_relations':'independently exhaustively scanned all nodes and actual fragments; no gap concatenation','native_calibration_excluded_TRAIN_TEST':True,'all_three_split_pairs_and_both_directions':'fresh all-node reversed query DB against same entire global target; every cross-component candidate scored; within already connected pairs skipped by tested exact connectivity identity','all_remote_homology_guarantee':False,'component_screen_summary':summary,'count_scope':'alignment forest witness edges and exhaustive exact/source checks; not a census of all within-component pairwise alignments','prefilter_input_contract':str(fresh/'INPUT.json')}
(out/'REPORT.json').write_text(json.dumps(report,indent=2))
if violations or new_component_relations:
 with (D/'independent_repair_edges.tsv').open('a') as f:
  for a,b,kind in violations+new_component_relations:f.write(f'{a}\t{b}\t{kind}\n')
print(json.dumps(report),flush=True)
