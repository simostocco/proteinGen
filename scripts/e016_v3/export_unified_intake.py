"""Preserve a complete provisional linked intake even if homology gates block."""
import os,json,sqlite3,hashlib,time,collections
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
R=Path(os.environ['E016_ARTIFACT_ROOT']);D=R/'release/intake_v1';D.mkdir(parents=True,exist_ok=True)
assert not (D/'INTAKE_MANIFEST.json').exists(),'intake already sealed'
resume=json.load(open(R/'mapping/INTAKE_EXPORT_PARTIAL_COMMIT.json')) if (R/'mapping/INTAKE_EXPORT_PARTIAL_COMMIT.json').exists() else {}
db=sqlite3.connect('file:'+str(R/'tables/unified_v2.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row;db.execute('BEGIN');assert db.execute("SELECT 1 FROM stages WHERE name='atomic_and_action_QA'").fetchone()
counts={}
for table,name in [('sources','raw_sources_manifest'),('entities','polymer_entities'),('observations','chain_observations'),('exclusions','exclusion_events'),('fragments','contiguous_fragment_links')]:
 if (name+'.parquet') in resume:
  path=D/(name+'.parquet');h=hashlib.sha256()
  with path.open('rb') as f:
   for block in iter(lambda:f.read(2**20),b''):h.update(block)
  assert h.hexdigest()==resume[path.name]['sha256'] and pq.ParquetFile(path).metadata.num_rows==resume[path.name]['rows']
  counts[name]=resume[path.name]['rows'];continue
 columns=db.execute('pragma table_info('+table+')').fetchall();schema=pa.schema([(r['name'],pa.int64() if r['type']=='INT' else pa.string()) for r in columns]);n=0;batch=[]
 with pq.ParquetWriter(D/(name+'.parquet'),schema,compression='zstd') as out:
  for r in db.execute('SELECT * FROM '+table):
   x=dict(r)
   if table=='entities':
    m=json.loads(x['metadata'])
    if 'deposited_lexeme_ref' in m:m['canonical_CIF_lexeme_ref']=m.pop('deposited_lexeme_ref');m['original_deposited_oneletter_string']='UNKNOWN_in_legacy_cache; native three-letter monomers retained'
    x['metadata']=json.dumps(m,separators=(',',':'))
   batch.append(x);n+=1
   if len(batch)==1024:out.write_table(pa.Table.from_pylist(batch,schema=schema));batch=[]
  if batch:out.write_table(pa.Table.from_pylist(batch,schema=schema))
 counts[name]=n;print(name,n,flush=True)
grouped={};entity_status=collections.Counter();methods=collections.Counter();source_entries=set()
for batch in pq.ParquetFile(D/'polymer_entities.parquet').iter_batches(batch_size=1024,columns=['sequence_hash','sequence','lm_eligible','status','metadata']):
 for x in batch.to_pylist():
  entity_status[x['status']]+=1
  if x['sequence_hash'] is not None:
   g=grouped.setdefault(x['sequence_hash'],[x['sequence'],0,0]);assert g[0]==x['sequence'];g[1]+=1;g[2]+=x['lm_eligible']
  if x['lm_eligible']:
   m=json.loads(x['metadata'])
   for item in m.get('methods') or [m.get('method')]:methods[item]+=1
rows=[{'sequence_id':h,'canonical_full_sequence':v[0],'length':len(v[0]),'entity_multiplicity':v[1],'verified_eligible_entity_occurrences':v[2],'lm_eligible':bool(v[2]),'split':'UNASSIGNED_NOT_CERTIFIED'} for h,v in sorted(grouped.items())]
pq.write_table(pa.Table.from_pylist(rows),D/'unique_sequences.parquet',compression='zstd');counts['unique_sequences']=len(rows)
ob_status=collections.Counter();bc_count=0;bc_ids=set();bc_entries=set()
for batch in pq.ParquetFile(D/'chain_observations.parquet').iter_batches(batch_size=2048,columns=['mapping_status','BC_eligible','sequence_hash','entry_id']):
 for x in batch.to_pylist():
  ob_status[x['mapping_status']]+=1
  if x['BC_eligible']:bc_count+=1;bc_ids.add(x['sequence_hash']);bc_entries.add(x['entry_id'])
for batch in pq.ParquetFile(D/'raw_sources_manifest.parquet').iter_batches(batch_size=4096,columns=['entry_id']):source_entries.update(batch.column(0).to_pylist())
quality={'entity_status':dict(entity_status),'observation_mapping_status':dict(ob_status),'BC_eligible_observations':bc_count,'BC_eligible_unique_sequences':len(bc_ids),'BC_contributing_entries':len(bc_entries),'distinct_PDB_entries':len(source_entries),'LM_length_bands':{},'LM_occurrence_methods':dict(methods),'multiplicity_histogram':dict(collections.Counter(r['entity_multiplicity'] for r in rows if r['lm_eligible']))}
for lo,hi in [(20,64),(65,128),(129,256),(257,384),(385,500)]:quality['LM_length_bands'][str(lo)+'-'+str(hi)]=sum(r['lm_eligible'] and lo<=r['length']<=hi for r in rows)
(D/'COUNTS_AND_QUALITY.json').write_text(json.dumps(quality,indent=2));db.commit();db.close()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
manifest={'status':'PROVISIONAL_INTAKE_ONLY_NOT_TRAINABLE','tables':counts,'counts_and_quality':quality,'authoritative_source_population':'historical223709 +priority2418DONE;22priorityresourceexclusions','raw_coordinates_duplicated':False,'homology_and_splits':'pending separate immutable certification','hashes':{p.name:sha(p) for p in D.iterdir() if p.is_file()},'primary_BC_fragment_policy':'complete observed full deposited construct only; no internal fragment falsely BOS-marked as full protein','legacy_caveats':'canonical sequence and full native three-letter monomers/mapping retained; original deposited oneletter text/additionalmethodlist UNKNOWN; natural gene termini UNKNOWN'}
(D/'INTAKE_MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(quality),flush=True)
