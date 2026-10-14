"""Compile a frozen DONE cohort into linked metadata tables. No raw reparse or splits."""
import os,json,gzip,hashlib,sqlite3,resource,time
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
R=Path(os.environ['E016_ARTIFACT_ROOT']); P=Path('/mnt/d/Simone/proteinGen_data/pdb_only_v3_20261013')
AA=set('ACDEFGHIKLMNPQRSTVWY'); start=time.monotonic()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def check_entity(e):
 seq=e.get('canonical_full_sequence')
 if e['status']=='VERIFIED':
  assert seq and len(seq)==e['full_length'] and set(seq)<=AA
  assert hashlib.sha256(seq.encode()).hexdigest()==e['full_sha256']
  assert len(e['native_monomers'])==len(seq)
 if e['lm_eligible']:assert e['status']=='VERIFIED' and 20<=len(seq)<=500 and e['method_allowed']
 return seq

def main():
 rows=[json.loads(x) for x in (R/'PDB_JOURNAL_FROZEN.jsonl').open()];latest={r['entry_id']:r for r in rows};done=[r for r in latest.values() if r['status']=='DONE']
 out=R/'tables/mapped_cohort_v1';out.mkdir(exist_ok=False)
 db=sqlite3.connect(out/'identities.sqlite');db.execute('pragma temp_store=FILE');db.execute('pragma cache_size=-8192')
 for table in ['raw_sources','polymer_entities','chain_observations','structure_views','quarantine']:
  db.execute(f'CREATE TABLE {table}(id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
 db.execute('CREATE TABLE sequences(hash TEXT PRIMARY KEY, sequence TEXT NOT NULL, lm_eligible INTEGER NOT NULL, entity_multiplicity INTEGER NOT NULL)')
 def insert(table,key,row):db.execute(f'INSERT INTO {table} VALUES (?,?)',(key,json.dumps(row,separators=(',',':'))))
 for index,r in enumerate(done):
  source=Path(r['source_path']);parsed=Path(r['parsed_path'])
  assert source.is_file() and source.stat().st_size==r['source_bytes']
  assert str(source.resolve()).startswith(str(P)+'/raw/')
  assert sha(parsed)==r['parsed_sha256']
  d=json.load(gzip.open(parsed,'rt')); assert d['entry_id']==r['entry_id'] and d['source_sha256']==r['source_sha256'] and d['source_path']==str(source)
  sid=r['source_sha256'];insert('raw_sources',sid,{**r,'raw_hash_verification':'reused validated acquisition hash; size/path checked','parsed_hash_reverified':True})
  entities={e['entity_id']:e for e in d['entities']};entkeys={}
  for i,e in enumerate(d['entities']):
   seq=check_entity(e);key=digest([sid,e['entity_id']]);entkeys[e['entity_id']]=key
   insert('polymer_entities',key,{**e,'stable_entity_id':key,'source_sha256':sid,'parsed_path':str(parsed),'parsed_entity_index':i,'split_status':'UNASSESSED'})
   if e['status']=='VERIFIED':
    h=e['full_sha256'];old=db.execute('SELECT sequence FROM sequences WHERE hash=?',(h,)).fetchone()
    if old:assert old[0]==seq,'SEQUENCE_HASH_COLLISION'
    db.execute('INSERT INTO sequences VALUES (?,?,?,1) ON CONFLICT(hash) DO UPDATE SET lm_eligible=max(lm_eligible,excluded.lm_eligible),entity_multiplicity=entity_multiplicity+1',(h,seq,int(e['lm_eligible'])))
   else:insert('quarantine','entity:'+key,{'role':'entity','id':key,'entry_id':r['entry_id'],'status':e['status'],'reason':e.get('reason'),'source_sha256':sid})
  obkeys={}
  for i,o in enumerate(d['observations']):
   key=digest([sid,o['observation_id']]);obkeys[o['observation_id']]=key;e=entities[o['entity_id']]
   if o['status']=='VERIFIED':
    assert e['status']=='VERIFIED' and o['full_sha256']==e['full_sha256'] and o['canonical_full_sequence']==e['canonical_full_sequence']
    positions=[m['label_seq_id'] for m in o['residue_mapping']];assert positions==list(range(1,e['full_length']+1))
    present=[m['label_seq_id'] for m in o['residue_mapping'] if not m['ca_missing']];assert present==o['observed_label_seq_ids']
    assert o['observed_ca_sequence']==''.join(e['canonical_full_sequence'][p-1] for p in present)
    assert o['missing_ca_positions']==[p for p in positions if p not in set(present)]
   meta={k:v for k,v in o.items() if k not in ['residue_mapping','backbone_alternates','canonical_full_sequence']}
   meta.update(stable_observation_id=key,stable_entity_id=entkeys[o['entity_id']],source_sha256=sid,parsed_path=str(parsed),parsed_observation_index=i,residue_mapping_ref=f'{parsed}#/observations/{i}/residue_mapping',atomic_alternates_ref=f'{parsed}#/observations/{i}/backbone_alternates',split_status='UNASSESSED')
   insert('chain_observations',key,meta)
   if o['status']!='VERIFIED':insert('quarantine','observation:'+key,{'role':'observation','id':key,'entry_id':r['entry_id'],'status':o['status'],'reason':'upstream mapping/canonicalization quarantine','source_sha256':sid})
  for i,s in enumerate(d['segments']):
   oid=obkeys[s['observation_id']];o=next(o for o in d['observations'] if o['observation_id']==s['observation_id']);assert o['status']=='VERIFIED'
   positions=list(range(s['first_label_seq_id'],s['last_label_seq_id']+1));assert len(positions)==s['length'] and all(p in o['observed_label_seq_ids'] for p in positions)
   assert s['sequence']==o['canonical_full_sequence'][positions[0]-1:positions[-1]]
   key=digest([sid,s['observation_id'],i]);insert('structure_views',key,{**s,'stable_observation_id':oid,'source_sha256':sid,'coordinate_ref':f'{parsed}#/observations/{d["observations"].index(o)}/residue_mapping','split_status':'UNASSESSED','joint_BC_status':'BLOCKED_FRAGMENT_POLICY_AND_SPLIT_CERTIFICATION'})
  db.commit()
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>2**30:raise MemoryError('TABLE_RAM_CAP')
  if index%50==0:print('mapped',index+1,'/',len(done),flush=True)
 # Stream small typed metadata payload batches; coordinates stay at original references.
 counts={}
 for table in ['raw_sources','polymer_entities','chain_observations','structure_views','quarantine']:
  counts[table]=db.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
  schema=pa.schema([('stable_id',pa.string()),('metadata_json',pa.string())])
  with pq.ParquetWriter(out/(table+'.parquet'),schema) as writer:
   cur=db.execute(f'SELECT id,payload FROM {table} ORDER BY id')
   while batch:=cur.fetchmany(256):writer.write_table(pa.Table.from_pylist([{'stable_id':x[0],'metadata_json':x[1]} for x in batch],schema=schema))
 schema=pa.schema([('sequence_sha256',pa.string()),('canonical_full_sequence',pa.string()),('lm_eligible',pa.bool_()),('entity_multiplicity',pa.int64()),('split_status',pa.string())])
 with pq.ParquetWriter(out/'unique_sequences.parquet',schema) as writer:
  cur=db.execute('SELECT hash,sequence,lm_eligible,entity_multiplicity FROM sequences ORDER BY hash')
  while batch:=cur.fetchmany(256):writer.write_table(pa.Table.from_pylist([dict(zip(schema.names,[h,s,bool(lm),m,'UNASSESSED'])) for h,s,lm,m in batch],schema=schema))
 counts['unique_verified_full_sequences']=db.execute('SELECT count(*) FROM sequences').fetchone()[0]
 counts['unique_lm_eligible_full_sequences']=db.execute('SELECT count(*) FROM sequences WHERE lm_eligible=1').fetchone()[0]
 counts['verified_entity_multiplicity_histogram']={str(m):n for m,n in db.execute('SELECT entity_multiplicity,count(*) FROM sequences GROUP BY entity_multiplicity')}
 counts['eligible_unique_length_bands']={f'{a}-{b}':db.execute('SELECT count(*) FROM sequences WHERE lm_eligible=1 AND length(sequence) BETWEEN ? AND ?',(a,b)).fetchone()[0] for a,b in [(20,64),(65,128),(129,256),(257,384),(385,500)]}
 counts.update(status='PROVISIONAL_MAPPED_COHORT_NOT_CERTIFIED',population='latest DONE records in frozen continuation journal; excludes unfinished priority and unmapped legacy sources',completed_entries=len(done),split_assignments=0,TRAIN_certified=False,raw_coordinate_bytes_duplicated=0,RSS_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,elapsed_seconds=time.monotonic()-start)
 db.close();(out/'SUMMARY.json').write_text(json.dumps(counts,indent=2));print(json.dumps(counts),flush=True)
if __name__=='__main__':main()
