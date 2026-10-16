"""Bounded source/sequence consistency inventory; no raw mmCIF reparse."""
from practical_split import *

def check(root):
 p,by,active=load();native=H/'all_db';checked=0
 with native.open('rb') as f:
  for line in (H/'all_db.index').open():
   k,off,n=map(int,line.split());f.seek(off);body=f.read(n);assert body==p['sequences'][k].encode()+b'\n\0';checked+=1
 assert checked==len(active)
 lookup=[tuple(map(int,l.split())) for l in (H/'all_db.lookup').read_text().splitlines()];assert all(k==n for k,n,_ in lookup) and len(lookup)==checked
 sources={};entry_ids=set();source_rows=0
 for batch in pq.ParquetFile(INTAKE/'raw_sources_manifest.parquet').iter_batches(batch_size=2048,columns=['source_sha','entry_id'],use_threads=False):
  for row in batch.to_pylist():sources.setdefault(row['source_sha'],set()).add(row['entry_id']);entry_ids.add(row['entry_id']);source_rows+=1
 entity_reasons=collections.Counter();eligible_entities=0;total_entities=0;entitysha={}
 for batch in pq.ParquetFile(INTAKE/'polymer_entities.parquet').iter_batches(batch_size=2048,use_threads=False):
  for e in batch.to_pylist():
   total_entities+=1
   assert e['source_sha'] in sources and e['entry_id'] in sources[e['source_sha']]
   if e['lm_eligible']:
    eligible_entities+=1;assert e['status']=='VERIFIED' and e['sequence_hash'] in by and e['sequence']==by[e['sequence_hash']]['canonical_full_sequence']
    entitysha[e['entity_key']]=(e['sequence_hash'],e['source_sha'],e['entry_id'])
   else:entity_reasons[str(e['reason'])]+=1
 qualified=0
 for batch in pq.ParquetFile(INTAKE/'chain_observations.parquet').iter_batches(batch_size=2048,columns=['entity_key','sequence_hash','source_sha','entry_id','BC_eligible','mapping_status','complete_CA'],use_threads=False):
  for o in batch.to_pylist():
   if o['BC_eligible']:
    assert entitysha[o['entity_key']]==(o['sequence_hash'],o['source_sha'],o['entry_id']);assert o['complete_CA']==1 and o['mapping_status']=='VERIFIED_LABEL_CORRESPONDENCE';qualified+=1
 exclusions=collections.Counter();roles=collections.Counter()
 for batch in pq.ParquetFile(INTAKE/'exclusion_events.parquet').iter_batches(batch_size=2048,columns=['role','reason'],use_threads=False):
  for e in batch.to_pylist():roles[e['role']]+=1;exclusions[e['reason']]+=1
 ineligible=pq.ParquetFile(INTAKE/'unique_sequences.parquet').metadata.num_rows-len(by)
 rawfiles=[]
 for base in [next((H/'full_scratch').glob('*/pref_0.index')).with_suffix(''),H/'independent_all_pool_v1/prefilter']:
  for f in sorted(base.parent.glob(base.name+'.*')):
   if f.name.rsplit('.',1)[-1].isdigit():rawfiles.append({'path':str(f),'bytes':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns,'verification':'existing immutable candidate index; source certificates/index hashes linked, data parts not rehashed'})
 bounded=[INTAKE/'INTAKE_MANIFEST.json',H/'POOL_FREEZE.json',H/'PREFILTER_CAPACITY_AUDIT.json',H/'all_db',H/'all_db.index',H/'all_db.lookup',H/'independent_all_pool_v1/INPUT.json',H/'independent_all_pool_v1/prefilter.index',next((H/'full_scratch').glob('*/pref_0.index'))]
 hashes={str(f):sha(f) for f in bounded}
 save(root/'SOURCE_INTEGRITY_CHECK.json',{'status':'PASS','native_database_strings_verified':checked,'eligible_sequences':len(by),'eligible_sequence_residues':sum(r['length'] for r in by.values()),'source_rows':source_rows,'distinct_PDB_entries':len(entry_ids),'total_polymer_entity_rows':total_entities,'eligible_polymer_entity_rows':eligible_entities,'noneligible_entity_rows':total_entities-eligible_entities,'noneligible_entity_reason_counts':dict(entity_reasons),'noneligible_canonical_unique_sequences':ineligible,'qualified_joint_rows_owner_verified':qualified,'exclusion_event_count':sum(exclusions.values()),'exclusion_events_by_role':dict(roles),'exclusion_events_by_reason':dict(exclusions),'new_split_exclusions':0,'hashes':hashes,'prefilter_part_metadata':rawfiles,'raw_source_validation':'inherits prior sealed source hashes and mapping checks; raw mmCIF not reparsed or massively rehashed'})
 print('SOURCE INTEGRITY PASS',checked,eligible_entities,qualified,flush=True)
if __name__=='__main__':
 import sys
 check(Path(sys.argv[1]))
