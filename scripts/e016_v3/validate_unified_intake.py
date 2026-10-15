"""Independent sequential foreign-key, canonical identity and proof-reference audit."""
import os,json,hashlib,collections
from pathlib import Path
import pyarrow.parquet as pq
R=Path(os.environ['E016_ARTIFACT_ROOT']);D=R/'release/intake_v1';out=R/'mapping/INTAKE_INTEGRITY_QA.json';assert not out.exists()
def rows(name,cols=None):
 for b in pq.ParquetFile(D/(name+'.parquet')).iter_batches(batch_size=1024,columns=cols,use_threads=False):yield from b.to_pylist()
sources={r['source_sha']:r['entry_id'] for r in rows('raw_sources_manifest',['source_sha','entry_id'])};entities={};reasons=collections.Counter();proof_events={}
for r in rows('exclusion_events'):
 if r['exclusion_id'].startswith('entity:'):proof_events[r['exclusion_id'][7:]]=json.loads(r['metadata'])
proof_reference_defects=[];sequences={}
for r in rows('polymer_entities'):
 assert r['source_sha'] in sources and sources[r['source_sha']]==r['entry_id'];entities[r['entity_key']]=(r['sequence_hash'],r['status'],r['lm_eligible']);m=json.loads(r['metadata'])
 if r['sequence_hash']:
  assert r['sequence'] and set(r['sequence'])<=set('ACDEFGHIKLMNPQRSTVWY');assert hashlib.sha256(r['sequence'].encode()).hexdigest()==r['sequence_hash']
  if r['sequence_hash'] in sequences:assert sequences[r['sequence_hash']]==r['sequence']
  else:sequences[r['sequence_hash']]=r['sequence']
 if r['lm_eligible']:assert r['status']=='VERIFIED' and 20<=len(r['sequence'])<=500
 event=proof_events.get(r['entity_key']);ref=m.get('native_monomers_ref')
 if r['status']=='VERIFIED' and event and ref==str(event.get('cache_path'))+'#/row/'+str(event.get('row'))+'/residue_names':proof_reference_defects.append({'entity_key':r['entity_key'],'entry_id':r['entry_id'],'stale_proof_reference':ref,'prior_rejection':event,'sequence_hash':r['sequence_hash']})
counts=collections.Counter();bounds={};locations={}
for r in rows('chain_observations'):
 assert r['entity_key'] in entities and r['source_sha'] in sources;expected,status,lm=entities[r['entity_key']]
 if r['mapping_status']=='SUPERSEDED_MAPPING_VIEW':assert not r['BC_eligible'];counts['superseded_aliases']+=1;continue
 assert expected==r['sequence_hash'],'current observation/entity sequence identity contradiction'
 assert r['coordinate_kind'] in ['legacy_cache','priority_v3','regression_v3'];path=Path(r['coordinate_path'])
 if str(path) not in locations:
  assert str(path.resolve()).startswith('/mnt/d/') and path.exists();locations[str(path)]=r['coordinate_kind']
 assert locations[str(path)]==r['coordinate_kind'] and r['coordinate_row']>=0
 if r['coordinate_kind']=='legacy_cache':
  if str(path) not in bounds:bounds[str(path)]=pq.ParquetFile(path).metadata.num_rows
  assert r['coordinate_row']<bounds[str(path)]
 if r['BC_eligible']:assert lm and status=='VERIFIED' and r['complete_CA'] and r['mapping_status']=='VERIFIED_LABEL_CORRESPONDENCE';counts['joint_BC_foreign_keys_and_flags_valid']+=1
 counts['current_observations_checked']+=1
report={'status':'PASS' if not proof_reference_defects else 'REPAIR_PROOF_REFERENCES_REQUIRED','source_hashes':len(sources),'entity_rows':len(entities),'exact_canonical_identities_all_lengths':len(sequences),'counts':dict(counts),'entity_proof_reference_defects':proof_reference_defects,'scope':'all sealed intake rows; canonical hashes/strings, source/entity/observation links, source formats/row bounds, BC eligibility flags, and stale upgraded entity-proof references'}
out.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='entity_proof_reference_defects'}),len(proof_reference_defects),flush=True)
