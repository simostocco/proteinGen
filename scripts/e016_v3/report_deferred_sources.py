import os,json,sqlite3,collections
from pathlib import Path
R=Path(os.environ['E016_ARTIFACT_ROOT']);P=Path('/mnt/d/Simone/proteinGen_data/pdb_only_v3_20261013');db=sqlite3.connect('file:'+str(R/'tables/unified_v2.sqlite')+'?mode=ro',uri=True);seqres=sqlite3.connect('file:'+str(P/'catalogs/discovery.sqlite')+'?mode=ro',uri=True)
latest={r['entry_id']:r for r in map(json.loads,(R/'mapping/PRIORITY_JOURNAL_FROZEN.jsonl').open())};out=[]
entries=sorted(e for e,r in latest.items() if r['status']!='DONE');official=collections.defaultdict(list)
for record in seqres.execute('SELECT entry_id,identifier,molecule,length,sha256 FROM seqres WHERE entry_id IN ('+','.join('?' for _ in entries)+')',entries):official[record[0]].append(record[1:])
for entry,r in sorted(latest.items()):
 if r['status']=='DONE':continue
 records=[]
 for identifier,molecule,length,seqhash in official[entry]:
  represented=db.execute("SELECT count(*),count(DISTINCT entry_id) FROM entities WHERE sequence_hash=? AND status='VERIFIED'",(seqhash,)).fetchone()
  records.append({'seqres_identifier':identifier,'molecule':molecule,'length':length,'exact_seqres_hash_represented_verified_entity_rows':represented[0],'exact_seqres_hash_represented_entry_count':represented[1]})
 out.append({'entry_id':entry,'status':'RESOURCE_DEFERRED_EXCLUDED_NOT_BIOLOGICAL_REJECTION','user_named_oversized':entry in ['12UT','29IZ'],'journal':r,'official_SEQRES_metadata_only':records,'affected_observations':'UNKNOWN: CIF not successfully parsed within frozen limits','SEQRES_caveat':'discovery-only chain records; not verified polymer entities or experimental observations; lack of exact match does not establish lack of homology'})
(R/'acquisition/DEFERRED_SOURCE_EVIDENCE.json').write_text(json.dumps({'priority_denominator':2440,'deferred_entries':len(out),'entries':out},indent=2));print('deferred',len(out),flush=True)
