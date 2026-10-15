"""Link the completed priority cohort and bounded authoritative regression overlays."""
import os,sys,json,gzip,hashlib,time,sqlite3,subprocess,collections
from pathlib import Path
import numpy as np
import build_unified_catalogue as C
R=C.R;P=C.P;db=C.db;h=C.h;sha=C.sha;AA=set('ACDEFGHIKLMNPQRSTVWY')
def add(row,kind):
 if db.execute('SELECT 1 FROM sources WHERE source_id=?',(h(kind+'|'+row['source_sha256']),)).fetchone():return
 parsed=Path(row['parsed_path']);d=json.load(gzip.open(parsed,'rt'));assert d['source_sha256']==row['source_sha256']
 if row.get('parsed_sha256'):assert sha(parsed)==row['parsed_sha256']
 entry=d['entry_id'];src=d['source_sha256'];source_id=h(kind+'|'+src)
 if db.execute('SELECT 1 FROM sources WHERE source_id=?',(source_id,)).fetchone():return
 db.execute('INSERT INTO sources VALUES (?,?,?,?,?,?)',(source_id,src,entry,d['source_path'],'completed',json.dumps({'kind':kind,'source_revision':d['revision_history'],'parsed_path':str(parsed),'parsed_sha256':sha(parsed),'api_revision_check':d.get('api_revision_check'),'source_sha_validation':'validated acquisition or bounded gold-source hash'})))
 ent={};seqs={}
 for i,e in enumerate(d['entities']):
  key=h(src+'|entity|'+str(e['entity_id']));ent[e['entity_id']]=key;seq=e['canonical_full_sequence'];seqhash=e['full_sha256'] if seq and set(seq)<=AA else None;seqs[e['entity_id']]=seqhash
  old=db.execute('SELECT sequence_hash,status FROM entities WHERE entity_key=?',(key,)).fetchone()
  if old and old[0] and seqhash and old[0]!=seqhash:raise RuntimeError('CANONICAL_SOURCE_ENTITY_CONTRADICTION')
  payload={**e,'raw_coordinate_source':d['source_path'],'source_revision':d['revision_history'],'parsed_entity_ref':str(parsed)+f'#/entities/{i}'}
  if old:
   # Same source bytes, freshly verified entity_poly_seq is authoritative; old weaker
   # cache evidence remains in the sealed source and the exclusion event ledger.
   db.execute('UPDATE entities SET sequence_hash=?,sequence=?,status=?,reason=?,lm_eligible=?,metadata=? WHERE entity_key=?',(seqhash,seq,e['status'],e.get('reason'),int(e['lm_eligible']),json.dumps(payload),key))
  else:db.execute('INSERT INTO entities VALUES (?,?,?,?,?,?,?,?,?,?)',(key,src,entry,e['entity_id'],seqhash,seq,e['status'],e.get('reason'),int(e['lm_eligible']),json.dumps(payload)))
  if e['status']!='VERIFIED':C.exclude('entity:'+key,entry,'entity',e.get('reason') or 'UNKNOWN',payload)
 for i,o in enumerate(d['observations']):
  label=o['label_asym_id'];model=str(o['model_num']);ek=ent[o['entity_id']];seqhash=seqs[o['entity_id']];seq=o['canonical_full_sequence'];mapping=o['residue_mapping']
  if o['status']=='VERIFIED':
   assert [x['label_seq_id'] for x in mapping]==list(range(1,len(seq)+1))
   assert o['observed_ca_sequence']==''.join(seq[x['label_seq_id']-1] for x in mapping if not x['ca_missing'])
  complete=not o['missing_ca_positions'];fullOK=complete and o['status']=='VERIFIED'
  lm=db.execute('SELECT lm_eligible FROM entities WHERE entity_key=?',(ek,)).fetchone()[0]
  bc=bool(fullOK and lm and o['geometry_method_resolution_eligible'] and not o['alternate_selection_ambiguous'] and len(o['continuity_mask'])==len(seq)-1 and all(o['continuity_mask']))
  oid=h('|'.join([src,label,model,'v3_coherent_backbone_v1']))
  old=db.execute('SELECT observation_id FROM observations WHERE source_sha=? AND label_chain=? AND model=?',(src,label,model)).fetchall()
  # One primary selected observation per experimental chain/model; retain old evidence
  # references as aliases, not an extra allegedly independent conformation.
  for (oldid,) in old:db.execute("UPDATE observations SET BC_eligible=0,BC_reason='SUPERSEDED_BY_SAME_SOURCE_AUTHORITATIVE_MAPPING',mapping_status='SUPERSEDED_MAPPING_VIEW' WHERE observation_id=?",(oldid,))
  meta={k:v for k,v in o.items() if k not in ['residue_mapping','backbone_alternates','canonical_full_sequence']};meta.update(parsed_observation_index=i,residue_mapping_ref=str(parsed)+f'#/observations/{i}/residue_mapping',backbone_alternates_ref=str(parsed)+f'#/observations/{i}/backbone_alternates',prior_view_aliases=[x[0] for x in old],source_revision=d['revision_history'],experimental_context_ref=str(parsed)+'#/context_metadata',atom_masks_scope='explicit atoms in backbone alternates and mapping; no fabricated missing atom')
  auth=o['auth_asym_ids'][0] if len(o['auth_asym_ids'])==1 else None
  db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(oid,ek,seqhash,src,entry,label,auth,model,'VERIFIED_LABEL_CORRESPONDENCE' if o['status']=='VERIFIED' else 'QUARANTINE',int(complete),int(bc),'PASS_COMPLETE_FULL_POLYMER_WITH_VERIFIED_C_N_CONTINUITY' if bc else 'FULL_CHAIN_COORDINATE_CONTINUITY_OR_QUALITY_INELIGIBLE',kind,str(parsed),i,json.dumps(meta,separators=(',',':'))))
  if o['status']=='VERIFIED' and seqhash:
   start=None
   miss=[x['ca_missing'] for x in mapping]
   for pos,m in enumerate(miss+[True]):
    if not m and start is None:start=pos
    if m and start is not None:
     if pos-start>=20:
      frag=seq[start:pos];db.execute('INSERT OR IGNORE INTO fragments VALUES (?,?,?)',(h(frag),frag,seqhash))
     start=None
 db.commit()
def main():
 assert C.completed('historical_complete'),'historical source population incomplete'
 assert (R/'acquisition/COMPLETE.json').exists(),'approved priority queue not terminal'
 latest={x['entry_id']:x for x in map(json.loads,(P/'DOWNLOAD_JOURNAL.jsonl').open())};queue={r['entry_id'] for r in json.load(open(P/'catalogs/download_queue.json'))}
 assert queue<=latest.keys()
 (R/'mapping/PRIORITY_JOURNAL_FROZEN.jsonl').write_text(''.join(json.dumps(latest[k],sort_keys=True)+'\n' for k in sorted(queue)))
 for cursor,entry in enumerate(sorted(queue)):
  row=latest[entry]
  if row['status']=='DONE':add(row,'priority_v3')
  else:C.exclude('intake:'+entry,entry,'intake',row.get('reason') or row['status'],row)
  if cursor%100==0:print('priority linked cursor',cursor,'entry',entry,flush=True)
 db.commit()
 # Only known bounded examples, same stable parser, no archive-wide raw parsing.
 targets={'3FM1','2QMI','7Z53','9QEX','6ZK9','6ZKC'};inventory={x['entry_id']:x for x in map(json.loads,(P/'catalogs/legacy_source_inventory.jsonl').open()) if x['entry_id'] in targets}
 output=R/'mapping/regression_sources';output.mkdir(exist_ok=True)
 for entry in sorted(targets):
  inv=inventory[entry];dest=output/(entry+'.json.gz')
  if not dest.exists():
   temp=output/(entry+'_broker.json')
   child=subprocess.run([sys.executable,'-B',str(R/'source/scripts/e016_v3/parse_entry_broker.py'),inv['path'],entry,str(temp)],capture_output=True,text=True,timeout=180)
   (output/(entry+'_broker.log')).write_text(child.stdout+child.stderr)
   if child.returncode:raise RuntimeError('REGRESSION_SOURCE_PARSE_FAILED:'+entry)
   d=json.load(open(temp));assert d['source_sha256']==inv['known_sha256']
   d['source_path']=inv['path'];d['api_revision_check']='UNKNOWN_historical_source'
   with gzip.GzipFile(str(dest)+'.part','wb',mtime=0) as f:f.write(json.dumps(d,separators=(',',':')).encode())
   os.replace(str(dest)+'.part',dest)
  d=json.load(gzip.open(dest,'rt'));add({'parsed_path':str(dest),'source_sha256':d['source_sha256']},'regression_v3')
 if not C.completed('intake_complete'):C.stage('intake_complete',{'priority_terminal':len(queue),'priority_DONE':sum(latest[k]['status']=='DONE' for k in queue),'raw_regression_entries':sorted(targets),'new_raw_archive_copies':0})
 db.execute('CREATE INDEX IF NOT EXISTS obs_hash ON observations(sequence_hash)');db.execute('CREATE INDEX IF NOT EXISTS obs_entity ON observations(entity_key)');db.execute('CREATE INDEX IF NOT EXISTS entity_hash ON entities(sequence_hash)');db.execute('CREATE INDEX IF NOT EXISTS entity_entry ON entities(entry_id)');db.execute('CREATE INDEX IF NOT EXISTS obs_bc ON observations(BC_eligible)');db.commit()
 stats={'source_entries':db.execute('SELECT count(DISTINCT entry_id) FROM sources').fetchone()[0],'entity_rows':db.execute('SELECT count(*) FROM entities').fetchone()[0],'observation_rows':db.execute('SELECT count(*) FROM observations').fetchone()[0],'BC_candidate_observations':db.execute('SELECT count(*) FROM observations WHERE BC_eligible=1').fetchone()[0],'entity_status':dict(db.execute('SELECT status,count(*) FROM entities GROUP BY status'))}
 (R/'mapping/INTAKE_COUNTS.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats),flush=True)
 db.execute('pragma wal_checkpoint(TRUNCATE)');db.close()
if __name__=='__main__':main()
