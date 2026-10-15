"""Resume historical authoritative-cache integration; raw atoms are never copied.
Complete full-polymer proof and label mapping are independent of coordinate coverage.
"""
import os,sys,json,hashlib,sqlite3,time,math,resource,collections,datetime,shutil
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
R=Path(os.environ['E016_ARTIFACT_ROOT']);ROOT=Path('/mnt/d/Simone/proteinGen');CACHE=Path('/mnt/d/Users/Simone Stocco/proteinGen_audits/sequence_data_readiness_raw_full_compact_v1');SIDE=CACHE.parent/'e006_rich_geometry_sidecars_v2';P=Path('/mnt/d/Simone/proteinGen_data/pdb_only_v3_20261013')
sys.path.insert(0,'/mnt/d/Simone/proteinGen_data/thesis_pdb_sequence_audit_v2/reproduce')
from core import cached_canonical,AA3,PARENTS,array
pa.set_cpu_count(1);pa.set_io_thread_count(1)
DB=Path(os.environ.get('E016_UNIFIED_DB',str(R/'tables/unified_v2.sqlite')));db=sqlite3.connect(DB);db.execute('pragma journal_mode=WAL');available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024;memtotal=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemTotal:')))*1024;dbbytes=DB.stat().st_size if DB.exists() else 0;sqlite_cache_kib=int(min(available-.2*memtotal,max(available/3,dbbytes*1.25))/1024);db.execute(f'pragma cache_size=-{sqlite_cache_kib}');db.execute('pragma wal_autocheckpoint=16384');db.execute('pragma temp_store=FILE');db.execute('pragma synchronous=FULL')
for ddl in [
 'CREATE TABLE IF NOT EXISTS stages(name TEXT PRIMARY KEY,payload TEXT)',
 'CREATE TABLE IF NOT EXISTS historical_samples(sample_id TEXT PRIMARY KEY,entry_id TEXT,auth_chain TEXT,resolution REAL)',
 'CREATE TABLE IF NOT EXISTS quality(entry_id TEXT PRIMARY KEY,resolution REAL)',
 'CREATE TABLE IF NOT EXISTS connectivity(key TEXT PRIMARY KEY,source_sha TEXT,sequence_hash TEXT,auth_chain TEXT,model TEXT,coordinates_sha TEXT,sidecar_path TEXT,row_number INT)',
 'CREATE INDEX IF NOT EXISTS connectivity_match ON connectivity(source_sha,sequence_hash,auth_chain,model)',
 'CREATE TABLE IF NOT EXISTS sources(source_id TEXT PRIMARY KEY,source_sha TEXT,entry_id TEXT,path TEXT,status TEXT,metadata TEXT)',
 'CREATE TABLE IF NOT EXISTS entities(entity_key TEXT PRIMARY KEY,source_sha TEXT,entry_id TEXT,entity_id TEXT,sequence_hash TEXT,sequence TEXT,status TEXT,reason TEXT,lm_eligible INT,metadata TEXT)',
 'CREATE TABLE IF NOT EXISTS observations(observation_id TEXT PRIMARY KEY,entity_key TEXT,sequence_hash TEXT,source_sha TEXT,entry_id TEXT,label_chain TEXT,auth_chain TEXT,model TEXT,mapping_status TEXT,complete_CA INT,BC_eligible INT,BC_reason TEXT,coordinate_kind TEXT,coordinate_path TEXT,coordinate_row INT,metadata TEXT)',
 'CREATE TABLE IF NOT EXISTS fragments(fragment_hash TEXT,sequence TEXT,owner_hash TEXT,PRIMARY KEY(fragment_hash,owner_hash))',
 'CREATE TABLE IF NOT EXISTS exclusions(exclusion_id TEXT PRIMARY KEY,entry_id TEXT,role TEXT,reason TEXT,metadata TEXT)',
 'CREATE INDEX IF NOT EXISTS obs_source_chain_model ON observations(source_sha,label_chain,model)',
 'CREATE TABLE IF NOT EXISTS progress(partition INT PRIMARY KEY,payload TEXT)']:
 db.execute(ddl)
def h(x):return hashlib.sha256(x.encode()).hexdigest()
def sha(p):
 d=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):d.update(b)
 return d.hexdigest()
def rows(p,columns=None,batch=64):
 for b in pq.ParquetFile(p).iter_batches(batch_size=batch,columns=columns,use_threads=False):yield from b.to_pylist()
def stage(name,payload):
 db.execute('INSERT INTO stages VALUES (?,?)',(name,json.dumps(payload)));db.commit()
def completed(name):return bool(db.execute('SELECT 1 FROM stages WHERE name=?',(name,)).fetchone())
def guard():
 if shutil.disk_usage(R).free<14*2**30:raise RuntimeError('BLOCKED_DISK_RESERVE_AND_NEXT4GiB_WORKSPACE')
 mem={x.split(':')[0]:int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if len(x.split())>=3}
 if mem['MemAvailable']<.1*mem['MemTotal']:raise RuntimeError('BLOCKED_ACTUAL_HOST_PRESSURE')
def exclude(key,entry,role,reason,meta):db.execute('INSERT OR IGNORE INTO exclusions VALUES (?,?,?,?,?)',(key,entry,role,reason,json.dumps(meta,separators=(',',':'))))
def validate_full_polymer(r,seq):
 labels=[int(x) for x in array(r['label_sequence_ids'])]
 names=array(r['residue_names']);tokens=array(r['residue_tokens']);auth=array(r['auth_sequence_ids']);ins=array(r['insertion_codes'])
 if not r.get('entity_id'):raise ValueError('ENTITY_ID_UNKNOWN')
 if (r.get('polymer_type') or '').strip("'\"")!='polypeptide(L)':raise ValueError('NONSTANDARD_PROTEIN_POLYMER_TYPE')
 if labels!=list(range(1,len(seq)+1)):raise ValueError('FULL_SCHEME_LABEL_COVERAGE_UNKNOWN')
 if not all(len(v)==len(seq) for v in [names,tokens,auth,ins]):raise ValueError('FULL_SCHEME_MAPPING_LENGTH_CONTRADICTION')
 if ''.join(AA3.get(PARENTS.get(n,n),'?') for n in names)!=seq or ''.join(tokens)!=seq:raise ValueError('FULL_MONOMER_CANONICAL_CONTRADICTION')
 return labels,names,tokens,auth,ins
def main():
 start=time.monotonic()
 if not completed('copied_engineering_metadata'):
  db.execute("ATTACH DATABASE ? AS old",(str(R/'tables/unified.sqlite'),))
  for table in ['historical_samples','quality','connectivity']:
   db.execute(f'INSERT OR IGNORE INTO {table} SELECT * FROM old.{table}')
  for name in ['historical_manifest','connectivity']:
   row=db.execute('SELECT payload FROM old.stages WHERE name=?',(name,)).fetchone()
   if row:db.execute('INSERT INTO stages VALUES (?,?)',(name,row[0]))
  db.commit();db.execute('DETACH DATABASE old');stage('copied_engineering_metadata',{'source':'rejected adapter attempt; only verified manifest/connectivity indexes reused; no quarantines copied'})

 if not completed('historical_manifest'):
  mf=ROOT/'data/full/processed_recovery/merged_manifest.parquet'
  for r in rows(mf,['sample_id','pdb_id','chain_id','resolution_angstrom'],batch=1024):
   entry=r['pdb_id'].upper();res=r['resolution_angstrom'];res=float(res) if res is not None and math.isfinite(res) and res>0 else None
   db.execute('INSERT OR IGNORE INTO historical_samples VALUES (?,?,?,?)',(r['sample_id'],entry,r['chain_id'],res))
   if res is not None:db.execute('INSERT INTO quality VALUES (?,?) ON CONFLICT(entry_id) DO UPDATE SET resolution=max(resolution,excluded.resolution)',(entry,res))
  stage('historical_manifest',{'path':str(mf),'sha256':sha(mf)})
 if not completed('connectivity'):
  n=good=0
  cursor=db.execute('SELECT max(sidecar_path) FROM connectivity').fetchone()[0]
  print('resume connectivity after atomically committed file',cursor,'SQLite cache KiB',sqlite_cache_kib,flush=True)
  columns=['sample_id','sequence','model_number','source_sha256','ca_coordinates','ca_mask','chain_continuity_mask','chain_break_mask','peptide_bond_threshold_angstrom']
  # Deterministic old evidence is reused, not old sampling weights or learned normalizers.
  for split in ['train','validation']:
   for path in sorted((SIDE/split).glob('*.parquet')):
    if cursor and str(path)<=cursor:continue
    for i,r in enumerate(rows(path,columns)):
     n+=1;seq=r['sequence'];L=len(seq)
     if not all(r['ca_mask']) or len(r['ca_mask'])!=L or len(r['chain_continuity_mask'])!=L-1 or not all(r['chain_continuity_mask']) or any(r['chain_break_mask']):continue
     row=db.execute('SELECT entry_id,auth_chain FROM historical_samples WHERE sample_id=?',(r['sample_id'],)).fetchone()
     if row is None:raise RuntimeError('SIDECAR_SAMPLE_MISSING_FROM_AUTHORITATIVE_MANIFEST')
     assert r['peptide_bond_threshold_angstrom']==2.0
     xy=np.asarray(r['ca_coordinates'],dtype='<f4');assert xy.shape==(L,3) and np.isfinite(xy).all()
     coord=hashlib.sha256(xy.tobytes()).hexdigest();seqhash=h(seq);key=h('|'.join([r['source_sha256'],seqhash,row[1],str(r['model_number']),coord]))
     db.execute('INSERT OR IGNORE INTO connectivity VALUES (?,?,?,?,?,?,?,?)',(key,r['source_sha256'],seqhash,row[1],str(r['model_number']),coord,str(path),i));good+=1
    db.commit();guard()
  stage('connectivity',{'inspected_this_attempt':n,'unbroken_CA_evidence_rows_this_attempt':good,'total_sidecar_rows':sum(pq.ParquetFile(p).metadata.num_rows for split in ['train','validation'] for p in (SIDE/split).glob('*.parquet')),'deduplicated_evidence_rows':db.execute('SELECT count(*) FROM connectivity').fetchone()[0],'CA_precision':'IEEEfloat32 equality against cached nativeCA','threshold_C_N_A':2.0})
  print('connectivity',n,good,flush=True)
 inventory={}
 for x in map(json.loads,(P/'catalogs/legacy_source_inventory.jsonl').open()):inventory[x['path']]=x
 quality={x:y for x,y in db.execute('SELECT entry_id,resolution FROM quality')}
 commits=sorted((CACHE/'partition_commits').glob('part-*.json'))
 for j,cp in enumerate(commits):
  if db.execute('SELECT 1 FROM progress WHERE partition=?',(j,)).fetchone():continue
  c=json.load(open(cp));source_table=CACHE/c['tables']['source_identity']['path'];chain_table=CACHE/c['tables']['protein_chain_models']['path']
  assert sha(source_table)==c['tables']['source_identity']['sha256'] and sha(chain_table)==c['tables']['protein_chain_models']['sha256'],'SEALED_CACHE_HASH_MISMATCH'
  sources={}
  for s in rows(source_table):
   path=Path(s['source_file']);path=path if path.is_absolute() else ROOT/path
   inv=inventory.get(str(path));assert inv is not None,'SOURCE_NOT_IN_FROZEN_HISTORICAL_INVENTORY'
   st=path.stat();assert st.st_size==inv['bytes'],'RAW_SIZE_CHANGED'
   if inv.get('known_sha256'):assert inv['known_sha256']==s['source_sha256'],'RAW_PRIOR_HASH_CONTRADICTION'
   expected=datetime.datetime.fromisoformat(inv['mtime_utc'].replace('Z','+00:00')).timestamp();assert abs(st.st_mtime-expected)<.001,'RAW_MTIME_CHANGED'
   sources[s['source_id']]={**s,'physical_path':str(path),'entry_id':inv['entry_id']}
   db.execute('INSERT OR IGNORE INTO sources VALUES (?,?,?,?,?,?)',(s['source_id'],s['source_sha256'],inv['entry_id'],str(path),s['parse_status'],json.dumps({'source_id_kind':'legacyphysicalsource','raw_sha_validation':'priorsealedhash+unchangedsize/mtime','inventory':inv,'cache_source_path':str(source_table)},separators=(',',':'))))
   if s['parse_status']!='completed':exclude('source:'+s['source_id'],inv['entry_id'],'source',s['error_type'] or 'SOURCE_PARSE_UNKNOWN',s)
  counts=collections.Counter();cols=pq.ParquetFile(chain_table).schema_arrow.names
  for i,r in enumerate(rows(chain_table,cols)):
   source=sources[r['source_id']];entry=r['pdb_id'].upper();assert entry==source['entry_id'],'SOURCE_ENTRY_ID_CONTRADICTION'
   rawsha=source['source_sha256'];entity=r.get('entity_id');ek=h(rawsha+'|entity|'+str(entity));label=r['label_asym_id'];model=str(r['model_id']);oid=h('|'.join([rawsha,str(label),model,'legacy_CA_occupancy_v1']))
   seq=None;reason=None;fullstatus='VERIFIED';mapping='UNKNOWN';ca_complete=False;bc=False;bc_reason='FULL_MAPPING_UNKNOWN';seqhash=None
   try:seq=cached_canonical(r.get('entity_poly_sequence'));seqhash=h(seq)
   except Exception as ex:fullstatus='QUARANTINE';reason='CANONICAL_FULL_UNKNOWN:'+str(ex)
   if seq is not None:
    try:
     labels,names,tokens,auth,ins=validate_full_polymer(r,seq)
    except Exception as ex:fullstatus='QUARANTINE';reason=str(ex)
   method=r.get('experimental_method');lm=fullstatus=='VERIFIED' and method in ['X-RAY DIFFRACTION','ELECTRON MICROSCOPY','SOLUTION NMR'] and 20<=len(seq)<=500
   if not entity:ek=h(rawsha+'|unknown_entity|'+str(label))
   old=db.execute('SELECT sequence_hash,status,reason FROM entities WHERE entity_key=?',(ek,)).fetchone()
   if old and old[0] is not None and seqhash is not None and old[0]!=seqhash:raise RuntimeError('STRONG_ENTITY_SEQUENCE_CONTRADICTION')
   emeta={'polymer_type':r.get('polymer_type'),'canonicalization':'v3_standard20_mse_v1+verified_v2_CIFlexeme_decoder','native_monomers_ref':str(chain_table)+f'#/row/{i}/residue_names','method':method,'resolution_angstrom':quality.get(entry),'BOS':'deposited_construct_start','natural_gene_termini':'UNKNOWN','native_gene_stability':'UNKNOWN','full_sequence_source':'cached_entity_poly_canonical_field','method_metadata_scope':'cached_experimental_method; additional_methods_not_retained','deposited_lexeme_ref':str(chain_table)+f'#/row/{i}/entity_poly_sequence'}
   if old is None:db.execute('INSERT INTO entities VALUES (?,?,?,?,?,?,?,?,?,?)',(ek,rawsha,entry,str(entity),seqhash,seq,fullstatus,reason,int(lm),json.dumps(emeta,separators=(',',':'))))
   elif old[1]!='VERIFIED' and fullstatus=='VERIFIED':db.execute('UPDATE entities SET status=?,reason=NULL,lm_eligible=? WHERE entity_key=?',('VERIFIED',int(lm),ek))
   if fullstatus!='VERIFIED':exclude('entity:'+ek,entry,'entity',reason,{'cache_path':str(chain_table),'row':i,'entity_id':entity,'sequence_hash':seqhash})
   meta={'experimental_method':method,'resolution_angstrom':quality.get(entry),'atom_mask_scope':'CA masks in cache; other atoms UNKNOWN unless linked sidecar','correspondence_ref':str(chain_table)+f'#/row/{i}','missing_ca_mask_ref':str(chain_table)+f'#/row/{i}/missing_calpha_mask','missing_residue_mask_ref':str(chain_table)+f'#/row/{i}/missing_residue_mask','selected_altlocs_ref':str(chain_table)+f'#/row/{i}/selected_altlocs','source_revision':'UNKNOWN; exact source hash retained','experimental_disorder':'UNKNOWN','observed_order_proxy':False,'chain_mapping_status':r['chain_mapping_status']}
   if fullstatus=='VERIFIED':
    try:
     miss=array(r['missing_calpha_mask']);missing=array(r['missing_residue_mask']);xy=array(r['selected_calpha_coordinates']);assert all(len(v)==len(seq) for v in [miss,missing,xy])
     present=[k for k,m in enumerate(miss) if not m];assert r['atom_sequence']==''.join(seq[k] for k in present)
     assert [k for k,a in enumerate(xy) if a is not None]==present
     if present:assert np.isfinite(np.asarray([xy[k] for k in present],dtype=float)).all()
     mapping='VERIFIED_LABEL_CORRESPONDENCE';ca_complete=not any(miss)
     meta.update(observed_CA_sha256=h(''.join(seq[k] for k in present)),missing_CA_count=sum(miss),missing_residue_count=sum(missing),CA_N_missing=present[0] if present else None,CA_C_missing=len(seq)-present[-1]-1 if present else None,auth_numbering_ref=str(chain_table)+f'#/row/{i}/auth_sequence_ids',insertion_ref=str(chain_table)+f'#/row/{i}/insertion_codes',model_altloc_policy='legacy_CA_occupancy_v1',observed_order_proxy=bool(present))
     # Only true contiguous label intervals: never concatenate across missing internals.
     startpos=None
     for pos,is_missing in enumerate(miss+[True]):
      if not is_missing and startpos is None:startpos=pos
      if is_missing and startpos is not None:
       if pos-startpos>=20:
        frag=seq[startpos:pos];db.execute('INSERT OR IGNORE INTO fragments VALUES (?,?,?)',(h(frag),frag,seqhash))
       startpos=None
     if not lm:bc_reason='SEQUENCE_LM_INELIGIBLE'
     elif not ca_complete:bc_reason='INCOMPLETE_CHAIN_FRAGMENT_EXCLUDED_PRIMARY'
     elif r['chain_mapping_status']!='unambiguous':bc_reason='AUTHOR_MAPPING_UNKNOWN'
     elif r['ambiguous_altloc_position_count'] or r['deterministic_altloc_tie_count']:bc_reason='ALTERNATE_CONFORMER_AMBIGUITY'
     elif not (method=='SOLUTION NMR' or quality.get(entry) is not None and quality[entry]<=(3 if method=='X-RAY DIFFRACTION' else 4)):bc_reason='GEOMETRY_RESOLUTION_UNKNOWN_OR_OUTSIDE_POLICY'
     else:
      coord=hashlib.sha256(np.asarray(xy,dtype='<f4').tobytes()).hexdigest()
      found=db.execute('SELECT sidecar_path,row_number FROM connectivity WHERE source_sha=? AND sequence_hash=? AND auth_chain=? AND model=? AND coordinates_sha=?',(rawsha,seqhash,r['auth_asym_id'],model,coord)).fetchone()
      if found:
       bc=True;bc_reason='PASS_COMPLETE_FULL_POLYMER_WITH_VERIFIED_C_N_CONTINUITY';meta.update(connectivity_ref=found[0]+f'#/row/{found[1]}',backbone_atom_masks_ref=found[0]+f'#/row/{found[1]}',CA_precision='native cached coordinates; matches sidecar at float32 source-artifact precision')
      else:bc_reason='PEPTIDE_CONTINUITY_UNASSESSED_NO_MATCHING_FULL_SIDECAR'
    except (ValueError,AssertionError,TypeError) as ex:mapping='QUARANTINE';bc_reason='MAPPING_CONTRADICTION:'+str(ex);exclude('observation:'+oid,entry,'observation',bc_reason,meta)
   db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(oid,ek,seqhash,rawsha,entry,label,r['auth_asym_id'],model,mapping,int(ca_complete),int(bc),bc_reason,'legacy_cache',str(chain_table),i,json.dumps(meta,separators=(',',':'))))
   counts['observations']+=1;counts['BC_eligible']+=bc;counts[fullstatus]+=1
  payload={'partition':j,'source_count':len(sources),'chain_rows':dict(counts),'source_table_sha256':c['tables']['source_identity']['sha256'],'chain_table_sha256':c['tables']['protein_chain_models']['sha256'],'RSS_peak':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'elapsed_seconds':time.monotonic()-start}
  db.execute('INSERT INTO progress VALUES (?,?)',(j,json.dumps(payload)));db.commit();guard();print(json.dumps(payload),flush=True)
 stage('historical_complete',{'partitions':len(commits),'sources':db.execute('SELECT count(*) FROM sources').fetchone()[0],'entities':db.execute('SELECT count(*) FROM entities').fetchone()[0],'observations':db.execute('SELECT count(*) FROM observations').fetchone()[0],'BC_observations':db.execute('SELECT count(*) FROM observations WHERE BC_eligible=1').fetchone()[0],'elapsed_seconds':time.monotonic()-start})
 db.execute('pragma wal_checkpoint(TRUNCATE)');db.close()
if __name__=='__main__':main()
