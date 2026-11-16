"""D:-only bounded v2 data corrections. Stages never train or infer."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import argparse,collections,gzip,hashlib,json,resource,shutil,signal,subprocess,sys,time
import pyarrow as pa
import pyarrow.parquet as pq
from core import reconstruct,choose_evidence,array,hash_sequence,length_stratum
REPO=Path('/mnt/d/Simone/proteinGen');HIST=Path('/home/simostocco/proteinGen-causal-rope')
V1=Path('/mnt/d/Simone/proteinGen_data/thesis_pdb_sequence_audit_v1')
RAW=Path('/mnt/d/Users/Simone Stocco/proteinGen_audits/sequence_data_readiness_raw_full_compact_v1')
SIDE=RAW.parent/'e006_rich_geometry_sidecars_v2'
HOM=REPO/'reports/experiments/E007_matrix_sequence_cogeneration/split_homology_audit_v1_publication_correction_v1/mmseqs/identity_30/cluster_assignments.parquet'
CORPUS=Path('/mnt/d/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization/scientific_corrected_protein_v1')
FROZEN=CORPUS.parents[2]/'e012_large_corpus_warmstart_20m_20261013_v1'
pa.set_cpu_count(1);pa.set_io_thread_count(1)
POLICY={'version':'e012_scope_screen_v2','identity':0.30,'coverage':0.80,'coverage_mode':0,'seq_id_mode':0,'alignment_mode':3,'search_type':1,'sensitivity':7.5,'max_seqs':10000,'evalue':0.001,'mask':1,'threads':1,'split_memory_limit':'2G','child_address_space_gib':3,'prepare_rss_stop_mib':1024,'per_search_timeout_seconds':1800,'search_disk_cap_gib':8,'scope':'published accepted-population clustering plus two complement directed searches and full-polymer validation-to-train search; heuristic detection, not exhaustive all-pairs proof','full_reconstruction_max_length':10000,'proposed_lm_full_length':[20,500],'canonicalization':'standard 20 amino acids plus MSE->MET; entity canonical and full scheme positions/monomers must agree; other chemistry unresolved'}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def save(out,name,d):
    p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(d,indent=2,sort_keys=True)+'\n'
    if p.exists():
        if json.loads(p.read_text())!=json.loads(text):raise RuntimeError('refuse overwriting existing v2 artifact: '+str(p))
        return
    p.write_text(text)
def rows(p,cols=None):
    for b in pq.ParquetFile(p).iter_batches(batch_size=64,columns=cols,use_threads=False):
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>1024*1024:raise MemoryError('v2 prepare RSS exceeded 1 GiB stop threshold')
        yield from b.to_pylist()
def jsonrows(p):
    with gzip.open(p,'rt') as f:
        for line in f:yield json.loads(line)
def gz_writer(p):return gzip.GzipFile(filename=str(p),mode='wb',mtime=0)
def load_population():
    rec={}
    for split in ['train','validation']:
        p=HIST/f'outputs/e012_causal_rope_sequence/pilot_v1/{split}.parquet'
        for r in rows(p):rec[r['sample_id']]={'sample_id':r['sample_id'],'split':split,'retained_sequence':r['sequence'],'retained_sha256':hash_sequence(r['sequence']),'retained_length':len(r['sequence'])}
    paths={s:REPO/f'data/full/splits_recovered_all_structures/{s}.parquet' for s in ['train','validation','test']}
    for s,p in paths.items():
        cols=['sample_id','sequence','pdb_id','chain_id','experimental_method','resolution_angstrom','cluster_id','source_file','terminal_trimming_applied','trimmed_n_terminal_residues','trimmed_c_terminal_residues']
        for r in rows(p,cols):
            sid=r['sample_id']
            if s!='test' and sid not in rec:continue
            if s=='test':rec[sid]={'sample_id':sid,'split':'test','retained_sequence':r['sequence'],'retained_sha256':hash_sequence(r['sequence']),'retained_length':len(r['sequence'])}
            if rec[sid]['retained_sequence']!=r['sequence']:raise ValueError('historical membership sequence contradiction')
            rec[sid].update({k:r.get(k) for k in cols if k not in ['sequence','sample_id']})
    assert collections.Counter(r['split'] for r in rec.values())=={'train':231743,'validation':23307,'test':25066}
    return rec,paths
def prepare(out):
    if (out/'PREPARE_COMPLETE.json').exists():raise RuntimeError('v2 prepare output exists; do not overwrite')
    start=time.time();out.mkdir(parents=True,exist_ok=True);save(out,'SCREENING_POLICY_V2.json',POLICY)
    if not (out/'RESOURCE_PLAN.md').exists():(out/'RESOURCE_PLAN.md').write_text('# Bounded v2 resource plan\n\nOne process; 64-row projected Parquet batches; one existing source partition at a time; stop above 1 GiB prepare peak RSS. No raw-mmCIF reparse. Existing partition hashes are verified. All outputs and scratch stay on D:. No historical SQLite writes.\n\nMMseqs searches are directed, never clustering: additional TRAIN to all VALIDATION; additional VALIDATION to all TRAIN; reconstructed full VALIDATION polymers to reconstructed full TRAIN polymers. Exact sequences are deduplicated in search FASTAs; every record is linked back. Policy: 30% identity, 80% both-sequence coverage, seq-id-mode 0, alignment-mode 3, sensitivity 7.5, e-value 0.001, mask 1, max-seqs 10000. One CPU thread, split-memory-limit 2G, child address-space 3 GiB, 8 GiB total search workspace, 1800 seconds per search. Preserve partial output and classify interrupted or saturated queries UNKNOWN. Candidate limits and heuristic sensitivity do not certify exhaustive absence of homology. TEST homology screening is not included; TEST membership remains provisional.\n\nFull polymers with missing/contradictory entity or complete-scheme evidence are UNKNOWN. Retained sequences are not substituted for full polymers. No training or model inference.\n')
    prov=[];v1prov=json.loads((V1/'PROVENANCE_SHA256.json').read_text());v1sha={r['path']:r['sha256'] for r in v1prov['inputs']}
    def verify(p,expected=None,kind='input'):
        digest=sha(p)
        if expected is not None and digest!=expected:raise ValueError('source hash contradiction: '+str(p))
        prov.append({'path':str(p),'sha256':digest,'bytes':p.stat().st_size,'expected_sha256':expected,'kind':kind});return digest
    for n in ['DATASET_AUDIT.md','DATASET_LINEAGE.md','DUPLICATION_REPORT.json','TERMINI_AND_ORDER_AUDIT.json','SPECIFIC_EXAMPLES.json','PROVENANCE_SHA256.json','OUTPUT_SHA256.json']:verify(V1/n,kind='immutable_v1')
    for split in ['train','validation']:
        p=HIST/f'outputs/e012_causal_rope_sequence/pilot_v1/{split}.parquet';verify(p,v1sha[str(p)])
    rec,paths=load_population()
    for p in paths.values():verify(p,v1sha[str(p)])
    accepted={};clusters=collections.defaultdict(lambda:{'train':[],'validation':[]});seqgroups=collections.defaultdict(set)
    verify(HOM,v1sha[str(HOM)])
    for r in rows(HOM,['sample_id','split','sequence_hash','cluster_id']):
        sid=r['sample_id'];assert rec[sid]['split']==r['split'] and rec[sid]['retained_sha256']==r['sequence_hash']
        accepted[sid]=r['cluster_id'];clusters[r['cluster_id']][r['split']].append(sid);seqgroups[r['sequence_hash']].add(r['cluster_id'])
    extra={s:sorted(sid for sid,r in rec.items() if r['split']==s and sid not in accepted) for s in ['train','validation']}
    assert len(extra['train'])==1611 and len(extra['validation'])==146
    save(out,'ADDITIONAL_E012_IDENTITIES.json',{s:[rec[sid] for sid in ids] for s,ids in extra.items()})
    mixed=[{'cluster_id':c,**v} for c,v in sorted(clusters.items()) if v['train'] and v['validation']];assert len(mixed)==280
    save(out,'PUBLISHED_MIXED_GROUPS.json',mixed)
    old_affected={sid for g in mixed for sid in g['validation']}
    # Preserve old assignment semantics, including the unassessed extra training population.
    extra_ids=set(extra['train'])|set(extra['validation'])
    for sid,r in rec.items():r.update(published_cluster_id=accepted.get(sid),published_mixed=sid in old_affected,additional_e012=sid in extra_ids)
    del accepted,clusters,seqgroups,mixed
    expected_source={}
    for p in sorted((SIDE/'eligibility_manifest.parquet').glob('*.parquet')):
        # v1 verified the sealed inventory; verify cache manifest bytes rather than raw files again.
        verify(p,v1sha[str(p)])
        for r in rows(p,['sample_id','source_sha256','eligible']):
            if r['sample_id'] in rec and r['eligible']:expected_source[r['sample_id']]=sys.intern(r['source_sha256'])
    cache_protocol=RAW/'sequence_readiness_protocol.json';verify(cache_protocol);verify(RAW/'run_config.json');verify(REPO/'src/protein_distance_diffusion/evaluation/raw_sequence_readiness.py')
    counts=collections.Counter();fullgroups={};done=set();part_summary=[];mappingdir=out/'structure_observations';mappingdir.mkdir(exist_ok=True)
    pop_path=out/'population_records.jsonl.gz';previous=None
    if pop_path.exists():
        attempt=out/'attempt_001_resource_stop';attempt.mkdir(exist_ok=True)
        backup=attempt/'population_records.jsonl.gz'
        if backup.exists():raise RuntimeError('preserve multiple failed attempts explicitly; do not overwrite')
        shutil.copyfile(Path(__file__),attempt/'run_at_resume.py');pop_path.rename(backup);previous=iter(jsonrows(backup))
        save(out,'PREPARE_RESOURCE_STOP.json',{'status':'first attempt stopped by 1 GiB RSS guard','completed_partition_count':len(list(mappingdir.glob('*.gz'))),'partial_population_preserved':str(backup),'resume_method':'reuse completed v2 observation partitions; stream completed metadata; unchanged scientific policy; no raw reparsing'})
    def count_mapping(r,mapping):
        if mapping:
            h=r['full_sha256'];g=fullgroups.setdefault(h,{'canonical_sequence':mapping['canonical_complete_sequence'],'full_sha256':h,'record_multiplicity':0,'split_counts':collections.Counter(),'method_counts':collections.Counter(),'fragment_records':0,'lm_eligible':r['full_lm_eligible']})
            assert g['canonical_sequence']==mapping['canonical_complete_sequence'];g['record_multiplicity']+=1;g['split_counts'][r['split']]+=1;g['method_counts'][r['experimental_method'] or 'UNKNOWN']+=1
            isfragment=r['retained_sha256']!=h;g['fragment_records']+=isfragment
            counts[(r['split'],'full_verified')]+=1;counts[(r['split'],'fragment')]+=isfragment;counts[(r['split'],'lm_eligible_records')]+=r['full_lm_eligible']
        counts[(r['split'],'records')]+=1
    with gz_writer(pop_path) as population:
        for j,cp in enumerate(sorted((RAW/'partition_commits').glob('part-*.json'))):
            verify(cp,kind='partition_inventory');commit=json.loads(cp.read_text());tables=commit['tables'];wanted=['source_identity','physical_candidates','protein_chain_models','residue_id_convention_evidence']
            for name in wanted:
                e=tables[name];p=RAW/e['path'];verify(p,e['sha256'],kind='sealed_raw_cache');assert pq.ParquetFile(p).metadata.num_rows==e['row_count']
            output=mappingdir/f'part-{j:06d}.jsonl.gz'
            if output.exists():
                if previous is None:raise RuntimeError('existing observations require preserved population resume evidence')
                local=collections.Counter()
                for ob in jsonrows(output):
                    r=next(previous);assert r['sample_id']==ob['sample_id'];sid=rec[r['sample_id']]['sample_id'];done.add(sid)
                    count_mapping(r,ob if ob['status']=='VERIFIED' else None);population.write((json.dumps(r,separators=(',',':'))+'\n').encode());local[ob['status']]+=1
                part_summary.append({'path':str(output.relative_to(out)),'records':sum(local.values()),'status_counts':dict(local),'sha256':sha(output),'reused_completed_v2_partition':True})
                if (j+1)%16==0:print('v2 resumed completed partitions',j+1,'records',len(done),'RSS MiB',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,flush=True)
                continue
            sources={r['source_id']:r for r in rows(RAW/tables['source_identity']['path'])}
            candidates={};by_sample=collections.defaultdict(list)
            for r in rows(RAW/tables['physical_candidates']['path']):
                if r['sample_id'] not in rec or str(r['model_number'])!='1' or not r['author_chain_match']:continue
                candidates[r['physical_candidate_id']]=r;by_sample[r['sample_id']].append(r)
            evidence=collections.defaultdict(list)
            ec=['sample_id','physical_candidate_id','source_id','strong_evidential_match','identity_mapping_available','identity_mapping_ambiguous','author_chain_match','mapped_indices','selected_sequence']
            for r in rows(RAW/tables['residue_id_convention_evidence']['path'],ec):
                if r['sample_id'] in by_sample and r['selected_sequence']==rec[r['sample_id']]['retained_sequence']:evidence[r['sample_id']].append(r)
            needs={};choices={}
            for sid,cs in by_sample.items():
                strong=None;error=None
                try:strong=choose_evidence(evidence[sid])
                except ValueError as e:error=str(e)
                phys={r['physical_candidate_id'] for r in cs}
                if strong is not None and strong[0] in phys:chosen=candidates[strong[0]];indices=list(strong[1])
                elif len(phys)==1 and error is None:chosen=cs[0];indices=None
                else:choices[sid]=(None,None,error or 'author-linked physical candidate ambiguous');continue
                key=(chosen['source_id'],chosen['entity_id'],chosen['auth_asym_id'],chosen['label_asym_id']);needs[key]=None;choices[sid]=(key,indices,error)
            cc=['source_id','pdb_id','entity_id','auth_asym_id','label_asym_id','model_id','entity_poly_sequence','chain_mapping_status','label_sequence_ids','auth_sequence_ids','insertion_codes','residue_names','residue_tokens','atom_sequence','missing_residue_mask','missing_calpha_mask','experimental_method','model_count','modified_residue_counts','unknown_residue_counts','coordinate_model_present','synthetic_model_row']
            for r in rows(RAW/tables['protein_chain_models']['path'],cc):
                key=(r['source_id'],r['entity_id'],r['auth_asym_id'],r['label_asym_id'])
                if str(r['model_id'])=='1' and key in needs:
                    if needs[key] is not None:raise ValueError('duplicate cached model-1 physical identity')
                    needs[key]=r
            output=mappingdir/f'part-{j:06d}.jsonl.gz';local=collections.Counter()
            with gz_writer(output) as obs:
                for sid in sorted(by_sample):
                    if sid in done:raise ValueError('sample appears across multiple source partitions')
                    done.add(sid);r=dict(rec[sid]);key,indices,error=choices[sid];mapping=None;source=None;chain=None
                    if key is not None:
                        source=sources.get(key[0]);chain=needs[key]
                        if source is None or chain is None:error='missing source/model evidence'
                        elif sid in expected_source and source['source_sha256']!=expected_source[sid]:error='source hash contradicts sealed E006 identity'
                        elif source['parse_status']!='completed':error='cached source parse not completed'
                        elif chain['pdb_id']!=r['pdb_id']:error='PDB identity contradiction'
                        elif chain['synthetic_model_row'] or not chain['coordinate_model_present']:error='synthetic/unobserved model row'
                        else:
                            try:mapping=reconstruct(chain,r['retained_sequence'],indices)
                            except ValueError as e:error=str(e)
                    status='VERIFIED' if mapping else 'UNKNOWN'
                    r.update(full_status=status,full_reason=error,full_sha256=mapping['canonical_complete_sha256'] if mapping else None,full_length=len(mapping['canonical_complete_sequence']) if mapping else None,
                      mapping_status='VERIFIED' if mapping and mapping['retained_label_seq_ids'] is not None else 'UNKNOWN',historical_bos_semantics=mapping['historical_bos_semantics'] if mapping else 'UNKNOWN',
                      source_id=source['source_id'] if source else None,source_sha256=source['source_sha256'] if source else None,
                      entity_id=key[1] if key else None,auth_asym_id=key[2] if key else None,label_asym_id=key[3] if key else None,
                      full_lm_eligible=bool(mapping and 20<=len(mapping['canonical_complete_sequence'])<=500),observation_part=str(output.relative_to(out)))
                    count_mapping(r,mapping);local[status]+=1
                    ob={'sample_id':sid,'split':r['split'],'status':status,'reason':error,'source_file':source['source_file'] if source else r.get('source_file'),'source_sha256':r['source_sha256'],'entity_id':r['entity_id'],'auth_asym_id':r['auth_asym_id'],'label_asym_id':r['label_asym_id'],'historical_retained_sequence':r['retained_sequence'],'experimental_method':r['experimental_method'],'resolution_angstrom':r['resolution_angstrom'],'model_number':1,'ensemble_model_count':chain['model_count'] if chain else None,'alternative_model_evidence':tables['protein_chain_models']['path'],'assembly_context':'UNKNOWN: cached chain evidence does not establish biological dependence','disorder_evidence':{'status':'unknown','evidence':[]}}
                    if mapping:ob.update(mapping)
                    obs.write((json.dumps(ob,separators=(',',':'))+'\n').encode());population.write((json.dumps(r,separators=(',',':'))+'\n').encode())
            part_summary.append({'path':str(output.relative_to(out)),'records':sum(local.values()),'status_counts':dict(local),'sha256':sha(output)})
            if (j+1)%16==0:print('v2 cached mapping partitions',j+1,'records',len(done),'RSS MiB',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,flush=True)
        for sid in sorted(set(rec)-done):
            r=rec[sid];r.update(full_status='UNKNOWN',full_reason='no author-linked cached physical candidate',full_sha256=None,full_length=None,mapping_status='UNKNOWN',historical_bos_semantics='UNKNOWN',full_lm_eligible=False,observation_part=None)
            population.write((json.dumps(r,separators=(',',':'))+'\n').encode());counts[(r['split'],'records')]+=1
    lm=[];full_info=[]
    for h,g in sorted(fullgroups.items()):
        info=dict(g);info.update(canonical_length=len(g['canonical_sequence']),length_stratum=length_stratum(len(g['canonical_sequence'])),full_exact_cross_split=len(g['split_counts'])>1,per_record_unique_sequence_weight=1/g['record_multiplicity'],homology_split_eligibility='UNKNOWN pending full-polymer screening and regrouping',full_lm_bos_semantics='deposited_polymer_N_terminus');full_info.append(info)
        if g['lm_eligible']:lm.append(info)
    # Frozen directed-search inputs are compact exact-string deduplicated FASTAs.
    inputs=out/'search_inputs';inputs.mkdir(exist_ok=True)
    fasta={}
    def emit(name,pairs):
        sequences={h:s for h,s in pairs};path=inputs/(name+'.fasta')
        with path.open('w') as f:
            for h,s in sorted(sequences.items()):f.write('>'+h+'\n'+s+'\n')
        fasta[name]={'path':str(path),'sha256':sha(path),'unique_sequences':len(sequences),'residues':sum(map(len,sequences.values()))};return sequences
    emit('retained_train',((r['retained_sha256'],r['retained_sequence']) for r in rec.values() if r['split']=='train'))
    emit('retained_validation',((r['retained_sha256'],r['retained_sequence']) for r in rec.values() if r['split']=='validation'))
    for s in ['train','validation']:emit('additional_'+s,((rec[sid]['retained_sha256'],rec[sid]['retained_sequence']) for sid in extra[s]))
    for s in ['train','validation']:emit('full_'+s,((h,g['canonical_sequence']) for h,g in fullgroups.items() if g['split_counts'][s]))
    population_counts=dict(collections.Counter(r['split'] for r in rec.values()));record_count=len(rec)
    del rec,expected_source,done
    for name,records in [('full_polymer_sequences.parquet',full_info),('proposed_lm_exact_unique.parquet',lm)]:
        writer=None
        try:
            for begin in range(0,len(records),256):
                table=pa.Table.from_pylist(records[begin:begin+256])
                # Freeze nested count fields to explicit JSON strings rather than batch-dependent structs.
                values=[{**r,'split_counts':json.dumps(r['split_counts'],sort_keys=True),'method_counts':json.dumps(r['method_counts'],sort_keys=True)} for r in records[begin:begin+256]]
                table=pa.Table.from_pylist(values)
                if writer is None:writer=pq.ParquetWriter(out/name,table.schema)
                writer.write_table(table)
        finally:
            if writer is not None:writer.close()
    # Canonical complete sequence counts are from authoritative cached constructs, not assumed native genes.
    summary={'population':'historical E012 TRAIN/VALIDATION plus canonical recovered TEST representatives','population_counts':population_counts,
       'counts':{s:{k:v for (sp,k),v in counts.items() if sp==s} for s in ['train','validation','test']},'full_sequence_unique':len(full_info),'full_lm_exact_unique_eligible':len(lm),
       'full_lm_unique_by_historical_split':{s:sum(g['lm_eligible'] and bool(g['split_counts'][s]) for g in fullgroups.values()) for s in ['train','validation','test']},
       'full_lm_cross_split_exact_groups':sum(g['lm_eligible'] and len(g['split_counts'])>1 for g in fullgroups.values()),'full_lm_length_distribution':dict(collections.Counter(length_stratum(len(g['canonical_sequence'])) for g in fullgroups.values() if g['lm_eligible'])),
       'full_lm_multiplicity_histogram':dict(collections.Counter(g['record_multiplicity'] for g in fullgroups.values() if g['lm_eligible'])),
       'prepare_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'elapsed_seconds':time.time()-start,'mapping_partitions':part_summary,'search_inputs':fasta,'published_validation_affected':len(old_affected),'raw_reparse_count':0}
    save(out,'DATASET_COUNTS_V2.json',summary);save(out,'PROVENANCE_V2.json',prov);save(out,'PREPARE_COMPLETE.json',{'population_sha256':sha(pop_path),'policy_sha256':sha(out/'SCREENING_POLICY_V2.json'),'count':record_count,'source_cache_partitions':len(part_summary),'metrics':{'peak_rss_mib':summary['prepare_peak_rss_mib'],'elapsed_seconds':summary['elapsed_seconds']}})
    print('v2 prepare complete',summary['counts'],'eligible unique full polymers',len(lm),flush=True)

def search(out):
    policy=json.loads((out/'SCREENING_POLICY_V2.json').read_text());assert policy==POLICY
    inputs=json.loads((out/'DATASET_COUNTS_V2.json').read_text())['search_inputs'];root=out/'mmseqs_search';root.mkdir(exist_ok=True)
    jobs=[('additional_train_vs_validation','additional_train','retained_validation'),('additional_validation_vs_train','additional_validation','retained_train'),('full_validation_vs_train','full_validation','full_train')]
    env=os.environ.copy();env.update(TMPDIR=str(root),TMP=str(root),TEMP=str(root),SQLITE_TMPDIR=str(root),MMSEQS_NUM_THREADS='1')
    for name,qname,tname in jobs:
        if (root/name/'COMPLETE.json').exists():print('Reuse completed v2 search',name,flush=True);continue
        q=inputs[qname];t=inputs[tname]
        assert sha(q['path'])==q['sha256'] and sha(t['path'])==t['sha256']
        dest=root/name;dest.mkdir(exist_ok=True);tsv=dest/'alignments.tsv'
        cmd=['/usr/bin/mmseqs','easy-search',q['path'],t['path'],str(tsv),str(dest/'tmp'),'--threads','1','--split-memory-limit','2G','-s','7.5','--min-seq-id','0.3','-c','0.8','--cov-mode','0','--seq-id-mode','0','--alignment-mode','3','--search-type','1','--max-seqs','10000','-e','0.001','--mask','1','--remove-tmp-files','0','--format-output','query,target,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen,qcov,tcov,evalue,bits','-v','2']
        save(dest,'COMMAND.json',{'command':cmd,'policy':policy,'query':q,'target':t,'mmseqs_version':subprocess.check_output(['/usr/bin/mmseqs','version'],text=True).strip()})
        def limits():resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        start=time.time();stop=None;peak=0
        with (dest/'search.log').open('w') as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,preexec_fn=limits,start_new_session=True)
            while proc.poll() is None:
                time.sleep(2)
                if time.time()-start>1800:stop='30-minute timeout'
                # Outputs are bounded to an entirely new D: tree; never remove existing caches.
                used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
                if used>8*1024**3:stop='8 GiB search disk cap'
                rss=0
                for pp in Path('/proc').iterdir():
                    if not pp.name.isdigit():continue
                    try:
                        stat=(pp/'stat').read_text().rsplit(')',1)[1].split()
                        if int(stat[2])!=proc.pid:continue
                        status=(pp/'status').read_text();rss+=int(status.split('VmRSS:')[1].split()[0])
                    except (OSError,IndexError,ValueError):pass
                peak=max(peak,rss)
                if rss>2*1024*1024:stop='2 GiB aggregate process-group RSS cap'
                if stop:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=30);break
        rc=proc.wait();report={'return_code':rc,'resource_stop':stop,'elapsed_seconds':time.time()-start,'process_group_peak_rss_mib':peak/1024,'result_path':str(tsv),'result_sha256':sha(tsv) if tsv.exists() else None,'assessed_query_sequences':q['unique_sequences'] if rc==0 and stop is None else 0,'status':'completed' if rc==0 and stop is None else 'UNKNOWN: incomplete search'}
        save(dest,'COMPLETE.json' if rc==0 and stop is None else 'INCOMPLETE.json',report);print(name,report,flush=True)
    save(out,'SEARCH_STAGE_COMPLETE.json',{'jobs':jobs,'all_completed':all((root/n/'COMPLETE.json').exists() for n,_,_ in jobs)})

def main():
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','search','report','finalize']);a.add_argument('--output',type=Path,required=True);args=a.parse_args();out=args.output
    if not str(out.resolve()).startswith('/mnt/d/'):raise ValueError('all v2 outputs must be D:-resident')
    os.environ.update(TMPDIR=str(out),TMP=str(out),TEMP=str(out),SQLITE_TMPDIR=str(out),XDG_CACHE_HOME=str(out/'cache'))
    if args.stage=='prepare':prepare(out)
    elif args.stage=='search':search(out)
    elif args.stage=='report':
        from report import report
        report(out)
    else:
        from finish import finish
        finish(out)
if __name__=='__main__':main()
