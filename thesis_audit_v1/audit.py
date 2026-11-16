"""Read-only historical audit. All outputs and SQLite temporaries use explicit D: root."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import argparse, collections, hashlib, itertools, json, math, resource, sqlite3, subprocess, sys, time
import numpy as np
import pyarrow.parquet as pq

REPO=Path('/mnt/d/Simone/proteinGen')
HIST=Path('/home/simostocco/proteinGen-causal-rope')
SIDE=Path('/mnt/d/Users/Simone Stocco/proteinGen_audits/e006_rich_geometry_sidecars_v2')
PAIR=SIDE.parent/'sequence_geometry_pairing_v1'
FROZEN=Path('/mnt/d/Simone/proteinGen_data/sequence_foundation/e012_large_corpus_warmstart_20m_20261013_v1')
STRATA=[(20,64),(65,128),(129,256),(257,384),(385,500)]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
    return h.hexdigest()
def sh(s): return hashlib.sha256(s.encode()).hexdigest()
def files(p): return sorted(p.rglob('*.parquet')) if p.is_dir() else [p]
def rows(p, cols=None):
    for f in files(p):
        pf=pq.ParquetFile(f)
        selected=None if cols is None else [c for c in cols if c in pf.schema_arrow.names]
        for b in pf.iter_batches(batch_size=256, columns=selected, use_threads=False):
            yield from b.to_pylist()
def clean(v): return str(v).strip().strip("'\"") if str(v).strip() not in ['.','?',''] else None
def metrics(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    a=x-x.mean(0);b=y-y.mean(0);u,_,vt=np.linalg.svd(a.T@b)
    d=np.eye(3);d[-1,-1]=np.linalg.det(u@vt);rot=u@d@vt
    rms=float(np.sqrt(np.mean(np.sum((a@rot-b)**2,axis=1))))
    dx=np.linalg.norm(x[:,None]-x[None,:],axis=-1);dy=np.linalg.norm(y[:,None]-y[None,:],axis=-1)
    mask=np.triu(np.ones(dx.shape,dtype=bool),k=3)
    cx=(dx<8)&mask;cy=(dy<8)&mask;union=(cx|cy).sum()
    return {'rmsd_angstrom':rms,'distance_rmse_angstrom':float(np.sqrt(np.mean((dx[mask]-dy[mask])**2))),
            'contact_jaccard':float((cx&cy).sum()/union) if union else None,'contact_union_count':int(union)}
def source_map(row):
    """Map stored CA to authoritative atom-site labels by unique coordinate + identity match.
    Refuse ambiguous correspondence; author numbering never determines orientation.
    """
    import gemmi
    from protein_distance_diffusion.constants import STANDARD_AA3_TO_1
    p=Path(row['source_path'])
    block=gemmi.cif.read_file(str(p)).sole_block()
    site=block.get_mmcif_category('_atom_site.')
    chain=row['sample_id'].split('_',1)[1]
    candidates=[]
    for i,atom in enumerate(site['label_atom_id']):
        if atom!='CA' or site['auth_asym_id'][i]!=chain or site.get('pdbx_PDB_model_num',['1']*len(site['label_atom_id']))[i]!='1': continue
        comp=site['label_comp_id'][i];aa=STANDARD_AA3_TO_1.get('MET' if comp=='MSE' else comp)
        label=clean(site['label_seq_id'][i])
        if aa is None or label is None: continue
        candidates.append((int(label),aa,np.array([float(site[k][i]) for k in ['Cartn_x','Cartn_y','Cartn_z']]),site['label_entity_id'][i],site['label_asym_id'][i],site['auth_seq_id'][i],site.get('pdbx_PDB_ins_code',['']*len(site['label_atom_id']))[i]))
    coords=np.asarray(row['ca_coordinates']);labels=[];entities=[];asym=[];auth=[];ins=[]
    for aa,xyz,valid in zip(row['sequence'],coords,row['ca_mask']):
        if not valid: raise ValueError('missing CA makes complete mapping inconclusive')
        found=[c for c in candidates if c[1]==aa and np.linalg.norm(c[2]-xyz)<0.001]
        distinct={(c[0],c[3],c[4]) for c in found}
        if len(distinct)!=1: raise ValueError('ambiguous/unmatched coordinate anchor')
        c=found[0];labels.append(c[0]);entities.append(c[3]);asym.append(c[4]);auth.append(c[5]);ins.append(c[6])
    if len(set(entities))!=1 or len(set(asym))!=1 or len(set(labels))!=len(labels): raise ValueError('entity or label ambiguity')
    poly=block.get_mmcif_category('_entity_poly.')
    i=poly['entity_id'].index(entities[0]);full=''.join(poly['pdbx_seq_one_letter_code_can'][i].split())
    full_raw=poly['pdbx_seq_one_letter_code'][i]
    # Canonical entity sequence is required to agree at every mapped residue.
    if any(j<1 or j>len(full) or full[j-1]!=aa for j,aa in zip(labels,row['sequence'])): raise ValueError('canonical entity sequence mismatch')
    increasing=all(a<b for a,b in zip(labels,labels[1:]));gaps=sum(b-a-1 for a,b in zip(labels,labels[1:]) if b>a)
    scheme=block.get_mmcif_category('_pdbx_poly_seq_scheme.')
    entity=block.get_mmcif_category('_entity.')
    context={k:v for k,v in entity.items() if k in ['id','pdbx_description','pdbx_mutation','pdbx_fragment','pdbx_ec','details']}
    expt=block.get_mmcif_category('_exptl.');ref=block.get_mmcif_category('_refine.');em=block.get_mmcif_category('_em_3d_reconstruction.')
    resolution=ref.get('ls_d_res_high',em.get('resolution',[None]))[0]
    assembly=block.get_mmcif_category('_pdbx_struct_assembly_gen.')
    models=set(site.get('pdbx_PDB_model_num',['1']))
    return {'sample_id':row['sample_id'],'entity_id':entities[0],'label_asym_id':asym[0],'auth_asym_id':chain,
      'deposited_full_sequence':full_raw,'canonical_full_sequence':full,'full_hash':sh(full),'retained_hash':sh(row['sequence']),
      'label_seq_ids':labels,'auth_seq_ids':auth,'insertion_codes':ins,'n_to_c':increasing,'internal_label_gaps':gaps,
      'n_terminal_missing':min(labels)-1,'c_terminal_missing':len(full)-max(labels),'full_length':len(full),
      'coordinate_length':len(coords),'method':expt.get('method',[None])[0],'resolution':resolution,
      'model_count':len(models),'entity_metadata':context,'assembly_generation':assembly,
      'unobserved_annotations':block.get_mmcif_category('_pdbx_unobs_or_zero_occ_residues.'),
      'sequence_differences':block.get_mmcif_category('_struct_ref_seq_dif.'),'source_sha256_verified':sha(p)==row['source_sha256']}

def main(out):
    start=time.time();out.mkdir(parents=True,exist_ok=True);os.environ['TMPDIR']=str(out);os.environ['SQLITE_TMPDIR']=str(out)
    def save(name,d): (out/name).write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
    provenance=[]
    def record(p,expected=None):
        p=Path(p);digest=sha(p);provenance.append({'path':str(p),'sha256':digest,'bytes':p.stat().st_size,'expected_sha256':expected,'expected_match':None if expected is None else expected==digest});return digest
    # SQLite holds population membership, not structure matrices; bounded cache and D: temp files.
    db=sqlite3.connect(out/'audit.sqlite');db.execute('PRAGMA cache_size=-32768');db.execute('PRAGMA temp_store=FILE');db.execute('PRAGMA journal_mode=DELETE')
    db.executescript('CREATE TABLE IF NOT EXISTS records(pop TEXT,id TEXT,seq TEXT,h TEXT,pdb TEXT,chain TEXT,n INT,method TEXT,res REAL,tn INT,tc INT,cluster TEXT,w REAL, PRIMARY KEY(pop,id)); CREATE TABLE IF NOT EXISTS eligibility(id TEXT PRIMARY KEY,split TEXT,eligible INT,reason TEXT);')
    if db.execute('SELECT count(*) FROM records').fetchone()[0]: raise RuntimeError('Use a fresh output root/database for reproducibility')
    splitroot=REPO/'data/full/splits_recovered_all_structures'
    populations={name:splitroot/f'{name}.parquet' for name in ['train','validation','test','validation_all_structures','test_all_structures']}
    populations['merged']=REPO/'data/full/processed_recovery/merged_manifest.parquet'
    populations.update({f'e012_{s}':HIST/f'outputs/e012_causal_rope_sequence/pilot_v1/{s}.parquet' for s in ['train','validation']})
    for pop,path in populations.items():
        record(path);batch=[]
        for r in rows(path):
            seq=r['sequence'];sid=r['sample_id'];pdb=r.get('pdb_id');chain=r.get('chain_id')
            # Recover identity from matching authoritative split membership after loading.
            batch.append((pop,sid,seq,sh(seq),pdb,chain,len(seq),r.get('experimental_method'),r.get('resolution_angstrom'),r.get('trimmed_n_terminal_residues'),r.get('trimmed_c_terminal_residues'),r.get('cluster_id'),r.get('sample_weight')))
            if len(batch)>=256: db.executemany('INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',batch);batch=[]
        db.executemany('INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',batch);db.commit();print('Loaded',pop,flush=True)
    db.execute('CREATE INDEX records_hash ON records(pop,h)')
    for pop,source in [('e012_train','train'),('e012_validation','validation')]:
        missing=db.execute('SELECT count(*) FROM records a LEFT JOIN records b ON a.id=b.id AND b.pop=? WHERE a.pop=? AND (b.id IS NULL OR a.seq!=b.seq)',(source,pop)).fetchone()[0]
        if missing: raise RuntimeError(f'{pop}: {missing} membership/sequence mismatches')
        for col in ['pdb','chain','method','res','tn','tc','cluster','w']:
            db.execute(f'UPDATE records AS a SET {col}=(SELECT b.{col} FROM records b WHERE b.id=a.id AND b.pop=?) WHERE a.pop=?',(source,pop))
    for f in files(SIDE/'eligibility_manifest.parquet'):
        record(f);db.executemany('INSERT INTO eligibility VALUES(?,?,?,?)',[(r['sample_id'],r['split'],r['eligible'],r['exclusion_reason']) for r in rows(f,['sample_id','split','eligible','exclusion_reason'])])
    db.commit()
    stats={}
    for pop in populations:
        counts=list(db.execute('SELECT h,count(*),count(DISTINCT pdb),length(seq) FROM records WHERE pop=? GROUP BY h',(pop,)))
        hist=collections.Counter(r[1] for r in counts);total=sum(r[1] for r in counts);unique=len(counts)
        entropy=-sum((r[1]/total)*math.log(r[1]/total) for r in counts)
        stats[pop]={'records':total,'unique_retained_sequences':unique,'excess_duplicate_records':total-unique,'multiplicity_histogram':dict(sorted(hist.items())),
        'max_multiplicity':max(r[1] for r in counts),'effective_inverse_multiplicity_total_weight':unique,'kish_effective_records_with_inverse_multiplicity':unique**2/sum(1/r[1] for r in counts),
        'entropy_effective_sequences_under_uniform_record_sampling':math.exp(entropy),'sequences_repeated_within_entry':sum(r[1]>r[2] for r in counts) if pop not in ['e012_train','e012_validation'] or True else None,
        'sequences_across_entries':sum(r[2]>1 for r in counts),'sum_existing_sample_weights':db.execute('SELECT sum(w) FROM records WHERE pop=?',(pop,)).fetchone()[0]}
        with (out/f'{pop}_sequence_frequencies.tsv').open('w') as f:
            f.write('retained_sequence_sha256\tmultiplicity\tpdb_entries\tlength\tper_record_inverse_multiplicity_weight\tuniform_record_probability\n')
            for h,c,p,n in counts:f.write(f'{h}\t{c}\t{p}\t{n}\t{1/c:.12g}\t{c/total:.12g}\n')
    concentrations={}
    for field,expr in [('length',"CASE WHEN n<=64 THEN '20-64' WHEN n<=128 THEN '65-128' WHEN n<=256 THEN '129-256' WHEN n<=384 THEN '257-384' ELSE '385-500' END"),('method',"coalesce(method,'UNKNOWN')"),('resolution',"CASE WHEN res IS NULL THEN 'UNKNOWN' WHEN res<=1.5 THEN '<=1.5' WHEN res<=2 THEN '(1.5,2]' WHEN res<=3 THEN '(2,3]' ELSE '>3' END"),('chain',"coalesce(chain,'UNKNOWN')"),('terminal_trim',"CASE WHEN coalesce(tn,0)+coalesce(tc,0)>0 THEN 'trimmed' ELSE 'untrimmed_or_missing_metadata' END")]:
        concentrations[field]=[dict(zip(['stratum','records','unique_sequences','records_in_repeated_sequences'],r)) for r in db.execute(f"SELECT {expr},count(*),count(DISTINCT h),sum(h IN (SELECT h FROM records WHERE pop='e012_train' GROUP BY h HAVING count(*)>1)) FROM records WHERE pop='e012_train' GROUP BY {expr}")]
    overlaps={}
    for key in ['id','h','pdb','cluster']:
        overlaps[key]=db.execute(f"SELECT count(*) FROM (SELECT DISTINCT {key} FROM records WHERE pop='e012_train' INTERSECT SELECT DISTINCT {key} FROM records WHERE pop='e012_validation')").fetchone()[0]
    # Actual sidecar rows must exactly match sequence-only projection; recompute E007 filters.
    acceptance=collections.defaultdict(collections.Counter);sidecount=collections.Counter();side_mismatch=0;selected={};panel=[]
    for lo,hi in STRATA:
        for low,high in [(2,2),(3,9),(10,10**9)]:
            groups=list(db.execute("SELECT h,count(*) c FROM records WHERE pop='e012_train' AND n BETWEEN ? AND ? GROUP BY h HAVING c BETWEEN ? AND ? ORDER BY h LIMIT 4",(lo,hi,low,high)))
            for h,c in groups:
                ids=[r[0] for r in db.execute("SELECT id FROM records WHERE pop='e012_train' AND h=? ORDER BY pdb,id LIMIT 4",(h,))]
                panel.append({'hash':h,'multiplicity':c,'length_stratum':[lo,hi],'multiplicity_stratum':[low,high],'sample_ids':ids});selected.update({sid:None for sid in ids})
    protocol=json.loads((SIDE/'protocol.json').read_text())
    for j,item in enumerate(protocol['shards']):
        if item['dataset'] not in ['train','validation']:continue
        f=SIDE/item['path'];record(f,item['sha256']);s=item['dataset']
        for r in rows(f,['sample_id','sequence','ca_mask','chain_continuity_mask','chain_break_mask']):
            sidecount[s]+=1
            hit=db.execute('SELECT seq FROM records WHERE pop=? AND id=?',('e012_'+s,r['sample_id'])).fetchone()
            side_mismatch+=int(hit is None or hit[0]!=r['sequence'])
            ca=r['ca_mask'];continuity=r['chain_continuity_mask'];breaks=r['chain_break_mask'];reasons=[]
            if len(ca)!=len(r['sequence']) or len(continuity)!=len(ca)-1 or breaks!=[not x for x in continuity]:raise ValueError('mask shape/complement defect')
            if not any(ca):reasons.append('no_valid_calpha')
            if not all(ca):reasons.append('missing_calpha')
            if not all(continuity):reasons.append('chain_break')
            acceptance[s]['+'.join(reasons) or 'accepted']+=1
        # Only selected groups materialize CA arrays, at most a single bounded batch.
        if s=='train':
            for r in rows(f,['sample_id','sequence','ca_coordinates','ca_mask','source_path','source_sha256','residue_ids','insertion_codes']):
                if r['sample_id'] in selected:selected[r['sample_id']]=r
        if (j+1)%64==0: print('Sidecars verified',j+1,flush=True)
    mapping=[];mapped={};pairs=[]
    for sid,row in selected.items():
        try:
            if row is None:raise ValueError('sidecar missing')
            m=source_map(row);mapped[sid]=m;mapping.append(m)
        except Exception as e:mapping.append({'sample_id':sid,'status':'INCONCLUSIVE','reason':str(e)})
    for group in panel:
        classifications=[]
        for a,b in itertools.combinations(group['sample_ids'],2):
            result={'a':a,'b':b,'retained_sequence_sha256':group['hash'],'same_entry':db.execute("SELECT pdb FROM records WHERE pop='e012_train' AND id=?",(a,)).fetchone()==db.execute("SELECT pdb FROM records WHERE pop='e012_train' AND id=?",(b,)).fetchone()}
            if a not in mapped or b not in mapped or not mapped[a]['n_to_c'] or not mapped[b]['n_to_c']:
                result.update(classification='inconclusive',reason='authoritative mapping unavailable or nonmonotone')
            elif mapped[a]['full_hash']!=mapped[b]['full_hash'] or mapped[a]['label_seq_ids']!=mapped[b]['label_seq_ids']:
                result.update(classification='inconclusive',reason='full polymers or retained label positions differ; identical cropped sequence does not prove residue correspondence',full_polymer_equal=mapped[a]['full_hash']==mapped[b]['full_hash'])
            else:
                m=metrics(selected[a]['ca_coordinates'],selected[b]['ca_coordinates']);result.update(m,aligned_residues=len(selected[a]['sequence']),full_polymer_equal=True)
                j=m['contact_jaccard'];result['classification']='structurally_similar' if m['rmsd_angstrom']<=2 and j is not None and j>=0.8 else 'structurally_heterogeneous' if m['rmsd_angstrom']>=4 and j is not None and j<=0.7 else 'inconclusive'
            classifications.append(result['classification']);pairs.append(result)
        group['classification']='structurally_heterogeneous' if 'structurally_heterogeneous' in classifications else 'structurally_similar' if classifications and all(c=='structurally_similar' for c in classifications) and group['multiplicity']==len(group['sample_ids']) else 'inconclusive'
    exclusions=[dict(zip(['split','eligible','reason','records'],r)) for r in db.execute('SELECT split,eligible,reason,count(*) FROM eligibility GROUP BY split,eligible,reason')]
    # Small configs and authoritative reports; record complete bytes without printing large protocols.
    refs=[SIDE/'protocol.json',SIDE/'schema.json',SIDE/'vocabulary.json',SIDE/'normalization.json',SIDE/'shard_hashes.sha256',PAIR/'protocol.json',PAIR/'input_hashes.sha256',splitroot/'workflow_metadata.json',splitroot/'split_report.json',splitroot/'leakage_audit.json',splitroot/'split_assignments.parquet',REPO/'data/full/processed_recovery/mmseqs_merged_all_structure_clusters.tsv']
    refs+=list((REPO/'configs').glob('*preprocess*'))+list((REPO/'configs').glob('split_recovered_all*'))+[REPO/'configs/e006_rich_geometry_sidecars.yaml',REPO/'configs/e007_split_homology_audit_v1.yaml']
    refs+=list((REPO/'src/protein_distance_diffusion/data').glob('*.py'))+list((HIST/'src/protein_sequence_generation').glob('*.py'))+[HIST/'scripts/run_e012_causal_rope.py',HIST/'configs/e012_causal_rope_sequence_v1.yaml']
    histreport=HIST/'reports/experiments/E012_causal_rope_sequence/pilot_v1'
    refs+=[histreport/n for n in ['data_integrity.json','results.json','FINAL_REPORT.md','contract.json','historical_model_audit.json']]
    refs+=[FROZEN/n for n in ['protocol.json','source_hashes.json','status.json','pause_verification.json','REPORT.json']]
    for p in refs:
        if p.is_file():record(p)
    for pop,path in [('pair_eligible_train',PAIR/'eligible_train.parquet'),('pair_eligible_validation',PAIR/'eligible_validation.parquet')]:
        count=0
        for f in files(path):record(f);count+=pq.ParquetFile(f).metadata.num_rows
        stats[pop]={'records':count}
    save('PROVENANCE_SHA256.json',{'inputs':provenance,'repository_heads':{str(p):subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip() for p in [REPO,HIST]},'hash_scope':'actual bytes; historical claim hashes are separate expected fields; no checkpoint or 20M corpus rehash'})
    save('DUPLICATION_REPORT.json',{'version':'thesis_pdb_sequence_audit_v1','status':'VERIFIED for manifest counts; structural panel is bounded descriptive evidence','sequence_definition':'exact retained canonical coordinate-associated sequence, not deposited full polymer','populations':stats,'concentrations_e012_train':concentrations,'leakage_e012_train_validation':overlaps,'sidecar_sequence_projection_mismatches':side_mismatch,'sidecar_counts':sidecount,'e006_exclusions':exclusions,'e007_coordinate_acceptance':acceptance,'structural_panel':panel,'structural_pairs':pairs,'pair_class_counts':collections.Counter(r['classification'] for r in pairs),'panel_method':'five length x three multiplicity strata; first four SHA256-sorted groups each; at most first four pdb/id-sorted records each; all within-selected-group pairs. No random-population inference. Complete matching canonical full polymers and identical label_seq_id positions required.','thresholds':{'similar':'RMSD <=2 A and CA contacts (<8 A, sequence separation >=3) Jaccard >=0.8','heterogeneous':'RMSD >=4 A and contact Jaccard <=0.7','other':'inconclusive; operational thresholds do not identify fold families'},'limitations':['No exhaustive full-polymer duplication or full-coordinate diversity census','Panel excludes NMR ensemble diversity beyond historical model 1','Homology clustering overlap is not an exhaustive pairwise homology exclusion guarantee']})
    save('TERMINI_AND_ORDER_AUDIT.json',{'version':'v1','mapping_examples':mapping,'denominators':{'selected_records':len(selected),'authoritatively_mapped':len(mapped),'n_to_c_mapped':sum(m['n_to_c'] for m in mapped.values()),'full_polymer_differs_from_retained':sum(m['full_hash']!=m['retained_hash'] for m in mapped.values()),'internal_gaps_mapped':sum(m['internal_label_gaps']>0 for m in mapped.values())},'scope':'panel; whole-population terminal metadata in DUPLICATION_REPORT','verified_source_behavior':{'mmcif':'numeric canonical label order; scheme provides unresolved positions; reject internal missing CA; recovery trims terminals <=25%; model 1; occupancy/altloc selection','legacy_pdb':'observed residues only, no SEQRES correspondence; unsupported residues skipped; insertion codes omitted from returned IDs; metadata/continuity/terminal policy kwargs ignored'},'unresolved_choices':['Legacy PDB cannot recover unobserved residues solely from coordinate topology','Author-number fallback does not establish polymer orientation','Unknown scheme monomers omitted; strict contiguity can miss unsupported terminal residues','Authoritative full entity sequence and mapping must become required in new dataset']})
    # Plots have standalone exportable SVG/PNG assets and plotted tables in JSON/TSV.
    os.environ['MPLCONFIGDIR']=str(out/'mplconfig')
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    for title,pop in [('Historical E012 TRAIN multiplicity','e012_train'),('Recovered TRAIN multiplicity','train')]:
        hist=stats[pop]['multiplicity_histogram'];fig,ax=plt.subplots(figsize=(6,4));ax.scatter([int(k) for k in hist],list(hist.values()),s=12);ax.set(xscale='log',yscale='log',xlabel='Records per retained exact sequence',ylabel='Number of sequences',title=title);fig.tight_layout();fig.savefig(out/f'{pop}_multiplicity.svg');plt.close(fig)
    valid=[p for p in pairs if 'rmsd_angstrom' in p]
    fig,ax=plt.subplots(figsize=(6,4));ax.scatter([p['rmsd_angstrom'] for p in valid],[p['contact_jaccard'] or 0 for p in valid],s=12);ax.set(xlabel='Aligned CA RMSD (A)',ylabel='Contact Jaccard',title=f'Bounded duplicate panel ({len(valid)} aligned pairs)');fig.tight_layout();fig.savefig(out/'structural_panel.svg');plt.close(fig)
    save('RUN_METRICS.json',{'elapsed_seconds':time.time()-start,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'batch_rows':256,'sqlite_cache_mib':32,'workers':1,'training_launched':False,'gpu_used':False,'output_root':str(out),'sidecar_hash_failures':sum(p['expected_match'] is False for p in provenance)})
    db.close();print(json.dumps({'counts':{k:v.get('records') for k,v in stats.items()},'pair_classes':dict(collections.Counter(r['classification'] for r in pairs)),'elapsed':time.time()-start}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--finalize',action='store_true');parser.add_argument('--review',action='store_true');a=parser.parse_args();sys.path.insert(0,str(REPO/'src'))
    if a.review:
        from review import review
        review(a.output)
    elif a.finalize:
        from report import finalize
        finalize(a.output)
        from review import review
        review(a.output)
    else:main(a.output)
