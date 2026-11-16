"""Publish frozen v2 membership, data design and prediction-reuse protocol."""
from pathlib import Path
import collections,gzip,json,math,os,resource,shutil,subprocess,sys,time
import numpy as np
import pyarrow.parquet as pq
from core import length_stratum,hash_sequence,future_bos_annotation
from run import REPO,HIST,V1,RAW,HOM,CORPUS,FROZEN,POLICY,rows,jsonrows,sha,save,gz_writer
class DSU:
    def __init__(self):self.parent={}
    def find(self,x):
        p=self.parent.setdefault(x,x)
        while p!=self.parent[p]:self.parent[p]=self.parent[self.parent[p]];p=self.parent[p]
        while x!=p:old=self.parent[x];self.parent[x]=p;x=old
        return p
    def union(self,a,b):
        a=self.find(a);b=self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)
    def join(self,nodes):
        ns=list(nodes)
        if ns:
            for x in ns[1:]:self.union(ns[0],x)
def fasta_records(path):
    header=None;seq=[]
    with Path(path).open() as f:
        for line in f:
            if line.startswith('>'):
                if header is not None:yield header,''.join(seq)
                header=line[1:].strip().split()[0];seq=[]
            else:seq.append(line.strip())
    if header is not None:yield header,''.join(seq)
def report(out):
    start=time.time();data=json.loads((out/'DATASET_COUNTS_V2.json').read_text());policy=json.loads((out/'SCREENING_POLICY_V2.json').read_text());assert policy==POLICY
    if (out/'REPORT_COMPLETE.json').exists():raise RuntimeError('v2 reports already complete; immutable publication')
    keep=['sample_id','split','pdb_id','experimental_method','resolution_angstrom','retained_length','retained_sha256','full_sha256','full_status','full_reason','full_length','mapping_status','historical_bos_semantics','full_lm_eligible','entity_id','published_cluster_id','published_mixed','additional_e012','observation_part']
    rec={r['sample_id']:{k:r.get(k) for k in keep} for r in jsonrows(out/'population_records.jsonl.gz')}
    train={sid for sid,r in rec.items() if r['split']=='train'};valid={sid for sid,r in rec.items() if r['split']=='validation'};test={sid for sid,r in rec.items() if r['split']=='test'}
    assert len(train)==231743 and len(valid)==23307 and len(test)==25066
    panels_path=HIST/'reports/experiments/E012_causal_rope_sequence/pilot_v1/panels.json';panels=json.loads(panels_path.read_text())
    assert all(len(ids)==2048 and set(ids)<=valid for ids in panels.values())
    groups=DSU();node_members=collections.defaultdict(list);ret_nodes={s:collections.defaultdict(set) for s in ['train','validation','test']};full_nodes={s:collections.defaultdict(set) for s in ['train','validation','test']}
    for sid,r in rec.items():
        node='published:'+r['published_cluster_id'] if r.get('published_cluster_id') else 'unassessed:'+sid
        r['screen_node']=node;groups.find(node);node_members[node].append(sid);ret_nodes[r['split']][r['retained_sha256']].add(node)
        if r.get('full_sha256'):full_nodes[r['split']][r['full_sha256']].add(node)
    # Exact strings preserve shared identities across clusters and new records.
    for h in set(ret_nodes['train'])|set(ret_nodes['validation']):groups.join(ret_nodes['train'].get(h,set())|ret_nodes['validation'].get(h,set()))
    exact=set(ret_nodes['train'])&set(ret_nodes['validation']);pdb_overlap={rec[s]['pdb_id'] for s in train}&{rec[s]['pdb_id'] for s in valid};assert not exact and not pdb_overlap
    jobs={};edges=[];saturated=set();failed=[]
    names=['additional_train_vs_validation','additional_validation_vs_train','full_validation_vs_train']
    for name in names:
        base=out/'mmseqs_search'/name
        if not (base/'COMPLETE.json').exists():jobs[name]={'status':'UNKNOWN','reason':json.loads((base/'INCOMPLETE.json').read_text()) if (base/'INCOMPLETE.json').exists() else 'not run'};failed.append(name);continue
        marker=json.loads((base/'COMPLETE.json').read_text());path=base/'alignments.tsv';assert sha(path)==marker['result_sha256']
        hits=collections.Counter();read_count=0;local=[]
        with path.open() as f:
            for line in f:
                fields=line.rstrip().split('\t');assert len(fields)==14
                q,t=fields[:2];ident=float(fields[2]);qcov=float(fields[10]);tcov=float(fields[11]);ev=float(fields[12]);hits[q]+=1;read_count+=1
                if ident<.3-1e-6 or qcov<.8-1e-6 or tcov<.8-1e-6 or ev>.001+1e-10:raise ValueError('alignment outside frozen policy')
                e={'job':name,'query_hash':q,'target_hash':t,'identity':ident,'qcov':qcov,'tcov':tcov,'aligned_length':int(fields[3]),'query_start':int(fields[4]),'query_end':int(fields[5]),'query_length':int(fields[6]),'target_start':int(fields[7]),'target_end':int(fields[8]),'target_length':int(fields[9]),'evalue':ev,'bits':float(fields[13])};local.append(e)
                if name=='additional_train_vs_validation':a=ret_nodes['train'][q];b=ret_nodes['validation'][t]
                elif name=='additional_validation_vs_train':a=ret_nodes['validation'][q];b=ret_nodes['train'][t]
                else:continue
                groups.join(a|b)
        sat={q for q,c in hits.items() if c>=10000};saturated|={(name,q) for q in sat};edges.extend(local)
        jobs[name]={**marker,'alignment_count':read_count,'aligned_query_count':len(hits),'queries_at_reported_hit_cap':len(sat),'limitation':'prefilter cap applies before alignment; unsaturated output does not prove unsaturated prefilter. Heuristic non-detection only.'}
    trainroots={groups.find(rec[s]['screen_node']) for s in train};retained_affected={s for s in valid if groups.find(rec[s]['screen_node']) in trainroots}
    retained_components=collections.defaultdict(lambda: {'train':[],'validation':[]})
    for s in train|valid:retained_components[groups.find(rec[s]['screen_node'])][rec[s]['split']].append(s)
    with gz_writer(out/'RETAINED_MIXED_GROUPS_V2.jsonl.gz') as f:
        for root,m in sorted(retained_components.items()):
            if m['train'] and m['validation']:f.write((json.dumps({'group_id':root,**{k:sorted(v) for k,v in m.items()}},separators=(',',':'))+'\n').encode())
    # Full-polymer exact identity and detected alignments add distinct evidence.
    full_exact=set(full_nodes['train'])&set(full_nodes['validation'])
    for h in set(full_nodes['train'])|set(full_nodes['validation']):groups.join(full_nodes['train'].get(h,set())|full_nodes['validation'].get(h,set()))
    for e in edges:
        if e['job']=='full_validation_vs_train':groups.join(full_nodes['validation'][e['query_hash']]|full_nodes['train'][e['target_hash']])
    trainroots={groups.find(rec[s]['screen_node']) for s in train};all_affected={s for s in valid if groups.find(rec[s]['screen_node']) in trainroots}
    all_components=collections.defaultdict(lambda:{'train':[],'validation':[]})
    for s in train|valid:all_components[groups.find(rec[s]['screen_node'])][rec[s]['split']].append(s)
    mixed_count=0
    with gz_writer(out/'KNOWN_MIXED_GROUPS_V2.jsonl.gz') as f:
        for root,m in sorted(all_components.items()):
            if m['train'] and m['validation']:
                mixed_count+=1;f.write((json.dumps({'group_id':root,'definition':'connected component of published cluster, exact retained/full strings and recorded directed alignments; not every pair is directly homologous',**{k:sorted(v) for k,v in m.items()}},separators=(',',':'))+'\n').encode())
    with gz_writer(out/'DETECTED_ALIGNMENT_EDGES_V2.jsonl.gz') as f:
        for e in edges:f.write((json.dumps(e,separators=(',',':'))+'\n').encode())
    retained_complete=not any(n in failed for n in names[:2]);full_complete=names[2] not in failed
    unknown_train_full=sum(rec[s].get('full_status')!='VERIFIED' for s in train);unknown_val_full=sum(rec[s].get('full_status')!='VERIFIED' for s in valid)
    retained_ok={s for s in valid if s not in retained_affected} if retained_complete else set()
    full_ok={s for s in retained_ok if rec[s]['full_status']=='VERIFIED' and s not in all_affected} if full_complete else set()
    # Every query sharing a saturated novel sequence remains UNKNOWN, including reverse-target exposure.
    if saturated:
        retained_ok=set();full_ok=set() # conservative whole-view fail closed rather than localizing an incomplete target scan.
    strict_ok=full_ok if unknown_train_full==0 and unknown_val_full==0 and retained_complete and full_complete and not saturated else set()
    views={'historical_validation':sorted(valid),'published_accepted_no_known_mixed_only':sorted(s for s in valid if rec[s].get('published_cluster_id') and not rec[s]['published_mixed']),
           'retained_complement_assessed_no_detected_relationship_v2':sorted(retained_ok),'retained_and_reconstructed_full_assessed_no_detected_relationship_v2':sorted(full_ok),'strict_entire_e012_full_polymer_scope_v2':sorted(strict_ok)}
    for sid,r in rec.items():r['v2_group_id']=groups.find(r['screen_node'])
    membership=[];panel_sets={name:set(ids) for name,ids in panels.items()}
    for sid in sorted(valid):
        r=rec[sid];membership.append({'sample_id':sid,'pdb_id':r['pdb_id'],'entity_id':r.get('entity_id'),'method':r['experimental_method'],'resolution_angstrom':r['resolution_angstrom'],'retained_length':r['retained_length'],'length_stratum':length_stratum(r['retained_length']),
            'retained_sha256':r['retained_sha256'],'full_sha256':r.get('full_sha256'),'full_mapping_status':r.get('full_status','UNKNOWN'),'full_mapping_reason':r.get('full_reason'),'additional_e012_validation':r['additional_e012'],
            'published_mixed':r['published_mixed'],'known_retained_mixed':sid in retained_affected,'known_any_representation_mixed':sid in all_affected,'group_id':groups.find(r['screen_node']),
            'retained_screen_status':'KNOWN_RELATIONSHIP' if sid in retained_affected else 'NO_DETECTED_RELATIONSHIP_UNDER_DECLARED_PROCEDURE' if sid in retained_ok else 'UNKNOWN',
            'full_screen_status':'KNOWN_RELATIONSHIP' if sid in all_affected else 'NO_DETECTED_RELATIONSHIP_WITH_ASSESSED_TRAIN_SUBSET' if sid in full_ok else 'UNKNOWN',
            'entire_training_scope_status':'KNOWN_RELATIONSHIP' if sid in all_affected else 'NO_DETECTED_RELATIONSHIP_UNDER_DECLARED_PROCEDURE' if sid in strict_ok else 'UNKNOWN',
            'primary_panel':sid in panel_sets['primary'],'independent_panel':sid in panel_sets['independent']})
    panelviews={name:{view:sorted(set(ids)&set(sids)) for view,sids in views.items()} for name,ids in panels.items()}
    tally={view:{'records':len(ids),'by_retained_length':dict(collections.Counter(length_stratum(rec[s]['retained_length']) for s in ids)),'by_method':dict(collections.Counter(rec[s]['experimental_method'] or 'UNKNOWN' for s in ids)),'by_resolution':dict(collections.Counter('UNKNOWN' if rec[s]['resolution_angstrom'] is None else '<=2A' if rec[s]['resolution_angstrom']<=2 else '>2A' for s in ids))} for view,ids in views.items()}
    summary={'version':'e012_validation_scope_v2','frozen_policy':policy,'original_denominators':{'train':231743,'validation':23307,'primary':2048,'independent':2048},'additional_records':{'train':1611,'validation':146},'search_jobs':jobs,
        'known_retained_mixed_validation':len(retained_affected),'known_combined_mixed_validation':len(all_affected),'published_mixed_validation':1578,'known_mixed_components':mixed_count,'full_exact_cross_train_validation_groups':len(full_exact),
        'unknown_full_polymer_train_records':unknown_train_full,'unknown_full_polymer_validation_records':unknown_val_full,'retained_screen_procedure_complete':retained_complete,'reconstructed_full_search_complete':full_complete,
        'entire_e012_homology_scope_fully_established':bool(strict_ok) and unknown_train_full==unknown_val_full==0,'retained_exact_cross_split_groups':len(exact),'pdb_entry_overlap':len(pdb_overlap),'query_hit_cap_flags':sorted(saturated),
        'views':views,'view_counts':tally,'frozen_panel_views':panelviews,'panel_view_counts':{name:{view:len(ids) for view,ids in vv.items()} for name,vv in panelviews.items()},'records':membership,
        'view_claims':{'historical_validation':'descriptive historical benchmark; contamination known','published_accepted_no_known_mixed_only':'insufficient for full E012 scope; not homology-safe','retained_complement_assessed_no_detected_relationship_v2':'usable conditional retained-sequence heuristic sensitivity view; no exhaustive guarantee','retained_and_reconstructed_full_assessed_no_detected_relationship_v2':'usable conditional known-full-polymer subset sensitivity view; unassessed TRAIN full polymers remain a limitation','strict_entire_e012_full_polymer_scope_v2':'usable only if all required representations are assessed; otherwise empty and blocked'}}
    save(out,'E012_VALIDATION_MEMBERSHIP_V2.json',summary)
    save(out,'FROZEN_PANEL_VIEWS_V2.json',{'original_panels_path':str(panels_path),'original_panels_sha256':sha(panels_path),'membership_sha256':sha(out/'E012_VALIDATION_MEMBERSHIP_V2.json'),'views':panelviews,'frozen_policy_sha256':sha(out/'SCREENING_POLICY_V2.json'),'inference_authorized':False})
    # Freeze reporting before any prediction reaggregation.
    protocol='''# Proposed evaluation protocol v2 — frozen before reaggregation

VERIFIED membership is in E012_VALIDATION_MEMBERSHIP_V2.json and immutable panel intersections in FROZEN_PANEL_VIEWS_V2.json. Historical panels remain unchanged. Report historical full, published-only diagnostic, retained complement-assessed, reconstructed-full complement-assessed and strict-entire-scope views separately. Conditional views must retain their scope qualifier; never call them universally homology-safe. No new selection by model loss or predicted confidence is allowed.

Report number of records, exact retained/full sequences, historical and v2 homology components, experimental provenance, fragment-start frequency, five retained-length strata and full-length distribution. Historical validation has one record per retained exact sequence, so retained-unique and record-weighted results coincide there. Full-polymer grouping can differ. For future repeated populations use per-record 1/m with multiplicity recomputed inside each frozen population. Never use old E006 weights as current inverse multiplicities.

Use equal-record CE, token-weighted CE from summed NLL, mean per-record perplexity versus exp(mean CE) as explicitly distinct estimands, accuracy, normal-minus-composition/global/length-unigram/bigram/trigram, paired normal-prefix-minus-shuffle, last8 and last1 differences, and neutral-length ablation where stored. Shuffles preserve composition at fixed historical targets; they are statistical controls, not negative folding labels. Exact length/composition-matched unigram, positional-unigram and simple n-gram baselines must be fitted to TRAIN only. New unique-sequence baseline parameters, position-level conditioning or new shuffle replicates need a new approved computation/inference protocol if predictions are missing.

Existing per-identity predictions can be filtered/reweighted and bootstrapped without model inference. Freeze this document's hash before computing descriptive reaggregations. No fresh causal forward pass, checkpoint loading, structural prediction or pLDDT labels. Use 1000 deterministic bootstrap draws seed 21212, equal-record resampling and equal-homology-component (component means) resampling separately. These intervals have different estimands. In sensitivity reporting, cross-fit nuisance baselines or retrain models only under a separate approved experiment. Original metric gates remain historical; do not retrospectively promote a corrected subset into a new certification gate.

The 20M UniRef50 warm-start checkpoint is an external-pretrained system. Protection must match exact evaluation sequence representations and certified protected hashes. Retained sequences can reuse verified protected membership/certification; newly reconstructed full polymers are a different target universe unless independently protected. Parent PDB-trained contamination is distinct from UniRef50 continuation overlap. All experimental-design conclusions remain qualified by PDB selection and context dependence.
'''
    (out/'PROPOSED_EVALUATION_PROTOCOL_V2.md').write_text(protocol);protocol_hash=sha(out/'PROPOSED_EVALUATION_PROTOCOL_V2.md')
    metrics=reaggregate(out,rec,panels,panelviews,protocol_hash)
    protection=pretraining(out,rec,panels,views)
    save(out,'PRETRAINING_PROTECTION_V2.json',protection)
    # Preserve the historical BOS-location annotation and add an explicit future fragment tag.
    with gz_writer(out/'BOS_SEMANTICS_CORRECTION_V2.jsonl.gz') as f:
        for sid,r in sorted(rec.items()):
            f.write((json.dumps({'sample_id':sid,'split':r['split'],'historical_bos_location':r['historical_bos_semantics'],**future_bos_annotation(r.get('full_sha256'),r['retained_sha256'],r['historical_bos_semantics'])},separators=(',',':'))+'\n').encode())
    # Stable schemas link full sequence unit and experimental observation, with unknowns explicit.
    fields={'sequence_unit_id':'SHA256 exact frozen canonical complete construct string','canonical_complete_sequence':'standard20+MSE parent; scheme and entity agreement required','canonical_complete_sha256':'exact hash','deposited_polymer_monomers':'ordered complete monomer list; never skip unsupported monomers','deposited_one_letter_lexeme':'string or null; legacy cache does not retain noncanonical one-letter field','canonicalization_policy':'pdb_full_scheme_standard_plus_MSE_v2','source_links':'all source/chain observation identities and sealed cache row/partition hashes','record_multiplicity':'current population denominator; separate per historical split','sampling_modes':['historical retained-record replay','uniform full-exact-unique','1/current-population-full-multiplicity'],'split':'unassigned until linked full-polymer/homology context grouping is frozen','full_bos_semantics':'deposited polymer N terminus','native_gene_terminal_status':'UNKNOWN unless external evidence documents processing/construct boundaries'}
    observation={'observation_id':'PDB/source revision hash + entity + label/auth chain + model + conformer policy','source_sha256':'recorded authoritative source content hash','source_current_rehash':'not performed; cached inventory and v1 sealed inputs reused','deposited_polymer_identity':'sequence_unit_id','model_number':'integer; current projection model1','alternate_model_evidence':'existing cached protein_chain_models table; retain future conformations only under prespecified evidence criteria','residue_mapping':'full label order; author ID and insertion code retained as identifiers; coordinate_index nullable','missing_residue_mask':'one value per full polymer position','missing_ca_mask':'separate from missing residue','historical_retained_sequence':'preserved fragment view','historical_bos_semantics':'polymer_N_terminus / fragment_start / UNKNOWN','experimental_method':'known or null','resolution':'known or null','context':'ligand/assembly/dependence UNKNOWN unless authoritative evidence supplied','disorder_evidence':'known/absent/unknown with source and conditions; default unknown','structural_order_proxy':'observation-specific known eligibility or unknown; never thermodynamic folding label'}
    observation['future_view_bos']='complete_deposited_polymer_start / fragment_start / UNKNOWN, separate from whether it starts at the deposited N terminus'
    save(out,'PDB_ONLY_MANIFEST_SCHEMA_V2.json',{'version':'pdb_only_linked_design_v2','status':'design proposal plus versioned candidate manifests; no training release','full_sequence_table':fields,'structure_observation_table':observation,'strict_invariants':['canonical full scheme labels exactly 1..N','canonical entity agrees with frozen monomer-parent policy','retained correspondence requires unique strong source evidence','increasing numeric label order independent of author numbering','no artificial adjacency across missing positions','unknown mappings excluded from strict released datasets','all linked exact/full/polymer/homology observations kept in one split'],'no_folding_label':True})
    write_reports(out,data,summary,protection,metrics,rec)
    # Focused fixtures are actual v1 problematic examples recovered from cached source evidence.
    wanted={'7z53_W','9qex_E','6zk9_b','6zkc_b'};fixtures={}
    for part in sorted({rec[s]['observation_part'] for s in wanted if rec[s].get('observation_part')}):
        for r in jsonrows(out/part):
            if r['sample_id'] in wanted:fixtures[r['sample_id']]=r
    save(out,'tests/actual_problematic_examples.json',fixtures)
    reproduce=out/'reproduce';reproduce.mkdir(exist_ok=True)
    for p in (REPO/'thesis_audit_v2').glob('*.py'):shutil.copyfile(p,reproduce/p.name)
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(out/'tests'))
    testcmd=[sys.executable,'-B',str(REPO/'thesis_audit_v2/test_v2.py'),'--root',str(out)]
    with (out/'tests/test_results.txt').open('w') as log:result=subprocess.run(testcmd,stdout=log,stderr=subprocess.STDOUT,env=env)
    if result.returncode:raise RuntimeError('v2 focused tests failed; publication refused')
    save(out,'FOCUSED_TEST_RESULTS.json',{'status':'passed','command':testcmd,'log_sha256':sha(out/'tests/test_results.txt'),'includes_actual_examples':sorted(fixtures),'training_or_inference':False})
    # Only small inputs/reports/prediction metadata are added; no 20M corpus rescreen or weights opened.
    prov=json.loads((out/'PROVENANCE_V2.json').read_text())
    for p in [panels_path,HIST/'reports/experiments/E012_causal_rope_sequence/pilot_v1/positions.json']+[REPO/'thesis_audit_v2'/n for n in ['core.py','run.py','report.py','documents.py','test_v2.py']]:prov.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
    save(out,'PROVENANCE_PUBLICATION_V2.json',prov)
    save(out,'REPORT_COMPLETE.json',{'version':'v2','membership_sha256':sha(out/'E012_VALIDATION_MEMBERSHIP_V2.json'),'frozen_protocol_sha256':protocol_hash,'prepare_raw_reparse_count':0,'report_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'elapsed_seconds':time.time()-start,'training_or_inference':False,'blockers':json.loads((out/'UNRESOLVED_BLOCKERS_V2.json').read_text())})
    save(out,'OUTPUT_SHA256.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='OUTPUT_SHA256.json' and 'mmseqs_search' not in p.parts})
    print('v2 publication complete',summary['panel_view_counts'],'known combined affected',len(all_affected),'unknown full TRAIN',unknown_train_full,flush=True)

def reaggregate(out,rec,panels,panelviews,protocol_hash):
    result={'frozen_protocol_sha256':protocol_hash,'model_inference':False,'scope':'existing 2000/5000/10000 per-identity stored predictions; no new metrics requiring missing logits','experiments':{}}
    paths={2000:HIST/'outputs/e012_causal_rope_sequence/pilot_v1',5000:HIST/'outputs/e012_causal_rope_sequence/continuation_2k_to_10k_v1',10000:HIST/'outputs/e012_causal_rope_sequence/continuation_2k_to_10k_v1'}
    for step,root in paths.items():
        for name in panels:
            p=root/(f'{name}_final_records.json' if step==2000 else f'{name}_final_records_{step:05d}.json')
            if not p.exists():continue
            source=json.loads(p.read_text());assert {r['sample_id'] for r in source}==set(panels[name]);lookup={r['sample_id']:r for r in source};outcomes={}
            for view,ids in panelviews[name].items():
                rr=[lookup[s] for s in ids]
                if not rr:outcomes[view]={'records':0,'status':'UNKNOWN/blocked: empty view'};continue
                ce=np.array([r['normal'] for r in rr]);lens=np.array([r['length'] for r in rr]);groupvals=collections.defaultdict(list)
                for r in rr:groupvals[rec[r['sample_id']]['v2_group_id']].append(r['normal'])
                gm=np.array([np.mean(v) for v in groupvals.values()]);rng=np.random.default_rng(21212)
                draws_record=[float(ce[rng.integers(len(ce),size=len(ce))].mean()) for _ in range(1000)]
                draws_group=[float(gm[rng.integers(len(gm),size=len(gm))].mean()) for _ in range(1000)]
                outcome={'records':len(rr),'tokens':int(lens.sum()),'exact_retained_unique':len({rec[r['sample_id']]['retained_sha256'] for r in rr}),
                  'equal_record_ce':float(ce.mean()),'token_ce':float(sum(r['loss_sum'] for r in rr)/lens.sum()),'perplexity_exp_mean_ce':float(np.exp(ce.mean())),
                  'equal_record_ce_ci95':np.quantile(draws_record,[.025,.975]).tolist(),'v2_homology_group_equal_mean_ce':float(gm.mean()),'v2_homology_group_ce_ci95':np.quantile(draws_group,[.025,.975]).tolist(),'v2_homology_groups':len(gm),
                  'normal_prefix_minus_shuffle':float(np.mean([r['prefix_normal']-r['shuffle'] for r in rr])),'normal_prefix_minus_last8':float(np.mean([r['prefix_normal']-r['last8'] for r in rr])),
                  'length_strata':{str(i):{'records':sum(r['stratum']==i for r in rr),'mean_ce':float(np.mean([r['normal'] for r in rr if r['stratum']==i])) if any(r['stratum']==i for r in rr) else None} for i in range(5)},
                  'limitation':'conditional subset sensitivity only; 1000 deterministic draws; group interval equal group means, different estimand from equal record mean; retained exact unique and record weighting coincide in original validation'}
                fullbins=collections.defaultdict(list)
                for r in rr:
                    if rec[r['sample_id']].get('full_sha256'):fullbins[rec[r['sample_id']]['full_sha256']].append(r['normal'])
                outcome['known_full_exact_unique']=len(fullbins);outcome['known_full_mapping_records']=sum(map(len,fullbins.values()));outcome['retained_prediction_equal_full_identity_group_mean_ce']=float(np.mean([np.mean(v) for v in fullbins.values()])) if fullbins else None
                outcomes[view]=outcome
            result['experiments'][f'{step}_{name}']={'prediction_path':str(p),'prediction_sha256':sha(p),'views':outcomes}
    save(out,'EXISTING_PREDICTION_REAGGREGATION_V2.json',result);return result

def pretraining(out,rec,panels,views):
    definition=CORPUS/'reports/protected_definition.json';d=json.loads(definition.read_text());protected_path=CORPUS/'protected/protected.fasta';assert sha(protected_path)==d['fasta_sha256']
    hashes={hash_sequence(s) for _,s in fasta_records(protected_path)};assert len(hashes)==d['unique_sequences']==53615
    targets=set(views['historical_validation']);missing=[s for s in targets if rec[s]['retained_sha256'] not in hashes];assert not missing
    certp=CORPUS/'corpus/certification.json';verify=CORPUS/'reports/verify_protected.json';prot=CORPUS/'CORRECTION_PROTOCOL.json';c=json.loads(certp.read_text());vr=json.loads(verify.read_text());cp=json.loads(prot.read_text())
    frozen=json.loads((FROZEN/'protocol.json').read_text());assert sha(certp)==frozen['dataset_hashes']['certification'] and sha(CORPUS/'corpus/SHA256SUMS')==frozen['dataset_hashes']['inventory']
    assert c['certified'] and c['protected_exact_overlap']==0 and c['protected_detected_homology_overlap']==0 and vr['zero_detected_violations']
    fullids={rec[s]['full_sha256'] for s in targets if rec[s].get('full_sha256')};fullprotected=fullids&hashes
    evidence={'historical_retained_validation_denominator':len(targets),'retained_validation_protected_exact_identity_membership':len(targets)-len(missing),'protected_unique_sequences':len(hashes),'certified_20m_exact_overlap':c['protected_exact_overlap'],'certified_20m_detected_homology_overlap':c['protected_detected_homology_overlap'],
       'verified_policy':{k:cp[k] for k in ['identity','coverage','cov_mode','sensitivity','direction','search_type','seq_id_mode','alignment_mode','max_seqs_independent_audit']},
       'panel_membership_verified':{name:sum(rec[s]['retained_sha256'] in hashes for s in ids) for name,ids in panels.items()},'known_full_validation_unique_sequences':len(fullids),'known_full_validation_unique_protected':len(fullprotected),'known_full_validation_unique_unprotected':len(fullids-fullprotected),
       'status':'VERIFIED protected membership and frozen certification linkage; independent 20M rescreen NOT RUN; heuristic homology guarantee only; full-polymer representations absent from protected set UNKNOWN',
       'scope_note':'continuation corpus excludes protected retained strings/detected homologs; PDB-trained parent contamination remains separately relevant; UniRef50 system is not PDB-only',
       'input_hashes':{str(p):sha(p) for p in [definition,protected_path,certp,verify,prot,CORPUS/'corpus/SHA256SUMS',FROZEN/'protocol.json',FROZEN/'audit.json']}}
    return evidence

def write_reports(out,data,membership,protection,metrics,rec):
    # Implemented below; counts are sourced from frozen manifests rather than filenames.
    from documents import write_documents
    write_documents(out,data,membership,protection,metrics,rec)
