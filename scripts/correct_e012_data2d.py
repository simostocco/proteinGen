"""Freeze validated protein typing and complete prefilter capacity in a new D: namespace."""
import argparse, csv, hashlib, importlib.util, json, os, shutil, sqlite3, subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('D:/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization') if os.name=='nt' else Path('/mnt/d/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization')
OLD=ROOT/'scientific';DIAG=ROOT/'data2d_batch200_diagnostic';NEW=ROOT/'scientific_corrected_protein_v1'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*1024**2):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def translated(s):
    s=str(s).replace('\\','/')
    if os.name=='nt':return Path(s.replace('/mnt/d','D:'))
    return Path('/mnt/d'+s[2:] if s.startswith('D:/') else s)
def write(p,obj):
    assert not p.exists(),f'Refuse to overwrite {p}'
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def ro(p):return sqlite3.connect(p.as_uri()+'?mode=ro',uri=True)

def validate_evidence():
    records=load(DIAG/'EVIDENCE.json')['files']
    checks={r['path']:sha(translated(r['path']))==r['sha256'] for r in records}
    assert all(checks.values()),'Original evidence changed'
    return checks

def report():
    offenders=load(DIAG/'OFFENDERS_TRACED.json');byid={x['query']:x for x in offenders}
    targets=load(DIAG/'PROTECTED_TARGETS.json');alignments=load(DIAG/'ALIGNMENTS.json')
    out=DIAG/'OFFENDING_ALIGNMENTS.tsv';assert not out.exists()
    fields=['query','candidate_hash','source_offset','length','historical','external','original_priority_rank','original_final_selection_rank','main_batch','target','fident','qcov','tcov','evalue','exact_sequence_match','protected_sequence']
    with out.open('w',newline='') as f:
        writer=csv.DictWriter(f,fields,delimiter='\t');writer.writeheader()
        for hit in alignments:
            x=byid[hit['query']]
            row={k:hit[k] for k in ('query','target','fident','qcov','tcov','evalue')}
            row.update({k:x[k] for k in ('length','historical','external','original_priority_rank','original_final_selection_rank','main_batch')})
            row.update(candidate_hash=x['hash'],source_offset=x['offset'],exact_sequence_match=x['sequence']==targets[hit['target']],protected_sequence=targets[hit['target']]);writer.writerow(row)
    facts={'offending_queries':len(offenders),'alignments':len(alignments),'protected_targets':len(targets),'root_cause_class':'B: search heuristic false negative requiring global correction','batch200_immediate_cause':'query database alphabet mismatch in main batch 241','secondary_confirmed_false_negative':load(DIAG/'CAP_OFFENDER.json'),'main_batch':241,'original_search_returned_hits':False,'original_per_batch_independent_survivor_search':False,'original_exact_only_exclusions':2671,'original_exclusions':802706,'original_clean_pool':23283238,'full_24M_rescreen_required':True,'affected_population_requires_complete_research':24085944,'fresh_independent_audit_required':20000000,'old_evidence_checks':validate_evidence(),'no_training':True}
    write(DIAG/'ROOT_CAUSE.json',facts)
    text='''# E012 V2C DATA2-D: protein typing and prefilter completeness correction

The final independent audit correctly stopped before corpus finalization. It found 16,607 historical survivor queries, 76,613 alignments and 6,516 protected targets. Every offender is from main-screen batch 241. Every alignment, candidate hash/source offset/length/provenance, protected sequence and original priority/final-selection rank is in OFFENDING_ALIGNMENTS.tsv; candidate sequences are in OFFENDERS_TRACED.json.

The immediate batch-200 cause is a query alphabet mismatch in main-screen batch 241. The first ten historical queries are poly-alanine proteins. MMseqs 15 createdb --dbtype 0 samples the first ten queries; its counter then remains at 10 (the next sampling condition requires divisibility by 100). All ten have >90% A/C/G/T/U/N, so it types the entire 85,944-protein database as nucleotide. Search translates it into 442,562 ORFs, returning an empty relation. The main screen correctly parsed that empty output and excluded only 2,671 exact canonical protected strings. Complement certification could not detect the incorrect database representation. The reversed independent audit starts with ordinary proteins, builds an amino-acid DB, and exposes the contradiction. This is deterministic alphabet selection, not stochastic alignment loss.

GLOBAL RECOVERY CLASSIFICATION B: the larger-cap control also proves a real search-completeness false negative outside the historical tail. External query s_43683a7be2017cb2fc9de98bd89dc4529d846ac1259257a0d5c34c1476936f20_6559485279 (UniRef50_A0AAE0GEI3, length 495, rowid 3,000,335, main batch 31) is absent from its original archived relation and exclusions SQLite. Its original full survivor-verification certificate records an empty relation too. With max-seqs 1000 it is missed; larger caps recover target p_ecf4906ab6fab8852ef18c7277271f30ec6c642865df96ecddd91c1996f47c4a at identity 0.301, qcov 0.867, tcov 0.875, E-value 9.573E-34. No parser error is involved. All original negative expectations remain unchanged as historical evidence; the newly observed qualifying alignment proves this old negative was false. CAP_REPRODUCTIONS.json and CAP_OFFENDER.json preserve this case. Therefore a localized 85,944-query correction is insufficient, and EXPECTED_CORRECTION.json is only the rejected alphabet-only diagnostic projection, not a corrected-corpus count.

The installed release source is preserved as createdb-15.cpp, from https://raw.githubusercontent.com/soedinglab/MMseqs2/15-6f452/src/util/createdb.cpp. The original/repeated empty searches and all restored hit relations are preserved under runs/. Explicit amino-acid typing gives identical sorted full hit relations in original/reversed batch-241 order. All fixed-cap offender-only main/audit repeats reproduce all 76,613 archived audit alignments. The original 64-query bounded 1/16-thread/cap controls give identical relations, but the broader high-cap offender panel adds nine weaker alignments without changing those 16,607 exclusions or nearest decisions. The 6,020-query panel adds a genuinely new excluded query, proving the cap failure. Raw output order varies; classifications, strict-greater nearest decisions and sorted complete relations are compared separately. Repeated cap-offender tests separate deterministic truncation from threading.

Protected FASTA and all protected DB/index bytes match the frozen target_index.complete.json. All 241 source batches reconstruct their original input-row and FASTA hashes exactly. Re-parsing every archived main result reproduces the SQLite exclusions and nearest-hit decisions exactly: no parser/bookkeeping defect. The 208 later prefix creation logs show amino-acid DBs; the first ten queries for each of the earlier 32 batches are independently recreated with the original autodetection policy and verified as amino acid. Three complete representative prefix batches and the 6,020-query equivalence panel are checked against archived full relations with explicit protein typing.

Corrected protocol: createdb --dbtype 1, assert serialized uncompressed amino-acid DBTYPE zero, explicit --search-type 1, and max-seqs 53615 (the complete immutable protected target entry count) in both production and audit. The bound cannot truncate a prefilter list of distinct protected targets. Candidate-to-protected orientation, canonical sequences, provenance, priority ordering, exact-string override, strict-greater nearest-hit tie handling, identity >=0.30, bidirectional coverage >=0.80/cov-mode 0, sensitivity 7.5, seq-id-mode 0, alignment-mode 3 and 16 threads are unchanged. Only query representation and search completeness change: no unintended nucleotide translation and no 1,000-target truncation. More complete relations can correctly increase exclusions and improve nearest-hit identities; the nearest decision algorithm is unchanged. The independent audit separately reconstructs query databases with easy-search, independently parses hits and exact-string violations, reverses FASTA order, and uses the full-target cap so it cannot repeat the diagnosed truncation. Independence does not rely solely on reversal.

Recovery requires a FULL new 24,085,944-candidate search, including every previously excluded candidate and all unselected reservoir candidates; zero old screen or audit batches are adopted. Reparsing cannot recover alignments that MMseqs omitted. New exclusions and deterministic priority selection are published in scientific_corrected_protein_v1. No original certificate, exclusion DB, audit checkpoint, failure report or previous recovery report is modified. A completely new 20M audit evaluates the corrected selection from batch 1; none of the old 199 zero-hit audit batches is reused. ANY disagreement remains DATA2-D and stops before final/certify/diversity/handoff. A separate full-audit gate verifies all 200 zero-hit certificates, hashes, exact protocol flags and durable row counts before finalization. No training action exists in the automatic chain.
'''
    p=DIAG/'DIAGNOSTIC_REPORT.md';assert not p.exists();p.write_text(text)

def initialize():
    assert os.name=='nt','Use native NTFS I/O for the bounded SQLite backup'
    validation=load(DIAG/'CORRECTED_VALIDATION.json');assert validation['passed']
    assert validation['root_cause_class']=='B' and validation['production_max_seqs']==53615 and validation['required_rescreen_population']==24085944
    scope=load(DIAG/'SCOPE_VALIDATION.json');assert scope['parser_bookkeeping_exact'] and scope['population']==24085944
    results={x['name']:x for x in load(DIAG/'REPRODUCTIONS.json')}
    assert results['original241-auto']['alignments']==results['original241-auto-repeat']['alignments']==0
    assert results['original241-amino']['sorted_relation_sha256']==results['reversed241-amino']['sorted_relation_sha256']==results['reversed241-auto']['sorted_relation_sha256']
    offender_relation=results['audit200-auto']['sorted_relation_sha256']
    assert all(results[f'offenders-{mode}-{rep}']['sorted_relation_sha256']==offender_relation for mode in ('main','audit') for rep in range(2))
    panel={r['sorted_relation_sha256'] for k,r in results.items() if k.startswith('panel-aa-') or k=='reversed-panel-amino'};assert len(panel)==1
    tests=DIAG/'tests-correction-final.xml';suites=ET.parse(tests).getroot().findall('testsuite')
    assert suites and all(int(s.get('failures','0'))==int(s.get('errors','0'))==0 for s in suites)
    assert sum(int(s.get('tests','0')) for s in suites)>=42
    validate_evidence()
    assert not NEW.exists(),'Do not reinitialize a result namespace'
    for name in ('stats','reports','manifests','filtered','protected','mmseqs','clusters','corpus','tmp','checksums','production/tmp','production/logs','production/reports'):(NEW/name).mkdir(parents=True)
    for name in ('resource_plan.json','protected_search_preflight.json','filtering_stats.json'):shutil.copy2(OLD/'stats'/name,NEW/'stats'/name)
    for name in ('source_provenance.json','protected_definition.json','PROTOCOL.md','filtering_stats.json'):shutil.copy2(OLD/'reports'/name,NEW/'reports'/name)
    for name in ('raw.complete.json','reservoir.complete.json'):shutil.copy2(OLD/'manifests'/name,NEW/'manifests'/name)
    shutil.copy2(OLD/'protected/protected.fasta',NEW/'protected/protected.fasta')
    shutil.copy2(OLD/'checksums/source_provenance.json',NEW/'checksums/source_provenance.json')
    with ro(OLD/'manifests/exclusions.sqlite') as src:
        old=json.loads(src.execute("SELECT v FROM production_progress WHERE k='checkpoint'").fetchone()[0])
    cp=dict(inputs=dict(old['inputs'],query_dbtype=1,max_seqs=53615),batch=0,screened=0,last_rowid=0,exclusions=0,slow_batches=0)
    with sqlite3.connect(NEW/'manifests/exclusions.sqlite') as dest:
        dest.execute('PRAGMA synchronous=FULL')
        dest.execute('CREATE TABLE removed(h BLOB,off INTEGER,protected TEXT,identity REAL,PRIMARY KEY(h,off)) WITHOUT ROWID')
        dest.execute('CREATE TABLE batches(name TEXT,batch INTEGER,n INTEGER,marker TEXT,PRIMARY KEY(name,batch))')
        dest.execute('CREATE TABLE production_progress(k TEXT PRIMARY KEY,v TEXT)')
        dest.execute("INSERT INTO production_progress VALUES('checkpoint',?)",(json.dumps(cp),));dest.commit()
    for name in ('OLD_OWNERS_STOPPED.json','FROZEN_DIAGNOSTICS.json'):shutil.copy2(ROOT/'screen_production/reports'/name,NEW/'production/reports'/name)
    protocol={'validated':True,'root_cause_class':'B','batch200_query_alphabet_defect':True,'query_dbtype':1,'search_type':1,'threads':16,'identity':.30,'coverage':.80,'cov_mode':0,'sensitivity':7.5,'seq_id_mode':0,'alignment_mode':3,'max_seqs_production':53615,'max_seqs_independent_audit':53615,'direction':'candidate_to_protected','population':24085944,'adopted_batches':[],'affected_research_population':24085944,'fresh_independent_audit_population':20000000,'diagnostics':str(DIAG),'validation_sha256':sha(DIAG/'CORRECTED_VALIDATION.json'),'scope_sha256':sha(DIAG/'SCOPE_VALIDATION.json'),'failure_action':'DATA2-D; stop','automatic_stages':['screen','verify-protected','final','certify','diversity','handoff'],'training':False}
    protocol.update(driver_sha256=sha(REPO/'scripts/run_e012_v2c_screen_production.py'),pipeline_sha256=sha(REPO/'scripts/complete_e012_v2c_pipeline.py'))
    write(NEW/'CORRECTION_PROTOCOL.json',protocol)
    write(NEW/'reports/FULL_RESCREEN_REQUIRED.json',{'scope_validation':scope,'checkpoint':cp,'old_results_frozen':True,'new_result_namespace':str(NEW),'old_results_adopted':False,'reason':'Confirmed max-seqs false-negative survivor in original batch 31; candidate-wide correction required'})
    print(json.dumps({'namespace':str(NEW),'next_batch':1,'re_search_population':24085944}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['report','initialize']);args=parser.parse_args();globals()[args.action]()
