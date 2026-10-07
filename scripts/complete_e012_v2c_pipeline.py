"""Reuse historical V2B downstream stages with all outputs under the D: V2C root."""
from pathlib import Path
import argparse,gzip,importlib.util,json,os,re,shutil,sqlite3,subprocess,time
spec=importlib.util.spec_from_file_location('v2c',Path(__file__).with_name('e012_reservoir_v2c.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ROOT=Path('D:/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization') if os.name=='nt' else Path('/mnt/d/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization')
m.configure_storage(ROOT)
SCI=ROOT/'scientific';ORIGINAL=ROOT.parent

def build():
    for name in ['stats','reports','manifests','filtered','protected','mmseqs','clusters','corpus','tmp','checksums']:(SCI/name).mkdir(parents=True,exist_ok=True)
    m.configure_storage(SCI)
    for key in ['TORCH_HOME','HF_HOME','MPLCONFIGDIR','NUMBA_CACHE_DIR']:os.environ[key]=str(SCI/'tmp'/key)
    ns=importlib.util.spec_from_file_location('native',Path(__file__).with_name('run_e012_v2b_native.py'))
    native=importlib.util.module_from_spec(ns);ns.loader.exec_module(native);legacy=native.load()
    report=SCI/'reports';oldreport=legacy.REPO/legacy.REPORT_REL
    for name in ['source_provenance.json','protected_definition.json','PROTOCOL.md']:
        dest=report/name
        if not dest.exists():shutil.copyfile(oldreport/name,dest)
    checksum_copy=SCI/'checksums/source_provenance.json'
    if not checksum_copy.exists():shutil.copyfile(ORIGINAL/'checksums/source_provenance.json',checksum_copy)
    for name in ['resource_plan.json','protected_search_preflight.json']:
        dest=SCI/'stats'/name
        if not dest.exists():shutil.copyfile(ORIGINAL/'stats'/name,dest)
    protected=SCI/'protected/protected.fasta'
    if not protected.exists():shutil.copyfile(ORIGINAL/'protected/protected.fasta',protected)
    assert m.sha_file(protected)=='1f6df25e70e90d41bfb6a86268372d652f9ddccff15d4b0bfaf00899dca419ae'
    rawmarker=SCI/'manifests/raw.complete.json'
    if not rawmarker.exists():shutil.copyfile(ORIGINAL/'manifests/raw.complete.json',rawmarker)
    legacy.DEFAULT_ROOT=SCI
    b=legacy.Build(SCI,legacy.REPO);b.report=report;b.raw=ORIGINAL/'raw/uniref50.fasta.gz'
    # Reuse the pinned target index READ ONLY; all search tmp/output roots are new D: paths.
    b.target_index=lambda:None
    def search(query,out,tmp):
        return ['mmseqs','easy-search',query,ORIGINAL/'mmseqs/protected_index/db',out,tmp,
            '--min-seq-id','0.30','-c','0.80','--cov-mode','0','-s','7.5','--seq-id-mode','0','--alignment-mode','3',
            '--threads','8','--split-memory-limit','4G','--max-seqs','1000','--remove-tmp-files','0',
            '--format-output','query,target,fident,qcov,tcov,evalue']
    b.search_command=search
    def command_on_d(command,log):
        b.guard();started=time.monotonic();rss=Path(log).with_suffix('.rss.txt')
        temporary=native.canonical(str(SCI/'tmp'))
        argv=['/usr/bin/env']+[f'{key}={temporary}' for key in ['TMPDIR','TMP','TEMP','XDG_CACHE_HOME']]
        argv+=['/usr/bin/time','-v','-o',native.canonical(str(rss))]+[native.canonical(str(x)) for x in command]
        if os.name=='nt':argv=['C:/Windows/System32/wsl.exe','--distribution','Ubuntu','--cd',native.canonical(str(SCI)),'--exec']+argv
        with Path(log).open('a') as handle:
            handle.write(json.dumps({'command':argv,'cwd':str(SCI),'all_scratch_on_D':True})+'\n');handle.flush()
            p=subprocess.Popen(argv,cwd=SCI,stdout=handle,stderr=subprocess.STDOUT)
            while p.poll() is None:
                try:b.guard()
                except BaseException:p.terminate();p.wait();raise
                time.sleep(1)
            assert p.returncode==0,f'Command failed ({p.returncode}); see {log}'
        match=re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',rss.read_text());assert match
        b.last_command_metrics={'wall_seconds':time.monotonic()-started,'maxrss_kib':int(match[1]),'method':'GNU time -v; D: cwd/TMPDIR/TMP/TEMP/cache'}
    b.run=command_on_d
    return legacy,b

def materialize(legacy,b):
    source=ROOT/'continuation/union/candidates.jsonl.gz';cert=ROOT/'continuation/union/union.complete.json'
    union=json.loads(cert.read_text());assert m.sha_file(source)==union['sha256']
    inputs={'union_sha256':union['sha256'],'baseline_commit':'f52bdd3','protected_sha256':m.sha_file(b.protected)}
    if b.completed('reservoir',inputs):return
    db=sqlite3.connect(b.reservoir);db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.execute('PRAGMA cache_size=-262144');db.execute('PRAGMA temp_store=FILE')
    db.execute("PRAGMA temp_store_directory='"+str(SCI/'tmp').replace("'","''")+"'")
    db.execute('CREATE TABLE IF NOT EXISTS candidates(h BLOB NOT NULL,off INTEGER NOT NULL,p BLOB,seq TEXT,source_id TEXT,multiplicity INTEGER,historical INTEGER,external INTEGER)')
    db.execute('CREATE TABLE IF NOT EXISTS load_progress(k TEXT PRIMARY KEY,v TEXT)')
    saved=db.execute("SELECT v FROM load_progress WHERE k='cursor'").fetchone()
    state=json.loads(saved[0]) if saved else {'n':0,'offset':0,'source_sha256':union['sha256']}
    assert state['source_sha256']==union['sha256']
    with b.stage('reservoir_finalize'),gzip.open(source,'rb') as f:
        f.seek(state['offset']);batch=[]
        while line:=f.readline():
            row=json.loads(line);seq=row['seq'];assert legacy.valid(seq)
            assert m.sequence_hash(seq).hex()==row['h'] and m.priority(seq).hex()==row['p']
            batch.append((bytes.fromhex(row['h']),row['off'],bytes.fromhex(row['p']),seq,row['source_id'],row['multiplicity'],row['historical'],row['external']))
            if len(batch)==100_000:
                db.executemany('INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?)',batch)
                state['n']+=len(batch);state['offset']=f.tell();batch=[]
                db.execute("INSERT OR REPLACE INTO load_progress VALUES('cursor',?)",(json.dumps(state),));db.commit();b.guard()
        if batch:
            db.executemany('INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?)',batch);state['n']+=len(batch);state['offset']=f.tell()
            db.execute("INSERT OR REPLACE INTO load_progress VALUES('cursor',?)",(json.dumps(state),));db.commit()
        assert state['n']==union['union_count']
        # Bulk index construction, avoiding 24M per-record random index updates.
        db.execute('CREATE UNIQUE INDEX IF NOT EXISTS candidate_key_index ON candidates(h,off)')
        db.execute('CREATE INDEX IF NOT EXISTS priority_index ON candidates(p,h,off,length(seq))');db.commit()
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.execute('PRAGMA journal_mode=DELETE');db.close()
        scan=json.loads((ROOT/'continuation/EXTERNAL_TAIL_COMPLETE.json').read_text())['state']
        stats={'raw_count':scan['raw'],'malformed':scan['malformed'],'length_valid':scan['length_valid'],
            'canonical_valid':scan['canonical'],'external_exact_unique_encountered':scan['unique'],'exact_duplicates_removed':scan['duplicates'],
            'external_reservoir_count':union['external_reservoir'],'historical_unique':87_930,'historical_added_to_reservoir':union['historical_added'],
            'cross_source_reservoir_overlap':union['cross_source'],'union_count':union['union_count'],
            'raw_statistics':legacy.Stats(scan['raw_stats']).summary(),'canonical_statistics':legacy.Stats(scan['valid_stats']).summary()}
        legacy.save(SCI/'stats/filtering_stats.json',stats);legacy.save(b.report/'filtering_stats.json',stats)
        b.complete('reservoir',inputs,[b.reservoir,SCI/'stats/filtering_stats.json'],statistics=stats)

def run(stage):
    verdict=json.loads((ROOT/'V2C_VERDICT.json').read_text());assert verdict['verdict']=='V2C-GO' and all(verdict['gates'].values())
    legacy,b=build()
    actions={'reservoir-finalize':lambda:materialize(legacy,b),'screen':b.screen,'verify-protected':lambda:b.screen(True),
        'final':b.final,'certify':b.certify,'diversity':b.diversity,'handoff':b.handoff}
    start=time.time();actions[stage]()
    if stage=='handoff':
        path=b.report/'chatgpt_handoff.json';handoff=legacy.json.loads(path.read_text())
        handoff['result_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=legacy.REPO,text=True).strip()
        handoff['source_baseline_commit']='f52bdd3';handoff['continuation_verdict']=verdict
        handoff['stage_timings']={p.stem:json.loads(p.read_text()) for p in (ROOT/'reports').glob('timing_*.json')}
        handoff['external_records_after_31m']=7_840_027
        handoff['recommended_next_experiment']='One fixed-budget E012 causal-RoPE data-scaling comparison between protected-clean historical unique TRAIN and the frozen V2C corpus, with unchanged architecture and evaluation panels.'
        legacy.save(path,handoff)
        text=(b.report/'FINAL_REPORT.md').read_text().replace('E012 V2B frozen corpus','E012 V2C frozen corpus')
        text+='\nV2C-GO: all nine frozen gates passed. End-to-end speedup lower bound: '+str(verdict['evidence']['speedup_lower_bound'])+'x.\n'
        (b.report/'FINAL_REPORT.md').write_text(text)
    m.atomic_json(ROOT/'reports'/f'stage_{stage}.json',{'stage':stage,'started_unix':start,'ended_unix':time.time(),
        'wall_seconds':time.time()-start,'d_free_bytes':shutil.disk_usage(ROOT).free,'training_launched':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['policy-preflight','reservoir-finalize','screen','verify-protected','final','certify','diversity','handoff']);stage=p.parse_args().stage
    if stage=='policy-preflight':
        legacy,b=build();old=legacy.Build.__new__(legacy.Build);old.root=ORIGINAL;old.protected=ORIGINAL/'protected/protected.fasta'
        args=[SCI/'tmp/policy_query.fasta',SCI/'tmp/policy_hits.tsv',SCI/'tmp/policy_search']
        expected=list(map(str,legacy.Build.search_command(old,*args)));actual=list(map(str,b.search_command(*args)))
        assert actual==expected
        m.atomic_json(ROOT/'reports/POLICY_PREFLIGHT.json',{'passed':True,'original_command':expected,'v2c_command':actual,'protected_sha256':m.sha_file(b.protected),'all_generated_paths_on_D':True})
        print('Historical MMseqs command/coverage policy and D: storage preflight passed')
    else:run(stage)
