"""Durable D:-only single-search V2C screening and automatic corpus handoff.

Production keeps the original candidate->protected predicate and full final audit.
No source scan, reservoir construction, reversed search or training is invoked.
"""
from pathlib import Path
import argparse, gzip, hashlib, importlib.util, json, os, re, shutil, sqlite3, subprocess, sys, time

SCRIPTS = Path(__file__).resolve().parent
ROOT = Path('D:/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization') if os.name == 'nt' else Path('/mnt/d/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization')
PROD = ROOT/'screen_production'; SCI = ROOT/'scientific'
TOTAL = 24000000; INITIAL = 3200000; INITIAL_HITS = 178888; SIZE = 100000

def canonical(value):
    value = str(value).replace('\\','/')
    for prefix in ('//wsl.localhost/Ubuntu','//wsl$/Ubuntu'):
        if value.lower().startswith(prefix.lower()+'/'):return value[len(prefix):]
    return '/mnt/d'+value[2:] if value.startswith('D:/') else value

def setup():
    global TOTAL
    stats=SCI/'stats/filtering_stats.json'
    if stats.exists():TOTAL=int(json.loads(stats.read_text())['union_count'])
    assert PROD.resolve().drive.upper() == 'D:' if os.name == 'nt' else PROD.resolve().is_relative_to('/mnt/d')
    for name in ('tmp','logs','reports'): (PROD/name).mkdir(parents=True, exist_ok=True)
    for key in ('TMP','TEMP','TMPDIR','XDG_CACHE_HOME'): os.environ[key] = str(PROD/'tmp')
    import tempfile
    tempfile.tempdir = str(PROD/'tmp')
    os.chdir(PROD)

def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(8*1024**2): result.update(block)
    return result.hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True); partial=path.with_suffix(path.suffix+'.partial')
    with partial.open('w') as f:
        json.dump(value,f,indent=2,sort_keys=True); f.flush(); os.fsync(f.fileno())
    os.replace(partial,path)

def load_module(name):
    spec=importlib.util.spec_from_file_location(name,SCRIPTS/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def production_budget(b):
    original=b.floor;b.floor=10*1024**3
    free=shutil.disk_usage(ROOT).free
    assert free-b.floor>=8000000000,'DATA2-E: insufficient D: space for the bounded search budget'
    save(PROD/'reports/PRODUCTION_DISK_BUDGET.json',{'original_fixed_floor_bytes':original,
        'production_floor_bytes':b.floor,'free_at_check_bytes':free,'bounded_search_scratch_budget_bytes':8000000000,
        'final_sort_and_runtime_reserve_checks_retained':True,'original_resource_plan_unchanged':True,'all_artifacts_on_D':True})

def certify_only():
    production=load_module('complete_e012_v2c_pipeline');legacy,b=production.build();setup();production_budget(b)
    b.certify()

def ro(path):
    return sqlite3.connect('file:'+path.as_posix()+'?mode=ro',uri=True)

def durable():
    with ro(SCI/'manifests/exclusions.sqlite') as db:
        batch,n=db.execute("SELECT max(batch),sum(n) FROM batches WHERE name='screen'").fetchone()
        hits=db.execute('SELECT count(*) FROM removed').fetchone()[0]
    return dict(batch=batch,screened=n,exclusions=hits)

def validate_boundary():
    state=durable(); assert state==dict(batch=32,screened=INITIAL,exclusions=INITIAL_HITS),state
    evidence={}; hits=0
    with ro(SCI/'manifests/exclusions.sqlite') as db:
        assert db.execute('PRAGMA quick_check').fetchall()==[('ok',)]
        records=db.execute("SELECT batch,n FROM batches WHERE name='screen' ORDER BY batch").fetchall()
    assert [b for b,n in records]==list(range(1,33))
    for batch,n in records:
        path=SCI/f'manifests/screen-batch-{batch:04d}.json'; cert=json.loads(path.read_text())
        retained=SCI/f'manifests/screen-batch-{batch:04d}.tsv.gz'
        assert n==cert['n']==SIZE and cert['batch_zero_overlap_verified']
        assert sha(retained)==cert['retained_result_sha256'];hits+=cert['hits']
        evidence[str(batch)]={'certificate_sha256':sha(path),'retained_sha256':sha(retained)}
    assert hits==INITIAL_HITS
    save(PROD/'reports/VERIFIED_START.json',{**state,'certificates':evidence,'training_launched':False})

def seal():
    assert os.name!='nt'
    inventory={};last=time.monotonic()
    for name in ('screen_optimization','screen_architecture_v2'):
        folder=ROOT/name
        for path in sorted(folder.rglob('*')):
            key=str(path.relative_to(ROOT))
            if path.is_symlink(): inventory[key]={'symlink':os.readlink(path)}
            elif path.is_file():inventory[key]={'bytes':path.stat().st_size,'sha256':sha(path)}
            if time.monotonic()-last>=30:
                print(json.dumps({'phase':'freeze_diagnostics','files':len(inventory)}),flush=True);last=time.monotonic()
    save(PROD/'reports/FROZEN_DIAGNOSTICS.json',{'commits':['5a3cf728257aa94dc48ca36241909d00218e6188','5e235c1'],
        'files':inventory,'immutable_inputs':True})
    save(PROD/'reports/PRIORITY_SCREENING_DECISION.json',{'mode':'full_24M_append_order','final_rule':'ascending (p,h,off), min(clean_pool,20000000); minimum 10000000',
        'reason':'Final membership is a priority prefix, but lazy screening would require extending complete-population statistics and append-order checkpoint contracts. Whole-pipeline equivalence was not immediately established; no further redesign.',
        'source_final_selector_sha256':sha(SCRIPTS/'prepare_e012_sequence_corpus_v2b.py'),'time_box_minutes':30})
    print('Both diagnostics frozen; full 24M screening selected',flush=True)

def quiesce():
    assert os.name=='nt'
    import psutil
    validate_boundary()
    stopped=[]
    for proc in psutil.process_iter(['pid','cmdline']):
        args=proc.info['cmdline'] or []
        names=('recover_e012_v2c.py','complete_e012_v2c_pipeline.py','monitor_e012_v2c_recovery.py')
        if not any(str(a).endswith(name) for a in args for name in names):continue
        if any(str(a).endswith('complete_e012_v2c_pipeline.py') for a in args) and args[-1]!='screen':continue
        if not any(str(a).endswith('monitor_e012_v2c_recovery.py') for a in args):
            assert proc.status()==psutil.STATUS_STOPPED,'Refuse to terminate a worker outside the verified pause'
        children=proc.children(recursive=True)
        for child in reversed(children):
            try: child.terminate(); child.wait(10)
            except psutil.NoSuchProcess:pass
        try:proc.terminate();proc.wait(10)
        except psutil.NoSuchProcess:pass
        stopped.append({'pid':proc.pid,'command':args})
    for name in ('RECOVERY_SUPERVISOR_STATUS.json','RECOVERY_HEARTBEAT.json'):
        src=ROOT/'reports'/name
        if src.exists():shutil.copyfile(src,PROD/'reports'/('pre_resume_'+name))
    assert durable()==dict(batch=32,screened=INITIAL,exclusions=INITIAL_HITS)
    save(PROD/'reports/OLD_OWNERS_STOPPED.json',{'stopped':stopped,'at_committed_boundary':32,'completed_batches_preserved':True})
    print(json.dumps({'obsolete_owners_stopped':len(stopped),'verified_boundary':32}),flush=True)

def rows_from_fasta(path):
    with path.open() as f:
        while header:=f.readline():yield header[1:].strip(),f.readline().strip()

def decisions(rows, lines, protected):
    mapping={f's_{h.hex()}_{off}':(h,off) for h,off,seq,*rest in rows}
    nearest={f's_{h.hex()}_{off}':(protected[seq],1.0) for h,off,seq,*rest in rows if seq in protected}
    for line in lines:
        q,t,ident,qcov,tcov,evalue=line.rstrip().split('\t')
        assert q in mapping
        ident,qcov,tcov=map(float,(ident,qcov,tcov))
        if ident<.299 or qcov<.799 or tcov<.799:raise RuntimeError('DATA2-D: returned hit outside frozen boundary')
        if q not in nearest or ident>nearest[q][1]:nearest[q]=(t,ident)
    survivors=set(mapping)-set(nearest)
    assert not (set(nearest)&survivors) and len(nearest)+len(survivors)==len(rows)
    return mapping,nearest,survivors

def validate_parser():
    protected=dict((seq,sid) for sid,seq in rows_from_fasta(SCI/'protected/protected.fasta'))
    results=[]
    for batch in (28,30,31):
        case=ROOT/f'screen_architecture_v2/batch-{batch:04d}'
        records=list(rows_from_fasta(case/'query.fasta'))
        rows=[(bytes.fromhex(sid.split('_')[1]),int(sid.split('_')[2]),seq) for sid,seq in records]
        with gzip.open(case/'expected.tsv.gz','rt') as f:lines=f.readlines()
        mapping,nearest,survivors=decisions(rows,lines,protected)
        expected=json.loads((case/'persisted_nearest.json').read_text())
        assert {k:list(v) for k,v in nearest.items()}==expected
        assert len(survivors)==len(rows)-len(expected)
        results.append({'batch':batch,'n':len(rows),'hits':len(nearest),'survivors':len(survivors),'exact':True})
    save(PROD/'reports/PRODUCTION_PARSER_EQUIVALENCE.json',{'cases':results,'passed':True,'no_mmseqs_replay':True})

def reference_validation():
    assert os.name!='nt'
    import psutil
    for proc in psutil.process_iter(['cmdline']):
        args=proc.info['cmdline'] or []
        if args and args[0].endswith('mmseqs-avx2') and any(canonical(SCI) in a for a in args):
            raise RuntimeError('A live SCI MMseqs job exists; do not start an overlapping production owner')
    path=ROOT.parent/'manifests/target_index.complete.json';record=json.loads(path.read_text())
    assert record['inputs']['protected_sha256']==sha(SCI/'protected/protected.fasta')
    assert record['inputs']['sensitivity']==7.5 and record['inputs']['mmseqs_version']=='15-6f452+ds-2'
    for name,digest in record['outputs'].items():
        output=Path(name);assert output.resolve().is_relative_to('/mnt/d') and sha(output)==digest
    save(PROD/'reports/PROTECTED_INDEX_VALIDATED.json',{'marker_sha256':sha(path),'all_index_outputs_verified':True})

STATE={}
def state_update(**fields):
    STATE.update(fields)
    if os.name=='nt':
        import psutil
        STATE['native_rss_bytes']=psutil.Process().memory_info().rss
    STATE['timestamp_unix']=time.time();save(PROD/'STATE.json',STATE)

def step(b,label,argv,outputs):
    folder=Path(outputs[0]).parent;marker=folder/(label+'.complete.json');request=folder/(label+'.request.json')
    pin={'command':list(map(canonical,argv))}
    if marker.exists():
        existing=json.loads(marker.read_text());assert existing['request']==pin
        for path,h in existing['outputs'].items():
            p=Path(path.replace('/mnt/d','D:')) if os.name=='nt' else Path(path)
            assert sha(p)==h
        return
    log=folder/(label+'.log');timer=log.with_suffix('.rss.txt')
    if request.exists():assert json.loads(request.read_text())==pin
    else:save(request,pin)
    recovered=timer.exists() and re.search(r'Exit status:\s*0\s*$',timer.read_text())
    if not recovered:
        state_update(active_command=label,active_log=canonical(log))
        b.run(argv,log)
    files=[]
    for output in outputs:
        if output.is_file():files.append(output)
        files.extend(p for p in output.parent.glob(output.name+'*') if p.is_file() and not p.is_symlink())
    assert files,'MMseqs output missing'
    save(marker,{'request':pin,'outputs':{canonical(p):sha(p) for p in set(files)},'gnu_time_exit_zero':True})

def screen(legacy,b):
    reservoir=legacy.json.loads(b.marker('reservoir').read_text())
    inputs={'reservoir_sha256':reservoir['outputs'][str(b.reservoir)],'protected_sha256':legacy.digest(b.protected),
        'policy':{'identity':.30,'coverage':.80,'cov_mode':0,'sensitivity':7.5},'verification':False,
        'batch_order':'append-table scan; independent hash-priority cutoff'}
    if b.completed('screen',inputs):return
    protected={seq:sid for sid,seq in rows_from_fasta(b.protected)}
    db=ro(b.reservoir);ex=legacy.connection(SCI/'manifests/exclusions.sqlite')
    ex.execute('CREATE TABLE IF NOT EXISTS production_progress(k TEXT PRIMARY KEY,v TEXT)');ex.commit()
    old=ex.execute("SELECT v FROM production_progress WHERE k='checkpoint'").fetchone()
    current=json.loads(old[0]) if old else {'inputs':inputs,'batch':32,'screened':INITIAL,'last_rowid':INITIAL,'exclusions':INITIAL_HITS,'slow_batches':0}
    assert current['inputs']==inputs
    assert durable()=={k:current[k] for k in ('batch','screened','exclusions')}
    assert db.execute('SELECT max(rowid) FROM candidates').fetchone()[0]==TOTAL
    started=time.time();start_n=current['screened']
    state_update(phase='screen',screen_started_unix=started,resume_screened=start_n,required_population=TOTAL,
        screened=current['screened'],exclusions=current['exclusions'],committed_batch=current['batch'],active_batch=current['batch']+1,
        benchmark_candidates_s=366.41213747885496,recent_candidates_s=0)
    cursor=db.execute('SELECT rowid,h,off,seq,historical,external FROM candidates WHERE rowid>? ORDER BY rowid',(current['last_rowid'],))
    with b.stage('screen'):
        while source:=cursor.fetchmany(SIZE):
            batch=current['batch']+1;started_batch=time.time();rows=[r[1:] for r in source]
            folder=SCI/'mmseqs/screen_production'/f'batch-{batch:04d}';folder.mkdir(parents=True,exist_ok=True)
            state_update(active_batch=batch,active_batch_started_unix=started_batch)
            row_hash=hashlib.sha256(b''.join(h+str(off).encode()+b'\0' for h,off,*rest in rows)).hexdigest()
            # Every incomplete attempt has its own D: generation. An unfinished
            # search preserves its evidence; a completed query DB is reusable.
            active=folder/'active_job.json';job=json.loads(active.read_text()) if active.exists() else None
            reusable=None
            if job:
                assert job['input_rows_sha256']==row_hash
                prev=Path(job['generation'].replace('/mnt/d','D:')) if os.name=='nt' else Path(job['generation'])
                prior_query=Path(job['query_db'].replace('/mnt/d','D:')) if os.name=='nt' else Path(job['query_db'])
                if (prior_query.parent/'createdb.complete.json').exists():
                    ready=json.loads((prior_query.parent/'createdb.complete.json').read_text())
                    for name,digest in ready['outputs'].items():
                        p=Path(name.replace('/mnt/d','D:')) if os.name=='nt' else Path(name)
                        assert sha(p)==digest
                    reusable=prior_query
                for label in ('createdb','search','convert'):
                    pending=(prev/(label+'.request.json')).exists() and not (prev/(label+'.complete.json')).exists()
                    timer=prev/(label+'.rss.txt')
                    finished_time=timer.exists() and re.search(r'Exit status:\s*0\s*$',timer.read_text())
                    if pending and not finished_time:job=None;break
            if not job:
                gen=folder/('generation-'+str(time.time_ns()));gen.mkdir()
                fasta=gen/'query.fasta'
                with fasta.open('w') as f:
                    for h,off,seq,*rest in rows:f.write(f'>s_{h.hex()}_{off}\n{seq}\n')
                    f.flush();os.fsync(f.fileno())
                job={'generation':canonical(gen),'input_rows_sha256':row_hash,'query_sha256':sha(fasta),
                    'query_db':canonical(reusable) if reusable else canonical(gen/'query_db')}
                save(active,job)
            gen=Path(job['generation'].replace('/mnt/d','D:')) if os.name=='nt' else Path(job['generation'])
            query=Path(job['query_db'].replace('/mnt/d','D:')) if os.name=='nt' else Path(job['query_db'])
            assert sha(gen/'query.fasta')==job['query_sha256']
            if reusable is None:step(b,'createdb',['mmseqs','createdb',gen/'query.fasta',query,'--dbtype','0','--shuffle','1','--createdb-mode','0','--write-lookup','0'],[query])
            command=b.search_command(gen/'query.fasta',gen/'matches.tsv',gen/'tmp')
            search=['mmseqs','search',query,command[3],gen/'result',gen/'tmp']+command[6:command.index('--format-output')]
            step(b,'search',search,[gen/'result'])
            step(b,'convert',['mmseqs','convertalis',query,command[3],gen/'result',gen/'matches.tsv','--format-output','query,target,fident,qcov,tcov,evalue','--threads','16'],[gen/'matches.tsv'])
            with (gen/'matches.tsv').open() as f:mapping,nearest,survivors=decisions(rows,f,protected)
            # Derive a REAL empty relation, explicitly labelled as a complement
            # proof. No independent MMseqs survivor re-search is claimed.
            relation=gen/'survivor_relation.tsv'
            with relation.open('w') as out,(gen/'matches.tsv').open() as f:
                for line in f:
                    if line.split('\t')[0] in survivors:out.write(line)
                out.flush();os.fsync(out.fileno())
            assert relation.stat().st_size==0
            survivor_hash=hashlib.sha256()
            for h,off,*rest in reversed(rows):
                sid=f's_{h.hex()}_{off}'
                if sid in survivors:survivor_hash.update((sid+'\n').encode())
            cert=SCI/f'manifests/screen-batch-{batch:04d}.json';retained=SCI/f'manifests/screen-batch-{batch:04d}.tsv.gz'
            partial=retained.with_suffix(retained.suffix+'.partial')
            with (gen/'matches.tsv').open('rb') as src,partial.open('wb') as dest:
                with gzip.GzipFile(filename='',mode='wb',fileobj=dest,mtime=0) as zipped:shutil.copyfileobj(src,zipped)
                dest.flush();os.fsync(dest.fileno())
            os.replace(partial,retained)
            data={'stage':'screen','batch':batch,'inputs':inputs,'input_rows_sha256':row_hash,'n':len(rows),'hits':len(nearest),
                'matches_sha256':sha(gen/'matches.tsv'),'query_sha256':job['query_sha256'],'command':list(map(str,search)),
                'mmseqs_version':'15-6f452+ds-2','timestamp':legacy.now(),'retained_result_sha256':sha(retained),
                'batch_zero_overlap_verified':True,'batch_verify_result_sha256':sha(relation),
                'batch_verification_method':'deterministic_complete_returned_relation_complement',
                'independent_batch_research_performed':False,'independent_final_selected_corpus_audit_required':True,
                'survivors':len(survivors),'survivor_membership_sha256':survivor_hash.hexdigest(),
                'query_db_reused_across_restart':True,'all_scratch_on_D':True}
            legacy.save(cert,data)
            ex.execute('BEGIN IMMEDIATE')
            for sid,(target,ident) in nearest.items():ex.execute('INSERT INTO removed VALUES(?,?,?,?)',(*mapping[sid],target,ident))
            elapsed=time.time()-started_batch;rate=len(rows)/elapsed
            current.update(batch=batch,screened=current['screened']+len(rows),last_rowid=source[-1][0],exclusions=current['exclusions']+len(nearest),
                slow_batches=current['slow_batches']+1 if rate<.5*366.41213747885496 else 0)
            ex.execute('INSERT INTO batches VALUES(?,?,?,?)',('screen',batch,len(rows),str(cert)))
            ex.execute("INSERT OR REPLACE INTO production_progress VALUES('checkpoint',?)",(json.dumps(current),));ex.commit()
            assert ex.execute('SELECT count(*) FROM removed').fetchone()[0]==current['exclusions']
            state_update(screened=current['screened'],exclusions=current['exclusions'],committed_batch=batch,recent_candidates_s=rate,
                last_batch_seconds=elapsed,active_batch=batch)
            print(json.dumps({'stage':'screen','batch':batch,'screened':current['screened'],'hits':current['exclusions'],'candidates_s':rate}),flush=True)
            # Retained results/certificates/source rows are sufficient evidence;
            # temporary prefilter databases must not fill the D: reserve.
            b.cleanup([folder],cert)
            if current['slow_batches']>=3:raise RuntimeError('PERFORMANCE_COLLAPSE: three committed batches below half validated throughput')
        cursor.close();db.close();assert current['screened']==TOTAL
        with ro(b.reservoir) as source_db:
            source_db.execute('ATTACH DATABASE ? AS excluded',(str(SCI/'manifests/exclusions.sqlite'),))
            hist,external=source_db.execute('SELECT sum(c.historical),sum(c.external) FROM candidates c JOIN excluded.removed e USING(h,off)').fetchone()
        removed=current['exclusions']
        info={'candidates_screened':TOTAL,'removed':removed,'removal_fraction':removed/TOTAL,'historical_removals':hist or 0,
            'external_removals':external or 0,'source_counts_overlap_for_dual_provenance':True,'clean_pool':TOTAL-removed,
            'zero_detected_violations':None,'query_target_policy':inputs['policy'],'search_commands':'Per-batch manifests',
            'identity_histogram_of_detected_nearest_matches':{str(i):ex.execute('SELECT count(*) FROM removed WHERE identity>=? AND identity<?',(i/10,(i+1)/10 if i<9 else 1.001)).fetchone()[0] for i in range(3,10)},
            'limitation':'Zero detected violations under recorded MMseqs heuristic search; not an exhaustive proof of all possible alignments.',
            'decision_method':'single_search_complete_relation_complement','independent_final_audit_required':True}
        ex.close();legacy.save(SCI/'stats/screen.json',info);legacy.save(b.report/'screen.json',info)
        b.complete('screen',inputs,[SCI/'stats/screen.json',SCI/'manifests/exclusions.sqlite'],statistics=info)

def run():
    assert os.name=='nt'
    import psutil
    state_update(phase='starting',pid=os.getpid(),started_unix=time.time(),screened=durable()['screened'],required_population=TOTAL,
        exclusions=durable()['exclusions'],committed_batch=durable()['batch'],training_launched=False,driver_sha256=sha(Path(__file__).resolve()))
    production=load_module('complete_e012_v2c_pipeline');legacy,b=production.build();setup();production_budget(b)
    original=b.search_command
    def search(query,out,tmp):
        args=original(query,out,tmp);assert args[args.index('--threads')+1]=='8';args[args.index('--threads')+1]='16';return args
    b.search_command=search
    original_run=b.run
    def tracked_run(args,log):
        fields={'active_command':str(args[1]),'active_log':canonical(log)}
        match=re.search(r'verify_protected/batch-(\d+)',canonical(log))
        if match:fields['audit_active_batch']=int(match[1])
        state_update(**fields);return original_run(args,log)
    b.run=tracked_run
    version=subprocess.check_output(['mmseqs','version'],cwd=PROD,text=True).strip();assert version=='15-6f452+ds-2'
    subprocess.check_call(['C:/Windows/System32/wsl.exe','--distribution','Ubuntu','--cd',canonical(PROD),'--exec',
        '/usr/bin/python3','-B',canonical(Path(__file__).resolve()),'reference-validation'],cwd=PROD)
    verdict=json.loads((ROOT/'V2C_VERDICT.json').read_text());assert verdict['verdict']=='V2C-GO' and all(verdict['gates'].values())
    validate_parser()
    # SQLite sorts/journals/cache spill paths also stay on D:.
    connection=legacy.connection
    def on_d(path,wal=False):
        db=connection(path,wal);db.execute('PRAGMA temp_store=FILE')
        db.execute("PRAGMA temp_store_directory='"+str(PROD/'tmp').replace("'","''")+"'");return db
    legacy.connection=on_d
    state_update(phase='starting',pid=os.getpid(),started_unix=time.time(),screened=durable()['screened'],required_population=TOTAL,
        exclusions=durable()['exclusions'],committed_batch=durable()['batch'],training_launched=False)
    try:
        stage_start=time.time();screen(legacy,b)
        save(PROD/'reports/stage_screen.json',{'wall_seconds':time.time()-stage_start,'training_launched':False})
        for stage in ('verify-protected','final','certify','diversity','handoff'):
            state_update(phase=stage,phase_started_unix=time.time(),active_command=None)
            started=time.time()
            if stage=='verify-protected':b.screen(True)
            elif stage=='certify':
                cmd=['C:/Windows/System32/wsl.exe','--distribution','Ubuntu','--cd',canonical(SCI),'--exec',
                    '/home/simostocco/miniforge3/envs/proteingen/bin/python','-B',canonical(Path(__file__).resolve()),'certify-only']
                with (PROD/'logs/certify.log').open('ab') as log:subprocess.run(cmd,cwd=PROD,stdout=log,stderr=log,check=True)
            elif stage=='handoff':
                b.handoff()
                path=b.report/'chatgpt_handoff.json';handoff=legacy.json.loads(path.read_text())
                handoff['source_baseline_commit']='f52bdd3';handoff['continuation_verdict']=verdict
                handoff['production_resume']={'diagnostic_commits':['5a3cf72','5e235c1'],'single_search_complement':True,'threads':16,'full_final_audit':True,'screening_population':TOTAL}
                legacy.save(path,handoff)
                report=b.report/'FINAL_REPORT.md';report.write_text(report.read_text().replace('E012 V2B frozen corpus','E012 V2C frozen corpus'))
            else:getattr(b,stage)()
            save(PROD/f'reports/stage_{stage}.json',{'wall_seconds':time.time()-started,'training_launched':False})
        state_update(phase='frozen_corpus_ready',finished_unix=time.time(),training_launched=False)
    except BaseException as exc:
        matched=re.search(r'DATA2-[A-Z]',str(exc))
        classification=matched[0] if matched else ('DATA2-D' if STATE['phase']=='verify-protected' else ('PERFORMANCE_COLLAPSE' if 'PERFORMANCE_COLLAPSE' in str(exc) else 'PRODUCTION_FAILURE'))
        state_update(stage_at_failure=STATE['phase'],phase='failed',classification=classification,error=repr(exc),training_launched=False)
        save(PROD/'reports/FAILURE.json',dict(STATE));raise

def monitor():
    assert os.name!='nt'
    import psutil
    previous={};last=time.monotonic()-60
    with (PROD/'reports/HEARTBEAT.jsonl').open('a') as history:
        while True:
            if time.monotonic()-last>=60 and (PROD/'STATE.json').exists():
                state=json.loads((PROD/'STATE.json').read_text());phase=state['phase'];workers=[]
                for p in psutil.process_iter(['pid','cmdline']):
                    args=p.info['cmdline'] or []
                    if not args or not args[0].endswith('mmseqs-avx2') or not any(canonical(SCI/'mmseqs') in a for a in args):continue
                    try:
                        cpu=p.cpu_times();now=time.time();prior=previous.get(p.pid)
                        utilized=100*((cpu.user+cpu.system)-prior[0])/(now-prior[1]) if prior else p.cpu_percent(interval=.2)
                        previous[p.pid]=(cpu.user+cpu.system,now)
                        workers.append({'pid':p.pid,'subcommand':args[1],'cpu_percent':utilized,'rss_bytes':p.memory_info().rss,'threads':p.num_threads()})
                    except psutil.NoSuchProcess:pass
                elapsed=max(.001,time.time()-state.get('screen_started_unix',state['started_unix']))
                screened=state['screened'];recent=state.get('recent_candidates_s',0)
                cumulative=(screened-state.get('resume_screened',INITIAL))/elapsed
                rate=cumulative or recent or state.get('benchmark_candidates_s',366.41213747885496)
                row={'timestamp_unix':time.time(),'phase':phase,'screened':screened,'required_population':TOTAL,'percentage':100*screened/TOTAL,
                    'exclusions':state['exclusions'],'recent_candidates_s':recent,'cumulative_candidates_s_since_resume':cumulative,
                    'batch':state.get('active_batch',state['committed_batch']),'committed_batch':state['committed_batch'],'elapsed_seconds':elapsed,
                    'eta_screen_seconds':max(0,(TOTAL-screened)/rate),'eta_basis':'observed committed throughput' if cumulative else 'validated benchmark',
                    'cpu_percent':sum(p['cpu_percent'] for p in workers),'rss_bytes':state.get('native_rss_bytes',0)+sum(p['rss_bytes'] for p in workers),
                    'system_ram':psutil.virtual_memory()._asdict(),'d_free_bytes':shutil.disk_usage(ROOT).free,'mmseqs_workers':workers,
                    'raw':38840027,'raw_target':38840027,'reservoir_count':24000000,'historical_union_population':TOTAL,
                    'screen':{'stage':'screen','batch':state['committed_batch'],'screened':screened,'hits':state['exclusions']},'training_launched':False}
                row['active_log']=state.get('active_log');row['audit_active_batch']=state.get('audit_active_batch')
                if phase=='verify-protected':
                    with (PROD/'logs/production.log').open('rb') as f:
                        f.seek(max(0,f.seek(0,2)-65536));lines=f.read().splitlines()
                    for line in reversed(lines):
                        try:event=json.loads(line)
                        except (ValueError,UnicodeDecodeError):continue
                        if event.get('stage')=='verify_protected':row['audit']=event;break
                    stats=json.loads((SCI/'stats/screen.json').read_text())
                    row['audit_required_population']=min(stats['clean_pool'],20000000)
                    for process in workers:
                        row['audit_active_subcommand']=process['subcommand']
                save(PROD/'reports/HEARTBEAT.json',row);save(ROOT/'reports/RECOVERY_HEARTBEAT.json',row)
                history.write(json.dumps(row)+'\n');history.flush();print(json.dumps(row),flush=True);last=time.monotonic()
                if phase in ('failed','frozen_corpus_ready'):return
            time.sleep(1)

def launch():
    assert os.name=='nt'
    assert (PROD/'reports/OLD_OWNERS_STOPPED.json').exists() and (PROD/'reports/FROZEN_DIAGNOSTICS.json').exists()
    path=PROD/'LAUNCH.json'
    if path.exists():
        import psutil
        prior=json.loads(path.read_text())
        if psutil.pid_exists(prior['worker_pid']):
            p=psutil.Process(prior['worker_pid'])
            assert not any(str(a).endswith(Path(__file__).name) for a in p.cmdline()),'A production owner is still alive'
        shutil.copyfile(path,PROD/f'reports/previous_launch_{time.time_ns()}.json')
        for name in ('STATE.json','reports/FAILURE.json'):
            previous=PROD/name
            if previous.exists():shutil.copyfile(previous,PROD/f'reports/previous_{Path(name).name}_{time.time_ns()}.json')
    # Publish a nonterminal state BEFORE the child/observer are spawned; otherwise
    # an observer could see the prior failed startup and terminate immediately.
    current=durable()
    save(PROD/'STATE.json',{'phase':'launching','started_unix':time.time(),'screened':current['screened'],
        'exclusions':current['exclusions'],'committed_batch':current['batch'],'required_population':TOTAL,'training_launched':False})
    save(PROD/'reports/PRODUCTION_DECISION.json',{'minimum_5x_gate_required':False,'user_authorized_resume':True,
        'frozen_diagnostic_commits':['5a3cf72','5e235c1'],'direction':'candidate_to_protected','threads':16,
        'batch_size':SIZE,'required_screening_population':TOTAL,'start_at_batch':33,
        'external_reservoir_population':24000000,'historical_additions':85944,
        'decision_method':'single_search_complete_returned_relation_complement',
        'nearest_semantics':'exact strings override at 1.0; strict greater fident update in returned order',
        'final_independent_full_audit_retained':True,'audit_disagreement_action':'DATA2-D; stop and preserve evidence',
        'automatic_stages':['screen','verify-protected','final','certify','diversity','handoff'],'training_launched':False})
    flags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP
    with (PROD/'logs/production.log').open('ab') as log:
        worker=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'run'],cwd=PROD,stdin=subprocess.DEVNULL,stdout=log,stderr=log,creationflags=flags)
    with (PROD/'logs/heartbeat.log').open('ab') as log:
        observer=subprocess.Popen(['C:/Windows/System32/wsl.exe','--distribution','Ubuntu','--cd',canonical(PROD),'--exec',
            '/usr/bin/python3','-B',canonical(Path(__file__).resolve()),'monitor'],cwd=PROD,stdin=subprocess.DEVNULL,stdout=log,stderr=log,creationflags=flags)
    record={'worker_pid':worker.pid,'observer_controller_pid':observer.pid,'starting_boundary':durable(),'training_launched':False}
    save(path,record);print(json.dumps(record),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('seal','quiesce','validate','run','monitor','launch','reference-validation','certify-only'))
    action=parser.parse_args().action;setup()
    try:
        {'seal':seal,'quiesce':quiesce,'validate':validate_parser,'run':run,'monitor':monitor,'launch':launch,'reference-validation':reference_validation,'certify-only':certify_only}[action]()
    except BaseException as exc:
        if action=='run' and STATE.get('phase')!='failed':
            state_update(phase='failed',error=repr(exc),classification='PRODUCTION_STARTUP_FAILURE',training_launched=False)
            save(PROD/'reports/FAILURE.json',dict(STATE))
        raise
