"""Read-only source evidence inventory; generated diagnostics always on D:."""
import argparse, gzip, hashlib, json, os, platform, shutil, sqlite3, subprocess, sys, time
from pathlib import Path

BASE = '/mnt/d/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2'
def path(s):
    return Path(s.replace('/mnt/d', 'D:') if os.name == 'nt' else s)
ROOT = path(BASE + '/v2c_optimization')
SCI = ROOT/'scientific'
DIAG = ROOT/'data2d_batch200_diagnostic'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(8*1024*1024): h.update(b)
    return h.hexdigest()
def save(name, obj):
    p=DIAG/name
    if p.exists(): raise RuntimeError('Evidence already exists: '+str(p))
    p.write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n')
def ro(p): return sqlite3.connect(p.as_uri()+'?mode=ro', uri=True)
def preserve():
    DIAG.mkdir(exist_ok=True)
    files=[ROOT/'screen_production'/n for n in ('STATE.json','reports/FAILURE.json','reports/HEARTBEAT.json','logs/production.log')]
    files += list((SCI/'mmseqs/verify_protected/batch-0200').glob('*.*'))
    files += [SCI/'protected/protected.fasta',SCI/'manifests/exclusions.sqlite',SCI/'manifests/verification_progress.sqlite', SCI/'manifests/screen-batch-0241.json', SCI/'manifests/screen-batch-0241.tsv.gz']
    files += [p for p in path(BASE+'/mmseqs/protected_index').glob('db*') if p.is_file()]
    files += list((SCI/'reports/command_logs/mmseqs/screen_production/batch-0241').rglob('*.*'))
    records=[]
    for p in files:
        if not p.is_file(): continue
        records.append({'path':str(p),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':sha(p)})
        if p.suffix in ('.log','.json','.txt','.gz') or p.name in ('query.fasta','matches.tsv'):
            dst=DIAG/'evidence'/p.relative_to(ROOT) if p.is_relative_to(ROOT) else DIAG/'evidence'/p.name
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    save('EVIDENCE.json',{'files':records,'disk':list(shutil.disk_usage(ROOT)),'python':sys.version,'platform':platform.platform()})
def inspect():
    audit=SCI/'mmseqs/verify_protected/batch-0200'
    alignments=[];queries=set();targets=set()
    for line in (audit/'matches.tsv').read_text().splitlines():
        q,t,i,qc,tc,e=line.split('\t');queries.add(q);targets.add(t)
        alignments.append({'query':q,'target':t,'fident':float(i),'qcov':float(qc),'tcov':float(tc),'evalue':float(e)})
    save('ALIGNMENTS.json',alignments)
    logs=[]
    for p in sorted((SCI/'reports/command_logs/mmseqs/screen_production').rglob('createdb.log')):
        text=p.read_text();typ=[x for x in text.splitlines() if x.startswith('Database type:')]
        logs.append({'path':str(p),'types':typ,'sha256':sha(p)})
    save('DB_TYPES.json',logs)
    db=ro(SCI/'filtered/reservoir.sqlite');ex=ro(SCI/'manifests/exclusions.sqlite')
    candidates=[]
    for rowid,h,off,seq,hist,ext,p in db.execute('SELECT rowid,h,off,seq,historical,external,p FROM candidates WHERE off<0'):
        sid=f's_{h.hex()}_{off}'
        if sid not in queries: continue
        candidates.append({'query':sid,'hash':h.hex(),'sequence_hash':hashlib.sha256(seq.encode()).hexdigest(),'offset':off,'sequence':seq,'length':len(seq),'rowid':rowid,'main_batch':(rowid-1)//100000+1,'historical':hist,'external':ext,'priority':p.hex() if isinstance(p,bytes) else p,'excluded':ex.execute('SELECT protected,identity FROM removed WHERE h=? AND off=?',(h,off)).fetchone()})
    save('OFFENDERS.json',candidates)
    save('SUMMARY.json',{'alignments':len(alignments),'queries':len(queries),'targets':len(targets),'matched_historical_queries':len(candidates),'main_batches':sorted(set(x['main_batch'] for x in candidates)),'ranges':{k:[min(x[k] for x in alignments),max(x[k] for x in alignments)] for k in ('fident','qcov','tcov','evalue')},'nucleotide_batches':[x['path'] for x in logs if 'Database type: Nucleotide' in x['types']]})
def prepare():
    db=ro(SCI/'filtered/reservoir.sqlite')
    rows=db.execute('SELECT h,off,seq,historical,external FROM candidates WHERE rowid>24000000 ORDER BY rowid').fetchall()
    def fasta(name, records):
        p=DIAG/name
        if p.exists(): raise RuntimeError('Already prepared '+name)
        p.write_text(''.join(f'>s_{h.hex()}_{off}\n{seq}\n' for h,off,seq,*_ in records))
        return sha(p)
    digest=fasta('original-batch241.fasta',rows)
    cert=json.loads((SCI/'manifests/screen-batch-0241.json').read_text())
    assert digest==cert['query_sha256']
    assert hashlib.sha256(b''.join(h+str(off).encode()+b'\0' for h,off,*_ in rows)).hexdigest()==cert['input_rows_sha256']
    fasta('reversed-batch241.fasta',list(reversed(rows)))
    offenders=json.loads((DIAG/'OFFENDERS.json').read_text());keys={x['query'] for x in offenders}
    fasta('offenders.fasta',[r for r in rows if f's_{r[0].hex()}_{r[1]}' in keys])
    # Includes the exact leading historical query responsible for autodetection.
    panel=rows[:32]+[r for r in rows if f's_{r[0].hex()}_{r[1]}' in keys][:32]
    fasta('panel.fasta',panel);fasta('reversed-panel.fasta',list(reversed(panel)))
    fasta('single-offender.fasta',[r for r in rows if f's_{r[0].hex()}_{r[1]}' in keys][:1])
    protected={}
    with (SCI/'protected/protected.fasta').open() as f:
        for line in f:
            if line.startswith('>'):sid=line[1:].strip();protected[sid]=''
            else:protected[sid]+=line.strip()
    targets={x['target'] for x in json.loads((DIAG/'ALIGNMENTS.json').read_text())}
    save('PROTECTED_TARGETS.json',{k:protected[k] for k in sorted(targets)})
    pin=json.loads(path(BASE+'/manifests/target_index.complete.json').read_text())
    checks={p:sha(path(p))==v for p,v in pin['outputs'].items()};assert all(checks.values())
    save('PREPARATION.json',{'batch241_reconstructed_sha256':digest,'rows':len(rows),'first_query':{'hash':rows[0][0].hex(),'off':rows[0][1],'seq':rows[0][2]},'protected_index_pins_verified':checks,'protected_fasta_pin_verified':sha(SCI/'protected/protected.fasta')==pin['inputs']['protected_sha256']})
def reproduce(validation=False, cap=False, fullcap=False):
    assert os.name!='nt'
    scratch=DIAG/'scratch';scratch.mkdir(exist_ok=True)
    env=dict(os.environ,**{k:str(scratch) for k in ('TMPDIR','TMP','TEMP','XDG_CACHE_HOME')})
    target=path(BASE+'/mmseqs/protected_index/db')
    settings=['--min-seq-id','0.30','-c','0.80','--cov-mode','0','-s','7.5','--seq-id-mode','0','--alignment-mode','3','--split-memory-limit','4G','--remove-tmp-files','0']
    def run(name,fa,mode='main',dbtype='0',threads=16,maxseq=1000):
        folder=DIAG/'runs'/name
        if (folder/'COMPLETE.json').exists(): return json.loads((folder/'COMPLETE.json').read_text())
        assert not folder.exists(), 'Incomplete diagnostic: preserve and name a fresh run'
        folder.mkdir(parents=True);out=folder/'matches.tsv'
        params=settings+['--threads',str(threads),'--max-seqs',str(maxseq)]
        if mode=='main':
            commands=[['mmseqs','createdb',fa,folder/'query_db','--dbtype',dbtype,'--shuffle','1','--createdb-mode','0','--write-lookup','0'], ['mmseqs','search',folder/'query_db',target,folder/'result',folder/'tmp']+params, ['mmseqs','convertalis',folder/'query_db',target,folder/'result',out,'--format-output','query,target,fident,qcov,tcov,evalue','--threads',str(threads)]]
        else:
            commands=[['mmseqs','easy-search',fa,target,out,folder/'tmp']+params+['--dbtype',dbtype,'--format-output','query,target,fident,qcov,tcov,evalue']]
        started=time.time()
        for i,c in enumerate(commands):
            with (folder/f'command-{i}.log').open('w') as log:
                log.write(json.dumps(list(map(str,c)))+'\n');log.flush()
                subprocess.run(['/usr/bin/time','-v','-o',str(folder/f'command-{i}.rss.txt')]+list(map(str,c)),cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        lines=out.read_text().splitlines();qs=sorted({x.split('\t')[0] for x in lines})
        result={'name':name,'fasta':str(fa),'mode':mode,'dbtype':dbtype,'threads':threads,'maxseq':maxseq,'seconds':time.time()-started,'alignments':len(lines),'queries':len(qs),'query_set_sha256':hashlib.sha256('\n'.join(qs).encode()).hexdigest(),'sorted_relation_sha256':hashlib.sha256('\n'.join(sorted(lines)).encode()).hexdigest(),'raw_sha256':sha(out)}
        (folder/'COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        return result
    if cap:
        results=[]
        for threads in (1,16):
            for maxseq in (1000,2000,5000,53615):
                for rep in range(2):results.append(run(f'cap-offender-t{threads}-m{maxseq}-r{rep}',DIAG/'cap-offender.fasta',dbtype='1',threads=threads,maxseq=maxseq))
        for rep in range(2):results.append(run(f'cap-offender-original8-r{rep}',DIAG/'cap-offender.fasta',threads=8))
        save('CAP_REPRODUCTIONS.json',results)
        return
    if fullcap:
        cases=[]
        def relation(name):return (DIAG/'runs'/name/'matches.tsv').read_text().splitlines()
        def nearest(lines):
            result={}
            for line in lines:
                q,t,i,*_=line.split('\t');i=float(i)
                if q not in result or i>result[q][1]:result[q]=(t,i)
            return result
        for batch in (37,117,219):
            r=run(f'fullcap-validation-batch{batch}',DIAG/f'validation-batch{batch}.fasta',dbtype='1',maxseq=53615)
            with gzip.open(SCI/f'manifests/screen-batch-{batch:04d}.tsv.gz','rt') as f:reference=[x.rstrip() for x in f]
            actual=relation(r['name']);assert set(reference)<=set(actual)
            r['new_hit_queries']=sorted(set(nearest(actual))-set(nearest(reference)));r['known_positives_retained']=True;cases.append(r)
        for rep in range(2):
            r=run(f'fullcap-offenders-audit-{rep}',DIAG/'offenders.fasta','audit',dbtype='1',maxseq=53615)
            baseline=relation('validation-offenders-audit-highcap');assert sorted(relation(r['name']))==sorted(baseline);cases.append(r)
            r=run(f'fullcap-equivalence6020-main-{rep}',ROOT/'screen_optimization/equivalence.fasta',dbtype='1',maxseq=53615)
            baseline=relation('validation-equivalence6020-audit-highcap');assert sorted(relation(r['name']))==sorted(baseline);cases.append(r)
        for fa,label in (('original-batch241.fasta','original'),('reversed-batch241.fasta','reversed')):
            r=run('fullcap-batch241-'+label,DIAG/fa,dbtype='1',maxseq=53615);cases.append(r)
        assert sorted(relation('fullcap-batch241-original'))==sorted(relation('fullcap-batch241-reversed'))
        probes=[]
        for folder in sorted((DIAG/'prefix-db-types').glob('batch-*')):
            assert (folder/'db.dbtype').read_bytes()==b'\0\0\0\0';probes.append(str(folder))
        assert len(probes)==32
        capresults=json.loads((DIAG/'CAP_REPRODUCTIONS.json').read_text())
        full=[x for x in capresults if x['maxseq']==53615];old=[x for x in capresults if x['maxseq']==1000]
        assert all(x['queries']==1 for x in full) and all(x['queries']==0 for x in old)
        assert len({x['sorted_relation_sha256'] for x in full})==1
        save('CORRECTED_VALIDATION.json',{'cases':cases,'passed':True,'root_cause_class':'B','secondary_query_alphabet_defect':True,'production_max_seqs':53615,'independent_audit_max_seqs':53615,'required_rescreen_population':24085944,'prefix_type_probes':probes,'old_expected_negative_proven_false':'s_43683a7be2017cb2fc9de98bd89dc4529d846ac1259257a0d5c34c1476936f20_6559485279'})
        return
    if validation:
        results=[]
        for fa in sorted((DIAG/'prefix-first-ten').glob('*.fasta')):
            folder=DIAG/'prefix-db-types'/fa.stem;folder.mkdir(parents=True,exist_ok=True)
            prefix=folder/'db'
            if not Path(str(prefix)+'.dbtype').exists():
                with (folder/'createdb.log').open('w') as log:
                    subprocess.run(['mmseqs','createdb',str(fa),str(prefix),'--dbtype','0','--shuffle','1','--write-lookup','0'],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            assert Path(str(prefix)+'.dbtype').read_bytes()==b'\0\0\0\0',fa
            results.append({'batch':fa.stem,'dbtype_bytes':0,'query_sha256':sha(fa)})
        assert len(results)==32
        for batch in (37,117,219):
            r=run(f'validation-batch{batch}-aa',DIAG/f'validation-batch{batch}.fasta',dbtype='1')
            expected=(SCI/f'manifests/screen-batch-{batch:04d}.tsv.gz')
            with gzip.open(expected,'rt') as f:reference=sorted(line.rstrip() for line in f)
            actual=sorted((DIAG/'runs'/r['name']/'matches.tsv').read_text().splitlines())
            assert actual==reference, f'Corrected full batch relation differs: {batch}'
            results.append(r)
        for rep in range(2):
            r=run(f'validation-equivalence6020-aa-{rep}',ROOT/'screen_optimization/equivalence.fasta',dbtype='1')
            reference=json.loads((ROOT/'screen_optimization/equivalence.expected-original-order.json').read_text())
            assert sorted((DIAG/'runs'/r['name']/'matches.tsv').read_text().splitlines())==sorted(line.rstrip() for line in reference)
            results.append(r)
        r=run('validation-offenders-audit-highcap',DIAG/'offenders.fasta','audit',dbtype='1',maxseq=53615)
        reference=(SCI/'mmseqs/verify_protected/batch-0200/matches.tsv').read_text().splitlines()
        def nearest(lines):
            result={}
            for line in lines:
                q,t,i,*_=line.split('\t');i=float(i)
                if q not in result or i>result[q][1]:result[q]=(t,i)
            return result
        actual=(DIAG/'runs'/r['name']/'matches.tsv').read_text().splitlines()
        assert set(reference)<=set(actual) and nearest(actual)==nearest(reference)
        r['added_weaker_alignments']=len(set(actual)-set(reference));r['exclusion_and_nearest_exact']=True
        results.append(r)
        r=run('validation-equivalence6020-audit-highcap',ROOT/'screen_optimization/equivalence.fasta','audit',dbtype='1',maxseq=53615)
        reference=json.loads((ROOT/'screen_optimization/equivalence.expected-original-order.json').read_text())
        actual=(DIAG/'runs'/r['name']/'matches.tsv').read_text().splitlines();reference=[line.rstrip() for line in reference]
        assert set(reference)<=set(actual) and nearest(actual)==nearest(reference)
        r['added_weaker_alignments']=len(set(actual)-set(reference));r['exclusion_and_nearest_exact']=True
        results.append(r)
        save('CORRECTED_VALIDATION.json',{'cases':results,'passed':True})
        return
    results=[]
    for rep in range(2):
        for mode in ('main','audit'):
            results.append(run(f'single-{mode}-{rep}',DIAG/'single-offender.fasta',mode))
            results.append(run(f'panel-{mode}-{rep}',DIAG/'panel.fasta',mode))
    results.append(run('original241-auto',DIAG/'original-batch241.fasta'))
    results.append(run('original241-auto-repeat',DIAG/'original-batch241.fasta'))
    results.append(run('reversed241-auto',DIAG/'reversed-batch241.fasta'))
    results.append(run('original241-amino',DIAG/'original-batch241.fasta',dbtype='1'))
    results.append(run('reversed241-amino',DIAG/'reversed-batch241.fasta',dbtype='1'))
    results.append(run('audit200-auto',SCI/'mmseqs/verify_protected/batch-0200/query.fasta','audit'))
    for rep in range(2):
        for mode in ('main','audit'):
            results.append(run(f'offenders-{mode}-{rep}',DIAG/'offenders.fasta',mode,dbtype='1'))
    for threads in (1,16):
        for maxseq in (1000,2000,5000,53615):
            for rep in range(2):
                results.append(run(f'panel-aa-t{threads}-m{maxseq}-r{rep}',DIAG/'panel.fasta',dbtype='1',threads=threads,maxseq=maxseq))
    results.append(run('reversed-panel-amino',DIAG/'reversed-panel.fasta',dbtype='1'))
    save('REPRODUCTIONS.json',results)
def scope():
    db=ro(SCI/'filtered/reservoir.sqlite');ex=ro(SCI/'manifests/exclusions.sqlite')
    protected={}
    with (SCI/'protected/protected.fasta').open() as f:
        for line in f:
            if line.startswith('>'):sid=line[1:].strip()
            elif line.strip():protected[line.strip()]=sid
    actual={(h,off):(t,i) for h,off,t,i in ex.execute('SELECT h,off,protected,identity FROM removed')}
    assert len(actual)==802706
    results=[];cursor=db.execute('SELECT h,off,seq,historical,external FROM candidates ORDER BY rowid')
    batch=0;firsts=DIAG/'prefix-first-queries';firsts.mkdir(exist_ok=True)
    while rows:=cursor.fetchmany(100000):
        batch+=1;certpath=SCI/f'manifests/screen-batch-{batch:04d}.json';ret=certpath.with_suffix('.tsv.gz')
        cert=json.loads(certpath.read_text());assert sha(ret)==cert['retained_result_sha256']
        ih=hashlib.sha256();fh=hashlib.sha256();mapping={};nearest={}
        for h,off,seq,*_ in rows:
            sid=f's_{h.hex()}_{off}';ih.update(h+str(off).encode()+b'\0');fh.update(f'>{sid}\n{seq}\n'.replace('\n','\r\n' if os.name=='nt' else '\n').encode());mapping[sid]=(h,off)
            if seq in protected:nearest[sid]=(protected[seq],1.0)
        assert ih.hexdigest()==cert['input_rows_sha256'];assert fh.hexdigest()==cert['query_sha256']
        rawhash=hashlib.sha256();lines=0
        with gzip.open(ret,'rb') as f:
            for raw in f:
                rawhash.update(raw);q,t,i,qc,tc,e=raw.decode().strip().split('\t');i,qc,tc=map(float,(i,qc,tc));assert q in mapping and i>=.299 and qc>=.799 and tc>=.799
                if q not in nearest or i>nearest[q][1]:nearest[q]=(t,i)
                lines+=1
        assert rawhash.hexdigest()==cert['matches_sha256'];assert len(nearest)==cert['hits']
        expected={mapping[q]:hit for q,hit in nearest.items()}
        present={k:actual[k] for k in mapping.values() if k in actual};assert expected==present
        if batch<=32:
            h,off,seq,*_=rows[0];(firsts/f'batch-{batch:04d}.fasta').write_text(f'>s_{h.hex()}_{off}\n{seq}\n')
        results.append({'batch':batch,'n':len(rows),'raw_alignments':lines,'exclusions':len(nearest),'original_query_and_rows_reconstructed_exactly':True,'archived_relation_and_sqlite_exact':True,'certificate_sha256':sha(certpath),'retained_sha256':sha(ret)})
        if batch%20==0:print(json.dumps({'scope_batch':batch}),flush=True)
    assert batch==241;save('SCOPE_VALIDATION.json',{'batches':results,'parser_bookkeeping_exact':True,'population':sum(x['n'] for x in results),'historical_exclusions_preserved':len(actual)})
    offenders=json.loads((DIAG/'OFFENDERS.json').read_text());bykey={(bytes.fromhex(x['hash']),x['offset']):x for x in offenders}
    # Covering priority index: sequences are not fetched again for rank metadata.
    original_rank=clean_rank=0
    for h,off in db.execute('SELECT h,off FROM candidates ORDER BY p,h,off'):
        original_rank+=1
        if (h,off) not in actual:clean_rank+=1
        if (h,off) in bykey:
            x=bykey[(h,off)];x['original_priority_rank']=original_rank;x['original_final_selection_rank']=clean_rank
    assert original_rank==24085944 and clean_rank==23283238
    for x in offenders:
        assert not x['excluded'] and x['main_batch']==241 and x['original_final_selection_rank']<=20000000
        x.update(original_query_membership='proven by exact reconstructed FASTA hash',original_result_contains_query=False,original_raw_result='empty',original_clean_survivor=True,independent_per_batch_survivor_search='not performed: frozen single-search complement architecture',classification='database/index discrepancy: nucleotide query database')
    save('OFFENDERS_TRACED.json',offenders)
def validation_prepare():
    db=ro(SCI/'filtered/reservoir.sqlite')
    firsts=DIAG/'prefix-first-ten';firsts.mkdir(exist_ok=True)
    for batch in range(1,33):
        rows=db.execute('SELECT h,off,seq FROM candidates WHERE rowid>? AND rowid<=? ORDER BY rowid',((batch-1)*100000,(batch-1)*100000+10)).fetchall()
        fa=firsts/f'batch-{batch:04d}.fasta'
        if not fa.exists():fa.write_text(''.join(f'>s_{h.hex()}_{off}\n{seq}\n' for h,off,seq in rows))
    for batch in (37,117,219):
        rows=db.execute('SELECT h,off,seq FROM candidates WHERE rowid>? AND rowid<=? ORDER BY rowid',((batch-1)*100000,batch*100000)).fetchall()
        fa=DIAG/f'validation-batch{batch}.fasta'
        if not fa.exists():fa.write_text(''.join(f'>s_{h.hex()}_{off}\n{seq}\n' for h,off,seq in rows))
        assert sha(fa)==json.loads((SCI/f'manifests/screen-batch-{batch:04d}.json').read_text())['query_sha256']
def validate():reproduce(validation=True)
def cap_prepare():
    sid='s_43683a7be2017cb2fc9de98bd89dc4529d846ac1259257a0d5c34c1476936f20_6559485279'
    db=ro(SCI/'filtered/reservoir.sqlite');h=bytes.fromhex(sid.split('_')[1]);off=int(sid.split('_')[2])
    row=db.execute('SELECT rowid,seq,source_id,historical,external FROM candidates WHERE h=? AND off=?',(h,off)).fetchone();assert row
    batch=(row[0]-1)//100000+1
    with gzip.open(SCI/f'manifests/screen-batch-{batch:04d}.tsv.gz','rt') as f:lines=[line.rstrip() for line in f if line.split('\t')[0]==sid]
    ex=ro(SCI/'manifests/exclusions.sqlite');removed=ex.execute('SELECT protected,identity FROM removed WHERE h=? AND off=?',(h,off)).fetchone()
    save('CAP_OFFENDER.json',{'query':sid,'rowid':row[0],'main_batch':batch,'sequence':row[1],'length':len(row[1]),'source_id':row[2],'historical':row[3],'external':row[4],'original_main_alignments':lines,'original_exclusion':removed})
    (DIAG/'cap-offender.fasta').write_text(f'>{sid}\n{row[1]}\n')
def cap_reproduce():reproduce(cap=True)
def validate_fullcap():reproduce(fullcap=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['preserve','inspect','prepare','reproduce','scope','validation_prepare','validate','cap_prepare','cap_reproduce','validate_fullcap']);args=a.parse_args()
    globals()[args.action]()
