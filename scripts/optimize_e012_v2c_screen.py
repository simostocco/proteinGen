"""Profile and optimize the frozen V2C screen; all runtime artifacts stay on D:."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import time
import shutil
import subprocess
import gzip
import signal
import hashlib
import re

def time_metrics(path):
    text=path.read_text()
    def value(label):
        match=re.search(re.escape(label)+r':\s*([^\n]+)',text)
        return match[1].strip() if match else None
    elapsed=value('Elapsed (wall clock) time (h:mm:ss or m:ss)')
    parts=list(map(float,elapsed.split(':'))) if elapsed else []
    wall=sum(v*60**i for i,v in enumerate(reversed(parts))) if parts else None
    def numeric(label,convert):
        found=value(label)
        return convert(found) if found is not None else None
    return {'wall_seconds':wall,'user_seconds':numeric('User time (seconds)',float),
            'system_seconds':numeric('System time (seconds)',float),'cpu_percent':value('Percent of CPU this job got'),
            'peak_rss_kib':numeric('Maximum resident set size (kbytes)',int),
            'filesystem_input_blocks':numeric('File system inputs',int),
            'filesystem_output_blocks':numeric('File system outputs',int)}

def report():
    out=ROOT/'screen_optimization'
    pause_record=json.loads((ROOT/'reports/SCREEN_OPTIMIZATION_PAUSE.json').read_text())
    profile=out/'profile/batch-0032'
    original={label:time_metrics(profile/name) for label,name in [('search','command.rss.txt'),('survivors','verify_command.rss.txt')]}
    original_wall=sum(v['wall_seconds'] for v in original.values())
    sampled={}
    for line in (profile/'samples.jsonl').read_text().splitlines():
        for p in json.loads(line)['processes']:
            if 'mmseqs-avx2' not in p['command'][0]:continue
            key=str(p['pid']);record=sampled.setdefault(key,{'subcommand':p['command'][1],'max_observed_threads':0,'max_rss_bytes':0,'max_io_counters':{}})
            record['max_observed_threads']=max(record['max_observed_threads'],p['threads'])
            record['max_rss_bytes']=max(record['max_rss_bytes'],p['rss'])
            for k,v in p['io'].items():record['max_io_counters'][k]=max(record['max_io_counters'].get(k,0),v)
    measures={p.parent.name:time_metrics(p) for p in out.glob('*/time.txt')}
    equivalence=json.loads((out/'equivalence.json').read_text())
    probe=json.loads((out/'probe.json').read_text()) if (out/'probe.json').exists() else None
    optimized_labels=['probe-createdb','probe-16','probe-16-convert','probe-survivors-16']
    optimized_wall=sum(measures[k]['wall_seconds'] for k in optimized_labels) if probe else None
    original_rate=100000/original_wall
    optimized_rate=50000/optimized_wall if optimized_wall else None
    speedup=optimized_rate/original_rate if optimized_rate else None
    thread_gain=measures['probe-8']['wall_seconds']/measures['probe-16']['wall_seconds'] if probe else None
    assert speedup is None or speedup<5, 'Speed target passed: investigate production promotion instead of stopping'
    batches=[]
    for batch in [28,29,30,31,32]:
        cert=json.loads((ROOT/f'scientific/manifests/screen-batch-{batch:04d}.json').read_text())
        batches.append({'batch':batch,'n':cert['n'],'hits':cert['hits'],'timestamp':cert['timestamp'],
                        'command':cert['command'],'detailed_logs_available':batch==32})
    with sqlite3.connect('file:'+(ROOT/'scientific/manifests/exclusions.sqlite').as_posix()+'?mode=ro',uri=True) as db:
        highest,n=db.execute("SELECT max(batch),sum(n) FROM batches WHERE name='screen'").fetchone()
        hits=db.execute('SELECT count(*) FROM removed').fetchone()[0]
        assert (highest,n,hits)==(pause_record['batch'],pause_record['screened'],pause_record['hits'])
        assert db.execute('PRAGMA quick_check').fetchall()==[('ok',)]
        certified_hits=0
        for batch,count in db.execute("SELECT batch,n FROM batches WHERE name='screen' ORDER BY batch"):
            cert=json.loads((ROOT/f'scientific/manifests/screen-batch-{batch:04d}.json').read_text())
            assert cert['n']==count and cert['batch_zero_overlap_verified']
            retained=ROOT/f'scientific/manifests/screen-batch-{batch:04d}.tsv.gz'
            digest=hashlib.sha256()
            with retained.open('rb') as f:
                while block:=f.read(8*1024**2):digest.update(block)
            assert digest.hexdigest()==cert['retained_result_sha256']
            certified_hits+=cert['hits']
        assert certified_hits==hits
    result={'status':'STOPPED_NO_5X_SPEEDUP','production_resumed':False,'training_launched':False,
        'pause':pause_record,'original_metrics':original,'sampled_process_io':sampled,'benchmark_metrics':measures,'recent_batches':batches,
        'equivalence':equivalence,'probe':probe,'original_candidates_s':original_rate,'optimized_candidates_s':optimized_rate,
        'speedup_vs_measured_original':speedup,'remaining_candidates':24000000-n,
        'same_input_first_search_16_vs_8_speedup':thread_gain,'fastest_completed_thread_probe':16 if probe else None,
        'projected_remaining_screen_hours':(24000000-n)/optimized_rate/3600 if optimized_rate else None,
        'scratch_snapshot_bytes':558390060,'scratch_snapshot_is_peak':False,
        'diagnosis':'CPU-bound sensitive prefilter and alignment, repeated survivor search; reference index already reused; query creation is <1% of search time.',
        'bulk_tests':'500k/1M/2M bounded at 180 seconds; not complete end-to-end timings.',
        'limitations':['Completed batches 1-31 GNU-time/logs deleted by preexisting cleanup; retained hit manifests preserved.',
                      'DrvFS GNU-time filesystem blocks are not authoritative physical D: bytes read/written.',
                      'Bounded search termination also stopped the GNU-time wrapper; its empty time files are unavailable metrics, not zero CPU/RSS. Completed probes have full GNU-time evidence.',
                      'Tiny completed replays plus thread and interrupted-MMseqs-restart equivalence; production scheduler migration not exercised because speed gate failed.',
                      '50k probe includes MMseqs database/search/conversion/survivor verification; excludes production SQLite commit and FASTA-export cost.'],
        'd_free_bytes':shutil.disk_usage(out).free}
    (out/'ANALYSIS.json').write_text(json.dumps(result,indent=2))
    rows=['| Search | Candidates | Wall s | User s | System s | CPU | RSS GiB |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for label,count in [('search',100000),('survivors',95326)]:
        v=original[label];rows.append(f"| Batch 32 {label} | {count:,} | {v['wall_seconds']:.2f} | {v['user_seconds']:.2f} | {v['system_seconds']:.2f} | {v['cpu_percent']} | {v['peak_rss_kib']/1024**2:.2f} |")
    text='V2C screening performance investigation\n\nProduction is paused at committed batch 32, 3,200,000 screened and 178,888 exclusions. All completed certificates and retained results remain on D:. Source, historical union and reservoir finalization were not rerun.\n\n'+'\n'.join(rows)+'\n\n'
    text+='The protected 53,615-sequence database/index is reused read-only. Each 100k query batch and its 95k reversed survivor FASTA independently create a query database; no candidate index is built. Both searches use eight OpenMP threads, AVX2, identity 0.30, bidirectional coverage 0.80 (mode 0), sensitivity 7.5, seq-id mode 0, alignment mode 3, max-seqs 1000 and the original default E-value 0.001.\n\n'
    text+='Batch 32: first-pass createdb 3.476 s, prefilter 348.902 s, alignment 61.894 s, convertalis 0.471 s. Survivor pass: createdb 3.097 s, prefilter 334.880 s, alignment 58.086 s, conversion 0.376 s. Startup/database construction is below 1%; eliminating it cannot deliver 5x. CPU use is about 780%, system CPU below 4 seconds per search. Temporary volume snapshot was 558,390,060 bytes; a true peak was not captured. GNU-time input blocks: first pass 2,849,088, survivor pass 2,795,600; output blocks reported zero, which must not be interpreted as no D: writes.\n\n'
    text+='Previous cleanup deleted detailed logs/time files for completed batches 1–31. Their commands, sizes, exclusions and checksummed retained hits survive. No missing telemetry has been reconstructed as measured data.\n\n'
    text+='Coverage mode 0 and aligned-column identity are symmetric thresholds, but reversing orientation is rejected: the per-query top-1000 prefilter cap would become a cap per protected sequence rather than per candidate; target masking, prefilter scoring and database-dependent E-values also require equivalence. A persistent original-direction query database was tested with frozen parameters and 4/8/16 threads. See equivalence.json for exact full-hit, excluded-ID, nearest-hit and interrupted/restarted-search checks on 6,020 candidates from batches 28, 30, 31 with 1,767 known hit pairs.\n\n'
    if speedup:
        text+=f'The completed 50k probe, including the survivor check, achieved {optimized_rate:.2f} candidates/s versus {original_rate:.2f} candidates/s in measured batch 32 ({speedup:.2f}x). It projects {(24000000-n)/optimized_rate/3600:.1f} hours for the remaining screen alone, before final independent verification. FASTA export and SQL commit would add time.\n\n'
        text+=f'On identical 50k inputs, the first search at 16 threads was {thread_gain:.2f}x faster than at eight threads. Sixteen was the fastest completed thread probe; a completed 500k/1M/2M batch-size winner was not established.\n\n'
    text+='500k/1M/2M original-direction persistent-database searches were bounded at 180 seconds each. These are scale probes, not completed end-to-end benchmarks; their partial artifacts and logs are preserved. No 5x end-to-end gain was demonstrated, so another multi-day screen was not launched. Verify-protected/final/certify/diversity/handoff remain pending.\n\nPrimary MMseqs documentation: https://github.com/soedinglab/MMseqs2/wiki\n'
    (out/'ANALYSIS.md').write_text(text)
    supervisor_status=ROOT/'reports/RECOVERY_SUPERVISOR_STATUS.json'
    preserved=out/'prior_supervisor_status.json'
    if not preserved.exists():shutil.copyfile(supervisor_status,preserved)
    supervisor_status.write_text(json.dumps({'phase':'screen_performance_paused','screen':pause_record,
        'reason':'Frozen equivalence passed on tiny replay; 5x screening speed target not demonstrated.',
        'analysis':str(out/'ANALYSIS.json'),'training_launched':False,'timestamp_unix':time.time()},indent=2))
    print(json.dumps({k:result[k] for k in ['status','original_candidates_s','optimized_candidates_s','speedup_vs_measured_original','projected_remaining_screen_hours']}),flush=True)

def status():
    import psutil
    record=json.loads((ROOT/'reports/SCREEN_OPTIMIZATION_PAUSE.json').read_text())
    workers=[]
    for p in psutil.process_iter(['pid','cmdline']):
        argv=p.info['cmdline'] or []
        if argv and argv[0].endswith('/mmseqs-avx2') and any(str(ROOT/'screen_optimization') in arg for arg in argv):
            try:
                workers.append({'pid':p.pid,'subcommand':argv[1],'cpu_percent':p.cpu_percent(interval=1),'rss_bytes':p.memory_info().rss,'threads':p.num_threads()})
            except psutil.NoSuchProcess:pass
    pulse={'screened':record['screened'],'total':24000000,'percent':100*record['screened']/24000000,
           'hits':record['hits'],'production_paused':True,'workers':workers,'d_free_bytes':shutil.disk_usage(ROOT).free}
    with (ROOT/'screen_optimization/host_status.jsonl').open('a') as f:
        f.write(json.dumps({'time':time.time(),**pulse})+'\n')
    print(json.dumps(pulse),flush=True)

POLICY = ['--min-seq-id', '0.30', '-c', '0.80', '--cov-mode', '0', '-s', '7.5',
          '--seq-id-mode', '0', '--alignment-mode', '3', '--split-memory-limit', '4G',
          '--max-seqs', '1000', '--remove-tmp-files', '0']
FORMAT = 'query,target,fident,qcov,tcov,evalue'

def prepare():
    """Native sequential reads, with tiny committed replays and new uncommitted inputs."""
    assert os.name == 'nt'
    while not (ROOT / 'reports/SCREEN_OPTIMIZATION_PAUSE.json').exists():
        time.sleep(1)
    out = ROOT / 'screen_optimization'
    out.mkdir(exist_ok=True)
    pause_record = json.loads((ROOT / 'reports/SCREEN_OPTIMIZATION_PAUSE.json').read_text())
    manifests = ROOT / 'scientific/manifests'
    chosen = [28, 30, 31]
    wanted = set()
    expected = []
    for batch in chosen:
        with gzip.open(manifests / f'screen-batch-{batch:04d}.tsv.gz', 'rt') as f:
            lines = list(f)
        # Include known-hit queries explicitly, plus a contiguous sample.
        wanted.update(line.split('\t')[0] for line in lines[:30])
        with sqlite3.connect('file:' + (ROOT / 'scientific/filtered/reservoir.sqlite').as_posix() + '?mode=ro', uri=True) as db:
            rows = db.execute('SELECT h,off,seq FROM candidates WHERE rowid>? AND rowid<=? ORDER BY rowid',
                              ((batch-1)*100000, (batch-1)*100000+2000)).fetchall()
        wanted.update(f's_{h.hex()}_{off}' for h, off, seq in rows)
        expected.extend(line.rstrip() for line in lines if line.split('\t')[0] in wanted)
    native_tmp = out / 'tmp'; native_tmp.mkdir(exist_ok=True)
    for key in ['TMP', 'TEMP', 'TMPDIR']: os.environ[key] = str(native_tmp)
    with sqlite3.connect('file:' + (ROOT / 'scientific/filtered/reservoir.sqlite').as_posix() + '?mode=ro', uri=True) as db:
        with (out / 'equivalence.fasta').open('w') as f:
            for h, off, seq in db.execute('SELECT h,off,seq FROM candidates WHERE rowid<=? ORDER BY rowid', (pause_record['screened'],)):
                sid = f's_{h.hex()}_{off}'
                if sid in wanted: f.write(f'>{sid}\n{seq}\n')
        for count in [500000, 1000000, 2000000]:
            with (out / f'new-{count}.fasta').open('w') as f:
                for h, off, seq in db.execute('SELECT h,off,seq FROM candidates WHERE rowid>? ORDER BY rowid LIMIT ?', (pause_record['screened'], count)):
                    f.write(f'>s_{h.hex()}_{off}\n{seq}\n')
    (out / 'equivalence.expected.json').write_text(json.dumps(expected))
    (out / 'preparation.json').write_text(json.dumps({'committed_replay_ids': len(wanted), 'batches': chosen,
        'known_hit_pairs': len(expected), 'remaining_start': pause_record['screened'], 'new_sizes': [500000,1000000,2000000]}, indent=2))
    print((out / 'preparation.json').read_text(), flush=True)

def benchmark():
    assert os.name != 'nt'
    out = ROOT / 'screen_optimization'
    tmp = out / 'tmp'; tmp.mkdir(exist_ok=True)
    for key in ['TMP', 'TEMP', 'TMPDIR', 'XDG_CACHE_HOME']: os.environ[key] = str(tmp)
    os.chdir(out)
    target = ROOT.parent / 'mmseqs/protected_index/db'
    results = json.loads((out/'benchmarks.json').read_text()) if (out/'benchmarks.json').exists() else []
    pause_record=json.loads((ROOT/'reports/SCREEN_OPTIMIZATION_PAUSE.json').read_text())
    floor=json.loads((ROOT/'scientific/stats/resource_plan.json').read_text())['starting_free_bytes']*3//10
    def run(label, command, timeout=None):
        existing=next((r for r in results if r['label']==label),None)
        if existing is not None:
            print(json.dumps({'reuse_completed_benchmark':label,'returncode':existing['returncode']}),flush=True)
            return existing['returncode']
        folder = out / label; folder.mkdir(exist_ok=True)
        start = time.monotonic()
        with (folder / 'command.log').open('w') as log:
            log.write(json.dumps(list(map(str, command)))+'\n'); log.flush()
            p = subprocess.Popen(['/usr/bin/time','-v','-o',str(folder/'time.txt'), *map(str,command)],
                                 stdout=log, stderr=log, start_new_session=True)
            last=start
            while p.poll() is None:
                elapsed=time.monotonic()-start
                free=shutil.disk_usage(out).free
                if free<floor or (timeout is not None and elapsed>=timeout):
                    os.killpg(p.pid, signal.SIGTERM);p.wait()
                    code='disk_reserve_stop' if free<floor else 'bounded_timeout';break
                if time.monotonic()-last>=60:
                    print(json.dumps({'phase':'benchmark','job':label,'job_seconds':elapsed,
                        'screened':pause_record['screened'],'total':24000000,'percent':100*pause_record['screened']/24000000,
                        'hits':pause_record['hits'],'production_paused':True,'d_free_bytes':free}),flush=True)
                    last=time.monotonic()
                time.sleep(1)
            else: code=p.returncode
        result = {'label': label, 'seconds': time.monotonic()-start, 'returncode': code}
        results.append(result); (out/'benchmarks.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(result), flush=True)
        return code
    db = out/'equivalence-db'
    assert run('create-equivalence-db',['mmseqs','createdb',out/'equivalence.fasta',db]) == 0
    query_ids=set()
    with (out/'equivalence.fasta').open() as f:
        for line in f:
            if line.startswith('>'):query_ids.add(line[1:].strip())
    expected=[]
    for batch in [28,30,31]:
        with gzip.open(ROOT/f'scientific/manifests/screen-batch-{batch:04d}.tsv.gz','rt') as f:
            expected.extend(line.rstrip() for line in f if line.split('\t')[0] in query_ids)
    (out/'equivalence.expected-original-order.json').write_text(json.dumps(expected))
    protected={}
    with (ROOT/'scientific/protected/protected.fasta').open() as f:
        for line in f:
            if line.startswith('>'): sid=line[1:].strip()
            elif line.strip(): protected[line.strip()]=sid
    exact={}
    with (out/'equivalence.fasta').open() as f:
        while header:=f.readline():
            seq=f.readline().strip()
            if seq in protected: exact[header[1:].strip()]=(protected[seq],1.0)
    def decisions(lines):
        nearest=dict(exact)
        for line in lines:
            query,target,ident,qcov,tcov,evalue=line.split('\t')
            ident=float(ident)
            assert ident>=.299 and float(qcov)>=.799 and float(tcov)>=.799
            if query not in nearest or ident>nearest[query][1]: nearest[query]=(target,ident)
        return nearest
    gates=[]
    for threads in [4,8,16]:
        label=f'equivalence-{threads}'
        folder=out/label; folder.mkdir(exist_ok=True)
        assert run(label,['mmseqs','search',db,target,folder/'result',folder/'tmp',*POLICY,'--threads',threads]) == 0
        assert run(label+'-convert',['mmseqs','convertalis',db,target,folder/'result',folder/'hits.tsv','--format-output',FORMAT,'--threads',threads]) == 0
        actual=(folder/'hits.tsv').read_text().splitlines()
        gates.append({'threads': threads, 'exact_all_hit_fields':set(actual)==set(expected),
                      'exact_excluded_ids_and_nearest_decisions':decisions(actual)==decisions(expected),
                      'expected_pairs':len(expected),'actual_pairs':len(actual),
                      'missing':len(set(expected)-set(actual)), 'extra':len(set(actual)-set(expected))})
    # A second invocation must reuse completed MMseqs steps and return identical output.
    folder=out/'equivalence-8'
    # MMseqs refuses an already published result DB; the scheduler must reuse
    # its verified completion instead of relaunching it. Preserve this evidence.
    assert run('restart-equivalence-8',['mmseqs','search',db,target,folder/'result',folder/'tmp',*POLICY,'--threads',8]) == 1
    assert set((folder/'hits.tsv').read_text().splitlines())==set(expected)
    interrupted=out/'interrupted-restart';interrupted.mkdir(exist_ok=True)
    restart_command=['mmseqs','search',db,target,interrupted/'result',interrupted/'tmp',*POLICY,'--threads',8]
    assert run('interrupt-equivalence',restart_command,timeout=2)=='bounded_timeout'
    assert run('resume-interrupted-equivalence',restart_command)==0
    assert run('resume-interrupted-convert',['mmseqs','convertalis',db,target,interrupted/'result',interrupted/'hits.tsv','--format-output',FORMAT,'--threads',8])==0
    restarted=(interrupted/'hits.tsv').read_text().splitlines()
    assert set(restarted)==set(expected) and decisions(restarted)==decisions(expected)
    (out/'equivalence.json').write_text(json.dumps({'gates':gates, 'restart_search_passed':True,'interrupted_restart_exact':True,
        'orientation_inversion_accepted':False, 'reason':'Symmetric coverage does not prove symmetric heuristic prefilter, per-query top-1000 cap, target masking or database-size E-values.'},indent=2))
    if not all(g['exact_all_hit_fields'] and g['exact_excluded_ids_and_nearest_decisions'] for g in gates):
        print('Equivalence failed; production remains paused.',flush=True); return
    # Finish a moderately sized probe, so speed conclusions do not rely on
    # an unfinished prefilter's progress bar or its initialization overhead.
    probe=out/'probe-50000.fasta'
    with (out/'new-500000.fasta').open() as src, probe.open('w') as dst:
        for _ in range(50000): dst.write(next(src)); dst.write(next(src))
    probe_db=out/'probe-db'
    assert run('probe-createdb',['mmseqs','createdb',probe,probe_db]) == 0
    for threads in [8,16]:
        label=f'probe-{threads}'; folder=out/label;folder.mkdir(exist_ok=True)
        assert run(label,['mmseqs','search',probe_db,target,folder/'result',folder/'tmp',*POLICY,'--threads',threads]) == 0
        assert run(label+'-convert',['mmseqs','convertalis',probe_db,target,folder/'result',folder/'hits.tsv','--format-output',FORMAT,'--threads',threads]) == 0
    baseline=(out/'probe-8/hits.tsv').read_text().splitlines()
    optimized=(out/'probe-16/hits.tsv').read_text().splitlines()
    assert set(baseline)==set(optimized), 'Tail probe thread equivalence failed'
    removed={line.split('\t')[0] for line in optimized}
    clean=[]
    with probe.open() as src:
        while header:=src.readline():
            seq=src.readline()
            if header[1:].strip() not in removed: clean.append((header,seq))
    clean_fasta=out/'probe-clean.fasta'
    with clean_fasta.open('w') as dst:
        for header,seq in reversed(clean): dst.write(header); dst.write(seq)
    assert run('probe-survivors-16',['mmseqs','easy-search',clean_fasta,target,out/'probe-clean.tsv',out/'probe-clean-tmp',*POLICY,'--threads',16,'--format-output',FORMAT]) == 0
    assert (out/'probe-clean.tsv').stat().st_size == 0
    (out/'probe.json').write_text(json.dumps({'n':50000,'survivors':len(clean),'excluded':len(removed),'exact_thread_equivalence':True,'zero_overlap_verified':True},indent=2))
    for count in [500000,1000000,2000000]:
        label=f'bulk-{count}';folder=out/label;folder.mkdir(exist_ok=True)
        assert run(label+'-createdb',['mmseqs','createdb',out/f'new-{count}.fasta',folder/'query']) == 0
        # Bounded diagnostic: do not spend hours on a compute-bound losing alternative.
        run(label,['mmseqs','search',folder/'query',target,folder/'result',folder/'tmp',*POLICY,'--threads',16],timeout=180)
    print('Bounded bulk benchmarks complete; production remains paused pending measured speed gate.',flush=True)

def capture():
    """Preserve live logs before the legacy post-commit cleanup removes them."""
    import psutil
    out = ROOT / 'screen_optimization/profile/batch-0032'
    out.mkdir(parents=True, exist_ok=True)
    folder = ROOT / 'scientific/mmseqs/screen/batch-0032'
    last = 0
    with (out / 'samples.jsonl').open('a') as samples:
        while not (ROOT / 'reports/SCREEN_OPTIMIZATION_PAUSE.json').exists():
            for name in ['command.log', 'command.rss.txt', 'verify_command.log', 'verify_command.rss.txt']:
                p = folder / name
                if p.exists():
                    try: shutil.copyfile(p, out / name)
                    except FileNotFoundError: pass
            if time.time() - last >= 5:
                processes = []
                for p in psutil.process_iter(['pid', 'name', 'cmdline']):
                    if p.info['name'].startswith('mmseqs') and any(str(folder) in a for a in p.info['cmdline']):
                        try:
                            processes.append({'pid': p.pid, 'command': p.info['cmdline'], 'cpu': p.cpu_times()._asdict(),
                                              'rss': p.memory_info().rss, 'io': p.io_counters()._asdict(), 'threads': p.num_threads()})
                        except psutil.NoSuchProcess: pass
                samples.write(json.dumps({'time': time.time(), 'processes': processes}) + '\n'); samples.flush()
                last = time.time()
            time.sleep(.1)
    print('Live batch telemetry preserved on D:', flush=True)

ROOT = Path('D:/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization') if os.name == 'nt' else Path('/mnt/d/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization')

def pause():
    import psutil
    assert os.name == 'nt'
    heartbeat=json.loads((ROOT/'reports/RECOVERY_HEARTBEAT.json').read_text())
    supervisor_pid=next(p['pid'] for p in heartbeat['workers'] if any(str(a).endswith('recover_e012_v2c.py') for a in p['argv']))
    worker_pid=next(p['pid'] for p in heartbeat['workers'] if any(str(a).endswith('complete_e012_v2c_pipeline.py') for a in p['argv']) and p['argv'][-1]=='screen')
    supervisor = psutil.Process(supervisor_pid)
    worker = psutil.Process(worker_pid)
    assert any(str(a).endswith('recover_e012_v2c.py') for a in supervisor.cmdline())
    assert worker.cmdline()[-1] == 'screen'
    if (ROOT/'reports/SCREEN_OPTIMIZATION_PAUSE.json').exists():
        raise RuntimeError('Pause evidence already exists; preserve and inspect it before another pause attempt')
    supervisor.suspend()
    checkpoint = ROOT / 'scientific/manifests/exclusions.sqlite'
    with sqlite3.connect('file:' + checkpoint.as_posix() + '?mode=ro', uri=True) as db:
        initial = db.execute("SELECT max(batch) FROM batches WHERE name='screen'").fetchone()[0]
        print(json.dumps({'waiting_for_commit': initial + 1, 'supervisor_suspended': supervisor.pid}), flush=True)
        while True:
            batch, count = db.execute("SELECT max(batch),sum(n) FROM batches WHERE name='screen'").fetchone()
            if batch > initial:
                worker.suspend()
                children = worker.children(recursive=True)
                for child in children:
                    try: child.suspend()
                    except psutil.NoSuchProcess: pass
                hits = db.execute('SELECT count(*) FROM removed').fetchone()[0]
                assert db.execute('PRAGMA quick_check').fetchall() == [('ok',)]
                for number, n, marker in db.execute("SELECT batch,n,marker FROM batches WHERE name='screen' ORDER BY batch"):
                    p = ROOT / 'scientific/manifests' / f'screen-batch-{number:04d}.json'
                    cert = json.loads(p.read_text())
                    assert cert['n'] == n and cert['batch_zero_overlap_verified']
                result = {'batch': batch, 'screened': count, 'hits': hits, 'worker_pid': worker.pid,
                          'supervisor_pid': supervisor.pid, 'suspended_children': [p.pid for p in children],
                          'committed_batches_preserved': True, 'time': time.time()}
                (ROOT / 'reports/SCREEN_OPTIMIZATION_PAUSE.json').write_text(json.dumps(result, indent=2))
                print(json.dumps(result), flush=True)
                return
            time.sleep(.05)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['pause', 'capture', 'prepare', 'benchmark', 'report', 'status'])
    args = parser.parse_args()
    {'pause': pause, 'capture': capture, 'prepare': prepare, 'benchmark': benchmark, 'report': report, 'status': status}[args.action]()
