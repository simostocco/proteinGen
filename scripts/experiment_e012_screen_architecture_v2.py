"""Isolated search-count/orientation experiment. Never resumes production.

Runtime files, SQLite experiments, logs, caches and MMseqs scratch stay on D:.
The first diagnostic and all production certificates are read-only inputs.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

ROOT = Path('D:/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization') if os.name == 'nt' else Path('/mnt/d/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization')
OUT = ROOT / 'screen_architecture_v2'
FIRST = ROOT / 'screen_optimization'
SCI = ROOT / 'scientific'
TARGET = ROOT.parent / 'mmseqs/protected_index/db'
BATCHES = (28, 30, 31)
FROZEN_COMMIT = '5a3cf728257aa94dc48ca36241909d00218e6188'
POLICY = ['--min-seq-id', '0.30', '-c', '0.80', '--cov-mode', '0', '-s', '7.5',
          '--seq-id-mode', '0', '--alignment-mode', '3', '--split-memory-limit', '4G',
          '--max-seqs', '1000', '--remove-tmp-files', '0']
FORMAT = 'query,target,fident,qcov,tcov,evalue'

def setup():
    assert OUT.resolve().drive.upper() == 'D:' if os.name == 'nt' else OUT.resolve().is_relative_to('/mnt/d')
    (OUT / 'tmp').mkdir(parents=True, exist_ok=True)
    for key in ('TMP', 'TEMP', 'TMPDIR', 'XDG_CACHE_HOME'):
        os.environ[key] = str(OUT / 'tmp')
    tempfile.tempdir = str(OUT / 'tmp')
    os.chdir(OUT)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    with partial.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.flush(); os.fsync(f.fileno())
    os.replace(partial, path)

def readonly(path):
    return sqlite3.connect('file:' + path.as_posix() + '?mode=ro', uri=True)

def production_state():
    with readonly(SCI / 'manifests/exclusions.sqlite') as db:
        batch, count = db.execute("SELECT max(batch),sum(n) FROM batches WHERE name='screen'").fetchone()
        removed = db.execute('SELECT count(*) FROM removed').fetchone()[0]
    assert (batch, count, removed) == (32, 3200000, 178888)
    return {'batch': batch, 'screened': count, 'exclusions': removed, 'production_resumed': False}

def audit():
    assert os.name != 'nt'
    production_state()
    commit = subprocess.check_output(['git', 'rev-parse', '5a3cf72'], cwd=Path(__file__).resolve().parent.parent, text=True).strip()
    assert commit == FROZEN_COMMIT
    inventory = {}
    start = last = time.monotonic()
    for p in sorted(FIRST.rglob('*')):
        if p.is_symlink():
            inventory[str(p.relative_to(FIRST))] = {'symlink': os.readlink(p)}
        elif p.is_file():
            inventory[str(p.relative_to(FIRST))] = {'bytes': p.stat().st_size, 'sha256': sha(p)}
        if time.monotonic() - last >= 30:
            print(json.dumps({'phase': 'seal_first_diagnostic', 'files': len(inventory), 'seconds': time.monotonic()-start}), flush=True)
            last = time.monotonic()
    save(OUT / 'FIRST_DIAGNOSTIC_FROZEN.json', {'commit': commit, 'files': inventory, 'read_only_inputs': True})
    help_result = subprocess.run(['mmseqs', 'search', '-h'], capture_output=True, text=True)
    (OUT / 'installed_search_help.txt').write_text(help_result.stdout + help_result.stderr)
    gpu = shutil.which('nvidia-smi')
    result = subprocess.run([gpu, '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], capture_output=True, text=True) if gpu else None
    save(OUT / 'GPU_CAPABILITY.json', {'mmseqs_version': subprocess.check_output(['mmseqs', 'version'], text=True).strip(),
        'installed_search_exposes_gpu': '--gpu' in help_result.stdout, 'nvidia_smi': gpu,
        'gpu_stdout': result.stdout if result else None, 'gpu_stderr': result.stderr if result else None,
        'gpu_returncode': result.returncode if result else None, 'installed_or_switched_build': False,
        'gpu_benchmark_authorized_by_equivalence': False})
    print('First diagnostic sealed; installed build/GPU capability recorded on D:', flush=True)

def fasta_rows(path):
    with path.open() as f:
        while header := f.readline():
            assert header.startswith('>')
            yield header[1:].strip(), f.readline().strip()

def protected():
    return {seq: sid for sid, seq in fasta_rows(SCI / 'protected/protected.fasta')}

def nearest(lines, sequences):
    exact = protected()
    result = {sid: (exact[seq], 1.0) for sid, seq in sequences.items() if seq in exact}
    for line in lines:
        candidate, target, ident, candidate_cov, protected_cov, evalue = line.split('\t')
        assert candidate in sequences
        ident = float(ident)
        assert ident >= .299 and float(candidate_cov) >= .799 and float(protected_cov) >= .799
        if candidate not in result or ident > result[candidate][1]:
            result[candidate] = (target, ident)
    return result

def prepare():
    assert os.name == 'nt'
    import psutil
    state = production_state()
    pause = json.loads((ROOT / 'reports/SCREEN_OPTIMIZATION_PAUSE.json').read_text())
    statuses = {str(pid): psutil.Process(pid).status() for pid in (pause['worker_pid'], pause['supervisor_pid'])}
    assert all(value == psutil.STATUS_STOPPED for value in statuses.values()), statuses
    save(OUT / 'PRODUCTION_PAUSED.json', {**state, 'native_statuses': statuses})
    shutil.copyfile(FIRST / 'equivalence.fasta', OUT / 'tiny.fasta')
    lines = json.loads((FIRST / 'equivalence.expected-original-order.json').read_text())
    (OUT / 'tiny.expected.tsv').write_text('\n'.join(lines) + '\n')
    with readonly(SCI / 'manifests/exclusions.sqlite') as db:
        assert db.execute('PRAGMA quick_check').fetchall() == [('ok',)]
        persisted = {f's_{h.hex()}_{off}': (target, ident) for h, off, target, ident in db.execute('SELECT h,off,protected,identity FROM removed')}
        certified_hits = 0
        for batch, count in db.execute("SELECT batch,n FROM batches WHERE name='screen' ORDER BY batch"):
            cert = json.loads((SCI / f'manifests/screen-batch-{batch:04d}.json').read_text())
            assert cert['n'] == count and cert['batch_zero_overlap_verified']
            assert sha(SCI / f'manifests/screen-batch-{batch:04d}.tsv.gz') == cert['retained_result_sha256']
            certified_hits += cert['hits']
        assert certified_hits == 178888
    with readonly(SCI / 'filtered/reservoir.sqlite') as db:
        for batch in BATCHES:
            start = time.monotonic()
            rows = db.execute('SELECT h,off,seq FROM candidates WHERE rowid>? AND rowid<=? ORDER BY rowid', ((batch-1)*100000, batch*100000)).fetchall()
            case = OUT / f'batch-{batch:04d}'; case.mkdir(exist_ok=True)
            with (case / 'query.fasta').open('w') as f:
                for h, off, seq in rows:
                    assert hashlib.sha256(seq.encode()).digest() == h
                    f.write(f'>s_{h.hex()}_{off}\n{seq}\n')
            cert = json.loads((SCI / f'manifests/screen-batch-{batch:04d}.json').read_text())
            assert sha(case / 'query.fasta') == cert['query_sha256']
            row_hash = hashlib.sha256(b''.join(h+str(off).encode()+b'\0' for h, off, seq in rows)).hexdigest()
            assert row_hash == cert['input_rows_sha256']
            shutil.copyfile(SCI / f'manifests/screen-batch-{batch:04d}.tsv.gz', case / 'expected.tsv.gz')
            keys = {f's_{h.hex()}_{off}' for h, off, seq in rows}
            save(case / 'persisted_nearest.json', {k: v for k, v in persisted.items() if k in keys})
            save(case / 'input.json', {'n': len(rows), 'input_rows_sha256': row_hash, 'query_sha256': cert['query_sha256'],
                'legacy_certificate_sha256': sha(SCI / f'manifests/screen-batch-{batch:04d}.json'), 'export_seconds': time.monotonic()-start})
            print(json.dumps({'phase': 'prepare_full_equivalence', 'batch': batch, 'n': len(rows)}), flush=True)

def command(label, argv):
    assert os.name != 'nt'
    folder = OUT / 'commands' / label; folder.mkdir(parents=True, exist_ok=True)
    marker = folder / 'complete.json'
    args = list(map(str, argv))
    if marker.exists():
        existing = json.loads(marker.read_text()); assert existing['command'] == args
        assert existing['returncode'] == 0
        return existing
    start = last = time.monotonic()
    with (folder / 'command.log').open('a') as log:
        log.write(json.dumps({'command': args, 'cwd': str(OUT)})+'\n'); log.flush()
        p = subprocess.Popen(['/usr/bin/time', '-v', '-o', str(folder / 'time.txt'), *args], cwd=OUT, stdout=log, stderr=log)
        while p.poll() is None:
            free = shutil.disk_usage(OUT).free
            floor = json.loads((SCI / 'stats/resource_plan.json').read_text())['starting_free_bytes']*3//10
            assert free >= floor, 'D: reserve breached'
            if time.monotonic()-last >= 60:
                print(json.dumps({'phase': 'architecture_experiment', 'job': label, 'seconds': time.monotonic()-start, 'screened': 3200000,
                    'production_paused': True, 'd_free_bytes': free}), flush=True)
                last = time.monotonic()
            time.sleep(1)
    record = {'command': args, 'seconds': time.monotonic()-start, 'returncode': p.returncode}
    assert p.returncode == 0, str(folder / 'command.log')
    save(marker, record)
    print(json.dumps({'completed': label, 'seconds': record['seconds']}), flush=True)
    return record

def compare(case, actual, expected, sequences):
    current = nearest(actual, sequences)
    previous = nearest(expected, sequences)
    missing = set(previous)-set(current); extra = set(current)-set(previous)
    changed = {k for k in previous.keys() & current.keys() if previous[k] != current[k]}
    result = {'case': case, 'n': len(sequences), 'expected_exclusions': len(previous), 'actual_exclusions': len(current),
        'same_exclusions': not missing and not extra, 'same_nearest': not changed,
        'same_all_qualifying_hit_fields': set(actual) == set(expected), 'missing_ids': sorted(missing), 'extra_ids': sorted(extra),
        'changed_nearest_ids': sorted(changed), 'missing_pairs': len(set(expected)-set(actual)), 'extra_pairs': len(set(actual)-set(expected))}
    result['same_qualifying_identity_and_coverage_relation'] = {tuple(line.split('\t')[:5]) for line in actual} == {tuple(line.split('\t')[:5]) for line in expected}
    result['changed_nearest_identity_ids'] = sorted(k for k in changed if previous[k][1] != current[k][1])
    result['changed_nearest_target_only_ids'] = sorted(k for k in changed if previous[k][1] == current[k][1])
    result['nearest_change_examples'] = {k: {'expected': previous[k], 'actual': current[k]} for k in sorted(changed)[:10]}
    save(OUT / (case+'.comparison.json'), result)
    return result

def certify(case, hits, sequences, meta):
    """Certify complement of the COMPLETE recorded relation, not all alignments."""
    start = time.monotonic()
    rows = hits.read_text().splitlines()
    excluded = nearest(rows, sequences)
    survivors = sorted(set(sequences)-set(excluded))
    survivor_set = set(survivors)
    assert all(line.split('\t')[0] not in survivor_set for line in rows)
    folder = OUT / 'certificates' / case; folder.mkdir(parents=True, exist_ok=True)
    result_file = folder / 'matches.tsv.gz'
    with hits.open('rb') as src, result_file.open('wb') as dst, gzip.GzipFile(filename='', mode='wb', fileobj=dst, mtime=0) as packed:
        shutil.copyfileobj(src, packed)
    with sqlite3.connect(folder / 'exclusions.sqlite') as db:
        db.execute('PRAGMA synchronous=FULL')
        db.execute('CREATE TABLE IF NOT EXISTS removed(id TEXT PRIMARY KEY,protected TEXT,identity REAL)')
        db.executemany('INSERT OR REPLACE INTO removed VALUES(?,?,?)', [(k,*v) for k,v in excluded.items()]); db.commit()
        assert dict((k,(t,i)) for k,t,i in db.execute('SELECT * FROM removed')) == excluded
    (folder / 'survivors.txt').write_text('\n'.join(survivors)+'\n')
    cert = {**meta, 'n': len(sequences), 'hits': len(excluded), 'survivors': len(survivors), 'matches_sha256': sha(hits),
        'retained_result_sha256': sha(result_file), 'survivor_ids_sha256': sha(folder / 'survivors.txt'),
        'exclusions_sha256': sha(folder / 'exclusions.sqlite'), 'complete_returned_relation_complement_verified': True,
        'independent_batch_research_performed': False, 'independent_final_selected_corpus_audit_required': True,
        'legacy_batch_verify_result_sha256_is_not_fabricated': True, 'certification_seconds': time.monotonic()-start}
    save(folder / 'certificate.json', cert)
    # Restart reads immutable inputs and validates every output before reuse.
    saved = json.loads((folder / 'certificate.json').read_text())
    assert sha(hits) == saved['matches_sha256'] and sha(result_file) == saved['retained_result_sha256']
    assert sha(folder / 'survivors.txt') == saved['survivor_ids_sha256'] and sha(folder / 'exclusions.sqlite') == saved['exclusions_sha256']
    save(folder / 'restart.json', {'all_output_checksums_verified': True, 'input_bound': saved['query_sha256'], 'replayed_production_batches': 0})
    return cert

def experiment():
    production_state()
    # A: test complement on the existing tiny relation without another search.
    sequences = dict(fasta_rows(OUT / 'tiny.fasta'))
    expected = (OUT / 'tiny.expected.tsv').read_text().splitlines()
    tiny_cert = certify('tiny', OUT / 'tiny.expected.tsv', sequences, {'query_sha256': sha(OUT / 'tiny.fasta')})
    save(OUT / 'A_TINY.json', tiny_cert)
    # B: reversed queries, unchanged bidirectional coverage. Map coordinates back.
    candidate = OUT / 'tiny-candidate-db'
    command('reverse-createdb', ['mmseqs', 'createdb', OUT / 'tiny.fasta', candidate])
    command('reverse-createindex', ['mmseqs', 'createindex', candidate, OUT / 'reverse-index-tmp', '-s', '7.5', '--threads', '8', '--split-memory-limit', '4G', '--remove-tmp-files', '0'])
    reversals = []
    for cap in (1000, len(sequences)):
        folder = OUT / f'reverse-{cap}'; folder.mkdir(exist_ok=True)
        policy = list(POLICY); policy[policy.index('--max-seqs')+1] = str(cap)
        command(f'reverse-{cap}', ['mmseqs', 'search', TARGET, candidate, folder / 'result', folder / 'tmp', *policy, '--threads', '8'])
        command(f'reverse-{cap}-convert', ['mmseqs', 'convertalis', TARGET, candidate, folder / 'result', folder / 'hits.tsv', '--format-output', FORMAT, '--threads', '8'])
        normalized = []
        for line in (folder / 'hits.tsv').read_text().splitlines():
            protected_id,candidate_id,ident,protected_cov,candidate_cov,evalue = line.split('\t')
            normalized.append('\t'.join((candidate_id,protected_id,ident,candidate_cov,protected_cov,evalue)))
        (folder / 'normalized.tsv').write_text('\n'.join(normalized)+'\n')
        reversals.append(compare(f'reverse-{cap}', normalized, expected, sequences))
    save(OUT / 'B_REVERSE_GATE.json', {'variants': reversals,
        'passed': any(r['same_exclusions'] and r['same_nearest'] and r['same_qualifying_identity_and_coverage_relation'] for r in reversals),
        'cap_expansion_is_retrieval_diagnostic_not_frozen_policy_approval': True})
    # A: full committed batches. Production files/SQL are never written.
    for batch in BATCHES:
        label = f'batch-{batch:04d}'; case = OUT / label
        sequences = dict(fasta_rows(case / 'query.fasta'))
        with gzip.open(case / 'expected.tsv.gz', 'rt') as f:
            expected = f.read().splitlines()
        folder = case / 'single-8'; folder.mkdir(exist_ok=True)
        command(label+'-single-8', ['mmseqs', 'easy-search', case / 'query.fasta', TARGET, folder / 'hits.tsv', folder / 'tmp', *POLICY, '--threads', '8', '--format-output', FORMAT])
        actual = (folder / 'hits.tsv').read_text().splitlines()
        result = compare(label+'-single-8', actual, expected, sequences)
        persisted = json.loads((case / 'persisted_nearest.json').read_text())
        result['same_persisted_nearest'] = {k:list(v) for k,v in nearest(actual,sequences).items()} == persisted
        assert all(result[k] for k in ('same_exclusions','same_nearest','same_all_qualifying_hit_fields','same_persisted_nearest')), result
        save(OUT / (label+'-single-8.comparison.json'), result)
        certify(label+'-single-8', folder / 'hits.tsv', sequences, json.loads((case / 'input.json').read_text()))
    save(OUT / 'A_FULL_GATE.json', {'passed': True, 'batches': list(BATCHES), 'n': 300000,
        'same_exclusions_survivors_nearest': True, 'certificate_scope': 'recorded_relation_complement',
        'final_full_independent_audit_required': True, 'legacy_certificates_unchanged': True})
    if any(r['same_exclusions'] and r['same_nearest'] and r['same_qualifying_identity_and_coverage_relation'] for r in reversals):
        raise RuntimeError('Reverse tiny gate needs full-batch investigation before promotion; production remains paused')

def threads():
    assert json.loads((OUT / 'A_FULL_GATE.json').read_text())['passed']
    assert (OUT / 'B_REVERSE_GATE.json').exists()
    case = OUT / 'batch-0031'; sequences = dict(fasta_rows(case / 'query.fasta'))
    with gzip.open(case / 'expected.tsv.gz', 'rt') as f: expected = f.read().splitlines()
    control = case / 'control-8'; control.mkdir(exist_ok=True)
    command('batch-0031-control-8', ['mmseqs', 'easy-search', case / 'query.fasta', TARGET, control / 'hits.tsv', control / 'tmp', *POLICY, '--threads', '8', '--format-output', FORMAT])
    control_result = compare('batch-0031-control-8', (control / 'hits.tsv').read_text().splitlines(), expected, sequences)
    assert control_result['same_exclusions'] and control_result['same_nearest'] and control_result['same_all_qualifying_hit_fields']
    folder = case / 'single-16'; folder.mkdir(exist_ok=True)
    command('batch-0031-single-16', ['mmseqs', 'easy-search', case / 'query.fasta', TARGET, folder / 'hits.tsv', folder / 'tmp', *POLICY, '--threads', '16', '--format-output', FORMAT])
    result = compare('batch-0031-single-16', (folder / 'hits.tsv').read_text().splitlines(), expected, sequences)
    assert result['same_exclusions'] and result['same_nearest'] and result['same_all_qualifying_hit_fields']
    certify('batch-0031-single-16', folder / 'hits.tsv', sequences, json.loads((case / 'input.json').read_text()))
    # Benchmark the required full final audit, without replacing it by a sample.
    # This is a throughput probe on a complete committed batch's survivors.
    excluded = nearest(expected, sequences)
    clean = [(sid, seq) for sid, seq in sequences.items() if sid not in excluded]
    export_start = time.monotonic()
    clean_fasta = case / 'audit-survivors.fasta'
    with clean_fasta.open('w') as f:
        for sid, seq in reversed(clean): f.write(f'>{sid}\n{seq}\n')
        f.flush(); os.fsync(f.fileno())
    export_seconds = time.monotonic()-export_start
    for count in (8, 16):
        audit_folder = case / f'audit-{count}'; audit_folder.mkdir(exist_ok=True)
        command(f'batch-0031-audit-{count}', ['mmseqs', 'easy-search', clean_fasta, TARGET, audit_folder / 'hits.tsv', audit_folder / 'tmp', *POLICY, '--threads', str(count), '--format-output', FORMAT])
        assert (audit_folder / 'hits.tsv').stat().st_size == 0
        save(audit_folder / 'certificate.json', {'n': len(clean), 'query_sha256': sha(clean_fasta),
            'matches_sha256': sha(audit_folder / 'hits.tsv'), 'zero_detected_violations': True,
            'export_seconds': export_seconds, 'scope': 'throughput probe; final all-selected-corpus audit remains required'})
    production_state()

def reverse_rescore():
    """Separate reversed retrieval from the frozen forward alignment predicate."""
    assert (OUT / 'B_REVERSE_GATE.json').exists()
    # MMseqs can store a DB prefix entirely as per-thread .0/.1/... shards.
    pref = sorted({p.with_suffix('').resolve() for p in (OUT / 'reverse-6020/tmp').glob('*/pref_0.dbtype')})
    assert len(pref) == 1, pref
    candidate = OUT / 'tiny-candidate-db'
    folder = OUT / 'reverse-pref-forward-align'; folder.mkdir(exist_ok=True)
    command('reverse-pref-transpose', ['mmseqs', 'swapresults', TARGET, candidate, pref[0], folder / 'pref', '--threads', '8'])
    command('reverse-pref-forward-align', ['mmseqs', 'align', candidate, TARGET, folder / 'pref', folder / 'result',
        '--min-seq-id', '0.30', '-c', '0.80', '--cov-mode', '0', '--seq-id-mode', '0', '--alignment-mode', '3',
        '-e', '0.001', '--max-rejected', '2147483647', '--max-accept', '2147483647', '--threads', '8'])
    command('reverse-pref-forward-convert', ['mmseqs', 'convertalis', candidate, TARGET, folder / 'result', folder / 'hits.tsv', '--format-output', FORMAT, '--threads', '8'])
    result = compare('reverse-pref-forward-align', (folder / 'hits.tsv').read_text().splitlines(),
        (OUT / 'tiny.expected.tsv').read_text().splitlines(), dict(fasta_rows(OUT / 'tiny.fasta')))
    save(OUT / 'B_FORWARD_RESCORE_GATE.json', result)
    if result['same_exclusions'] and result['same_nearest'] and result['same_qualifying_identity_and_coverage_relation']:
        raise RuntimeError('Rescored reverse retrieval passed tiny gate: full committed-batch tests required before promotion')

def restart_certificates():
    """Real process exits around a FULL SQLite commit; only experimental files."""
    work = OUT / 'restart-proof'; work.mkdir(exist_ok=True)
    script = work / 'crash_worker.py'
    script.write_text('''import hashlib,json,os,sqlite3,sys
from pathlib import Path
root=Path(sys.argv[1]); mode=sys.argv[2]
cert=root.parent/'certificates/tiny/certificate.json'
source=root.parent/'certificates/tiny/exclusions.sqlite'
with sqlite3.connect('file:'+source.as_posix()+'?mode=ro',uri=True) as db:
    rows=db.execute('SELECT id,protected,identity FROM removed').fetchall()
db=sqlite3.connect(root/'progress.sqlite')
db.execute('PRAGMA synchronous=FULL');db.execute('PRAGMA cache_size=1')
db.execute('BEGIN IMMEDIATE')
db.executemany('INSERT INTO removed VALUES(?,?,?)',rows)
db.execute('INSERT INTO batches VALUES(1,?,?,?)',(len(rows),hashlib.sha256(cert.read_bytes()).hexdigest(),str(cert)))
with (root/'orphan-certificate.json').open('w') as f:
    json.dump(json.loads(cert.read_text()),f);f.flush();os.fsync(f.fileno())
if mode=='before_commit':os._exit(93)
db.commit();os._exit(94)
''')
    progress = work / 'progress.sqlite'
    assert not progress.exists(), 'Preserve an existing restart experiment rather than overwriting it'
    with sqlite3.connect(progress) as db:
        db.execute('CREATE TABLE removed(id TEXT PRIMARY KEY,protected TEXT,identity REAL)')
        db.execute('CREATE TABLE batches(batch INTEGER PRIMARY KEY,hits INTEGER,certificate_sha256 TEXT,marker TEXT)'); db.commit()
    before = subprocess.run([sys.executable, '-B', str(script), str(work), 'before_commit'])
    assert before.returncode == 93
    with sqlite3.connect(progress) as db:
        assert db.execute('SELECT count(*) FROM removed').fetchone()[0] == 0
        assert db.execute('SELECT count(*) FROM batches').fetchone()[0] == 0
    after = subprocess.run([sys.executable, '-B', str(script), str(work), 'after_commit'])
    assert after.returncode == 94
    with readonly(progress) as db:
        batch, hits, expected_hash, marker = db.execute('SELECT * FROM batches').fetchone()
        assert batch == 1 and hits == 274 and sha(Path(marker)) == expected_hash
        assert db.execute('SELECT count(*) FROM removed').fetchone()[0] == 274
    validations = []
    for path in sorted((OUT / 'certificates').glob('*/certificate.json')):
        cert = json.loads(path.read_text()); folder = path.parent
        assert sha(folder / 'matches.tsv.gz') == cert['retained_result_sha256']
        assert sha(folder / 'survivors.txt') == cert['survivor_ids_sha256']
        assert sha(folder / 'exclusions.sqlite') == cert['exclusions_sha256']
        validations.append({'case': folder.name, 'all_output_checksums_verified': True})
    production_state()
    save(OUT / 'RESTART_CERTIFICATES.json', {'precommit_crash': 'orphan certificate ignored; no committed exclusions/batch',
        'postcommit_crash': 'SQL checkpoint authoritative; certificate hash verified; search reused',
        'certificate_checks': validations, 'production_replayed_or_modified': False})
    print('Experimental pre/post-commit crash recovery and certificate checks passed', flush=True)

def report():
    # Include publication and restart checksum validation, which occur AFTER
    # the inner certificate's bookkeeping timer, in the complete-work model.
    timing_path = OUT / 'COMPLETE_CERTIFICATION_TIMING.json'
    if not timing_path.exists():
        case = OUT / 'batch-0031'; inputs = json.loads((case / 'input.json').read_text())
        started = time.monotonic()
        certify('complete-timing-single-16', case / 'single-16/hits.tsv', dict(fasta_rows(case / 'query.fasta')), inputs)
        complete_timing = {'single_certificate_including_publish_and_restart_seconds': time.monotonic()-started, 'audit_certification': {}}
        for count in (8, 16):
            started = time.monotonic(); folder = OUT / 'complete-timing' / f'audit-{count}'; folder.mkdir(parents=True, exist_ok=True)
            source = json.loads((case / f'audit-{count}/certificate.json').read_text())
            assert sha(case / 'audit-survivors.fasta') == source['query_sha256']
            assert (case / f'audit-{count}/hits.tsv').stat().st_size == 0
            with sqlite3.connect(folder / 'progress.sqlite') as db:
                db.execute('PRAGMA synchronous=FULL')
                db.execute('CREATE TABLE progress(n INTEGER,query_sha256 TEXT,result_sha256 TEXT)')
                db.execute('INSERT INTO progress VALUES(?,?,?)', (source['n'], source['query_sha256'], source['matches_sha256'])); db.commit()
                assert db.execute('SELECT n FROM progress').fetchone()[0] == source['n']
            save(folder / 'certificate.json', source)
            assert json.loads((folder / 'certificate.json').read_text()) == source
            complete_timing['audit_certification'][str(count)] = {'publish_and_checksum_seconds': time.monotonic()-started,
                'fasta_export_seconds': source['export_seconds']}
        save(timing_path, complete_timing)
    complete_timing = json.loads(timing_path.read_text())
    spec = importlib.util.spec_from_file_location('frozen_diagnostic', Path(__file__).with_name('optimize_e012_v2c_screen.py'))
    frozen = importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen)
    metrics = {p.parent.name: frozen.time_metrics(p) for p in (OUT / 'commands').glob('*/time.txt')}
    a = json.loads((OUT / 'A_FULL_GATE.json').read_text())
    b = json.loads((OUT / 'B_REVERSE_GATE.json').read_text())
    sequences = dict(fasta_rows(OUT / 'tiny.fasta')); expected = (OUT / 'tiny.expected.tsv').read_text().splitlines()
    reverse_details = [compare(f'reverse-{cap}', (OUT / f'reverse-{cap}/normalized.tsv').read_text().splitlines(), expected, sequences) for cap in (1000, 6020)]
    rescore = json.loads((OUT / 'B_FORWARD_RESCORE_GATE.json').read_text())
    batches = []
    for batch in BATCHES:
        label = f'batch-{batch:04d}-single-8'
        cert = json.loads((OUT / 'certificates' / label / 'certificate.json').read_text())
        batches.append({'batch': batch, 'n': cert['n'], 'exclusions': cert['hits'], 'survivors': cert['survivors'],
            'search_seconds': metrics[label]['wall_seconds'], 'export_seconds': cert['export_seconds'],
            'certification_seconds': cert['certification_seconds'], 'exact_equivalence': json.loads((OUT / (label+'.comparison.json')).read_text())})
    base8 = metrics['batch-0031-control-8']['wall_seconds']
    fast16 = metrics['batch-0031-single-16']['wall_seconds']
    audit8 = metrics['batch-0031-audit-8']['wall_seconds']
    audit16 = metrics['batch-0031-audit-16']['wall_seconds']
    fast_cert = json.loads((OUT / 'certificates/batch-0031-single-16/certificate.json').read_text())
    n_survivors = fast_cert['survivors']
    # Same-input workload accounting: baseline does one candidate search, one
    # survivor re-search and the final independent audit; new method omits only
    # the per-batch re-search. The final 20M audit is charged to BOTH methods.
    remaining_batches = 20800000 / 100000
    final_audit_blocks = 20000000 / n_survivors
    export = fast_cert['export_seconds']; cert_seconds = complete_timing['single_certificate_including_publish_and_restart_seconds']
    audit8_work = audit8 + sum(complete_timing['audit_certification']['8'].values())
    audit16_work = audit16 + sum(complete_timing['audit_certification']['16'].values())
    original_remaining = remaining_batches*(export+base8+audit8_work+cert_seconds) + final_audit_blocks*audit8_work
    optimized_remaining = remaining_batches*(export+fast16+cert_seconds) + final_audit_blocks*audit16_work
    speedup = original_remaining/optimized_remaining
    assert speedup < 5, 'Acceptance target reached: production implementation/promotion required'
    first_inventory = json.loads((OUT / 'FIRST_DIAGNOSTIC_FROZEN.json').read_text())
    verified = 0; last = time.monotonic()
    for name, record in first_inventory['files'].items():
        path = FIRST / name
        if 'symlink' in record: assert os.readlink(path) == record['symlink']
        else: assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256']
        verified += 1
        if time.monotonic()-last >= 30:
            print(json.dumps({'phase': 'verify_first_diagnostic_unchanged', 'files': verified}), flush=True); last=time.monotonic()
    result = {'decision': 'STOPPED_BELOW_5X', 'production': production_state(), 'frozen_first_commit': FROZEN_COMMIT,
        'first_artifact_hashes_reverified': verified, 'A': a, 'B_direct': reverse_details, 'B_forward_rescore': rescore,
        'full_batch_benchmarks': batches, 'command_metrics': metrics,
        'complete_certificate_publication_and_validation_timers': complete_timing,
        'same_input_16_vs_8_single_search_speedup': base8/fast16,
        'single_search_screen_candidates_s_including_export_certification': 100000/(export+fast16+cert_seconds),
        'remaining_work_speedup_including_final_full_audit': speedup,
        'remaining_original_hours_model': original_remaining/3600,
        'remaining_single_search_16_hours_model': optimized_remaining/3600,
        'final_audit_required_candidates': 20000000, 'remaining_screen_candidates': 20800000,
        'end_to_end_estimate_is_model_not_completed_20M_audit': True,
        'restart': json.loads((OUT / 'RESTART_CERTIFICATES.json').read_text()),
        'retrieval_diagnosis': json.loads((OUT / 'B_RETRIEVAL_DIAGNOSIS.json').read_text()) if (OUT / 'B_RETRIEVAL_DIAGNOSIS.json').exists() else None,
        'threading_comparison_waited_for_cpu_capacity': True,
        'peer_workload_suspended': False,
        'gpu': json.loads((OUT / 'GPU_CAPABILITY.json').read_text()), 'training_launched': False,
        'd_free_bytes': shutil.disk_usage(OUT).free}
    save(OUT / 'REPORT.json', result)
    lines = ['E012 second screening-architecture diagnostic', '',
        'Decision: production stays paused at batch 32, 3,200,000 screened, 178,888 exclusions. No source scan, historical union, reservoir finalization, production screening or training was launched.', '',
        'The first attempt is frozen at '+FROZEN_COMMIT+'. Its D: artifact hashes were sealed before this experiment and reverified afterwards.', '',
        'A — one-search relation and complement', '',
        'Let C be the input candidate-ID set, R the entire successful frozen MMseqs output relation, and E the independently detected exact protected-string IDs. Then H = E union projection_candidate(R), and S = C minus H. Every pair in R has its candidate in H; therefore no pair in R has its candidate in S. Nearest identity and target are derived in original returned order, seeded by exact matches at identity 1.0; the strict-greater identity update preserves ties. Multiple protected hits never multiply candidate exclusions.', '',
        'This proves completeness relative to the recorded heuristic relation, not the set of every possible biological alignment. The frozen 1,000-result prefilter cap and masking remain part of the operational search. With the immutable protected index, fixed sensitivity, one split and target database size, query scoring/filtering is per sequence. Removing other queries or reversing FASTA order does not change this operational predicate. The v15 prefilter source processes each query with matchQuery; its k-mer/splitting choices are target/index based in this configuration. Alignment E-value calibration uses target amino-acid database size, which is unchanged by removing candidate queries.', '',
        'The protocol explicitly requests independent batch survivor re-search AND an independent final selected-corpus search. The complement proof can replace the repeated decision calculation, but must not impersonate independent evidence. Proposed certificates retain query/result/input-row checksum meanings and add an explicit recorded-relation-complement scope. They do not fabricate the legacy empty independent-search hash. Existing certificates are untouched. The final FULL selected-corpus audit remains mandatory; a sample cannot certify the entire selected corpus.', '',
        '| Full committed batch | Candidates | Exclusions | Survivors | Single search, 8 threads | Certification |',
        '|---|---:|---:|---:|---:|---:|']
    for record in batches:
        lines.append(f"| {record['batch']} | {record['n']:,} | {record['exclusions']:,} | {record['survivors']:,} | {record['search_seconds']:.2f}s | {record['certification_seconds']:.2f}s |")
    lines += ['', 'All 300,000 candidates match retained full hit fields, excluded IDs, survivor complements, persisted nearest protected target/identity and exact-sequence overrides. Candidate canonical sequence hashes, FASTA hashes and input-row hashes also match the original certificates.', '',
        'B — reversal', '',
        'For the frozen bidirectional policy, keep --cov-mode 0, -c 0.80 and --min-seq-id 0.30 with seq-id-mode 0/alignment-mode 3. On transposition, candidate coverage is reverse tcov and protected coverage is reverse qcov. A candidate is excluded if ANY protected query qualifies; nearest identity must be a candidate-level maximum, not a per-protected-query best hit. These alignment thresholds are symmetric, but the engine is not generally reciprocal: per-query result caps, query composition correction, target masking and target-size-dependent E-values change. Expanding the reverse cap to all 6,020 targets is a retrieval diagnostic, not approval to change the frozen candidate-side cap.', '']
    for r in reverse_details:
        lines.append(f"{r['case']}: expected {r['expected_exclusions']} exclusions; found {r['actual_exclusions']}; missed {len(r['missing_ids'])}; added {len(r['extra_ids'])}; changed nearest identities {len(r['changed_nearest_identity_ids'])}, target-only ties {len(r['changed_nearest_target_only_ids'])}. Exact identity/coverage relation match: {r['same_qualifying_identity_and_coverage_relation']}.")
    lines += ['', 'A third experiment transposed the uncapped reversed PREFILTER relation and realigned pairs in the ORIGINAL candidate-to-protected direction using the frozen identity, coverage, alignment, composition correction and E-value defaults. This corrects the alignment predicate and target-size calibration while testing whether reversed retrieval is sufficient.',
        f"Forward-rescored retrieval: expected {rescore['expected_exclusions']} exclusions, found {rescore['actual_exclusions']}; missed {len(rescore['missing_ids'])}; added {len(rescore['extra_ids'])}; changed nearest records {len(rescore['changed_nearest_ids'])}. Qualifying identity/coverage relation match: {rescore['same_qualifying_identity_and_coverage_relation']}.", '',
        'The tiny reversal gate failed. Full committed-batch reversal and larger target blocks were therefore not launched. There is no analytical guarantee that directional masked/prefiltered retrieval is a superset of the original top-1000 candidate search; adding a forward full prefilter to recover that relation would restore the fundamental expensive work.', '',
        'Inspection of the frozen prefilter rows isolates an actual retrieval difference rather than merely rounded E-values. The added candidate/protected pair qualifies in the ORIGINAL alignment direction at identity 0.301, candidate coverage 0.867, protected coverage 0.875 and E-value 9.573E-34, but was absent from its original forward prefilter row, which contained only 92 results. Two original qualifying pairs were lost on another candidate; its original row contained 150 results, including the nearest identity-0.353 pair. Thus these counterexamples occur below the 1,000-result cap: enlarging the reverse cap cannot make the directional retrieval relations identical. The added exclusion is an implementation mismatch, not a claim of a biological false positive. See B_RETRIEVAL_DIAGNOSIS.json.', '',
        'C — throughput including certification and independent audit', '',
        'The initial full-batch equivalence runs shared the host with an unrelated eight-worker E010 job. That job finished naturally. The controlled 8/16 single-search and audit comparisons waited for free CPU capacity; no peer workload was suspended and no simultaneous MMseqs jobs were launched.', '',
        f'Same-input single search: eight threads {base8:.2f}s; sixteen threads {fast16:.2f}s ({base8/fast16:.2f}x). The complete {n_survivors:,}-survivor audit returned zero hits at both thread counts: eight {audit8:.2f}s, sixteen {audit16:.2f}s.',
        f'Single-search screening including native FASTA export and checksum/SQLite/complement certification: {result["single_search_screen_candidates_s_including_export_certification"]:.2f} candidates/s. Remaining-work model charges all 20.8M unscreened candidates plus ALL 20M final audit candidates. Original remaining workflow: {original_remaining/3600:.1f} hours; proposed single-search/16-thread workflow: {optimized_remaining/3600:.1f} hours; modeled gain {speedup:.2f}x. This is a measured per-block throughput model, not a completed 20M audit or certification claim.', '',
        'No exact tested architecture demonstrates >=5x complete screening/audit throughput. Production stays paused; verify-protected -> final -> certify -> diversity -> handoff remain pending. Residual cost is sensitivity-7.5 per-candidate prefiltering/alignment plus the mandatory independent final audit.', '',
        'Restart/certificates: actual experimental child exits before and after FULL SQLite commit verified orphan-certificate rejection, SQL-authoritative resume, unchanged exclusion counts and reuse of already-completed results. All complement-certificate output hashes passed restart validation; no production batch was replayed into production state.', '',
        'D — GPU: RTX 5060 is visible, but installed MMseqs 15-6f452+ds-2 search has no --gpu option. No build was installed/switched and no inequivalent GPU algorithm was benchmarked.', '',
        'Primary references: https://raw.githubusercontent.com/soedinglab/MMseqs2/15-6f452/src/prefiltering/Prefiltering.cpp ; https://raw.githubusercontent.com/soedinglab/MMseqs2/15-6f452/src/alignment/Alignment.cpp ; https://github.com/soedinglab/MMseqs2/wiki/Home/726c6045e91bb3554ec2dbac02cf16272d17a83c', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'decision': result['decision'], 'speedup_including_final_audit': speedup,
        'modeled_remaining_hours': optimized_remaining/3600, 'production': result['production']}), flush=True)

def finish():
    last = time.monotonic()
    while not (OUT / 'A_FULL_GATE.json').exists():
        if time.monotonic()-last >= 60:
            print('Waiting for full A/B gates; production remains paused', flush=True); last=time.monotonic()
        time.sleep(1)
    reverse_rescore()
    import psutil
    while psutil.cpu_percent(interval=2) * psutil.cpu_count() / 100 > 1.2:
        print('Waiting for CPU capacity before 16-thread tests; no peer job is modified', flush=True)
        time.sleep(30)
    threads()
    restart_certificates()
    report()

def hold_finish():
    import psutil
    matches = []
    for p in psutil.process_iter(['cmdline']):
        args = p.info['cmdline'] or []
        if args and args[-1] == 'finish' and any(str(a).endswith(Path(__file__).name) for a in args): matches.append(p)
    assert len(matches) == 1
    p = matches[0]; assert not p.children(), 'Do not interrupt an active MMseqs child'
    p.terminate(); p.wait(10)
    save(OUT / 'FINISH_HELD_FOR_CPU.json', {'pid': p.pid, 'reason': 'Replace waiting supervisor with CPU-capacity-aware supervisor', 'active_experiment_interrupted': False})
    print('Stopped only the idle experimental finish supervisor; A/B tests continue', flush=True)

def host_status():
    import psutil
    processes = list(psutil.process_iter(['pid', 'name']))
    for p in processes:
        try: p.cpu_percent()
        except psutil.NoSuchProcess: pass
    time.sleep(1)
    busy = []
    for p in processes:
        try:
            cpu = p.cpu_percent()
            if cpu > 5: busy.append({'pid': p.pid, 'name': p.info['name'], 'cpu_percent': cpu,
                'rss_bytes': p.memory_info().rss, 'threads': p.num_threads()})
        except (psutil.NoSuchProcess, psutil.AccessDenied): pass
    snapshot = {'time': time.time(), 'production': production_state(), 'logical_cpus': psutil.cpu_count(),
        'physical_cpus': psutil.cpu_count(logical=False), 'busy_processes': sorted(busy, key=lambda p: -p['cpu_percent'])[:12],
        'd_free_bytes': shutil.disk_usage(OUT).free}
    with (OUT / 'host_status.jsonl').open('a') as f: f.write(json.dumps(snapshot)+'\n')
    print(json.dumps(snapshot), flush=True)

def retrieval_diagnosis():
    """Inspect a few frozen prefilter rows; never rebuild either relation."""
    def lookup(prefix):
        with Path(str(prefix)+'.lookup').open() as f:
            return {columns[1]: int(columns[0]) for line in f if (columns := line.split())}
    def read_row(prefix, key):
        with Path(str(prefix)+'.index').open() as f:
            found = next((line.split() for line in f if int(line.split()[0]) == key), None)
        if found is None: return []
        offset, length = int(found[1]), int(found[2])
        shards = [prefix] if prefix.is_file() else sorted(
            (p for p in prefix.parent.glob(prefix.name+'.*') if re.fullmatch(re.escape(prefix.name)+r'\.\d+', p.name)),
            key=lambda p: int(p.suffix[1:]))
        for p in shards:
            size = p.stat().st_size
            if offset >= size: offset -= size; continue
            assert offset+length <= size
            with p.open('rb') as f: f.seek(offset); data=f.read(length)
            return [line.split('\t') for line in data.rstrip(b'\0').decode().splitlines()]
        raise RuntimeError('MMseqs DB offset exceeds its physical shards')
    original_db = FIRST / 'equivalence-db'
    query_keys = lookup(original_db); target_keys = lookup(TARGET)
    reverse_query_keys = lookup(OUT / 'tiny-candidate-db')
    prefixes = sorted({p.with_suffix('').resolve() for p in (FIRST / 'equivalence-8/tmp').glob('*/pref_0.dbtype')})
    assert len(prefixes) == 1
    case = json.loads((OUT / 'B_FORWARD_RESCORE_GATE.json').read_text())
    expected = (OUT / 'tiny.expected.tsv').read_text().splitlines()
    actual = (OUT / 'reverse-pref-forward-align/hits.tsv').read_text().splitlines()
    added = set(actual)-set(expected); missing = set(expected)-set(actual)
    proof = []
    for label, differences in [('added_forward_qualifying_pair', added), ('lost_original_qualifying_pair', missing)]:
        for line in sorted(differences):
            cid, prot, *fields = line.split('\t')
            original = read_row(prefixes[0], query_keys[cid])
            transposed = read_row(OUT / 'reverse-pref-forward-align/pref', reverse_query_keys[cid])
            proof.append({'type': label, 'candidate': cid, 'protected': prot, 'identity_cov_evalue': fields,
                'original_forward_prefilter_count': len(original),
                'present_in_original_forward_prefilter': any(int(row[0]) == target_keys[prot] for row in original),
                'transposed_reverse_prefilter_count': len(transposed),
                'present_in_transposed_reverse_prefilter': any(int(row[0]) == target_keys[prot] for row in transposed)})
    save(OUT / 'B_RETRIEVAL_DIAGNOSIS.json', {'pairs': proof, 'source_prefilter_sha256': sha(Path(str(prefixes[0])+'.index')),
        'first_attempt_read_only': True, 'extra_candidate_count': len(case['extra_ids'])})
    print(json.dumps(proof), flush=True)

def verify_paused():
    assert os.name == 'nt'
    import psutil
    state = production_state()
    pause = json.loads((ROOT / 'reports/SCREEN_OPTIMIZATION_PAUSE.json').read_text())
    states = {str(pid): psutil.Process(pid).status() for pid in (pause['worker_pid'], pause['supervisor_pid'])}
    assert all(value == psutil.STATUS_STOPPED for value in states.values())
    certificates = {}; total_hits = 0
    for batch in range(1, 33):
        path = SCI / f'manifests/screen-batch-{batch:04d}.json'; cert = json.loads(path.read_text())
        retained = SCI / f'manifests/screen-batch-{batch:04d}.tsv.gz'
        assert cert['n'] == 100000 and cert['batch_zero_overlap_verified']
        assert sha(retained) == cert['retained_result_sha256']
        certificates[str(batch)] = {'certificate_sha256': sha(path), 'retained_sha256': sha(retained)}
        total_hits += cert['hits']
    assert total_hits == 178888
    save(OUT / 'FINAL_PRODUCTION_PAUSED.json', {**state, 'native_statuses': states, 'verified_certificates': certificates})
    print(json.dumps({**state, 'native_statuses': states, 'all_32_certificates_verified': True}), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('audit', 'prepare', 'experiment', 'reverse-rescore', 'threads', 'restart-certificates', 'report', 'finish', 'host-status', 'hold-finish', 'retrieval-diagnosis', 'verify-paused'))
    action = parser.parse_args().action
    setup()
    {'audit': audit, 'prepare': prepare, 'experiment': experiment, 'reverse-rescore': reverse_rescore,
     'threads': threads, 'restart-certificates': restart_certificates, 'report': report, 'finish': finish,
     'host-status': host_status, 'hold-finish': hold_finish, 'retrieval-diagnosis': retrieval_diagnosis, 'verify-paused': verify_paused}[action]()
