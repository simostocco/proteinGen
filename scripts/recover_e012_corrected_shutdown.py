"""Audit and preserve corrected E012 shutdown evidence; use the existing runner."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]
BASE = Path('D:/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2') if os.name == 'nt' else Path('/mnt/d/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2')
ROOT = BASE / 'v2c_optimization/scientific_corrected_protein_v1'

def native(value):
    return Path(value.replace('/mnt/d', 'D:')) if os.name == 'nt' else Path(value.replace('D:', '/mnt/d').replace('\\', '/'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def ro(path):
    return sqlite3.connect(Path(path).as_uri() + '?mode=ro', uri=True)

def audit():
    protocol = read(ROOT / 'CORRECTION_PROTOCOL.json')
    assert protocol['validated'] and protocol['failure_action'] == 'DATA2-D; stop'
    for key, filename in [('driver_sha256', 'run_e012_v2c_screen_production.py'), ('pipeline_sha256', 'complete_e012_v2c_pipeline.py')]:
        assert sha(REPO / 'scripts' / filename) == protocol[key]
    assert sha(REPO / 'scripts/prepare_e012_sequence_corpus_v2b.py') == read(ROOT / 'manifests/screen.complete.json')['script_sha256']
    assert protocol['max_seqs_production'] == protocol['max_seqs_independent_audit'] == 53615
    assert protocol['threads'] == 16 and protocol['query_dbtype'] == protocol['search_type'] == 1
    assert protocol['direction'] == 'candidate_to_protected' and protocol['adopted_batches'] == []
    assert (protocol['identity'], protocol['coverage'], protocol['cov_mode'], protocol['sensitivity'], protocol['seq_id_mode'], protocol['alignment_mode']) == (0.3, 0.8, 0, 7.5, 0, 3)
    diagnostic = BASE / 'v2c_optimization/data2d_batch200_diagnostic'
    assert sha(diagnostic / 'SCOPE_VALIDATION.json') == protocol['scope_sha256']
    assert sha(diagnostic / 'CORRECTED_VALIDATION.json') == protocol['validation_sha256']
    marker = read(ROOT / 'manifests/screen.complete.json')
    for path, expected in marker['outputs'].items():
        assert sha(native(path)) == expected, path
    reservoir = read(ROOT / 'manifests/reservoir.complete.json')
    reservoir_path = next(native(p) for p in reservoir['outputs'] if p.endswith('reservoir.sqlite'))
    assert sha(reservoir_path) == marker['inputs']['reservoir_sha256']
    print('Reservoir and corrected screen output hashes verified', flush=True)
    protected = sha(ROOT / 'protected/protected.fasta')
    assert protected == marker['inputs']['protected_sha256'] == '1f6df25e70e90d41bfb6a86268372d652f9ddccff15d4b0bfaf00899dca419ae'
    index_path = BASE / 'manifests/target_index.complete.json'
    index = read(index_path)
    assert index['inputs'] == dict(mmseqs_version='15-6f452+ds-2', protected_sha256=protected, sensitivity=7.5)
    for path, expected in index['outputs'].items():
        assert sha(native(path)) == expected, path
    assert sha(index_path) == read(ROOT / 'production/reports/PROTECTED_INDEX_VALIDATED.json')['marker_sha256']
    inventory = {}
    def pin(path):
        inventory[str(path)] = dict(sha256=sha(path), bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
    for name in ['CORRECTION_PROTOCOL.json', 'manifests/screen.complete.json', 'manifests/exclusions.sqlite', 'stats/screen.json', 'protected/protected.fasta']:
        pin(ROOT / name)
    for stage, dbname, expected_count in [('screen', 'exclusions.sqlite', 241), ('verify_protected', 'verification_progress.sqlite', 36)]:
        with ro(ROOT / 'manifests' / dbname) as db:
            assert db.execute('PRAGMA integrity_check').fetchall() == [('ok',)]
            batches = db.execute('SELECT batch,n,marker FROM batches WHERE name=? ORDER BY batch', (stage,)).fetchall()
            assert [b for b, _, _ in batches] == list(range(1, expected_count + 1))
            removed = db.execute('SELECT count(*) FROM removed').fetchone()[0]
            assert removed == (829092 if stage == 'screen' else 0)
            if stage == 'screen':
                cp = json.loads(db.execute("SELECT v FROM production_progress WHERE k='checkpoint'").fetchone()[0])
                assert cp['batch'] == 241 and cp['screened'] == 24085944 and cp['exclusions'] == 829092
                assert cp['inputs'] == marker['inputs']
        total_hits = 0
        for batch, n, recorded_path in batches:
            cert_path = ROOT / 'manifests' / f'{stage}-batch-{batch:04d}.json'
            assert native(recorded_path) == cert_path
            cert = read(cert_path)
            assert cert['n'] == n and cert['batch'] == batch
            retained = cert_path.with_suffix('.tsv.gz')
            assert sha(retained) == cert['retained_result_sha256']
            with gzip.open(retained, 'rb') as f:
                assert hashlib.sha256(f.read()).hexdigest() == cert['matches_sha256']
            if stage == 'verify_protected':
                assert cert['hits'] == 0 and cert['mmseqs_version'] == '15-6f452+ds-2'
                args = cert['command']
                for flag, value in [('--min-seq-id','0.30'),('-c','0.80'),('--cov-mode','0'),('-s','7.5'),('--seq-id-mode','0'),('--alignment-mode','3'),('--max-seqs','53615'),('--dbtype','1'),('--search-type','1'),('--threads','16')]:
                    assert args[args.index(flag)+1] == value
                assert native(args[3]) == BASE / 'mmseqs/protected_index/db'
                assert cert['inputs']['protected_sha256'] == protected
            total_hits += cert['hits']
            pin(cert_path); pin(retained)
        assert sum(n for _, n, _ in batches) == (24085944 if stage == 'screen' else 3600000)
        assert total_hits == (829092 if stage == 'screen' else 0)
    assert not (ROOT / 'manifests/verify_protected.complete.json').exists()
    assert not (ROOT / 'manifests/verify_protected-batch-0037.json').exists()
    assert not (ROOT / 'production/reports/FAILURE.json').exists()
    # Reconstruct the frozen independent population/order, checking every durable
    # batch input and reversed FASTA hash without generating query data on disk.
    with ro(reservoir_path) as db:
        db.execute('PRAGMA cache_size=-524288')
        db.execute('PRAGMA temp_store=MEMORY')
        db.execute('ATTACH DATABASE ? AS excluded', (str(ROOT / 'manifests/exclusions.sqlite'),))
        db.execute('PRAGMA excluded.cache_size=-65536')
        where = 'WHERE NOT EXISTS(SELECT 1 FROM excluded.removed e WHERE e.h=c.h AND e.off=c.off)'
        cutoff = db.execute('SELECT c.p,c.h,c.off FROM candidates c '+where+' ORDER BY c.p,c.h,c.off LIMIT 1 OFFSET 19999999').fetchone()
        cursor = db.execute('SELECT c.h,c.off,c.seq FROM candidates c '+where+' AND (c.p,c.h,c.off)<=(?,?,?) ORDER BY c.rowid LIMIT 20000000', cutoff)
        for batch in range(1, 38):
            rows = cursor.fetchmany(100000)
            rowhash = hashlib.sha256(b''.join(h+str(off).encode()+b'\0' for h,off,_ in rows)).hexdigest()
            # The existing owner writes FASTA with Windows text-mode CRLF.
            queryhash = hashlib.sha256(b''.join(f'>s_{h.hex()}_{off}\r\n{seq}\r\n'.encode() for h,off,seq in reversed(rows))).hexdigest()
            if batch <= 36:
                cert = read(ROOT / f'manifests/verify_protected-batch-{batch:04d}.json')
                assert rowhash == cert['input_rows_sha256'] and queryhash == cert['query_sha256'], batch
            else:
                assert queryhash == sha(ROOT / 'mmseqs/verify_protected/batch-0037/query.fasta')
            if batch % 6 == 0:
                print(f'Input and query hashes verified through batch {batch}', flush=True)
    historical = BASE / 'v2c_optimization'
    for folder in [historical / 'data2d_batch200_diagnostic', historical / 'scientific/manifests', historical / 'screen_production/reports']:
        for path in folder.rglob('*'):
            if path.is_file() and not path.is_symlink():
                pin(path)
    print('All read-only recovery checks passed', flush=True)
    return dict(timestamp_unix=time.time(), last_durable_batch=36, first_uncommitted_batch=37,
                screened=3600000, total=20000000, hits=0, screen_marker_sha256=sha(ROOT / 'manifests/screen.complete.json'),
                batch37_input_rows_sha256=rowhash, batch37_query_sha256=queryhash,
                interrupted_batch='37: search completed; certificate and SQLite transaction absent',
                protected_sha256=protected, frozen_inventory=inventory,
                d_free_bytes=shutil.disk_usage(ROOT).free, training_launched=False)

def prepare():
    evidence = audit()
    destination = ROOT / 'production/reports' / time.strftime('shutdown_recovery_%Y%m%dT%H%M%SZ', time.gmtime())
    destination.mkdir()
    for filename in ['production/STATE.json', 'production/LAUNCH.json', 'production/reports/HEARTBEAT.json', 'manifests/verification_progress.sqlite']:
        shutil.copy2(ROOT / filename, destination / Path(filename).name)
    active = ROOT / 'mmseqs/verify_protected/batch-0037'
    evidence['preserved_batch37_files'] = {str(p.relative_to(active)): sha(p) for p in active.rglob('*') if p.is_file() and not p.is_symlink()}
    # Same-volume rename preserves all uncommitted evidence. The unchanged
    # legacy transaction recovery recreates batch 37 from the pinned population.
    active.rename(destination / 'uncommitted_batch-0037')
    evidence['action'] = 'Preserve complete uncommitted batch 37 by D: rename; replay with existing corrected runner'
    (destination / 'AUDIT.json').write_text(json.dumps(evidence, indent=2))
    print(json.dumps({'record':str(destination), 'last_durable_batch':36, 'replay_batch':37}), flush=True)

def post():
    destination = sorted((ROOT / 'production/reports').glob('shutdown_recovery_*'))[-1]
    evidence = read(destination / 'AUDIT.json')
    for path, before in evidence['frozen_inventory'].items():
        current = native(path)
        assert sha(current) == before['sha256'], path
        assert current.stat().st_mtime_ns == before['mtime_ns'], path
    with ro(ROOT / 'manifests/verification_progress.sqlite') as db:
        batches = db.execute("SELECT batch,n FROM batches WHERE name='verify_protected' ORDER BY batch").fetchall()
        assert db.execute('SELECT count(*) FROM removed').fetchone()[0] == 0
        assert [b for b,n in batches] == list(range(1, len(batches)+1))
    assert len(batches) >= 37, 'Wait for the first new durable batch'
    cert = read(ROOT / 'manifests/verify_protected-batch-0037.json')
    assert cert['input_rows_sha256'] == evidence['batch37_input_rows_sha256']
    assert cert['query_sha256'] == evidence['batch37_query_sha256']
    assert cert['hits'] == 0 and cert['n'] == 100000
    retained = ROOT / 'manifests/verify_protected-batch-0037.tsv.gz'
    assert sha(retained) == cert['retained_result_sha256']
    with gzip.open(retained, 'rb') as f:
        assert not f.read(1)
    old = read(ROOT / 'manifests/verify_protected-batch-0036.json')['command']
    assert cert['command'] == [x.replace('batch-0036', 'batch-0037') for x in old]
    launch = read(ROOT / 'production/LAUNCH.json')
    state = read(ROOT / 'production/STATE.json')
    heartbeat = read(ROOT / 'production/reports/HEARTBEAT.json')
    assert state['phase'] != 'failed' and not (ROOT / 'production/reports/FAILURE.json').exists()
    assert state['training_launched'] is False and heartbeat['training_launched'] is False
    assert time.time() - heartbeat['timestamp_unix'] < 180
    stats = dict(timestamp_unix=time.time(), first_new_committed_batch=37,
                 committed_batch=batches[-1][0], audit_screened=sum(n for b,n in batches), audit_total=20000000,
                 hits=0, worker_pid=launch['worker_pid'], observer_controller_pid=launch['observer_controller_pid'],
                 phase=state['phase'], heartbeat=heartbeat, d_free_bytes=shutil.disk_usage(ROOT).free,
                 frozen_hashes_and_mtimes_unchanged=True, historical_evidence_unchanged=True,
                 replay_query_hash_unchanged=True, replay_input_rows_hash_unchanged=True,
                 first_new_certificate_sha256=sha(ROOT / 'manifests/verify_protected-batch-0037.json'),
                 training_launched=False)
    (destination / 'POST_RESUME.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps({k:v for k,v in stats.items() if k!='heartbeat'}), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['audit', 'prepare', 'post'])
    args = p.parse_args()
    if args.action == 'audit':
        result = audit()
        result.pop('frozen_inventory')
        print(json.dumps(result), flush=True)
    elif args.action == 'prepare':
        prepare()
    else:
        post()
