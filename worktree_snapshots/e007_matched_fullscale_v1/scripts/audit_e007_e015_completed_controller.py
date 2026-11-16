"""Bounded, read-only verification of the completed comparison; never launches jobs."""

import collections
import datetime
import hashlib
import json
import math
import os
import pathlib
import subprocess
import time

BASE = pathlib.Path('/mnt/d/Simone/proteinGen')
COMPARISON = BASE / 'e007_e015_epoch1_comparison_v1'
E015 = BASE / 'e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1'
DEADLINE = time.monotonic() + 55
HASHES = {}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    path = pathlib.Path(path)
    assert path.is_relative_to('/mnt/d'), str(path)
    if path not in HASHES:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            while block := stream.read(1024 * 1024):
                assert time.monotonic() < DEADLINE, '55-second audit budget exhausted'
                h.update(block)
        HASHES[path] = h.hexdigest()
    return HASHES[path]


def ledger(root, expected):
    jobs = read(root / 'colabfold/jobs.json')
    statuses = dict(collections.Counter(j['status'] for j in jobs.values()))
    assert len(jobs) == expected and statuses == {'COMPLETE': expected}, statuses
    tool_commit = read(root / 'toolchain_manifest.json')['ColabFold']['commit']
    for key, job in jobs.items():
        sequence = job['sequence']
        assert key == sequence['sequence_id']
        assert hashlib.sha256(sequence['sequence'].encode()).hexdigest() == sequence['sequence_sha256']
        # Reused controls retain the original target and command fingerprint.
        target_hash = sha(job['target_backbone'])
        fingerprint = hashlib.sha256(json.dumps(dict(
            sequence=sequence['sequence_sha256'], target=target_hash,
            command=job['command'], tool_commit=tool_commit,
        ), sort_keys=True).encode()).hexdigest()
        assert fingerprint == job['fingerprint'], ('fingerprint', key)
        for filename, digest in job['files'].items():
            assert sha(filename) == digest, ('file hash', key, filename)
        points = []
        for line in pathlib.Path(job['pdb']).read_text().splitlines():
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                points.append([float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))])
        assert len(points) == sequence['length'] and all(math.isfinite(x) for p in points for x in p), key
        scores = read(pathlib.Path(job['score_json']))
        assert len(scores['plddt']) == sequence['length'], key
    return dict(completed=expected, remaining=0, statuses=statuses,
                categories=dict(collections.Counter(j['sequence']['source_category'] for j in jobs.values())),
                ledger=str(root / 'colabfold/jobs.json'), progress=str(root / 'progress.json'),
                logs=str(root / 'logs'), prediction_hashes_verified=True)


def main():
    start = time.monotonic()
    external = E015 / 'external_designability_loop_v1'
    result = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  e015=ledger(external, 82),
                  e007=ledger(COMPARISON / 'e007_inverse_folding', 62))
    for root in (external, COMPARISON):
        manifest = read(root / 'MANIFEST.json')
        files = manifest['files']
        if isinstance(files, dict):
            files = [dict(path=p, sha256=h) for p, h in files.items()]
        for item in files:
            assert sha(root / item['path']) == item['sha256'], ('manifest hash', item['path'])
    progress = read(COMPARISON / 'progress.json')
    assert progress['stage'] == 'REPORT_COMPLETE' and not progress['pending']
    assert progress['E007_epoch1_paused'] and progress['E015_training_paused']
    e007 = read(BASE / '.e014_worktree/e007_matched_fullscale_v1/STATE.json')
    assert e007['phase'] == 'paused_epoch1_comparative_evaluation'
    assert e007['total_exposures'] == 230132 and e007['update'] == 180817
    assert read(E015 / 'production/STATE.json')['phase'].startswith('paused')
    process_lines = subprocess.check_output(['ps', '-eo', 'pid,ppid,etime,args'], text=True).splitlines()
    names = ('colabfold_batch', 'protein_mpnn_run.py', 'run_e015_integrated.py',
             'supervise_e015_training.py', 'e007_epoch1_mission.py', 'e007_external_designability.py',
             'e015_external_designability.py', 'run_e007_fullscale.py', 'supervise_e007_epoch1_hold.py')
    workers = [line for line in process_lines if any(name in line for name in names)]
    assert not workers, workers
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                                   '--format=csv,noheader'], text=True, timeout=5).strip()
    assert not gpu, gpu
    locks = pathlib.Path('/proc/locks').read_text()
    ownership_locks = {}
    for path in (COMPARISON / 'PIPELINE_OWNER.lock', E015 / 'GPU_WORKER.lock',
                 E015 / 'SUPERVISOR.lock', BASE / '.e014_worktree/e007_matched_fullscale_v1/OWNER.lock',
                 BASE / '.e014_worktree/e007_matched_fullscale_v1/SUPERVISOR_OWNER.lock'):
        st = path.stat()
        held = []
        for line in locks.splitlines():
            fields = line.split()
            for field in fields:
                parts = field.split(':')
                if len(parts) == 3:
                    try:
                        identity = (int(parts[0], 16), int(parts[1], 16), int(parts[2]))
                    except ValueError:
                        continue
                    if identity == (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino):
                        held.append(line)
        assert not held, ('Ownership lock held', str(path), held)
        ownership_locks[str(path)] = 'unheld'
    controller = read(COMPARISON / 'AFTER_EPOCH_CONTROLLER.json')
    assert not pathlib.Path(f"/proc/{controller['pid']}").exists(), 'Historical controller PID exists; inspect identity'
    result.update(status='COMPLETE_VERIFIED_NO_RESTART', comparison_progress=str(COMPARISON / 'progress.json'),
                  previous_controller_pid=controller['pid'], previous_controller_exited=True,
                  current_worker_pid=None, gpu_compute_owners=[], ownership_locks=ownership_locks,
                  new_controller_needed=False, stale_running_entries=0,
                  e007_training_paused=True, e015_training_paused=True,
                  remaining_screening_seconds=0, prediction_files_preserved=True,
                  recovery_action='None: all refolds and comparison complete. Do not queue-after-epoch or resume training.',
                  stale_e015_progress_remaining=read(external / 'progress.json').get('remaining_refolds'),
                  completion_source='Verified colabfold/jobs.json and final comparison manifest; phase progress merges retain stale fields.',
                  verified_unique_files=len(HASHES), elapsed_seconds=time.monotonic() - start)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
