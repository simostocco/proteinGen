"""Infrastructure-only recovery of the sealed E017 V9.5 schedule.

No sampler, reward, filtering, goal, checkpoint, or statistical implementation.
All inference uses the frozen evaluate.publication entrypoint and common.load.
"""
import collections
import concurrent.futures
import copy
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path('/mnt/d/Simone/proteinGen_data/e017_v95_expanded_validation_20261028_v1')
sys.path.insert(0, str(ROOT / 'source'))
import common as C
import evaluate as E

REC = ROOT / 'recovery_20261029_v1'


def write(name, obj):
    C.atomic(REC / name, obj)


def schedule(m):
    return [(arm, policy, update, i, task, seeds)
            for arm, policy, update in C.ARMS
            for i, task, seeds in C.specs(m, arm)]


def key(spec):
    return f'{spec[0]}_{spec[3]:05d}'


def suffix(plan, present):
    keys = [key(s) for s in plan]
    assert len(keys) == len(set(keys))
    assert not present - set(keys), 'UNKNOWN_BATCH'
    frontier = next((n for n, k in enumerate(keys) if k not in present), len(keys))
    assert set(keys[:frontier]) == present, 'NONCONTIGUOUS_COMMITTED_FRONTIER'
    return frontier, plan[frontier:]


def audit(m, original=None):
    plan = schedule(m)
    lookup = {key(s): s for s in plan}
    qualification = C.read(ROOT / 'engineering/QUALIFICATION.json')
    paths = sorted((ROOT / 'commits').glob('*.json'))

    def check(p):
        arm, policy, update, i, task, seeds = lookup[p.stem]
        fp = qualification['trained_model' if update else 'initial_model']['actor_fingerprint']
        seal = C.read(p)
        assert seal['path'] == f'results/{p.stem}.json.gz'
        assert (seal['arm'], seal['index'], seal['seeds'], seal['actor_fingerprint']) == (arm, i, seeds, fp)
        result = ROOT / seal['path']
        assert C.sha(result) == seal['SHA256'], ('CORRUPT_BATCH', p.stem)
        obj = C.read(result)
        E.validate(obj, task, seeds, fp, policy)
        assert obj['arm'] == arm and obj['index'] == i
        counts = collections.Counter(batches=1, attempts=8)
        for row in obj['records']:
            n = len(row['sequence'])
            assert row['sequence'] == task['fixed_sequence'][:n]
            complete = row['metrics']['complete']
            aborted = row['termination'] is not None
            assert int(complete) + int(aborted) == 1
            counts['completed'] += int(complete)
            counts['aborted'] += int(aborted)
        return p.stem, dict(commit_SHA256=C.sha(p), result_SHA256=seal['SHA256'], UTC=seal['UTC']), counts, arm

    inventory = {}
    totals = collections.Counter()
    arms = collections.defaultdict(collections.Counter)
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for n, (name, hashes, counts, arm) in enumerate(pool.map(check, paths), 1):
            assert name not in inventory
            inventory[name] = hashes
            totals.update(counts)
            arms[arm].update(counts)
            if n % 1000 == 0:
                print('AUDIT_VERIFIED', n, flush=True)
    present = set(inventory)
    frontier, missing = suffix(plan, present)
    assert {p.name for p in (ROOT / 'results').iterdir()} == {k + '.json.gz' for k in present}, 'ORPHAN_OR_PARTIAL_RESULT'
    assert {p.name for p in (ROOT / 'commits').iterdir()} == {k + '.json' for k in present}
    if original is not None:
        for name, hashes in original.items():
            assert inventory[name] == hashes, ('ORIGINAL_COMMIT_CHANGED', name)
    return dict(PASS=True, UTC=C.now(), totals=dict(totals), arms={k: dict(v) for k, v in arms.items()},
                frontier=frontier, next_batch=key(missing[0]) if missing else None,
                missing_batches=len(missing), missing_by_arm=dict(collections.Counter(s[0] for s in missing)),
                inventory=inventory)


def locks():
    handles = []
    for name in ['RECOVERY.lock', 'EVALUATION.lock']:
        handle = (ROOT / 'runtime' / name).open('a+')
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        handles.append(handle)
    return handles


def infrastructure_tests(m):
    fixture = REC / 'fixtures'
    fixture.mkdir()
    (fixture / 'engineering').mkdir()
    base = C.read(ROOT / 'engineering/TRAIN_filtered_00000.json.gz')
    task, seeds, fp = base['task'], base['seeds'], base['actor_fingerprint']
    calls = []
    class Engine:
        calls = {}
    def sample(e, t, policy, count, ss):
        assert t == task and count == 8 and policy == 'C'
        calls.append(list(ss))
        records = copy.deepcopy(base['records'])
        for j, row in enumerate(records):
            row['seed'] = ss[j]
        return records
    original_root = E.R
    checks = []
    try:
        E.R = fixture
        fake = {'sample': sample}
        obj = E.publication(Engine(), fake, task, seeds, 'cursor', 'C', 0, fp, train=True)
        original_hash = C.sha(fixture / 'engineering/cursor_00000.json.gz')
        E.publication(Engine(), fake, task, seeds, 'cursor', 'C', 0, fp, train=True)
        assert len(calls) == 1 and original_hash == C.sha(fixture / 'engineering/cursor_00000.json.gz')
        checks.append('committed publication reused without inference or mutation')
        orphan = copy.deepcopy(obj)
        orphan['arm'] = 'adopt'
        C.atomic(fixture / 'engineering/adopt_00000.json.gz', orphan)
        E.publication(None, {'sample': lambda *a: (_ for _ in ()).throw(AssertionError('INFERENCE'))}, task, seeds, 'adopt', 'C', 0, fp, train=True)
        assert len(calls) == 1
        checks.append('finalized interrupted publication adopted without generation')
        partial = fixture / 'engineering/partial_00001.json.gz.part'
        partial.write_bytes(b'interrupted gzip fixture')
        next_seeds = [C.seed('TRAIN recovery fixture', j) for j in range(8)]
        E.publication(Engine(), fake, task, next_seeds, 'partial', 'C', 1, fp, train=True)
        assert not partial.exists() and calls == [seeds, next_seeds]
        checks.append('partial output ignored; exact requested seeds atomically published once')
        bad = copy.deepcopy(obj)
        bad['seeds'] = list(seeds)
        bad['seeds'][0] += 1
        try:
            E.validate(bad, task, seeds, fp, 'C')
        except AssertionError:
            checks.append('mismatched seed binding rejected')
        else:
            raise AssertionError('BAD_SEED_ACCEPTED')
        aborted = copy.deepcopy(obj)
        row = aborted['records'][0]
        row['sequence'] = row['sequence'][:4]
        row['geometry'] = row['geometry'][:4]
        row['termination'] = 'synthetic finite proposal exhaustion'
        row['metrics'].update(complete=False, success=False, hard_valid=False, backbone_valid=False)
        E.validate(aborted, task, seeds, fp, 'C')
        assert sum(x['metrics']['complete'] for x in aborted['records']) + sum(x['termination'] is not None for x in aborted['records']) == 8
        row['metrics']['success'] = True
        try:
            E.validate(aborted, task, seeds, fp, 'C')
        except AssertionError:
            checks.append('aborted attempts retained and successful abort rejected')
        else:
            raise AssertionError('SUCCESSFUL_ABORT_ACCEPTED')
    finally:
        E.R = original_root
    plan = schedule(m)
    frontier, remaining = suffix(plan, {key(s) for s in plan[:6709]})
    assert frontier == 6709 and key(remaining[0]) == 'trained_raw_01589' and len(remaining) == 1995
    assert [key(s) for s in plan[:frontier] + remaining] == [key(s) for s in plan]
    assert suffix(plan, {key(s) for s in plan}) == (8704, [])
    try:
        suffix(plan, {key(s) for s in plan[:6709]} | {key(plan[6710])})
    except AssertionError:
        checks.append('noncontiguous cursor rejected; no skipped or duplicated schedule entries')
    else:
        raise AssertionError('SCHEDULE_HOLE_ACCEPTED')
    assert C.read(ROOT / 'engineering/REPORT_ENTRYPOINT_SMOKE.json')['source_SHA256'] == C.sha(ROOT / 'source/analyse.py')
    assert C.read(ROOT / 'engineering/STATISTICAL_SELF_TESTS.json')['PASS']
    assert C.read(ROOT / 'engineering/TERMINAL_COUNTER_SELF_TEST.json')['source_SHA256'] == C.sha(ROOT / 'engineering/seal_terminal.py')
    checks.append('frozen report and terminal-counter qualification matches current source')
    return checks


def qualify():
    assert not REC.exists(), 'RECOVERY_DIRECTORY_ALREADY_EXISTS'
    REC.mkdir()
    m = C.frozen()
    held = locks()
    for name in ['EVALUATION_COMPLETED.json', 'ANALYSIS_COMPLETED.json', 'WORKER_EXIT.json', 'PROVENANCE_SHA256.json']:
        assert not (ROOT / name).exists(), name
    assert not list((ROOT / 'runtime').glob('*FAILURE*'))
    assert not (C.R3 / 'production/STOP_RECORD.json').exists()
    assert shutil.disk_usage(ROOT).free > m['min_free_launch_bytes']
    # Preserve exact bytes of the interrupted run before changing runtime records.
    saved = REC / 'original_interruption'
    saved.mkdir()
    originals = ['EVALUATION_LOG.txt', 'LAUNCH.json', 'runtime/LATEST_STATUS.json', 'runtime/WORKER.json', 'runtime/TERMINAL_SEAL_LAUNCH.json', 'runtime/TERMINAL_SEAL_LOG.txt']
    for name in originals:
        shutil.copyfile(ROOT / name, saved / name.replace('/', '__'))
    evidence = {}
    for label, command in {
        'previous_boot_tail': ['journalctl', '-b', '-1', '--utc', '--no-pager', '-n', '30'],
        'previous_kernel_failure_search': ['journalctl', '-b', '-1', '-k', '--utc', '--no-pager', '--grep=Out of memory|oom-kill|Killed process|NVRM|Xid|I/O error|EXT4-fs error'],
        'boots': ['journalctl', '--list-boots', '--utc', '--no-pager'],
        'kernel_current': ['dmesg', '--time-format', 'iso'],
        'gpu': ['nvidia-smi'],
    }.items():
        result = subprocess.run(command, capture_output=True, text=True)
        (REC / (label + '.txt')).write_text(result.stdout + result.stderr)
        evidence[label] = dict(returncode=result.returncode, path=label+'.txt')
    write('TERMINATION_INVESTIGATION.json', dict(UTC=C.now(), classification='E_UNRESOLVED_EXACT_CAUSE',
          evidence=evidence, conclusion='Previous boot journal ends near frontier; current boot has unclean shutdown recovery. External interruption plausible, exact trigger unproven. No evaluator failure record or registered stop.', TEST_content_read=False))
    result = audit(m)
    assert result['totals'] == dict(batches=6709, attempts=53672, completed=53279, aborted=393)
    assert result['next_batch'] == 'trained_raw_01589' and result['missing_by_arm'] == {'trained_raw':203, 'trained_filtered':1792}
    write('PRE_RESUME_INTEGRITY.json', result)
    checks = infrastructure_tests(m)
    for spec in m['models'].values():
        assert C.sha(spec['path']) == spec['SHA256'], ('CHECKPOINT_CHANGED', spec['path'])
    for path, h in m['data_SHA256'].items():
        assert C.sha(path) == h
    P, F, compatibility = C.load()
    contract = C.read(C.Q / 'COMPATIBILITY_CONTRACT.json')
    assert P['bindings']() == F['P']['bindings']() == contract['original_binding']
    checks.append('current live source/checkpoint/data/reward binding contract passes unchanged')
    # CPU-only checkpoint tensor equality and serialized binding checks.
    import torch
    a = torch.load(C.R3 / 'production/checkpoints/milestone_05000.pt', map_location='cpu', weights_only=False)
    b = torch.load(C.R3 / 'production/checkpoints/recovery_0.pt', map_location='cpu', weights_only=False)
    assert a['binding'] == b['binding'] == P['bindings']()
    assert a['actor'].keys() == b['actor'].keys()
    assert all(torch.equal(a['actor'][k], b['actor'][k]) for k in a['actor'])
    checks.append('final milestone and recovery actor tensors/buffers exactly equal')
    write('RECOVERY_QUALIFICATION.json', dict(PASS=True, UTC=C.now(), checks=checks,
          manifest_SHA256=C.sha(ROOT / 'VALIDATION_PANEL_MANIFEST.json'), infrastructure_SHA256=C.sha(__file__),
          scientific_source_SHA256=m['source_SHA256'], missing_batches=1995, missing_attempts=15960,
          next_batch='trained_raw_01589', VALIDATION_forward_passes=0, TRAIN_forward_passes=0,
          optimizer_updates=0, TEST=False, original_results_preserved=True))
    print('RECOVERY_QUALIFICATION_PASS', checks, flush=True)


def generate():
    held = locks()
    m = C.frozen()
    q = C.read(REC / 'RECOVERY_QUALIFICATION.json')
    assert q['PASS'] and q['infrastructure_SHA256'] == C.sha(__file__)
    assert not (ROOT / 'EVALUATION_COMPLETED.json').exists()
    original = C.read(REC / 'PRE_RESUME_INTEGRITY.json')
    current = audit(m, original['inventory'])
    assert current['totals'] == original['totals'], 'FRONTIER_CHANGED_SINCE_QUALIFICATION'
    plan = schedule(m)
    frontier, todo = suffix(plan, set(current['inventory']))
    assert len(todo) == 1995 and all(s[2] == 5000 for s in todo)
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], check=True, capture_output=True, text=True)
    assert not gpu.stdout.strip(), 'CONFLICTING_GPU_WORKER'
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    def forbidden(*a, **kw):
        raise AssertionError('OPTIMIZER_FORBIDDEN')
    torch.optim.Adam = torch.optim.AdamW = forbidden
    P, F, compatibility = C.load()
    engine, _ = P['K']['load_engine']('C')
    engine.actor.eval().requires_grad_(False)
    metadata = F['checkpoint'](engine, 5000)
    fp = metadata['actor_fingerprint']
    assert fp == C.read(ROOT / 'engineering/QUALIFICATION.json')['trained_model']['actor_fingerprint']
    started = C.now()
    t0 = time.monotonic()
    old_status = C.read(REC / 'original_interruption/runtime__LATEST_STATUS.json')
    original_seconds = old_status['elapsed_seconds'] + (datetime.datetime.fromisoformat(max(v['UTC'] for v in original['inventory'].values())) - datetime.datetime.fromisoformat(old_status['UTC'])).total_seconds()
    new = 0
    last_progress = 0
    write('RESUME_STARTED.json', dict(UTC=started, PID=os.getpid(), original_attempts=53672, next_batch=key(todo[0]), planned_new_attempts=15960, actor=metadata, original_active_seconds=original_seconds, TEST=False, TRAINING_UPDATES=0))
    C.atomic(ROOT / 'runtime/WORKER.json', dict(PID=os.getpid(), UTC=started, status='RESUMED', manifest_SHA256=q['manifest_SHA256'], next_batch=key(todo[0])))
    def interrupted(signum, frame):
        raise RuntimeError(f'EXTERNAL_SIGNAL_{signum}')
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, interrupted)
    for arm, policy, update, i, task, seeds in todo:
        assert not (ROOT / 'commits' / f'{arm}_{i:05d}.json').exists(), 'DUPLICATE_ATTEMPT_PREVENTED'
        assert not (ROOT / 'results' / f'{arm}_{i:05d}.json.gz').exists(), 'UNEXPECTED_ORPHAN_RESULT'
        assert not (ROOT / 'runtime/RECOVERY_STOP').exists(), 'REQUESTED_RECOVERY_STOP'
        assert shutil.disk_usage(ROOT).free > m['operational_reserve_bytes'] + 256*2**20, 'STORAGE_RESERVE'
        assert original_seconds + time.monotonic()-t0 < m['max_wall_hours']*3600, 'PREDECLARED_ACTIVE_WALL_LIMIT'
        obj = E.publication(engine, P, task, seeds, arm, policy, i, fp)
        new += 8
        elapsed = time.monotonic()-t0
        if elapsed-last_progress >= 60 or new == 8 or new == 15960:
            last_progress = elapsed
            status = dict(status='RUNNING_RESUMED', UTC=C.now(), PID=os.getpid(), arm=arm,
                          batch_index=i, groups=frontier+new//8, total_groups=8704,
                          generation_attempts=53672+new, target_attempts=69632,
                          new_attempts=new, resumed_seconds=elapsed, original_active_seconds=original_seconds,
                          D_free_GiB=shutil.disk_usage(ROOT).free/2**30)
            C.atomic(ROOT / 'runtime/LATEST_STATUS.json', status)
            print(json.dumps(status), flush=True)
        # Every batch is independently durable; actor/parent verified at arm boundaries.
        if i == 1791:
            engine.prior.verify()
            assert P['c']['state_digest'](engine.parent.state_dict()) == engine.fp
            assert P['c']['state_digest'](engine.actor.state_dict()) == fp
    assert new == 15960
    assert all(p.grad is None and not p.requires_grad for model in [engine.actor, engine.parent, engine.prior.model] for p in model.parameters())
    final = audit(m, original['inventory'])
    assert final['totals']['attempts'] == 69632 and final['missing_batches'] == 0
    write('FINAL_BATCH_INTEGRITY.json', final)
    for spec in m['models'].values():
        assert C.sha(spec['path']) == spec['SHA256']
    C.frozen()
    write('GENERATION_RECOVERY_COMPLETED.json', dict(PASS=True, UTC=C.now(), interrupted=True, original_attempts=53672, new_attempts=new, final_attempts=69632, resumed_seconds=time.monotonic()-t0, original_results_unchanged=True, model_files_unchanged=True, actor_tensors_unchanged=True, J1_frozen=True, TEST=False, TRAINING_UPDATES=0))
    done = dict(status='GENERATION_COMPLETED_ANALYSIS_PENDING', UTC=C.now(), PID=os.getpid(), groups=8704, attempts=69632,
                seconds=original_seconds+time.monotonic()-t0, started=C.read(ROOT/'runtime/WORKER.json')['UTC'],
                original_started=C.read(REC/'original_interruption/runtime__WORKER.json')['UTC'], resumed=True,
                original_attempts=53672, new_attempts=new, manifest_SHA256=q['manifest_SHA256'], TRAINING_UPDATES=0, TEST=False)
    C.atomic(ROOT / 'EVALUATION_COMPLETED.json', done)
    C.atomic(ROOT / 'runtime/LATEST_STATUS.json', done)
    print('RESUMED_GENERATION_COMPLETED', new, flush=True)


def supervise():
    exitcodes = {}
    for label, script, args in [('generation', Path(__file__), ['generate']), ('analysis', ROOT/'source/analyse.py', [])]:
        with (REC / (label.upper()+'_LOG.txt')).open('x') as log:
            result = subprocess.run([str(ROOT/'run.sh'), str(script), *args], stdout=log, stderr=subprocess.STDOUT)
        exitcodes[label] = result.returncode
        write(label.upper()+'_EXIT.json', dict(UTC=C.now(), exit_code=result.returncode))
        assert result.returncode == 0, f'{label.upper()}_FAILED'
    C.atomic(ROOT/'WORKER_EXIT.json', dict(UTC=C.now(), PID=os.getpid(), exit_code=0, status='COMPLETED', child_exitcodes=exitcodes, generation_attempts=69632, resumed=True, TRAINING_UPDATES=0, TEST=False))
    write('RECOVERY_EXECUTION_HISTORY.json', dict(UTC=C.now(), original_termination_cause='UNRESOLVED; external interruption plausible from boot evidence', original_attempts=53672, resumed_new_attempts=15960, final_attempts=69632, original_results_preserved=True, resumed=True, analysis_source_unchanged=True, child_exitcodes=exitcodes, TEST_seal='UNKNOWN_HISTORICAL_ACCESS_AUDIT_UNAVAILABLE', TEST_content_read=False, TRAINING_UPDATES=0))
    (REC/'RECOVERY_HISTORY.md').write_text('# Interrupted evaluation recovery\n\nThe original evaluation stopped after 53,672 committed attempts. Its exact termination cause remains unresolved; boot evidence supports an external interruption. The original log and status bytes are preserved in original_interruption/.\n\nRecovery began at zero-based trained_raw batch 1589 and generated only the remaining 1,995 batches (15,960 attempts). All original commitments and result hashes were reverified unchanged. Frozen scientific sources, checkpoint files, goals, seeds, filters, reward, and statistical implementation were preserved. No neural training occurred. This was not an uninterrupted run.\n\nThe original registered analysis was executed only after all 69,632 attempts passed the complete batch audit. TEST was not accessed; complete historical access evidence is unavailable.\n')


def pipeline():
    handle = (ROOT/'runtime/RECOVERY_PIPELINE.lock').open('a+')
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    worker = subprocess.Popen([str(ROOT/'run.sh'), str(Path(__file__)), 'supervise'])
    write('SUPERVISOR_LAUNCH.json', dict(UTC=C.now(), PID=worker.pid, pipeline_PID=os.getpid(), no_automatic_restart=True))
    code = worker.wait()
    assert code == 0, 'RECOVERY_SUPERVISOR_FAILED'
    # Parent is reaped and all production/report logs are now closed.
    with (ROOT/'runtime/RECOVERY_TERMINAL_LOG.txt').open('x') as log:
        result = subprocess.run([str(ROOT/'run.sh'), str(ROOT/'engineering/seal_terminal.py')], stdout=log, stderr=subprocess.STDOUT)
    assert result.returncode == 0, 'TERMINAL_PUBLICATION_FAILED'
    status = C.read(ROOT/'runtime/TERMINAL_SEAL_STATUS.json')
    assert status['status'] == 'COMPLETED' and status['PASS']
    C.atomic(ROOT/'runtime/RECOVERY_PIPELINE_EXIT.json', dict(UTC=C.now(), exit_code=0, status='COMPLETED', attempts=69632, TRAINING_UPDATES=0))


if __name__ == '__main__':
    mode = sys.argv[1]
    try:
        {'qualify':qualify, 'generate':generate, 'supervise':supervise, 'pipeline':pipeline}[mode]()
    except BaseException as error:
        if REC.exists():
            C.atomic(ROOT/'runtime'/f'RECOVERY_FAILURE_{mode}_{os.getpid()}.json', dict(UTC=C.now(), PID=os.getpid(), error=repr(error), traceback=traceback.format_exc(), no_automatic_restart=True))
        raise
