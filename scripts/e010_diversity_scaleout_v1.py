"""D:-only diversity pilot of the immutable 73f3330 privileged teacher.

No solver modifications, gold solves, training, or full-corpus generation.
"""
import os
import sys
from pathlib import Path

REPO = Path('/mnt/d/Simone/proteinGen')
ROOT = REPO / 'outputs/e010_structural_teacher_diversity_scaleout_v1'


def storage_environment():
    for key in ('TMPDIR', 'TMP', 'TEMP'):
        os.environ[key] = str(ROOT / 'tmp')
    for key in ('XDG_CACHE_HOME', 'MPLCONFIGDIR', 'CUDA_CACHE_PATH', 'TORCH_HOME'):
        os.environ[key] = str(ROOT / 'cache')
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    sys.dont_write_bytecode = True


storage_environment()
sys.path.insert(0, str(REPO / 'scripts'))
import run_e010_constraint_projected_v1 as frozen
storage_environment()
import argparse
import collections
import concurrent.futures
import hashlib
import json
import multiprocessing
import resource
import time

import numpy as np
import torch

pilot = frozen.pilot
CACHE = REPO / 'reports/experiments/E010_global_equivariant_expressivity/phase4b_real_denoiser_v1/phase4b_real_denoiser_v1.final'
CHECKPOINT = CACHE.parent / 'phase4b_training_v1.final/latest.pt'
TRAIN_MANIFEST = REPO / 'reports/experiments/E010_global_equivariant_expressivity/phase4a_multicorruption_v2/training_seed_manifest.json'
SPLITS = REPO / 'data/full/splits/split_assignments.parquet'
STRATA = ['20-64', '65-128', '129-256', '257-384', '385-500']
CONDITIONS = [50, 250, 450]
TEACHER_SHA = '0fb50ebe07d261d00845d46a43b0847aa3d855d4eec4250094cce80d06d12d33'
CONFIG = dict(experiment='e010_structural_teacher_diversity_scaleout_v1', parent_commit='73f3330',
    teacher_sha256=TEACHER_SHA, teacher_method='projection_2', teacher_solves=2, K=16,
    identities=2000, per_length_bin=400, conditions=CONDITIONS,
    namespace='e010_structural_teacher_diversity_scaleout_v1',
    selection='round-robin length/method/resolution cells; within cell round-robin sequence clusters; SHA256 identity ranking',
    portions='split-group SHA256 first 64 bits modulo 10: 0..5 reference, 6..7 holdout_A, 8..9 holdout_B; all identity conditions together',
    eligibility='TRAIN split and prior TRAIN source manifest; all 3 cache conditions; original source pins; all frozen Context gates at P0; no repair metric filtering',
    P0='unchanged Phase4B frozen global refiner, CPU float32 two-thread single-example inference over existing denoiser cache; teacher CPU float64 one thread',
    exact_zero_tail='stop at first bitwise exact-zero action; time-independent deterministic teacher gives identical future outputs',
    useful='nonzero action and normalized local-MSE rollout reduction >1e-6; reporting only, never alters teacher actions',
    horizons=[1, 2, 4, 8, 12, 16], prepare_workers=4, teacher_workers=8,
    decision=dict(hard_safety_violations_max=0, identity_useful_fraction_min=.70,
        every_length_condition_useful_fraction_min=.50, mean_relative_local_mse_gain_min=.05,
        every_length_condition_mean_relative_local_mse_gain_min=.01,
        projected_cached_eligible_corpus_wall_hours_max=24.,
        require_both_heldout_portions=True, no_automatic_student_training=True, no_full_corpus_generation=True),
    no_new_SLSQP=True, no_neural_training=True, GPU_used=False,
)
MODEL = None


def sha(path):
    return pilot.sha(path)


def rank(value):
    return hashlib.sha256(f"{CONFIG['namespace']}|{value}".encode()).hexdigest()


def make_dirs():
    assert str(ROOT.resolve()).startswith('/mnt/d/')
    for name in ('', 'tmp', 'cache', 'logs', 'inputs', 'trajectories', 'validation'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)


def acceptable_runner_sources():
    """Preserve completed actions from the pre-fix runner; teacher math is identical."""
    hashes = {sha(__file__)}
    amendment = ROOT / 'archive_validation_amendment.json'
    if amendment.exists():
        previous = json.loads(amendment.read_text())
        assert sha(previous['previous_runner_snapshot']) == previous['previous_runner_sha256']
        assert previous['teacher_sha256'] == TEACHER_SHA
        hashes.add(previous['previous_runner_sha256'])
    return hashes


def json_canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def initialize(with_model=False):
    global MODEL
    frozen.old.initialize()
    assert sha(frozen.__file__) == TEACHER_SHA
    storage_environment()
    if with_model:
        from protein_distance_diffusion.models.e010_hybrid_local import load_frozen_hybrid
        torch.set_default_dtype(torch.float32)
        try:
            torch.manual_seed(41047)
            MODEL = load_frozen_hybrid(CHECKPOINT).global_model.eval()
        finally:
            torch.set_default_dtype(torch.float64)


def pin_parent():
    assert sha(frozen.__file__) == TEACHER_SHA
    baseline = json.loads((frozen.ROOT / 'results.json').read_text())
    completion = json.loads((frozen.ROOT / 'COMPLETED.json').read_text())
    assert completion['status'] == 'complete'
    for path, digest in completion['files'].items():
        assert sha(path) == digest, path
    for path, digest in baseline['source_sha256'].items():
        assert sha(path) == digest, path
    artifact = dict(status='completed_frozen', commit='73f3330', do_not_optimize_gold_benchmark=True,
        provisional_teacher='projection_2', parent_completion_sha256=sha(frozen.ROOT / 'COMPLETED.json'),
        frozen_teacher_protocol_sha256=sha(frozen.ROOT / 'protocol.json'), teacher_source_sha256=TEACHER_SHA,
        scientific_result=dict(one_projection=[15.98, 19.07], one_relinearization=[39.21, 55.51],
            unsafe_gradient_rescues=[103, 252], safety_all_pass=True, archived_speedup_approx=1068,
            post_scalar_residual_active_span_percent=3.01,
            interpretation='relinearization improvement implicates nonlinear/changing constraint geometry'),
        exact_benchmark_summary=baseline['summary'])
    destination = ROOT / 'parent_freeze.json'
    if destination.exists():
        assert json.loads(destination.read_text()) == artifact
    else:
        pilot.atomic_json(destination, artifact)
    protocol = ROOT / 'protocol.json'
    if protocol.exists():
        assert json.loads(protocol.read_text()) == CONFIG
    else:
        pilot.atomic_json(protocol, CONFIG)


def input_data(row):
    result, target = [], None
    for record in row['examples']:
        path = CACHE / record['shard']
        with np.load(path, allow_pickle=False) as archive:
            lo, hi = archive['offsets'][record['index_in_shard']:record['index_in_shard'] + 2]
            source = archive['prediction'][lo:hi].copy()
            y = archive['target'][lo:hi].copy()
        assert pilot.digest(source) == record['prediction_sha256']
        assert pilot.digest(y) == record['target_sha256']
        assert source.shape == y.shape == (row['length'], 3)
        assert np.isfinite(source).all() and np.isfinite(y).all()
        if target is None:
            target = y
        else:
            assert np.array_equal(target, y), 'target changed across conditions'
        result.append(source)
    return np.stack(result), target


def prepare_candidate(row):
    destination = ROOT / 'inputs' / f"{row['sample_id']}.npz"
    marker = destination.with_suffix('.json')
    if marker.exists():
        result = json.loads(marker.read_text())
        assert result['candidate_sha256'] == rank(json.dumps(row, sort_keys=True))
        if result['eligible']:
            assert result['archive_sha256'] == sha(destination)
        return result
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    assert sha(REPO / row['source_path']) == row['source_sha256']
    try:
        with np.load(REPO / row['source_path'], allow_pickle=False) as archive:
            assert str(archive['sample_id'].item()) == row['sample_id']
            assert archive['ca_coordinates'].shape == (row['length'], 3)
            assert archive['residue_mask'].shape == (row['length'],) and archive['residue_mask'].all()
            assert np.isfinite(archive['ca_coordinates']).all()
        sources, target = input_data(row)
        mask = np.ones((1, row['length']), dtype=bool)
        predictions = []
        for source in sources:
            torch.set_num_threads(2)
            with torch.no_grad():
                pg = MODEL(torch.from_numpy(source)[None], torch.from_numpy(mask))['prediction'].numpy().copy()
            torch.set_num_threads(1)
            b = dict(pg=torch.from_numpy(pg.astype(np.float64)), source=torch.from_numpy(source[None].astype(np.float64)),
                     target=torch.from_numpy(target[None].astype(np.float64)), mask=torch.from_numpy(mask))
            ctx = pilot.Context(b)
            assert ctx.safety(b['pg'])['passes']
            assert float(ctx.baseline['local']) > 0 and float(ctx.baseline['chiral']) > 0 and float(ctx.baseline['aligned']) > 0
            assert pilot.local_representation(b['pg'], b['mask'])['eligible'].any()
            predictions.append(pg[0])
        frozen.old.atomic_npz(destination, P0=np.stack(predictions), source=sources, target=target, mask=mask[0],
                              conditions=np.array(CONDITIONS))
        result = dict(eligible=True, archive_sha256=sha(destination), P0_sha256=pilot.digest(np.stack(predictions)))
    except (ValueError, AssertionError, KeyError, OSError) as exc:
        result = dict(eligible=False, reason=f'{type(exc).__name__}: {exc}')
    result.update(identity=row['sample_id'], candidate_sha256=rank(json.dumps(row, sort_keys=True)),
        preparation_wall_seconds=time.perf_counter() - start_wall, preparation_cpu_seconds=time.process_time() - start_cpu)
    pilot.atomic_json(marker, result)
    return result


def candidate_order(rows):
    """Balanced metadata cells and cluster representatives without score filtering."""
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in rows:
        cells[row['metadata_cell']][row['cluster_id']].append(row)
    ordered_cells = {}
    for cell, clusters in cells.items():
        queues = [collections.deque(sorted(values, key=lambda r: (r['rank_sha256'], r['sample_id'])))
                  for _, values in sorted(clusters.items(), key=lambda kv: rank(f'cluster|{kv[0]}'))]
        order = []
        while any(queues):
            for queue in queues:
                if queue:
                    order.append(queue.popleft())
        ordered_cells[cell] = collections.deque(order)
    order = []
    while any(ordered_cells.values()):
        for cell in sorted(ordered_cells):
            if ordered_cells[cell]:
                order.append(ordered_cells[cell].popleft())
    return order


def prepare():
    import pandas as pd
    make_dirs()
    pin_parent()
    if (ROOT / 'selection.json').exists():
        validate_selection()
        return
    start = time.perf_counter()
    metadata = pd.read_parquet(SPLITS)
    assert metadata.sample_id.is_unique
    membership = dict(zip(metadata.sample_id, metadata.split))
    train = metadata[metadata.split == 'train'].set_index('sample_id')
    forbidden_groups = set(metadata[metadata.split != 'train'].split_group_id)
    original_panel = json.loads(Path('/home/simostocco/proteinGen-hybrid-local-global/reports/experiments/E010_global_equivariant_expressivity/phase4d_hybrid_local_global_v1/tiny_panel.json').read_text())
    excluded_panel_ids = {r['sample_id'] for r in original_panel['identities']}
    corpus = json.loads(TRAIN_MANIFEST.read_text())['identities']
    cache = json.loads((CACHE / 'manifest.json').read_text())
    chosen = {}
    for shard in cache['shards']:
        for index, record in enumerate(shard['records']):
            if record['split'] != 'train':
                continue
            key = (record['sample_id'], record['timestep'])
            if key not in chosen or (record['seed'], record['schedule_index']) < (chosen[key]['seed'], chosen[key]['schedule_index']):
                chosen[key] = dict(record, shard=shard['shard'], archive_sha256=shard['archive_sha256'], index_in_shard=index)
    candidates = []
    for row in corpus:
        sid = row['sample_id']
        if sid in excluded_panel_ids or membership.get(sid) != 'train' or any((sid, c) not in chosen for c in CONDITIONS):
            continue
        meta = train.loc[sid]
        assert meta.split_group_id not in forbidden_groups
        resolution = float(meta.resolution_angstrom)
        bucket = 'missing' if not np.isfinite(resolution) else 'le2' if resolution <= 2 else '2to3' if resolution <= 3 else 'gt3'
        examples = [chosen[(sid, c)] for c in CONDITIONS]
        assert all(r['length'] == row['length'] and r['source_sha256'] == row['source_sha256'] and r['mask_all_valid'] for r in examples)
        group_bucket = int(rank(f'portion|{meta.split_group_id}')[:16], 16) % 10
        candidates.append(dict(row, examples=examples, rank_sha256=rank(f'identity|{sid}'), cluster_id=str(meta.cluster_id),
            split_group_id=str(meta.split_group_id), sequence_hash=str(meta.sequence_hash), pdb_id=str(meta.pdb_id),
            experimental_method=str(meta.experimental_method), resolution_bucket=bucket,
            metadata_cell=f'{meta.experimental_method}|{bucket}', split='train',
            portion='reference' if group_bucket < 6 else 'holdout_A' if group_bucket < 8 else 'holdout_B'))
    shard_hashes = {r['shard']: r['archive_sha256'] for row in candidates for r in row['examples']}
    for filename, digest in shard_hashes.items():
        assert sha(CACHE / filename) == digest, filename
    pilot.atomic_json(ROOT / 'candidate_manifest.json', dict(candidates=candidates, excluded_prior_panel=sorted(excluded_panel_ids),
        input_sha256={str(p): sha(p) for p in (TRAIN_MANIFEST, SPLITS, CACHE / 'manifest.json', CHECKPOINT)},
        cache_shard_sha256=shard_hashes, structural_train_catalog_count=len(train),
        verified_train_source_cohort_count=len(corpus), all_condition_cached_count=len(candidates),
        candidates_by_length=dict(collections.Counter(r['stratum'] for r in candidates)),
        structural_train_by_length={s:int(((train.length >= int(s.split('-')[0])) & (train.length <= int(s.split('-')[1]))).sum()) for s in STRATA}))
    queues = {s:collections.deque(candidate_order([r for r in candidates if r['stratum'] == s])) for s in STRATA}
    selected, exclusions = [], []
    context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=context, initializer=initialize, initargs=(True,)) as pool:
        while any(sum(r['stratum'] == s for r in selected) < 400 for s in STRATA):
            batch = []
            for s in STRATA:
                remaining = 400 - sum(r['stratum'] == s for r in selected)
                for _ in range(min(remaining, 20)):
                    if not queues[s]:
                        raise RuntimeError(f'Insufficient teacher-compatible identities in {s}')
                    batch.append(queues[s].popleft())
            # Ordered map makes selection independent of worker completion order.
            for row, result in zip(batch, pool.map(prepare_candidate, batch)):
                if result['eligible']:
                    selected.append(dict(row, input_archive=str(ROOT / 'inputs' / f"{row['sample_id']}.npz"),
                        input_archive_sha256=result['archive_sha256']))
                else:
                    exclusions.append(result)
            pilot.atomic_json(ROOT / 'preparation_progress.json', dict(selected=len(selected), excluded=len(exclusions),
                by_length=dict(collections.Counter(r['stratum'] for r in selected)), elapsed_seconds=time.perf_counter() - start))
            print(f'SELECTION {len(selected)}/2000 excluded={len(exclusions)} elapsed={time.perf_counter()-start:.1f}', flush=True)
    selected.sort(key=lambda r: (STRATA.index(r['stratum']), r['sample_id']))
    for index, row in enumerate(selected):
        row['index'] = index
    pilot.atomic_json(ROOT / 'eligibility_exclusions.json', dict(exclusions=exclusions, no_repair_metric_filtering=True))
    manifest = dict(identities=selected, source_sha256=sha(__file__), protocol_sha256=sha(ROOT / 'protocol.json'),
        candidate_manifest_sha256=sha(ROOT / 'candidate_manifest.json'), preparation_wall_seconds=time.perf_counter()-start,
        conditions=CONDITIONS, identity_count=2000, trajectory_count=6000,
        selected_by_length=dict(collections.Counter(r['stratum'] for r in selected)),
        selected_by_portion=dict(collections.Counter(r['portion'] for r in selected)),
        unique_sequence_clusters=len({r['cluster_id'] for r in selected}), unique_split_groups=len({r['split_group_id'] for r in selected}))
    pilot.atomic_json(ROOT / 'selection.json', manifest)
    validate_selection()


def validate_selection():
    selection = json.loads((ROOT / 'selection.json').read_text())
    assert selection['source_sha256'] in acceptable_runner_sources()
    assert selection['protocol_sha256'] == sha(ROOT / 'protocol.json')
    assert selection['candidate_manifest_sha256'] == sha(ROOT / 'candidate_manifest.json')
    rows = selection['identities']
    assert len(rows) == len({r['sample_id'] for r in rows}) == 2000
    assert all(r['split'] == 'train' and [e['timestep'] for e in r['examples']] == CONDITIONS for r in rows)
    assert all(sum(r['stratum'] == s for r in rows) == 400 for s in STRATA)
    for group in {r['split_group_id'] for r in rows}:
        assert len({r['portion'] for r in rows if r['split_group_id'] == group}) == 1
    for row in rows:
        assert sha(row['input_archive']) == row['input_archive_sha256']
    return selection


def metrics(item, p):
    return pilot.scalar_metrics(item['ctx'], p)


def item_from_input(row, condition_index):
    with np.load(row['input_archive'], allow_pickle=False) as archive:
        b = {k:torch.from_numpy(archive[key][condition_index][None].astype(np.float64))
             for k, key in (('pg', 'P0'), ('source', 'source'))}
        b['target'] = torch.from_numpy(archive['target'][None].astype(np.float64))
        b['mask'] = torch.from_numpy(archive['mask'][None].copy())
    return dict(b=b, ctx=pilot.Context(b))


def validate_trajectory(row, marker):
    result = json.loads(marker.read_text())
    assert result['runner_sha256'] in acceptable_runner_sources()
    assert result['selection_sha256'] == sha(ROOT / 'selection.json')
    assert result['archive_sha256'] == sha(marker.with_suffix('.npz'))
    with np.load(marker.with_suffix('.npz'), allow_pickle=False) as archive:
        offsets = archive['offsets']
        assert offsets[-1] == len(archive['delta']) == len(archive['eligibility'])
        for i, condition in enumerate(CONDITIONS):
            item = item_from_input(row, i)
            p = item['b']['pg'].clone()
            stored = result['conditions'][str(condition)]
            assert item['ctx'].safety(p)['passes']
            assert len(stored['steps']) == offsets[i+1] - offsets[i]
            for t, (a, e) in enumerate(zip(archive['delta'][offsets[i]:offsets[i+1]], archive['eligibility'][offsets[i]:offsets[i+1]])):
                rep = pilot.local_representation(p, item['b']['mask'])['eligible'].numpy()
                assert np.array_equal(e, rep)
                checked = frozen.old.checks(item, p, a)
                assert json_canonical(checked['after']) == stored['steps'][t]['after']
                p = p + torch.from_numpy(a)
            if stored['first_zero_step'] is not None:
                assert (archive['delta'][offsets[i+1] - 1] == 0).all()
            assert json_canonical(metrics(item, p)) == stored['final']
    return result


def run_identity(row):
    destination = ROOT / 'trajectories' / f"identity_{row['index']:04d}.json"
    if destination.exists():
        begin_wall, begin_cpu = time.perf_counter(), time.process_time()
        validate_trajectory(row, destination)
        pilot.atomic_json(ROOT / 'validation' / destination.name, dict(passes=True, reused_checkpoint=True,
            archive_sha256=sha(destination.with_suffix('.npz')), record_sha256=sha(destination),
            wall_seconds=time.perf_counter()-begin_wall, cpu_seconds=time.process_time()-begin_cpu,
            validator_source_sha256=sha(__file__)))
        return row['index']
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    deltas, eligibility, offsets, results = [], [], [0], {}
    for i, condition in enumerate(CONDITIONS):
        item = item_from_input(row, i)
        p = item['b']['pg'].clone()
        baseline = metrics(item, p)
        steps = []
        first_zero = None
        for step in range(16):
            e = pilot.local_representation(p, item['b']['mask'])['eligible'].numpy().copy()
            before = metrics(item, p)
            a, cert = frozen.teacher(item, p, solves=2)
            checked = frozen.old.checks(item, p, a)
            p = p + torch.from_numpy(a)
            deltas.append(a)
            eligibility.append(e)
            norms = np.linalg.norm(a, axis=-1)[e]
            trials = cert['trials']
            chosen = next((dict(iteration=t['iteration'], scale=t['scale']) for t in trials
                if t['safety']['passes'] and t['local'] == cert['objective_after']), None)
            if cert['objective_after'] == cert['objective_before']:
                chosen = None
            step_record = dict(step=step+1, before=before, **checked, latency_seconds=cert['latency_seconds'],
                local_objective_before=cert['objective_before'], local_objective_after=cert['objective_after'],
                action_rms=float(np.sqrt(np.mean(norms**2))), action_mean=float(norms.mean()),
                action_norm_quantiles=np.quantile(norms, [0., .25, .5, .75, 1.]).tolist(),
                exact_zero=bool((a == 0).all()), shield=dict(safety_pass=True,
                    exact_checks=len(trials), useful_safe_trials=sum(t['useful'] for t in trials),
                    unsafe_trials=sum(not t['safety']['passes'] for t in trials),
                    accepted_proposal=chosen, exact_zero_fallback=bool((a == 0).all()),
                    convex_converged=[trials[k]['converged'] for k in (0, 4)],
                    solver_sweeps=[trials[k]['sweeps'] for k in (0, 4)],
                    chosen_candidates=[dict(iteration=t['iteration'], scale=t['scale']) for t in trials
                        if t['safety']['passes'] and t['local'] == cert['objective_after']]))
            steps.append(step_record)
            if step_record['exact_zero']:
                first_zero = step+1
                break
        offsets.append(len(deltas))
        results[str(condition)] = dict(baseline=baseline, final=metrics(item, p), steps=steps,
            evaluated_steps=len(steps), nonzero_steps=sum(not s['exact_zero'] for s in steps), first_zero_step=first_zero,
            implicit_zero_tail_steps=16-len(steps) if first_zero is not None else 0,
            horizons={str(k):steps[min(k, len(steps))-1]['after'] for k in CONFIG['horizons']})
    frozen.old.atomic_npz(destination.with_suffix('.npz'), delta=np.stack(deltas), eligibility=np.stack(eligibility),
        offsets=np.array(offsets), conditions=np.array(CONDITIONS))
    record = dict(index=row['index'], identity=row['sample_id'], length=row['length'], length_bin=row['stratum'],
        portion=row['portion'], metadata_cell=row['metadata_cell'], cluster_id=row['cluster_id'], split_group_id=row['split_group_id'],
        input_archive=row['input_archive'], input_archive_sha256=row['input_archive_sha256'],
        archive_sha256=sha(destination.with_suffix('.npz')), runner_sha256=sha(__file__), teacher_sha256=TEACHER_SHA,
        selection_sha256=sha(ROOT / 'selection.json'), conditions=results,
        worker_wall_seconds=time.perf_counter()-start_wall, worker_cpu_seconds=time.process_time()-start_cpu,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    pilot.atomic_json(destination, record)
    validation_start = time.perf_counter()
    validate_trajectory(row, destination)
    record['archive_validation_wall_seconds'] = time.perf_counter() - validation_start
    record['worker_wall_seconds'] = time.perf_counter() - start_wall
    record['worker_cpu_seconds'] = time.process_time() - start_cpu
    record['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    pilot.atomic_json(destination, record)
    pilot.atomic_json(ROOT / 'validation' / destination.name, dict(passes=True, reused_checkpoint=False,
        archive_sha256=sha(destination.with_suffix('.npz')), record_sha256=sha(destination),
        wall_seconds=record['archive_validation_wall_seconds'], validator_source_sha256=sha(__file__)))
    return row['index']


def run():
    make_dirs()
    pin_parent()
    selection = validate_selection()
    # Source model is not loaded during teacher execution; only prepared P0 is read.
    start = time.perf_counter()
    marker = ROOT / 'run_history.json'
    history = json.loads(marker.read_text()) if marker.exists() else []
    entry = dict(start_unix=time.time(), workers=8, runner_sha256=sha(__file__),
        previous_completed=len(list((ROOT / 'trajectories').glob('*.json'))))
    history.append(entry)
    pilot.atomic_json(marker, history)
    rows = sorted(selection['identities'], key=lambda r: (-r['length'], r['index']))
    context = multiprocessing.get_context('spawn')
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=8, mp_context=context, initializer=initialize)
    try:
        futures = [pool.submit(run_identity, row) for row in rows]
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            index = future.result()
            completed += 1
            if completed % 10 == 0 or completed == 2000:
                pilot.atomic_json(ROOT / 'progress.json', dict(completed=completed, total=2000, last_index=index,
                    elapsed_seconds=time.perf_counter()-start))
                print(f'TEACHER {completed}/2000 elapsed={time.perf_counter()-start:.1f}', flush=True)
    except BaseException:
        # Stop this launcher's own children immediately; retain all durable checkpoints.
        for process in list(pool._processes.values()):
            process.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        entry.update(status='failed', wall_seconds=time.perf_counter()-start, finished_unix=time.time())
        pilot.atomic_json(marker, history)
        raise
    else:
        pool.shutdown(wait=True)
    entry['wall_seconds'] = time.perf_counter()-start
    entry['finished_unix'] = time.time()
    entry['status'] = 'complete'
    pilot.atomic_json(marker, history)
    print('SCALEOUT_TEACHER_COMPLETE', flush=True)


def test():
    initialize(with_model=True)
    # Live P0 parity against already-frozen cached inputs; no gold optimizer calls.
    data, _ = pilot.load()
    outcomes = []
    for index in (0, 3, 30):
        item = data[index]
        source = item['b']['source'].float()
        torch.set_num_threads(2)
        with torch.no_grad():
            actual = MODEL(source, item['b']['mask'])['prediction']
        torch.set_num_threads(1)
        assert torch.equal(actual.double(), item['b']['pg'])
        action, _ = frozen.teacher(item, item['b']['pg'], solves=2)
        with np.load(frozen.ROOT / 'trajectories' / f'trajectory_{index:02d}.npz', allow_pickle=False) as z:
            assert np.array_equal(action, z['projection_2_one_0'])
        frozen.old.checks(item, item['b']['pg'], action)
        if (action == 0).all():
            repeated, _ = frozen.teacher(item, item['b']['pg'], solves=2)
            assert np.array_equal(action, repeated)
        outcomes.append(dict(index=index, input_parity=True, action_parity=True))
    pilot.atomic_json(ROOT / 'tests.json', dict(passed=True, frozen_P0_and_action_parity=outcomes,
        teacher_sha256=sha(frozen.__file__), GPU_used=False, optimizer_calls=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'test', 'run'])
    args = parser.parse_args()
    make_dirs()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'test':
        pin_parent()
        test()
    else:
        assert json.loads((ROOT / 'tests.json').read_text())['passed']
        run()


if __name__ == '__main__':
    main()
