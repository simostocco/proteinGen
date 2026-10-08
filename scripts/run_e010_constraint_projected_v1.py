"""Frozen constraint-span diagnostic and bounded convex privileged projection.

All artifacts live on D:. Historical modules are read-only; no gold solves.
"""
import os
import sys
from pathlib import Path

ROOT = Path('/mnt/d/Simone/proteinGen/outputs/fast_privileged_constraint_projected_teacher_v1')
for key in ('TMPDIR', 'TMP', 'TEMP'):
    os.environ[key] = str(ROOT / 'tmp')
for key in ('XDG_CACHE_HOME', 'MPLCONFIGDIR', 'CUDA_CACHE_PATH'):
    os.environ[key] = str(ROOT / 'cache')
os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import run_e010_privileged_gradient_v1 as old

# Importing the historical benchmark changes environment paths; restore ours.
for key in ('TMPDIR', 'TMP', 'TEMP'):
    os.environ[key] = str(ROOT / 'tmp')
for key in ('XDG_CACHE_HOME', 'MPLCONFIGDIR', 'CUDA_CACHE_PATH'):
    os.environ[key] = str(ROOT / 'cache')
import argparse
import concurrent.futures
import json
import subprocess
import time

import numpy as np
import torch
from scipy.optimize import nnls

pilot = old.pilot
METHODS = ['projection_1', 'projection_2']
CONFIG = dict(
    experiment='fast_privileged_constraint_projected_teacher_v1', gold_states=278, trajectories=20,
    active_normalized_margin=1e-5, near_active_normalized_margin=.01,
    ball_active_margin_angstrom=1e-5, ball_near_margin_angstrom=.001,
    desired='per-residue normalized negative exact local gradient, 0.10 A; eligibility zeros',
    solver='Dykstra Euclidean projection onto linear halfspaces and product of residue balls',
    maximum_convex_sweeps=128, iterate_tolerance=1e-9, linear_feasibility_tolerance=1e-9,
    global_linear_bound=0., quartet_linear_bound=0., action_cap_angstrom=.10,
    trust_region='same per-residue 0.10 A total-action ball about input; no extra trust parameter',
    solves={'projection_1': 1, 'projection_2': 2},
    relinearization='at first convex endpoint; same desired action and original total-action balls; at most one',
    backtracking=[1., .5, .25, .125], fallback='exact zero; retain best exact-safe local-nonworsening candidate',
    rollout_horizon=16, retention='ratio of absolute mean-local-RMSE repair sums, matched durable gold horizon',
    span='fit nonnegative scalar alpha to gold against eligible unnormalized -g; then project residual onto inward active rows',
    variance='uncentered action energy (not centered statistical variance); zero gold excluded from cosine only',
    multipliers='NNLS residual fit to inward rows -dc/dP, lambda>=0; equality eligibility unrestricted',
    ball_diagnostic='origin ball Jacobian is zero; endpoint gold ball normals reported separately, inward -delta_gold/norm',
    diagnostic_not_exact_KKT=True, no_new_gold_solves=True, no_neural_training=True, workers=4,
)


def linearize(item, p):
    x = p.detach().clone().requires_grad_(True)
    with pilot.float64_defaults():
        values = item['ctx'].values(x)
        c = item['ctx'].constraints(x)
        g = torch.autograd.grad(values['local'], x, retain_graph=True)[0]
        first = torch.stack([torch.autograd.grad(v, x, retain_graph=True)[0] for v in c[:2]])
        rest = torch.func.jacrev(item['ctx'].quartets)(x.detach())
    j = torch.cat((first, rest)).detach().numpy().reshape(len(c), -1)
    families = ['aligned_rmsd', 'continuous_chirality']
    for name, (lo, hi) in item['ctx'].quartets.slices.items():
        families.extend([name] * (hi - lo))
    assert len(families) == len(c) and np.isfinite(j).all()
    return g.detach().numpy(), c.detach().numpy(), j, np.array(families)


def convex_projection(desired, c, j, eligible, offset):
    """Bounded Dykstra, with exact halfspace and product-ball proximal maps.

    Drop only halfspaces proved unreachable inside the action ball using their
    support function. Eligibility is eliminated algebraically.
    """
    shape = desired.shape
    columns = np.repeat(eligible.reshape(-1), 3)
    a = j[:, columns]
    b = -c + j @ offset.reshape(-1)
    reachable = .1 * np.linalg.norm(a.reshape(len(a), -1, 3), axis=-1).sum(-1)
    keep = (b < reachable) & (np.linalg.norm(a, axis=-1) > 1e-14)
    a, b = a[keep], b[keep]
    norm = np.linalg.norm(a, axis=-1)
    a = a / norm[:, None]
    b = b / norm
    x = desired.reshape(-1)[columns].copy()
    dual = np.zeros(len(a))
    ball_dual = np.zeros_like(x)
    converged = False
    for sweep in range(CONFIG['maximum_convex_sweeps']):
        previous = x.copy()
        for k in range(len(a)):
            y = x + dual[k] * a[k]
            dual[k] = max(0., float(a[k] @ y - b[k]))
            x = y - dual[k] * a[k]
        y = x + ball_dual
        residue = y.reshape(-1, 3)
        size = np.linalg.norm(residue, axis=-1, keepdims=True)
        x = (residue * np.minimum(1., .1 / np.maximum(size, 1e-300))).reshape(-1)
        ball_dual = y - x
        violation = max(0., float(np.max(a @ x - b, initial=0.)))
        if np.linalg.norm(x - previous) <= CONFIG['iterate_tolerance'] and violation <= CONFIG['linear_feasibility_tolerance']:
            converged = True
            break
    full = np.zeros(desired.size)
    full[columns] = x
    action, _ = pilot.project(full.reshape(shape))
    return action, dict(sweeps=sweep + 1, converged=converged, selected_halfspaces=len(a),
                        normalized_linear_violation=violation,
                        original_linear_violation=float(np.max(c + j @ (action - offset).reshape(-1), initial=0.)),
                        projection_distance=float(np.linalg.norm(action - desired)))


def teacher(item, origin, solves, prepared=None):
    started = time.perf_counter()
    eligible = pilot.local_representation(origin, item['b']['mask'])['eligible'].numpy()
    g, c, j, _ = prepared if prepared is not None else linearize(item, origin)
    size = np.linalg.norm(g, axis=-1, keepdims=True)
    desired = np.divide(-.1 * g, size, out=np.zeros_like(g), where=size > 0) * eligible[..., None]
    incumbent = np.zeros_like(g)
    before = best = float(item['ctx'].values(origin)['local'])
    trials = []
    offset = np.zeros_like(g)
    for iteration in range(solves):
        if iteration:
            _, c, j, _ = linearize(item, origin + torch.from_numpy(offset))
        proposal, info = convex_projection(desired, c, j, eligible, offset)
        for scale in CONFIG['backtracking']:
            action, _ = pilot.project(proposal * scale)
            p = origin + torch.from_numpy(action)
            safety = item['ctx'].safety(p)
            objective = float(item['ctx'].values(p)['local'])
            trials.append(dict(iteration=iteration, scale=scale, safety=safety, local=objective,
                               useful=safety['passes'] and objective < before, **info))
            if safety['passes'] and objective < best:
                incumbent, best = action, objective
        offset = proposal
    assert item['ctx'].safety(origin + torch.from_numpy(incumbent))['passes']
    return incumbent, dict(latency_seconds=time.perf_counter() - started, trials=trials,
                           objective_before=before, objective_after=best)


def fit_span(r, rows):
    if len(rows) == 0:
        return np.zeros_like(r), np.zeros_like(r), []
    norms = np.linalg.norm(rows, axis=-1)
    rows = rows[norms > 1e-14] / norms[norms > 1e-14, None]
    if len(rows) == 0:
        return np.zeros_like(r), np.zeros_like(r), []
    coef = np.linalg.lstsq(rows.T, r, rcond=1e-10)[0]
    lam, _ = nnls(rows.T, r, maxiter=10 * len(rows) + 100)
    return coef @ rows, lam @ rows, lam.tolist()


def cosine(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / norm) if norm > 1e-15 else None


def diagnostic(item, origin, gold, prepared):
    g, c, j, families = prepared
    eligible = pilot.local_representation(origin, item['b']['mask'])['eligible'].numpy()
    columns = np.repeat(eligible.reshape(-1), 3)
    d = gold.reshape(-1)[columns]
    v = -g.reshape(-1)[columns]
    alpha = max(0., float(d @ v / max(v @ v, 1e-300)))
    base = alpha * v
    r = d - base
    energy = float(d @ d)
    result = dict(alpha=alpha, gold_energy=energy, residual_energy=float(r @ r),
                  descent_explained_energy=energy - float(r @ r), cosine_before=cosine(d, v),
                  constraint_values=c.tolist(), margins=(-c).tolist(),
                  eligibility_ineligible_count=int((~eligible).sum()),
                  eligibility_gold_energy=float(np.square(gold[~eligible]).sum()),
                  origin_ball_jacobian_norm=0., fits={})
    for name, threshold in [('active', 1e-5), ('near', .01)]:
        selection = c >= -threshold
        rows = -j[selection][:, columns]
        correction, compatible, lam = fit_span(r, rows)
        fit = dict(count=int(selection.sum()), indices=np.flatnonzero(selection).tolist(),
                   additional_explained_energy=float(r @ r - np.square(r - correction).sum()),
                   compatible_additional_explained_energy=float(r @ r - np.square(r - compatible).sum()),
                   cosine_after=cosine(d, base + correction), compatible_cosine_after=cosine(d, base + compatible),
                   compatible_residual_energy=float(np.square(r - compatible).sum()),
                   nonnegative_multipliers=lam, family_contributions={})
        # Leave-one-family-out gives conditional contribution; overlaps need not add.
        for family in sorted(set(families)):
            alone = -j[selection & (families == family)][:, columns]
            solo, _, _ = fit_span(r, alone)
            without, _, _ = fit_span(r, -j[selection & (families != family)][:, columns])
            fit['family_contributions'][family] = dict(count=len(alone), solo_explained_energy=float(r @ r - np.square(r - solo).sum()),
                conditional_explained_energy=float(np.square(r - without).sum() - np.square(r - correction).sum()))
        norms = np.linalg.norm(gold[eligible], axis=-1)
        ball = norms >= .1 - (1e-5 if name == 'active' else .001)
        ball_rows = np.zeros((int(ball.sum()), len(d)))
        for k, residue in enumerate(np.flatnonzero(ball)):
            ball_rows[k, residue * 3:residue * 3 + 3] = -d[residue * 3:residue * 3 + 3] / norms[residue]
        all_rows = np.concatenate((rows, ball_rows))
        combined, combined_nn, _ = fit_span(r, all_rows)
        fit['action_ball'] = dict(endpoint_active=int(ball.sum()),
            extra_explained_energy=float(np.square(r - correction).sum() - np.square(r - combined).sum()),
            combined_explained_energy=float(r @ r - np.square(r - combined).sum()),
            compatible_combined_explained_energy=float(r @ r - np.square(r - combined_nn).sum()))
        result['fits'][name] = fit
    return result


def worker(index):
    old.initialize()
    data, frozen = pilot.load()
    item = data[index]
    path = ROOT / 'trajectories' / f'trajectory_{index:02d}.json'
    if path.exists():
        validate(path, item)
        return
    arrays = item['arrays']
    record = item['row']['record']
    common = dict(index=index, identity=record['sample_id'], condition=record['condition'],
                  length=int(item['b']['mask'].sum()), length_bin=record['stratum'])
    one, diag, saved = [], [], {}
    for step, gold in enumerate(arrays['delta']):
        origin = torch.from_numpy(arrays['states'][step])
        started = time.perf_counter()
        prepared = linearize(item, origin)
        setup = time.perf_counter() - started
        diag.append(dict(common, step=step, **diagnostic(item, origin, gold, prepared)))
        saved[f'jacobian_{step}'] = prepared[2]
        saved[f'gradient_{step}'] = prepared[0]
        saved[f'families_{step}'] = prepared[3]
        before = pilot.scalar_metrics(item['ctx'], origin)
        gold_after = pilot.scalar_metrics(item['ctx'], torch.from_numpy(arrays['states'][step + 1]))
        desired_size = np.linalg.norm(prepared[0], axis=-1, keepdims=True)
        eligible = arrays['eligibility'][step]
        raw = np.divide(-.1 * prepared[0], desired_size, out=np.zeros_like(gold), where=desired_size > 0) * eligible[..., None]
        raw, _ = pilot.project(raw)
        raw_safe = item['ctx'].safety(origin + torch.from_numpy(raw))['passes']
        for method, solves in zip(METHODS, (1, 2)):
            action, cert = teacher(item, origin, solves, prepared)
            cert['latency_seconds'] += setup
            checked = old.checks(item, origin, action)
            repair = before['mean_local_rmse'] - checked['after']['mean_local_rmse']
            one.append(dict(common, step=step, method=method, before=before, gold_after=gold_after,
                gold_repair=before['mean_local_rmse'] - gold_after['mean_local_rmse'], repair=repair,
                raw_gradient_safe=raw_safe, rescued=not raw_safe and repair > 1e-8,
                exact_zero=bool((action == 0).all()), action_rms=float(np.sqrt(np.mean(np.square(action[eligible]).sum(-1)))),
                **old.comparison(action, gold, eligible), **checked, **cert))
            saved[f'{method}_one_{step}'] = action
        print(f'GOLD_STATE {index} {step + 1}/{len(arrays["delta"])}', flush=True)
    roll = []
    for method, solves in zip(METHODS, (1, 2)):
        states = [arrays['pg'].copy()]
        actions, steps = [], []
        baseline = pilot.scalar_metrics(item['ctx'], torch.from_numpy(states[0]))
        for step in range(16):
            origin = torch.from_numpy(states[-1])
            before = pilot.scalar_metrics(item['ctx'], origin)
            action, cert = teacher(item, origin, solves)
            checked = old.checks(item, origin, action)
            actions.append(action)
            states.append(states[-1] + action)
            steps.append(dict(common, method=method, step=step, before=before, exact_zero=bool((action == 0).all()), **checked, **cert))
        gold_final = pilot.scalar_metrics(item['ctx'], torch.from_numpy(arrays['states'][-1]))
        matched = steps[len(arrays['delta']) - 1]['after']
        roll.append(dict(common, method=method, matched_gold_horizon=len(arrays['delta']), baseline=baseline,
            gold_final=gold_final, matched_final=matched, P16=steps[-1]['after'], steps=steps,
            gold_repair=baseline['mean_local_rmse'] - gold_final['mean_local_rmse'],
            repair=baseline['mean_local_rmse'] - matched['mean_local_rmse']))
        saved[f'{method}_states'] = np.stack(states)
        saved[f'{method}_delta'] = np.stack(actions)
        print(f'ROLLOUT {index} {method}', flush=True)
    old.atomic_npz(path.with_suffix('.npz'), **saved)
    pilot.atomic_json(path, dict(one_step=one, rollouts=roll, diagnostic=diag,
        protocol_sha256=pilot.sha(ROOT / 'protocol.json'), source_sha256=pilot.sha(__file__),
        gold_manifest_sha256=frozen['manifest_sha256'], archive_sha256=pilot.sha(path.with_suffix('.npz'))))
    validate(path, item)


def validate(path, item):
    result = json.loads(path.read_text())
    assert result['protocol_sha256'] == pilot.sha(ROOT / 'protocol.json')
    assert result['source_sha256'] == pilot.sha(__file__)
    assert result['gold_manifest_sha256'] == pilot.sha(old.PILOT / 'gold/transitions.json')
    assert result['archive_sha256'] == pilot.sha(path.with_suffix('.npz'))
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as archive:
        for method in METHODS:
            for step in range(len(item['arrays']['delta'])):
                old.checks(item, torch.from_numpy(item['arrays']['states'][step]), archive[f'{method}_one_{step}'])
            states, actions = archive[f'{method}_states'], archive[f'{method}_delta']
            for step, action in enumerate(actions):
                assert np.array_equal(states[step] + action, states[step + 1])
                old.checks(item, torch.from_numpy(states[step]), action)
    return result


def launch(index):
    with (ROOT / 'logs' / f'trajectory_{index:02d}.log').open('a', buffering=1) as log:
        result = subprocess.run([sys.executable, '-B', __file__, '--worker', str(index)], stdout=log, stderr=log)
    if result.returncode:
        raise RuntimeError(f'Worker {index} failed; see D: logs')
    print(f'TRAJECTORY_COMPLETE {index}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker)
        return
    old.initialize()
    for path in (ROOT, ROOT / 'tmp', ROOT / 'cache', ROOT / 'logs', ROOT / 'trajectories'):
        path.mkdir(parents=True, exist_ok=True)
    data, frozen = pilot.load()
    for path, digest in frozen['source_sha256'].items():
        assert pilot.sha(path) == digest
    gradient_freeze = json.loads((old.ROOT / 'FINAL_FROZEN.json').read_text())
    # Validate original completion manifest without executing the gradient experiment.
    for path, digest in json.loads((old.ROOT / 'COMPLETED.json').read_text())['files'].items():
        assert pilot.sha(path) == digest
    protocol = ROOT / 'protocol.json'
    if protocol.exists():
        assert json.loads(protocol.read_text()) == CONFIG
    else:
        pilot.atomic_json(protocol, CONFIG)
    pilot.atomic_json(ROOT / 'provenance.json', dict(source_sha256=pilot.sha(__file__),
        gradient_freeze_sha256=pilot.sha(old.ROOT / 'FINAL_FROZEN.json'), gold_manifest_sha256=frozen['manifest_sha256'],
        gradient_conclusion=gradient_freeze, recovery='HEAD c97ac6a, clean tree, no interrupted projected artifacts found'))
    if args.test:
        test(data)
        return
    assert (ROOT / 'tests.json').exists()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for completed in concurrent.futures.as_completed([pool.submit(launch, i) for i in sorted(data)]):
            completed.result()
    print('BENCHMARK_COMPLETE', flush=True)


def test(data):
    item = data[0]
    origin = item['b']['pg']
    prepared = linearize(item, origin)
    g, c, j, _ = prepared
    direction = np.sin(np.arange(origin.numel())).reshape(origin.shape)
    direction /= np.linalg.norm(direction)
    epsilon = 1e-5
    plus = origin + torch.from_numpy(direction * epsilon)
    minus = origin - torch.from_numpy(direction * epsilon)
    fd = (item['ctx'].constraints(plus).numpy() - item['ctx'].constraints(minus).numpy()) / (2 * epsilon)
    assert np.allclose(fd, j @ direction.reshape(-1), rtol=1e-5, atol=1e-8)
    objective_fd = float((item['ctx'].values(plus)['local'] - item['ctx'].values(minus)['local']) / (2 * epsilon))
    assert np.isclose(objective_fd, np.sum(g * direction), rtol=1e-5, atol=1e-8)
    # Independent analytic projection: x<=0 and a 0.10 sphere.
    desired = np.array([[[.1, .1, 0.]]])
    action, info = convex_projection(desired, np.array([0.]), np.array([[1., 0., 0.]]), np.array([[True]]), np.zeros_like(desired))
    assert np.allclose(action, [[[0., .1, 0.]]], atol=1e-9) and info['converged']
    for solves in (1, 2):
        action, _ = teacher(item, origin, solves, prepared)
        old.checks(item, origin, action)
        repeat, _ = teacher(item, origin, solves, prepared)
        assert np.array_equal(action, repeat)
    pilot.atomic_json(ROOT / 'tests.json', dict(passed=True, exact_jacobian_directional_error=float(np.abs(fd - j @ direction.reshape(-1)).max()),
        analytic_convex_projection=True, deterministic=True, exact_safety=True, gold_solvers_disabled=True))


if __name__ == '__main__':
    main()
