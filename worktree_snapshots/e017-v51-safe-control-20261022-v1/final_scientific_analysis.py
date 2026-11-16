"""V5.1 v1: summarize saved outcomes only. No policy evaluation or optimization.

Run with ../run.sh; all outputs and scratch are on D:. Bootstrap is post hoc
analysis of the prospectively specified endpoints, not another model evaluation.
"""
from __future__ import annotations

import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

R = Path(__file__).resolve().parent.parent
V5 = R.parent / 'e017_v5_goal_observability_20261022_v1'
V4 = R.parent / 'e017_v4_multistep_20261022_v1'
E10 = Path('/mnt/d/Simone/proteinGen/outputs/e010_student_direction_observability_v1')
VERSION = 'e017_v51_final_scientific_analysis_v1'
SEED = 1736517
REPS = 20000
INPUTS = {}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 2**20), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    p = Path(p)
    INPUTS[str(p.resolve())] = sha(p)
    return json.loads(p.read_text())


def write(p, value):
    p = Path(p)
    temp = p.with_name(p.name + '.part')
    with temp.open('x') as f:
        f.write(value if isinstance(value, str) else json.dumps(value, indent=2, allow_nan=False))
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, p)
    fd = os.open(p.parent, os.O_DIRECTORY)
    os.fsync(fd)
    os.close(fd)


def key(r):
    return r['task_id'], r['seed']


def endpoint(r, metric):
    if metric == 'joint_J':
        return float(r['valid']) * math.exp(-.5 * ((r['position_A']/8)**2 + (r['orientation_deg']/35)**2))
    if metric == 'collision':
        return float(r['collision']['has_collision'])
    return float(r[metric])


METRICS = ['position_A', 'orientation_deg', 'success', 'valid', 'translation_hit', 'rotation_hit', 'KL', 'joint_J', 'collision']


def paired(a, b):
    aa = {key(r): r for r in a}
    bb = {key(r): r for r in b}
    assert aa.keys() == bb.keys()
    groups = sorted({r['group'] for r in a})
    sums, sizes = [], []
    for g in groups:
        rows = [r for r in b if r['group'] == g]
        assert all(aa[key(r)]['identity'] == r['identity'] and aa[key(r)]['group'] == g for r in rows)
        sums.append([sum(endpoint(r, m)-endpoint(aa[key(r)], m) for r in rows) for m in METRICS])
        sizes.append(len(rows))
    sums = np.asarray(sums)
    sizes = np.asarray(sizes)
    draws = np.random.default_rng(SEED).integers(0, len(groups), (REPS, len(groups)))
    samples = sums[draws].sum(1)/sizes[draws].sum(1)[:, None]
    out = {}
    for j, m in enumerate(METRICS):
        out[m] = dict(delta=float(sums[:, j].sum()/sizes.sum()), CI95=np.quantile(samples[:, j], [.025, .975]).tolist(),
                      CI98_333333=np.quantile(samples[:, j], [.05/6, 1-.05/6]).tolist(),
                      equal_group_delta=float(np.mean(sums[:, j]/sizes)))
    return dict(attempts=len(a), groups=len(groups), group_ids=groups, attempts_per_group=sizes.tolist(), endpoints=out)


def stats(xs):
    x = np.asarray(xs, dtype=float)
    return dict(mean=float(x.mean()), median=float(np.median(x)), min=float(x.min()), max=float(x.max()), p95=float(np.quantile(x, .95)))


def correlation(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None


def physical_checks(x, lim):
    d = np.linalg.norm(np.diff(x, axis=0), axis=1)
    ix = np.arange(len(x))
    pairs = (ix[:, None]-ix[None, :] >= 3)
    dist = np.linalg.norm(x[:, None]-x[None, :], axis=-1)
    count = int(((dist < 2) & pairs).sum())
    rg = float(np.sqrt(np.square(x-x.mean(0)).sum(-1).mean())/len(x)**(1/3))
    crosses = np.linalg.norm(np.cross(np.diff(x, axis=0)[1:], np.diff(x, axis=0)[:-1]), axis=-1)
    checks = dict(nonfinite=not bool(np.isfinite(x).all()),
                  adjacent_spacing=bool(d.min() < lim['adjacent_low_A'] or d.max() > lim['adjacent_high_A']),
                  nonlocal_clashes=count*100/len(x) > lim['clashes_per100_max'],
                  fragment_Rg=not lim['Rg_normalized_low'] <= rg <= lim['Rg_normalized_high'],
                  degenerate_frame=bool((crosses <= 1e-8).any()))
    return checks, count


def analyze_rollouts(d, manifest, lim):
    ts = {t['task_id']: t for t in manifest['tasks']}
    causes = collections.Counter()
    first = collections.Counter()
    collisionfirst = collections.Counter()
    pairtypes = collections.Counter()
    invalid = []
    for r in d['records']:
        t = ts[r['task_id']]
        assert t['identity'] == r['identity'] and t['protected_group_id'] == r['group']
        assert r['seed'] in [7101, 7102]
        x = np.asarray(t['native_prefix'] + r['tail'], dtype=float)
        checks, count = physical_checks(x, lim)
        assert (not any(checks.values())) == r['valid'], (d['label'], r['task_id'], checks)
        assert math.isclose(count*100/len(x), r['clashes_per100'], abs_tol=1e-10)
        if not r['valid']:
            reasons = [k for k, v in checks.items() if v]
            causes.update(reasons)
            earliest = None
            for step in range(r['H']+1):
                c, _ = physical_checks(x[:t['t']+step], lim)
                if any(c.values()):
                    earliest = dict(step=step, criteria=[k for k, v in c.items() if v])
                    break
            first[str(earliest['step'])] += 1
            invalid.append(dict(task_id=r['task_id'], identity=r['identity'], group=r['group'], seed=r['seed'],
                                criteria=reasons, first_fixed_criterion_failure=earliest,
                                first_new_collision_step=r['collision']['first_collision_step'],
                                collision_pairs=r['collision']['collision_pairs']))
        if r['collision']['has_collision']:
            collisionfirst[str(r['collision']['first_collision_step'])] += 1
            pairtypes.update(p['pair_type'] for p in r['collision']['collision_pairs'])
    step = [r['step_KL'] for r in d['records'] if r.get('step_KL')]
    return dict(invalid=len(invalid), invalid_causes=dict(causes),
                first_invalid_step_counts=dict(first), first_new_collision_step_counts=dict(collisionfirst),
                new_collision_pair_types=dict(pairtypes), invalid_records=invalid,
                KL_by_step_mean=np.mean(step, axis=0).tolist() if step else None,
                KL_by_step_p95=np.quantile(step, .95, axis=0).tolist() if step else None,
                KL_position_correlation=correlation([r['KL'] for r in d['records']], [r['position_A'] for r in d['records']]),
                sampled_tail_spacing=stats([r[k] for r in d['records'] for k in ['adjacent_min_A', 'adjacent_max_A']]),
                mean_angle_rad=stats([r['angle_mean_rad'] for r in d['records']]),
                mean_torsion_rad=stats([r['torsion_mean_rad'] for r in d['records']]))


def compute():
    seal = read(R/'SEALED_CONFIG.json')
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == seal['source']
    historical = read(R/'HISTORICAL_H4_COLLISION_AND_KL_ANALYSIS.json')
    # Stored native-calibrated limits are immutable; never infer them from V5.1.
    lim = historical['summary']['limits']
    manifests = {p.stem: read(p) for p in (R/'manifests').glob('*.json')}
    evaluations = {}
    duplicate = {}
    raw = {}
    for p in sorted((R/'results').glob('*.json')):
        d = read(p)
        if 'summary' not in d or 'records' not in d:
            continue
        raw[p.stem] = d
        if '_MATCHED100_' in p.stem:
            canonical = p.stem.replace('_MATCHED100', '')
            original = read(p.with_name(canonical+'.json'))
            assert d['records'] == original['records'] and d['summary'] == original['summary'] and d['direction'] == original['direction']
            duplicate[p.stem] = canonical
            continue
        rows = d['records']
        assert len(rows) == d['summary']['attempts'] and len({key(r) for r in rows}) == len(rows)
        for m in ['position_A', 'orientation_deg', 'KL']:
            assert math.isclose(np.mean([r[m] for r in rows]), d['summary'][m], abs_tol=1e-10)
        for m in ['success', 'valid', 'translation_hit', 'rotation_hit', 'finite']:
            assert sum(r[m] for r in rows) == d['summary'][m]
        manifest = manifests[f"{d['panel']}_H{d['H']}_{d['condition']}"]
        assert {(t['task_id'], ns) for t in manifest['tasks'] for ns in [7101,7102]} == {key(r) for r in rows}
        analysis = analyze_rollouts(d, manifest, lim)
        evaluations[p.stem] = dict(source=str(p.resolve()), summary=d['summary'], direction=d['direction'],
                                   joint_J=float(np.mean([endpoint(r,'joint_J') for r in rows])),
                                   panel=d['panel'], H=d['H'], update=d['update'], condition=d['condition'],
                                   failure_analysis=analysis)
    panels = {}
    for name, m in manifests.items():
        panels[name] = dict(tasks=len(m['tasks']), attempts=2*len(m['tasks']),
                            identities=sorted({t['identity'] for t in m['tasks']}),
                            protected_groups=sorted({t['protected_group_id'] for t in m['tasks']}),
                            task_ids=[t['task_id'] for t in m['tasks']], seeds=[7101,7102],
                            source=f'manifests/{name}.json')
    train_tasks = read(V5/'manifests/TRAIN.json')['tasks']
    train_groups = {t['protected_group_id'] for t in train_tasks}
    assert all(not train_groups.intersection(p['protected_groups']) for p in panels.values())
    assert not set(panels['DEV_H4_native']['protected_groups']) & set(panels['AUDIT_H4_native']['protected_groups'])
    comparisons = {}
    for panel, H, condition in [('AUDIT',4,'native'),('AUDIT',4,'shifted'),('AUDIT',2,'native'),('DEV',4,'native'),('DEV',2,'native')]:
        suffix=f'{panel}_H{H}_{condition}'
        baseline=raw[f'A_00100_{suffix}']['records']
        for arm in 'BCD':
            comparisons[f'{arm}-A_100_{suffix}'] = paired(baseline,raw[f'{arm}_00100_{suffix}']['records'])
        initial=raw[f'INITIAL_00000_{suffix}']['records']
        for arm in 'ABCD':
            comparisons[f'{arm}100-INITIAL_{suffix}'] = paired(initial,raw[f'{arm}_00100_{suffix}']['records'])
        if panel=='DEV':
            for arm in 'ABCD':
                comparisons[f'{arm}250-INITIAL_{suffix}']=paired(initial,raw[f'{arm}_00250_{suffix}']['records'])
    for H in [2,4]:
        suffix=f'AUDIT_H{H}_native'
        comparisons[f'INITIAL-J1_{suffix}']=paired(raw[f'FROZEN_J1_00000_{suffix}']['records'],raw[f'INITIAL_00000_{suffix}']['records'])
    train = {}
    logs = {}
    for arm in 'ABCD':
        p=R/f'results/{arm}_updates.jsonl'
        INPUTS[str(p)] = sha(p)
        rows=[json.loads(s) for s in p.read_text().splitlines()]
        logs[arm]=rows
        assert [r['update'] for r in rows]==list(range(1,251))
        train[arm]=dict(updates=len(rows), presentations=sum(len(r['identities']) for r in rows),
                        distinct_identities=len({i for r in rows for i in r['identities']}),
                        gradient_norm=stats([r['gradient_norm'] for r in rows]),
                        clipped_updates=sum(r['gradient_norm']>5 for r in rows),
                        clip_threshold=5., clipping_fraction=sum(r['gradient_norm']>5 for r in rows)/len(rows),
                        groups={g:stats([r['groups'][g] for r in rows]) for g in rows[0]['groups']},
                        dual_final=rows[-1]['dual'], dual_max=max(r['dual'] for r in rows),
                        windows={}, outcome_position_orientation_hits='not logged on TRAIN; loss is not a substitute')
        for start,end in [(1,25),(76,100),(101,125),(226,250)]:
            window=rows[start-1:end]
            train[arm]['windows'][f'{start}-{end}']=dict(terms={k:float(np.mean([r['terms'][k] for r in window])) for k in rows[0]['terms']},
                                                         gradient_norm=stats([r['gradient_norm'] for r in window]),
                                                         dual_mean=float(np.mean([r['dual'] for r in window])))
    assert all([r['task_ids'] for r in logs[a]]==[r['task_ids'] for r in logs['A']] for a in 'BCD')
    kl_precision = {}
    for arm in 'ABCD':
        for H in [2,4]:
            a=raw[f'INITIAL_00000_DEV_H{H}_native']['records']
            b=raw[f'{arm}_00250_DEV_H{H}_native']['records']
            aa={key(r):r for r in a}
            kl_precision[f'{arm}_H{H}_250_minus_initial']=dict(
                paired_KL_change_vs_position_gain_correlation=correlation([r['KL']-aa[key(r)]['KL'] for r in b],[aa[key(r)]['position_A']-r['position_A'] for r in b]),
                interpretation='positive means larger KL increases accompany larger position gains; descriptive, not causal')
    out=dict(schema=VERSION, created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             source_commit=seal['source'], sealed=seal, limits=lim, panels=panels,
             evaluations=evaluations, duplicate_evaluations=duplicate, comparisons=comparisons, TRAIN=train,
             matched_TRAIN_order_verified=True, KL_precision_associations=kl_precision,
             bootstrap=dict(seed=SEED, replicates=REPS, unit='paired protected identity/group; retain all task/seed rows per sampled group',
                            estimator='attempt-weighted ratio of resampled group sums/counts; equal-group point estimates also provided',
                            intervals='percentile; nominal95 and two-sided Bonferroni98.333333 for three safety arms versus A',
                            scope='conditional on fixed panels and two evaluation seeds, one training seed; not training-seed uncertainty',
                            caution='No resampling of attempts independently; secondary comparisons exploratory, no simultaneous family claim beyond primary three-arm endpoint family'),
             limitations=['TRAIN position/orientation/hit panels absent; cannot reconstruct from scalar loss',
                          'No per-action-component Gaussian means/log-stds in saved evaluations: KL component attribution unavailable',
                          '+250 shifted and AUDIT evaluations absent by stop control flow; not run retrospectively',
                          'No policy-gradient conflict cosines; clipping is aggregate evidence only',
                          'No new IK feasibility or selected-policy density evaluation; scalar KL is a sampled conditional mean, not whole-trajectory KL certification'])
    write(R/'source/final_analysis_work.json', out)
    print(json.dumps(dict(evaluations=len(evaluations), duplicates=len(duplicate), comparison_count=len(comparisons),
                          TRAIN={a:{k:train[a][k] for k in ['distinct_identities','clipping_fraction','dual_final']} for a in 'ABCD'})))


if __name__=='__main__':
    compute()
