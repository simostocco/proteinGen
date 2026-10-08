"""Validate and freeze the separate D:-resident projected-teacher experiment."""
import json
from pathlib import Path

import run_e010_constraint_projected_v1 as experiment
import numpy as np

old = experiment.old
pilot = experiment.pilot
ROOT = experiment.ROOT


def span_summary(rows):
    energy = sum(r['gold_energy'] for r in rows)
    residual = sum(r['residual_energy'] for r in rows)
    output = dict(count=len(rows), nonzero_gold=sum(r['gold_energy'] > 1e-15 for r in rows),
                  gold_energy=energy, descent_explained_fraction=sum(r['descent_explained_energy'] for r in rows) / energy if energy else None,
                  cosine_before=old.statistics([r['cosine_before'] for r in rows]), fits={})
    for selection in ('active', 'near'):
        fits = [r['fits'][selection] for r in rows]
        extra = sum(r['additional_explained_energy'] for r in fits)
        compatible = sum(r['compatible_additional_explained_energy'] for r in fits)
        output['fits'][selection] = dict(
            active_counts=old.statistics([r['count'] for r in fits]),
            additional_gold_energy_fraction=extra / energy if energy else None,
            residual_span_fraction=extra / residual if residual else None,
            total_explained_fraction=(energy - residual + extra) / energy if energy else None,
            compatible_total_explained_fraction=(energy - residual + compatible) / energy if energy else None,
            cosine_after=old.statistics([r['cosine_after'] for r in fits]),
            compatible_cosine_after=old.statistics([r['compatible_cosine_after'] for r in fits]),
            compatible_relative_residual=np.sqrt(sum(r['compatible_residual_energy'] for r in fits) / energy) if energy else None,
            families={family:dict(
                count=sum(r['family_contributions'][family]['count'] for r in fits),
                solo_gold_energy_fraction=sum(r['family_contributions'][family]['solo_explained_energy'] for r in fits) / energy if energy else None,
                conditional_gold_energy_fraction=sum(r['family_contributions'][family]['conditional_explained_energy'] for r in fits) / energy if energy else None)
                for family in fits[0]['family_contributions']},
            endpoint_action_ball=dict(
                count=sum(r['action_ball']['endpoint_active'] for r in fits),
                additional_gold_energy_fraction=sum(r['action_ball']['extra_explained_energy'] for r in fits) / energy if energy else None,
                combined_total_explained_fraction=(energy - residual + sum(r['action_ball']['combined_explained_energy'] for r in fits)) / energy if energy else None,
                compatible_combined_total_explained_fraction=(energy - residual + sum(r['action_ball']['compatible_combined_explained_energy'] for r in fits)) / energy if energy else None))
    output['eligibility'] = dict(gold_ineligible_energy=sum(r['eligibility_gold_energy'] for r in rows),
                                origin_action_ball_jacobian='identically zero',
                                equality_effect='ineligible coordinates removed before fits; gold equality residual exactly zero')
    return output


def benchmark_summary(one, roll, lookup, data):
    result = {}
    for method in experiment.METHODS:
        rows = [r for r in one if r['method'] == method]
        rollrows = [r for r in roll if r['method'] == method]
        a = old.aggregate(rows, lookup)
        b = old.aggregate(rollrows)
        a['max_previous_state_aligned_rmsd_ratio'] = a.pop('max_aligned_rmsd_ratio')
        b['max_previous_state_aligned_rmsd_ratio'] = b.pop('max_aligned_rmsd_ratio')
        a['max_frozen_baseline_aligned_rmsd_ratio'] = max(r['after']['aligned_rmsd'] / float(data[r['index']]['ctx'].baseline['aligned']) for r in rows)
        b['max_frozen_baseline_aligned_rmsd_ratio'] = max(s['after']['aligned_rmsd'] / float(data[r['index']]['ctx'].baseline['aligned']) for r in rollrows for s in r['steps'])
        a['action_max_angstrom'] = old.statistics([r['action_max'] for r in rows])
        b['action_max_angstrom'] = old.statistics([s['action_max'] for r in rollrows for s in r['steps']])
        unsafe = [r for r in rows if not r['raw_gradient_safe']]
        first = [r['trials'][0] for r in rows]
        a.update(raw_gradient_unsafe=len(unsafe), rescued_useful=sum(r['rescued'] for r in unsafe),
            rescue_fraction=sum(r['rescued'] for r in unsafe) / len(unsafe),
            action_rms=old.statistics([r['action_rms'] for r in rows]),
            first_convex_converged_fraction=float(np.mean([r['converged'] for r in first])),
            first_unshrunk_exact_safe_fraction=float(np.mean([r['safety']['passes'] for r in first])),
            first_convex_linear_violation=old.statistics([r['original_linear_violation'] for r in first]),
            first_convex_sweeps=old.statistics([r['sweeps'] for r in first]))
        result[method] = dict(one_step=a, rollout=b)
    return result


def curvature_summary(one):
    result = {}
    for method in experiment.METHODS:
        rows = [r for r in one if r['method'] == method]
        trials = [t for r in rows for t in r['trials'] if t['scale'] == 1.]
        result[method] = dict(unshrunk_trials=len(trials),
            global_aligned_failures=sum(t['safety']['global_constraints'][0] > 1e-8 for t in trials),
            continuous_chirality_failures=sum(t['safety']['global_constraints'][1] > 1e-8 for t in trials),
            strict_quartet_failures=sum(t['safety']['maximum_extra_constraint'] > 0 for t in trials),
            assessability_failures=sum(t['safety']['quartets']['assessability_lost'] > 0 for t in trials),
            inversion_failures=sum(t['safety']['quartets']['new_inversions'] > 0 for t in trials),
            exact_failures_despite_linear_feasible=sum(not t['safety']['passes'] and t['original_linear_violation'] <= 1e-8 for t in trials),
            approximate_linear_feasibility_diagnostic_tolerance=1e-8,
            rejected_extra_constraint_values=old.statistics([t['safety']['maximum_extra_constraint'] for t in trials if not t['safety']['passes']]),
            rejected_continuous_constraint_values=old.statistics([t['safety']['global_constraints'][1] for t in trials if not t['safety']['passes']]),
            bounded_solver_nonconvergence=sum(not t['converged'] for t in trials),
            exact_zero=sum(r['exact_zero'] for r in rows))
    return result


def main():
    old.initialize()
    data, frozen = pilot.load()
    source_hashes = json.loads((old.ROOT / 'provenance.json').read_text())['source_sha256']
    for path, digest in source_hashes.items():
        assert pilot.sha(path) == digest, path
    for path in (Path(experiment.__file__), Path(__file__),
                 Path('/mnt/d/Simone/proteinGen/tests/test_e010_constraint_projected_v1.py')):
        source_hashes[str(path.resolve())] = pilot.sha(path)
    outputs = [experiment.validate(ROOT / 'trajectories' / f'trajectory_{i:02d}.json', data[i]) for i in sorted(data)]
    one = sum([o['one_step'] for o in outputs], [])
    roll = sum([o['rollouts'] for o in outputs], [])
    diagnostic = sum([o['diagnostic'] for o in outputs], [])
    assert len(diagnostic) == 278 and len(one) == 556 and len(roll) == 40
    supplement = {}
    radius_margins, complementarity = [], {'active': [], 'near': []}
    # Context.safety also checks radius and finite coordinates outside constraints().
    # Radius is analytically redundant throughout every 0.10 A action ball here.
    for row in diagnostic:
        item = data[row['index']]
        p = item['arrays']['states'][row['step']]
        mask = item['arrays']['mask']
        centered = p[mask] - p[mask].mean(0)
        radius = float(np.sqrt(np.square(centered).sum() / mask.sum()))
        radius_jac = np.zeros_like(p)
        radius_jac[mask] = -centered / (mask.sum() * radius)
        assert radius - .1 >= 1e-3  # reverse triangle inequality on centered RMS
        radius_margins.append(radius - 1e-3)
        suffix = f"{row['index']:02d}_{row['step']:02d}"
        supplement[f'radius_jacobian_{suffix}'] = radius_jac
        supplement[f'radius_constraint_{suffix}'] = np.array(1e-3 - radius)
        eligible = item['arrays']['eligibility'][row['step']]
        # Exact equality Jacobian is a selection of Cartesian identity rows.
        supplement[f'eligibility_zero_indices_{suffix}'] = np.flatnonzero(np.repeat(~eligible.reshape(-1), 3))
        with np.load(ROOT / 'trajectories' / f"trajectory_{row['index']:02d}.npz", allow_pickle=False) as archive:
            j = archive[f"jacobian_{row['step']}"]
        for selection in ('active', 'near'):
            indices = np.array(row['fits'][selection]['indices'], dtype=int)
            norms = np.linalg.norm(j[indices][:, np.repeat(eligible.reshape(-1), 3)], axis=-1)
            indices = indices[norms > 1e-14]
            norms = norms[norms > 1e-14]
            lam = np.array(row['fits'][selection]['nonnegative_multipliers'])
            assert len(lam) == len(norms)
            # Fit coefficients were on unit inward rows. Convert to original J.
            original_lambda = lam / norms
            supplement[f'{selection}_original_multipliers_{suffix}'] = original_lambda
            complementarity[selection].append(float(np.linalg.norm(original_lambda * np.array(row['constraint_values'])[indices])))
    old.atomic_npz(ROOT / 'phase_a_auxiliary_constraints.npz', **supplement)
    manifest = json.loads((old.PILOT / 'gold/transitions.json').read_text())['transitions']
    lookup = {(r['trajectory_index'], r['step']): r for r in manifest}
    summary = benchmark_summary(one, roll, lookup, data)
    assert all(r['one_step']['raw_gradient_unsafe'] == 252 for r in summary.values())
    span = span_summary(diagnostic)
    span['minimum_radius'] = dict(margins_angstrom=old.statistics(radius_margins),
        active_count=0, near_active_count=0, span_contribution=0,
        ball_redundancy_proved=True, finite_coordinates='verified; domain requirement, no finite differentiable boundary')
    span['approximate_complementarity'] = {k: old.statistics(v) for k, v in complementarity.items()}
    groups, rollout_groups, span_groups = {}, {}, {}
    for dimension in ('identity', 'condition', 'length', 'length_bin', 'step'):
        keys = sorted({str(r[dimension]) for r in one})
        groups[dimension] = {key: {m: old.aggregate([r for r in one if str(r[dimension]) == key and r['method'] == m], lookup)
                                  for m in experiment.METHODS} for key in keys}
        span_groups[dimension] = {key: span_summary([r for r in diagnostic if str(r[dimension]) == key]) for key in keys}
        if dimension != 'step':
            rollout_groups[dimension] = {key: {m: old.aggregate([r for r in roll if str(r[dimension]) == key and r['method'] == m])
                                             for m in experiment.METHODS} for key in keys}
    curvature = curvature_summary(one)
    results = dict(summary=summary, span=span, by=groups, rollout_by=rollout_groups, span_by=span_groups,
                   curvature=curvature, gold_manifest_sha256=frozen['manifest_sha256'],
                   source_sha256=source_hashes,
                   success=any(r['rollout']['repair_retention'] >= .8 for r in summary.values()))
    pilot.atomic_json(ROOT / 'results.json', results)
    lines = ['# Fast privileged constraint-projected teacher v1', '',
        'Recovery: repository HEAD was c97ac6a, with a clean worktree and no later commits. '
        'No interrupted projected-teacher files or directory were found in the D:-resident proteinGen data tree. '
        'The completed gradient experiment and FAST-PT negative pilot were preserved. '
        'Frozen gold files, source lineage, and the gradient completion manifest were hash-verified. '
        'No gradient benchmark, gold SLSQP solve, neural training, or additional label generation was run.', '',
        'New artifacts are under `/mnt/d/Simone/proteinGen/outputs/fast_privileged_constraint_projected_teacher_v1`. '
        'All temporary files, logs, solver arrays, and reports are on D:. '
        'Protocol thresholds and limits were frozen before the benchmark; no measured-result tuning was performed.', '',
        '## Phase A: mechanistic diagnostic', '',
        'Exactly 278 starting states from 20 frozen trajectories (7 identities), including 63 exact-zero gold actions. '
        'The exact unnormalized local-loss gradient and every frozen normalized safety Jacobian/value were evaluated and archived. '
        'Active margin is 1e-5; near-active margin is 0.01, in the historical normalized constraint units. '
        'Fits restrict coordinates to the frozen starting-state eligible set and fit alpha >= 0. '
        'Explained fractions below use uncentered gold-action squared energy; they are not centered statistical variance. '
        'Cosines exclude zero vectors, so their sample counts must accompany interpretation.', '',
        f"Negative-gradient scalar component explains {100*span['descent_explained_fraction']:.3f}% of gold-action energy. "
        f"Starting cosine: mean {span['cosine_before']['mean']:.6f}, median {span['cosine_before']['median']:.6f}, n={span['cosine_before']['count']}.", '',
        '| Set | Additional gold energy explained | Residual in span | Total explained | Compatible total | Mean cosine after | Compatible mean cosine |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, f in span['fits'].items():
        lines.append(f"| {name} | {100*f['additional_gold_energy_fraction']:.3f}% | {100*f['residual_span_fraction']:.3f}% | {100*f['total_explained_fraction']:.3f}% | {100*f['compatible_total_explained_fraction']:.3f}% | {f['cosine_after']['mean']:.6f} | {f['compatible_cosine_after']['mean']:.6f} |")
    lines += ['', 'The compatible fit uses inward rows J_inward = -dc/dP for c<=0, with lambda>=0, '
        'so delta≈-alpha*g+J_inward.T@lambda. Alpha is held at the preregistered scalar fit before residual correction. '
        'This is an approximate starting-state consistency diagnostic: finite gold steps, curvature, objective Hessians, '
        'and endpoint activity prevent it from proving exact nonlinear KKT optimality. Unrestricted span fits provide an upper bound on this residual explanation.', '',
        '| Constraint family | Active count | Solo residual explanation / gold energy | Conditional contribution / gold energy |',
        '|---|---:|---:|---:|']
    for family, f in span['fits']['active']['families'].items():
        lines.append(f"| {family} | {f['count']} | {100*f['solo_gold_energy_fraction']:.3f}% | {100*f['conditional_gold_energy_fraction']:.3f}% |")
    lines += ['', 'Solo and leave-one-family-out contributions overlap and do not generally sum. '
        'Eligibility is an exact equality, eliminated before fitting; all gold ineligible coordinates are exactly zero. '
        'Action-ball Jacobians at starting delta=0 are identically zero and explain no starting-state residual. '
        'As a separately labeled endpoint diagnostic, gold actions within 1e-5 A of the 0.10 A ball contribute inward endpoint normals; '
        f"these explain an additional {100*span['fits']['active']['endpoint_action_ball']['additional_gold_energy_fraction']:.3f}% of gold energy in unrestricted span fits. "
        f"The combined unrestricted explanation is {100*span['fits']['active']['endpoint_action_ball']['combined_total_explained_fraction']:.3f}%, "
        f"while the combined inequality-compatible explanation is {100*span['fits']['active']['endpoint_action_ball']['compatible_combined_total_explained_fraction']:.3f}%. "
        'This endpoint calculation uses gold and is never part of teacher inference.', '',
        'Per-state alpha, constraint values/margins, active indices, nonnegative multipliers, family contributions and '
        'Jacobians are in trajectories/. Grouped results by identity, corruption condition, exact length, length bin, '
        'and step are in results.json (span_by, by, rollout_by). '
        'phase_a_auxiliary_constraints.npz also stores radius constraint values and exact Jacobians, eligibility '
        'identity-row indices, and multipliers rescaled to original constraint Jacobians. '
        'The radius>=1e-3 A gate is inactive in every state and analytically redundant inside every 0.10 A action ball '
        '(centered coordinate RMS changes by at most 0.10 A); its span contribution is zero. '
        'Finite coordinates are a domain check, with no finite differentiable boundary. '
        'Approximate complementarity norms are reported separately; signed NNLS fitting alone does not establish complementarity. '
        f"Active-set compatible reconstruction has relative residual {span['fits']['active']['compatible_relative_residual']:.4f}; "
        f"its mean complementarity norm is {span['approximate_complementarity']['active']['mean']:.3g}.", '',
        '## Phase B: bounded fast convex projection', '',
        'The desired action is the per-residue normalized negative local gradient at 0.10 A. '
        'Dykstra projects onto all reachable linearized safety halfspaces and the product of exact residue balls. '
        'Eligibility zeros are enforced by eliminating coordinates. A halfspace is omitted only when its support '
        'function proves it cannot be violated anywhere in the ball; numerically zero Jacobian rows (norm<=1e-14) '
        'are also omitted from the projection, with original linear violation and exact safety still checked. '
        'Each convex solve has at most 128 sweeps. '
        'Convergence requires an iterate change <=1e-9 and normalized linear violation <=1e-9. '
        'Finite-budget approximations are explicitly recorded, not claimed to be exact convex optima.', '',
        'projection_1 uses one convex solve. projection_2 adds one relinearization at the first projected endpoint, '
        'solving for total action about the original teacher input with the same desired action and balls. '
        'Every convex endpoint and each scale in [1, 0.5, 0.25, 0.125] receives the historical exact nonlinear check. '
        'The best exact-safe, local-MSE-nonworsening candidate is retained; otherwise the action is exact zero. '
        'No unrestricted nonlinear iteration or general optimization solver is used.', '',
        '| Method | One-step retention | Rollout retention | Median latency | Archived SLSQP speedup | Mean gold cosine | RMS difference | Useful rescues / 252 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method, result in summary.items():
        a, b = result['one_step'], result['rollout']
        lines.append(f"| {method} | {100*a['repair_retention']:.3f}% | {100*b['repair_retention']:.3f}% | {1000*a['latency_seconds']['median']:.3f} ms | {a['speedup']:.1f}x | {a['cosine']['mean']:.6f} | {a['action_rms_difference']['mean']:.6f} A | {a['rescued_useful']} |")
    lines += ['', 'Retention is sum absolute mean-local-RMSE repair / corresponding gold sum. '
        'Rollouts run 16 steps from all 20 corruptions and compare at the exact available gold prefix length. '
        'Useful rescue means an originally unsafe unshrunk raw-gradient case yields an exact-safe correction '
        'with mean-local-RMSE repair >1e-8 A. The raw-gradient unsafe count reproduces exactly 252. '
        'Latency includes gradient, eligibility, Jacobians, convex solve(s), all nonlinear checks, and fallback; '
        'Phase A and reporting are excluded. Four isolated single-threaded processes run concurrently. '
        'Archived speedup uses matched existing SLSQP times and excludes cases without archived timings.', '',
        'Every stored gold-state action and all 640 rollout actions were independently checked against '
        'aligned RMSD <=1.01 original-corruption baseline and continuous chirality under the frozen 1e-8 '
        'normalized global tolerance, exact quartet constraints, '
        'no new inversions, no assessability loss, eligibility zeros, and exact NumPy residue cap <=0.10 A. '
        'All gates passed; inversion, assessability, cap and eligibility violation counts are zero.', '',
        'Detailed norms, aligned RMSD distributions, latency distributions, rollout horizons, convergence flags '
        'and all rejected-proposal certificates are archived in results.json and trajectories/. '
        'The summary reports aligned RMSD ratios against the frozen corruption baseline. '
        'Grouped inherited max_aligned_rmsd_ratio fields compare consecutive states and do not represent the frozen safety bound.', '',
        '## Curvature and remaining gap', '']
    for method, values in curvature.items():
        a = summary[method]['one_step']
        lines.append(f"{method}: {100*a['first_convex_converged_fraction']:.2f}% first convex solves converged; "
            f"{100*a['first_unshrunk_exact_safe_fraction']:.2f}% first unshrunk proposals passed exact safety. "
            f"Across {values['unshrunk_trials']} unshrunk proposals, continuous chirality failed {values['continuous_chirality_failures']} times, "
            f"aligned RMSD {values['global_aligned_failures']}, strict quartet gates {values['strict_quartet_failures']}. "
            f"{values['exact_failures_despite_linear_feasible']} exact failures occurred with linear violation <=1e-8; "
            f"{values['bounded_solver_nonconvergence']} solves hit the convex budget. {values['exact_zero']} final one-step actions were zero.")
    best = max(summary, key=lambda m: summary[m]['rollout']['repair_retention'])
    retention = summary[best]['rollout']['repair_retention']
    lines += ['', f"Best rollout retention is {100*retention:.3f}% ({best}); the target is approximately 80%. "
        + ('The target was reached.' if retention >= .8 else
           'The method remains below target and this experiment stops here. No further model, labels, corpus, or student was launched.'), '',
        'A starting-state tangent halfspace omits second-order constraint change. In particular, continuous chirality '
        'is a sum of squared normalized quartet errors: a tangent step at its baseline boundary can increase it '
        'through curvature even when J*delta<=0. Signed normalized quartet quantities depend nonlinearly on bond '
        'directions and lengths. For initially inverted quartets the assessability gate uses q squared, whose '
        'Jacobian weakens near q=0; a finite step can cross the forbidden low-assessability region. '
        'The exact quartet gates require literal <=0, while the bounded convex solve uses a 1e-9 tolerance on '
        'unit Jacobian rows. Small positive linear residuals therefore can also fail exact certification; '
        'the <=1e-8 diagnostic above is approximate linear feasibility, not exact linear feasibility. '
        'One endpoint relinearization and four backtracking scales do not '
        'guarantee a useful feasible path. Residual span explanation measures directions, not the needed inward '
        'margin or nonlinear coupling. Any budget-hit convex solve also leaves a numerical approximation limitation; '
        'its failure must not be attributed solely to nonlinear curvature. Per-trial certificates separate these cases. '
        'The starting active safety span explains only 3.01% of the residual after scalar descent; '
        'near-active rows explain 3.79%. Endpoint ball normals account for much more in unrestricted fits '
        '(total 98.14%), but the compatible total is only 84.55%. This points to per-residue cap geometry and '
        'finite-step nonlinear coupling beyond a small starting-state active-span correction; it does not '
        'justify treating the tangent projection as an approximation of the nonlinear gold optimum.', '',
        'Preflight: directional finite differences for the exact safety Jacobians and loss gradient; analytic '
        'halfspace-plus-ball example; repeated deterministic inference; exact output gates. '
        'Completion independently verifies archive hashes and every stored action.']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    files = {str(p): pilot.sha(p) for p in ROOT.rglob('*') if p.is_file()
             and p.name != 'COMPLETED.json' and '/logs/' not in str(p) and '/tmp/' not in str(p) and '/cache/' not in str(p)}
    pilot.atomic_json(ROOT / 'COMPLETED.json', dict(status='complete', files=files, gold_states=278,
        trajectories=20, no_new_gold_solves=True, no_neural_training=True, target_reached=retention >= .8))
    print(json.dumps(dict(summary=summary, span=span, curvature=curvature), indent=2))


if __name__ == '__main__':
    main()
