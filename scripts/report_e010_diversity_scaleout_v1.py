"""Freeze measured diversity-pilot outcomes; never start student training."""
import json

import e010_diversity_scaleout_v1 as e
import numpy as np

ROOT = e.ROOT


def stats(values):
    return e.frozen.old.statistics(values)


def condition_rows(records):
    rows = []
    for identity in records:
        for condition, result in identity['conditions'].items():
            baseline, final = result['baseline'], result['final']
            mse_gain = (baseline['local_mse'] - final['local_mse']) / baseline['local_mse']
            rows.append(dict(identity=identity['identity'], index=identity['index'], condition=int(condition),
                length=identity['length'], length_bin=identity['length_bin'], portion=identity['portion'],
                metadata_cell=identity['metadata_cell'], cluster_id=identity['cluster_id'],
                split_group_id=identity['split_group_id'], **result,
                relative_mse_gain=mse_gain, absolute_mse_gain=baseline['local_mse']-final['local_mse'],
                rmse_gain=baseline['mean_local_rmse']-final['mean_local_rmse'],
                relative_rmse_gain=(baseline['mean_local_rmse']-final['mean_local_rmse']) / baseline['mean_local_rmse'],
                aligned_rmsd_change=final['aligned_rmsd']-baseline['aligned_rmsd'],
                useful=result['nonzero_steps'] > 0 and mse_gain > 1e-6))
    return rows


def aggregate(rows):
    identities = {r['identity'] for r in rows}
    useful_identities = {r['identity'] for r in rows if r['useful']}
    steps = [s for r in rows for s in r['steps']]
    baseline_mse_sum = sum(r['baseline']['local_mse'] for r in rows)
    mse_gain_sum = sum(r['absolute_mse_gain'] for r in rows)
    baseline_rmse_sum = sum(r['baseline']['mean_local_rmse'] for r in rows)
    horizons = {}
    for k in (1, 2, 4, 8, 12, 16):
        gains = [(r['baseline']['local_mse']-r['horizons'][str(k)]['local_mse']) / r['baseline']['local_mse'] for r in rows]
        rmse_gains = [(r['baseline']['mean_local_rmse']-r['horizons'][str(k)]['mean_local_rmse']) / r['baseline']['mean_local_rmse'] for r in rows]
        horizons[str(k)] = dict(relative_mse_gain=stats(gains), relative_mean_local_rmse_gain=stats(rmse_gains),
            aggregate_mse_reduction=sum(r['baseline']['local_mse']-r['horizons'][str(k)]['local_mse'] for r in rows) / baseline_mse_sum,
            aggregate_rmse_reduction=sum(r['baseline']['mean_local_rmse']-r['horizons'][str(k)]['mean_local_rmse'] for r in rows) / baseline_rmse_sum)
    late = {}
    total_inference_seconds = sum(s['latency_seconds'] for s in steps)
    for k in (4, 8, 12):
        additional = sum(r['horizons'][str(k)]['local_mse']-r['final']['local_mse'] for r in rows)
        late[str(k)] = dict(aggregate_additional_mse_reduction=additional / baseline_mse_sum,
            fraction_of_total_mse_repair=additional / mse_gain_sum if mse_gain_sum > 0 else None,
            mean_identity_condition_additional_relative_mse_gain=float(np.mean([
                (r['horizons'][str(k)]['local_mse']-r['final']['local_mse']) / r['baseline']['local_mse'] for r in rows])),
            trajectories_with_nonzero_actions_after_k=sum(any(not s['exact_zero'] for s in r['steps'][k:]) for r in rows))
        late[str(k)]['fraction_of_measured_inference_cost'] = sum(s['latency_seconds'] for r in rows for s in r['steps'][k:])/total_inference_seconds
        late[str(k)]['evaluated_steps_after_k'] = sum(len(r['steps'][k:]) for r in rows)
    violation_counts = dict(new_inversions=sum(s['new_inversions'] for s in steps),
        assessability_losses=sum(s['assessability_loss'] for s in steps),
        cap_violations=sum(s['action_cap_violations'] for s in steps),
        eligibility_violations=sum(s['eligibility_violations'] for s in steps),
        exact_safety_failures=sum(not s['after']['safety']['passes'] for s in steps),
        local_objective_increases=sum(s['local_objective_after'] > s['local_objective_before'] for s in steps))
    return dict(identities=len(identities), trajectories=len(rows), evaluated_steps=len(steps),
        useful_identity_fraction=len(useful_identities)/len(identities), useful_trajectory_fraction=float(np.mean([r['useful'] for r in rows])),
        nonzero_identity_fraction=len({r['identity'] for r in rows if r['nonzero_steps'] > 0})/len(identities),
        mean_relative_local_mse_gain=float(np.mean([r['relative_mse_gain'] for r in rows])),
        zero_at_first_step_fraction=float(np.mean([r['first_zero_step'] == 1 for r in rows])),
        mean_initial_local_mse=float(np.mean([r['baseline']['local_mse'] for r in rows])),
        mean_initial_continuous_chirality=float(np.mean([r['baseline']['continuous_chiral'] for r in rows])),
        relative_local_mse_gain=stats([r['relative_mse_gain'] for r in rows]),
        relative_mean_local_rmse_gain=stats([r['relative_rmse_gain'] for r in rows]),
        total_mse_gain_sum=mse_gain_sum, total_mean_local_rmse_gain_sum=sum(r['rmse_gain'] for r in rows),
        aggregate_mse_reduction=mse_gain_sum/baseline_mse_sum,
        aggregate_mean_local_rmse_reduction=sum(r['rmse_gain'] for r in rows)/baseline_rmse_sum,
        aligned_rmsd_change=stats([r['aligned_rmsd_change'] for r in rows]),
        aligned_rmsd_relative_change=stats([r['aligned_rmsd_change']/r['baseline']['aligned_rmsd'] for r in rows]),
        max_frozen_baseline_aligned_rmsd_ratio=max(s['after']['aligned_rmsd']/r['baseline']['aligned_rmsd'] for r in rows for s in r['steps']),
        continuous_chirality_change=stats([r['final']['continuous_chiral']-r['baseline']['continuous_chiral'] for r in rows]),
        repaired_inversions_sum=sum(r['baseline']['inversions']-r['final']['inversions'] for r in rows),
        violation_counts=violation_counts, horizons=horizons, benefit_after_k=late,
        first_zero_step=dict(collections_counter(str(r['first_zero_step']) for r in rows)),
        nonzero_step_distribution=dict(collections_counter(str(r['nonzero_steps']) for r in rows)),
        implicit_zero_tail_steps=sum(r['implicit_zero_tail_steps'] for r in rows),
        action_rms=stats([s['action_rms'] for s in steps]), action_max=stats([s['action_max'] for s in steps]),
        latency_seconds=stats([s['latency_seconds'] for s in steps]),
        convex_solve_budget_hits=sum(not value for s in steps for value in s['shield']['convex_converged']),
        rejected_unsafe_proposals=sum(s['shield']['unsafe_trials'] for s in steps))


def collections_counter(values):
    import collections
    return collections.Counter(values)


def decision(rows, throughput):
    thresholds = e.CONFIG['decision']
    output = {}
    for portion in ('reference', 'holdout_A', 'holdout_B'):
        selected = [r for r in rows if r['portion'] == portion]
        total = aggregate(selected)
        cells = {}
        for length in e.STRATA:
            for condition in e.CONDITIONS:
                subset = [r for r in selected if r['length_bin'] == length and r['condition'] == condition]
                result = aggregate(subset)
                cells[f'{length}|{condition}'] = dict(identities=result['identities'],
                    useful_fraction=result['useful_identity_fraction'], mean_mse_gain=result['mean_relative_local_mse_gain'],
                    passed=result['useful_identity_fraction'] >= thresholds['every_length_condition_useful_fraction_min']
                        and result['mean_relative_local_mse_gain'] >= thresholds['every_length_condition_mean_relative_local_mse_gain_min'])
        output[portion] = dict(useful_identity_fraction=total['useful_identity_fraction'],
            mean_relative_local_mse_gain=total['mean_relative_local_mse_gain'], cells=cells,
            passed=not any(total['violation_counts'].values())
                and total['useful_identity_fraction'] >= thresholds['identity_useful_fraction_min']
                and total['mean_relative_local_mse_gain'] >= thresholds['mean_relative_local_mse_gain_min']
                and all(v['passed'] for v in cells.values()))
    safe = not any(aggregate(rows)['violation_counts'].values())
    feasible = throughput['cache_covered_projected_teacher_wall_hours_8_workers'] <= thresholds['projected_cached_eligible_corpus_wall_hours_max']
    passed = safe and feasible and output['holdout_A']['passed'] and output['holdout_B']['passed']
    return dict(passed=passed, zero_hard_safety_violations=safe, feasible_throughput=feasible,
        portions=output, corpus_frozen_for_student=passed, student_training_started=False,
        full_corpus_generation_started=False,
        next_action='frozen pilot corpus available for a separately authorized non-privileged student pilot' if passed else
            'stop; diagnose failed strata and throughput without changing the frozen teacher')


def main():
    e.pin_parent()
    selection = e.validate_selection()
    candidate = json.loads((ROOT / 'candidate_manifest.json').read_text())
    for path, digest in candidate['input_sha256'].items():
        assert e.sha(path) == digest, path
    for filename, digest in candidate['cache_shard_sha256'].items():
        assert e.sha(e.CACHE / filename) == digest, filename
    for row in selection['identities']:
        assert e.sha(e.REPO / row['source_path']) == row['source_sha256']
    records = []
    selection_digest = e.sha(ROOT / 'selection.json')
    accepted_runners = e.acceptable_runner_sources()
    validator_digest = e.sha(e.__file__)
    for row in selection['identities']:
        path = ROOT / 'trajectories' / f"identity_{row['index']:04d}.json"
        record = json.loads(path.read_text())
        assert record['runner_sha256'] in accepted_runners
        assert record['selection_sha256'] == selection_digest
        assert record['input_archive_sha256'] == row['input_archive_sha256']
        assert record['archive_sha256'] == e.sha(path.with_suffix('.npz'))
        verification = json.loads((ROOT / 'validation' / path.name).read_text())
        assert verification['passes'] and verification['archive_sha256'] == record['archive_sha256']
        assert verification['record_sha256'] == e.sha(path)
        assert verification['validator_source_sha256'] == validator_digest
        records.append(record)
    assert len(records) == 2000
    rows = condition_rows(records)
    assert len(rows) == 6000
    summary = aggregate(rows)
    assert not any(summary['violation_counts'].values()), summary['violation_counts']
    groups = {dimension:{key:aggregate([r for r in rows if str(r[dimension]) == key])
        for key in sorted({str(r[dimension]) for r in rows})}
        for dimension in ('length_bin', 'condition', 'portion', 'metadata_cell')}
    cells = {f'{s}|{c}':aggregate([r for r in rows if r['length_bin'] == s and r['condition'] == c])
             for s in e.STRATA for c in e.CONDITIONS}
    # Exact per-residue norm distribution from compact accepted action archives.
    norms = []
    for record in records:
        with np.load(ROOT / 'trajectories' / f"identity_{record['index']:04d}.npz", allow_pickle=False) as archive:
            norms.append(np.linalg.norm(archive['delta'], axis=-1)[archive['eligibility']])
    all_norms = np.concatenate(norms)
    summary['per_residue_action_norms'] = dict(count=len(all_norms), mean=float(all_norms.mean()),
        quantiles={str(q):float(np.quantile(all_norms, q)) for q in (0., .01, .1, .25, .5, .75, .9, .99, 1.)},
        zero_fraction=float(np.mean(all_norms == 0.)))
    preparation_results = [json.loads(p.read_text()) for p in (ROOT / 'inputs').glob('*.json')]
    preparation_cpu = sum(r['preparation_cpu_seconds'] for r in preparation_results)
    worker_cpu = sum(r['worker_cpu_seconds'] for r in records)
    checkpoint_validation_CPU = sum(json.loads(p.read_text()).get('cpu_seconds', 0.)
        for p in (ROOT / 'validation').glob('*.json') if json.loads(p.read_text())['reused_checkpoint'])
    worker_cpu += checkpoint_validation_CPU
    worker_wall = sum(r['worker_wall_seconds'] for r in records)
    reused_validation_wall = {p.stem:json.loads(p.read_text()).get('wall_seconds', 0.)
        for p in (ROOT / 'validation').glob('*.json') if json.loads(p.read_text())['reused_checkpoint']}
    worker_wall += sum(reused_validation_wall.values())
    runs = json.loads((ROOT / 'run_history.json').read_text())
    run_wall = sum(r['wall_seconds'] for r in runs)
    throughput = dict(GPU_wall_seconds=0., GPU_used=False, preparation_wall_seconds=selection['preparation_wall_seconds'],
        preparation_CPU_seconds=preparation_cpu, teacher_generation_wall_seconds=run_wall,
        teacher_worker_CPU_seconds=worker_cpu, teacher_worker_wall_seconds_sum=worker_wall,
        reused_checkpoint_independent_validation_CPU_seconds=checkpoint_validation_CPU,
        reused_checkpoint_independent_validation_wall_seconds_sum=sum(reused_validation_wall.values()),
        CPU_accounting_caveat='completed trajectory CPU plus checkpoint revalidation; interrupted in-flight work without a durable record is not included',
        identity_throughput_per_hour=2000/run_wall*3600, trajectory_throughput_per_hour=6000/run_wall*3600,
        structural_step_throughput_per_second=summary['evaluated_steps']/run_wall,
        wall_time_includes_post_archive_independent_safety_verification=True,
        end_to_end_wall_seconds=selection['preparation_wall_seconds']+run_wall,
        CPU_wall_projection_assumption='observed 8-worker effective throughput, reweighted by corpus length counts; no new denoiser inference included')
    covered_counts = candidate['candidates_by_length']
    catalog_counts = candidate['structural_train_by_length']
    per_length_cost = {s:sum(r['worker_wall_seconds'] + reused_validation_wall.get(f"identity_{r['index']:04d}", 0.)
        for r in records if r['length_bin'] == s)/400 for s in e.STRATA}
    candidate_by_id = {r['sample_id']:r for r in candidate['candidates']}
    prep_by_length = {s:[r for r in preparation_results if candidate_by_id[r['identity']]['stratum'] == s] for s in e.STRATA}
    preparation_effective_workers = sum(r['preparation_wall_seconds'] for r in preparation_results)/selection['preparation_wall_seconds']
    prep_cost = {s:float(np.mean([r['preparation_wall_seconds'] for r in prep_by_length[s]])) for s in e.STRATA}
    effective_workers = worker_wall/run_wall
    covered_seconds = sum(covered_counts[s]*per_length_cost[s] for s in e.STRATA)
    catalog_seconds = sum(catalog_counts[s]*per_length_cost[s] for s in e.STRATA)
    throughput.update(effective_concurrent_workers=effective_workers,
        per_identity_three_condition_worker_wall_seconds_by_length=per_length_cost,
        cache_covered_candidate_identities=candidate['all_condition_cached_count'],
        structural_TRAIN_catalog_identities=candidate['structural_train_catalog_count'],
        verified_TRAIN_source_cohort_identities=candidate['verified_train_source_cohort_count'],
        cache_covered_projected_teacher_worker_hours=covered_seconds/3600,
        cache_covered_projected_teacher_wall_hours_8_workers=covered_seconds/effective_workers/3600,
        entire_structural_TRAIN_catalog_projected_teacher_worker_hours=catalog_seconds/3600,
        entire_structural_TRAIN_catalog_projected_teacher_wall_hours_8_workers=catalog_seconds/effective_workers/3600,
        P0_preparation_effective_workers=preparation_effective_workers,
        P0_preparation_worker_wall_seconds_by_length=prep_cost,
        cache_covered_projected_P0_wall_hours_4_workers=sum(covered_counts[s]*prep_cost[s] for s in e.STRATA)/preparation_effective_workers/3600,
        catalog_projected_P0_wall_hours_4_workers=sum(catalog_counts[s]*prep_cost[s] for s in e.STRATA)/preparation_effective_workers/3600,
        projection_caveat='catalog projection is conditional on teacher-compatible input eligibility; full exact eligibility is not measured outside pilot, and absent denoiser-cache generation cost is excluded')
    gate = decision(rows, throughput)
    results = dict(summary=summary, by=groups, length_condition_cells=cells, throughput=throughput, decision=gate,
        selection_sha256=e.sha(ROOT / 'selection.json'), protocol_sha256=e.sha(ROOT / 'protocol.json'),
        runner_sha256=e.sha(e.__file__), report_source_sha256=e.sha(__file__),
        parent_freeze_sha256=e.sha(ROOT / 'parent_freeze.json'),
        selection_diversity={k:selection[k] for k in ('selected_by_length', 'selected_by_portion', 'unique_sequence_clusters', 'unique_split_groups')},
        eligibility_exclusions=json.loads((ROOT / 'eligibility_exclusions.json').read_text()))
    e.pilot.atomic_json(ROOT / 'results.json', results)
    write_report(results)
    if gate['passed']:
        corpus_files = {str(p):e.sha(p) for folder in ('inputs', 'trajectories') for p in (ROOT/folder).glob('*.npz')}
        e.pilot.atomic_json(ROOT / 'CORPUS_FROZEN.json', dict(status='frozen_for_nonprivileged_student_pilot',
            files=corpus_files, selection_sha256=e.sha(ROOT/'selection.json'), results_sha256=e.sha(ROOT/'results.json'),
            student_input_rule='P_t and masks/eligibility only; exclude target, source denoiser metadata and privileged metrics from model features',
            split_rule='all conditions/states of one identity and split_group_id remain together',
            student_training_started=False))
    files = {str(p):e.sha(p) for p in ROOT.rglob('*') if p.is_file()
        and p.name != 'COMPLETED.json' and not any(f'/{s}/' in str(p) for s in ('logs', 'tmp', 'cache'))}
    e.pilot.atomic_json(ROOT/'COMPLETED.json', dict(status='complete', files=files, identities=2000, trajectories=6000,
        decision_passed=gate['passed'], teacher_frozen=True, no_SLSQP=True, no_student_training=True, no_full_corpus=True))
    print(json.dumps(dict(summary=summary, throughput=throughput, decision=gate), indent=2))


def write_report(result):
    a, performance, gate = result['summary'], result['throughput'], result['decision']
    lines = ['# E010 privileged structural teacher diversity scale-out pilot v1', '',
        'The completed 73f3330 constraint-projected benchmark is frozen. This is a new TRAIN diversity pilot, '
        'not a gold-surrogate benchmark. The one-relinearization teacher source and all solver/projection '
        'thresholds remain unchanged. No SLSQP, neural training, full-corpus generation, or student training ran.', '',
        '## Cohort, inputs, and held-out portions', '',
        'Exactly 2,000 eligible TRAIN identities, 400 in each existing length bin, with conditions 50/250/450 '
        '(6,000 trajectories). Selection is deterministic: round-robin experimental-method/resolution cells '
        'and sequence-cluster representatives within each length bin, with SHA256 rankings. '
        'The existing 20-identity Phase4D panel is excluded. Non-TRAIN identities and split groups are excluded. '
        'Only candidates with all three cache conditions and all frozen baseline safety gates are eligible. '
        'No teacher-repair metric affects selection. Eligibility exclusions are recorded explicitly.', '',
        'Existing TRAIN denoiser-cache source/target bytes are hash-verified. P0 is generated by the unchanged '
        'Phase4B global checkpoint in CPU float32 with two threads, matching the historical input lineage. '
        'Teacher geometry and safety are CPU float64 on one thread. Three archived examples passed bitwise '
        'P0 and teacher-action parity. A preflight-only one-to-two-thread input-inference correction is recorded '
        'in preflight_amendment.json; teacher settings did not change.', '',
        'An archive-validation serialization bug was corrected during generation: JSON turns tuple-valued '
        'constraint-slice metadata into lists, so reconstructed metric dictionaries now compare canonical '
        'JSON representations. Existing P0/delta archives were preserved and independently validated. '
        'archive_validation_amendment.json and code/runner_before_archive_fix.py pin the prior runner. '
        'The fix changes validation/restart bookkeeping only; the teacher and its thresholds are unchanged. '
        'validation/ contains a separate pass certificate for every identity.', '',
        'TRAIN pilot portions are assigned by split-group hash: 60% of hash buckets map to reference, '
        '20% to holdout_A and 20% to holdout_B. Actual identity proportions can differ because group sizes vary; '
        'the exact counts are reported below. All conditions/states of an identity and its split group remain together. '
        'These are held-out reporting portions of TRAIN, not the structural validation/test splits. '
        'The teacher is already frozen, and no portion is used for tuning.', '',
        'Selection diversity: '+json.dumps(result['selection_diversity'], sort_keys=True)+'.', '',
        '## Repair and exact safety', '',
        f"Useful nonzero correction reaches {100*a['useful_identity_fraction']:.2f}% of identities and "
        f"{100*a['useful_trajectory_fraction']:.2f}% of identity/condition trajectories. Useful means nonzero "
        'action plus normalized rollout local-MSE repair >1e-6; this is a reporting threshold only.', '',
        f"Mean per-trajectory local-MSE reduction is {100*a['mean_relative_local_mse_gain']:.3f}%; "
        f"ratio-of-sums MSE reduction is {100*a['aggregate_mse_reduction']:.3f}%. "
        f"Ratio-of-sums mean-local-RMSE reduction is {100*a['aggregate_mean_local_rmse_reduction']:.3f}%.", '',
        '| Horizon | Mean relative local-MSE improvement | Aggregate MSE improvement | Mean relative local-RMSE improvement |',
        '|---|---:|---:|---:|']
    for k, value in a['horizons'].items():
        lines.append(f"| {k} | {100*value['relative_mse_gain']['mean']:.3f}% | {100*value['aggregate_mse_reduction']:.3f}% | {100*value['relative_mean_local_rmse_gain']['mean']:.3f}% |")
    lines += ['', '| Length | Useful identities | Useful trajectories | Mean local-MSE gain |', '|---|---:|---:|---:|']
    for key, value in result['by']['length_bin'].items():
        lines.append(f"| {key} | {100*value['useful_identity_fraction']:.2f}% | {100*value['useful_trajectory_fraction']:.2f}% | {100*value['mean_relative_local_mse_gain']:.3f}% |")
    lines += ['', '| Condition | Useful identities | Mean local-MSE gain |', '|---|---:|---:|']
    for key, value in result['by']['condition'].items():
        lines.append(f"| {key} | {100*value['useful_identity_fraction']:.2f}% | {100*value['mean_relative_local_mse_gain']:.3f}% |")
    lines += ['', 'All accepted states were checked online and independently rechecked from compact archived '
        'P0+deltas before worker completion. Frozen gates retain the 1e-8 normalized global tolerance and literal '
        'nonpositive quartet constraints; eligibility zeros and physical residue cap <=0.10 A are exact. '
        'Safety/objective violation counts: '+json.dumps(a['violation_counts'], sort_keys=True)+'.', '',
        f"Aligned RMSD change: mean {a['aligned_rmsd_change']['mean']:.6f} A, "
        f"median {a['aligned_rmsd_change']['median']:.6f} A; maximum frozen-baseline ratio "
        f"{a['max_frozen_baseline_aligned_rmsd_ratio']:.12f}. "
        f"Mean continuous-chirality change: {a['continuous_chirality_change']['mean']:.6g}; "
        f"total inversions repaired: {a['repaired_inversions_sum']}.", '',
        'Action-norm distribution over eligible residue actions: '+json.dumps(a['per_residue_action_norms'], sort_keys=True)+'.', '',
        'First exact-zero step distribution (`None` means all 16 steps evaluated without an exact zero): '
        +json.dumps(a['first_zero_step'], sort_keys=True)+'. '
        f"Implicit zero-tail steps: {a['implicit_zero_tail_steps']}. The teacher has no time/step input: an exact-zero "
        'action leaves the complete state unchanged, so deterministic future outputs repeat. Only exact zeros '
        'terminate; no near-zero or low-repair truncation was added.', '',
        '## Late-step benefit', '', '| After step | Additional baseline-normalized MSE repair | Share of total MSE repair | Share of inference cost |', '|---|---:|---:|---:|']
    for k, value in a['benefit_after_k'].items():
        fraction = value['fraction_of_total_mse_repair']
        lines.append(f"| {k} | {100*value['aggregate_additional_mse_reduction']:.3f}% | {100*fraction:.3f}% | {100*value['fraction_of_measured_inference_cost']:.3f}% |")
    late12 = a['benefit_after_k']['12']['fraction_of_total_mse_repair']
    lines += ['', ('K=16 retains measurable late benefit: steps 13-16 supply ' if late12 >= .05 else
                  'Late benefit is small: steps 13-16 supply ')+f'{100*late12:.3f}% of total MSE repair. '
        'This describes the cost/benefit of K=16; the teacher horizon was not changed.', '',
        '## Throughput and full-TRAIN cost projection', '',
        f"Teacher generation wall time: {performance['teacher_generation_wall_seconds']/3600:.3f} h; "
        f"summed worker CPU time: {performance['teacher_worker_CPU_seconds']/3600:.3f} h; "
        f"throughput: {performance['identity_throughput_per_hour']:.2f} identities/h or "
        f"{performance['trajectory_throughput_per_hour']:.2f} trajectories/h. "
        f"P0/eligibility preparation wall time: {performance['preparation_wall_seconds']/3600:.3f} h. "
        'GPU wall time is zero. Teacher wall time includes independent archive safety verification '
        'and elapsed time of the interrupted validator run, so this is the observed pilot cost rather '
        'than an uninterrupted steady-state timing. CPU accounting includes completed trajectories '
        'and checkpoint revalidation; interrupted in-flight work without a durable record is excluded.', '',
        f"Length-reweighted projection for the {performance['cache_covered_candidate_identities']} all-condition "
        f"cache-covered TRAIN candidates: {performance['cache_covered_projected_teacher_wall_hours_8_workers']:.3f} h "
        f"wall, {performance['cache_covered_projected_teacher_worker_hours']:.3f} worker-hours. "
        f"Projection for the {performance['structural_TRAIN_catalog_identities']} structural TRAIN catalog identities: "
        f"{performance['entire_structural_TRAIN_catalog_projected_teacher_wall_hours_8_workers']:.3f} h wall, "
        f"{performance['entire_structural_TRAIN_catalog_projected_teacher_worker_hours']:.3f} worker-hours. "
        'These assume pilot-like teacher-compatible eligibility, all three conditions, observed zero tails, and '
        'observed effective eight-worker throughput. Full exact input eligibility is not measured outside the '
        'pilot. New denoiser-cache generation for uncached identities is excluded, so the catalog estimate '
        'is a conditional teacher-only cost, not a full-pipeline promise. No full corpus was generated.', '',
        f"Additional frozen-global P0/eligibility preparation projection: "
        f"{performance['cache_covered_projected_P0_wall_hours_4_workers']:.3f} h for the cache-covered candidates, "
        f"{performance['catalog_projected_P0_wall_hours_4_workers']:.3f} h for the catalog at measured four-worker throughput. "
        'These still exclude producing absent denoiser cache entries.', '',
        '## Decision gate', '',
        'The gate was frozen before generation: zero hard-safety violations; >=70% useful identities and '
        '>=5% mean local-MSE gain in both held-out portions; >=50% useful identities and >=1% mean local-MSE '
        'gain in every held-out length/condition cell; cached-eligible projected wall time <=24 h.', '',
        '**Gate '+('passed' if gate['passed'] else 'failed')+'**. '+gate['next_action']+'.', '']
    for portion, value in gate['portions'].items():
        failed = {k:v for k,v in value['cells'].items() if not v['passed']}
        lines.append(f"{portion}: useful identities {100*value['useful_identity_fraction']:.2f}%, "
            f"mean local-MSE gain {100*value['mean_relative_local_mse_gain']:.3f}%, passed={value['passed']}. "
            'Failed cells: '+json.dumps(failed, sort_keys=True)+'.')
    failed_cells = sorted({cell for portion in ('holdout_A', 'holdout_B')
        for cell, value in gate['portions'][portion]['cells'].items() if not value['passed']})
    if failed_cells:
        lines += ['', 'Failed-cell diagnosis uses the full pilot cell, with held-out pass/fail listed above. '
            'Unsafe proposals were rejected, so high rejection rates describe shielding pressure rather than accepted safety violations.', '',
            '| Failed cell | Useful trajectories | Exact zero at first step | Convex budget hits | Unsafe proposal fraction | Mean action RMS |',
            '|---|---:|---:|---:|---:|---:|']
        for cell in failed_cells:
            value = result['length_condition_cells'][cell]
            step_count = value['evaluated_steps']
            lines.append(f"| {cell} | {100*value['useful_trajectory_fraction']:.2f}% | "
                f"{100*value['zero_at_first_step_fraction']:.2f}% | "
                f"{100*value['convex_solve_budget_hits']/(2*step_count):.2f}% | "
                f"{100*value['rejected_unsafe_proposals']/(8*step_count):.2f}% | {value['action_rms']['mean']:.6f} A |")
    lines += ['', 'If cells fail, the fixed teacher is not broadened or retuned: low repair may reflect '
        'exact-zero shielding near strict quartet/chirality boundaries, finite-step curvature, or the frozen '
        'convex iteration limit. Per-cell zero tails, action norms, budget hits, rejected-proposal counts and '
        'repair distributions in results.json provide the diagnosis. Any decision failure stops downstream '
        'corpus promotion; no student is trained automatically.', '',
        '## Artifacts and reconstruction', '',
        'Everything generated is on D: under '+str(ROOT)+'. '
        'selection.json pins exact identities, conditions, source/cache hashes, metadata and portions. '
        'candidate_manifest.json and eligibility_exclusions.json preserve eligibility accounting. '
        'inputs/<identity>.npz stores three P0/source arrays, one shared target and mask. '
        'trajectories/identity_*.npz stores variable-length accepted deltas, eligibility, condition offsets and '
        'no redundant P_t arrays. Starting from the referenced float32 P0 cast to float64, add deltas in stored '
        'order; beyond the first exact zero, the state remains unchanged through K=16. '
        'JSON records contain baseline/step objectives, aligned RMSD, chirality/inversions/assessability, '
        'norms, solver/check outcomes and timing. results.json groups length, condition, metadata and pilot portions. '
        'CORPUS_FROZEN.json is emitted only if the decision gate passes. COMPLETED.json pins the completed '
        'experiment. Teacher-visible target X is never authorized as a future student input.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
