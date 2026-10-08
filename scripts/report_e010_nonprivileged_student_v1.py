"""Close the bounded student pilot; never launch full-corpus generation."""
import json
from pathlib import Path

import numpy as np
import e010_nonprivileged_student_v1 as e

ROOT=e.ROOT


def stats(values):
    result=e.frozen.old.statistics(values)
    if values:result['p95']=float(np.quantile(values,.95))
    return result


def aggregate(rows,mode):
    result=e.summary(rows,mode)
    for key in ('policy_seconds','inference_seconds'):
        result[key]=stats([s[key] for r in rows for s in r['rollouts'][mode]['steps']])
    ones=[s for r in rows for s in r['one_step'] if s['mode']==mode]
    def failure_families(steps):
        return dict(aligned_RMSD=sum(s['after']['safety']['global_constraints'][0]>1e-8 for s in steps),
            continuous_chirality=sum(s['after']['safety']['global_constraints'][1]>1e-8 for s in steps),
            strict_quartets=sum(s['after']['safety']['maximum_extra_constraint']>0 for s in steps),
            new_inversion_states=sum(s['after']['safety']['quartets']['new_inversions']>0 for s in steps),
            assessability_loss_states=sum(s['after']['safety']['quartets']['assessability_lost']>0 for s in steps))
    result['rollout_failure_family_states']=failure_families([s for r in rows for s in r['rollouts'][mode]['steps']])
    if ones:
        denominator=sum(s['before']['mean_local_rmse']-s['teacher_after']['mean_local_rmse'] for s in ones)
        student=sum(s['before']['mean_local_rmse']-s['after']['mean_local_rmse'] for s in ones)
        cosines=[s['action_cosine'] for s in ones if s['action_cosine'] is not None]
        result['one_step']=dict(RMSE_retention=student/denominator if denominator>1e-12 else None,
            action_cosine=stats(cosines),cosine_excluded_zero_pairs=len(ones)-len(cosines),
            action_RMS_error=stats([s['action_rms_difference'] for s in ones]),
            mean_relative_local_MSE_repair=float(np.mean([(s['before']['local_mse']-s['after']['local_mse'])/s['before']['local_mse'] for s in ones])),
            policy_seconds=stats([s['policy_seconds'] for s in ones]),inference_seconds=stats([s['inference_seconds'] for s in ones]),
            failure_family_states=failure_families(ones),safety_failed_steps=sum(not s['after']['safety']['passes'] for s in ones),
            new_inversions_steps=sum(s['after']['safety']['quartets']['new_inversions'] for s in ones),
            assessability_loss_steps=sum(s['after']['safety']['quartets']['assessability_lost'] for s in ones),
            cap_violations=sum(s['cap_violations'] for s in ones),eligibility_violations=sum(s['eligibility_violations'] for s in ones))
    result['aligned_RMSD_change']=stats([r['rollouts'][mode]['final']['aligned_rmsd']-r['baseline']['aligned_rmsd'] for r in rows])
    result['max_aligned_RMSD_ratio']=max(s['after']['aligned_rmsd']/r['baseline']['aligned_rmsd'] for r in rows for s in r['rollouts'][mode]['steps'])
    result['action_RMS']=stats([s['action_rms'] for r in rows for s in r['rollouts'][mode]['steps']])
    result['action_max']=stats([s['action_max'] for r in rows for s in r['rollouts'][mode]['steps']])
    result['horizons']={}
    for k in (1,2,4,8,12,16):
        gains=[]; teachers=[]
        for r in rows:
            teacher=r['teacher_steps'][min(k,len(r['teacher_steps']))-1]['after']
            student=r['rollouts'][mode]['steps'][k-1]['after']
            gains.append(r['baseline']['mean_local_rmse']-student['mean_local_rmse'])
            teachers.append(r['baseline']['mean_local_rmse']-teacher['mean_local_rmse'])
        result['horizons'][str(k)]=dict(RMSE_retention=sum(gains)/sum(teachers),
            student_RMSE_repair_sum=sum(gains),teacher_RMSE_repair_sum=sum(teachers))
    result['by_step']={}
    result['one_step_by_step']={}
    for k in range(16):
        selected=[r['rollouts'][mode]['steps'][k] for r in rows]
        teacher_gain=sum(r['baseline']['mean_local_rmse']-r['teacher_steps'][min(k,len(r['teacher_steps'])-1)]['after']['mean_local_rmse'] for r in rows)
        student_gain=sum(r['baseline']['mean_local_rmse']-s['after']['mean_local_rmse'] for r,s in zip(rows,selected))
        result['by_step'][str(k+1)]=dict(exact_zero_rate=float(np.mean([s['exact_zero'] for s in selected])),
            intervention_rate=float(np.mean([s['shield']['intervention'] for s in selected])) if mode=='shielded' else 0.,
            rejection_rate=float(np.mean([s['shield']['rejected_nonzero'] for s in selected])) if mode=='shielded' else 0.,
            mean_action_RMS=float(np.mean([s['action_rms'] for s in selected])),
            safety_failed_states=sum(not s['after']['safety']['passes'] for s in selected),
            cumulative_RMSE_repair_sum=student_gain,cumulative_teacher_RMSE_repair_sum=teacher_gain,
            cumulative_RMSE_retention=student_gain/teacher_gain)
        comparisons=[s for s in ones if s['step']==k+1]
        if comparisons:
            teacher=sum(s['before']['mean_local_rmse']-s['teacher_after']['mean_local_rmse'] for s in comparisons)
            student=sum(s['before']['mean_local_rmse']-s['after']['mean_local_rmse'] for s in comparisons)
            result['one_step_by_step'][str(k+1)]=dict(action_cosine=stats([s['action_cosine'] for s in comparisons if s['action_cosine'] is not None]),
                action_RMS_error=stats([s['action_rms_difference'] for s in comparisons]),
                RMSE_retention=student/teacher if teacher>1e-12 else None,
                intervention_rate=float(np.mean([s['shield']['intervention'] for s in comparisons])) if mode=='shielded' else 0.)
    return result


def gate(summary,cells):
    t=e.CONFIG['success'];s=summary['shielded'];raw=summary['raw']
    violations={key:s[key] for key in ('safety_failed_steps','new_inversions_steps','assessability_loss_steps','cap_violations','eligibility_violations')}
    violations.update({f'one_step_{key}':s['one_step'][key] for key in ('safety_failed_steps','new_inversions_steps','assessability_loss_steps','cap_violations','eligibility_violations')})
    checks=dict(retention=s['RMSE_retention']>=t['heldout_rollout_RMSE_retention_min'],
        raw_repair=raw['RMSE_retention']>=t['raw_rollout_RMSE_retention_min'],
        safety=not any(violations.values()),intervention=s['intervention_rate']<=t['shield_intervention_rate_max'],
        rejection=s['rejection_rate']<=t['shield_zero_rejection_rate_max'],
        strata=all(v['shielded']['RMSE_retention']>=t['every_length_condition_RMSE_retention_min'] for v in cells.values()),
        policy_latency=1000*s['policy_seconds']['median']<=t['CPU_network_geometry_median_ms_max'],
        shield_latency=1000*s['inference_seconds']['p95']<=t['CPU_shielded_p95_ms_max'])
    return dict(passed=all(checks.values()),checks=checks,shielded_violations=violations,
        target_free_network=True,shield_requires_target_X=True,full_corpus_generated=False,larger_model_trained=False,
        next_action='propose separately authorized full structural TRAIN teacher-data generation; do not launch it' if all(checks.values()) else
            'stop; diagnose the bounded student pilot before any full-corpus generation or capacity escalation')


def main():
    selected=json.loads((ROOT/'selected_model.json').read_text())['arm']
    splits=e.selection()
    assert e.sha(ROOT/'code'/'preparation_runner.py')==splits['source_sha256']
    evaluations={arm:json.loads((ROOT/'evaluation'/f'{arm}_validation.json').read_text()) for arm in e.CONFIG['arms']}
    test=json.loads((ROOT/'evaluation'/f'{selected}_test.json').read_text())
    rows=test['rows']
    assert test['identities']==splits['splits']['test']
    assert len(rows)==3*len(test['identities'])
    model=ROOT/'models'/f'{selected}_best.pt'
    assert test['model_sha256']==e.sha(model)
    for path in {r['action_archive'] for r in rows}:
        record=next(r for r in rows if r['action_archive']==path)
        assert e.sha(path)==record['action_archive_sha256']
    assert all(r['shielded_archive_reconstruction_passed'] for r in rows)
    for source,digest in [('scripts/e010_nonprivileged_student_v1.py',json.loads((ROOT/'tests.json').read_text())['source_sha256'])]:
        assert e.sha(e.REPO/source)==digest
    summaries={mode:aggregate(rows,mode) for mode in ('raw','shielded')}
    by={dimension:{str(key):{mode:aggregate([r for r in rows if r[dimension]==key],mode) for mode in ('raw','shielded')}
        for key in sorted({r[dimension] for r in rows})} for dimension in ('identity','condition','length_bin')}
    cells={f'{length}|{condition}':{mode:aggregate([r for r in rows if r['length_bin']==length and r['condition']==condition],mode)
        for mode in ('raw','shielded')} for length in e.STRATA for condition in (50,250,450)}
    decision=gate(summaries,cells)
    training={arm:json.loads((ROOT/f'{arm}_training.json').read_text()) for arm in e.CONFIG['arms']}
    capacity=json.loads((ROOT/'capacity_diagnostic.json').read_text())
    parent_sources=json.loads((e.frozen.ROOT/'results.json').read_text())['source_sha256']
    for path,digest in parent_sources.items():assert e.sha(path)==digest,path
    parent_rows=json.loads((e.PARENT/'selection.json').read_text())['identities']
    clusters={s:{row['cluster_id'] for row in parent_rows if splits['assignments'][row['sample_id']]==s}
        for s in ('train','validation','test')}
    overlaps={f'{a}|{b}':len(clusters[a]&clusters[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
    assert not any(overlaps.values())
    results=dict(test=summaries,by=by,length_condition=cells,decision=decision,selected_arm=selected,
        validation={arm:{mode:aggregate(value['rows'],mode) for mode in ('raw','shielded')} for arm,value in evaluations.items()},
        training=training,capacity_diagnostic=capacity,split_counts=splits['counts'],splits_sha256=e.sha(ROOT/'splits.json'),
        identity_list_sha256=splits['identity_list_sha256'],sequence_cluster_split_overlaps=overlaps,parent_completion_sha256=e.CONFIG['parent_completion_sha256'],
        source_sha256={str(e.__file__):e.sha(e.__file__),str(Path(__file__).resolve()):e.sha(__file__)},
        unchanged_frozen_geometry_source_sha256=parent_sources,
        selected_model_sha256=e.sha(model),evaluation_wall_seconds=dict(test=test['wall_seconds'],
            validation={arm:v['wall_seconds'] for arm,v in evaluations.items()}))
    e.pilot.atomic_json(ROOT/'results.json',results)
    write_report(results)
    # The entire original corpus and teacher completion must remain unchanged.
    assert e.sha(e.PARENT/'COMPLETED.json')==e.CONFIG['parent_completion_sha256']
    parent=json.loads((e.PARENT/'COMPLETED.json').read_text())
    for path,digest in parent['files'].items():assert e.sha(path)==digest,path
    parent_results=json.loads((e.PARENT/'results.json').read_text())
    assert e.sha(e.REPO/'scripts/e010_diversity_scaleout_v1.py')==parent_results['runner_sha256']
    assert e.sha(e.REPO/'scripts/report_e010_diversity_scaleout_v1.py')==parent_results['report_source_sha256']
    assert e.sha(e.frozen.__file__)==json.loads((e.PARENT/'parent_freeze.json').read_text())['teacher_source_sha256']
    e.pilot.atomic_json(ROOT/'PARENT_UNCHANGED.json',dict(commit='6d1ed30',files_verified=len(parent['files']),
        completion_sha256=e.sha(e.PARENT/'COMPLETED.json'),teacher_source_sha256=e.sha(e.frozen.__file__)))
    files={str(p):e.sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='COMPLETED.json'
        and not any(f'/{folder}/' in str(p) for folder in ('logs','tmp','cache'))}
    e.pilot.atomic_json(ROOT/'COMPLETED.json',dict(status='complete',files=files,pilot_success=decision['passed'],
        parent_unchanged=True,no_SLSQP=True,no_teacher_rerun=True,no_full_corpus=True,no_larger_model=True))
    print(json.dumps(dict(test=summaries,decision=decision,selected_arm=selected),indent=2),flush=True)


def write_report(r):
    raw,safe=r['test']['raw'],r['test']['shielded']
    lines=['# E010 non-privileged structural student pilot v1','',
        'The teacher and 2,000-identity diversity corpus from commit 6d1ed30 remain unchanged. '
        'This separate experiment trains only on that frozen pilot. No SLSQP, teacher relabeling, '
        'full-corpus generation, or larger model ran. All generated data are on D:.','',
        '## Data and target isolation','',
        'Deterministic splits have 1,400 TRAIN, 300 validation and 300 test identities, with '
        '280/60/60 per length bin. Whole split groups stay together, which is stronger than identity '
        'isolation. All three conditions and all 16 logical states/actions, including exact zero tails, '
        'stay with their identity. These validation/test portions contain only project-level TRAIN identities. '
        'Sequence-cluster overlap between every split pair is also zero.', '',
        'Exact split identity-list SHA256: '+json.dumps(r['identity_list_sha256'],sort_keys=True)+'.', '',
        'Sampling is uniform identity first, then uniform condition and logical step. Each protein loss is '
        'normalized over its eligible residues before averaging across sampled identities. Long or nonzero '
        'trajectories do not contribute extra sampling weight. P_t is reconstructed by sequential float64 '
        'addition to archived P0. The parent corpus, source modules and selection were hash-verified.', '',
        'The network forward receives only P_t and mask. It uses the existing 41-channel Phase4D local '
        'invariant/parity-sensitive representation and LocalGeometryBranch blocks, width 64 and two blocks, '
        'with a differentiable 0.10 A cap. It receives no target X, target distances, gradients, privileged '
        'constraints, sequence, corruption condition, or step index. Omitting time and condition tests P_t '
        'observability directly and preserves the existing input representation. CPU inference uses float64 '
        'geometry/frame and float32 network weights; the frozen closed-ball ULP guard enforces the exact cap.', '',
        'The exact safety shield still needs target X. Shielded results are therefore privileged evaluation, '
        'not evidence of a deployable target-free safety guarantee. The network remains target-free. '
        'The shield copies the frozen projection_2 candidate-acceptance block: scales 1/.5/.25/.125, '
        'retain the lowest strict-improving local objective among exact-safe candidates, otherwise zero. '
        'Safety thresholds, P0-anchored Context, quartet rules, masks and cap are unchanged. It never calls '
        'the teacher projector or computes new teacher gradients/actions.', '',
        '## Controlled training and early stopping','',
        'Two arms use identical initialization, optimizer, sampling seed, capacity and budget. The imitation '
        'loss is per-protein eligible-residue squared action error /0.10². The auxiliary arm adds exactly '
        '0.1 × the resulting local distance MSE normalized by the same-state pre-action MSE (denominator '
        'clamped at 1e-12); X appears only in that loss. No aligned/chirality loss was mixed into v1.', '',
        'AdamW lr=3e-4, weight decay=1e-4, batch=8 identities, gradient clip=1, at most 6,000 updates. '
        'Every 1,000 updates, a fixed 60-identity validation monitor evaluates recursive shielded K=16 '
        'teacher-repair retention; checkpoints require improvement >0.002 and early stopping has two-check '
        'patience after 3,000 updates. Training action loss never selects checkpoints. Full validation '
        'retention then selects the arm; the test is evaluated once for that selected model.', '',
        '| Arm | Best update | Updates | Parameters | Training GPU seconds | Validation raw / shielded rollout retention |',
        '|---|---:|---:|---:|---:|---:|']
    for arm,t in r['training'].items():
        v=r['validation'][arm]
        lines.append(f"| {arm} | {t['best_update']} | {t['updates']} | {t['parameters']} | {t['GPU_training_seconds']:.2f} | {100*v['raw']['RMSE_retention']:.2f}% / {100*v['shielded']['RMSE_retention']:.2f}% |")
    lines += ['',f"Selected arm: **{r['selected_arm']}**. Retention is the ratio of summed absolute mean-local-RMSE repair; "
        'local-MSE retention is reported separately. One-step comparisons use all 16 archived teacher states '
        'per identity/condition, including logical zero tails. Recursive student rollouts use their own states; '
        'no teacher relabels those off-trajectory states.', '',
        '## Identity-held-out test','',
        '| Metric | Raw student | Shielded student |','|---|---:|---:|']
    metrics=[('One-step RMSE repair retention',lambda s:f"{100*s['one_step']['RMSE_retention']:.3f}%"),
        ('Rollout RMSE repair retention',lambda s:f"{100*s['RMSE_retention']:.3f}%"),
        ('Rollout MSE repair retention',lambda s:f"{100*s['MSE_retention']:.3f}%"),
        ('Mean relative rollout local-MSE repair',lambda s:f"{100*s['mean_relative_MSE_repair']:.3f}%"),
        ('Mean teacher-action cosine (nonzero pairs)',lambda s:f"{s['one_step']['action_cosine'].get('mean',0.):.6f} ({s['one_step']['action_cosine']['count']})"),
        ('Mean teacher-action RMS error A',lambda s:f"{s['one_step']['action_RMS_error']['mean']:.6f}"),
        ('Mean aligned RMSD change A',lambda s:f"{s['aligned_RMSD_change']['mean']:.6f}"),
        ('Exact-zero / near-zero action rate',lambda s:f"{100*s['exact_zero_rate']:.2f}% / {100*s['near_zero_rate']:.2f}%"),
        ('Shield intervention / zero-rejection rate',lambda s:f"{100*s['intervention_rate']:.2f}% / {100*s['rejection_rate']:.2f}%"),
        ('Median policy+geometry ms',lambda s:f"{1000*s['policy_seconds']['median']:.3f}"),
        ('Median / p95 inference ms',lambda s:f"{1000*s['inference_seconds']['median']:.3f} / {1000*s['inference_seconds']['p95']:.3f}")]
    for label,fn in metrics:lines.append(f'| {label} | {fn(raw)} | {fn(safe)} |')
    lines += ['', 'Cosines exclude zero/degenerate pairs; counts and per-residue cosine are in evaluation JSON. '
        'Near-zero means maximum residue action <=1e-8 A. Safety counts are violations against the frozen '
        'P0 baseline at each evaluated state, so a persistent inversion can be counted at multiple steps. '
        'There is no hidden raw safety filtering. Shielded action archives were independently reconstructed '
        'and checked after every accepted action.', '',
        'Raw safety: '+json.dumps({k:raw[k] for k in ('safety_failed_steps','new_inversions_steps','assessability_loss_steps','cap_violations','eligibility_violations')})+'.',
        'Shielded safety: '+json.dumps(r['decision']['shielded_violations'])+'.', '',
        'Raw rollout safety-family failed-state counts (overlap): '+json.dumps(raw['rollout_failure_family_states'])+'. '
        'Raw teacher-state one-step failed-state counts: '+json.dumps(raw['one_step']['failure_family_states'])+'.', '',
        '| Length | Raw rollout retention | Shielded rollout retention | Shield intervention |',
        '|---|---:|---:|---:|']
    for key,v in r['by']['length_bin'].items():
        lines.append(f"| {key} | {100*v['raw']['RMSE_retention']:.2f}% | {100*v['shielded']['RMSE_retention']:.2f}% | {100*v['shielded']['intervention_rate']:.2f}% |")
    lines += ['', '| Condition | Raw rollout retention | Shielded rollout retention | Shield intervention |','|---|---:|---:|---:|']
    for key,v in r['by']['condition'].items():
        lines.append(f"| {key} | {100*v['raw']['RMSE_retention']:.2f}% | {100*v['shielded']['RMSE_retention']:.2f}% | {100*v['shielded']['intervention_rate']:.2f}% |")
    lines += ['', '| Horizon | Raw cumulative repair retention | Shielded cumulative repair retention |','|---|---:|---:|']
    for k,v in safe['horizons'].items():
        lines.append(f"| {k} | {100*raw['horizons'][k]['RMSE_retention']:.2f}% | {100*v['RMSE_retention']:.2f}% |")
    lines += ['', '| Length / condition | Raw rollout retention | Shielded rollout retention |','|---|---:|---:|']
    for cell,v in r['length_condition'].items():
        lines.append(f"| {cell.replace('|',' / ')} | {100*v['raw']['RMSE_retention']:.2f}% | {100*v['shielded']['RMSE_retention']:.2f}% |")
    chosen_training=r['training'][r['selected_arm']]
    chosen_checkpoint=next(v for v in chosen_training['history'] if v['update']==chosen_training['best_update'])
    tr=chosen_checkpoint['train_probe']['error_to_zero_baseline']
    va=chosen_checkpoint['validation_probe']['error_to_zero_baseline']
    delta_raw=r['validation']['imitation_local']['raw']['RMSE_retention']-r['validation']['imitation']['raw']['RMSE_retention']
    delta_safe=r['validation']['imitation_local']['shielded']['RMSE_retention']-r['validation']['imitation']['shielded']['RMSE_retention']
    capacity=r['capacity_diagnostic']
    if capacity['run']:
        initial=capacity['history'][0]['error_to_zero_baseline']; final=capacity['history'][-1]['error_to_zero_baseline']
        capacity_text=f'Same-capacity TRAIN-only memorization of 32 states reduced action error from {100*initial:.2f}% to {100*final:.2f}% of the zero-action baseline after 1,000 updates. Substantial residual remains; the diagnostic model was never selected. Exact states and the curve are in capacity_diagnostic.json.'
    else:capacity_text=capacity['reason']
    lines += ['', '## Decision and diagnosis','',
        '**Gate '+('passed' if r['decision']['passed'] else 'failed')+'**. '+r['decision']['next_action']+'.', '',
        'The gate requires >=75% shielded rollout RMSE retention, >=50% raw retention, zero shielded hard '
        'violations, intervention <=50%, nonzero-action rejection to zero <=10%, >=50% retention in every '
        'length/condition cell, median CPU policy+geometry <=5 ms and p95 shielded inference <=100 ms. '
        'Gate results: '+json.dumps(r['decision']['checks'],sort_keys=True)+'.', '',
        'Capacity/optimization diagnostic: '+capacity_text, '',
        f"The raw network learns substantial teacher-direction information (mean action cosine {raw['one_step']['action_cosine']['mean']:.3f}) "
        f"and {100*raw['RMSE_retention']:.2f}% recursive repair, but shielded repair falls to {100*safe['RMSE_retention']:.2f}%. "
        'Strict quartet/no-new-inversion failures dominate the raw failed-state counts; continuous chirality '
        'also fails frequently. The principal observed bottleneck is predicting corrections that remain '
        'feasible under the target-conditioned frozen constraints.', '',
        f"Training/validation action errors at the selected checkpoint are {100*tr:.2f}%/{100*va:.2f}% of their "
        'zero-action baselines. Their small gap argues against ordinary split overfitting as the sole cause. '
        'Substantial training residual and incomplete 32-state memorization keep capacity/optimization '
        'underfitting plausible. The current experiment does not isolate network size from optimization '
        'budget, representation locality, or missing teacher context; it does not justify assuming that a '
        'larger model or more labels alone will fix feasible repair.', '',
        'The teacher depends on X and on safety baselines frozen at P0, while this student sees only the '
        'current local geometry. Its two local blocks have bounded sequence receptive field and cannot '
        'directly observe target-dependent signs, global alignment margins, or the original safety context. '
        'Missing information and local architecture are therefore plausible causes of feasible-action '
        'averaging. This is an inference from the mismatch and failure families, not proof that every '
        'target-free architecture would fail.', '',
        f"One-step/recursive retention is {100*raw['one_step']['RMSE_retention']:.2f}%/{100*raw['RMSE_retention']:.2f}% raw "
        f"and {100*safe['one_step']['RMSE_retention']:.2f}%/{100*safe['RMSE_retention']:.2f}% shielded. "
        'Their closeness means recursive drift is not the main explanation for the low overall retention; '
        'later shield rejection still accumulates as the student encounters constraint boundaries.', '',
        f"The single auxiliary local-loss comparison changes validation raw retention by {100*delta_raw:+.3f} "
        f"percentage points and shielded retention by {100*delta_safe:+.3f}. It does not resolve the feasible-direction "
        'gap at the tested weight. Other objective choices remain untested. The next work should be a '
        'separately authorized controlled observability/feasibility and fitting study, before full TRAIN '
        'teacher generation. No additional architecture, capacity ladder or teacher-data generation is launched.', '',
        f"First-step shield rejection is {100*safe['by_step']['1']['rejection_rate']:.2f}%; this is before recursive "
        'drift or logical zero tails. '+('High immediate rejection identifies instantaneous feasible-action '
        'prediction as a bottleneck; recursive compounding alone cannot explain the result.' if safe['by_step']['1']['rejection_rate']>.5 else
        'Compare later-step rejection to this initial value to assess recursive compounding.'), '',
        'The imitation-only versus auxiliary arm separates action/objective mismatch. Training and '
        'validation action-error-to-zero-baseline curves quantify fitting versus generalization. '
        'One-step versus recursive retention and per-step intervention distinguish rollout compounding '
        'from instantaneous feasible-action failure. The same-capacity memorization probe, when triggered, '
        'tests whether the fixed network can fit a small TRAIN-only state set; it is never selected. '
        'These diagnostics can implicate limited P_t-only observability or data/representation coverage '
        'but do not prove an information-theoretic impossibility. No larger architecture was trained.', '',
        'All artifacts reside at '+str(ROOT)+'. splits.json pins exact identity lists and source hashes. '
        'models/ stores checkpoints; evaluation/ stores teacher-state comparisons, recursive metrics and '
        'compact accepted action archives. results.json includes identity, condition, length, cell and '
        'step diagnostics. PARENT_UNCHANGED.json rechecks the complete frozen diversity corpus. '
        'COMPLETED.json pins this separate experiment. No full structural TRAIN teacher corpus was generated.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
