"""Raw held-out directional correctness and feasibility of controlled variants."""
import collections
import json
import numpy as np
import train_e010_student_direction_observability_v1 as t
e=t.e


def auc(values,labels):
    from scipy.stats import rankdata
    y=np.asarray(labels,bool);n=int(y.sum());m=len(y)-n
    return float((rankdata(values)[y].sum()-n*(n+1)/2)/(n*m)) if n and m else None


def summary(rows,step=None):
    one=[s for r in rows for s in (r['one_step'] if step is None else [r['one_step'][step-1]])]
    roll=[s for r in rows for s in (r['rollouts']['raw']['steps'] if step is None else [r['rollouts']['raw']['steps'][step-1]])]
    student=sum(r['baseline']['mean_local_rmse']-(r['rollouts']['raw']['final'] if step is None else r['rollouts']['raw']['steps'][step-1]['after'])['mean_local_rmse'] for r in rows)
    teacher=sum(r['baseline']['mean_local_rmse']-(r['teacher_final'] if step is None else r['one_step'][step-1]['teacher_after'])['mean_local_rmse'] for r in rows)
    one_student=sum(s['before']['mean_local_rmse']-s['after']['mean_local_rmse'] for s in one)
    one_teacher=sum(s['before']['mean_local_rmse']-s['teacher_after']['mean_local_rmse'] for s in one)
    d=dict(identities=len({r['identity'] for r in rows}),trajectories=len(rows),raw_K16_retention=student/teacher if teacher else None,
        raw_one_step_retention=one_student/one_teacher if one_teacher else None,
        teacher_action_cosine=e.old.frozen.old.statistics([s['action_cosine'] for s in one if s['action_cosine'] is not None]),
        teacher_action_RMS_error=e.old.frozen.old.statistics([s['action_rms_difference'] for s in one]),
        directional_error_margin_failure_AUC=auc([s['directional_risk_ratio'] for s in one],[s['hard_failed'] for s in one]))
    for mode,states in [('one_step',one),('rollout',roll)]:
        d[mode]=dict(states=len(states),hard_failed_states=sum(s['hard_failed'] for s in states),hard_failure_rate=float(np.mean([s['hard_failed'] for s in states])),
            new_inversions=sum(s['after']['safety']['quartets']['new_inversions'] for s in states),
            assessability_losses=sum(s['after']['safety']['quartets']['assessability_lost'] for s in states),
            cap_violations=sum(s['cap_violations'] for s in states),eligibility_violations=sum(s['eligibility_violations'] for s in states),
            exact_zero_rate=float(np.mean([s['exact_zero'] for s in states])),near_zero_rate=float(np.mean([s['near_zero'] for s in states])),
            policy_seconds=e.old.frozen.old.statistics([s['policy_seconds'] for s in states]))
    d['mean_aligned_RMSD_change']=float(np.mean([(r['rollouts']['raw']['final'] if step is None else r['rollouts']['raw']['steps'][step-1]['after'])['aligned_rmsd']-r['baseline']['aligned_rmsd'] for r in rows]))
    return d


def prepare_decision():
    protocol=dict(written_before_phase_C_raw_test_evaluation=True,selection_split='validation',test_used_for_selection=False,
        material_context_effect='mean teacher-action cosine gain >=0.02, raw K16 unsafe-state rate decrease >=0.10 absolute, raw K16 repair retention >=90% of reproduced baseline; unsafe-rate effect positive in every length bin and condition',
        minimal_context_priority=['B','C','D','E'],
        preliminary_deployable_screen='raw K16 repair retention >=0.50 and unsafe-state rate <=0.01; exact frozen gates retain their existing tolerances',
        no_automatic_followup_training=True)
    if (t.ROOT/'decision_protocol.json').exists():assert json.loads((t.ROOT/'decision_protocol.json').read_text())==protocol
    else:e.write(t.ROOT/'decision_protocol.json',protocol)


def main():
    assert (t.ROOT/'RUNS_COMPLETED.json').exists();result={'summaries':{},'strata':[],'by_identity':[],'training':{}}
    for variant in t.VARIANTS:
        result['summaries'][variant]={};result['training'][variant]=json.loads((t.ROOT/f'{variant}_training.json').read_text())
        for split in ('validation','test'):
            data=json.loads((t.ROOT/'evaluation'/f'{variant}_{split}.json').read_text());rows=data['rows']
            result['summaries'][variant][split]=summary(rows)
            for dimension,key in [('length','length_bin'),('condition','condition'),('identity','identity')]:
                groups=collections.defaultdict(list)
                for row in rows:groups[row[key]].append(row)
                for label,subset in groups.items():
                    entry=dict(variant=variant,split=split,dimension=dimension,label=label,**summary(subset))
                    result['by_identity' if dimension=='identity' else 'strata'].append(entry)
            for step in range(1,17):result['strata'].append(dict(variant=variant,split=split,dimension='step',label=step,**summary(rows,step)))
    baseline=result['summaries']['A']['validation'];gates={}
    for variant in ('B','C','D','E'):
        value=result['summaries'][variant]['validation']
        strata=[r for r in result['strata'] if r['variant']==variant and r['split']=='validation' and r['dimension'] in ('length','condition')]
        checks=dict(cosine=value['teacher_action_cosine']['mean']-baseline['teacher_action_cosine']['mean']>=.02,
            raw_feasibility=baseline['rollout']['hard_failure_rate']-value['rollout']['hard_failure_rate']>=.10,
            repair=value['raw_K16_retention']>=.9*baseline['raw_K16_retention'],
            stability=all(r['rollout']['hard_failure_rate']<next(b for b in result['strata'] if b['variant']=='A' and b['split']=='validation' and b['dimension']==r['dimension'] and b['label']==r['label'])['rollout']['hard_failure_rate'] for r in strata))
        gates[variant]=dict(material_effect=all(checks.values()),checks=checks,
            preliminary_deployable=value['raw_K16_retention']>=.5 and value['rollout']['hard_failure_rate']<=.01)
    passing=[v for v in ('B','C','D','E') if gates[v]['material_effect']]
    result['decision']=dict(context_gates=gates,minimal_useful_context=passing[0] if passing else None,selection_split='validation',test_used_for_selection=False)
    winner=passing[0] if passing else None
    if winner:
        value=result['summaries'][winner]['test'];base=result['summaries']['A']['test']
        checks=dict(cosine=value['teacher_action_cosine']['mean']-base['teacher_action_cosine']['mean']>=.02,
            raw_feasibility=base['rollout']['hard_failure_rate']-value['rollout']['hard_failure_rate']>=.10,
            repair=value['raw_K16_retention']>=.9*base['raw_K16_retention'])
        result['decision']['locked_context_test_confirmation']=dict(context=winner,checks=checks,confirmed=all(checks.values()),
            preliminary_deployable=value['raw_K16_retention']>=.5 and value['rollout']['hard_failure_rate']<=.01)
    e.write(t.ROOT/'results.json',result)
    lines=['','## Phase C: fixed-capacity controlled training','',
        'Phase B opened the preregistered validation gate for denoiser geometry. Five action-imitation-only controls retain width64, two blocks and 118,723 trainable parameters. Context uses a fixed rank-preserving projection before the existing blocks. No model receives X, teacher gradients or constraints. E uses source denoiser geometry alone; it excludes coarse latent, initial correction and privileged teacher-path history.', '',
        'A geometry-only reproduction is bitwise identical to fc76211 after 1,000 updates and selects the same checkpoint under the original monitor. The original identity-first sampler, seed, AdamW settings, update ladder and checkpoint rule are preserved. That checkpoint rule scores an X-based shield offline on the original 60 validation identities; it is never part of the raw evaluation or a deployment solution. This protocol precedence was recorded before context-effect scoring.', '',
        '| Variant | Inputs | Selected update | Test raw K16 retention | Unsafe raw states | Teacher-action cosine | Action RMS error (Å) | Directional-error AUC | Median latency (ms) |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    names={'A':'geometry','B':'+ condition','C':'+ sequence','D':'+ sequence + condition','E':'+ denoiser geometry'}
    for variant in t.VARIANTS:
        s=result['summaries'][variant]['test'];lines.append(f"| {variant} | {names[variant]} | {result['training'][variant]['best_update']} | {100*s['raw_K16_retention']:.2f}% | {100*s['rollout']['hard_failure_rate']:.2f}% | {s['teacher_action_cosine']['mean']:.4f} | {s['teacher_action_RMS_error']['mean']:.5f} | {s['directional_error_margin_failure_AUC']:.4f} | {1000*s['rollout']['policy_seconds']['median']:.3f} |")
    lines+=['', 'One-step action agreement and AUC are measured on frozen teacher states. AUC is the tie-aware discrimination of offline first-order error-to-teacher-margin risk against exact safety failure, with the same 1e-4 finite-difference convention as the feasibility pilot. It is descriptive, not an inference feature, fitted classifier or calibrated safety probability. Recursive safety counts include persistent failures at subsequent states.', '',
        '| Split | Variant | One-step repair retention | K16 repair retention | Unsafe one-step / rollout states | New inversions | Assessability losses | Cap / eligibility violations | Exact-zero / near-zero rollout rate |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for split in ('validation','test'):
        for variant in t.VARIANTS:
            s=result['summaries'][variant][split];r=s['rollout'];o=s['one_step']
            lines.append(f"| {split} | {variant} | {100*s['raw_one_step_retention']:.2f}% | {100*s['raw_K16_retention']:.2f}% | {o['hard_failed_states']} / {r['hard_failed_states']} | {r['new_inversions']} | {r['assessability_losses']} | {r['cap_violations']} / {r['eligibility_violations']} | {100*r['exact_zero_rate']:.2f}% / {100*r['near_zero_rate']:.2f}% |")
    lines+=['','| Test stratum | A repair / unsafe rate | E repair / unsafe rate | E mean action cosine |','|---|---:|---:|---:|']
    for row in result['strata']:
        if row['variant']!='E' or row['split']!='test' or (row['dimension']=='step' and row['label'] not in (1,2,4,8,12,16)):continue
        b=next(v for v in result['strata'] if v['variant']=='A' and v['split']=='test' and v['dimension']==row['dimension'] and v['label']==row['label'])
        lines.append(f"| {row['dimension']} {row['label']} | {100*b['raw_K16_retention']:.2f}% / {100*b['rollout']['hard_failure_rate']:.2f}% | {100*row['raw_K16_retention']:.2f}% / {100*row['rollout']['hard_failure_rate']:.2f}% | {row['teacher_action_cosine']['mean']:.4f} |")
    lines+=['', f"Validation-selected minimal useful context: **{result['decision']['minimal_useful_context']}**. Material effect requires ≥0.02 mean action-cosine gain, ≥10 percentage-point raw unsafe-state reduction, ≥90% of baseline repair retention, and positive feasibility effects in every length bin and condition. Test does not select the context. A separate preliminary deployment screen requires ≥50% repair retention and ≤1% unsafe states; a material context effect alone does not make a model deployable.", '',
        'Complete identity/length/condition/step results, aligned RMSD changes, action norms/errors, zero behavior, raw gates, latency and training wall/GPU time are in phase_C/results.json. Per-state exact metrics and compact raw action archives are in phase_C/evaluation/. No additional capacity, teacher data, SLSQP, full corpus or final training was launched.']
    lines+=['','## Diagnosis and stop','']
    if winner:
        confirmation=result['decision']['locked_context_test_confirmation']
        lines.append(f"The validation-selected context is {winner}; locked test confirmation is {confirmation['confirmed']}. Preliminary raw deployment safety screen: {confirmation['preliminary_deployable']}. A context gain is evidence that available information helps; it does not establish a safe deployable teacher surrogate when hard failures remain frequent.")
    else:
        lines.append('No context meets the combined raw directional-correctness/feasibility/repair gate. Phase B nevertheless shows informative denoiser geometry, so a null training result cannot be interpreted as proof that this information is absent. The tested embedding, local architecture, action-imitation objective and nonlinear feasible-policy boundaries can prevent explanatory information from becoming safe predictions. Smaller effects remain visible in the tables.')
    value=result['summaries']['E']['test'];base=result['summaries']['A']['test']
    lines.append(f"Denoiser geometry changes mean action cosine by {value['teacher_action_cosine']['mean']-base['teacher_action_cosine']['mean']:+.4f}, raw K16 unsafe-state rate by {100*(value['rollout']['hard_failure_rate']-base['rollout']['hard_failure_rate']):+.2f} percentage points, and teacher-repair retention by {100*(value['raw_K16_retention']-base['raw_K16_retention']):+.2f} percentage points on test. Its one-step versus K16 retention is {100*value['raw_one_step_retention']:.2f}% versus {100*value['raw_K16_retention']:.2f}%; the step strata describe recursive compounding separately.")
    lines.append('Finite-neighbor opposition remains evidence of residual privileged-policy ambiguity in these representations, not proof of identical-input collisions or fundamental unobservability. Sequence and condition alone show little explanatory effect; teacher-path history has a large association but is not a deployment-equivalent trajectory. Single-seed controls and the fixed context embedding limit architecture-level conclusions. Stop at this completed comparison without expanding capacity, context combinations, teacher data or final training.')
    report=e.ROOT/'REPORT.md';text=report.read_text();marker='\n## Phase C: fixed-capacity controlled training\n'
    if marker in text:text=text.split(marker)[0]
    report.write_text(text+'\n'.join(lines)+'\n')
    print(json.dumps(result['decision']),flush=True)


if __name__=='__main__':
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='prepare':prepare_decision()
    else:main()
