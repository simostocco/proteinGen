"""Aggregate the preregistered offline feasibility-margin diagnostic."""
import collections
import json
import time
import numpy as np
import e010_student_feasibility_margin_v1 as e


def bucket():
    return dict(identities=set(),trajectories=0,roll_teacher=0.,roll_student=0.,one_teacher=0.,one_student=0.,
        roll_states=0,one_states=0,roll_failed=0,one_failed=0,roll_new_inversions=0,roll_assessability_loss=0,
        one_new_inversions=0,one_assessability_loss=0,roll_cap=0,one_cap=0,roll_eligibility=0,one_eligibility=0,
        roll_exact_zero=0,roll_near_zero=0,one_exact_zero=0,one_near_zero=0,aligned_change=0.,
        roll_failure_families=collections.Counter(),one_failure_families=collections.Counter())


def add(b,r,alpha,step=None):
    b['identities'].add(r['identity']);b['trajectories']+=1
    roll=r['rollouts'][alpha]['steps'];one=r['one_step'][alpha]
    final=roll[-1]['after'] if step is None else roll[step-1]['after']
    teacher=r['teacher_final'] if step is None else one[step-1]['teacher_after']
    b['roll_student']+=r['baseline']['mean_local_rmse']-final['mean_local_rmse']
    b['roll_teacher']+=r['baseline']['mean_local_rmse']-teacher['mean_local_rmse']
    b['aligned_change']+=final['aligned_rmsd']-r['baseline']['aligned_rmsd']
    if step is not None:roll=[roll[step-1]];one=[one[step-1]]
    for mode,states in [('roll',roll),('one',one)]:
        for s in states:
            b[f'{mode}_states']+=1;b[f'{mode}_failed']+=s['hard_failed']
            q=s['after']['safety']['quartets']
            b[f'{mode}_new_inversions']+=q['new_inversions'];b[f'{mode}_assessability_loss']+=q['assessability_lost']
            b[f'{mode}_cap']+=s['cap_violations'];b[f'{mode}_eligibility']+=s['eligibility_violations']
            b[f'{mode}_exact_zero']+=s['exact_zero'];b[f'{mode}_near_zero']+=s['near_zero']
            b[f'{mode}_failure_families'].update({k:int(v) for k,v in s['failures'].items()})
            if mode=='one':
                b['one_student']+=s['before']['mean_local_rmse']-s['after']['mean_local_rmse']
                b['one_teacher']+=s['before']['mean_local_rmse']-s['teacher_after']['mean_local_rmse']


def finalize(b):
    d=dict(b);d['identities']=len(b['identities'])
    d['rollout_retention']=b['roll_student']/b['roll_teacher'] if b['roll_teacher'] else None
    d['one_step_retention']=b['one_student']/b['one_teacher'] if b['one_teacher'] else None
    d['mean_aligned_rmsd_change']=b['aligned_change']/b['trajectories']
    for mode in ('roll','one'):
        for value in ('failed','exact_zero','near_zero'):
            d[f'{mode}_{value}_rate']=b[f'{mode}_{value}']/b[f'{mode}_states']
    return d


def stats(values):
    return e.old.frozen.old.statistics(values) if values else None


def auc(values,labels):
    # Tie-aware Mann-Whitney statistic; descriptive discrimination, no fitted model.
    x=np.asarray(values);y=np.asarray(labels,dtype=bool)
    order=np.argsort(x,kind='stable');xs=x[order];ys=y[order]
    total=0.;negative=0;i=0
    while i<len(x):
        j=i+1
        while j<len(x) and xs[j]==xs[i]:j+=1
        n=int((~ys[i:j]).sum());p=int(ys[i:j].sum());total+=p*(negative+.5*n);negative+=n;i=j
    return total/(y.sum()*(~y).sum()) if y.any() and (~y).any() else None


def main():
    assert (e.ROOT/'PHASE_AB_COMPLETED.json').exists(), 'Full phases A/B must finish before the decision gate'
    started=time.perf_counter(); groups=collections.defaultdict(bucket)
    geometry=collections.defaultdict(list); margins=collections.defaultdict(list)
    for i,path in enumerate(sorted((e.ROOT/'records').glob('*.json')),1):
        payload=json.loads(path.read_text())
        assert payload['protocol_sha256']==e.old.sha(e.ROOT/'protocol.json')
        for r in payload['rows']:
            for alpha in r['rollouts']:
                for category,label in [('all','all'),('length',r['length_bin']),('condition',str(r['condition']))]:
                    add(groups[(r['split'],alpha,category,label)],r,alpha)
                for step in range(1,17):add(groups[(r['split'],alpha,'step',str(step))],r,alpha,step)
            for g in r['geometry']:
                g.update(length_bin=r['length_bin'],condition=r['condition'],identity=r['identity'])
                geometry[r['split']].append(g)
                if g['teacher_nonzero']:
                    for family,m in g['families'].items():margins[(r['split'],family)].append(m['teacher_min'])
        if i%100==0:print(json.dumps(dict(aggregated=i,wall_seconds=time.perf_counter()-started)),flush=True)
    summaries={split:{alpha:finalize(groups[(split,alpha,'all','all')]) for alpha in map(str,e.ALPHAS)} for split in ('validation','test')}
    assert i==600
    assert all(s['identities']==300 and s['trajectories']==900 and s['roll_states']==s['one_states']==14400 for ss in summaries.values() for s in ss.values())
    diagnostics={}
    for split,gs in geometry.items():
        labels=[g['raw_failed'] for g in gs];risks=[g['directional_risk_ratio'] for g in gs];errors=[g['error_l2'] for g in gs]
        bins=[];edges=e.PROTOCOL['risk_bins']
        for lo,hi in zip(edges[:-1],edges[1:]):
            chosen=[g for g in gs if lo<=g['directional_risk_ratio']<hi]
            bins.append(dict(lo=lo,hi=hi,count=len(chosen),failure_rate=float(np.mean([g['raw_failed'] for g in chosen])) if chosen else None))
        diagnostics[split]=dict(states=len(gs),nonzero_teacher_states=sum(g['teacher_nonzero'] for g in gs),
            teacher_unsafe_states=sum(not g['teacher_safe'] for g in gs),error_l2=stats(errors),
            action_cosine=stats([g['action_cosine'] for g in gs if g.get('action_cosine') is not None]),
            error_magnitude_failure_AUC=auc(errors,labels),directional_margin_risk_failure_AUC=auc(risks,labels),risk_bins=bins,
            first_order_crossing_confusion=dict(true_positive=sum(r>1 and f for r,f in zip(risks,labels)),
                false_positive=sum(r>1 and not f for r,f in zip(risks,labels)),
                true_negative=sum(r<=1 and not f for r,f in zip(risks,labels)),
                false_negative=sum(r<=1 and f for r,f in zip(risks,labels))),
            within_length_AUC={length:dict(error_magnitude=auc([g['error_l2'] for g in gs if g['length_bin']==length],[g['raw_failed'] for g in gs if g['length_bin']==length]),
                directional_margin=auc([g['directional_risk_ratio'] for g in gs if g['length_bin']==length],[g['raw_failed'] for g in gs if g['length_bin']==length])) for length in e.old.STRATA},
            teacher_cap_slack=stats([g['teacher_cap_slack'] for g in gs if g['teacher_nonzero']]),
            teacher_near_any_native_boundary={str(t):float(np.mean([min(m['teacher_min'] for m in g['families'].values())<=t for g in gs if g['teacher_nonzero']])) for t in e.PROTOCOL['near_boundary_normalized_thresholds']},
            teacher_action_ball_within_1e_6_A=float(np.mean([g['teacher_cap_slack']<=1e-6 for g in gs if g['teacher_nonzero']])),
            teacher_boundary_by_family={family:dict(slack=stats(vals),near_boundary={str(t):float(np.mean(np.asarray(vals)<=t)) for t in e.PROTOCOL['near_boundary_normalized_thresholds']}) for (s,family),vals in margins.items() if s==split})
    val=summaries['validation'];base=val['1.0'];half=val['0.5']
    opens=half['roll_failed_rate']<=.5*base['roll_failed_rate'] and half['rollout_retention']>=.25
    gate=dict(supports_margin_target_training=opens,rule=e.PROTOCOL['phase_C_gate'],
        validation_failure_reduction=1-half['roll_failed_rate']/base['roll_failed_rate'],validation_half_retention=half['rollout_retention'],test_used=False)
    pareto={split:[alpha for alpha,s in ss.items() if not any(
        t['rollout_retention']>=s['rollout_retention'] and t['roll_failed_rate']<=s['roll_failed_rate'] and
        (t['rollout_retention']>s['rollout_retention'] or t['roll_failed_rate']<s['roll_failed_rate'])
        for other,t in ss.items() if other!=alpha)] for split,ss in summaries.items()}
    result=dict(protocol_sha256=e.old.sha(e.ROOT/'protocol.json'),summaries=summaries,geometry=diagnostics,phase_C_gate=gate,pareto_nondominated_alphas=pareto,
        strata=[dict(split=s,alpha=a,category=c,label=l,**finalize(b)) for (s,a,c,l),b in groups.items() if c!='all'])
    e.write(e.ROOT/'results_AB.json',result)
    lines=['# E010 student feasibility margin v1 — offline phases A/B','',
        'Frozen baseline: fc76211. No retraining, new teacher actions, SLSQP solves, X-based inference shield, or per-sample alpha selection. All artifacts reside on D:.',
        '','The same frozen 300 validation and 300 test identities are evaluated, with all three conditions and 16 logical states/rollout steps per trajectory. Family failures overlap; persistent inversions are recounted per state.',
        '','## Repair versus safety','', '| Split | Alpha | One-step retention | K16 retention | Unsafe K16 states | Unsafe rate | New inversion observations | Assessability losses |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for split in ('validation','test'):
        for alpha in map(str,e.ALPHAS):
            s=summaries[split][alpha]
            lines.append(f"| {split} | {alpha} | {100*s['one_step_retention']:.2f}% | {100*s['rollout_retention']:.2f}% | {s['roll_failed']}/{s['roll_states']} | {100*s['roll_failed_rate']:.2f}% | {s['roll_new_inversions']} | {s['roll_assessability_loss']} |")
    lines+=['','| Test alpha | Unsafe one-step states | Aligned-RMSD failed rollout states | Continuous chirality failed states | Signed failed states | Mean aligned-RMSD change (Å) | Exact zero / near zero |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for alpha,s in summaries['test'].items():
        f=s['roll_failure_families']
        lines.append(f"| {alpha} | {s['one_failed']}/{s['one_states']} | {f['aligned']} | {f['continuous_chirality']} | {f['signed']} | {s['mean_aligned_rmsd_change']:.6f} | {100*s['roll_exact_zero_rate']:.2f}% / {100*s['roll_near_zero_rate']:.2f}% |")
    lines+=['',f"Action-cap and eligibility violations across the curve: {sum(s['roll_cap']+s['one_cap']+s['roll_eligibility']+s['one_eligibility'] for ss in summaries.values() for s in ss.values())}.",
        '', '## Failure geometry','', 'Constraint slacks are exact native normalized inequalities, reported separately by family. The error-direction risk is a central finite-difference first-order diagnostic at the teacher endpoint, with step 1e-4 along the action error. It is not an exact geometric distance; small/zero endpoint slacks amplify the ratio. Action-ball slack is separately measured in Å. Eligibility is an exact equality and cannot have a positive buffer. Raw Cartesian error is recorded but the frozen global hard gate is aligned RMSD.', '']
    for split,g in diagnostics.items():
        lines.append(f"{split}: teacher unsafe states {g['teacher_unsafe_states']}; error-magnitude AUC {g['error_magnitude_failure_AUC']:.3f}; directional margin-risk AUC {g['directional_margin_risk_failure_AUC']:.3f}.")
        for family,m in g['teacher_boundary_by_family'].items():
            lines.append(f"- {family}: median endpoint slack {m['slack']['median']:.6g}; within 1e-5 of boundary {100*m['near_boundary']['1e-05']:.2f}%.")
        lines.append(f"- Teacher action-ball slack: median {g['teacher_cap_slack']['median']:.6g} Å.")
        lines.append(f"- Within 1e-5 of any native normalized hard boundary: {100*g['teacher_near_any_native_boundary']['1e-05']:.2f}% of nonzero teacher states. Within 1e-6 Å of the action ball: {100*g['teacher_action_ball_within_1e_6_A']:.2f}%.")
        lines+=['','| Directional error / teacher margin | States | Raw failure rate |','|---|---:|---:|']
        for b in g['risk_bins']:
            rate=f"{100*b['failure_rate']:.2f}%" if b['failure_rate'] is not None else '—'
            lines.append(f"| [{b['lo']:g}, {b['hi']:g}) | {b['count']} | {rate} |")
        lines.append('')
    lines+=['## Preregistered conditional gate','',f"Phase C authorized by its scientific gate: **{opens}**. Validation alpha 0.5 failure reduction: {100*gate['validation_failure_reduction']:.2f}%; original-teacher repair retention: {100*gate['validation_half_retention']:.2f}%. Required: at least 50% failure reduction and at least 25% retention. Test was not used to open this gate.",'',
        'The frozen Phase C buffer ladder is 0.001/0.01/0.05 in the existing normalized inequality units, with a corresponding inward action-cap buffer. Largest passing rho would be searched on {1,0.99,...,0}; transitions with no positive-buffer candidate would be flagged unbufferable rather than counted as buffered zeros.', '',
        'Full per-identity, condition, length, and rollout-step results are in results_AB.json and records/. Exact start/teacher/student constraint vectors and directional derivatives are in geometry/. Recursive actions are in actions/ and reconstruct from the frozen P0.']
    if not opens:
        lines+=['','## Decision','',
            'Stop after phases A/B. The preregistered validation gate does not support automatic margin-target retraining. No marginized dataset, new model, larger model, new teacher data, full corpus, or SLSQP solve was generated.', '',
            'Boundary sensitivity may still contribute, but reducing global action magnitude is insufficient to produce a deployable repair/safety point. Persistent failures at smaller alpha indicate that directional feasibility and baseline-anchored cumulative constraints matter, not only imitation-error amplitude. The first-order margin diagnostic is associative and cannot establish causality or prove an observability limit.', '',
            'This closed gate is an operational decision for this preregistered pilot; it does not prove that marginized-target training cannot help. No alpha below 0.2 was tested, and no claims are made about those untested scales.', '',
            'A subsequent separately authorized study could examine inference-observable geometric feasibility and constraint-family-specific directional errors. This experiment does not introduce new context or an X-based inference shield.']
    (e.ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(gate=gate,summaries=summaries)),flush=True)


if __name__=='__main__':main()
