"""Read-only teacher evaluation and conditional frozen transition publication."""
import json
from statistics import median
import numpy as np
from scripts.run_e010_sequential_teacher_v16d import OUT
from scripts.run_e010_phase4d_recurrent_capacity_v3 import write
from scripts.report_e010_local_feasibility_v7 import summarize, percent


def main():
    assert json.loads((OUT/'reproduction_complete.json').read_text())['exact']
    contract=json.loads((OUT/'execution_contract.json').read_text())
    run=json.loads((OUT/'run_complete.json').read_text())
    records=[json.loads((OUT/'examples'/f'example_{i:02d}.json').read_text()) for i in range(60)]
    states=[summarize([r['states'][t] for r in records]) for t in range(17)]
    gains={c:{str(t):-percent(states[t]['by_condition'][c],states[0]['by_condition'][c],'mean_local_rmse')
              for t in (1,2,4,8,12,16)} for c in ('50','250','450')}
    offsets={c:{k:100*(1-states[-1]['by_condition'][c]['local_rmse'][k]/states[0]['by_condition'][c]['local_rmse'][k])
                for k in ('1','2','3')} for c in gains}
    allsafe=all(s['safety']['passes'] and s['finite'] and not s['collapse'] and s['frame_assessability_preserved']
                and (not s['step'] or s['exact_stored_step_max']<=.1)
                for r in records for s in r['states'])
    useful=all(v>0 for group in offsets.values() for v in group.values())
    counts=sum(s['eligible_corrections'] for r in records for s in r['states'][1:])
    fractions={str(a):sum(s['correction_fractions'][str(a)]*s['eligible_corrections']
                          for r in records for s in r['states'][1:])/counts for a in (.04,.06,.08,.095)}
    late=np.mean([r['solver_steps'][7]['accepted_normalized_local']-r['solver_steps'][15]['accepted_normalized_local']
                  for r in records])
    gates=allsafe and gains['450']['16']>=5 and useful
    classification=('TEACH16-C' if fractions['0.04']>.01 or late>1e-6 else 'TEACH16-B') if gates else 'TEACH16-E'
    logs=[v for r in records for v in r['solver_steps']]
    correction_by_step=[]
    for t in range(1,17):
        rows=[r['states'][t] for r in records]
        cosine=[v for s in rows for v in s.get('consecutive_cosines',[])]
        correction_by_step.append(dict(step=t,rms_mean=float(np.mean([s['step_correction_rms'] for s in rows])),
             maximum=max(s['exact_stored_step_max'] for s in rows),
             fractions={str(a):sum(s['correction_fractions'][str(a)]*s['eligible_corrections'] for s in rows)/sum(s['eligible_corrections'] for s in rows) for a in (.04,.06,.08,.095)},
             consecutive_cosine_quantiles=np.quantile(cosine,[0,.1,.5,.9,1]).tolist() if cosine else None))
    result=dict(classification=classification,teacher_frozen_for_distillation=gates,trajectories=60,
        transition_slots=960,solver='SciPy SLSQP',workers=run['workers'],runtime_seconds=run['wall_seconds'],
        peak_worker_rss_bytes=max(r['peak_rss_bytes'] for r in records),
        conservative_total_worker_rss_bytes=run['workers']*max(r['peak_rss_bytes'] for r in records),
        K_max=16,s_max_angstrom=.1,zero_actions=sum(s['zero_action'] for s in logs),all_stepwise_safety=allsafe,
        local_offsets_improve_all_conditions=useful,condition_gains_pct=gains,offset_gains_pct=offsets,
        marginal_gains_pp={c:{'K4_to_K8':g['8']-g['4'],'K8_to_K12':g['12']-g['8'],'K12_to_K16':g['16']-g['12'],'K8_to_K16':g['16']-g['8'],
            'fraction_final_gain_after_8':(g['16']-g['8'])/g['16']} for c,g in gains.items()},
        by_condition_correction_fractions={c:{str(v):sum(s['correction_fractions'][str(v)]*s['eligible_corrections'] for r in records if str(r['record']['condition'])==c for s in r['states'][1:])/sum(s['eligible_corrections'] for r in records if str(r['record']['condition'])==c for s in r['states'][1:]) for v in (.04,.06,.08,.095)} for c in gains},
        eligible_correction_fractions=fractions,late_normalized_local_MSE_gain=float(late),
        useful_nonzero_actions_after_8=sum(not s['zero_action'] for r in records for s in r['solver_steps'][8:]),
        maximum_correction_angstrom=max(s['exact_stored_step_max'] for r in records for s in r['states'][1:]),
        max_inward_guard_angstrom=max(s['maximum_inward_guard_angstrom'] for s in logs),
        max_total_projection_angstrom=max(s['maximum_total_projection_angstrom'] for s in logs),
        iterations_min_median_max=[min(s['iterations'] for s in logs),median(s['iterations'] for s in logs),max(s['iterations'] for s in logs)],
        solver_success_steps=sum(s['success'] for s in logs),states=states,correction_by_step=correction_by_step,
        v13_reference_gains_pct={'50':19.1694,'250':8.3859,'450':8.5007},
        cuda_used=False,neural_training_launched=False,globally_optimal_oracle_claim=False)
    write(OUT/'teacher_result.json',result)
    if gates:
        manifest=json.loads((OUT/'transition_candidates.json').read_text())
        manifest.update(authorized_for_distillation=True,classification=classification,
                        teacher_result='teacher_result.json',frozen=True)
        write(OUT/'frozen_transition_manifest.json',manifest)
        from protein_distance_diffusion.training.e010_phase4d_diagnostic import file_hash
        write(OUT/'transition_manifest_hash.json',dict(sha256=file_hash(OUT/'frozen_transition_manifest.json')))
    lines=['# V16D sequential privileged teacher','',f'Classification: **{classification}**. Frozen for distillation: **{gates}**.',
           '', 'No neural training or CUDA. Teacher knows target X; student will not. These are safe reproducible policy trajectories, not globally optimal solutions.',
           '', '| Condition | K1 | K2 | K4 | K8 | K12 | K16 | i+1 / i+2 / i+3 final gain % |',
           '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for c,g in gains.items():
        lines.append('| '+c+' | '+' | '.join(f'{g[str(t)]:.6f}' for t in (1,2,4,8,12,16))+' | '+ '/'.join(f'{offsets[c][k]:.6f}' for k in ('1','2','3'))+' |')
    lines+=['','| Condition | Aligned change % | Continuous chiral change % | Inversions Pg / P16 | New / repaired | Assessability Pg / P16 | Net RMS/max Å | Path RMS/max Å |',
            '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for c in gains:
        b,f=states[0]['by_condition'][c],states[-1]['by_condition'][c]
        group=[r for r in records if str(r['record']['condition'])==c]
        q=[r['states'][-1]['safety']['quartets'] for r in group]
        lines.append(f"| {c} | {percent(f,b,'aligned_rmsd'):.6f} | {percent(f,b,'continuous_chiral_loss'):.6f} | {b['chirality_inversions']}/{f['chirality_inversions']} | {sum(v['new_inversions'] for v in q)}/{sum(v['repaired_inversions'] for v in q)} | {b['chirality_assessable']}/{f['chirality_assessable']} | {f['displacement_rms']:.6f}/{f['displacement_max']:.6f} | {f['path_length_rms']:.6f}/{f['path_length_max']:.6f} |")
    lines+=['','| Length stratum | K4 gain % | K8 gain % | K16 gain % | Aligned final change % |',
            '| --- | --- | --- | --- | --- |']
    for s,b in states[0]['by_stratum'].items():
        lines.append('| '+s+' | '+' | '.join(f"{-percent(states[t]['by_stratum'][s],b,'mean_local_rmse'):.6f}" for t in (4,8,16))+f" | {percent(states[-1]['by_stratum'][s],b,'aligned_rmsd'):.6f} |")
    lines+=['',f"60 complete trajectories / 960 slots; zero actions {result['zero_actions']}. Workers {run['workers']}; wall time {run['wall_seconds']/60:.2f} min; peak worker RSS {result['peak_worker_rss_bytes']/2**30:.3f} GiB; conservative summed peak {result['conservative_total_worker_rss_bytes']/2**30:.3f} GiB.",
            '',f"Fractions of eligible corrections >.04/.06/.08/.095 Å: {fractions}. Maximum correction {result['maximum_correction_angstrom']:.17g} Å. Maximum ULP guard adjustment {result['max_inward_guard_angstrom']:.6g} Å; full radial projection adjustment {result['max_total_projection_angstrom']:.6g} Å.",
            '',f"Useful nonzero actions after step8: {result['useful_nonzero_actions_after_8']}; mean normalized local-MSE improvement K8→K16: {late:.6g}. Solver success {result['solver_success_steps']}/960 is telemetry, not a stationarity or optimality claim.",
            '', 'Full per-state safety, hashes, solver histories and length/condition aggregates are recorded in JSON. Every saved coordinate, correction, eligibility, frame, metric and safety decision was independently reproduced without optimization. Large trajectories remain outside Git in untracked_states. The transition manifest includes no-op slots.',
            '', 'Recommended next experiment: prepare a matched oracle-to-neural trajectory-distillation protocol using the frozen teacher corpus, with target X excluded from student inputs; do not launch training automatically.' if gates else 'Recommended next experiment: audit the failed one-step teacher numerical/quality gate before authorizing a distillation corpus.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(classification,gains,flush=True)

if __name__=='__main__': main()
