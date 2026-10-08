"""Finalize mechanistic and stratified rollout analysis of the frozen benchmark."""
import os
from pathlib import Path
import sys

ROOT=Path('/mnt/d/Simone/proteinGen_data/e010_phase4d/fast_privileged_gradient_teacher_v1')
os.environ.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT/'tmp'),TMP=str(ROOT/'tmp'),TEMP=str(ROOT/'tmp'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import json
import tempfile
import numpy as np
import torch
import run_e010_privileged_gradient_v1 as benchmark


def cosine_distribution(rows):
    values=np.array([r['gold_negative_gradient_cosine'] for r in rows if r['gold_negative_gradient_cosine'] is not None])
    return dict(distribution=benchmark.statistics(values),total=len(rows),excluded=len(rows)-len(values),
                positive_fraction=float(np.mean(values>0)) if len(values) else None,
                cosine_over_point8_fraction=float(np.mean(values>=.8)) if len(values) else None,
                cosine_over_point95_fraction=float(np.mean(values>=.95)) if len(values) else None)


def main():
    benchmark.initialize()
    completed=json.loads((ROOT/'COMPLETED.json').read_text())
    for path,digest in completed['files'].items():assert benchmark.pilot.sha(path)==digest,path
    data,frozen=benchmark.pilot.load()
    outputs=[benchmark.validate_trajectory(index,data) for index in sorted(data)]
    results=json.loads((ROOT/'results.json').read_text())
    diagnostic=results['gradient_diagnostic_rows']
    grouped_diagnostic={dimension:{key:cosine_distribution([r for r in diagnostic if str(r[dimension])==key]) for key in sorted({str(r[dimension]) for r in diagnostic})} for dimension in ('identity','condition','length','length_bin','step')}
    unshrunk=[];gradient_errors=[]
    for index,item in data.items():
        with np.load(ROOT/'trajectories'/f'trajectory_{index:02d}.npz',allow_pickle=False) as archive:
            gradients=archive['gradient'].copy()
        for step,g in enumerate(gradients):
            origin=torch.from_numpy(item['arrays']['states'][step]);eligible=item['arrays']['eligibility'][step]
            norm=np.linalg.norm(g,axis=-1,keepdims=True)
            direction=np.divide(-g,norm,out=np.zeros_like(g),where=norm>0)*eligible[...,None]
            perturbation=torch.from_numpy(direction);epsilon=1e-5
            finite_difference=float((item['ctx'].values(origin+epsilon*perturbation)['local']-item['ctx'].values(origin-epsilon*perturbation)['local'])/(2*epsilon))
            analytic=float((torch.from_numpy(g)*perturbation).sum())
            assert np.isclose(finite_difference,analytic,rtol=1e-6,atol=1e-8),(index,step)
            gradient_errors.append(abs(finite_difference-analytic))
            raw,_=benchmark.pilot.project(direction*.1)
            safety=item['ctx'].safety(origin+torch.from_numpy(raw));q=safety['quartets']
            unshrunk.append(dict(index=index,step=step,identity=item['row']['record']['sample_id'],condition=item['row']['record']['condition'],
                                 passes=safety['passes'],aligned_failure=safety['global_constraints'][0]>1e-8,
                                 continuous_chirality_failure=safety['global_constraints'][1]>1e-8,
                                 exact_quartet_failure=safety['maximum_extra_constraint']>0,
                                 new_inversion_failure=q['new_inversions']>0,assessability_failure=q['assessability_lost']>0,
                                 safety=safety))
    failure_counts={key:sum(row[key] for row in unshrunk) for key in ('passes','aligned_failure','continuous_chirality_failure','exact_quartet_failure','new_inversion_failure','assessability_failure')}
    lookup={(r['trajectory_index'],r['step']):r for r in json.loads((benchmark.PILOT/'gold/transitions.json').read_text())['transitions']}
    rollout_steps=[]
    rollouts=sum([o['rollouts'] for o in outputs],[])
    for rollout in rollouts:
        item=data[rollout['index']];baseline=rollout['baseline'];n=rollout['matched_gold_horizon']
        for row in rollout['steps'][:n]:
            gold=lookup[(row['index'],row['step'])]
            gold_after=gold['metrics_after']
            rollout_steps.append(dict(row,gold_repair=baseline['mean_local_rmse']-gold_after['mean_local_rmse'],
                                      repair=baseline['mean_local_rmse']-row['after']['mean_local_rmse']))
    by_step={step:{m:benchmark.aggregate([r for r in rollout_steps if r['step']==step and r['method']==m],lookup) for m in benchmark.METHODS} for step in sorted({r['step'] for r in rollout_steps})}
    summary=results['summary'];rollout_speedup={}
    frozen_safety_ratios={}
    for method in benchmark.METHODS:
        rows=[r for r in rollout_steps if r['method']==method];pairs=[]
        for row in rows:
            source=lookup[(row['index'],row['step'])]
            seconds=(source.get('solver_metadata') or {}).get('runtime_seconds') or (source.get('timing') or {}).get('solver_wall_seconds')
            if seconds:pairs.append((seconds,row['latency_seconds']))
        rollout_speedup[method]=dict(matched_steps=len(pairs),speedup=sum(p[0] for p in pairs)/sum(p[1] for p in pairs),
                                    median_latency_seconds=float(np.median([r['latency_seconds'] for r in rows])))
        accepted=[r for output in outputs for r in output['one_step'] if r['method']==method]+[s for r in rollouts if r['method']==method for s in r['steps']]
        frozen_safety_ratios[method]=dict(max_aligned_rmsd_ratio_to_original_baseline=max(r['after']['aligned_rmsd']/float(data[r['index']]['ctx'].baseline['aligned']) for r in accepted),
                                          max_continuous_chirality_ratio_to_original_baseline=max(r['after']['continuous_chiral']/float(data[r['index']]['ctx'].baseline['chiral']) for r in accepted))
    additional=dict(gold_gradient_by=grouped_diagnostic,rollout_retention_by_structural_step=by_step,
                    rollout_matching_speedup=rollout_speedup,
                    cosine_global=cosine_distribution(diagnostic),unshrunk_point10_safety_counts=failure_counts,
                    unshrunk_point10_safety_rows=unshrunk,frozen_safety_ratios=frozen_safety_ratios,
                    gradient_finite_difference_check=dict(states_checked=278,passed=True,max_absolute_error=max(gradient_errors)))
    benchmark.pilot.atomic_json(ROOT/'mechanistic_and_rollout_analysis.json',additional)
    lines=['# Fast privileged gradient teacher v1: completed analysis\n',
           'FAST-PT-v1 remains a completed negative pilot. Its architecture and all original outputs were preserved. '
           'This separate experiment uses no neural training, SLSQP, fitted parameters, final student or full corpus. All artifacts are on D:.\n',
           'Benchmark: **278 accepted gold states, 15 complete + 5 partial trajectories, 7 identities**. '
           'The frozen grid is 0.01, 0.025, 0.05, 0.075 and 0.10 Å. '
           'The 2/4-step methods recompute the exact CPU float64 local-objective gradient at each internal state and cap the total correction relative to the external input at 0.10 Å. '
           'Rollouts use 16 external actions; retention compares each rollout at its exact available gold horizon.\n',
           '| Method | One-step repair / gold % | Rollout repair / gold % | Median one-step ms | One-step speedup | Median rollout-action ms | Rollout speedup |\n|---|---:|---:|---:|---:|---:|---:|']
    for method in benchmark.METHODS:
        s=summary[method];r=rollout_speedup[method]
        lines.append(f"| {method} | {100*s['one_step']['repair_retention']:.3f} | {100*s['rollout']['repair_retention']:.3f} | {1000*s['one_step']['latency_seconds']['median']:.3f} | {s['one_step']['speedup']:.1f}x | {1000*r['median_latency_seconds']:.3f} | {r['speedup']:.1f}x |")
    best=max(benchmark.METHODS,key=lambda m:summary[m]['rollout']['repair_retention']);retention=summary[best]['rollout']['repair_retention']
    grid=summary['best_grid']['rollout']['repair_retention']
    two=summary['projected_2']['rollout']['repair_retention'];four=summary['projected_4']['rollout']['repair_retention']
    lines+=['\n'+f'Best matched rollout: **{best}, {100*retention:.3f}%**. Residual gap to 80%: **{max(0,80-100*retention):.3f} percentage points**. '
            f'2-step change versus best-grid: {100*(two-grid):+.3f} points; 4-step change versus best-grid: {100*(four-grid):+.3f} points. '
            'No additional method complexity was introduced after observing these results.\n']
    c=additional['cosine_global'];d=c['distribution']
    lines.append(f"Gold action versus -grad L_local: **{c['total']-c['excluded']} assessed / 278**, {c['excluded']} excluded zero/degenerate pairs. "
                 f"Positive cosine: **{100*c['positive_fraction']:.3f}%**. Cosine mean/median/p10/p90: **{d['mean']:.6f}/{d['median']:.6f}/{d['p10']:.6f}/{d['p90']:.6f}**; "
                 f"range {d['min']:.6f}–{d['max']:.6f}. Fraction cosine ≥0.8: {100*c['cosine_over_point8_fraction']:.3f}%; ≥0.95: {100*c['cosine_over_point95_fraction']:.3f}%.\n")
    if d['median']>=.8:
        interpretation='Gold directions largely align with steepest descent. The residual repair gap cannot be attributed simply to a systematically orthogonal direction; cap, magnitudes and nonlinear boundary feasibility remain possible explanations.'
    elif c['positive_fraction']>=.8:
        interpretation='Gold actions mostly have a descent component but depart substantially from raw steepest descent. This supports systematic constraint-shaped or preconditioned directions, rather than a pure normalized-gradient teacher. Cosine alone does not identify the particular preconditioner or prove which constraint causes the difference.'
    else:
        interpretation='Gold directions are not consistently aligned with the local negative gradient. Endpoint improvement can coexist with poor infinitesimal alignment; this diagnostic does not by itself establish a particular preconditioner.'
    lines.append('Mechanistic interpretation: '+interpretation+'\n')
    lines.append('Safety audit of the already-benchmarked 0.10 Å normalized-gradient proposals **before shrinking** (278 states; failures overlap): '
                 +json.dumps(failure_counts,sort_keys=True)+'. These are proposal failures, not violations by accepted/shielded actions.\n')
    lines.append('| Method | Action cosine (assessed count) | Action RMS diff Å | Zero actions % | Aligned RMSD mean Å | Inversions | Assessability loss | Cap violations |\n|---|---:|---:|---:|---:|---:|---:|---:|')
    for method in benchmark.METHODS:
        r=summary[method]['one_step'];c=r['cosine']
        lines.append(f"| {method} | {c.get('mean','N/A')} ({c['count']}) | {r['action_rms_difference']['mean']:.6f} | {100*r['zero_action_fraction']:.3f} | {r['aligned_rmsd']['mean']:.6f} | {r['new_inversions']} | {r['assessability_loss']} | {r['action_cap_violations']} |")
    for dimension in ('identity','condition','length_bin'):
        lines.append('\n'+f'Repair retention by {dimension} (one-step / matched rollout, %):\n')
        lines.append('| Group | '+' | '.join(benchmark.METHODS)+' |\n|---|'+'---:|'*len(benchmark.METHODS))
        for key,values in results['by'][dimension].items():
            cells=[]
            for method in benchmark.METHODS:
                one=values[method]['repair_retention'];roll=results['rollout_by'][dimension][key][method]['repair_retention']
                cells.append(f'{100*one:.2f} / {100*roll:.2f}')
            lines.append('| '+key+' | '+' | '.join(cells)+' |')
    lines+=['\nFrozen safety passes for every stored gold-state prediction and rollout action. '
            'The historical safety gates include aligned RMSD relative to the original corruption, continuous chirality relative to its frozen baseline (with historical numerical tolerance), exact quartet constraints, no new inversions and assessability preservation. '
            'Eligibility and exact NumPy norm ≤0.10 Å were independently rechecked. Current-state local MSE never increases.\n',
            'Every one of the 278 archived gradients also passed an independent central directional finite-difference check. '
            'Aligned RMSD and continuous-chirality ratios to the original frozen baseline are explicitly recorded in the supplemental JSON.\n',
            'All stratified metrics (including exact length and structural step), one-step repair/gold repair, RMS differences, aligned RMSD, safety counts, latency and speedups are in [results.json](results.json). '
            'Gold-gradient cosine by every requested dimension and rollout retention by structural step are in [mechanistic_and_rollout_analysis.json](mechanistic_and_rollout_analysis.json). '
            'Per-state actions, gradients, rollout states and safety metrics are in trajectories/.\n',
            'Speedups use the 230 gold steps with durable archived solver timing; the 48 unmatched steps are excluded from timing comparisons. '
            'They compare archived solver runtime to new CPU wall time under four-worker concurrency, rather than simultaneous same-hardware reruns. '
            'Reported repair retention is sum absolute mean-local-RMSE repair / sum matching gold repair. '
            'Cosines exclude zero-vector pairs and are therefore conditional on assessed actions. '
            'These seven identities remain a limited benchmark, even if a method meets the target.\n',
            '[Frozen protocol](protocol.json), [source and gold provenance](provenance.json), [verification tests](tests.json), [completion manifest](COMPLETED.json).\n']
    text='\n'.join(lines);path=ROOT/'ANALYSIS_REPORT.md'
    fd,tmp=tempfile.mkstemp(dir=ROOT,suffix='.tmp')
    with os.fdopen(fd,'w') as stream:stream.write(text);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path);benchmark.pilot.fsync_dir(ROOT)
    # Verify the sealed neural pilot has remained unchanged throughout this experiment.
    negative=json.loads((benchmark.PILOT/'FAST_PT_V1_FROZEN_NEGATIVE.json').read_text())
    for path,digest in negative['files'].items():assert benchmark.pilot.sha(path)==digest,path
    files={str(p):benchmark.pilot.sha(p) for p in ROOT.rglob('*') if p.is_file() and '/cache/' not in str(p) and '/tmp/' not in str(p) and '/logs/' not in str(p) and p.name!='FINAL_FROZEN.json'}
    benchmark.pilot.atomic_json(ROOT/'FINAL_FROZEN.json',dict(status='completed',files=files,analysis_source_sha256=benchmark.pilot.sha(__file__),
                                                          FAST_PT_v1_unchanged=True,no_new_gold_solves=True,no_neural_training=True))
    print(json.dumps(dict(best_method=best,rollout_retention=retention,gradient_cosine=additional['cosine_global']),indent=2))


if __name__=='__main__':main()
