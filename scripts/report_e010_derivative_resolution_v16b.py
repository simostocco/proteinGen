"""Publish fixed-state audit evidence; no optimizer or teacher-quality metrics."""
import json
from pathlib import Path
from scripts.audit_e010_derivative_resolution_v16b import OUT
from scripts.run_e010_phase4d_recurrent_capacity_v3 import write


def main():
    result=json.loads((OUT/'audit_result.json').read_text())
    curves=[json.loads(p.read_text()) for p in sorted((OUT/'curves').glob('*.json'))]
    lines=['# V16B length-500 derivative resolution audit','',
        '**FD-D — mixed: small-step finite-difference under-resolution plus a separate analytic chain-scaling precision defect.**','',
        'The exact V16A check state, deterministic directions and every available historical derivative comparison were reproduced bitwise. V16A has no saved derivative-state coordinate hash; new state/input/eligibility hashes document the recovered state without claiming nonexistent historical hashes. Historical V16A and all scientific settings remain unchanged.',
        '', '## Checks and recovery','',
        'The nine checks cross phase 0/.7/1.3 with dimensionless h=1e-6/1e-5/1e-4. Each includes a scalar normalized local objective and a 1,495-component vector of safety constraints (500 case): aligned RMSD, continuous chirality, signed no-new-inversion, unsigned assessability, frame assessability and bond assessability. Ball constraints are not part of the historical nine-check vector; its construction is preserved exactly. Every vector component must pass 1e-8 + 1e-5*abs(analytic derivative). Full family-wise records and all vector components are retained in curve JSON and reproducible ignored numerical arrays with versioned hashes.',
        '', '| Phase | h | Worst vector family | Analytic Jd | Centered FD | Absolute error | Relative error | Frozen allowed error | Failed vector rows | Objective |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for c in result['historical_length_500_checks']:
        fs=c['vector_constraint_families']; name=max(fs,key=lambda k:fs[k]['maximum_tolerance_ratio']); f=fs[name]
        lines.append(f"| {c['phase']} | {c['historical_h']:.0e} | {name} | {f['analytic']:.12g} | {f['fd']:.12g} | {f['absolute_error']:.6g} | {f['relative_error']:.6g} | {f['tolerance']:.6g} | {c['vector_failures']} | {'pass' if c['scalar_objective']['failures']==0 else 'FAIL'} |")
    lines+=['','Historical failures are exactly (phase0,h1e-6):32 rows, (phase.7,h1e-6):50 rows, (phase1.3,h1e-6):36 rows. All six larger-h checks and all nine scalar-objective checks pass unchanged tolerances.',
            '', '## Physical perturbations','',
            '| Phase | h | Intended max Å | Intended RMS Å | Stored max Å | Stored RMS Å |',
            '| --- | --- | --- | --- | --- | --- |']
    for c in result['historical_length_500_checks']:
        p=c['physical']; lines.append(f"| {c['phase']} | {c['historical_h']:.0e} | {p['intended_max_angstrom']:.6g} | {p['intended_rms_angstrom']:.6g} | {p['stored_max_angstrom']:.6g} | {p['stored_rms_angstrom']:.6g} |")
    lines+=['','z is dimensionless; physical delta=.10*z at eligible residues. RMS is across all masked residues (endpoints zero). The full 13-value grid retains intended/stored physical perturbations for every h, including unchanged-coordinate fractions.',
            '', '## Resolution curve: length500, maximum across all three directions','',
            '| h | Max vector absolute error | Max error / frozen allowed error | Failed rows, summed | All checks pass |',
            '| --- | --- | --- | --- | --- |']
    high=[c for c in curves if c['length']==500]
    for j,h in enumerate(result['epsilon_grid']):
        fs=[v for c in high for k,v in c['samples'][j]['families'].items() if k!='objective']
        lines.append(f"| {h:.0e} | {max(f['maximum_absolute_error'] for f in fs):.6g} | {max(f['maximum_tolerance_ratio'] for f in fs):.6g} | {sum(f['failures'] for f in fs)} | {all(c['samples'][j]['passes_all'] for c in high)} |")
    lines+=['','## Cancellation at the three failed checks','',
            '| Phase | Worst family | f(x) | f+ | f− | Absolute numerator | eps64 * local function scale | Numerator / roundoff scale | Reliable relative digits |',
            '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for c in high:
        s=next(s for s in c['samples'] if s['h']==1e-6)
        fs={k:v for k,v in s['families'].items() if k!='objective'}; name=max(fs,key=lambda k:fs[k]['maximum_tolerance_ratio']); f=fs[name]
        import math
        digits=max(0,min(16,-math.log10(max(f['relative_error'],1e-16))))
        lines.append(f"| {c['phase']} | {name} | {f['f0']:.17g} | {f['f_plus']:.17g} | {f['f_minus']:.17g} | {f['numerator']:.6g} | {f['roundoff_local_scale']:.6g} | {f['numerator_to_roundoff']:.6g} | {digits:.3f} |")
    lines+=['','Small h produces physical vector changes only ~4.6e-9 Å at coordinates near 998 Å. Subtraction of nearby translated coordinates to form bonds introduces finite precision before the centered numerator is formed. Function numerators are tiny relative to O(1) normalized constraint values; the local eps64 comparison is a lower bound, not a complete forward-error bound for this geometry pipeline. Error increases strongly as h shrinks, then decreases in the larger-h window. This supports cancellation/under-resolution rather than an h-independent derivative error as the cause of the three historical failures.',
            '', '## Analytic precision defect','',
            'The independent analytic paths are (A) the historical full coordinate Jacobian followed by the stored mask chain; (B) direct autograd JVP through the entire current-geometry function; plus scalar reverse-autograd family contractions. Objective and global constraints agree. Quartet rows do not pass the unchanged preregistered analytic-consistency tolerance 1e-12+1e-10*abs(Jd).',
            '',f"V16A `.1 * Boolean eligibility` promotes to torch.float32, yielding **0.10000000149011612** rather than float64 0.1. The historical quartet derivatives therefore have multiplicative factor **{result['chain_bias_factor']:.17g}**. Maximum analytic discrepancy across lengths/directions is **{result['analytic_max_disagreement']:.6g}**; the factor model explains it to **{result['chain_factor_model_max_error']:.6g}**. Maximum scalar-contraction discrepancy is **{result['scalar_analytic_max_disagreement']:.6g}**. No historical source was repaired in V16B. This small defect is separate from the much larger small-h finite-difference errors and prevents an FD-A claim that all analytic paths agree.",
            '', '## Length controls, stable windows and Richardson','',
            '| Length | Phase | Passing grid h | Proposed future h |',
            '| --- | --- | --- | --- |']
    for c in curves:
        lines.append(f"| {c['length']} | {c['phase']} | {', '.join(f'{h:.0e}' for h in c['stable_grid_values'])} | 1e-4 |")
    lines+=['','Common window across every objective/vector component, direction and length: **1e-5,3e-5,1e-4**, with original FD tolerances. The lower edge shifts upward with length: all-direction length12 window begins at3e-8, length32 at1e-7, length500 at1e-5.',
            '', '| Length | Phase | h | Max abs(Dh−Dhalf) | Max Richardson extrapolate error | Half-step all-pass |',
            '| --- | --- | --- | --- | --- | --- |']
    for c in curves:
        for r in c['richardson']:
            if r['h'] in (1e-5,3e-5,1e-4):
                lines.append(f"| {c['length']} | {c['phase']} | {r['h']:.0e} | {r['maximum_Dh_Dhalf_difference']:.6g} | {r['maximum_extrapolated_absolute_error']:.6g} | {r['half_passes_all']} |")
    lines+=['','Full Dh and Dhalf vectors are retained with hashes. The half-step comparison supports resolution in the stable window; Richardson does not consistently improve a roundoff-dominated estimate and is not used to replace the analytic derivative. No clear truncation-dominated O(h²) regime emerges within the fixed grid ending at1e-4; no larger h was added after inspection.',
            '', '## Proposal and compliance','',
            'Proposed future validation rule: **fixed dimensionless h=1e-4**, contingent on separately repairing/validating the float32 chain-scaling defect in a new version. This was proposed from numerical scale before running the grid; its length500 maximum/RMS physical perturbations are about4.6e-7/4.46e-7 Å. The proposal is not applied to V16A. Historical tolerances, bounds, constraints and solver settings remain immutable.',
            '', 'The audit/reproducer itself invokes no optimizer and uses no teacher-quality outcomes. A validation-command mistake included one inherited test that executed **two zero-start length12 synthetic SLSQP solves**; no real-panel examples or saved endpoints were touched. Those test outputs are excluded from all audit evidence and conclusions. This task-level deviation is explicitly recorded in compliance_deviation.json; subsequent validation uses only optimizer-guarded audit tests. The historical scientific panel was never launched, CUDA was never used, and no neural training occurred.',
            '', 'Recommended next action: **prepare a versioned correction of the Boolean-mask chain to explicit float64, then validate it independently at the unchanged tolerances before considering any teacher replay**.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__': main()
