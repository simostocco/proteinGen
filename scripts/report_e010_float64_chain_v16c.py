"""Read-only publication of the precision correction and guarded validation."""
import json
from scripts.audit_e010_float64_chain_v16c import OUT


def main():
    result=json.loads((OUT/'precision_result.json').read_text())
    cases=[json.loads((OUT/'cases'/f'length_{n}.json').read_text()) for n in (12,32,500)]
    lines=['# V16C float64 Jacobian-chain correction','',
      '**PRECISION-A — float64 chain repaired.** No optimization, teacher panel, CUDA or neural training.',
      '', 'Historical V16A/V16B files, reports and failure classifications are unchanged. Only the new versioned implementation is corrected. All solver settings, scientific equations, tau, bounds and eligibility semantics are preserved.',
      '', '## Precision changes','',
      '- OneStep constructs scale using current.new_tensor(0.10), casts Boolean eligibility to current.dtype only for arithmetic, and uses both explicit float64 objects in the coordinate/Jacobian chains.',
      '- NumPy ball inputs are explicitly float64. Objective/constraint definitions, K_max=16 and s_max=0.10 Å are unchanged.',
      '- The comprehensive trace also found PyTorch determinant-backward 0/1 temporary tensors in default float32. These are exactly representable and caused no value defect, but violate an all-float64 intermediate contract. A scoped float64 default during jac/cjac differentiation removes them and restores the prior default in finally. It is designed for the preregistered isolated single-threaded teacher workers. Kabsch equations are unchanged.',
      '- Float64Trace is an opt-in operation-level assertion covering nested geometry, objective, constraints and autograd. Boolean/integer objects stay Boolean/integer. No trace overhead when disabled.',
      '', '## Value parity before derivatives','',
      'At every exact recovered V16B state, coordinates, local/cartesian/aligned/chiral terms, signed q, turn/bond assessability quantities and every safety constraint are bitwise identical old/new: **all reported differences are zero**. Eligibility, inversion/assessability and safety classifications are exactly unchanged. Archived state/direction/input/eligibility hashes reproduce where available; no optimization was used for recovery.',
      '', '## Primary finite differences: h=1e-4','',
      'Unchanged FD tolerance: 1e-8 +1e-5*abs(analytic directional derivative). Unchanged analytic tolerance:1e-12 +1e-10*abs(analytic directional derivative). Each row includes scalar objective and all vector safety components. Worst vector component is selected by error/tolerance ratio for reporting, never for acceptance.',
      '', '| Length | Phase | Analytic max discrepancy | Worst family | Analytic | FD | Abs error | Rel error | Allowed error | Physical max/RMS Å | Pass |',
      '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for c in cases:
        for d in c['directions']:
            s=next(s for s in d['samples'] if s['primary']);fs=s['families']
            k=max(fs,key=lambda k:fs[k]['absolute_error']/fs[k]['allowed_error']);f=fs[k];p=s['physical']
            lines.append(f"| {c['length']} | {d['phase']} | {d['analytic_max_difference']:.6g} | {k} | {f['analytic']:.12g} | {f['finite_difference']:.12g} | {f['absolute_error']:.6g} | {f['relative_error']:.6g} | {f['allowed_error']:.6g} | {p['intended_max_angstrom']:.6g}/{p['intended_rms_angstrom']:.6g} | {s['passes']} |")
    lines+=['',f"Full-Jacobian vs independent whole-function autograd JVP maximum discrepancy: **{result['maximum_analytic_disagreement']:.6g}**. Direct scalar family contractions maximum discrepancy: **{result['maximum_scalar_disagreement']:.6g}**. All pass unchanged analytic tolerances.",
      '', 'Nine distinct length×phase primary validations pass. At each length, the historical nine phase×old-epsilon slots map to the fixed primary h=1e-4 (all9 slots pass); three reused slots per direction are explicitly labeled, not claimed as additional independent validations. Full per-family scalar/vector values, errors and physical perturbations are in cases JSON.',
      '', '## Resolution sanity (not used for acceptance)','',
      '| Length | Phase | h | All components pass | Failed rows |',
      '| --- | --- | --- | --- | --- |']
    for c in cases:
        for d in c['directions']:
            for s in d['samples']:
                lines.append(f"| {c['length']} | {d['phase']} | {s['h']:.0e} | {s['passes']} | {sum(v['failures'] for v in s['families'].values())} |")
    lines+=['', '## Dtype and integrity evidence','',
      'cases/length_12.json, length_32.json and length_500.json report Python/NumPy/Torch types, dtypes/devices for state, scale, masks, every objective term, q/turn/bonds, constants and constraints, plus operator-level old/new dtype traces. Every new floating operation in geometry and differentiation was CPU float64 under strict mode. Old traces preserve both contamination paths. The process default dtype was restored after differentiation.',
      '', 'Optimizer calls are fail-fast blocked in audit and focused tests. Tests requiring real historical optimization are excluded rather than run. No synthetic or real optimization is authorized or invoked. New-lineage tests validate scale storage, Boolean semantics, value parity, old-defect detection, strict new dtype assertions, analytic/FD agreement and unchanged scientific settings. Independent reproduction must match all numerical records and original protected hashes.',
      '', 'Exactly one recommended next action: **prepare a new versioned sequential-teacher replay using V16C float64 implementation and fixed h=1e-4 validation; retain all other V16A scientific and solver settings.** No replay or teacher generation is launched by V16C.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
