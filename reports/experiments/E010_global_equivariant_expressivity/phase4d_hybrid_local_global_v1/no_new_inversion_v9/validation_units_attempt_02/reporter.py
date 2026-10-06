"""Record the numerical preflight blocker without launching a scientific oracle."""

import json
import subprocess

import torch

from protein_distance_diffusion.models.e010_hybrid_local import PSEUDOSCALAR_INDEX, local_representation
from protein_distance_diffusion.training import e010_no_new_inversion_v9 as v9
from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins, file_hash
from protein_distance_diffusion.training.e010_recurrent_capacity import batch
from scripts.run_e010_no_new_inversion_v9 import OUT, setup
from scripts.run_e010_phase4d_recurrent_capacity_v3 import ROOT, load_cache, write

cfg = setup()
cache, manifest = load_cache()
records = []
for i in range(60):
    b = {k: v.double() if v.is_floating_point() else v for k, v in batch(cache, [i], "cpu").items()}
    q = v9.QuartetConstraints(b)
    assert bool((q(b["pg"]) <= 0).all())
    assert torch.equal(q.q0, local_representation(b["pg"], b["mask"])["features"][:, 1:-2, PSEUDOSCALAR_INDEX])
    record = q.telemetry(b["pg"], cfg["constraints"]["active_tolerance"])
    assert record["new_inversions"] == record["assessability_lost"] == 0
    records.append(dict(index=i, record=cache["records"][i], quartets=record))
write(
    OUT / "baseline_exact_feasibility.json",
    dict(examples=60, all_exactly_feasible=True, raw_quantity_bitwise_evaluator_match=True, records=records),
)
pref = json.loads((OUT / "synthetic_preflight.json").read_text())
assert not pref["passed"]
assert not (OUT / "B").exists(), "A scientific panel result must not be concealed by blocker reporting"
pins = {
    str(ROOT / p): file_hash(ROOT / p) for p in subprocess.check_output(["git", "ls-files"], text=True).splitlines()
}
assert (
    subprocess.check_output(["git", "diff", "--name-only", "e293857953b064439645f1c2ae36fdd67b36fe2b"], text=True) == ""
)
assert_file_pins(manifest["protected_input_sha256"])
write(
    OUT / "protected_hashes.json",
    dict(
        historical_sha256=pins,
        protected_input_sha256=manifest["protected_input_sha256"],
        cache_sha256=manifest["cache_sha256"],
    ),
)
summary = dict(
    classification="INV-D",
    scientific_panel_optimized=False,
    scientific_examples_optimized=0,
    condition_gains_pct={"50": None, "250": None, "450": None},
    baseline_assessable=sum(r["quartets"]["assessable"] for r in records),
    baseline_correct=sum(r["quartets"]["correct"] for r in records),
    baseline_inverted=sum(r["quartets"]["inverted"] for r in records),
    condition_baseline_inversions={
        str(c): sum(r["quartets"]["inverted"] for r in records if r["record"]["condition"] == c) for c in (50, 250, 450)
    },
    threshold=v9.THRESHOLD,
    config=cfg,
    synthetic_converged=sum(r["optimizer"]["converged"] for r in pref["records"]),
    synthetic_total=2,
    blockers=[
        "length32 synthetic expanded constraint solve fails retained gtol at xtol despite cap2000",
        "baseline34 direction1.3 finite differences fail frozen mixed tolerance at all three epsilon scales",
    ],
    historical_artifacts_unchanged=True,
    cuda_used=False,
    neural_training_launched=False,
)
write(OUT / "numerical_blocker_result.json", summary)
lines = [
    "# E010 Phase4D V9 numerical preflight result",
    "",
    "**INV-D — numerically inconclusive. Scientific panel NOT launched.**",
    "",
    "The exact signed-quartet formulation is baseline-feasible on all60 fixed examples; al"
    "l13,029 raw q quantities match the historical evaluator bitwise. Baseline correct7,53"
    "6; inverted5,493. Correct quartets receive target-signed q>=nextafter(1e-6,+inf). Inv"
    "erted quartets receive unsigned q²>=tau², with no target-sign restriction. Historical"
    " frame/bond assessability floors are supported explicitly. New constraints admit no r"
    "elaxed final feasibility tolerance.",
    "",
    "K=8 and s_max=.04 Å, targets, Pg, objective and continuous safety constraints remain "
    "unchanged. Sparse exact Jacobian storage is required for the expanded inequalities. E"
    "xact Hessian-vector products remain in use.",
    "",
    "The1000-cap synthetic result is archived. The same length32 V8 control converges50 it"
    "erations. One globally applicable preflight-only cap revision to2000 was permitted an"
    "d frozen before scientific outcomes. No further cap/tolerance/solver change occurred.",
    "",
    "| Synthetic length | Converged | Iterations | Termination | Optimality | Ball KKT | E"
    "xact feasible | New inversions | Assessment loss | Active signed | Minimum signed mar"
    "gin |",
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
]
for r in pref["records"]:
    o = r["optimizer"]
    q = o["quartets"]
    vals = [
        r["length"],
        o["converged"],
        o["iterations"],
        o["message"],
        o["optimality"],
        o["stationarity"]["normalized_ball_kkt_max"],
        o["constraint_feasible"],
        q["new_inversions"],
        q["assessability_lost"],
        q["active_signed"],
        q["minimum_signed_margin"],
    ]
    lines.append("| " + " | ".join(map(str, vals)) + " |")
lines += [
    "",
    "The length32 result passes physical KKT/projection thresholds and every constraint/di"
    "screte gate, but fails the retained1e-12 parameter-space convergence requirement. Its"
    " xtol termination at1807 iterations means another cap extension alone is not justifie"
    "d. Preserve this failure; do not reinterpret it as a converged oracle.",
    "",
    "Baseline directional finite differences initially failed input15 at1e-5. The unchange"
    "d V8 epsilon set[1e-6,1e-5,1e-4] resolves that cancellation, without loosening tolera"
    "nces. Input34/direction1.3 still fails: best scaled error1.2050065 >1. Sparse-chain a"
    "nd direct autograd directional agreement is within3.75e-16 there. This is not evidenc"
    "e of a Jacobian implementation defect; the complete preregistered finite-difference r"
    "equirement remains unvalidated. Its full diagnosis is recorded. No scientific panel w"
    "as opened.",
    "",
    "Condition50/250/450 local gains, offsets, aligned/chiral changes, final inversions, a"
    "ctive constraints, minimum margins and P0-P8/path/saturation metrics are **not evalua"
    "ted for V9**. V8 historical safe gains19.2207/8.4448/8.5719% are comparisons only and"
    " cannot substitute for a V9 result. The complete historical target is neither passed "
    "nor disproved.",
    "",
    "Validation:130 focused CPU tests pass, including sign-crossing rejection, inverted-to"
    "-correct permission, assessability support, exact sparse/autograd Jacobian agreement "
    "and float64 Hessian finite differences. All60 baselines satisfy the exact formulation"
    "; historical/source/input hashes remain intact. Synthetic reproduction evidence is se"
    "parate from scientific convergence. No CUDA, neural training, E010 mutation, environm"
    "ent change, bound sweep or development evaluation.",
    "",
    "Recommended next experiment: a preregistered numerical conditioning audit of the V9 e"
    "ndpoint constraints and radial-variable trust-constr solve, retaining K=8/s_max=.04 a"
    "nd the exact scientific feasible set, before any panel optimization.",
    "",
]
(OUT / "RESULTS.md").write_text("\n".join(lines).rstrip() + "\n")
print(summary)
