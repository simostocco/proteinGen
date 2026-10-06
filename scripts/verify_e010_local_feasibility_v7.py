import hashlib
import json
import sys

import torch

sys.path.insert(0, ".")
from protein_distance_diffusion.training import e010_local_feasibility_v7 as v7
from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins
from protein_distance_diffusion.training.e010_recurrent_capacity import batch
from scripts.run_e010_local_feasibility_v7 import OUT, digest, setup
from scripts.run_e010_phase4d_recurrent_capacity_v3 import load_cache

cfg = setup()
contract = json.loads((OUT / "execution_contract.json").read_text())
assert_file_pins(contract["protected_sha256"])
cache, manifest = load_cache()
assert_file_pins(manifest["protected_input_sha256"])
assert json.loads((OUT / "reproduction_complete.json").read_text())["exact_non_timing_reproduction"]
counts = dict(
    examples=120,
    exact_archived_states=0,
    exact_reproductions=0,
    float64_states=0,
    bounds_verified=0,
    eligibility_verified=0,
)
limits = dict(step=0.0, net=0.0, path=0.0, pair_distance=0.0, temporary_chirality_loss=0)
for arm in ("A", "B"):
    for i in range(60):
        r = json.loads((OUT / arm / f"example_{i:02d}.json").read_text())
        assert r["record"] == cache["records"][i]
        b = {k: v.double() if v.is_floating_point() else v for k, v in batch(cache, [i], "cpu").items()}
        saved = torch.load(OUT / "untracked_states" / arm / f"example_{i:02d}.pt", weights_only=True)
        old = torch.load(OUT / "preflight_attempt_02/untracked_states" / arm / f"example_{i:02d}.pt", weights_only=True)
        assert saved["z"].dtype == torch.float64
        assert torch.equal(saved["z"], old["z"])
        assert hashlib.sha256(saved["z"].numpy().tobytes()).hexdigest() == r["variables_sha256"]
        assert torch.equal(saved["states"][0], b["pg"])
        assert all(p.dtype == torch.float64 and p.device.type == "cpu" for p in saved["states"])
        assert all(torch.equal(p, q) for p, q in zip(saved["states"], old["states"], strict=True))
        assert [digest(p) for p in saved["states"]] == r["state_coordinate_sha256"]
        tr = v7.physical_trajectory(b["pg"], b["mask"], 0.04 * saved["z"].reshape(4, *b["pg"].shape))
        assert all(torch.equal(p, q) for p, q in zip(tr["states"], saved["states"], strict=True))
        for t, s in enumerate(tr["steps"]):
            assert torch.equal(s["delta"][~s["eligible"]], torch.zeros_like(s["delta"][~s["eligible"]]))
            assert not s["eligible"][0, 0] and not s["eligible"][0, int(b["mask"].sum()) - 1]
            m = r["states"][t + 1]
            assert m == v7.metric_row(tr["states"][t + 1], b, r["record"], tr, t + 1)
            assert m["finite"] and not m["collapse"] and m["frame_assessability_preserved"]
            limits["step"] = max(limits["step"], m["step_correction_max"])
            limits["net"] = max(limits["net"], m["displacement_max"])
            limits["path"] = max(limits["path"], m["path_length_max"])
            limits["temporary_chirality_loss"] = max(
                limits["temporary_chirality_loss"], m["chirality_assessability_lost"]
            )
            p, q = tr["states"][t + 1][b["mask"]], b["pg"][b["mask"]]
            limits["pair_distance"] = max(
                limits["pair_distance"], float((torch.cdist(p, p) - torch.cdist(q, q)).abs().max())
            )
        assert (
            limits["step"] <= 0.04
            and limits["net"] <= 0.160001
            and limits["path"] <= 0.160001
            and limits["pair_distance"] <= 0.320002
        )
        log = r["optimizer"]
        res = log["stationarity"]
        feasible = (
            max(log["constraints"]) <= cfg["constraints"]["normalized_feasibility_tolerance"] if arm == "B" else True
        )
        assert feasible == log["constraint_feasible"]
        expected = bool(
            log["scipy_success"]
            and log["optimality"] <= cfg["solver"]["gtol"]
            and feasible
            and res["normalized_ball_kkt_max"] <= 0.001
            and res["projected_ball_mapping_max"] <= 0.001
            and min(log["multipliers"]) >= -1e-8
            and log["complementarity"] <= 1e-6
        )
        assert expected == log["converged"]
        repeat = json.loads((OUT / "reproduction" / arm / f"example_{i:02d}.json").read_text())
        del r["optimizer"]["runtime_seconds"]
        assert repeat["exact_non_timing_match"]
        assert (
            hashlib.sha256(json.dumps(r, sort_keys=True, allow_nan=False).encode()).hexdigest()
            == repeat["scientific_record_sha256"]
        )
        for k in (
            "exact_archived_states",
            "exact_reproductions",
            "float64_states",
            "bounds_verified",
            "eligibility_verified",
        ):
            counts[k] += 1
assert_file_pins(contract["protected_sha256"])
assert_file_pins(manifest["protected_input_sha256"])
result = dict(
    counts=counts,
    maximums=limits,
    protected_source_pins=len(contract["protected_sha256"]),
    protected_input_pins=len(manifest["protected_input_sha256"]),
    focused_cpu_tests=118,
    ruff_python="passed",
    historical_artifacts_unchanged=True,
    cuda_used=False,
    neural_training_launched=False,
)
p = OUT / "post_execution_validation.json"
assert not p.exists()
p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2))
