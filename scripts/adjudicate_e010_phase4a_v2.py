#!/usr/bin/env python3
"""Read-only CPU adjudication of a completed E010 Phase 4A v2 run."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/experiments/E010_global_equivariant_expressivity/phase4a_supervised_generalization_v2"
FINAL = OUT / "phase4a_training_v2.final"
PUBLISH = OUT / "phase4a_v2_scientific_review_v1"
BOUNDARIES = (5, 10, 20, 35, 50)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    def norm(x):
        if isinstance(x, torch.Tensor):
            t = x.detach().contiguous().cpu()
            return {
                "tensor": str(t.dtype),
                "shape": list(t.shape),
                "sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest(),
            }
        if isinstance(x, np.ndarray):
            a = np.ascontiguousarray(x)
            return {"array": str(a.dtype), "shape": list(a.shape), "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
        if isinstance(x, dict):
            return {str(k): norm(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
        if isinstance(x, (list, tuple)):
            return [norm(v) for v in x]
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        return {"repr": repr(x), "type": type(x).__qualname__}

    raw = json.dumps(norm(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def finite_tree(x):
    if isinstance(x, torch.Tensor):
        return bool(torch.isfinite(x).all())
    if isinstance(x, np.ndarray):
        return bool(np.isfinite(x).all())
    if isinstance(x, dict):
        return all(finite_tree(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return all(finite_tree(v) for v in x)
    if isinstance(x, float):
        return math.isfinite(x)
    return True


def bootstrap(values, reps, seed):
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    for st in range(0, reps, 256):
        n = min(256, reps - st)
        means[st : st + n] = arr[rng.integers(0, len(arr), (n, len(arr)))].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci95_percentile": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "replicates": reps,
        "unit": "development_identity",
        "quantity": "paired_percentage_improvement",
    }


def validate_and_review():
    if not FINAL.is_dir():
        raise FileNotFoundError(FINAL)
    if (OUT / "phase4a_training_v2.staging").exists():
        raise ValueError("both final and staging lifecycle paths exist")
    config_path = ROOT / "configs/e010_phase4a_supervised_generalization_v2.yaml"
    config = yaml.safe_load(config_path.read_text())
    prep = json.loads((OUT / "preparation_manifest.json").read_text())
    compat = json.loads((OUT / "phase4a_v2_compatibility_review.json").read_text())
    baseline = json.loads((OUT / "development_corrupted_baseline.json").read_text())
    metrics = json.loads((FINAL / "training_metrics.json").read_text())
    auth_false = {"downstream": False, "prospective": False, "phase4b": False}
    if (
        prep.get("training_identity_count") != 2048
        or prep.get("development_identity_count") != 320
        or prep.get("planned_optimizer_updates") != 1150
    ):
        raise ValueError("preparation counts mismatch")
    if (
        prep.get("training_started") is not False
        or prep.get("prospective_accessed") is not False
        or prep.get("phase4b_prepared") is not False
    ):
        raise ValueError("preparation authorization evidence mismatch")
    if compat.get("status") != "reviewed_non_authorizing_exact_continuation" or compat.get("authorization") != {
        "downstream": False,
        "new_run": False,
        "phase4b": False,
        "prospective": False,
    }:
        raise ValueError("compatibility authorization evidence mismatch")
    restricted = config.get("restrictions", {})
    if any(restricted.get(k) is not False for k in restricted):
        raise ValueError("configuration authorization/restriction field is not false")
    if (
        baseline.get("count") != 320
        or baseline.get("prospective_split_accessed") is not False
        or baseline.get("authorizes_downstream") is not False
    ):
        raise ValueError("baseline evidence contract mismatch")
    base = {r["sample_id"]: r for r in baseline["per_identity_baseline_and_paired_change_slots"]}
    if len(base) != 320 or not finite_tree(baseline):
        raise ValueError("baseline identities or values invalid")

    journal = []
    raw = (FINAL / "journal.jsonl").read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("truncated journal")
    for line in raw.splitlines():
        row = json.loads(line)
        digest = row.pop("record_sha256", None)
        expected = hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        if digest != expected:
            raise ValueError("journal record hash mismatch")
        row["record_sha256"] = digest
        journal.append(row)
    if len(journal) != 1150 or [x.get("global_update") for x in journal] != list(range(1, 1151)):
        raise ValueError("journal update count/prefix invalid")
    if any(
        x.get("epoch") != (i - 1) // 23 + 1
        or x.get("step_in_epoch") != (i - 1) % 23 + 1
        or x.get("training_started") is not True
        or x.get("prospective_split_accessed") is not False
        for i, x in enumerate(journal, 1)
    ):
        raise ValueError("journal deterministic schedule/authorization mismatch")
    if not finite_tree(journal):
        raise ValueError("non-finite journal metric")

    exposure_files = [FINAL / f"development_exposure_{e:02d}.json" for e in BOUNDARIES]
    results = []
    states = {}
    expected_ids = set(base)
    for e, path in zip(BOUNDARIES, exposure_files, strict=True):
        d = json.loads(path.read_text())
        cp = FINAL / f"checkpoint_at_exposure_{e:02d}.pt"
        if d.get("exposure") != e or d.get("evaluation_complete") is not True or d.get("development_count") != 320:
            raise ValueError(f"boundary {e} incomplete")
        rows = d.get("development_per_identity", [])
        ids = [r.get("sample_id") for r in rows]
        if len(ids) != 320 or set(ids) != expected_ids or len(set(ids)) != 320:
            raise ValueError(f"boundary {e} development panel mismatch")
        if any(r.get("prospective") is True or "prospective" in str(r.get("sample_id", "")).lower() for r in rows):
            raise ValueError("prospective identity detected")
        if not finite_tree(d):
            raise ValueError(f"boundary {e} has non-finite stored metrics")
        digest = sha(cp)
        if d.get("checkpoint_sha256") != digest:
            raise ValueError(f"boundary {e} checkpoint hash mismatch")
        state = torch.load(cp, map_location="cpu", weights_only=False)
        states[e] = state
        if state.get("global_update") != e * 23 or state.get("schedule_cursor") != e * 23:
            raise ValueError(f"boundary {e} checkpoint cursor mismatch")
        exposures = state.get("identity_exposures", {})
        if len(exposures) != 2048 or set(exposures.values()) != {e}:
            raise ValueError(f"training identities do not each have {e} exposures at boundary {e}")
        if not finite_tree(state):
            raise ValueError(f"boundary {e} checkpoint has non-finite state")
        recorded = d.get("metrics", {})
        recomputed = np.asarray([r["aligned_rmse_angstrom"] for r in rows], dtype=float)
        paired_baseline = np.asarray(
            [base[r["sample_id"]]["corrupted_input_aligned_rmse_angstrom"] for r in rows], dtype=float
        )
        if not math.isclose(
            float(recomputed.mean()),
            float(recorded.get("refined_mean_aligned_rmse_angstrom", float("nan"))),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError(f"boundary {e} stored refined mean disagrees with per-identity results")
        if not math.isclose(
            float(paired_baseline.mean()),
            float(recorded.get("baseline_mean_aligned_rmse_angstrom", float("nan"))),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError(f"boundary {e} stored baseline mean disagrees with pinned baseline")
        results.append((d, rows))
    final_state = torch.load(FINAL / "latest.pt", map_location="cpu", weights_only=False)
    selected_state = torch.load(FINAL / "selected_checkpoint.pt", map_location="cpu", weights_only=False)
    if (
        final_state.get("global_update") != 1150
        or final_state.get("schedule_cursor") != 1150
        or set(final_state.get("identity_exposures", {}).values()) != {50}
    ):
        raise ValueError("latest state is not exposure 50")
    if not finite_tree(final_state) or not finite_tree(selected_state):
        raise ValueError("latest or selected checkpoint non-finite")
    matches = [e for e, s in states.items() if canonical(s.get("model")) == canonical(selected_state.get("model"))]
    if len(matches) != 1:
        raise ValueError("selected model does not match exactly one boundary model")
    selected_e = matches[0]
    rule = min(
        results, key=lambda pair: (float(np.mean([r["aligned_rmse_angstrom"] for r in pair[1]])), pair[0]["exposure"])
    )[0]["exposure"]
    if (
        selected_e != rule
        or metrics.get("selected_exposure") != rule
        or sha(FINAL / "selected_checkpoint.pt") != metrics.get("selected_checkpoint_sha256")
    ):
        raise ValueError("selected exposure/checkpoint violates predeclared rule")
    if (
        metrics.get("training_updates") != 1150
        or metrics.get("sample_exposures") != 102400
        or metrics.get("per_identity_exposure_count", {}).get("all_exactly_50") is not True
    ):
        raise ValueError("training metrics completion contract mismatch")
    if (
        metrics.get("authorizes_downstream") is not False
        or metrics.get("downstream_authorized") is not False
        or metrics.get("prospective_authorized") is not False
        or metrics.get("phase4b_authorized") is not False
        or metrics.get("prospective_split_accessed") is not False
        or metrics.get("phase4b_prepared") is not False
        or metrics.get("authorization") != auth_false
    ):
        raise ValueError("training metrics authorization evidence mismatch")
    if any(r.get("selected") is not (r.get("exposure") == selected_e) for r in [x[0] for x in results]):
        raise ValueError("boundary selected flags inconsistent")

    reps = int(config["statistics"]["bootstrap_replicates"])
    seed = int(config["statistics"]["seed"])
    reviewed = []
    for d, rows in results:
        ordered = {r["sample_id"]: r for r in rows}
        paired = []
        for sid, b in base.items():
            r = ordered[sid]
            g0 = b["geometry_telemetry"]
            g1 = r["geometry_telemetry"]
            paired.append(
                {
                    "sample_id": sid,
                    "length": b["length"],
                    "stratum": b["stratum"],
                    "baseline_rmse": b["corrupted_input_aligned_rmse_angstrom"],
                    "refined_rmse": r["aligned_rmse_angstrom"],
                    "finite": r["finite"],
                    "baseline_geometry": g0,
                    "refined_geometry": g1,
                }
            )
        b = np.array([x["baseline_rmse"] for x in paired])
        a = np.array([x["refined_rmse"] for x in paired])
        imp = (b - a) / b
        overall = float(imp.mean())
        strata = {}
        for _idx, s in enumerate(config["selection"]["strata"]):
            sub = [x for x in paired if x["stratum"] == s["name"]]
            bb = np.array([x["baseline_rmse"] for x in sub])
            aa = np.array([x["refined_rmse"] for x in sub])
            strata[s["name"]] = {
                "count": len(sub),
                "baseline_rmse": float(bb.mean()),
                "refined_rmse": float(aa.mean()),
                "paired_percentage_improvement": float(((bb - aa) / bb).mean()),
                "paired_rmse_improvement_fraction_of_mean": float((bb.mean() - aa.mean()) / bb.mean()),
            }
        lengths = np.array([x["length"] for x in paired])
        slope0 = float(np.polyfit(lengths, b, 1)[0])
        slope1 = float(np.polyfit(lengths, a, 1)[0])
        dist_keys = [f"i_plus_{k}_distance_rmse_angstrom" for k in (1, 2, 3)]
        dist = {}
        for k in dist_keys:
            x = np.array([r["baseline_geometry"][k] for r in paired])
            y = np.array([r["refined_geometry"][k] for r in paired])
            dist[k] = {
                "baseline_rmse": float(x.mean()),
                "refined_rmse": float(y.mean()),
                "improvement_fraction": float((x.mean() - y.mean()) / x.mean()),
            }
        local0 = np.mean([[x["baseline_geometry"][k] for k in dist_keys] for x in paired], axis=1)
        local1 = np.mean([[x["refined_geometry"][k] for k in dist_keys] for x in paired], axis=1)
        inv0 = sum(x["baseline_geometry"]["chirality_inversions"] for x in paired)
        trip0 = sum(x["baseline_geometry"]["chirality_triplets"] for x in paired)
        inv1 = sum(x["refined_geometry"]["chirality_inversions"] for x in paired)
        trip1 = sum(x["refined_geometry"]["chirality_triplets"] for x in paired)
        coord = [
            x["sample_id"]
            for x in paired
            if x["refined_geometry"]["prediction_radius_gyration_angstrom"] < 1
            or x["refined_geometry"]["prediction_radius_gyration_angstrom"]
            / max(x["refined_geometry"]["target_radius_gyration_angstrom"], 1e-8)
            < 0.5
        ]
        diversity = {}
        for s in config["selection"]["strata"]:
            rows_s = [x for x in paired if x["stratum"] == s["name"]]
            pr = np.array([x["refined_geometry"]["prediction_radius_gyration_angstrom"] for x in rows_s])
            tr = np.array([x["refined_geometry"]["target_radius_gyration_angstrom"] for x in rows_s])
            ratio = float(pr.std() / tr.std()) if tr.std() > 1e-12 else None
            diversity[s["name"]] = {
                "predicted_radius_gyration_sd": float(pr.std()),
                "target_radius_gyration_sd": float(tr.std()),
                "sd_ratio_prediction_to_target": ratio,
                "collapse": bool(ratio is not None and ratio < 0.5),
            }
        mean_ratio = float((b.mean() - a.mean()) / b.mean())
        checks = {
            "overall_rmse_reduction_ge_30pct": mean_ratio >= 0.30,
            "every_length_stratum_reduction_ge_20pct": all(
                v["paired_rmse_improvement_fraction_of_mean"] >= 0.20 for v in strata.values()
            ),
            "no_length_stratum_worsens": all(v["refined_rmse"] <= v["baseline_rmse"] for v in strata.values()),
            "error_vs_length_slope_not_increased": slope1 <= slope0,
            "finite_outputs": all(x["finite"] and math.isfinite(x["refined_rmse"]) for x in paired),
            "chirality_inversion_rate_not_increased": inv1 / max(trip1, 1) <= inv0 / max(trip0, 1),
            "mean_i_plus_1_i_plus_2_i_plus_3_rmse_reduction_ge_20pct": (local0.mean() - local1.mean()) / local0.mean()
            >= 0.20,
            "no_coordinate_collapse": not coord,
            "no_diversity_collapse": not any(v["collapse"] for v in diversity.values()),
        }
        checks = {k: bool(v) for k, v in checks.items()}
        reviewed.append(
            {
                "exposure": d["exposure"],
                "global_update": d["global_update"],
                "selected": d["exposure"] == selected_e,
                "checkpoint_sha256": d["checkpoint_sha256"],
                "overall": {
                    "baseline_aligned_rmse": float(b.mean()),
                    "refined_aligned_rmse": float(a.mean()),
                    "rmse_reduction_fraction_of_means": mean_ratio,
                    "paired_percentage_improvement": overall,
                    "paired_percentage_improvement_bootstrap_95_ci": bootstrap(imp, reps, seed),
                },
                "length_strata": strata,
                "local_distance_rmse": dist,
                "mean_local_distance_improvement_fraction": float((local0.mean() - local1.mean()) / local0.mean()),
                "chirality": {
                    "baseline_inversions": inv0,
                    "baseline_triplets": trip0,
                    "baseline_rate": inv0 / max(trip0, 1),
                    "refined_inversions": inv1,
                    "refined_triplets": trip1,
                    "refined_rate": inv1 / max(trip1, 1),
                },
                "error_vs_length_slope": {"baseline": slope0, "refined": slope1},
                "coordinate_collapse_identities": coord,
                "diversity_by_stratum": diversity,
                "gate_checks": checks,
                "gate_adjudication": "passed" if all(checks.values()) else "failed",
            }
        )
    selected_review = next(b for b in reviewed if b["selected"])
    passed = all(selected_review["gate_checks"].values())
    classification = "pass_all_predeclared_gates" if passed else "fail_one_or_more_predeclared_gates"
    return {
        "schema": "e010_phase4a_v2_scientific_review_v1",
        "classification": classification,
        "selected_exposure": selected_e,
        "selection_rule": "lowest development mean aligned RMSE; earliest boundary wins exact ties",
        "validation": {
            "journal_updates": 1150,
            "training_identity_count": 2048,
            "all_training_identities_exactly_50_exposures": True,
            "development_identity_count": 320,
            "all_five_boundaries_match_pinned_development_panel": True,
            "prospective_identity_or_result_detected": False,
            "all_stored_metrics_finite": True,
            "all_boundary_checkpoint_hashes_match": True,
            "selected_model_matches_exactly_one_boundary": True,
            "latest_is_exposure_50": True,
            "authorization_fields_false": True,
        },
        "authorization": {
            "downstream": False,
            "prospective": False,
            "phase4b": False,
            "phase4b_preparation_supported_only_after_human_review": passed,
        },
        "boundaries": reviewed,
    }


def main():
    report = validate_and_review()
    PUBLISH.mkdir(parents=True, exist_ok=True)
    review_json = PUBLISH / "review.json"
    review_md = PUBLISH / "review.md"
    review_json.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    lines = [
        "# Phase 4A v2 scientific review",
        "",
        f"**Classification:** `{report['classification']}`  ",
        f"**Selected exposure:** {report['selected_exposure']}  ",
        (
            "**Authorization:** all downstream, prospective, and Phase 4B authorization "
            "fields remain false. A pass supports only preparation of Phase 4B after "
            "human review."
        ),
        "",
        "## Boundary results",
        "",
    ]
    for b in report["boundaries"]:
        ci = b["overall"]["paired_percentage_improvement_bootstrap_95_ci"]["ci95_percentile"]
        chiral = b["chirality"]
        collapsed_strata = ", ".join(k for k, v in b["diversity_by_stratum"].items() if v["collapse"]) or "none"
        lines += [
            f"### Exposure {b['exposure']}",
            "",
            (
                f"Overall aligned RMSE: {b['overall']['baseline_aligned_rmse']:.6f} → "
                f"{b['overall']['refined_aligned_rmse']:.6f}; paired percentage "
                f"improvement {b['overall']['paired_percentage_improvement']:.2%}; "
                f"deterministic paired bootstrap 95% CI [{ci[0]:.2%}, "
                f"{ci[1]:.2%}]."
            ),
            "",
            "| Length stratum | Baseline RMSE | Refined RMSE | Improvement |",
            "|---|---:|---:|---:|",
        ]
        for name, v in b["length_strata"].items():
            lines.append(
                f"| {name} | {v['baseline_rmse']:.6f} | {v['refined_rmse']:.6f} | "
                f"{v['paired_rmse_improvement_fraction_of_mean']:.2%} |"
            )
        lines += ["", "Local distance RMSE:", ""]
        for name, v in b["local_distance_rmse"].items():
            lines.append(
                f"- {name}: {v['baseline_rmse']:.6f} → {v['refined_rmse']:.6f} ("
                f"{v['improvement_fraction']:.2%} improvement)"
            )
        lines += [
            f"- Mean i+1/i+2/i+3 improvement: {b['mean_local_distance_improvement_fraction']:.2%}",
            (
                f"- Chirality inversions: {chiral['baseline_inversions']}/{chiral['baseline_triplets']} "
                f"({chiral['baseline_rate']:.4%}) "
                f"→ {chiral['refined_inversions']}/{chiral['refined_triplets']} ({chiral['refined_rate']:.4%})"
            ),
            (
                f"- Error versus length slope: {b['error_vs_length_slope']['baseline']:.8g} → "
                f"{b['error_vs_length_slope']['refined']:.8g}"
            ),
            (
                f"- Coordinate collapse identities: {len(b['coordinate_collapse_identities'])}; "
                f"diversity collapse strata: {collapsed_strata}"
            ),
            "- Gates: " + ", ".join(f"{k}={str(v).lower()}" for k, v in b["gate_checks"].items()),
            f"- Boundary adjudication: **{b['gate_adjudication']}**",
            " ",
        ]
    review_md.write_text("\n".join(lines) + "\n")
    inventory = []
    for p in sorted(PUBLISH.iterdir()):
        if p.is_file() and p.name not in {"artifact_inventory.json", "SHA256SUMS.json"}:
            inventory.append({"path": p.name, "size_bytes": p.stat().st_size, "sha256": sha(p)})
    inv = {
        "schema": "e010_phase4a_v2_review_inventory_v1",
        "source_final_recursive_inventory": json.loads(
            Path("/tmp/phase4a_v2_final_pre_edit_inventory.json").read_text()
        ),
        "published_files": inventory,
    }
    (PUBLISH / "artifact_inventory.json").write_text(json.dumps(inv, indent=2, sort_keys=True) + "\n")
    sums = {p.name: sha(p) for p in sorted(PUBLISH.iterdir()) if p.is_file() and p.name != "SHA256SUMS.json"}
    (PUBLISH / "SHA256SUMS.json").write_text(json.dumps(sums, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "selected_exposure": report["selected_exposure"],
                "published": str(PUBLISH.relative_to(ROOT)),
                "sha256": sums,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
