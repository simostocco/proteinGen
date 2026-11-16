"""Refresh a compact D:-only launch snapshot without importing CUDA libraries."""
# Embedded report paragraphs intentionally exceed the source line limit.
# ruff: noqa: E501

from __future__ import annotations

import datetime
import hashlib
import json
import math
from pathlib import Path

RUN = Path("/mnt/d/Simone/proteinGen_data/sequence_foundation/e012_large_corpus_warmstart_20m_20261013_v1")


def read(name):
    path = RUN / name
    return json.loads(path.read_text()) if path.exists() else None


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    base = read("REPORT.json")
    status = read("status.json")
    candidates = sorted((RUN / "evaluations").glob("light_*.json"))
    latest = candidates[-1] if candidates else None
    light = read(str(latest.relative_to(RUN))) if latest else None
    parent = read("evaluations/parent.json")
    parent_light = {}
    strata = [(20, 64), (65, 128), (129, 256), (257, 384), (385, 500)]
    for name in ["primary", "independent"]:
        records = read(f"evaluations/parent_{name}_likelihood.json")
        selected = [row for low, high in strata for row in [r for r in records if low <= r["length"] <= high][:52]]
        assert len(selected) == 260
        parent_light[name] = {
            "proteins": len(selected),
            "sequence_perplexity": math.exp(sum(r["normal"] for r in selected) / len(selected)),
            "sample_ids": [r["sample_id"] for r in selected],
        }
    generation = read("generation/parent.json")
    telemetry = status.get("telemetry", {})
    source = Path(__file__).resolve().with_name("run_e012_large_corpus.py")
    assert sha(source) == read("final_replay.json")["source_hashes"]["scripts/run_e012_large_corpus.py"]
    assert sha(base["audit"]["parent_path"]) == base["audit"]["parent_sha256"]
    recoveries = [
        read(path.name + ".json")
        for path in RUN.glob("*.pt")
        if path.name.startswith(("recovery_", "production_initial", "milestone_"))
    ]
    newest = max(recoveries, key=lambda record: record["counters"]["updates"])
    assert sha(newest["path"]) == newest["sha256"]
    base.update(
        {
            "snapshot_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "status": status,
            "launch_command": read("launch_command.json"),
            "latest_light_evaluation": light,
            "parent_light_reference": parent_light,
            "latest_light_evaluation_path": str(latest) if latest else None,
            "final_replay": read("final_replay.json"),
            "verified_recovery_checkpoint": newest,
            "parent_checkpoint_unchanged": True,
            "source_matches_verified_replay": True,
            "resume_command": f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={source.parents[1]}/src /home/simostocco/miniforge3/envs/proteingen/bin/python -B {source} train --resume {newest['path']}",
            "evaluation_preflight": read("evaluation_preflight.json"),
            "evaluation_failure_recovery": read("evaluation_failure_recovery.json"),
            "scientific_conclusion": "Pending the 1M frozen likelihood and order-control evaluation. Early likelihood and smoke loss do not establish improvement in order learning or protein stability.",
        }
    )
    with (RUN / "LAUNCH_REPORT.json").open("w") as handle:
        json.dump(base, handle, indent=2, sort_keys=True)
        handle.write("\n")
    lines = [
        "# E012 large-corpus warm-start launch",
        "",
        f"Snapshot: {base['snapshot_utc']}. Active PID: {status.get('pid')}. State: {status['state']}.",
        "",
        f"Counters: {status['counters']['updates']:,} optimizer updates; {status['counters']['exposures']:,} unique sequences; {status['counters']['tokens']:,} residue tokens.",
        f"Measured production throughput: {telemetry.get('sequences_per_second', 0):.1f} sequences/sec and {telemetry.get('tokens_per_second', 0):.1f} residue tokens/sec.",
        f"ETA to 1M exposures: {telemetry.get('eta_1m_seconds', 0) / 60:.1f} minutes. Full 20M epoch projection: {telemetry.get('full_20m_epoch_estimate_seconds', 0) / 3600:.1f} hours. Rates include periodic work after production started; full milestone evaluation adds time.",
        "",
        "Strict 15,533,952-parameter parent load; fresh AdamW and deterministic schedule. Physical batch 16, accumulation 4, effective batch 64. Both 100/1000 operational smokes and exact checkpoint replay passed. Full 20M manifest consumer audit found zero exact overlap with all 23,307 historical validation sequences; certified protected homology exclusions retained.",
        "",
        "| Frozen panel | Full historical parent perplexity | Matched light parent perplexity | Production light perplexity |",
        "|---|---:|---:|---:|",
    ]
    for name in ["primary", "independent"]:
        new = f"{light[name]['aggregate']['sequence_perplexity']:.5f}" if light else "pending"
        lines.append(
            f"| {name} | {parent[name]['aggregate']['sequence_perplexity']:.5f} | {parent_light[name]['sequence_perplexity']:.5f} | {new} |"
        )
    lines.extend(
        [
            "",
            "The full parent panels have 2,048 sequences each; light panels have 260 each. For an identical-panel comparison, the machine-readable report includes their full records and frozen source panel definitions. Early light-panel values are descriptive; full-panel production comparison and paired order-sensitivity changes are due at 1M.",
            "",
            "Historical parent normal-minus-shuffled-prefix CE: "
            + "; ".join(
                f"{name} {parent[name]['context']['normal_minus_control']['shuffle']['delta']:.5f} nats"
                for name in ["primary", "independent"]
            )
            + ".",
            "Historical independent model-minus-corpus-composition CE: "
            + f"{parent['independent']['aggregate']['model_minus_corpus_composition']['delta']:.8f} nats (confidence interval spans zero). Detectable order sensitivity and competitive composition prediction are separate findings.",
            "",
            f"Parent generation: {generation['unique']}/{generation['n']} exact-unique canonical samples; all requested lengths met. Composition entropy {generation['composition_entropy_bits']:.3f} bits. No exact match in 4,096 scheduled TRAIN plus 4,096 protected references. Highest shortlisted SequenceMatcher ratio {max(r['sequence_matcher_ratio'] for r in generation['nearest']):.3f}; heuristic similarity is not a sensitive full-corpus homology search.",
            "",
            "Parent examples:",
            "",
            *[f"`{sequence}`" for sequence in generation["sequences"][:2]],
            "",
            "The initial evaluation failure occurred before production update 1. Its update-zero checkpoint, log and metadata were preserved; the corrected evaluator preflight and final replay passed before relaunch.",
            "",
            "Training automatically stops at exactly 1,000,000 unique exposures, with an immutable checkpoint, full frozen likelihood/order evaluation, bounded generation and a durable paused status for scientific review. The predeclared 20M scheduler horizon and saved raw states preserve a later extension under the same scientific protocol.",
            "",
            "E007 and E015 production remain paused; external designability and structural evaluation remain stopped. All generated artifacts, caches and scratch are on D:. Existing committed and uncommitted work, parent weights and the historical reservoir are preserved.",
            "",
            "Full provenance, configuration, checkpoint paths and resume command: `REPORT.json`. Current telemetry: `status.json` and `telemetry.jsonl`. Exact launch command: `launch_command.json`. Scientific outcome remains pending the 1M review.",
        ]
    )
    (RUN / "LAUNCH_REPORT.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "status": status["state"],
                "counters": status["counters"],
                "pid": status.get("pid"),
                "latest_validation": str(latest) if latest else None,
            }
        )
    )


if __name__ == "__main__":
    main()
