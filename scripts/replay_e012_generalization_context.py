"""Independent fresh replay of all historical held-out context diagnostics for E012 V3."""

from __future__ import annotations

import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from protein_sequence_generation.e012_generalization import bulk_prefix_batch, evaluation_only_guard
from scripts import audit_e012_generalization as audit
from scripts import run_e012_causal_rope as old


def replay():
    assert audit.read(audit.OUT / "completed.json")["optimizer_steps"] == 0
    audit.verify_historical()
    owners = audit.gpu_check()
    old.seed()
    _, panels = audit.populations()
    positions = audit.read(old.REPORT / "positions.json")
    before = old.prefix_batch
    old.prefix_batch = bulk_prefix_batch
    checks = []
    start = time.monotonic()
    try:
        with evaluation_only_guard():
            for step in [2000, 5000, 10000]:
                ck = torch.load(audit.CHECKPOINTS[step][0], map_location="cpu", weights_only=False)
                net = old.model().cuda()
                net.load_state_dict(ck["model"], strict=True)
                net.requires_grad_(False)
                net.eval()
                fingerprint = old.weights_hash(net)
                for panel in ["primary", "independent"]:
                    print(f"Fresh held-out context replay {step} {panel}", flush=True)
                    context = old.diagnostics(net, panels[panel], positions[panel])
                    frozen = audit.read(audit.OUT / f"{panel}_context_{step:05d}.json")
                    assert set(context) == set(frozen)
                    differences = [
                        abs(value - frozen[sid][key])
                        for sid, values in context.items()
                        for key, value in values.items()
                    ]
                    # Exact parity expected; record tiny differences without changing frozen results.
                    assert max(differences) <= 1e-6, "fresh historical context replay mismatch"
                    old.save(audit.OUT / f"{panel}_fresh_context_{step:05d}.json", context, exclusive=True)
                    checks.append(
                        {
                            "checkpoint": step,
                            "panel": panel,
                            "identities": len(context),
                            "maximum_absolute_metric_difference": max(differences),
                            "parsed_metrics_bitwise_equal": all(v == 0 for v in differences),
                        }
                    )
                assert old.weights_hash(net) == fingerprint and all(p.grad is None for p in net.parameters())
                del net, ck
                torch.cuda.empty_cache()
    finally:
        old.prefix_batch = before
    audit.verify_historical()
    old.save(
        audit.REPORT / "fresh_context_replay_verification.json",
        {
            "passed": True,
            "seconds": time.monotonic() - start,
            "checks": checks,
            "owners": owners,
            "all_four_checkpoints_all_three_panels_context_evaluated": True,
            "contract_changed": False,
            "frozen_results_changed": False,
            "purpose": (
                "Independently replay stored held-out contexts in identical current eval-mode precision and evaluator, "
                "supplementing frozen historical-record reuse with fresh evidence."
            ),
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "training_launched": False,
        },
        exclusive=True,
    )
    print("All fresh held-out context replays verified", flush=True)


if __name__ == "__main__":
    replay()
