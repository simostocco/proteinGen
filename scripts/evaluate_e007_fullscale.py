"""Reuse the validated frozen diagnostics with all writes redirected to this control."""

# ruff: noqa: I001, E501
from e007_fullscale_runtime import RUN, DATA, MODEL_CONFIG, configure, atomic_json, sha, torch, yaml
import argparse
import json
import time
from pathlib import Path
from protein_distance_diffusion.models.e014_e007_control import E007ScaleupControl

LABELS = ("e007_epoch_2", "e007_epoch_1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("epoch", type=int, choices=(1, 2))
    epoch = parser.parse_args().epoch
    target = RUN / "epoch2_comparison"
    target.mkdir(exist_ok=True)
    # The reused modules expose their output root; no historical reports are edited.
    import e007_e014_comparison_protocol as protocol
    import collect_e007_e014_comparison as collector
    import analyze_e007_e014_comparison as analyzer

    configure()
    protocol.OUT = target
    collector.OUT = target
    collector.E007_LABELS = LABELS
    collector.SELECTED = LABELS
    analyzer.OUT = target
    analyzer.E007_LABELS = LABELS
    inventory = json.loads((RUN / "checkpoint_inventory.json").read_text())
    selected = []
    for e in range(1, epoch + 1):
        boundary = next(r for r in inventory["checkpoints"] if r["label"] == f"epoch_{e}_fraction_1.00")
        assert sha(boundary["path"]) == boundary["sha256"]
        selected.append(dict(boundary, label=f"e007_epoch_{e}"))
    atomic_json(target / "checkpoint_inventory.json", dict(selected=selected))
    atomic_json(
        target / "PROTOCOL.json",
        dict(
            reference_sha256=sha(collector.OLD / "shared_reference_descriptors.parquet"),
            training_matched=True,
            primary_E007="e007_epoch_2",
            primary_E014="epoch_2_fraction_1.00",
            bootstrap_draws=5000,
            no_prospective=True,
            minimum_prefix_endpoint=350,
        ),
    )

    def load(label):
        r = next(r for r in selected if r["label"] == label)
        path = Path(r["path"])
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = E007ScaleupControl(**yaml.safe_load(MODEL_CONFIG.read_text())["model"]).eval()
        model.load_state_dict(state["model"], strict=True)
        assert sum(p.numel() for p in model.parameters()) == 7586505
        return model, state, path

    collector.load_e007 = load
    for r in selected:
        label = r["label"]
        if not (target / f"{label}.INFERENCE_COMPLETE.json").exists():
            for suffix in ("paired.parquet", "descriptors.parquet", "gaussian_descriptors.parquet"):
                file = target / f"{label}.{suffix}"
                if file.exists():
                    file.rename(file.with_name(file.name + f".incomplete_attempt_{time.time_ns()}"))
    collector.collect()
    for r in selected:
        analyzer.analyze(r["label"])
    if epoch == 2:
        expected = "61a3acfe1024c9a6adcb21bfd864b3d0980e978d2fb90cbb02d9f1d095b9d902"
        assert sha(DATA / "e014/checkpoints/epoch_2_fraction_1.00.pt") == expected
        analyzer.finish()
        stats = json.loads((target / "bootstrap_intervals.json").read_text())["primary"]
        by = {r["metric"]: r for r in stats}
        better_v = by["aggregate_v_mse"]["ci_familywise"][1] < 0
        geometry_e007 = by["aggregate_local_mse"]["ci_familywise"][0] > 0
        geometry_e014 = by["aggregate_local_mse"]["ci_familywise"][1] < 0
        chiral_e007 = any(by[m]["ci_familywise"][0] > 0 for m in ("torsion_angular_400", "inversion_rate_400"))
        chiral_e014 = any(by[m]["ci_familywise"][1] < 0 for m in ("torsion_angular_400", "inversion_rate_400"))
        classification = (
            "M-E007-C"
            if better_v and geometry_e007 and chiral_e007
            else "M-E007-A"
            if geometry_e007 and chiral_e007
            else "M-E007-B"
            if better_v and geometry_e014 and chiral_e014
            else "unresolved_requires_scientific_review"
        )
        atomic_json(
            target / "COMPARISON_COMPLETE.json",
            dict(
                primary_classification=classification,
                training_matched=True,
                complete_epochs=2,
                exposures=460264,
                training=False,
                full_sampling=False,
                primary=stats,
                automatic_classification_rule="Conservative familywise directional checks; no equivalence inferred from nonsignificance; D requires scientific review of shared limitations",
            ),
        )
        lines = [
            "# Matched E007 epoch-2 comparison",
            "",
            f"Classification: **{classification}**. Two complete epochs, 460,264 exposures. E014 epoch 2 is the fixed primary comparator.",
            "",
            "| Primary metric | E007 epoch 2 | E014 epoch 2 | E014−E007 | 95% paired CI |",
            "| --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| {r['metric']} | {r['e007']:.8g} | {r['e014']:.8g} | {r['difference']:.8g} | {r['ci95']} |"
            for r in stats
        ]
        lines += [
            "",
            "All length-stratum, high-noise, marginal and Gaussian-prefix metrics and intervals are stored alongside this report. No full t=0 sampling or prospective holdout inspection occurred. A mixed or unresolved directional result requires scientific review rather than forcing one of A/B/C/D.",
        ]
        (target / "REPORT.md").write_text("\n".join(lines) + "\n")
    atomic_json(
        RUN / f"EPOCH_{epoch}_DIAGNOSTICS_COMPLETE.json",
        dict(epoch=epoch, outputs=str(target), minimum_sampling_endpoint=350, prospective_inspected=False),
    )


if __name__ == "__main__":
    main()
