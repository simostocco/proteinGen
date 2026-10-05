from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from protein_distance_diffusion.models.e007_frozen_prior_geometry import (
    InvariantGeometryConditioner,
    conditioner_parameter_count,
)
from protein_distance_diffusion.training import e007_frozen_prior_geometry as phase4c

CONFIG = Path("configs/e007_frozen_prior_geometry_conditioning_v1.yaml")


def _rows() -> list[dict]:
    return [
        {
            "sample_id": f"x{index}",
            "sequence": "A" * length,
            "coordinate_rigid_shape_sha256": f"{index:064x}",
        }
        for index, length in enumerate([40, 45, 80, 85, 140, 145, 300, 305, 450, 455])
    ]


def test_invariant_conditioner_is_translation_rotation_invariant_and_masked() -> None:
    torch.manual_seed(3)
    model = InvariantGeometryConditioner(rbf_bins=8, hidden_width=16, shared_output_width=32).eval()
    coordinates = torch.randn(2, 7, 3)
    mask = torch.tensor([[True] * 7, [True] * 5 + [False] * 2])
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    first = model(coordinates, mask)
    second = model(coordinates @ rotation + 17.0, mask)
    assert torch.allclose(first, second, atol=2e-5, rtol=1e-5)
    assert torch.count_nonzero(first[1, 5:]) == 0


def test_null_conditioning_and_fixed_width_projection_preserve_budget() -> None:
    model = InvariantGeometryConditioner(rbf_bins=8, hidden_width=16, shared_output_width=32)
    coordinates = torch.randn(1, 5, 3)
    mask = torch.ones(1, 5, dtype=torch.bool)
    null = model(coordinates, mask, null_geometry=True)
    assert torch.equal(null[:, :1], null[:, 1:2])
    assert model.for_prior_width(null, 20).shape == (1, 5, 20)
    with pytest.raises(ValueError, match="exceeds"):
        model.for_prior_width(null, 33)


def test_every_conditioner_group_receives_finite_gradients() -> None:
    model = InvariantGeometryConditioner(rbf_bins=8, hidden_width=16, shared_output_width=32)
    output = model(torch.randn(2, 9, 3), torch.ones(2, 9, dtype=torch.bool))
    output.square().mean().backward()
    assert all(phase4c._parameter_groups_have_gradients(model).values())


def test_analytical_and_instantiated_parameter_budgets_match_exactly() -> None:
    config = phase4c._load_config(CONFIG)
    model = InvariantGeometryConditioner(**config["conditioner"])
    budget = phase4c.trainable_budget(config)
    assert budget["esm2_150m"] == conditioner_parameter_count(model)
    assert budget["esm2_150m"] == budget["progen2_151m"]
    assert budget["difference"] == 0
    assert budget["material_difference"] is False


def test_same_bin_derangement_is_deterministic_nonself_and_order_independent() -> None:
    config = phase4c._load_config(CONFIG)
    rows = _rows()
    first = phase4c.same_length_bin_derangement(rows, seed=7, strata=config["length_strata"])
    second = phase4c.same_length_bin_derangement(list(reversed(rows)), seed=7, strata=config["length_strata"])
    assert first == second
    assert all(source != donor for source, donor in first.items())
    lengths = {row["sample_id"]: len(row["sequence"]) for row in rows}
    assert all(
        phase4c.length_stratum(lengths[source], config["length_strata"])
        == phase4c.length_stratum(lengths[donor], config["length_strata"])
        for source, donor in first.items()
    )
    assert set(first) == set(first.values())


@pytest.mark.parametrize("arm", ["correct_geometry", "null_geometry"])
def test_nonshuffled_arms_never_construct_donors(monkeypatch: pytest.MonkeyPatch, arm: str) -> None:
    monkeypatch.setattr(
        phase4c,
        "same_length_bin_derangement",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not be called")),
    )
    singleton = [{"sample_id": "only", "sequence": "A" * 40}]
    assert phase4c.build_arm_donor_maps(
        arm, singleton, singleton, seed=1, strata=phase4c._load_config(CONFIG)["length_strata"]
    ) == ({}, {})


def test_shuffled_arm_rejects_singleton_before_any_model_factory_is_needed() -> None:
    singleton = [{"sample_id": "only", "sequence": "A" * 40}]
    with pytest.raises(ValueError, match="same-bin donor: 20-64"):
        phase4c.build_arm_donor_maps(
            "shuffled_geometry",
            singleton,
            singleton,
            seed=1,
            strata=phase4c._load_config(CONFIG)["length_strata"],
        )


def test_candidates_share_identical_bijective_mapping_and_donor_geometry_differs() -> None:
    rows = _rows()
    config = phase4c._load_config(CONFIG)
    esm = phase4c.build_arm_donor_maps("shuffled_geometry", rows, rows, seed=11, strata=config["length_strata"])
    progen = phase4c.build_arm_donor_maps("shuffled_geometry", rows, rows, seed=11, strata=config["length_strata"])
    assert esm == progen
    mapping = esm[0]
    by_id = {row["sample_id"]: row for row in rows}
    assert all(
        by_id[recipient]["coordinate_rigid_shape_sha256"] != by_id[donor]["coordinate_rigid_shape_sha256"]
        and by_id[recipient]["sequence"] == "A" * len(by_id[recipient]["sequence"])
        for recipient, donor in mapping.items()
    )


def test_smoke_panel_requires_two_eligible_samples_in_every_exercised_bin() -> None:
    config = phase4c._load_config(CONFIG)
    counts = phase4c.validate_panel_bin_minimum(_rows(), strata=config["length_strata"], minimum=2, split="train")
    assert set(counts.values()) == {2}
    with pytest.raises(ValueError, match="minimum contradiction"):
        phase4c.validate_panel_bin_minimum(_rows()[:-1], strata=config["length_strata"], minimum=2, split="train")


def test_deterministic_mask_is_reproducible_and_nonempty() -> None:
    first = phase4c.deterministic_mask(37, fraction=0.15, seed=9)
    assert torch.equal(first, phase4c.deterministic_mask(37, fraction=0.15, seed=9))
    assert not torch.equal(first, phase4c.deterministic_mask(37, fraction=0.15, seed=10))
    assert 0 < int(first.sum()) < len(first)


def test_bootstrap_and_candidate_classification_use_positive_improvement() -> None:
    interval = phase4c.paired_bootstrap_interval([0.2, 0.3, 0.4], seed=1, replicates=500)
    assert interval["lower_95"] > 0
    effect = {
        control: {
            "bootstrap_95": interval,
            "by_length_stratum": {"short": 0.2, "long": 0.3},
        }
        for control in ("null_geometry", "shuffled_geometry")
    }
    assert phase4c.classify_candidate(effect, length_strata=["short", "long"]) == ("usable_geometry_conditioning")
    broken = copy.deepcopy(effect)
    broken["null_geometry"]["bootstrap_95"]["lower_95"] = -0.1
    assert phase4c.classify_candidate(broken, length_strata=["short", "long"]) == ("no_supported_held_out_improvement")


def test_exact_checkpoint_resume_restores_state_and_cursor(tmp_path: Path) -> None:
    torch.manual_seed(5)
    model = InvariantGeometryConditioner(rbf_bins=8, hidden_width=16, shared_output_width=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    coordinates = torch.randn(1, 6, 3)
    mask = torch.ones(1, 6, dtype=torch.bool)
    model(coordinates, mask).square().mean().backward()
    optimizer.step()
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    path = tmp_path / "checkpoint.pt"
    phase4c.atomic_checkpoint(
        path,
        phase4c.checkpoint_payload(model, optimizer, update=17, cursor=23, config_sha256="abc"),
    )
    clone = InvariantGeometryConditioner(rbf_bins=8, hidden_width=16, shared_output_width=32)
    clone_optimizer = torch.optim.AdamW(clone.parameters(), lr=1e-3)
    assert phase4c.restore_checkpoint(path, clone, clone_optimizer, config_sha256="abc") == (17, 23)
    assert all(torch.equal(clone.state_dict()[name], value) for name, value in expected.items())
    with pytest.raises(ValueError, match="checkpoint contract"):
        phase4c.restore_checkpoint(path, clone, clone_optimizer, config_sha256="different")


def test_plan_verifies_phase4b_v3_and_has_no_side_effects(tmp_path: Path) -> None:
    config = phase4c._load_config(CONFIG)
    config["smoke_output_dir"] = str(tmp_path / "smoke")
    config["pilot_output_dir"] = str(tmp_path / "pilot")
    path = tmp_path / "config.yaml"
    import yaml

    path.write_text(yaml.safe_dump(config, sort_keys=False))
    result = phase4c.plan(path)
    assert result["isolated_process_count"] == 6
    assert result["trainable_parameter_budget"]["difference"] == 0
    assert result["dataset_scanned"] is False
    assert result["model_created"] is False
    assert result["optimizer_created"] is False
    assert not Path(config["smoke_output_dir"]).exists()
    assert not Path(config["pilot_output_dir"]).exists()


def test_parent_failure_atomically_finalizes_heartbeat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = phase4c._load_config(CONFIG)
    config["smoke_output_dir"] = str(tmp_path / "smoke")
    config["pilot_output_dir"] = str(tmp_path / "pilot")
    path = tmp_path / "config.yaml"
    import yaml

    path.write_text(yaml.safe_dump(config, sort_keys=False))

    def fail(config_path: str | Path, *, mode: str, resume: bool = False):
        del config_path, resume
        staging = Path(config[f"{mode}_output_dir"]).with_name(f".{Path(config[f'{mode}_output_dir']).name}.inprogress")
        staging.mkdir()
        phase4c._atomic_json(staging / "heartbeat.json", {"status": "running"})
        raise phase4c.WorkerFailure(
            "esm2_150m",
            "shuffled_geometry",
            {
                "status": "failed",
                "optimizer_updates": 0,
                "execution": {
                    "model_created": False,
                    "forward_performed": False,
                    "backward_performed": False,
                    "optimizer_created": False,
                    "checkpoint_written": False,
                },
            },
        )

    monkeypatch.setattr(phase4c, "_run_impl", fail)
    with pytest.raises(phase4c.WorkerFailure):
        phase4c.run(path, mode="smoke")
    heartbeat = json.loads((tmp_path / ".smoke.inprogress" / "heartbeat.json").read_text())
    assert heartbeat["status"] == "failed"
    assert heartbeat["failed_candidate"] == "esm2_150m"
    assert heartbeat["failed_arm"] == "shuffled_geometry"
    assert heartbeat["optimizer_updates"] == 0
    assert heartbeat["model_created"] is False
    assert heartbeat["authorizes_training"] is False


def test_phase4b_v3_hash_tampering_is_refused(tmp_path: Path) -> None:
    config = phase4c._load_config(CONFIG)
    config["prerequisites"]["phase4b_v3_report"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash contradiction"):
        phase4c.verify_prerequisites(config)


def test_all_publication_authorizations_are_false() -> None:
    assert all(value is False for value in phase4c.NON_AUTHORIZING.values())
    report = json.loads(
        Path(
            "reports/experiments/E007_matrix_sequence_cogeneration/pretrained_sequence_prior_smoke_v3/report.json"
        ).read_text()
    )
    assert report["decision"] == "esm2_and_progen2_advance"
