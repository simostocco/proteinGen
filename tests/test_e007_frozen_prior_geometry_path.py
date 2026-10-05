from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from protein_distance_diffusion.evaluation import e007_frozen_prior_geometry_path as diagnostic
from protein_distance_diffusion.models.e007_frozen_prior_geometry import InvariantGeometryConditioner
from protein_distance_diffusion.models.e007_frozen_prior_geometry_capacity import (
    MediumInvariantGeometryConditioner,
)
from protein_distance_diffusion.training import e007_frozen_prior_geometry_capacity as phase4c1

CONFIG = Path("configs/e007_frozen_prior_geometry_path_diagnostic_v1.yaml")


class _Tokenizer:
    def __init__(self) -> None:
        self.values = {token: index for index, token in enumerate("ACDEFGHIKLMNPQRSTVWY12")}

    def token_to_id(self, value: str) -> int:
        return self.values[value]

    def encode(self, value: str):
        ids = [self.values[token] for token in value]
        return SimpleNamespace(ids=ids, tokens=list(value))

    def decode(self, values: list[int], *, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        inverse = {value: key for key, value in self.values.items()}
        return "".join(inverse[value] for value in values)


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layer = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.layer.weight)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        del attention_mask
        return torch.tanh(self.layer(hidden))


class _Prior(nn.Module):
    def __init__(self, width: int = 16, blocks: int = 3) -> None:
        super().__init__()
        self.config = SimpleNamespace(n_embd=width)
        self.embedding = nn.Embedding(32, width)
        self.transformer = SimpleNamespace(h=nn.ModuleList([_Block(width) for _ in range(blocks)]))
        self.head = nn.Linear(width, 32)

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def forward(self, *, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        hidden = inputs_embeds
        for block in self.transformer.h:
            hidden = block(hidden, attention_mask)
        return SimpleNamespace(logits=self.head(hidden))


def _row(sample_id: str = "recipient", length: int = 6) -> dict:
    coordinates = torch.tensor(
        [[float(index), float(index % 2), float((index * index) % 3)] for index in range(length)],
        dtype=torch.float32,
    )
    coordinates -= coordinates.mean(dim=0)
    return {
        "sample_id": sample_id,
        "sequence": "ACDEFG"[:length],
        "coordinates": coordinates,
        "residue_mask": torch.ones(length, dtype=torch.bool),
        "continuity_mask": torch.ones(max(length - 1, 0), dtype=torch.bool),
    }


def _small() -> InvariantGeometryConditioner:
    return InvariantGeometryConditioner(rbf_bins=8, hidden_width=12, shared_output_width=16)


def _medium() -> MediumInvariantGeometryConditioner:
    return MediumInvariantGeometryConditioner(
        rbf_bins=8,
        separation_bins=9,
        separation_width=4,
        pair_width=8,
        residue_width=12,
        shared_output_width=16,
        message_blocks=4,
        injection_depths=(0, 1, 2),
    )


def test_configuration_pins_completed_phase4c1_and_bounded_contract() -> None:
    config = diagnostic._load_config(CONFIG)
    assert config["prerequisites"]["phase4c1_report"]["sha256"] == (
        "31c2b560385d1a46182da6b642bac0713fd7815f925e99b10eff4bdf5fa16361"
    )
    assert config["prerequisites"]["phase4c1_protocol"]["sha256"] == (
        "a3ca81a848d5c6900ceb67c82d5cebbe0774a07499f89d642d8ce595626cdfab"
    )
    assert sum(1 for name, _ in diagnostic._iter_pins(config) if name.startswith("checkpoint:")) == 12
    assert config["optimization_diagnostic"]["maximum_updates"] == 200
    assert config["gate_sweep"]["effective_gate_values"] == [0.0, 0.018, 0.05, 0.25, 0.5, 1.0]


def test_injection_contract_is_explicit_and_causal() -> None:
    contract = diagnostic.injection_contract()
    assert contract["small"]["depths"] == ["input_embedding"]
    assert contract["medium"]["depths"] == [0, 5, 11]
    assert contract["alignment"]["bos_eos_padding"] == "exact zero"
    assert contract["alignment"]["target_exposure"] is False
    assert "geometry donor alone changes" in contract["alignment"]["correct_shuffled_pairing"]


@pytest.mark.parametrize("capacity", ["small", "medium"])
def test_diagnostic_forward_gate_zero_identity_and_gate_one_sensitivity(capacity: str) -> None:
    torch.manual_seed(4)
    prior = _Prior()
    tokenizer = _Tokenizer()
    conditioner = _small() if capacity == "small" else _medium()
    recipient = _row()
    donor = _row("donor")
    donor["coordinates"] = donor["coordinates"].flip(0).clone()
    zero_correct = diagnostic.diagnostic_forward(
        prior,
        tokenizer,
        conditioner,
        capacity,
        recipient,
        geometry_row=recipient,
        arm="correct_geometry",
        effective_gate=0.0,
        device=torch.device("cpu"),
    )
    zero_shuffled = diagnostic.diagnostic_forward(
        prior,
        tokenizer,
        conditioner,
        capacity,
        recipient,
        geometry_row=donor,
        arm="shuffled_geometry",
        effective_gate=0.0,
        device=torch.device("cpu"),
    )
    assert torch.equal(zero_correct["canonical_logits"], zero_shuffled["canonical_logits"])
    active = diagnostic.diagnostic_forward(
        prior,
        tokenizer,
        conditioner,
        capacity,
        recipient,
        geometry_row=donor,
        arm="shuffled_geometry",
        effective_gate=1.0,
        device=torch.device("cpu"),
    )
    assert diagnostic.paired_logit_effect(zero_correct, active)["logit_rms_difference"] > 0
    assert active["bos_exact_zero"] and active["eos_exact_zero"] and active["padding_exact_zero"]
    assert len(active["injections"]) == (1 if capacity == "small" else 3)
    assert all(row["injected_residual_rms"] > 0 for row in active["injections"])


@pytest.mark.parametrize("capacity", ["small", "medium"])
def test_conditioner_is_o3_invariant_and_pair_perturbation_changes_representation(capacity: str) -> None:
    torch.manual_seed(8)
    conditioner = _small() if capacity == "small" else _medium()
    row = _row()
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    moved = copy.deepcopy(row)
    moved["coordinates"] = row["coordinates"] @ rotation + 13.0
    original, first = diagnostic._conditioner_forward(
        conditioner, capacity, row, arm="correct_geometry", effective_gate=1.0, device=torch.device("cpu")
    )
    transformed, _ = diagnostic._conditioner_forward(
        conditioner, capacity, moved, arm="correct_geometry", effective_gate=1.0, device=torch.device("cpu")
    )
    assert torch.allclose(original, transformed, atol=2e-5, rtol=1e-5)
    perturbed = copy.deepcopy(row)
    perturbed["coordinates"][1, 0] += 0.25
    changed, second = diagnostic._conditioner_forward(
        conditioner, capacity, perturbed, arm="correct_geometry", effective_gate=1.0, device=torch.device("cpu")
    )
    assert not torch.equal(first["pair_encoder_output"], second["pair_encoder_output"])
    assert float((original - changed).square().mean().sqrt().detach()) > 0
    assert (
        diagnostic.single_pair_distance_response(
            conditioner,
            capacity,
            distance=3.8,
            perturbation=0.25,
        )
        > 0
    )


def test_representation_statistics_and_difference_are_finite_and_hashed() -> None:
    first = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    second = first + 0.5
    mask = torch.tensor([[True, False]])
    stats = diagnostic.tensor_statistics(first, mask)
    assert stats["finite"] is True
    assert stats["element_count"] == 2
    assert len(stats["sha256"]) == 64
    difference = diagnostic.representation_difference(first, second, mask)
    assert difference["rms_difference"] == pytest.approx(0.5)
    assert difference["cosine_distance"] > 0


def test_o3_scale_aware_comparison_passes_measured_roundoff_and_rejects_noninvariance() -> None:
    reference = torch.full((8, 20), 34.282169342041016, requires_grad=True)
    measured = reference.detach() + 3.750364703591913e-5
    result = diagnostic.o3_tensor_comparison(
        reference,
        measured,
        mask=None,
        absolute_tolerance=2e-5,
        relative_tolerance=1e-5,
    )
    assert result["passed"] is True
    assert result["rms_error"] == pytest.approx(3.750364703591913e-5, rel=0.02)
    assert result["rms_threshold"] > result["rms_error"]
    broken = diagnostic.o3_tensor_comparison(
        reference,
        reference.detach() + 0.1,
        mask=None,
        absolute_tolerance=2e-5,
        relative_tolerance=1e-5,
    )
    assert broken["passed"] is False


def test_o3_structural_validation_rejects_padding_masks_transform_and_order() -> None:
    row = _row()
    row["residue_mask"][-1] = False
    row["coordinates"][-1] = 0
    row["continuity_mask"][-1] = False
    transformed, metadata = diagnostic._o3_transformed_row(row)
    assert diagnostic._validate_o3_structure(row, transformed, metadata)["padding_exact_zero"] is True

    invalid_padding = copy.deepcopy(row)
    invalid_padding["coordinates"][-1, 0] = 1
    with pytest.raises(ValueError, match="padding"):
        diagnostic._o3_transformed_row(invalid_padding)

    altered_order = copy.deepcopy(transformed)
    altered_order["sequence"] = transformed["sequence"][::-1]
    with pytest.raises(ValueError, match="structural contradiction"):
        diagnostic._validate_o3_structure(row, altered_order, metadata)

    invalid_transform = copy.deepcopy(metadata)
    invalid_transform["matrix"][0][0] += 0.1
    with pytest.raises(ValueError, match="structural contradiction"):
        diagnostic._validate_o3_structure(row, transformed, invalid_transform)

    with pytest.raises(ValueError, match="mask shape"):
        diagnostic.o3_tensor_comparison(
            torch.ones(2, 3),
            torch.ones(2, 3),
            mask=torch.ones(4, dtype=torch.bool),
            absolute_tolerance=0,
            relative_tolerance=0,
        )


@pytest.mark.parametrize("capacity", ["small", "medium"])
def test_o3_end_to_end_diagnostic_preserves_masks_and_padding(capacity: str) -> None:
    torch.manual_seed(12)
    prior = _Prior()
    tokenizer = _Tokenizer()
    conditioner = _small() if capacity == "small" else _medium()
    row = _row()
    result = diagnostic.o3_invariance_diagnostic(
        prior,
        tokenizer,
        conditioner,
        capacity,
        row,
        reference_output=None,
        device=torch.device("cpu"),
        absolute_tolerance=2e-5,
        relative_tolerance=1e-5,
    )
    assert result["status"] == "passed"
    assert result["structural_checks"]["canonical_pair_distances"] is True
    assert result["sequence_separation"]["relative_tolerance"] == 0
    assert result["canonical_logits"]["finite"] is True


def test_gate_sweep_summary_is_paired_length_stratified_and_bootstrapped() -> None:
    config = diagnostic._load_config(CONFIG)
    config["gate_sweep"]["bootstrap_replicates"] = 100
    records = []
    for index, length in enumerate((64, 64, 128, 128)):
        effect = float(index + 1) / 100
        records.append(
            {
                "capacity": "small",
                "gate": 0.5,
                "length": length,
                "correct_vs_gate_zero": {"cross_entropy_change": effect},
                "correct_vs_shuffled": {"cross_entropy_change": effect * 2},
                "correct_vs_null": {"cross_entropy_change": effect * 3},
            }
        )
    summary = diagnostic.summarize_gate_sweep(records, config)
    assert set(summary["small/gate=0.5"]["by_length"]) == {"64", "128"}
    assert summary["small/gate=0.5"]["correct_vs_shuffled"]["lower_95"] > 0


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        ({"minimum_downstream_logit_rms_response": 0.0}, "conditioning_path_defect"),
        (
            {"minimum_downstream_logit_rms_response": 1.0, "minimum_correct_shuffled_representation_rms": 0.0},
            "representation_not_discriminative",
        ),
        (
            {
                "minimum_downstream_logit_rms_response": 1.0,
                "minimum_correct_shuffled_representation_rms": 1.0,
                "learned_gate_correct_shuffled_effect": 0.0,
                "less_suppressive_correct_shuffled_effect": 0.1,
            },
            "gate_suppression_supported",
        ),
        (
            {
                "minimum_downstream_logit_rms_response": 1.0,
                "minimum_correct_shuffled_representation_rms": 1.0,
                "bounded_correct_arm_overfit_advantage": 0.2,
            },
            "optimization_insufficient",
        ),
        (
            {
                "minimum_downstream_logit_rms_response": 1.0,
                "minimum_correct_shuffled_representation_rms": 1.0,
                "bounded_correct_arm_overfit_advantage": 0.0,
            },
            "architecture_not_using_matching_geometry",
        ),
        ({}, "bounded_diagnostic_inconclusive"),
    ],
)
def test_decision_categories_do_not_use_a_scalar_score(evidence: dict, expected: str) -> None:
    result = diagnostic.classify_diagnostic(evidence)
    assert result["classification"] == expected
    assert result["scalar_composite_score_used"] is False
    assert result["protein_geometry_declared_uninformative"] is False


def test_conditioner_checkpoint_loading_is_strict_and_weights_only(tmp_path: Path) -> None:
    model = _small()
    state = copy.deepcopy(model.state_dict())
    checkpoint = {
        "version": phase4c1.VERSION,
        "configuration_sha256": "a" * 64,
        "capacity": "small",
        "arm": "correct_geometry",
        "update": 1000,
        "conditioner": state,
        "optimizer": {"must_not_be_loaded": True},
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    record = {"path": str(path), "sha256": diagnostic.sha256_file(path)}
    loaded = diagnostic.load_conditioner_state(
        model,
        record,
        capacity="small",
        arm="correct_geometry",
        update=1000,
        phase4c1_configuration_sha256="a" * 64,
    )
    assert loaded["optimizer"] == {"must_not_be_loaded": True}
    with pytest.raises(ValueError, match="checkpoint contradiction"):
        diagnostic.load_conditioner_state(
            model,
            record,
            capacity="medium",
            arm="correct_geometry",
            update=1000,
            phase4c1_configuration_sha256="a" * 64,
        )


def test_plan_only_is_side_effect_free_and_never_touches_execution_paths(tmp_path: Path, monkeypatch) -> None:
    config = diagnostic._load_config(CONFIG)
    config["output_dir"] = str(tmp_path / "output")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.setattr(diagnostic, "verify_prerequisites", lambda _config: {"fixture": "f" * 64})
    monkeypatch.setattr(diagnostic, "_zero_update_diagnostics", lambda *_args: pytest.fail("model path called"))
    monkeypatch.setattr(phase4c1, "select_panel", lambda *_args, **_kwargs: pytest.fail("dataset scanned"))
    result = diagnostic.plan(path)
    assert result["model_created"] is False
    assert result["cuda_initialized"] is False
    assert result["optimizer_created"] is False
    assert result["dataset_scanned"] is False
    assert result["output_created"] is False
    assert not Path(config["output_dir"]).exists()


def test_execution_failure_publishes_terminal_non_authorizing_heartbeat(tmp_path: Path, monkeypatch) -> None:
    config = diagnostic._load_config(CONFIG)
    config["output_dir"] = str(tmp_path / "output")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.setattr(diagnostic, "verify_prerequisites", lambda _config: {})

    def fail_after_forward(_config, execution):
        execution["dataset_scanned"] = True
        execution["model_created"] = True
        execution["forward_performed"] = True
        raise RuntimeError("controlled failure")

    monkeypatch.setattr(diagnostic, "_zero_update_diagnostics", fail_after_forward)
    with pytest.raises(RuntimeError, match="controlled failure"):
        diagnostic.run(path)
    staging = tmp_path / ".output.inprogress"
    heartbeat = json.loads((staging / "heartbeat.json").read_text())
    assert heartbeat["status"] == "failed"
    assert heartbeat["error_type"] == "RuntimeError"
    assert heartbeat["execution"] == {
        "dataset_scanned": True,
        "model_created": True,
        "forward_performed": True,
        "backward_performed": False,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "checkpoint_written": False,
        "metric_published": False,
        "trajectory_published": False,
    }
    assert all(heartbeat[key] is False for key in diagnostic.NON_AUTHORIZING)
    assert not Path(config["output_dir"]).exists()


def test_atomic_report_protocol_separation_and_inventory_non_self_reference(tmp_path: Path) -> None:
    scientific = tmp_path / "zero.json"
    report = tmp_path / "report.json"
    protocol = tmp_path / "protocol.json"
    diagnostic.atomic_json(scientific, {"scientific": [1, 2, 3]})
    diagnostic.atomic_json(report, {"results": "scientific"})
    diagnostic.atomic_json(protocol, {"report_sha256": diagnostic.sha256_file(report)})
    inventory = diagnostic._artifact_inventory(tmp_path, [scientific])
    paths = {row["path"] for row in inventory["artifacts"]}
    assert paths == {"zero.json"}
    assert diagnostic.sha256_file(report) != diagnostic.sha256_file(protocol)
    assert "report.json" not in paths and "protocol.json" not in paths


def test_paired_control_identity_hashes_are_order_independent() -> None:
    panel = ["a", "b", "c"]
    donors = {"a": "b", "b": "c", "c": "a"}
    assert diagnostic.canonical_sha256(panel) == diagnostic.canonical_sha256(list(panel))
    assert diagnostic.canonical_sha256(donors) == diagnostic.canonical_sha256(dict(reversed(list(donors.items()))))
    assert hashlib.sha256(json.dumps(donors, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == (
        diagnostic.canonical_sha256(donors)
    )


def test_summary_uses_paired_final_evaluations_without_scalar_score() -> None:
    config = diagnostic._load_config(CONFIG)
    config["gate_sweep"]["bootstrap_replicates"] = 100
    zero = {
        "sensitivity_records": [{"perturbed_logit_rms_response": 0.01}],
        "minimum_correct_shuffled_representation_rms": 0.02,
        "gate_sweep_summary": {
            "medium/gate=0.018": {"correct_vs_shuffled": {"mean": 0.001}},
            "medium/gate=0.5": {"correct_vs_shuffled": {"mean": 0.02}},
        },
    }
    results = []
    final_update = str(config["optimization_diagnostic"]["maximum_updates"])
    for capacity in diagnostic.CAPACITIES:
        for policy in diagnostic.POLICIES:
            for arm, cross_entropy in (
                ("correct_geometry", 1.0),
                ("shuffled_geometry", 1.2),
                ("null_geometry", 1.3),
            ):
                results.append(
                    {
                        "capacity": capacity,
                        "policy": policy,
                        "arm": arm,
                        "evaluations": {
                            final_update: {
                                "records": [
                                    {"sample_id": "a", "cross_entropy": cross_entropy},
                                    {"sample_id": "b", "cross_entropy": cross_entropy + 0.1},
                                ]
                            }
                        },
                    }
                )
    summary = diagnostic._summarize(zero, results, config)
    assert summary["optimization_case_count"] == 12
    assert summary["decision"]["classification"] == "gate_suppression_supported"
    assert summary["paired_bootstrap_policy"]["scalar_composite_score_used"] is False
    assert summary["optimization_paired_effects"]["medium/gate_logit_zero"]["correct_vs_shuffled"][
        "mean"
    ] == pytest.approx(0.2)
