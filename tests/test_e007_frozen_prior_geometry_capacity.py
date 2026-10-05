from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from protein_distance_diffusion.models.e007_frozen_prior_geometry_capacity import (
    MediumInvariantGeometryConditioner,
    trainable_parameter_count,
)
from protein_distance_diffusion.training import e007_frozen_prior_geometry_capacity as phase4c1

CONFIG = Path("configs/e007_frozen_prior_geometry_capacity_v1.yaml")


def _config() -> dict:
    return phase4c1._load_config(CONFIG)


def _rows(lengths: list[int]) -> list[dict]:
    return [
        {
            "sample_id": f"sample-{index}",
            "sequence": "A" * length,
            "coordinate_sha256": f"coordinate-{index}",
            "coordinate_rigid_shape_sha256": f"shape-{index}",
        }
        for index, length in enumerate(lengths)
    ]


class _Tokenizer:
    def __init__(self) -> None:
        alphabet = "ACDEFGHIKLMNPQRSTVWY12"
        self.values = {token: index for index, token in enumerate(alphabet)}

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
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[torch.Tensor] = []

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        del attention_mask
        self.seen.append(hidden.detach().clone())
        return hidden


class _Prior(nn.Module):
    def __init__(self, width: int = 1024) -> None:
        super().__init__()
        self.config = SimpleNamespace(n_embd=width)
        self.embedding = nn.Embedding(32, width)
        self.transformer = SimpleNamespace(h=nn.ModuleList([_Block() for _ in range(12)]))
        self.blocks = self.transformer.h
        self.head = nn.Linear(width, 32)

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def forward(self, *, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        hidden = inputs_embeds
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        return SimpleNamespace(logits=self.head(hidden))


def test_exact_parameter_counts_and_medium_architecture() -> None:
    config = _config()
    assert phase4c1.analytical_parameter_counts(config) == {
        "small": 337_921,
        "medium": 3_608_999,
    }
    medium = MediumInvariantGeometryConditioner(**config["conditioners"]["medium"])
    assert trainable_parameter_count(medium) == 3_608_999
    assert len(medium.message_blocks) == 4
    assert medium.injection_depths == (0, 5, 11)
    assert max(medium.gate_values()) < 0.02


def test_capacity_initialization_is_matched_within_capacity_and_distinct_between_capacities() -> None:
    config = _config()
    first = phase4c1.initialize_conditioner("small", config, device=torch.device("cpu"))
    second = phase4c1.initialize_conditioner("small", config, device=torch.device("cpu"))
    medium = phase4c1.initialize_conditioner("medium", config, device=torch.device("cpu"))
    assert phase4c1.parameter_hash(first) == phase4c1.parameter_hash(second)
    assert phase4c1.parameter_hash(first) != phase4c1.parameter_hash(medium)


def test_exact_length_derangement_is_deterministic_bijective_and_shape_distinct() -> None:
    rows = _rows([64, 64, 64, 64, 128, 128])
    first = phase4c1.exact_length_derangement(rows, seed=11)
    second = phase4c1.exact_length_derangement(list(reversed(rows)), seed=11)
    assert first == second
    assert set(first) == set(first.values())
    by_id = {row["sample_id"]: row for row in rows}
    assert all(source != donor for source, donor in first.items())
    assert all(len(by_id[source]["sequence"]) == len(by_id[donor]["sequence"]) for source, donor in first.items())
    assert all(
        by_id[source]["coordinate_rigid_shape_sha256"] != by_id[donor]["coordinate_rigid_shape_sha256"]
        for source, donor in first.items()
    )


def test_exact_length_derangement_rejects_singletons_and_same_shapes() -> None:
    with pytest.raises(ValueError, match="length=64 count=1"):
        phase4c1.exact_length_derangement(_rows([64]), seed=1)
    rows = _rows([64, 64])
    rows[1]["coordinate_rigid_shape_sha256"] = rows[0]["coordinate_rigid_shape_sha256"]
    with pytest.raises(ValueError, match="distinct-shape"):
        phase4c1.exact_length_derangement(rows, seed=1)


def test_small_and_medium_reuse_identical_donor_mapping() -> None:
    rows = _rows([64, 64, 128, 128])
    small = phase4c1.exact_length_derangement(rows, seed=7603)
    medium = phase4c1.exact_length_derangement(rows, seed=7603)
    assert small == medium


def test_deterministic_panel_selection_preserves_exact_length_pairs_per_stratum() -> None:
    config = _config()
    rows = []
    for stratum in config["length_strata"]:
        length = int(stratum["minimum"])
        rows.extend(_rows([length] * 4))
        for index, row in enumerate(rows[-4:]):
            row["sample_id"] = f"{stratum['name']}-{index}"
            row["coordinate_rigid_shape_sha256"] = f"{stratum['name']}-shape-{index}"
    selected = phase4c1.deterministic_exact_length_selection(
        list(reversed(rows)), count=20, seed=7, strata=config["length_strata"]
    )
    assert len(selected) == len({row["sample_id"] for row in selected}) == 20
    mapping = phase4c1.exact_length_derangement(selected, seed=8)
    lengths = {row["sample_id"]: len(row["sequence"]) for row in selected}
    assert all(lengths[source] == lengths[donor] for source, donor in mapping.items())


def test_geometry_alignment_zeros_bos_eos_and_preserves_biology() -> None:
    biological = torch.randn(2, 7, 12)
    framed = phase4c1._framed_geometry(biological, 7, framed_length=12)
    assert framed.shape == (2, 12, 12)
    assert torch.count_nonzero(framed[:, 0]) == 0
    assert torch.count_nonzero(framed[:, 8:]) == 0
    assert torch.equal(framed[:, 1:8], biological)
    with pytest.raises(ValueError, match="too short"):
        phase4c1._framed_geometry(biological, 7, framed_length=8)


def test_medium_conditioner_is_o3_invariant_masked_and_sequence_independent() -> None:
    torch.manual_seed(4)
    model = MediumInvariantGeometryConditioner(
        rbf_bins=8,
        separation_bins=9,
        separation_width=4,
        pair_width=8,
        residue_width=12,
        shared_output_width=16,
        message_blocks=4,
        injection_depths=(0, 1, 2),
    ).eval()
    coordinates = torch.randn(1, 6, 3)
    mask = torch.tensor([[True, True, True, True, False, False]])
    continuity = torch.tensor([[True, True, True, False, False]])
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    first = model(coordinates, mask, continuity)
    second = model(coordinates @ rotation + 9.0, mask, continuity)
    assert torch.allclose(first, second, atol=2e-5, rtol=1e-5)
    assert torch.count_nonzero(first[:, 4:]) == 0
    # The conditioner API has no sequence-token input, so targets cannot enter this path.
    assert not any("token" in name or "sequence" in name for name, _value in model.named_parameters())


def test_multi_depth_injection_reaches_early_middle_late_only() -> None:
    prior = _Prior()
    conditioner = MediumInvariantGeometryConditioner(
        rbf_bins=8,
        separation_bins=9,
        separation_width=4,
        pair_width=8,
        residue_width=12,
        shared_output_width=1024,
        message_blocks=4,
        initial_gate_logit=0.0,
        injection_depths=(0, 5, 11),
    )
    framed = torch.ones(1, 6, 1024)
    inputs = torch.zeros_like(framed)
    with phase4c1._medium_hooks(prior, conditioner, framed):
        prior(inputs_embeds=inputs, attention_mask=torch.ones(1, 6, dtype=torch.long))
    assert torch.count_nonzero(prior.blocks[0].seen[-1]) > 0
    assert torch.count_nonzero(prior.blocks[4].seen[-1]) > 0  # Carries the early residual.
    assert torch.all(prior.blocks[5].seen[-1] > prior.blocks[4].seen[-1])
    assert torch.all(prior.blocks[11].seen[-1] > prior.blocks[10].seen[-1])


def test_medium_component_gradient_coverage_and_frozen_prior_hash() -> None:
    torch.manual_seed(9)
    prior = _Prior()
    tokenizer = _Tokenizer()
    for parameter in prior.parameters():
        parameter.requires_grad_(False)
    before = phase4c1.parameter_hash(prior)
    conditioner = MediumInvariantGeometryConditioner(
        rbf_bins=8,
        separation_bins=9,
        separation_width=4,
        pair_width=8,
        residue_width=12,
        shared_output_width=1024,
        message_blocks=4,
        injection_depths=(0, 5, 11),
    )
    row = {
        "sample_id": "x",
        "sequence": "ACDEFG",
        "coordinates": torch.randn(6, 3),
        "residue_mask": torch.ones(6, dtype=torch.bool),
        "continuity_mask": torch.ones(5, dtype=torch.bool),
    }
    loss, metrics = phase4c1.conditioned_progen_loss(
        prior,
        tokenizer,
        conditioner,
        "medium",
        row,
        geometry_row=row,
        arm="correct_geometry",
        device=torch.device("cpu"),
    )
    loss.backward()
    assert all(phase4c1.gradient_coverage(conditioner, "medium").values())
    assert metrics["bos_geometry_norm"] == metrics["eos_geometry_norm"] == 0
    assert phase4c1.parameter_hash(prior) == before


def test_checkpoint_exactly_restores_scheduler_rng_cursor_and_evaluations(tmp_path: Path) -> None:
    config = _config()
    model = phase4c1.initialize_conditioner("small", config, device=torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    torch.manual_seed(88)
    expected_next = torch.rand(3)
    torch.manual_seed(88)
    payload = phase4c1.checkpoint_payload(
        model,
        optimizer,
        scheduler,
        capacity="small",
        arm="correct_geometry",
        update=17,
        cursor=23,
        evaluations={"0": {"loss": 1.0}},
        config_sha256="abc",
        panel_hashes={"train": "t", "validation": "v"},
        donor_mapping_sha256="d",
    )
    path = tmp_path / "latest.pt"
    phase4c1.atomic_checkpoint(path, payload)
    clone = phase4c1.initialize_conditioner("small", config, device=torch.device("cpu"))
    clone_optimizer = torch.optim.AdamW(clone.parameters(), lr=1e-3)
    clone_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(clone_optimizer, T_max=10)
    update, cursor, evaluations = phase4c1.restore_checkpoint(
        path,
        clone,
        clone_optimizer,
        clone_scheduler,
        capacity="small",
        arm="correct_geometry",
        config_sha256="abc",
        panel_hashes={"train": "t", "validation": "v"},
        donor_mapping_sha256="d",
    )
    assert (update, cursor, evaluations) == (17, 23, {"0": {"loss": 1.0}})
    assert torch.equal(torch.rand(3), expected_next)
    assert phase4c1.parameter_hash(model) == phase4c1.parameter_hash(clone)


def test_capacity_decision_uses_effects_without_scalar_score() -> None:
    config = _config()
    supported = {
        "usable_geometry_conditioning": True,
        "effects": {
            name: {"bootstrap_95": {"mean": value}}
            for name, value in (("null_geometry", 0.1), ("shuffled_geometry", 0.12))
        },
    }
    comparisons = {"small": supported, "medium": copy.deepcopy(supported)}
    decision = phase4c1.capacity_decision(comparisons, config)
    assert decision["classification"] == "small_capacity_sufficient"
    assert decision["scalar_composite_score_used"] is False
    comparisons["medium"]["effects"]["null_geometry"]["bootstrap_95"]["mean"] = 0.2
    comparisons["medium"]["effects"]["shuffled_geometry"]["bootstrap_95"]["mean"] = 0.22
    assert phase4c1.capacity_decision(comparisons, config)["classification"] == "medium_capacity_preferred"


def test_memory_gates_accept_equality_and_reject_excess_or_nonfinite() -> None:
    limits = _config()["memory"]
    equal = {
        "peak_rss_mib": float(limits["maximum_rss_mib"]),
        "peak_cuda_allocated_mib": float(limits["maximum_cuda_allocated_mib"]),
        "peak_cuda_reserved_mib": float(limits["maximum_cuda_reserved_mib"]),
    }
    phase4c1.enforce_memory_limits(equal, limits)
    for key in equal:
        invalid = dict(equal)
        invalid[key] += 0.01
        with pytest.raises(MemoryError, match=key):
            phase4c1.enforce_memory_limits(invalid, limits)
    nonfinite = dict(equal)
    nonfinite["peak_cuda_reserved_mib"] = float("nan")
    with pytest.raises(MemoryError, match="peak_cuda_reserved_mib"):
        phase4c1.enforce_memory_limits(nonfinite, limits)


def test_failed_worker_finalizes_terminal_non_authorizing_heartbeat(tmp_path: Path, monkeypatch) -> None:
    config = _config()
    config["smoke_output_dir"] = str(tmp_path / "smoke")
    config["pilot_output_dir"] = str(tmp_path / "pilot")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))

    def fail(_path: str | Path, *, mode: str, resume: bool = False):
        del resume
        output = Path(config[f"{mode}_output_dir"])
        staging = output.with_name(f".{output.name}.inprogress")
        staging.mkdir()
        phase4c1.atomic_json(staging / "heartbeat.json", {"status": "running"})
        raise phase4c1.WorkerFailure(
            "medium",
            "shuffled_geometry",
            {"optimizer_updates": 0, "execution": {"model_created": False}},
        )

    monkeypatch.setattr(phase4c1, "_run_impl", fail)
    with pytest.raises(phase4c1.WorkerFailure):
        phase4c1.run(path, mode="smoke")
    heartbeat = json.loads((tmp_path / ".smoke.inprogress" / "heartbeat.json").read_text())
    assert heartbeat["status"] == "failed"
    assert heartbeat["failed_capacity"] == "medium"
    assert heartbeat["failed_arm"] == "shuffled_geometry"
    assert all(heartbeat[key] is False for key in phase4c1.NON_AUTHORIZING)


def test_plan_only_has_no_dataset_model_cuda_optimizer_or_output_side_effects(tmp_path: Path) -> None:
    config = _config()
    config["smoke_output_dir"] = str(tmp_path / "smoke")
    config["pilot_output_dir"] = str(tmp_path / "pilot")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    result = phase4c1.plan(path)
    assert result["isolated_process_count"] == 6
    assert result["parameter_counts"] == {"small": 337_921, "medium": 3_608_999}
    assert result["dataset_scanned"] is False
    assert result["model_created"] is False
    assert result["cuda_initialized"] is False
    assert result["optimizer_created"] is False
    assert result["output_created"] is False
    assert not Path(config["smoke_output_dir"]).exists()
    assert not Path(config["pilot_output_dir"]).exists()


def test_capacity_authorization_forwards_explicit_relocation_policy(monkeypatch) -> None:
    config = _config()
    observed = {}

    def authorize(_root, **kwargs):
        observed.update(kwargs)
        return object()

    monkeypatch.setattr(phase4c1, "authorize_rich_geometry_dataset", authorize)
    phase4c1._authorize(config)
    assert observed["protected_input_relocations"] == config["dataset"]["protected_input_relocations"]
