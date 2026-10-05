from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from protein_distance_diffusion.data.rich_geometry import RichDatasetAuthorization
from protein_distance_diffusion.data.rich_geometry_sidecars import (
    SIDECAR_SCHEMA_VERSION,
    rich_geometry_schema,
)
from protein_distance_diffusion.evaluation.e006_stage_a_context import (
    DIAGNOSTIC_VERSION,
    REQUIRED_SEQUENCE_COLUMNS,
    PanelMember,
    SequenceOnlyRichDataset,
    _aggregate,
    _allocate_independent_quotas,
    _bucket,
    _donor_map,
    _evaluate_model,
    _gate,
    _parse_history,
    _step0_payload,
    select_panels,
    training_unigram,
    validate_config,
    validate_sequence_schema_columns,
)
from protein_distance_diffusion.training.checkpointing import save_checkpoint
from protein_distance_diffusion.training.rich_codesign_production import verify_stage_a_context_gate


class _MetadataDataset:
    split = "validation"

    def __init__(self, count: int) -> None:
        self.rows = [(index, f"sample-{index:05d}", (32, 96, 192, 320, 480)[index % 5]) for index in range(count)]

    def iter_metadata(self):
        yield from self.rows


class _Rows:
    split = "validation"

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class _ContextModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward_sequence_pretraining(self, token_ids: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        previous = torch.roll(token_ids, 1, dims=1).clamp(2, 21)
        logits = torch.nn.functional.one_hot(previous, num_classes=22).float() * (3.0 * self.scale)
        return logits * residue_mask[..., None]


def _authorization(tmp_path: Path, rows: list[dict]) -> RichDatasetAuthorization:
    shard = tmp_path / "validation" / "part-000.parquet"
    shard.parent.mkdir(parents=True)
    table = pa.Table.from_pylist(rows)
    token_index = table.schema.get_field_index("token_ids")
    table = table.set_column(
        token_index,
        "token_ids",
        pa.array([row["token_ids"] for row in rows], type=pa.list_(pa.int16())),
    )
    pq.write_table(table, shard, row_group_size=2)
    protocol = {
        "shards": [{"dataset": "validation", "path": str(shard.relative_to(tmp_path))}],
    }
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    (tmp_path / "schema.json").write_text(
        json.dumps({"columns": {field.name: str(field.type) for field in table.schema}})
    )
    return RichDatasetAuthorization(
        root=tmp_path,
        protocol_sha256="protocol",
        schema_sha256="schema",
        vocabulary_sha256="vocabulary",
        normalization_sha256="normalization",
        shard_inventory_sha256="inventory",
        split_counts={"train": 0, "validation": len(rows)},
        observed_shard_hashes={},
    )


def _row(sample_id: str, tokens: list[int], split: str = "validation") -> dict:
    return {
        "sample_id": sample_id,
        "split": split,
        "sequence": "A" * len(tokens),
        "token_ids": tokens,
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "source_path": f"raw/{sample_id}.cif.gz",
        "source_sha256": "1" * 64,
        "npz_path": f"processed/{sample_id}.npz",
        "npz_sha256": "2" * 64,
        "mapping_evidence_sha256": "3" * 64,
        "ca_coordinates": [[0.0, 0.0, 0.0] for _ in tokens],
    }


def _config() -> dict:
    return {
        "version": DIAGNOSTIC_VERSION,
        "architecture_version": "e006_rich_geometry_codesign_v1",
        "seed": 7,
        "training_seed": 6,
        "device": "cpu",
        "output_dir": "unused",
        "model": {"pad_token_id": 0},
        "panels": {
            "original_size": 256,
            "independent_size": 2048,
            "minimum_per_nonempty_bucket": 64,
            "maximum_length": 500,
            "expected_original_sample_id_sha256": "a" * 64,
        },
        "corruption": {"mask_fractions": [0.3]},
        "baselines": {"smoothing": 1.0},
        "bootstrap": {"iterations": 100, "seed": 4},
        "interpretation_gate": {
            "unigram_improvement_nats": 0.05,
            "context_degradation_nats": 0.05,
            "minimum_passing_strata_fraction": 0.5,
            "sensitivity_thresholds_nats": [0.025, 0.05],
        },
        "evaluation": {"batch_size": 2},
        "memory": {"maximum_rss_mib": 4096},
    }


def test_sequence_only_reader_projects_columns_and_enforces_split(tmp_path: Path, monkeypatch) -> None:
    rows = [_row("a", [2, 3, 4]), _row("b", [5, 6])]
    authorization = _authorization(tmp_path, rows)
    observed_columns = []
    original = pq.ParquetFile.read_row_group

    def tracked(self, row_group, columns=None, **kwargs):
        observed_columns.append(tuple(columns or ()))
        return original(self, row_group, columns=columns, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", tracked)
    dataset = SequenceOnlyRichDataset(authorization, split="validation")
    assert dataset[0]["sample_id"] == "a"
    assert "ca_coordinates" not in dataset[0]
    assert all("ca_coordinates" not in columns for columns in observed_columns)
    assert all("experimental_method" not in columns for columns in observed_columns)

    bad = [_row("wrong", [2, 3], split="train")]
    bad_root = tmp_path / "bad"
    bad_dataset = SequenceOnlyRichDataset(_authorization(bad_root, bad), split="validation")
    with pytest.raises(ValueError, match="physical split"):
        _ = bad_dataset[0]


def test_exact_definitive_sidecar_schema_supports_minimal_sequence_projection() -> None:
    definitive = rich_geometry_schema()
    result = validate_sequence_schema_columns(definitive.names, source="definitive v2 Arrow schema")
    assert "experimental_method" not in definitive.names
    assert result["required_projected_columns"] == list(REQUIRED_SEQUENCE_COLUMNS)
    assert result["missing_required_columns"] == []
    assert result["constructs_rich_pair_features"] is False
    assert result["feature_complexity"] == "O(N)"
    with pytest.raises(ValueError, match="token_ids"):
        validate_sequence_schema_columns(
            [name for name in definitive.names if name != "token_ids"],
            source="definitive v2 Arrow schema",
        )


def test_panels_are_exact_deterministic_disjoint_and_order_independent() -> None:
    dataset = _MetadataDataset(3000)
    first, diagnostics = select_panels(
        dataset,
        original_size=256,
        independent_size=2048,
        training_seed=6006,
        diagnostic_seed=6106,
    )
    second, _ = select_panels(
        dataset,
        original_size=256,
        independent_size=2048,
        training_seed=6006,
        diagnostic_seed=6106,
    )
    reversed_dataset = _MetadataDataset(0)
    reversed_dataset.rows = list(reversed(dataset.rows))
    reordered, _ = select_panels(
        reversed_dataset,
        original_size=256,
        independent_size=2048,
        training_seed=6006,
        diagnostic_seed=6106,
    )
    assert [item.sample_id for item in first["original_validation"]] == [
        item.sample_id for item in second["original_validation"]
    ]
    assert [item.sample_id for item in first["independent_diagnostic"]] == [
        item.sample_id for item in reordered["independent_diagnostic"]
    ]
    assert len(first["original_validation"]) == 256
    assert len(first["independent_diagnostic"]) == 2048
    assert not (
        {item.sample_id for item in first["original_validation"]}
        & {item.sample_id for item in first["independent_diagnostic"]}
    )
    assert diagnostics["panels_disjoint"] is True
    assert diagnostics["plan_gate"]["passed"] is True
    allocation = diagnostics["independent_panel_allocation"]
    assert allocation["candidate_population_before_original_panel_exclusion"]["total"] == 3000
    assert allocation["candidate_population_after_original_panel_exclusion"]["total"] == 2744
    assert sum(allocation["requested_allocation_by_length_bucket"].values()) == 2048
    assert allocation["requested_allocation_by_length_bucket"] == allocation["realized_allocation_by_length_bucket"]
    assert all(count >= 64 for count in allocation["realized_allocation_by_length_bucket"].values())
    assert diagnostics["panels"]["original_validation"]["sample_id_sha256"] == (
        "7769d3eaca69048d5fc57f11631d55d118bd6e799e1d7d03868196fd5b6f2554"
    )
    assert diagnostics["panels"]["independent_diagnostic"]["sample_id_sha256"] == (
        "bbf8ef0ea9a4e5a2d2a1d7b854bd9844e1c1a67ae2f2f99d4fb5e994ee0ccfac"
    )


@pytest.mark.parametrize(
    ("length", "boundary"),
    [(64, 64), (65, 128), (128, 128), (129, 256), (256, 256), (257, 384), (384, 384), (385, 500), (500, 500)],
)
def test_contextual_length_bucket_boundaries(length: int, boundary: int) -> None:
    assert _bucket(length) == boundary


@pytest.mark.parametrize("length", [0, -1, 501])
def test_contextual_length_buckets_reject_out_of_range_lengths(length: int) -> None:
    with pytest.raises(ValueError, match="sequence length"):
        _bucket(length)


def test_largest_remainder_allocation_is_proportional_and_exact() -> None:
    quotas, diagnostics = _allocate_independent_quotas(
        {64: 1000, 128: 500, 256: 250, 384: 125, 500: 125},
        count=1000,
        minimum_per_nonempty_bucket=64,
    )
    assert sum(quotas.values()) == 1000
    assert all(value >= 64 for value in quotas.values())
    assert quotas[64] > quotas[128] > quotas[256]
    assert diagnostics["allocation_method"].startswith("minimum_then_largest_remainder")


def test_undersupplied_bucket_is_fully_selected_and_redistributed() -> None:
    quotas, diagnostics = _allocate_independent_quotas(
        {64: 1990, 128: 58, 256: 100, 384: 100, 500: 100},
        count=2048,
        minimum_per_nonempty_bucket=64,
    )
    assert sum(quotas.values()) == 2048
    assert quotas[128] == 58
    assert quotas[256] >= 64 and quotas[384] >= 64 and quotas[500] >= 64
    assert diagnostics["redistribution_events"] == [
        {
            "event": "minimum_quota_shortage",
            "length_bucket": "65-128",
            "available": 58,
            "configured_minimum": 64,
            "included": 58,
            "shortage": 6,
        }
    ]


def test_multiple_exhausted_buckets_use_deterministic_global_capacity() -> None:
    available = {64: 2400, 128: 20, 256: 30, 384: 100, 500: 100}
    first, diagnostics = _allocate_independent_quotas(
        available,
        count=2048,
        minimum_per_nonempty_bucket=64,
    )
    second, _ = _allocate_independent_quotas(
        available,
        count=2048,
        minimum_per_nonempty_bucket=64,
    )
    assert first == second
    assert first[128] == 20
    assert first[256] == 30
    assert sum(first.values()) == 2048
    assert [item["length_bucket"] for item in diagnostics["redistribution_events"]] == ["65-128", "129-256"]


def test_genuinely_insufficient_independent_population_fails_clearly() -> None:
    with pytest.raises(ValueError, match=r"requested=2048, available=2000"):
        _allocate_independent_quotas(
            {64: 400, 128: 400, 256: 400, 384: 400, 500: 400},
            count=2048,
            minimum_per_nonempty_bucket=64,
        )


def test_observed_short_sequence_domination_pattern_is_not_reproduced() -> None:
    dataset = _MetadataDataset(0)
    lengths = [32] * 2400 + [96] * 100 + [192] * 100 + [320] * 100 + [480] * 100
    dataset.rows = [(index, f"domination-{index:05d}", length) for index, length in enumerate(lengths)]
    panels, diagnostics = select_panels(
        dataset,
        original_size=256,
        independent_size=2048,
        training_seed=6006,
        diagnostic_seed=6106,
    )
    counts = diagnostics["independent_panel_allocation"]["realized_allocation_by_length_bucket"]
    assert len(panels["independent_diagnostic"]) == 2048
    assert counts["1-64"] < 1990
    assert all(counts[label] >= 64 for label in ("65-128", "129-256", "257-384", "385-500"))
    assert diagnostics["plan_gate"]["passed"] is True


def test_duplicate_validation_sample_ids_are_rejected() -> None:
    dataset = _MetadataDataset(3000)
    dataset.rows[-1] = (dataset.rows[-1][0], dataset.rows[0][1], dataset.rows[-1][2])
    with pytest.raises(ValueError, match="sample_id is duplicated"):
        select_panels(
            dataset,
            original_size=256,
            independent_size=2048,
            training_seed=6006,
            diagnostic_seed=6106,
        )


def test_unigram_uses_training_tokens_only_and_smoothing() -> None:
    class Train:
        def iter_token_ids(self):
            yield 0, "train-a", [2, 2, 3]
            yield 1, "train-b", [2, 4]

    result = training_unigram(Train(), smoothing=1.0)
    assert result["training_sample_count"] == 2
    assert result["token_count"] == 5
    assert result["token_counts"][:3] == [3, 1, 1]
    assert sum(result["token_frequencies"]) == pytest.approx(1.0)


def test_paired_conditions_are_deterministic_mask_targets_and_use_no_geometry(tmp_path: Path) -> None:
    rows = [
        _row("a", [2, 3, 4, 5]),
        _row("b", [3, 4, 5, 6]),
        _row("c", [4, 5, 6, 7]),
        _row("d", [5, 6, 7, 8]),
    ]
    dataset = _Rows(rows)
    panel = [PanelMember(index, row["sample_id"], 4, 64) for index, row in enumerate(rows)]
    panels = {"original_validation": panel}
    unigram = {
        "token_frequencies": [0.05] * 20,
        "length_bucketed": {"64": {"token_frequencies": [0.05] * 20}},
    }
    model = _ContextModel().eval()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    first, changes = _evaluate_model(
        model,
        model_variant="best",
        dataset=dataset,
        panels=panels,
        unigram=unigram,
        config=_config(),
        device=torch.device("cpu"),
        heartbeat_path=None,
    )
    second, _ = _evaluate_model(
        model,
        model_variant="best",
        dataset=dataset,
        panels=panels,
        unigram=unigram,
        config=_config(),
        device=torch.device("cpu"),
        heartbeat_path=None,
    )
    assert first == second
    assert changes
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert {record["subset"] for record in first} == {"all_valid", "corrupted", "visible"}
    assert all(record["metrics"]["normal"]["token_count"] > 0 for record in first if record["subset"] == "corrupted")


def _gate_records(context_effect: float, unigram_effect: float) -> list[dict]:
    records = []
    for panel in ("original_validation", "independent_diagnostic"):
        for fraction in (0.15, 0.30):
            for bucket in (64, 128):
                metrics = {}
                for condition, ce in {
                    "normal": 2.0,
                    "visible_shuffle": 2.0 + context_effect,
                    "permuted_conditioning": 2.0 + context_effect,
                    "null_conditioning": 2.0 + context_effect,
                    "uniform": 3.0,
                    "training_unigram": 2.0 + unigram_effect,
                    "length_bucketed_unigram": 2.0 + unigram_effect,
                }.items():
                    metrics[condition] = {"token_count": 5, "ce": ce, "top1": 0.2, "top3": 0.4, "top5": 0.6}
                records.append(
                    {
                        "model_variant": "best",
                        "panel": panel,
                        "sample_id": f"{panel}-{fraction}-{bucket}",
                        "length_bucket": bucket,
                        "mask_fraction": fraction,
                        "subset": "corrupted",
                        "metrics": metrics,
                    }
                )
    return records


@pytest.mark.parametrize(
    ("context_effect", "unigram_effect", "classification"),
    [
        (0.2, 0.2, "contextual_learning_verified"),
        (0.2, 0.0, "marginal_frequency_only"),
        (0.0, 0.2, "conditioning_path_ineffective"),
    ],
)
def test_interpretation_gate_classifications(context_effect, unigram_effect, classification) -> None:
    aggregates = _aggregate(_gate_records(context_effect, unigram_effect), bootstrap_iterations=100, bootstrap_seed=1)
    assert _gate(aggregates, _config())["classification"] == classification


def test_interpretation_gate_can_be_inconclusive() -> None:
    records = _gate_records(0.2, 0.2)
    records[0]["metrics"]["training_unigram"]["ce"] = 2.0
    aggregates = _aggregate(records, bootstrap_iterations=100, bootstrap_seed=1)
    assert _gate(aggregates, _config())["classification"] == "inconclusive"


def test_step0_requires_hash_and_full_compatibility(tmp_path: Path) -> None:
    best = {
        "architecture_version": "architecture",
        "dataset_identity": "dataset",
        "calibration_sha256": "calibration",
        "production_selection_sha256": "selection",
        "model": {"weight": torch.ones(2)},
    }
    payload = {
        **{key: best[key] for key in best if key != "model"},
        "optimizer_step": 0,
        "stage": "sequence-pretrain",
        "status": "recovery_only",
        "authorizes_training": False,
        "model": {"weight": torch.zeros(2)},
    }
    path = tmp_path / "step0.pt"
    save_checkpoint(path, payload)
    digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    loaded, status = _step0_payload({"step0_checkpoint": {"path": str(path), "sha256": digest}}, best)
    assert loaded["optimizer_step"] == 0
    assert status["status"] == "verified"
    payload["dataset_identity"] = "wrong"
    save_checkpoint(path, payload)
    changed = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="compatibility"):
        _step0_payload({"step0_checkpoint": {"path": str(path), "sha256": changed}}, best)


def test_history_parses_nested_losses_validation_and_overflows(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "\n".join(
            json.dumps(item)
            for item in (
                {"optimizer_step": 1, "losses": {"sequence": 3.0}, "amp_overflows_total": 0},
                {"optimizer_step": 2, "losses": {"sequence": 2.0}, "amp_overflows_total": 1},
            )
        )
    )
    validation = tmp_path / "validation.jsonl"
    validation.write_text(json.dumps({"optimizer_step": 2, "sequence_cross_entropy": 2.5}) + "\n")

    def sha(path):
        return __import__("hashlib").sha256(path.read_bytes()).hexdigest()

    config = {
        "training_history": {
            "rolling_window_records": 1,
            "files": [
                {"kind": "metrics", "path": str(metrics), "sha256": sha(metrics)},
                {"kind": "validation", "path": str(validation), "sha256": sha(validation)},
            ],
        }
    }
    result = _parse_history(config)
    assert result["metrics_record_count"] == 2
    assert result["amp_overflow_count"] == 1
    assert result["best_selection"]["optimizer_step"] == 2


def test_context_gate_is_non_authorizing_strict_and_tamper_evident(tmp_path: Path) -> None:
    report = {
        "status": "completed",
        "version": DIAGNOSTIC_VERSION,
        "classification": "contextual_learning_verified",
        "scientific_gate": {
            "classification": "contextual_learning_verified",
            "acceptable_for_stage_b": True,
        },
        "checkpoint_sha256": "checkpoint",
        "dataset_protocol_sha256": "dataset",
        "protected_inputs_unchanged": True,
        "authorizes_training": False,
        "authorizes_joint_training": False,
        "authorizes_evaluation": False,
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    assert (
        verify_stage_a_context_gate(
            path,
            digest,
            checkpoint_sha256="checkpoint",
            dataset_protocol_sha256="dataset",
            acceptable_classifications=["contextual_learning_verified"],
        )["classification"]
        == "contextual_learning_verified"
    )
    with pytest.raises(ValueError, match="classification"):
        verify_stage_a_context_gate(
            path,
            digest,
            checkpoint_sha256="checkpoint",
            dataset_protocol_sha256="dataset",
            acceptable_classifications=["inconclusive"],
        )


def test_production_config_is_valid_and_stage_b_gate_is_unpinned() -> None:
    from protein_distance_diffusion.config import load_yaml
    from protein_distance_diffusion.training.rich_codesign_production import validate_phase3_config

    config = load_yaml("configs/e006_stage_a_context_diagnostic.yaml")
    validate_config(config)
    joint = load_yaml("configs/e006_rich_geometry_joint_train.yaml")
    assert joint["stage_a"]["checkpoint_path"].endswith(
        "sequence_pretrain_v5_warm_start/checkpoints/best_context_verified.pt"
    )
    assert joint["stage_a"]["requires_contextual_checkpoint"] is True
    assert joint["stage_a"]["checkpoint_sha256"] is None
    assert joint["stage_a"]["context_diagnostic_required"] is True
    assert joint["stage_a"]["context_diagnostic_sha256"] is None
    assert joint["stage_a"]["acceptable_context_classifications"] == ["contextual_learning_verified"]
    with pytest.raises(ValueError, match="authorized Stage-A checkpoint"):
        validate_phase3_config(joint, mode="joint-train")


def test_donor_permutation_stays_within_length_bucket() -> None:
    panel = [
        PanelMember(0, "a", 60, 64),
        PanelMember(1, "b", 64, 64),
        PanelMember(2, "c", 100, 128),
        PanelMember(3, "d", 120, 128),
    ]
    donors = _donor_map(panel)
    assert all(donors[item.sample_id].length_bucket == item.length_bucket for item in panel)
    assert all(donors[item.sample_id].sample_id != item.sample_id for item in panel)


def test_synthetic_run_publishes_atomic_non_authorizing_artifacts(tmp_path: Path, monkeypatch) -> None:
    from protein_distance_diffusion.evaluation import e006_stage_a_context as context

    rows = [
        _row("a", [2, 3, 4, 5]),
        _row("b", [3, 4, 5, 6]),
        _row("c", [4, 5, 6, 7]),
        _row("d", [5, 6, 7, 8]),
    ]
    validation = _Rows(rows)
    panel = [PanelMember(index, row["sample_id"], 4, 64) for index, row in enumerate(rows)]
    authorization = RichDatasetAuthorization(
        root=tmp_path,
        protocol_sha256="dataset-protocol",
        schema_sha256="schema",
        vocabulary_sha256="vocabulary",
        normalization_sha256="normalization",
        shard_inventory_sha256="inventory",
        split_counts={"train": 2, "validation": 4},
        observed_shard_hashes={},
    )
    protected = tmp_path / "checkpoint.pt"
    protected.write_bytes(b"immutable")
    digest = __import__("hashlib").sha256(protected.read_bytes()).hexdigest()
    config = _config()
    config.update(
        output_dir=str(tmp_path / "output"),
        checkpoint={
            "path": str(protected),
            "sha256": digest,
            "optimizer_step": 3,
            "validation_sequence_cross_entropy": 2.0,
        },
        protected_stage_a_inputs=[{"path": str(protected), "sha256": digest}],
        training_history={"files": []},
        step0_checkpoint={"path": None, "sha256": None},
    )
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(config))

    class Train:
        def iter_token_ids(self):
            yield 0, "train-a", [2, 3, 4]
            yield 1, "train-b", [2, 4, 5]

    prepared = {
        "authorization": authorization,
        "checkpoint": {
            "optimizer_step": 3,
            "selected_validation_sequence_cross_entropy": 2.0,
            "model": {},
        },
        "validation": validation,
        "train": Train(),
        "panels": {"original_validation": panel, "independent_diagnostic": panel},
        "panel_diagnostics": {"panels_disjoint": True},
        "schema_projection": validate_sequence_schema_columns(
            rich_geometry_schema().names,
            source="definitive v2 Arrow schema",
        ),
        "protected_stage_a_inputs": {str(protected): digest},
    }
    monkeypatch.setattr(context, "prepare_diagnostic", lambda _config: prepared)
    monkeypatch.setattr(context, "_authorization", lambda _config: authorization)
    monkeypatch.setattr(context, "_protected_hashes", lambda _authorization: {"dataset": "unchanged"})
    monkeypatch.setattr(context, "SequenceOnlyRichDataset", lambda _authorization, split: Train())
    monkeypatch.setattr(context, "_load_model", lambda *_args, **_kwargs: _ContextModel().eval())
    monkeypatch.setattr(
        context,
        "_parse_history",
        lambda _config: {
            "metrics_record_count": 1,
            "rolling_sequence_loss": [],
            "validation_trajectory": [{"optimizer_step": 3, "sequence_cross_entropy": 2.0}],
            "best_selection": {"optimizer_step": 3, "sequence_cross_entropy": 2.0},
            "amp_overflow_count": 0,
            "final_amp_scale": None,
        },
    )
    monkeypatch.setattr(context, "_step0_payload", lambda *_args: (None, {"status": "unavailable"}))
    result = context.run_diagnostic(path)
    output = Path(config["output_dir"])
    protocol = json.loads((output / "protocol.json").read_text())
    heartbeat = json.loads((output / "heartbeat.json").read_text())
    assert result["status"] == protocol["status"] == heartbeat["status"] == "completed"
    assert protocol["authorizes_training"] is False
    assert protocol["authorizes_joint_training"] is False
    assert protected.read_bytes() == b"immutable"
