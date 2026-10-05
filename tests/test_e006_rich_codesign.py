from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.data.rich_geometry import (
    RICH_PAIR_FEATURE_DIM,
    RICH_RESIDUE_FEATURE_DIM,
    RichGeometryDataset,
    _resolve_protected_input,
    authorize_rich_geometry_dataset,
    collate_rich_geometry,
    deterministic_length_bucket_sample,
    invariant_rich_features,
    validate_rich_row,
)
from protein_distance_diffusion.evaluation.e006_geometry_source_audit import sha256_file
from protein_distance_diffusion.models.rich_codesign import E006LossWeights, E006RichGeometryCoDesign, e006_losses
from protein_distance_diffusion.training.rich_codesign_smoke import (
    _forward,
    _synthetic_row,
    plan_e006_smoke,
    run_e006_smoke,
    stratified_metrics,
)


def _metadata(root: Path) -> dict[str, str]:
    values = {
        "schema.json": {
            "schema_version": "e006_rich_geometry_sidecar_v2",
            "dense_pair_features_stored": False,
        },
        "vocabulary.json": {
            "version": "canonical_20_pad_mask_v1",
            "tokens": ["<PAD>", "<MASK>", *list("ACDEFGHIKLMNPQRSTVWY")],
        },
        "normalization.json": {"schema_version": "e006_rich_geometry_sidecar_v2"},
    }
    hashes = {}
    for name, payload in values.items():
        path = root / name
        path.write_text(json.dumps(payload))
        hashes[name] = sha256_file(path)
    return hashes


def _authorized_dataset(
    tmp_path: Path,
    *,
    duplicate: bool = False,
    mode: str = "full",
    authorization: bool = True,
) -> tuple[Path, dict[str, str]]:
    root = tmp_path / "rich"
    root.mkdir(parents=True)
    metadata = _metadata(root)
    shards = []
    counts = {}
    for split in ("train", "validation"):
        directory = root / split
        directory.mkdir()
        rows = [
            _synthetic_row("duplicate" if duplicate else f"{split}-{index}", length, split)
            for index, length in enumerate((7, 11))
        ]
        path = directory / "part-000000.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=1)
        relative = str(path.relative_to(root))
        shards.append({"path": relative, "dataset": split, "row_count": len(rows), "sha256": sha256_file(path)})
        counts[split] = len(rows)
    inventory = root / "shard_hashes.sha256"
    inventory.write_text("".join(f"{record['sha256']}  {record['path']}\n" for record in shards))
    protocol = {
        "status": "completed",
        "mode": mode,
        "schema_version": "e006_rich_geometry_sidecar_v2",
        "torsion_convention_version": "e006_backbone_torsion_sincos_v1",
        "authorizes_training": authorization,
        "authorizes_definitive_dataset": authorization,
        "unexplained_failure_count": 0,
        "protected_inputs_unchanged": True,
        "observed_phase1_inputs_unchanged": True,
        "input_hashes_before": {},
        "input_hashes_after": {},
        "eligible_split_counts": counts,
        "definitive_observed_counts": {"total": 4, "eligible": 4, "excluded": 0},
        "processed_samples": 4,
        "shards": shards,
    }
    protocol_path = root / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    metadata.update(
        {
            "protocol.json": sha256_file(protocol_path),
            "shard_hashes.sha256": sha256_file(inventory),
        }
    )
    return root, metadata


def _authorize(root: Path, hashes: dict[str, str]):
    return authorize_rich_geometry_dataset(
        root,
        expected_protocol_sha256=hashes["protocol.json"],
        expected_schema_sha256=hashes["schema.json"],
        expected_vocabulary_sha256=hashes["vocabulary.json"],
        expected_normalization_sha256=hashes["normalization.json"],
        expected_shard_inventory_sha256=hashes["shard_hashes.sha256"],
    )


def _small_model() -> E006RichGeometryCoDesign:
    return E006RichGeometryCoDesign(
        sequence_hidden_dim=24,
        sequence_layers=2,
        sequence_heads=4,
        sequence_feedforward_dim=48,
        sequence_dropout=0.0,
        max_length=16,
        rich_hidden_dim=24,
        fusion_layers=(0, 1),
        minimum_fusion_capacity_ratio=0.01,
        minimum_fusion_parameters=100,
        geometry_model={
            "base_channels": 4,
            "channel_multipliers": (1, 2),
            "residual_blocks_per_level": 1,
            "group_norm_groups": 2,
            "attention_heads": 1,
            "use_bottleneck_attention": False,
            "time_embedding_dim": 16,
            "length_embedding_dim": 16,
            "max_length": 16,
        },
    )


def test_authorization_accepts_full_and_rejects_pilot_or_hash_contradiction(tmp_path: Path) -> None:
    root, hashes = _authorized_dataset(tmp_path)
    authorization = _authorize(root, hashes)
    assert authorization.split_counts == {"train": 2, "validation": 2}
    with pytest.raises(ValueError, match="protocol SHA-256"):
        authorize_rich_geometry_dataset(
            root,
            expected_protocol_sha256="0" * 64,
            expected_schema_sha256=hashes["schema.json"],
            expected_vocabulary_sha256=hashes["vocabulary.json"],
            expected_normalization_sha256=hashes["normalization.json"],
            expected_shard_inventory_sha256=hashes["shard_hashes.sha256"],
        )
    pilot_root, pilot_hashes = _authorized_dataset(tmp_path / "pilot", mode="pilot", authorization=False)
    with pytest.raises(ValueError, match="authorization contradiction"):
        _authorize(pilot_root, pilot_hashes)


def test_protected_input_relocation_preserves_content_hash_gate(tmp_path: Path) -> None:
    recorded_root = tmp_path / "recorded"
    verification_root = tmp_path / "preserved"
    recorded = recorded_root / "samples" / "7ssn_D.npz"
    preserved = verification_root / "samples" / "7ssn_D.npz"
    recorded.parent.mkdir(parents=True)
    preserved.parent.mkdir(parents=True)
    recorded.write_bytes(b"case-colliding-active-payload")
    preserved.write_bytes(b"authoritative-payload")
    expected = hashlib.sha256(preserved.read_bytes()).hexdigest()
    resolved, method = _resolve_protected_input(
        str(recorded),
        expected,
        [{"recorded_root": str(recorded_root), "verification_root": str(verification_root)}],
    )
    assert resolved == preserved.resolve()
    assert method == "relocated_verification_path"

    preserved.write_bytes(b"not-authoritative")
    with pytest.raises(ValueError, match="protected input hash contradiction"):
        _resolve_protected_input(
            str(recorded),
            expected,
            [{"recorded_root": str(recorded_root), "verification_root": str(verification_root)}],
        )


def test_protected_input_relocation_rejects_invalid_roots_and_escape(tmp_path: Path) -> None:
    expected = hashlib.sha256(b"payload").hexdigest()
    with pytest.raises(ValueError, match="roots must be absolute"):
        _resolve_protected_input(
            str(tmp_path / "recorded" / "value"),
            expected,
            [{"recorded_root": "relative", "verification_root": str(tmp_path)}],
        )
    with pytest.raises(ValueError, match="relocation schema"):
        _resolve_protected_input(
            str(tmp_path / "recorded" / "value"),
            expected,
            [{"recorded_root": str(tmp_path), "verification_root": str(tmp_path), "extra": "invalid"}],
        )
    with pytest.raises(ValueError, match="duplicate protected input relocation root"):
        _resolve_protected_input(
            str(tmp_path / "recorded" / "value"),
            expected,
            [
                {"recorded_root": str(tmp_path / "recorded"), "verification_root": str(tmp_path)},
                {"recorded_root": str(tmp_path / "recorded"), "verification_root": str(tmp_path)},
            ],
        )


def test_authorization_reports_relocated_protected_inputs(tmp_path: Path) -> None:
    root, hashes = _authorized_dataset(tmp_path)
    recorded_root = tmp_path / "recorded"
    verification_root = tmp_path / "preserved"
    recorded = recorded_root / "sample.npz"
    preserved = verification_root / "sample.npz"
    recorded_root.mkdir()
    verification_root.mkdir()
    recorded.write_bytes(b"case-collision")
    preserved.write_bytes(b"authoritative")
    expected = hashlib.sha256(preserved.read_bytes()).hexdigest()
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["input_hashes_before"] = {str(recorded): expected}
    protocol["input_hashes_after"] = dict(protocol["input_hashes_before"])
    protocol_path.write_text(json.dumps(protocol))
    hashes["protocol.json"] = sha256_file(protocol_path)
    authorization = authorize_rich_geometry_dataset(
        root,
        expected_protocol_sha256=hashes["protocol.json"],
        expected_schema_sha256=hashes["schema.json"],
        expected_vocabulary_sha256=hashes["vocabulary.json"],
        expected_normalization_sha256=hashes["normalization.json"],
        expected_shard_inventory_sha256=hashes["shard_hashes.sha256"],
        protected_input_relocations=[
            {"recorded_root": str(recorded_root), "verification_root": str(verification_root)}
        ],
    )
    assert authorization.protected_input_resolution_counts == {
        "recorded_path": 0,
        "relocated_verification_path": 1,
    }
    assert authorization.relocated_protected_inputs == (str(recorded),)


def test_plan_only_authorizes_without_publishing_or_training(tmp_path: Path) -> None:
    root, hashes = _authorized_dataset(tmp_path)
    config = load_yaml("configs/e006_rich_geometry_codesign_smoke.yaml")
    config["dataset"].update(
        directory=str(root),
        protocol_sha256=hashes["protocol.json"],
        schema_sha256=hashes["schema.json"],
        vocabulary_sha256=hashes["vocabulary.json"],
        normalization_sha256=hashes["normalization.json"],
        shard_inventory_sha256=hashes["shard_hashes.sha256"],
    )
    config["smoke"].update(output_dir=str(tmp_path / "must-not-exist"), maximum_length=16)
    config["model"] = {
        "sequence_hidden_dim": 24,
        "sequence_layers": 2,
        "sequence_heads": 4,
        "sequence_feedforward_dim": 48,
        "sequence_dropout": 0.0,
        "max_length": 16,
        "rich_hidden_dim": 24,
        "fusion_layers": [0, 1],
        "minimum_fusion_capacity_ratio": 0.01,
        "minimum_fusion_parameters": 100,
        "geometry_model": {
            "base_channels": 4,
            "channel_multipliers": [1, 2],
            "residual_blocks_per_level": 1,
            "group_norm_groups": 2,
            "attention_heads": 1,
            "use_bottleneck_attention": False,
            "time_embedding_dim": 16,
            "length_embedding_dim": 16,
            "max_length": 16,
        },
    }
    plan = plan_e006_smoke(config)
    assert plan["dataset_protocol_sha256"] == hashes["protocol.json"]
    assert plan["dataset_metadata_sha256"]["schema_sha256"] == hashes["schema.json"]
    assert plan["training_performed"] is False
    assert plan["authorizes_training"] is False
    assert not (tmp_path / "must-not-exist").exists()


def test_authorization_rejects_duplicate_cross_split_identity_and_corrupt_shard(tmp_path: Path) -> None:
    duplicate_root, duplicate_hashes = _authorized_dataset(tmp_path / "duplicate", duplicate=True)
    with pytest.raises(ValueError, match="duplicate or cross-split"):
        _authorize(duplicate_root, duplicate_hashes)
    root, hashes = _authorized_dataset(tmp_path / "corrupt")
    path = root / "train" / "part-000000.parquet"
    path.write_bytes(path.read_bytes() + b"altered")
    with pytest.raises(ValueError, match="SHA-256 contradiction"):
        _authorize(root, hashes)


def test_lazy_dataset_join_and_length_bucket_selection_are_bounded(tmp_path: Path) -> None:
    root, hashes = _authorized_dataset(tmp_path)
    dataset = RichGeometryDataset(_authorize(root, hashes), split="train")
    assert len(dataset) == 2
    assert dataset[0]["split"] == "train"
    assert dataset.cached_row_count == 1
    selected = deterministic_length_bucket_sample(dataset, count=2, seed=9)
    assert selected == deterministic_length_bucket_sample(dataset, count=2, seed=9)
    with pytest.raises(ValueError, match="split ownership"):
        validate_rich_row({**dataset[0], "split": "validation"}, split="train")
    with pytest.raises(ValueError, match="only 1 selectable"):
        deterministic_length_bucket_sample(dataset, count=2, seed=9, maximum_length=7)


def test_invariant_features_survive_proper_rigid_transform() -> None:
    row = _synthetic_row("protein", 9)
    first = invariant_rich_features(row)
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    translation = np.asarray([12.0, -7.0, 3.0], dtype=np.float32)
    transformed = copy.deepcopy(row)
    for atom in ("n", "ca", "c", "o", "cb"):
        coordinates = np.asarray(row[f"{atom}_coordinates"], dtype=np.float32)
        transformed[f"{atom}_coordinates"] = (coordinates @ rotation.T + translation).tolist()
    second = invariant_rich_features(transformed)
    assert torch.allclose(first["residue_features"], second["residue_features"], atol=2e-5, rtol=0)
    assert torch.allclose(first["pair_features"], second["pair_features"], atol=2e-5, rtol=0)
    assert torch.equal(first["pair_feature_mask"], second["pair_feature_mask"])


def test_mixed_length_collation_preserves_all_masks_and_cb_provenance() -> None:
    batch = collate_rich_geometry([_synthetic_row("a", 7), _synthetic_row("b", 11)])
    assert batch["sequence_token_ids"].shape == (2, 16)
    assert batch["rich_residue_features"].shape == (2, 16, RICH_RESIDUE_FEATURE_DIM)
    assert batch["rich_pair_features"].shape == (2, 16, 16, RICH_PAIR_FEATURE_DIM)
    assert not batch["residue_mask"][0, 7:].any()
    assert not batch["pair_mask"][0, :, 7:, :].any()
    assert not batch["pair_feature_mask"][0, :, 7:, :].any()
    assert (batch["native_cb_mask"] | batch["pseudo_cb_mask"])[batch["residue_mask"]].all()
    with pytest.raises(MemoryError, match="pair feature budget"):
        invariant_rich_features(_synthetic_row("large", 20), maximum_pair_elements=399)
    with pytest.raises(MemoryError, match="padded pair feature budget"):
        collate_rich_geometry([_synthetic_row("padded", 9)], maximum_pair_elements=143)


def test_e006_fusion_is_material_multi_depth_and_parameter_partition_is_exact() -> None:
    model = _small_model()
    counts = model.parameter_counts()
    assert counts["geometry_to_sequence_fusion"] > 2304
    assert sum(value for key, value in counts.items() if key != "total_model") == counts["total_model"]
    with pytest.raises(ValueError, match="parameter floor"):
        E006RichGeometryCoDesign(
            sequence_hidden_dim=24,
            sequence_layers=2,
            sequence_heads=4,
            sequence_feedforward_dim=48,
            max_length=16,
            rich_hidden_dim=8,
            fusion_layers=(0,),
            minimum_fusion_parameters=1_000_000,
            geometry_model={
                "base_channels": 4,
                "channel_multipliers": (1, 2),
                "residual_blocks_per_level": 1,
                "group_norm_groups": 2,
                "attention_heads": 1,
                "use_bottleneck_attention": False,
                "time_embedding_dim": 16,
                "length_embedding_dim": 16,
                "max_length": 16,
            },
        )


def test_forward_backward_masks_and_sequence_only_geometry_isolation() -> None:
    torch.manual_seed(6)
    model = _small_model()
    batch = collate_rich_geometry([_synthetic_row("a", 7), _synthetic_row("b", 11)])
    config = {
        "seed": 6,
        "diffusion": {"steps": 20},
        "objective": {
            "mask_fraction": 0.3,
            "geometry_noise_std": 0.1,
            "conditioning_dropout_probability": 0.0,
            "sequence_weight": 1.0,
            "geometry_weight": 0.1,
            "consistency_weight": 0.01,
            "auxiliary_warmup_steps": 0,
        },
    }
    outputs, losses, _ = _forward(model, batch, config=config, step=0, mode="learned_geometry_gating")
    losses["total"].backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    assert not outputs["sequence_logits"][0, 7:].any()
    model.eval()
    with torch.no_grad():
        first, _, _ = _forward(model, batch, config=config, step=1, mode="sequence_only")
        changed = copy.deepcopy(batch)
        changed["rich_residue_features"] = torch.randn_like(batch["rich_residue_features"]) * 100
        changed["rich_pair_features"] = torch.randn_like(batch["rich_pair_features"]) * 100
        second, _, _ = _forward(model, changed, config=config, step=1, mode="sequence_only")
    assert torch.equal(first["sequence_logits"], second["sequence_logits"])
    assert not first["fusion_gates"].any()


def test_losses_report_unweighted_and_weighted_components() -> None:
    model = _small_model()
    batch = collate_rich_geometry([_synthetic_row("a", 8)])
    outputs, _, _ = _forward(
        model,
        batch,
        config={
            "seed": 1,
            "diffusion": {"steps": 10},
            "objective": {
                "mask_fraction": 0.5,
                "geometry_noise_std": 0.1,
                "conditioning_dropout_probability": 0.0,
                "sequence_weight": 1.0,
                "geometry_weight": 0.1,
                "consistency_weight": 0.01,
                "auxiliary_warmup_steps": 0,
            },
        },
        step=0,
        mode="learned_geometry_gating",
    )
    masked = torch.zeros_like(batch["residue_mask"])
    masked[:, 0] = True
    losses = e006_losses(
        outputs,
        sequence_targets=batch["sequence_token_ids"],
        masked_token_mask=masked,
        geometry_target=batch["distance_matrices"],
        weights=E006LossWeights(1.0, 0.1, 0.01),
    )
    assert set(losses) == {
        "sequence",
        "geometry",
        "consistency",
        "sequence_weighted",
        "geometry_weighted",
        "consistency_weighted",
        "total",
    }


def test_stratified_metrics_keeps_each_required_dimension() -> None:
    base = {
        "mask_fraction": 0.3,
        "geometry_corruption_level": 0.1,
        "diffusion_timestep": 9,
        "protein_length": 64,
        "experimental_method": "X-RAY",
        "pseudo_cb_coverage": 0.2,
        "frame_coverage": 1.0,
        "torsion_coverage": 0.9,
        "conditioning_mode": "learned_geometry_gating",
        "sequence_cross_entropy": 2.0,
        "perplexity": 7.0,
        "top1_accuracy": 0.2,
        "top3_accuracy": 0.4,
        "top5_accuracy": 0.5,
        "geometry_loss": 0.1,
        "consistency_loss": 0.2,
    }
    result = stratified_metrics([base, {**base, "conditioning_mode": "sequence_only"}])
    assert {row["dimension"] for row in result} == {
        "mask_fraction",
        "geometry_corruption_level",
        "diffusion_timestep",
        "protein_length",
        "experimental_method",
        "pseudo_cb_coverage",
        "frame_coverage",
        "torsion_coverage",
        "conditioning_mode",
    }


def _smoke_config(tmp_path: Path) -> Path:
    config = load_yaml("configs/e006_rich_geometry_codesign_smoke.yaml")
    config["smoke"].update(
        synthetic=True,
        synthetic_lengths=[8],
        output_dir=str(tmp_path / "smoke"),
        device="cpu",
        sample_count=1,
        steps=2,
        overfit_steps=2,
        maximum_length=8,
    )
    config["model"].update(
        sequence_hidden_dim=24,
        sequence_layers=2,
        sequence_heads=4,
        sequence_feedforward_dim=48,
        sequence_dropout=0.0,
        max_length=16,
        rich_hidden_dim=24,
        fusion_layers=[0, 1],
        minimum_fusion_capacity_ratio=0.01,
        minimum_fusion_parameters=100,
        geometry_model={
            "base_channels": 4,
            "channel_multipliers": [1, 2],
            "residual_blocks_per_level": 1,
            "group_norm_groups": 2,
            "attention_heads": 1,
            "use_bottleneck_attention": False,
            "time_embedding_dim": 16,
            "length_embedding_dim": 16,
            "max_length": 16,
        },
    )
    config["objective"]["auxiliary_warmup_steps"] = 0
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_synthetic_train_smoke_is_non_authorizing_and_reload_deterministic(tmp_path: Path) -> None:
    report = run_e006_smoke(_smoke_config(tmp_path), mode="train-smoke")
    assert report["status"] == "completed"
    assert report["authorizes_definitive_training"] is False
    assert report["checkpoint_reload_deterministic"] is True
    assert report["overfit_diagnostic"]["fixed_mask_noise_and_timestep"] is True
    assert report["overfit_diagnostic"]["total_reduced"] is True
    assert set(report["losses_by_step"][0]["gradient_norms"]) == {"sequence", "geometry", "fusion"}
    assert report["stratified_validation"]


def test_synthetic_loader_smoke_checks_both_splits_and_invariance(tmp_path: Path) -> None:
    path = _smoke_config(tmp_path)
    config = json.loads(path.read_text())
    config["smoke"].update(sample_count=2, synthetic_lengths=[7, 8])
    path.write_text(json.dumps(config))
    report = run_e006_smoke(path, mode="loader-smoke")
    assert report["loader_checks"]["both_splits_present"] is True
    assert report["loader_checks"]["rigid_transform_invariant"] is True


def test_failed_smoke_atomically_publishes_non_authorizing_protocol(tmp_path: Path) -> None:
    path = _smoke_config(tmp_path)
    config = json.loads(path.read_text())
    config["smoke"]["steps"] = 17
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="steps"):
        run_e006_smoke(path, mode="train-smoke")
    report = json.loads((tmp_path / "smoke" / "train-smoke" / "protocol.json").read_text())
    assert report["status"] == "failed"
    assert report["authorizes_definitive_training"] is False
    assert json.loads((tmp_path / "smoke" / "train-smoke" / "heartbeat.json").read_text())["status"] == "failed"


def test_definitive_phase1_hash_is_pinned_without_accessing_shards() -> None:
    config = load_yaml("configs/e006_rich_geometry_codesign_smoke.yaml")
    assert config["dataset"]["protocol_sha256"] == ("653d1a88c4366401345287d6c3eea2267c363945bac429374db2058015913b8b")
    assert (
        hashlib.sha256(Path(config["dataset"]["directory"], "protocol.json").read_bytes()).hexdigest()
        == config["dataset"]["protocol_sha256"]
    )


@pytest.mark.parametrize("index", [(0,), [0], slice(0, 1), True])
def test_rich_geometry_dataset_rejects_non_scalar_indices(index: object) -> None:
    dataset = object.__new__(RichGeometryDataset)
    dataset._ends = [1]
    with pytest.raises(TypeError, match="scalar integers"):
        dataset[index]  # type: ignore[index]


def test_rich_geometry_dataset_accepts_numpy_integer_index() -> None:
    dataset = object.__new__(RichGeometryDataset)
    dataset._ends = []
    with pytest.raises(IndexError):
        dataset[np.int64(0)]
