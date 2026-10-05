from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from protein_distance_diffusion.data.rich_geometry import _resolve_protected_input
from protein_distance_diffusion.training.e007_local_backbone_repair import (
    _phase3f_dataset_config,
    validate_pilot_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/e007_local_backbone_repair_pilot_phase3i2_final_v1.yaml"
EXPECTED_7SSN = "99b26c65e29601430dea6810feff24f7b4a6a8c8ed255de6621448e8e2fdafd2"
RELOCATIONS = [
    {
        "recorded_root": "/home/simostocco/proteinGen/data",
        "verification_root": "/home/simostocco/proteinGen.pre-relocation-20260922T064202Z/data",
    }
]


def test_production_final_config_uses_preserved_7ssn_bytes() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    assert config["dataset"]["protected_input_relocations"] == RELOCATIONS
    source = _phase3f_dataset_config(config)
    assert source["dataset"]["protected_input_relocations"] == RELOCATIONS
    recorded = "/home/simostocco/proteinGen/data/full/processed/samples/7ssn_D.npz"
    resolved, kind = _resolve_protected_input(recorded, EXPECTED_7SSN, RELOCATIONS)
    assert kind == "relocated_verification_path"
    assert resolved.stat().st_size == 9211
    assert hashlib.sha256(resolved.read_bytes()).hexdigest() == EXPECTED_7SSN


def test_physical_case_collided_root_does_not_resolve_authoritative_file() -> None:
    recorded = "/home/simostocco/proteinGen/data/full/processed/samples/7ssn_D.npz"
    bad_mapping = [
        {"recorded_root": RELOCATIONS[0]["recorded_root"], "verification_root": "/mnt/d/Simone/proteinGen/data"}
    ]
    with pytest.raises(ValueError, match="hash contradiction"):
        _resolve_protected_input(recorded, EXPECTED_7SSN, bad_mapping)


def test_missing_preserved_root_and_hash_contradiction_fail_closed(tmp_path: Path) -> None:
    recorded = "/home/simostocco/proteinGen/data/full/processed/samples/7ssn_D.npz"
    relocation = [
        {
            "recorded_root": "/home/simostocco/proteinGen/data",
            "verification_root": str(tmp_path / "missing-preserved-data"),
        }
    ]
    with pytest.raises(ValueError, match="hash contradiction"):
        _resolve_protected_input(recorded, EXPECTED_7SSN, relocation)
    target = tmp_path / "present"
    target_file = target / "full/processed/samples/7ssn_D.npz"
    target_file.parent.mkdir(parents=True)
    target_file.write_bytes(b"contradictory")
    relocation[0]["verification_root"] = str(target)
    with pytest.raises(ValueError, match="hash contradiction"):
        _resolve_protected_input(recorded, EXPECTED_7SSN, relocation)


def test_escaping_relocated_path_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "verification"
    root.mkdir()
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"authoritative")
    (root / "item.npz").symlink_to(outside)
    relocation = [{"recorded_root": "/recorded/data", "verification_root": str(root)}]
    digest = hashlib.sha256(outside.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="escapes its root"):
        _resolve_protected_input("/recorded/data/item.npz", digest, relocation)


def test_production_read_only_contract_reconstructs_full_protected_panels() -> None:
    result = validate_pilot_contract(CONFIG)
    inventory = result["protected_dataset_inventory"]
    assert inventory["total"] == 3736
    assert inventory["direct_resolutions"] == 3725
    assert inventory["relocated_resolutions"] == 11
    assert inventory["missing"] == inventory["contradictory"] == 0
    relocated = {row["identity"]: row["sha256"] for row in inventory["relocated_identities"]}
    assert relocated["/home/simostocco/proteinGen/data/full/processed/samples/7ssn_D.npz"] == EXPECTED_7SSN
    assert result["training_panel_updates"] == 500
    assert len(result["evaluation_panel_identities"]) == 5
    assert result["coefficient_lookup_timesteps_verified"] == list(range(500))
    assert result["matched_arm_identity_sha256"]
    for flag in (
        "staging_created",
        "checkpoint_loaded",
        "model_created",
        "optimizer_created",
        "cuda_initialized",
        "forward_pass",
        "backward_pass",
        "sampling_performed",
    ):
        assert result[flag] is False
    assert result["optimizer_updates"] == 0
    assert all(
        result[field] is False
        for field in (
            "authorizes_training",
            "authorizes_production_training",
            "authorizes_phase3j",
            "authorizes_joint_training",
            "authorizes_sequence_conditioning",
            "authorizes_downstream_generation",
        )
    )
