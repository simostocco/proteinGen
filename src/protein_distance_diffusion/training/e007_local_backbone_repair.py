"""E007 Phase-3I.2 bounded local-backbone objective/sampler experiment."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

VERSION = "e007_local_backbone_repair_v1"
ARMS = ("v_only", "v_plus_local")
LEGACY_ARMS = ("matched_v_only", "matched_v_plus_local")
SAMPLERS = ("native_reverse", "local_guided_reverse")
LOCAL_TERMS = ("adjacent", "i_plus_2", "i_plus_3", "bond_angle_cosine", "discontinuity", "clash")
GRADIENT_AUDIT_UPDATES = (0, *range(1, 11), 25, 50, 100, 250, 500)
DECISION_CATEGORIES = {
    "objective_correction_supported",
    "sampler_correction_supported",
    "combined_correction_supported",
    "local_improvement_with_global_degradation",
    "length_384_500_failure_persists",
    "no_material_local_geometry_improvement",
    "inconclusive_bounded_pilot",
}
NON_AUTHORIZING = {
    "authorizes_training": False,
    "authorizes_production_training": False,
    "authorizes_joint_training": False,
    "authorizes_phase3j": False,
    "authorizes_sequence_conditioning": False,
    "authorizes_downstream_generation": False,
}


class CudaMemoryTelemetry:
    """Per-device allocator observations; resets delimit phases, never erase run peaks."""

    def __init__(self, cuda: Any, device: Any) -> None:
        self.cuda = cuda
        self.device = device
        self.run_allocated = 0.0
        self.run_reserved = 0.0
        self.total = float(cuda.get_device_properties(device).total_memory) / 2**20
        self.snapshot()  # Validate existing allocator state before the first reset.
        cuda.reset_peak_memory_stats(device)

    def snapshot(self) -> dict[str, float]:
        cuda, device = self.cuda, self.device
        cuda.synchronize(device)
        stats = cuda.memory_stats(device)
        active_bytes = int(stats["active_bytes.all.current"])
        values = {
            "current_cuda_allocated_mib": float(cuda.memory_allocated(device)) / 2**20,
            "current_cuda_reserved_mib": float(cuda.memory_reserved(device)) / 2**20,
            "current_cuda_active_bytes": active_bytes,
            "current_cuda_active_mib": active_bytes / 2**20,
            "phase_peak_cuda_allocated_mib": float(cuda.max_memory_allocated(device)) / 2**20,
            "phase_peak_cuda_reserved_mib": float(cuda.max_memory_reserved(device)) / 2**20,
            "device_total_cuda_mib": self.total,
            "device_total_cuda_bytes": int(self.total * 2**20),
        }
        if any(not math.isfinite(value) or value < 0 for value in values.values()) or self.total <= 0:
            raise MemoryError(f"invalid CUDA memory observation: {values}")
        allocated, reserved, active = (values[f"current_cuda_{key}_mib"] for key in ("allocated", "reserved", "active"))
        peak_allocated = values["phase_peak_cuda_allocated_mib"]
        peak_reserved = values["phase_peak_cuda_reserved_mib"]
        if (
            not (allocated <= reserved <= peak_reserved <= self.total)
            or not (active <= reserved)
            or not (allocated <= peak_allocated <= peak_reserved)
        ):
            raise MemoryError(f"physically inconsistent CUDA memory observation: {values}")
        self.run_allocated = max(self.run_allocated, peak_allocated)
        self.run_reserved = max(self.run_reserved, peak_reserved)
        return {
            **values,
            "run_peak_cuda_allocated_mib": self.run_allocated,
            "run_peak_cuda_reserved_mib": self.run_reserved,
        }

    def end_phase(self) -> dict[str, float]:
        """Capture transient peaks before resetting the next phase's allocator counters."""
        observed = self.snapshot()
        self.cuda.reset_peak_memory_stats(self.device)
        return observed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _phase3f_dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    """Load Phase 3F source config with the final pilot's validated relocation policy."""
    relocations = config.get("dataset", {}).get("protected_input_relocations")
    from protein_distance_diffusion.data.rich_geometry import validate_protected_input_relocations

    validate_protected_input_relocations(relocations)
    source = yaml.safe_load(Path("configs/e007_coordinate_real_pilot_v1.yaml").read_text())
    source["dataset"]["protected_input_relocations"] = relocations
    return source


def _resolve_reviewed_coordinate_source(config: dict[str, Any], *, load_checkpoint: bool = False) -> dict[str, Any]:
    """Resolve the hash-pinned Phase 3F architecture, step-9000 weights and relocation policy."""
    from protein_distance_diffusion.evaluation import e007_denoiser_sampler_localization as localization

    audit_source = localization.load_config(config["dataset_source_config"])
    model_source = audit_source.get("model_source", {})
    source_path = Path(model_source.get("phase3f_config_path", ""))
    expected_source_hash = model_source.get("phase3f_config_sha256")
    if not source_path.is_file() or not expected_source_hash:
        raise ValueError("reviewed coordinate source config pin is missing")
    if _sha256_file(source_path) != expected_source_hash:
        raise ValueError("reviewed coordinate source config hash mismatch")
    source = yaml.safe_load(source_path.read_text())
    required_source = ("model", "optimizer", "numerics", "dataset")
    missing = [key for key in required_source if not isinstance(source, dict) or key not in source]
    if missing or not isinstance(source.get("model"), dict) or not source["model"]:
        raise ValueError(f"reviewed coordinate source config missing required keys: {missing or ['model fields']}")
    if not all(key in source["optimizer"] for key in ("learning_rate", "weight_decay", "betas")):
        raise ValueError("reviewed coordinate source config has incomplete optimizer contract")
    production = load_config("configs/e007_local_backbone_repair_pilot_phase3i2_final_v1.yaml")
    if production["dataset"].get("protected_input_relocations") != config["dataset"].get("protected_input_relocations"):
        raise ValueError("final pilot relocation mapping changed")
    source = _phase3f_dataset_config(production)
    source["model"] = yaml.safe_load(source_path.read_text())["model"]
    checkpoint_spec = config["selected_checkpoint"]
    checkpoint_path = Path(checkpoint_spec["path"])
    if _sha256_file(checkpoint_path) != checkpoint_spec["sha256"]:
        raise ValueError("step-9000 checkpoint hash mismatch")
    result = {
        "source": source,
        "audit_source": audit_source,
        "source_config_path": source_path,
        "checkpoint_path": checkpoint_path,
    }
    if load_checkpoint:
        import torch

        result["checkpoint"] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return result


COEFFICIENT_TABLE_SHA256 = "6672eea18135f35545526d2c7d18e437ae2caf9355f785360bfdc1ee93e304fe"
V3_PINS = {
    "configuration_sha256": "381a3c862c30d32a6a8a5a235d1f49804145aa420d8f6a72b13b060a6f9866b7",
    "report_sha256": "c51a9c694616378c857828968ac358e057040f197f3285db0c9d9410eda81fb6",
    "protocol_sha256": "0c95805fcaf4ca4187b592a1015f33270cd535db6520af562dad05577094a50c",
    "plan_sha256": "f1e1ce6ac28974baaabf249814476264e8dceb9735ddcec21781cc0d8258dbe4",
}
COEFFICIENT_ANCHORS = (25, 250, 425, 499)
PREFLIGHT_TIMESTEPS = (0, 12, 25, 75, 150, 225, 250, 300, 375, 425, 450, 475, 499)
PREFLIGHT_V2_TIMESTEPS = (
    0,
    6,
    12,
    18,
    25,
    50,
    75,
    112,
    150,
    187,
    225,
    237,
    250,
    275,
    300,
    337,
    375,
    400,
    425,
    437,
    450,
    462,
    475,
    487,
    499,
)
PREFLIGHT_V1_HASHES = {
    "configuration_sha256": "5f2c9be4048e25d50db6a5f09eef2e262ebce1b9b90a5c45fb6dcb3abe9b8d15",
    "report_sha256": "5863aaec69815a979885116b43117929311e7c1a872018a6f914d3a6b00bb238",
    "protocol_sha256": "dccefdd84c30e619886b040f3db3605ed0c0127018dc27ef3d4c47eeb9c78ec1",
    "cells_sha256": "47fca0c920efce27954569d92b18dc01a5a9459156143825565940c0a13e4031",
}


def load_coefficient_table(path: str | Path, *, expected_sha256: str = COEFFICIENT_TABLE_SHA256) -> dict[str, Any]:
    """Load and reconstruct the immutable 500-row positive coefficient schedule."""

    def no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate coefficient-table key: {key}")
            result[key] = value
        return result

    raw = json.loads(Path(path).read_text(), object_pairs_hook=no_duplicate_pairs)
    if raw.get("version") == "e007_phase3i2_budget_medium_coefficient_table_v3":
        return _load_coefficient_table_v3(raw, path, expected_sha256)
    expected_keys = {
        "version",
        "policy",
        "selected_tier",
        "timesteps",
        "coefficient_table_sha256",
        "protected_v3_hashes",
    }
    if raw.get("version") == "e007_phase3i2_budget_medium_coefficient_table_v2":
        expected_keys.add("safety_derivation")
    if set(raw) != expected_keys:
        raise ValueError("coefficient table schema mismatch")
    if raw["version"] == "e007_phase3i2_budget_medium_coefficient_table_v2":
        return _load_coefficient_table_v2(
            raw, raw["coefficient_table_sha256"] if expected_sha256 == COEFFICIENT_TABLE_SHA256 else expected_sha256
        )
    if raw["version"] != "e007_phase3i2_budget_medium_coefficient_table_v1":
        raise ValueError("coefficient table version mismatch")
    if raw["policy"] != "log_linear_positive_between_25_250_425_499; clamp_below_25; exact_499":
        raise ValueError("coefficient table policy/version mismatch")
    if raw["selected_tier"] != "budget_medium":
        raise ValueError("coefficient table tier mismatch")
    if raw["protected_v3_hashes"] != V3_PINS:
        raise ValueError("coefficient-table protected v3 hashes mismatch")
    verify_v3_evidence()
    content = {key: raw[key] for key in ("policy", "selected_tier", "timesteps")}
    observed_sha = _canonical_sha(content)
    if raw["coefficient_table_sha256"] != observed_sha:
        raise ValueError("coefficient table embedded hash mismatch")
    if observed_sha != expected_sha256:
        raise ValueError("coefficient table hash mismatch")
    rows = raw["timesteps"]
    if list(rows) != [str(index) for index in range(500)]:
        raise ValueError("coefficient table must contain ordered unique timesteps 0..499")
    names = set(LOCAL_TERMS)
    for timestep, row in rows.items():
        if set(row) != {"v_mse", "local_terms"} or float(row["v_mse"]) != 1.0:
            raise ValueError(f"coefficient row {timestep} schema/v-MSE mismatch")
        if set(row["local_terms"]) != names:
            raise ValueError(f"coefficient row {timestep} term keys mismatch")
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in row["local_terms"].values()):
            raise ValueError(f"coefficient row {timestep} has invalid coefficient")
    anchors = {}
    calibration_report = json.loads(
        Path(
            "reports/experiments/E007_matrix_sequence_cogeneration/local_backbone_repair_calibration_v3/report.json"
        ).read_text()
    )
    for timestep in COEFFICIENT_ANCHORS:
        anchors[timestep] = {name: float(rows[str(timestep)]["local_terms"][name]) for name in LOCAL_TERMS}
        report_cell = next(
            cell for cell in calibration_report["cells"] if int(cell["identity"]["timestep"]) == timestep
        )
        for name in LOCAL_TERMS:
            calibrated = float(report_cell["candidate_coefficients"]["budget_medium"]["terms"][name]["coefficient"])
            if not math.isclose(anchors[timestep][name], calibrated, rel_tol=1e-12, abs_tol=0.0):
                raise ValueError(f"coefficient table anchor differs from protected v3 at {timestep}/{name}")
    for timestep in range(500):
        if timestep < 25:
            expected = anchors[25]
        elif timestep == 499:
            expected = anchors[499]
        else:
            left = max(anchor for anchor in COEFFICIENT_ANCHORS if anchor <= timestep)
            right = min(anchor for anchor in COEFFICIENT_ANCHORS if anchor >= timestep)
            if left == right:
                expected = anchors[left]
            else:
                fraction = (timestep - left) / (right - left)
                expected = {
                    name: math.exp(
                        math.log(anchors[left][name]) * (1 - fraction) + math.log(anchors[right][name]) * fraction
                    )
                    for name in LOCAL_TERMS
                }
        for name in LOCAL_TERMS:
            actual = float(rows[str(timestep)]["local_terms"][name])
            if not math.isclose(actual, expected[name], rel_tol=1e-12, abs_tol=0.0):
                raise ValueError(f"coefficient interpolation reconstruction mismatch at {timestep}/{name}")
    return {**content, "sha256": observed_sha, "anchors": anchors, "protected_v3_hashes": raw["protected_v3_hashes"]}


def _load_coefficient_table_v2(raw: Mapping[str, Any], expected_sha256: str) -> dict[str, Any]:
    required = {
        "version",
        "policy",
        "selected_tier",
        "timesteps",
        "coefficient_table_sha256",
        "protected_v3_hashes",
        "safety_derivation",
    }
    if set(raw) != required or raw["selected_tier"] != "budget_medium":
        raise ValueError("v2 coefficient table schema/tier mismatch")
    if raw["protected_v3_hashes"] != V3_PINS:
        raise ValueError("v2 coefficient table protected calibration hashes mismatch")
    derivation = raw["safety_derivation"]
    if derivation.get("preflight_v1_hashes") != PREFLIGHT_V1_HASHES:
        raise ValueError("v2 coefficient table preflight-v1 evidence pins mismatch")
    v1_dir = Path("reports/experiments/E007_matrix_sequence_cogeneration/local_backbone_repair_dense_preflight_v1")
    v1_paths = {
        "configuration_sha256": Path("configs/e007_local_backbone_repair_dense_preflight_v1.yaml"),
        "report_sha256": v1_dir / "report.json",
        "protocol_sha256": v1_dir / "protocol.json",
        "cells_sha256": v1_dir / "cells.jsonl",
    }
    if any(_sha256_file(path) != PREFLIGHT_V1_HASHES[key] for key, path in v1_paths.items()):
        raise ValueError("protected preflight-v1 evidence changed")
    cells_path = v1_dir / "cells.jsonl"
    cells = [json.loads(line) for line in cells_path.read_text().splitlines() if line.strip()]
    maxima = {
        int(t): max(float(c["combined_auxiliary_v_ratio"]) for c in cells if int(c["identity"]["timestep"]) == int(t))
        for t in derivation["safety_anchors"]
    }
    anchors = derivation["anchors"]
    if [int(a["timestep"]) for a in anchors] != list(map(int, derivation["safety_anchors"])):
        raise ValueError("v2 safety-anchor rows mismatch")
    base = load_coefficient_table("configs/e007_local_backbone_repair_budget_medium_coefficients_v1.json")
    safety_values = {}
    for anchor in anchors:
        t = int(anchor["timestep"])
        observed = maxima[t]
        factor = min(1.0, 0.12 / observed)
        if not math.isclose(float(anchor["observed_max_combined_ratio"]), observed, rel_tol=0, abs_tol=1e-15):
            raise ValueError(f"v2 observed maximum mismatch at {t}")
        if not math.isclose(float(anchor["safety_factor"]), factor, rel_tol=1e-14, abs_tol=0):
            raise ValueError(f"v2 safety factor mismatch at {t}")
        safety_values[t] = factor
    rows = raw["timesteps"]
    if list(rows) != [str(i) for i in range(500)]:
        raise ValueError("v2 coefficient schedule must contain ordered rows 0..499")
    for t in range(500):
        left = max(a for a in safety_values if a <= t)
        right = min(a for a in safety_values if a >= t)
        if left == right:
            factor = safety_values[left]
        else:
            q = (t - left) / (right - left)
            factor = math.exp(math.log(safety_values[left]) * (1 - q) + math.log(safety_values[right]) * q)
        row = rows[str(t)]
        if float(row["v_mse"]) != 1.0 or set(row["local_terms"]) != set(LOCAL_TERMS):
            raise ValueError(f"v2 coefficient row schema mismatch at {t}")
        for term in LOCAL_TERMS:
            expected = float(base["timesteps"][str(t)]["local_terms"][term]) * factor
            if not math.isclose(float(row["local_terms"][term]), expected, rel_tol=1e-12, abs_tol=0):
                raise ValueError(f"v2 coefficient reconstruction mismatch at {t}/{term}")
    content = {key: raw[key] for key in ("policy", "selected_tier", "timesteps")}
    digest = _canonical_sha({**content, "safety_derivation": derivation})
    if digest != raw["coefficient_table_sha256"] or digest != expected_sha256:
        raise ValueError("v2 coefficient table hash mismatch")
    return {
        **content,
        "sha256": digest,
        "anchors": {int(a["timestep"]): a for a in anchors},
        "safety_derivation": derivation,
        "protected_v3_hashes": raw["protected_v3_hashes"],
    }


def _load_coefficient_table_v3(raw: Mapping[str, Any], path: str | Path, expected_sha256: str) -> dict[str, Any]:
    """Validate the high-noise schedule as a mechanical positive rescaling of v2."""
    derivation = raw.get("derivation")
    policy = raw.get("policy")
    if raw.get("selected_tier") != "budget_medium" or not isinstance(derivation, Mapping):
        raise ValueError("v3 coefficient table schema/tier mismatch")
    if policy != {
        "anchors": {"0": 1.0, "450": 1.0, "475": 0.8, "487": 0.6, "499": 0.375},
        "interpolation": "positive_log_linear_between_anchors",
        "scope": "all_six_local_terms; v_mse_unchanged",
    }:
        raise ValueError("v3 multiplier policy mismatch")
    parent_path = Path("configs/e007_local_backbone_repair_budget_medium_coefficients_v2.json")
    parent_file_sha = _sha256_file(parent_path)
    parent = load_coefficient_table(
        parent_path,
        expected_sha256="d6bd01692d114c385143635a21a8db0475128d41c91f329f641c7611d3f83bd0",
    )
    parent_canonical = parent["sha256"]
    policy_sha = _canonical_sha(policy)
    if derivation != {
        "parent_version": "e007_phase3i2_budget_medium_coefficient_table_v2",
        "parent_canonical_sha256": parent_canonical,
        "parent_file_sha256": parent_file_sha,
        "multiplier_policy_canonical_sha256": policy_sha,
    }:
        raise ValueError("v3 parent/policy derivation hash mismatch")
    rows = raw.get("timesteps", {})
    if list(rows) != [str(i) for i in range(500)]:
        raise ValueError("v3 coefficient schedule must contain ordered rows 0..499")
    anchors = {0: 1.0, 450: 1.0, 475: 0.8, 487: 0.6, 499: 0.375}

    def multiplier(t: int) -> float:
        left = max(a for a in anchors if a <= t)
        right = min(a for a in anchors if a >= t)
        if left == right:
            return anchors[left]
        q = (t - left) / (right - left)
        return math.exp(math.log(anchors[left]) * (1 - q) + math.log(anchors[right]) * q)

    for t in range(500):
        row = rows[str(t)]
        if set(row) != {"v_mse", "local_terms", "multiplier"} or float(row["v_mse"]) != 1.0:
            raise ValueError(f"v3 coefficient row schema mismatch at {t}")
        factor = multiplier(t)
        if not math.isclose(float(row["multiplier"]), factor, rel_tol=1e-14, abs_tol=0):
            raise ValueError(f"v3 multiplier interpolation mismatch at {t}")
        if set(row["local_terms"]) != set(LOCAL_TERMS):
            raise ValueError(f"v3 coefficient term schema mismatch at {t}")
        for term in LOCAL_TERMS:
            value = float(row["local_terms"][term])
            expected = float(parent["timesteps"][str(t)]["local_terms"][term]) * factor
            if not math.isfinite(value) or value <= 0 or not math.isclose(value, expected, rel_tol=1e-14, abs_tol=0):
                raise ValueError(f"v3 coefficient reconstruction/positivity mismatch at {t}/{term}")
    content = {
        "version": raw["version"],
        "policy": policy,
        "derivation": derivation,
        "selected_tier": raw["selected_tier"],
        "timesteps": rows,
    }
    digest = _canonical_sha(content)
    if raw.get("coefficient_table_sha256") != digest or expected_sha256 not in (COEFFICIENT_TABLE_SHA256, digest):
        raise ValueError("v3 coefficient table canonical hash mismatch")
    return {
        **content,
        "sha256": digest,
        "anchors": anchors,
        "multiplier_policy_sha256": policy_sha,
        "parent_canonical_sha256": parent_canonical,
        "parent_file_sha256": parent_file_sha,
        "file_sha256": _sha256_file(Path(path)),
    }


def coefficient_tensor(table: Mapping[str, Any], timesteps: Any, device: Any) -> Any:
    """Gather a coefficient row for each sample; only this compact tensor moves to device."""
    import torch

    cpu_times = timesteps.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    if bool(((cpu_times < 0) | (cpu_times >= 500)).any()):
        raise ValueError("timestep outside declared coefficient schedule")
    cpu_rows = torch.tensor(
        [
            [float(table["timesteps"][str(int(time))]["local_terms"][name]) for name in LOCAL_TERMS]
            for time in cpu_times
        ],
        dtype=torch.float64,
    )
    return cpu_rows.to(device=device)


def _pilot_training_objective(
    arm: str,
    prediction: Any,
    corruption: Mapping[str, Any],
    prepared: Mapping[str, Any],
    diffusion: Any,
    settings: Mapping[str, Any],
    table: Mapping[str, Any],
    device: Any,
) -> tuple[Any, Any, Mapping[str, Any], Mapping[str, Any], Any, Any]:
    """Keep the control arm on coordinate-v MSE without evaluating local terms."""
    from protein_distance_diffusion.training import e007_coordinate_real_pilot as phase3f

    v_loss = phase3f.uniform_coordinate_v_mse(prediction, corruption["batch"].coordinate_v_target, corruption["mask"])
    if arm == "v_only":
        return v_loss, v_loss, {}, {}, None, None
    if arm != "v_plus_local":
        raise ValueError(f"unknown pilot training arm: {arm}")
    predicted_x0 = diffusion.reconstruct_x0(
        corruption["batch"].noisy_coordinates,
        corruption["timesteps"],
        prediction,
        corruption["mask"],
    )
    raw = local_backbone_losses(
        predicted_x0,
        prepared["coordinates"].to(device),
        corruption["mask"],
        prepared["chain_continuity_mask"].to(device),
        settings,
        per_structure=True,
    )
    coefficient_rows = coefficient_tensor(table, corruption["timesteps"], device)
    weights = {name: coefficient_rows[:, index].to(dtype=prediction.dtype) for index, name in enumerate(LOCAL_TERMS)}
    auxiliary, weighted = weighted_local_objective(raw, weights)
    return v_loss + auxiliary, v_loss, raw, weighted, coefficient_rows, auxiliary


def _initialize_pilot_arm(
    model_class: Any,
    model_config: Mapping[str, Any],
    checkpoint_model: Mapping[str, Any],
    optimizer_config: Mapping[str, Any],
    device: Any,
) -> tuple[Any, Any, Any]:
    """Construct an independent arm from immutable weights and fresh training state."""
    import torch

    model = model_class(**model_config).to(device)
    model.load_state_dict(checkpoint_model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        betas=tuple(optimizer_config["betas"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return model, optimizer, scheduler


def _gradient_vector(loss: Any, parameters: Sequence[Any], *, retain_graph: bool = False) -> Any:
    import torch

    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, create_graph=False, allow_unused=True)
    vector = torch.cat([gradient.reshape(-1) for gradient in gradients if gradient is not None])
    del gradients
    return vector


def _finite_counts(vector: Any) -> dict[str, Any]:
    import torch

    finite = int(torch.isfinite(vector).sum().item())
    total = int(vector.numel())
    return {
        "finite_count": finite,
        "total_count": total,
        "non_finite_count": total - finite,
        "all_finite": finite == total,
    }


def _upper_safety_gate_failures(cell: Mapping[str, Any]) -> list[str]:
    """Return unconditional upper-safety failures, including for timestep zero."""
    ratios = cell["individual_term_v_ratios"]
    failures = []
    if any(value is None or value > 0.20 for value in ratios.values()):
        failures.append("individual_ratio")
    if cell["combined_auxiliary_v_ratio"] is None or cell["combined_auxiliary_v_ratio"] > 0.20:
        failures.append("combined_ratio")
    if cell["total_v_ratio"] is None or not 0.80 <= cell["total_v_ratio"] <= 1.30:
        failures.append("total_ratio")
    return failures


def _lower_activity_timesteps(timesteps: Sequence[int], exempt_timesteps: Sequence[int]) -> tuple[int, ...]:
    exemptions = set(map(int, exempt_timesteps))
    return tuple(int(t) for t in timesteps if int(t) not in exemptions)


def _json_finite(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _fingerprint(value: Any) -> str:
    """Stable-enough in-process state fingerprint for audit mutation detection."""
    import torch

    def normalize(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            return (
                "tensor",
                str(tensor.dtype),
                tuple(tensor.shape),
                hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
            )
        if isinstance(item, Mapping):
            return tuple((str(key), normalize(item[key])) for key in sorted(item, key=str))
        if isinstance(item, (tuple, list)):
            return tuple(normalize(child) for child in item)
        if isinstance(item, (str, int, float, bool, type(None))):
            return item
        return repr(item)

    return hashlib.sha256(pickle.dumps(normalize(value), protocol=5)).hexdigest()


def _zero_update_drift_cell(
    model: Any,
    diffusion: Any,
    row: Mapping[str, Any],
    length: int,
    timestep_value: int,
    update: int,
    update_identity: Mapping[str, Any] | None,
    config: Mapping[str, Any],
    table: Mapping[str, Any],
    device: Any,
    settings: Mapping[str, Any],
    downsample_factor: int,
) -> dict[str, Any]:
    """Evaluate one cell with no tensor references shared with the next cell."""
    import gc

    import torch

    from protein_distance_diffusion.training.e007_coordinate_real_loader_smoke import prepare_coordinate_batch
    from protein_distance_diffusion.training.e007_coordinate_real_pilot import uniform_coordinate_v_mse

    prepared = clean = mask = continuity = timestep = generator = batch = None
    prediction = v_loss = predicted_x0 = raw = coeff = weighted = parameters = None
    v_vec = aux_vec = total_vec = term_vec = None
    try:
        prepared = prepare_coordinate_batch([row], float(config["coordinate_scale_angstrom"]), downsample_factor)
        clean = prepared["coordinates"].to(device)
        mask = prepared["residue_mask"].to(device)
        continuity = prepared["chain_continuity_mask"].to(device)
        timestep = torch.tensor([timestep_value], dtype=torch.long, device=device)
        generator = torch.Generator(device=device).manual_seed(8431000 + update * 1000 + length * 10 + timestep_value)
        batch = diffusion.make_training_batch(clean, mask, timesteps=timestep, generator=generator)
        prediction = model(batch.noisy_coordinates, timestep, prepared["lengths"].to(device), mask, continuity)[
            "v_prediction"
        ]
        v_loss = uniform_coordinate_v_mse(prediction, batch.coordinate_v_target, mask)
        predicted_x0 = diffusion.reconstruct_x0(batch.noisy_coordinates, timestep, prediction, mask)
        raw = local_backbone_losses(predicted_x0, clean, mask, continuity, settings, per_structure=True)
        coeff = coefficient_tensor(table, timestep, device)[0].to(prediction.dtype)
        weighted = {name: raw[name][0] * coeff[index] for index, name in enumerate(LOCAL_TERMS)}
        parameters = tuple(model.parameters())
        v_vec = _gradient_vector(v_loss, parameters, retain_graph=True)
        v_norm = float(torch.linalg.vector_norm(v_vec).detach().cpu())
        finite = {"v": _finite_counts(v_vec)}
        ratios = {}
        for index, name in enumerate(LOCAL_TERMS):
            term_vec = _gradient_vector(weighted[name], parameters, retain_graph=index < len(LOCAL_TERMS) - 1)
            ratios[name] = float(torch.linalg.vector_norm(term_vec).detach().cpu()) / max(v_norm, 1e-30)
            finite[name] = _finite_counts(term_vec)
            aux_vec = term_vec if aux_vec is None else aux_vec + term_vec
            term_vec = None
        assert aux_vec is not None
        total_vec = v_vec + aux_vec
        aux_ratio = float(torch.linalg.vector_norm(aux_vec).detach().cpu()) / max(v_norm, 1e-30)
        total_ratio = float(torch.linalg.vector_norm(total_vec).detach().cpu()) / max(v_norm, 1e-30)
        total_v_cosine = float(
            (
                torch.dot(total_vec, v_vec)
                / (torch.linalg.vector_norm(total_vec) * torch.linalg.vector_norm(v_vec)).clamp_min(1e-30)
            )
            .detach()
            .cpu()
        )
        finite["auxiliary"] = _finite_counts(aux_vec)
        finite["total"] = _finite_counts(total_vec)
        record = {
            "update": update,
            "update_identity": json.loads(json.dumps(_json_finite(dict(update_identity or {})), allow_nan=False)),
            "length": length,
            "timestep": timestep_value,
            "sample_id": str(row["sample_id"]),
            "coefficient_row": {name: float(coeff[index].detach().cpu()) for index, name in enumerate(LOCAL_TERMS)},
            "finite_counts": finite,
            "individual_ratios": ratios,
            "combined_auxiliary_ratio": aux_ratio,
            "total_v_ratio": total_ratio,
            "total_v_cosine": total_v_cosine,
            "loss_values": {
                "v": float(v_loss.detach().cpu()),
                "raw": {name: float(raw[name][0].detach().cpu()) for name in LOCAL_TERMS},
                "weighted": {name: float(weighted[name].detach().cpu()) for name in LOCAL_TERMS},
            },
        }
        if aux_ratio > 0.20:
            record["warning"] = "combined_auxiliary_v_ratio_exceeds_0.20"
        return record
    finally:
        del prepared, clean, mask, continuity, timestep, generator, batch
        del prediction, v_loss, predicted_x0, raw, coeff, weighted, parameters
        del v_vec, aux_vec, total_vec, term_vec
        gc.collect()
        if getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()


def zero_update_drift_audit(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    diffusion: Any,
    panel: Sequence[tuple[int, Mapping[str, Any]]],
    config: Mapping[str, Any],
    table: Mapping[str, Any],
    device: Any,
    *,
    update: int,
    data_cursor: int,
    audit_timesteps: Sequence[int] | None = None,
    update_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fixed audit using autograd.grad only; preserves and verifies all training state."""
    from protein_distance_diffusion.evaluation.e007_denoiser_sampler_localization import (
        load_config as load_localization_config,
    )

    protected_before = verify_protected_evidence(config)
    table_before = table["sha256"]
    model_before = _fingerprint(model.state_dict())
    optimizer_before = _fingerprint(optimizer.state_dict())
    scheduler_before = _fingerprint(scheduler.state_dict())
    rng_state_before = _rng_state()
    rng_before = _fingerprint(rng_state_before)
    cpu_rng_before = _fingerprint(
        {"python": rng_state_before["python"], "numpy": rng_state_before["numpy"], "torch": rng_state_before["torch"]}
    )
    cuda_rng_before = _fingerprint(rng_state_before["cuda"])
    mode_before = model.training
    cursor_before = data_cursor
    cursor_hash_before = _fingerprint(data_cursor)
    # Training gradients are not part of the audit input. Clear optimizer.step() leftovers
    # before taking the restoration snapshot so every cell starts with no parameter grads.
    optimizer.zero_grad(set_to_none=True)
    grads_before = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
    settings = dict(config["local_objective"])
    settings["coordinate_scale_angstrom"] = float(config["coordinate_scale_angstrom"])
    downsample_factor = int(load_localization_config(config["dataset_source_config"])["expected_downsample_factor"])
    records, violations, warnings = [], [], []
    timesteps = tuple(map(int, audit_timesteps or COEFFICIENT_ANCHORS))
    dynamic_gates = config.get("dynamic_preflight", {})
    descriptive_combined = (
        dynamic_gates.get("version") == "e007_phase3i2_dynamic_stability_preflight_v6"
        or config.get("pilot_audit_semantics") == "reviewed_v6"
    )
    individual_limit = 0.20 if dynamic_gates or descriptive_combined else 0.25
    combined_limit = float(dynamic_gates.get("maximum_combined_auxiliary_v_ratio", 0.20)) if dynamic_gates else 0.25
    warning_limit = (
        float(dynamic_gates.get("combined_auxiliary_warning_threshold", combined_limit)) if dynamic_gates else 0.20
    )
    total_minimum, total_maximum = (0.80, 1.30) if dynamic_gates or descriptive_combined else (0.75, 1.35)
    try:
        model.eval()
        for length, row in panel:
            for timestep_value in timesteps:
                record = _zero_update_drift_cell(
                    model,
                    diffusion,
                    row,
                    length,
                    timestep_value,
                    update,
                    update_identity,
                    config,
                    table,
                    device,
                    settings,
                    downsample_factor,
                )
                records.append(record)
                finite = record["finite_counts"]
                ratios = record["individual_ratios"]
                aux_ratio = record["combined_auxiliary_ratio"]
                total_ratio = record["total_v_ratio"]
                if any(not item["all_finite"] for item in finite.values()):
                    violations.append({"identity": record, "reason": "non_finite_gradient"})
                if any(
                    not math.isfinite(value)
                    for collection in (record["loss_values"]["raw"], record["loss_values"]["weighted"])
                    for value in collection.values()
                ) or not math.isfinite(record["loss_values"]["v"]):
                    violations.append({"identity": record, "reason": "non_finite_loss"})
                if any(not math.isfinite(value) or value > individual_limit for value in ratios.values()):
                    violations.append({"identity": record, "reason": "individual_ratio"})
                if aux_ratio > warning_limit:
                    warnings.append(
                        {
                            "update": update,
                            "sample_id": record["sample_id"],
                            "length": length,
                            "timestep": timestep_value,
                            "combined_auxiliary_v_ratio": aux_ratio,
                            "total_v_ratio": total_ratio,
                            "total_v_cosine": record.get("total_v_cosine"),
                            "classification": (
                                "descriptive_warning"
                                if descriptive_combined
                                else (
                                    "accepted_dynamic_warning_below_hard_ceiling"
                                    if aux_ratio <= combined_limit
                                    else "hard_ceiling_exceeded"
                                )
                            ),
                        }
                    )
                if not math.isfinite(aux_ratio) or (not descriptive_combined and aux_ratio > combined_limit):
                    violations.append({"identity": record, "reason": "combined_auxiliary_ratio"})
                if not math.isfinite(total_ratio) or not total_minimum <= total_ratio <= total_maximum:
                    violations.append({"identity": record, "reason": "total_v_ratio"})
                if descriptive_combined and not math.isfinite(record.get("total_v_cosine", math.nan)):
                    violations.append({"identity": record, "reason": "non_finite_total_v_cosine"})
    finally:
        for parameter, gradient in zip(model.parameters(), grads_before, strict=True):
            parameter.grad = gradient
        model.train(mode_before)
    protected_after = verify_protected_evidence(config)
    rng_state_after = _rng_state()
    state_after = {
        "model": _fingerprint(model.state_dict()),
        "optimizer": _fingerprint(optimizer.state_dict()),
        "scheduler": _fingerprint(scheduler.state_dict()),
        "rng": _fingerprint(rng_state_after),
        "cpu_rng": _fingerprint(
            {"python": rng_state_after["python"], "numpy": rng_state_after["numpy"], "torch": rng_state_after["torch"]}
        ),
        "cuda_rng": _fingerprint(rng_state_after["cuda"]),
        "data_cursor": _fingerprint(data_cursor),
    }
    expected_after = {
        "model": model_before,
        "optimizer": optimizer_before,
        "scheduler": scheduler_before,
        "rng": rng_before,
        "cpu_rng": cpu_rng_before,
        "cuda_rng": cuda_rng_before,
        "data_cursor": cursor_hash_before,
    }
    if (
        state_after != expected_after
        or protected_after != protected_before
        or table["sha256"] != table_before
        or cursor_before != data_cursor
    ):
        violations.append({"identity": {"update": update}, "reason": "audit_mutated_state_or_protected_hash"})
    return _json_finite(
        {
            "update": update,
            "records": records,
            "state_hashes_before": {
                "model": model_before,
                "optimizer": optimizer_before,
                "scheduler": scheduler_before,
                "rng": rng_before,
                "cpu_rng": cpu_rng_before,
                "cuda_rng": cuda_rng_before,
                "data_cursor": cursor_hash_before,
                "protected": protected_before,
                "coefficient_table": table_before,
            },
            "state_hashes_after": state_after,
            "violations": violations,
            "warnings": warnings,
            "warning_count": len(warnings),
            "pass": not violations,
        }
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or config.get("version") != VERSION:
        raise ValueError("E007 Phase-3I.2 configuration version contradiction")
    dynamic = config.get("dynamic_preflight")
    valid_arms = (("v_only",),) if dynamic else (ARMS, LEGACY_ARMS)
    if tuple(config.get("arms", ())) not in valid_arms or tuple(config.get("samplers", ())) != SAMPLERS:
        raise ValueError("E007 Phase-3I.2 factorial contract changed")
    if list(map(int, config.get("lengths", ()))) != [64, 128, 256, 384, 500]:
        raise ValueError("E007 Phase-3I.2 length contract changed")
    pilot = config["pilot"]
    dynamic = config.get("dynamic_preflight")
    if dynamic:
        if tuple(config.get("arms", ())) != ("v_only",):
            raise ValueError("dynamic preflight must use only v_only")
        if int(pilot["maximum_optimizer_updates_per_arm"]) != 10 or list(map(int, pilot["evaluation_updates"])) != list(
            range(11)
        ):
            raise ValueError("dynamic preflight update/audit schedule mismatch")
    else:
        if int(pilot["maximum_optimizer_updates_per_arm"]) != 500:
            raise ValueError("E007 Phase-3I.2 pilot must stop at 500 updates")
        if list(map(int, pilot["evaluation_updates"])) != [0, 50, 100, 250, 500]:
            raise ValueError("E007 Phase-3I.2 evaluation schedule changed")
    objective = config["local_objective"]
    candidates = objective.get("candidate_coefficient_sets", {})
    if not candidates:
        raise ValueError("E007 Phase-3I.2 coefficient candidates are absent")
    for name, weights in candidates.items():
        if set(weights) != set(LOCAL_TERMS):
            raise ValueError(f"E007 Phase-3I.2 coefficient term contradiction: {name}")
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in weights.values()):
            raise ValueError(f"E007 Phase-3I.2 invalid coefficient: {name}")
    strengths = list(map(float, config["guidance"]["strengths"]))
    if not strengths or any(not math.isfinite(value) or value <= 0 for value in strengths):
        raise ValueError("E007 Phase-3I.2 guidance strengths must be finite and positive")
    return config


def verify_protected_evidence(config: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    records = [
        {
            "path": config["selected_checkpoint"]["path"],
            "sha256": config["selected_checkpoint"]["sha256"],
        },
        *config["protected_evidence"],
    ]
    for record in records:
        path = Path(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"E007 Phase-3I.2 protected input is absent: {path}")
        digest = _sha256_file(path)
        if digest != record["sha256"]:
            raise ValueError(f"E007 Phase-3I.2 protected-input hash contradiction: {path}")
        observed[path.as_posix()] = digest
    localization = json.loads(
        Path(
            "reports/experiments/E007_matrix_sequence_cogeneration/denoiser_sampler_localization_v1/report.json"
        ).read_text()
    )
    required = {"combined_denoiser_and_sampler_failure", "chirality_symmetry_limitation"}
    classifications = localization.get("summary", {}).get("decision_categories", ())
    if not required.issubset(set(classifications)):
        raise ValueError("E007 Phase-3I.2 prerequisite localization classifications changed")
    return observed


def verify_v3_evidence() -> dict[str, str]:
    records = {
        "configuration_sha256": (
            "configs/e007_local_backbone_repair_calibration_v3.yaml",
            V3_PINS["configuration_sha256"],
        ),
        "report_sha256": (
            "reports/experiments/E007_matrix_sequence_cogeneration/local_backbone_repair_calibration_v3/report.json",
            V3_PINS["report_sha256"],
        ),
        "protocol_sha256": (
            "reports/experiments/E007_matrix_sequence_cogeneration/local_backbone_repair_calibration_v3/protocol.json",
            V3_PINS["protocol_sha256"],
        ),
        "plan_sha256": (
            "reports/experiments/E007_matrix_sequence_cogeneration/local_backbone_repair_calibration_v3_plan.json",
            V3_PINS["plan_sha256"],
        ),
    }
    observed = {name: _sha256_file(Path(path)) for name, (path, _) in records.items() if Path(path).is_file()}
    if set(observed) != set(records) or any(observed[name] != expected for name, (_, expected) in records.items()):
        raise ValueError("protected v3 evidence hash mismatch")
    report = json.loads(Path(records["report_sha256"][0]).read_text())
    if report.get("selected_coefficient_set") != "budget_medium":
        raise ValueError("protected v3 selected tier mismatch")
    return observed


def validate_pilot_contract(
    config_path: str | Path, *, refuse_unreviewed: bool = False, allow_staging_for_resume: bool = False
) -> dict[str, Any]:
    """Read-only metadata and hash checks; does not import torch or create output."""
    path = Path(config_path)
    config = load_config(path)
    if "reviewed_v6_decision_path" in config:
        return _validate_reviewed_v6_pilot_contract(
            path, config, allow_staging_for_resume=allow_staging_for_resume, refuse_unreviewed=refuse_unreviewed
        )
    table_path = config["local_objective"].get("coefficient_table_path")
    if not table_path:
        raise ValueError("pilot config has no coefficient table path")
    table = load_coefficient_table(table_path)
    v3 = verify_v3_evidence()
    pins = config.get("v3_pins", {})
    if pins.get("coefficient_table_sha256", table["sha256"]) != table["sha256"]:
        raise ValueError("pilot coefficient-table pin mismatch")
    observed_evidence = {}
    validation_identities = {}
    schedule = []
    paired_identity_hash = None
    protected_dataset = None
    protected = verify_protected_evidence(config)
    if "deviation_record_path" in config:
        evidence_pins = {
            "dense_preflight_report_sha256": (
                config.get("dense_preflight_report_path"),
                "b3d3d5cc6e55e9e966b2316700b1b9700f19adcc55842dcf45d08f43ff57487a",
            ),
            "dense_preflight_protocol_sha256": (
                config.get("dense_preflight_protocol_path"),
                "bc88820f7b555f3b98df02c5122860db92949c76f37142691cc426638c6e98a9",
            ),
            "dense_preflight_cells_sha256": (
                config.get("dense_preflight_cells_path"),
                "5bb1b6bbfad540e29cf7b788ac481911a9ff9771bb0fc541e9aaeaf69f7ded13",
            ),
            "coefficient_table_file_sha256": (
                table_path,
                "e8e95e885984e9bb41d9040db69c0ef246cdd77a54b01b69db1133e543e18983",
            ),
        }
        observed_evidence = {}
        for key, (evidence_path, expected_hash) in evidence_pins.items():
            if not evidence_path or not Path(evidence_path).is_file():
                raise ValueError(f"required pinned evidence is absent: {key}")
            observed_evidence[key] = _sha256_file(Path(evidence_path))
            if observed_evidence[key] != expected_hash:
                raise ValueError(f"pinned evidence hash mismatch: {key}")
        if table["sha256"] != "d6bd01692d114c385143635a21a8db0475128d41c91f329f641c7611d3f83bd0":
            raise ValueError("coefficient schedule canonical hash mismatch")
        deviation_path = Path(config["deviation_record_path"])
        observed_evidence["deviation_record_sha256"] = _sha256_file(deviation_path)
        deviation = json.loads(deviation_path.read_text())
        expected_deviation = {
            "record_version": "e007_phase3i2_reviewed_dense_preflight_deviation_v1",
            "review_status": "reviewed",
            "sample_id": "5gl4_C",
            "length": 256,
            "timestep": 50,
            "observed_combined_auxiliary_v_ratio": 0.20471774690164152,
            "declared_preflight_limit": 0.2,
            "relative_exceedance": 0.0235887345082076,
            "all_tensors_finite": True,
            "all_individual_term_gates_passed": True,
            "total_v_ratio": 1.1416259901059571,
            "classification": "accepted_minor_preflight_deviation_for_bounded_exploratory_pilot",
            "scientific_gate_changed": False,
            "authorizes_later_phases": False,
            "authorized_scope": "bounded_exploratory_pilot_only",
            "downstream_authorizations": {
                "phase3j": False,
                "production_training": False,
                "joint_training": False,
                "sequence_conditioning": False,
                "downstream_generation": False,
            },
        }
        if deviation != expected_deviation:
            raise ValueError("reviewed dense-preflight deviation record mismatch")
        source = _phase3f_dataset_config(config)
        from protein_distance_diffusion.training import e007_coordinate_real_pilot as phase3f

        authorization = phase3f._authorize(source)
        protected_dataset = {
            "total": authorization.protected_input_total,
            "direct_resolutions": authorization.protected_input_resolution_counts.get("recorded_path", 0),
            "relocated_resolutions": authorization.protected_input_resolution_counts.get(
                "relocated_verification_path", 0
            ),
            "missing": 0,
            "contradictory": 0,
            "relocated_identities": list(authorization.relocated_protected_input_identities),
        }
        selected = phase3f._select_rows(source, authorization)
        schedule = _paired_schedule(source, selected["train"])
        if len(schedule) != 500 or any(not row["sample_ids"] for row in schedule):
            raise ValueError("complete paired training panel reconstruction failed")
        validation_identities = {
            name: [str(row["sample_id"]) for row in rows] for name, rows in selected["validation"].items()
        }
        if not validation_identities or any(not identities for identities in validation_identities.values()):
            raise ValueError("complete evaluation panel reconstruction failed")
        paired_identity_hash = _canonical_sha(
            {"schedule": schedule, "validation": validation_identities, "arms": list(ARMS)}
        )
        canonical_panel_hashes = {
            "training": _canonical_sha(
                {name: [str(row["sample_id"]) for row in rows] for name, rows in selected["train"].items()}
            ),
            "evaluation": _canonical_sha(validation_identities),
        }
        matched_arm_identity_components = {
            "sample_schedule_sha256": _canonical_sha(schedule),
            "timestep_noise_corruption_seed_schedule_sha256": _canonical_sha(
                {
                    "paired_seed": int(config["pilot"]["paired_seed"]),
                    "updates": [
                        {
                            "update": int(row["update"]),
                            "seed": int(config["pilot"]["paired_seed"]) + int(row["update"]) + 1,
                        }
                        for row in schedule
                    ],
                }
            ),
            "evaluation_identity_sha256": _canonical_sha(validation_identities),
        }
    if any(pins.get(key) != value for key, value in V3_PINS.items()):
        raise ValueError("pilot protected v3 pin mismatch")
    expected_schedule = [0, 50, 100, 250, 500]
    if list(map(int, config["pilot"]["evaluation_updates"])) != expected_schedule:
        raise ValueError("pilot update/evaluation schedule mismatch")
    if config["local_objective"].get("batch_reduction_policy") != (
        "mean_over_structures_including_zero_for_mask_ineligible"
    ):
        raise ValueError("per-structure auxiliary batch-reduction policy mismatch")
    if config["pilot"].get("maximum_optimizer_updates_per_arm") != 500:
        raise ValueError("pilot update count mismatch")
    monitor = config.get("online_drift_monitor", {})
    if (
        monitor.get("updates") != expected_schedule
        or monitor.get("timesteps") != list(COEFFICIENT_ANCHORS)
        or monitor.get("lengths") != list(config["lengths"])
        or monitor.get("fail_closed") is not True
        or monitor.get("individual_ratio_max") != 0.25
        or monitor.get("combined_auxiliary_ratio_max") != 0.25
        or monitor.get("total_ratio_min") != 0.75
        or monitor.get("total_ratio_max") != 1.35
        or monitor.get("clip_auxiliary_for_audit") is not False
        or (
            "gradient_audit" in config
            and tuple(map(int, config["gradient_audit"].get("updates", ()))) != GRADIENT_AUDIT_UPDATES
        )
    ):
        raise ValueError("drift monitoring contract mismatch")
    source_objective = yaml.safe_load(Path("configs/e007_coordinate_real_pilot_v1.yaml").read_text())["objective"]
    if source_objective != {
        "name": "uniform_valid_coordinate_v_mse",
        "timestep_sampling": "uniform_discrete",
        "timestep_weighting": "none",
        "auxiliary_losses": [],
    }:
        raise ValueError("matched-arm timestep/noise source contract mismatch")
    dense_hash = pins.get("dense_preflight_report_sha256")
    resume_schema = config.get("resume_schema", {})
    required_resume_fields = {
        "optimizer_update",
        "next_required_audit_boundary",
        "model",
        "optimizer",
        "scheduler",
        "rng_state",
        "sample_cursor",
        "protected_hashes",
        "coefficient_table_sha256",
    }
    if resume_schema.get("version") != 1 or set(resume_schema.get("required", ())) != required_resume_fields:
        raise ValueError("pilot recovery-state schema mismatch")
    if dense_hash is not None:
        report_path = config.get("dense_preflight_report_path")
        if not report_path or not Path(report_path).is_file() or _sha256_file(Path(report_path)) != dense_hash:
            raise ValueError("reviewed dense-preflight report path/hash mismatch")
        dense_report = json.loads(Path(report_path).read_text())
        failures = dense_report.get("gate_failures", [])
        accepted_failure = {
            "gate": "combined_ratio",
            "identity": {"sample_id": "5gl4_C", "length": 256, "timestep": 50},
        }
        if dense_report.get("status") != "completed" or failures != [accepted_failure]:
            raise ValueError("dense-preflight report failures differ from the reviewed exception")
    reviewed = bool(config.get("pilot_reviewed", False)) and bool(config.get("pilot_authorized", False))
    blocked = dense_hash is None or not reviewed
    result = {
        "mode": "validate_pilot_contract",
        "status": "blocked_pending_reviewed_dense_preflight" if blocked else "pilot_contract_validated",
        "configuration_sha256": _sha256_file(path),
        "coefficient_table_sha256": table["sha256"],
        "coefficient_table_file_sha256": observed_evidence.get("coefficient_table_file_sha256"),
        "deviation_record_sha256": observed_evidence.get("deviation_record_sha256"),
        "coefficient_rows": 500,
        "coefficient_schedule_reconstructed": True,
        "v3_hashes": v3,
        "protected_hashes": protected,
        "pinned_evidence_hashes": observed_evidence,
        "deviation_record_verified": True,
        "training_panel_updates": len(schedule),
        "training_schedule_first_10": schedule[:10],
        "evaluation_panel_identities": validation_identities,
        "training_panel_counts": {
            name: len(rows) for name, rows in (selected["train"].items() if "deviation_record_path" in config else [])
        },
        "evaluation_panel_counts": {
            name: len(rows)
            for name, rows in (selected["validation"].items() if "deviation_record_path" in config else [])
        },
        "canonical_panel_identity_sha256": canonical_panel_hashes if "deviation_record_path" in config else {},
        "matched_arm_identity_components": (
            matched_arm_identity_components if "deviation_record_path" in config else {}
        ),
        "matched_arm_identity_sha256": paired_identity_hash,
        "protected_dataset_inventory": protected_dataset,
        "coefficient_lookup_timesteps_verified": list(range(500)),
        "gradient_audit_updates": list(GRADIENT_AUDIT_UPDATES),
        "matched_arm_contract": (
            "identical checkpoint, fresh optimizer/scheduler, ordered samples, timesteps, noise, updates, evaluations"
        ),
        "resume_schema": (
            "versioned state includes update, RNG, optimizer, scheduler, sample cursor and next audit boundary"
        ),
        "dense_preflight_report_sha256": dense_hash,
        "pilot_reviewed": bool(config.get("pilot_reviewed", False)),
        "pilot_authorized": bool(config.get("pilot_authorized", False)),
        "model_created": False,
        "staging_created": False,
        "checkpoint_loaded": False,
        "cuda_initialized": False,
        "optimizer_created": False,
        "output_created": False,
        "forward_pass": False,
        "backward_pass": False,
        "sampling_performed": False,
        "optimizer_updates": 0,
        **NON_AUTHORIZING,
    }
    if blocked and refuse_unreviewed:
        raise ValueError("pilot execution blocked: reviewed dense-preflight hash and review flags are required")
    return result


def _validate_reviewed_lifecycle_smoke(decision: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Check the reviewed smoke and restrict v4 to the authorized path change."""
    base = Path("reports/experiments/E007_matrix_sequence_cogeneration")
    smoke = base / "local_backbone_repair_pilot_phase3i2_lifecycle_smoke_v3"
    expected = {
        "configuration": (
            Path("configs/e007_local_backbone_repair_pilot_phase3i2_reviewed_v6_v3.yaml"),
            "6127415514d8b0206d462ea9b1fa261fc5b4158829da9c33dd644a90d288dee7",
        ),
        "execution_log": (
            Path("logs/e007_local_backbone_repair_pilot_phase3i2_v3_lifecycle_smoke.log"),
            "552e4322794eaab876accd10cf5375d4bfb35c0fc6408249a7aee85bd68127c0",
        ),
        "drift_audits": (
            smoke / "arms/v_only/drift_audits.json",
            "e14dba43779cfaf6f0a8fb2af3d15e6e90c648bdc35823d1c767f76b174d84ee",
        ),
        "evaluations": (
            smoke / "arms/v_only/evaluations.json",
            "9f23328f1c22ed46ff95fa976928e2d37037a223845c90c87e8575024f6e7579",
        ),
        "latest": (smoke / "arms/v_only/latest.pt", "39d52d9177c4af79701c1b9c5f7084ca3ff6d29441bb03f538181dd959d9249f"),
        "memory_telemetry": (
            smoke / "arms/v_only/memory_telemetry.jsonl",
            "ad4592c251bf42c56849bfe7db57d67b87e33daced7938643d3814ebe7b75931",
        ),
        "metrics": (
            smoke / "arms/v_only/metrics.jsonl",
            "71d947d18ebbaa849e168aab25756d1517f47f328e110c67c2c5bf3d923d30b5",
        ),
        "heartbeat": (smoke / "heartbeat.json", "39296ee5ec4fd2b7be5c06b23a0079f48a2b05ca6bc9ebd8a30345dfbc173775"),
        "panel_manifest": (
            smoke / "panel_manifest.json",
            "42d9d1c5dd787a555c63475772936aa2e6f8646f546dd80181e4b19900caf656",
        ),
        "smoke_contract": (
            smoke / "smoke_contract.json",
            "cf79dcada8b7afc7ea287d65e755cc0cc0176f58e6156dc4dec1138df469a91f",
        ),
    }
    if decision.get("lifecycle_smoke_evidence") != {
        name: {"path": path.as_posix(), "sha256": digest} for name, (path, digest) in expected.items()
    }:
        raise ValueError("reviewed lifecycle smoke evidence pin mismatch")
    for name, (path, digest) in expected.items():
        if _sha256_file(path) != digest:
            raise ValueError(f"reviewed lifecycle smoke evidence changed: {name}")
    if (
        decision.get("pilot_v3_preparation_sha256")
        != "4a143f8b2b85a92ec5745e58cd42efcd5c0cd06de8ec93247fefefd4940093b9"
    ):
        raise ValueError("pilot-v3 preparation pin mismatch")
    if (
        _sha256_file(base / "local_backbone_repair_phase3i2_pilot_v3_preparation_v1.json")
        != decision["pilot_v3_preparation_sha256"]
    ):
        raise ValueError("pilot-v3 preparation changed")
    smoke_result = json.loads((smoke / "smoke_contract.json").read_text())
    memory = smoke_result.get("memory_telemetry", {})
    limits = config["memory"]
    if decision.get("lifecycle_smoke_review") != {
        "status": "completed",
        "arm": "v_only",
        "optimizer_updates": 25,
        "completed_audit_updates": list(range(11)) + [25],
        "update_zero_evaluation_lifecycle_completed": True,
        "scientific_pilot_result": False,
        "peak_rss_mib": 3237.99609375,
        "run_peak_cuda_allocated_mib": 5871.71630859375,
        "run_peak_cuda_reserved_mib": 6394.0,
        "device_total_cuda_mib": 8150.5625,
        "post_cleanup_current_cuda_allocated_active_mib": 152.66015625,
        "post_cleanup_current_cuda_reserved_mib": 266.0,
    }:
        raise ValueError("reviewed lifecycle smoke summary mismatch")
    if (
        smoke_result.get("status") != "completed"
        or smoke_result.get("arm") != "v_only"
        or smoke_result.get("optimizer_updates") != 25
        or smoke_result.get("completed_audit_updates") != list(range(11)) + [25]
        or smoke_result.get("update_zero_evaluation_lifecycle_completed") is not True
        or smoke_result.get("scientific_pilot_result") is not False
        or any(smoke_result.get(key) is not False for key in NON_AUTHORIZING)
        or memory.get("peak_rss_mib") != 3237.99609375
        or memory.get("run_peak_cuda_allocated_mib") != 5871.71630859375
        or memory.get("run_peak_cuda_reserved_mib") != 6394.0
        or memory.get("device_total_cuda_mib") != 8150.5625
        or memory.get("current_cuda_allocated_mib") != 152.66015625
        or memory.get("current_cuda_active_mib") != 152.66015625
        or memory.get("current_cuda_reserved_mib") != 266.0
        or memory["peak_rss_mib"] > limits["maximum_rss_mib"]
        or memory["run_peak_cuda_allocated_mib"] > limits["maximum_cuda_allocated_mib"]
        or memory["run_peak_cuda_reserved_mib"] > limits["maximum_cuda_reserved_mib"]
    ):
        raise ValueError("reviewed lifecycle smoke result contradiction")
    previous = load_config(expected["configuration"][0])
    allowed = {"output_dir", "reviewed_v6_decision_path", "authorization"}
    if {key: value for key, value in config.items() if key not in allowed} != {
        key: value for key, value in previous.items() if key not in allowed
    }:
        raise ValueError("pilot-v4 scientific contract differs from pilot-v3")
    if config["authorization"] != {**previous["authorization"], "pilot_authorized": True}:
        raise ValueError("pilot-v4 authorization scope mismatch")
    if decision.get("pilot_v3_configuration_sha256") != expected["configuration"][1]:
        raise ValueError("pilot-v3 configuration pin mismatch")
    if config["output_dir"] != (base / "local_backbone_repair_pilot_phase3i2_reviewed_v6_v4").as_posix():
        raise ValueError("pilot-v4 output path mismatch")
    if (
        config["reviewed_v6_decision_path"]
        != (base / "local_backbone_repair_phase3i2_lifecycle_smoke_reviewed_pilot_v4_decision_v1.json").as_posix()
    ):
        raise ValueError("pilot-v4 decision path mismatch")


def _validate_reviewed_v6_pilot_contract(
    path: Path, config: Mapping[str, Any], *, allow_staging_for_resume: bool = False, refuse_unreviewed: bool = False
) -> dict[str, Any]:
    """Fail closed on the exact reviewed pilot, using files and metadata only."""
    decision_path = Path(config["reviewed_v6_decision_path"])
    decision = json.loads(decision_path.read_text())
    prepared_v3 = decision.get("record_version") == "e007_phase3i2_pilot_v3_preparation_v1"
    reviewed_v4 = decision.get("record_version") == "e007_phase3i2_lifecycle_smoke_reviewed_pilot_v4_decision_v1"
    if (
        reviewed_v4
        and _sha256_file(decision_path) != "8ec9ea090fd9bc8c90af0d63a2c3266d92e71021d44b5eceafc35251aaa570b6"
    ):
        raise ValueError("reviewed pilot-v4 decision changed")
    if (
        not (prepared_v3 or reviewed_v4)
        and decision.get("record_version") != "e007_phase3i2_reviewed_v6_bounded_pilot_decision_v1"
    ):
        raise ValueError("reviewed-v6 decision version mismatch")
    if (
        decision.get("review_status") != ("prepared" if prepared_v3 else "reviewed")
        or decision.get("immutable") is not True
    ):
        raise ValueError("reviewed-v6 decision is not immutable and reviewed")
    if prepared_v3 or reviewed_v4:
        original = Path(
            "reports/experiments/E007_matrix_sequence_cogeneration/local_backbone_repair_phase3i2_reviewed_v6_decision_v1.json"
        )
        if decision.get("reviewed_v6_decision_sha256") != _sha256_file(original):
            raise ValueError("pilot-v3 original reviewed-v6 decision pin mismatch")
        incident = decision.get("failed_v2_incident", {})
        if incident.get("path") != "logs/e007_local_backbone_repair_pilot_phase3i2_reviewed_v6_v2.log" or incident.get(
            "sha256"
        ) != _sha256_file(Path(incident["path"])):
            raise ValueError("pilot-v3 failed-v2 incident pin mismatch")
        incident_record = decision.get("failed_v2_incident_record", {})
        if incident_record.get("path") != (
            "reports/experiments/E007_matrix_sequence_cogeneration/"
            "local_backbone_repair_phase3i2_failed_v2_memory_incident_v1.json"
        ) or incident_record.get("sha256") != _sha256_file(Path(incident_record["path"])):
            raise ValueError("pilot-v3 failed-v2 incident record mismatch")
        failed = json.loads(Path(incident_record["path"]).read_text())["failed_staging"]
        expected_names = {
            "drift_audits.json",
            "evaluations.json",
            "latest.pt",
            "metrics.jsonl",
            "heartbeat.json",
            "panel_manifest.json",
        }
        if set(failed["files"]) != expected_names:
            raise ValueError("pilot-v3 failed-v2 staging inventory mismatch")
        for item in failed["files"].values():
            artifact = Path(failed["path"]) / item["relative_path"]
            if _sha256_file(artifact) != item["expected_sha256"]:
                raise ValueError(f"pilot-v3 failed-v2 artifact changed: {artifact}")
        if prepared_v3 and refuse_unreviewed:
            raise ValueError("pilot-v3 is prepared for review and lifecycle smoke; execution is not authorized")
    if reviewed_v4:
        _validate_reviewed_lifecycle_smoke(decision, config)
    if decision.get("pilot_configuration_path") != path.as_posix() or decision.get(
        "pilot_configuration_sha256"
    ) != _sha256_file(path):
        raise ValueError("reviewed-v6 pilot configuration hash mismatch")
    if (decision.get("scientific_decision"), decision.get("authorized_scope")) != (
        (
            "prepare_identical_bounded_matched_arm_pilot_v3",
            "exactly_500_successful_optimizer_updates_per_arm_in_pilot_v3",
        )
        if prepared_v3
        else (
            "authorize_bounded_matched_arm_pilot_v4_only",
            "exactly_500_successful_optimizer_updates_per_arm_in_pilot_v4",
        )
        if reviewed_v4
        else (
            "authorize_bounded_matched_arm_pilot_v2_only",
            "exactly_500_successful_optimizer_updates_per_arm_in_reviewed_v6_pilot_v2",
        )
    ):
        raise ValueError("reviewed-v6 decision scope mismatch")
    if (
        decision.get("authorizes_pilot_execution") is not (not prepared_v3)
        or decision.get("pilot_completion_non_authorizing_pending_scientific_review") is not True
        or decision.get("downstream_authorizations")
        != {
            "phase3j": False,
            "production_training": False,
            "joint_training": False,
            "sequence_conditioning": False,
            "downstream_generation": False,
        }
    ):
        raise ValueError("reviewed-v6 downstream authorization mismatch")
    expected_artifacts = {
        "config": "99537329559db7b0012182fab0781a7c76e4195d99e95003da9e721f307d7b6c",
        "report": "455cfad6fe4a3a5814838c35e2a9479836890972651806bb7bffb46e9c24b093",
        "protocol": "762fcd396fe56ec8cccf7cf9e93e807cafec4cc3094c25fa46881151eefc2523",
        "heartbeat": "d8659ee9af3ec43212e1592387fdc3ed8e1dc119fc3a5c45e0266b9ec511d585",
        "drift_audits": "cb3be957a04a5c062fc19cb796cb634fc003f79e5a8995ab72ed35d725b78b4b",
        "recovery_checkpoint": "101619215120ac608dac54aecae7398a4df49f6169f83251ef7e6983c2d44b7f",
        "metrics": "51493259fd9a1a198236a402b66a655956877b0d6fd3d3214102e473a8e1107f",
        "execution_log": "574bf516f89f6b347bd13aedf47e5bf533d80d74854b77caa5279435af146000",
    }
    artifacts = decision.get("v6_artifacts", {})
    if set(artifacts) != set(expected_artifacts):
        raise ValueError("reviewed-v6 artifact inventory mismatch")
    for name, digest in expected_artifacts.items():
        item = artifacts[name]
        if item.get("sha256") != digest or _sha256_file(Path(item["path"])) != digest:
            raise ValueError(f"reviewed-v6 artifact mismatch: {name}")
    report = json.loads(Path(artifacts["report"]["path"]).read_text())
    audits = json.loads(Path(artifacts["drift_audits"]["path"]).read_text())
    records = [record for audit in audits for record in audit["records"]]
    summary = decision.get("v6_summary", {})
    if (
        report.get("status") != "passed"
        or len(records) != summary.get("audit_cells")
        or any(not audit.get("pass") or audit.get("violations") for audit in audits)
        or report.get("warning_count") != summary.get("descriptive_combined_gradient_warnings")
        or max(record["combined_auxiliary_ratio"] for record in records)
        != summary.get("maximum_combined_auxiliary_v_ratio")
        or min(record["total_v_ratio"] for record in records) != summary.get("minimum_total_v_ratio")
        or max(record["total_v_ratio"] for record in records) != summary.get("maximum_total_v_ratio")
        or min(record["total_v_cosine"] for record in records) != summary.get("minimum_total_v_cosine")
        or summary.get("hard_violations") != 0
        or summary.get("all_other_hard_gates_passed") is not True
    ):
        raise ValueError("reviewed-v6 scientific evidence contradiction")
    coefficient = decision["coefficient_schedule"]
    table_path = Path(config["local_objective"]["coefficient_table_path"])
    if (
        coefficient
        != {"path": table_path.as_posix(), "sha256": "1bcbd0fe6c4253799daf4c0a542c81aa640eac8ff88810bebbd957a826d606d8"}
        or _sha256_file(table_path) != coefficient["sha256"]
    ):
        raise ValueError("reviewed-v6 coefficient schedule file mismatch")
    table = load_coefficient_table(table_path)
    if table["sha256"] != config["local_objective"]["coefficient_table_sha256"] or len(table["timesteps"]) != 500:
        raise ValueError("reviewed-v6 coefficient lookup mismatch")
    pins = config.get("v3_pins", {})
    if (
        any(pins.get(key) != value for key, value in V3_PINS.items() if key in pins)
        or pins.get("coefficient_table_sha256") != table["sha256"]
    ):
        raise ValueError("reviewed-v6 calibration and coefficient pins mismatch")
    checkpoint = decision["selected_checkpoint"]
    if checkpoint != config["selected_checkpoint"] or checkpoint["optimizer_update"] != 9000:
        raise ValueError("reviewed-v6 initialization mismatch")
    protected = verify_protected_evidence(config)
    v3 = verify_v3_evidence()
    if not _calibration_ready(config, raise_on_failure=False):
        raise ValueError("reviewed-v6 calibration is not ready")
    if tuple(config["arms"]) != ARMS or tuple(config["samplers"]) != SAMPLERS:
        raise ValueError("reviewed-v6 matched-arm identity mismatch")
    pilot = config["pilot"]
    evaluations = list(map(int, pilot["evaluation_updates"]))
    if (
        pilot["maximum_optimizer_updates_per_arm"] != 500
        or pilot["successful_optimizer_updates_per_arm"] != 500
        or evaluations != [0, 50, 100, 250, 500]
        or pilot["initialization"] != "step_9000_model_weights_only"
        or pilot["optimizer_state"] != "fresh_identical_per_arm"
        or pilot["scheduler_state"] != "fresh_identical_per_arm"
        or pilot["recovery_checkpoint_frequency"] != 1
        or pilot["paired_seed"] != 3914001
        or pilot.get("paired_training_stream") != "same_ordered_sample_batch_timestep_noise_corruption_seed_per_update"
        or pilot.get("paired_evaluation_stream") != "same_validation_rows_and_sampling_seeds_per_update"
    ):
        raise ValueError("reviewed-v6 matched update/evaluation/recovery contract mismatch")
    publication = config["publication"]
    if (
        publication.get("primary_comparison")
        != "paired_local_backbone_validity_v_plus_local_minus_v_only_native_reverse"
        or publication.get("required_safeguards")
        != [
            "denoising_objective",
            "global_topology",
            "diversity",
            "chirality",
            "finite_coordinate_rate",
            "sampling_completion",
        ]
        or publication.get("publish_per_length_results") is not True
        or publication.get("publish_paired_bootstrap_intervals") is not True
        or publication.get("single_composite_score") is not False
        or publication.get("completion_authorizes_downstream") is not False
        or publication.get("local_backbone_validity_criteria")
        != {
            "maximum_adjacent_rmse_angstrom": 1.0,
            "maximum_discontinuity_fraction": 0.0,
            "maximum_clash_fraction": 0.0,
        }
        or publication.get("maximum_chirality_positive_fraction_shift") != 0.05
    ):
        raise ValueError("reviewed-v6 scientific comparison and safeguard contract mismatch")
    monitor = config["online_drift_monitor"]
    if (
        config.get("pilot_audit_semantics") != "reviewed_v6"
        or monitor.get("updates") != list(GRADIENT_AUDIT_UPDATES)
        or config.get("gradient_audit", {}).get("updates") != list(GRADIENT_AUDIT_UPDATES)
        or monitor.get("timesteps") != [25, 250, 425, 450, 475, 487, 499]
        or monitor.get("lengths") != config["lengths"]
        or monitor.get("individual_ratio_max") != 0.2
        or monitor.get("combined_auxiliary_warning_threshold") != 0.2
        or "combined_auxiliary_ratio_max" in monitor
        or monitor.get("total_ratio_min") != 0.8
        or monitor.get("total_ratio_max") != 1.3
        or monitor.get("record_total_v_cosine_descriptively") is not True
        or monitor.get("fail_closed") is not True
        or monitor.get("clip_auxiliary_for_audit") is not False
    ):
        raise ValueError("reviewed-v6 drift gate mismatch")
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    expected_name = "reviewed_v6_v4" if reviewed_v4 else "reviewed_v6_v3" if prepared_v3 else "reviewed_v6_v2"
    if output.exists() or (staging.exists() != allow_staging_for_resume) or expected_name not in output.name:
        raise ValueError("reviewed-v6 pilot output/staging is not fresh")
    if allow_staging_for_resume:
        heartbeat = staging / "heartbeat.json"
        if not heartbeat.is_file() or json.loads(heartbeat.read_text()).get("status") != "running":
            raise ValueError("reviewed-v6 pilot cannot resume failed or unrecognized staging")
    if config["resume_schema"].get("version") != 1 or set(config["resume_schema"].get("required", ())) != {
        "optimizer_update",
        "next_required_audit_boundary",
        "model",
        "optimizer",
        "scheduler",
        "rng_state",
        "sample_cursor",
        "protected_hashes",
        "coefficient_table_sha256",
    }:
        raise ValueError("reviewed-v6 exact-state resume schema mismatch")
    from protein_distance_diffusion.training import e007_coordinate_real_pilot as phase3f

    source = _phase3f_dataset_config(config)
    authorization = phase3f._authorize(source)
    selected = phase3f._select_rows(source, authorization)
    schedule = _paired_schedule(source, selected["train"])
    if len(schedule) != 500 or any(not row["sample_ids"] for row in schedule):
        raise ValueError("reviewed-v6 paired training schedule mismatch")
    validation = {name: [str(row["sample_id"]) for row in rows] for name, rows in selected["validation"].items()}
    if len(validation) != len(config["lengths"]) or any(not rows for rows in validation.values()):
        raise ValueError("reviewed-v6 paired evaluation panel mismatch")
    return {
        "mode": "validate_pilot_contract",
        "status": "pilot_contract_prepared" if prepared_v3 else "pilot_contract_validated",
        "configuration_sha256": _sha256_file(path),
        "decision_sha256": _sha256_file(decision_path),
        "v6_artifact_hashes": expected_artifacts,
        "coefficient_table_file_sha256": coefficient["sha256"],
        "coefficient_table_sha256": table["sha256"],
        "coefficient_lookup_timesteps_verified": list(range(500)),
        "selected_checkpoint_sha256": checkpoint["sha256"],
        "protected_hashes": protected,
        "v3_hashes": v3,
        "protected_dataset_inventory": {
            "total": authorization.protected_input_total,
            "direct_resolutions": authorization.protected_input_resolution_counts.get("recorded_path", 0),
            "relocated_resolutions": authorization.protected_input_resolution_counts.get(
                "relocated_verification_path", 0
            ),
            "relocated_identities": list(authorization.relocated_protected_input_identities),
        },
        "training_panel_updates": len(schedule),
        "training_schedule_sha256": _canonical_sha(schedule),
        "training_schedule_first_10": schedule[:10],
        "evaluation_panel_identities": validation,
        "matched_arm_identity_sha256": _canonical_sha(
            {"schedule": schedule, "validation": validation, "arms": list(ARMS)}
        ),
        "evaluation_updates": evaluations,
        "gradient_audit_updates": list(GRADIENT_AUDIT_UPDATES),
        "output_dir": output.as_posix(),
        "staging_dir": staging.as_posix(),
        "model_created": False,
        "checkpoint_loaded": False,
        "cuda_initialized": False,
        "optimizer_created": False,
        "output_created": False,
        "staging_created": False,
        "forward_pass": False,
        "backward_pass": False,
        "sampling_performed": False,
        "optimizer_updates": 0,
        **NON_AUTHORIZING,
    }


def validate_coefficient_table(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    observed = verify_v3_evidence()
    table_path = config["local_objective"]["coefficient_table_path"]
    table = load_coefficient_table(table_path)
    return {
        "mode": "validate_coefficient_table",
        "status": "validated_read_only",
        "coefficient_table_sha256": table["sha256"],
        "rows": len(table["timesteps"]),
        "v3_hashes": observed,
        "model_created": False,
        "cuda_initialized": False,
        "output_created": False,
        **NON_AUTHORIZING,
    }


def validate_dynamic_preflight_contract(config_path: str | Path) -> dict[str, Any]:
    """Validate the dynamic preflight without staging, model construction, or CUDA imports."""
    path = Path(config_path)
    config = load_config(path)
    contract = config.get("dynamic_preflight")
    if not contract or contract.get("version") not in {
        "e007_phase3i2_dynamic_stability_preflight_v1",
        "e007_phase3i2_dynamic_stability_preflight_v2",
        "e007_phase3i2_dynamic_stability_preflight_v3",
        "e007_phase3i2_dynamic_stability_preflight_v4",
        "e007_phase3i2_dynamic_stability_preflight_v5",
        "e007_phase3i2_dynamic_stability_preflight_v6",
    }:
        raise ValueError("dynamic-preflight contract version mismatch")
    table_path = config["local_objective"].get("coefficient_table_path")
    table = load_coefficient_table(table_path)
    if table["sha256"] != config["local_objective"].get("coefficient_table_sha256"):
        raise ValueError("dynamic-preflight coefficient-table pin mismatch")
    expected_anchors = {0: 1.0, 450: 1.0, 475: 0.8, 487: 0.6, 499: 0.375}
    if table.get("anchors") != expected_anchors:
        raise ValueError("dynamic-preflight schedule anchors mismatch")
    if contract.get("updates") != 10 or contract.get("audit_updates") != list(range(11)):
        raise ValueError("dynamic-preflight must execute 10 updates and audits 0..10")
    if contract.get("lengths") != [64, 128, 256, 384, 500] or contract.get("timesteps") != [
        25,
        250,
        425,
        450,
        475,
        487,
        499,
    ]:
        raise ValueError("dynamic-preflight audit grid mismatch")
    if contract.get("objective") != "v_only" or tuple(config["arms"]) != ("v_only",):
        raise ValueError("dynamic-preflight objective must remain v_only")
    version_suffix = contract["version"].rsplit("_", 1)[-1]
    expected_final = (
        "reports/experiments/E007_matrix_sequence_cogeneration/"
        f"local_backbone_repair_dynamic_stability_preflight_{version_suffix}"
    )
    expected_staging = (
        "reports/experiments/E007_matrix_sequence_cogeneration/"
        f".local_backbone_repair_dynamic_stability_preflight_{version_suffix}.inprogress"
    )
    if contract.get("final_output_dir") != expected_final or contract.get("staging_output_dir") != expected_staging:
        raise ValueError("dynamic-preflight publication paths changed")
    if contract["final_output_dir"] == contract["staging_output_dir"]:
        raise ValueError("dynamic-preflight final and staging paths must differ")
    if version_suffix in {"v3", "v4", "v5", "v6"}:
        pins = contract.get("smoke_evidence", {})
        required = (
            "report",
            "protocol",
            "artifact_inventory",
            "log",
            "configuration",
            "coefficient_schedule",
            "incident_record",
            "checkpoint",
        )
        if set(pins) != set(required):
            raise ValueError("dynamic-preflight v3 smoke evidence pins are incomplete")
        for name, pin in pins.items():
            if _sha256_file(Path(pin["path"])) != pin["sha256"]:
                raise ValueError(f"dynamic-preflight v3 {name} hash mismatch")
    if version_suffix in {"v4", "v5", "v6"}:
        pins = contract.get("lifecycle_evidence", {})
        required = {"failed_v3_log", "failed_v3_heartbeat", "smoke_report", "smoke_protocol", "smoke_inventory"}
        if set(pins) != required:
            raise ValueError("dynamic-preflight v4 lifecycle evidence pins are incomplete")
        for name, pin in pins.items():
            if _sha256_file(Path(pin["path"])) != pin["sha256"]:
                raise ValueError(f"dynamic-preflight v4 {name} hash mismatch")
        smoke = json.loads(Path(pins["smoke_report"]["path"]).read_text())
        if (
            smoke.get("status") != "passed"
            or smoke.get("smoke_kind") != "multi_cell_lifecycle"
            or smoke.get("optimizer_updates") != 0
            or smoke.get("sampling_performed") is not False
            or [(cell["length"], cell["timestep"]) for cell in smoke["measurements"]]
            != [(64, 25), (128, 250), (500, 499)]
            or any(cell["audit_state_restored"] is not True for cell in smoke["measurements"])
        ):
            raise ValueError("dynamic-preflight v4 multi-cell smoke evidence contradiction")
    if any(
        config.get("authorization", {}).get(field) is not False
        for field in (
            "pilot_authorized",
            "authorizes_training",
            "authorizes_production_training",
            "authorizes_phase3j",
            "authorizes_joint_training",
            "authorizes_sequence_conditioning",
            "authorizes_downstream_generation",
        )
    ):
        raise ValueError("dynamic-preflight authorization fields must all remain false")
    if config.get("pilot_authorized") is not False or config.get("pilot_reviewed") is not False:
        raise ValueError("dynamic-preflight pilot review and authorization must remain false")
    forbidden = (
        "sampling_performed",
        "reverse_sampling",
        "guided_sampling",
        "trajectory_generation",
        "expensive_evaluation_sampling",
    )
    if any(contract.get(key) is not False for key in forbidden):
        raise ValueError("dynamic-preflight sampling path is forbidden")
    if version_suffix in {"v5", "v6"}:
        pin = contract.get("reviewed_v4_warning", {})
        if pin.get("path") != (
            "reports/experiments/E007_matrix_sequence_cogeneration/"
            "local_backbone_repair_dynamic_stability_v4_warning_review_v1.json"
        ) or _sha256_file(Path(pin["path"])) != pin.get("sha256"):
            raise ValueError("dynamic-preflight v5 reviewed warning pin mismatch")
        review = json.loads(Path(pin["path"]).read_text())
        warning = review.get("warning", {})
        if (
            review.get("status") != "reviewed"
            or review.get("classification") != "accepted_dynamic_warning_below_hard_ceiling"
            or review.get("source_status") != "failed_closed"
            or review.get("source_failed_update_boundary") != 5
            or (warning.get("update"), warning.get("sample_id"), warning.get("length"), warning.get("timestep"))
            != (5, "2e19_A", 64, 425)
            or warning.get("combined_auxiliary_v_ratio") != 0.23559981839253288
            or warning.get("total_v_ratio") != 0.9288470670534813
            or warning.get("all_individual_gates_passed") is not True
            or warning.get("all_gradients_finite") is not True
            or warning.get("state_restored") is not True
            or review.get("pilot_authorized") is not False
            or review.get("authorizes_pilot_execution") is not False
            or review.get("authorizes_downstream_work") is not False
        ):
            raise ValueError("dynamic-preflight v5 reviewed warning contradiction")
        for evidence in review["v4_evidence"].values():
            if _sha256_file(Path(evidence["path"])) != evidence["sha256"]:
                raise ValueError("dynamic-preflight v5 source evidence hash mismatch")
    if version_suffix == "v6":
        incident_pin = contract.get("v5_incident_review", {})
        incident_path = (
            "reports/experiments/E007_matrix_sequence_cogeneration/"
            "local_backbone_repair_dynamic_stability_v5_incident_review_v1.json"
        )
        if incident_pin.get("path") != incident_path or _sha256_file(Path(incident_path)) != incident_pin.get("sha256"):
            raise ValueError("dynamic-preflight v6 incident review pin mismatch")
        incident = json.loads(Path(incident_path).read_text())
        if (
            incident.get("status") != "reviewed"
            or incident.get("source_status") != "failed_closed"
            or incident.get("failed_update_boundary") != 7
            or incident.get("audited_cell_count") != 280
            or incident.get("hard_violation_count") != 1
            or incident.get("prior_boundaries_passed") is not True
            or incident.get("prior_warning_count") != 1
            or incident.get("failed_cell", {}).get("combined_auxiliary_v_ratio") != 0.3132148858397767
            or incident.get("failed_cell", {}).get("total_v_ratio") != 0.9970738914566103
            or incident.get("failed_cell", {}).get("violation_reason") != "combined_auxiliary_ratio"
            or incident.get("failed_cell", {}).get("all_individual_gates_passed") is not True
            or incident.get("failed_cell", {}).get("all_gradients_finite") is not True
            or incident.get("pilot_authorized") is not False
            or incident.get("authorizes_pilot_execution") is not False
        ):
            raise ValueError("dynamic-preflight v6 incident review contradiction")
        for evidence in incident["v5_evidence"].values():
            if _sha256_file(Path(evidence["path"])) != evidence["sha256"]:
                raise ValueError("dynamic-preflight v6 source evidence hash mismatch")
    if (
        contract.get("maximum_individual_term_ratio") != 0.20
        or (
            version_suffix != "v6"
            and contract.get("maximum_combined_auxiliary_v_ratio") != (0.25 if version_suffix == "v5" else 0.20)
        )
        or (version_suffix == "v6" and "maximum_combined_auxiliary_v_ratio" in contract)
        or (version_suffix in {"v5", "v6"} and contract.get("combined_auxiliary_warning_threshold") != 0.20)
        or (version_suffix == "v6" and contract.get("total_v_cosine_telemetry") is not True)
        or contract.get("total_v_ratio_min") != 0.80
        or contract.get("total_v_ratio_max") != 1.30
        or contract.get("fail_closed") is not True
        or contract.get("exact_all_finite") is not True
    ):
        raise ValueError("dynamic-preflight safety gates changed")
    if contract.get("planned_forward_count") != 395 or contract.get("planned_backward_count") != 2705:
        raise ValueError("dynamic-preflight workload estimate contradicts the declared 10-update grid")
    production_config = load_config("configs/e007_local_backbone_repair_pilot_phase3i2_final_v1.yaml")
    if int(production_config["pilot"]["maximum_optimizer_updates_per_arm"]) != 500:
        raise ValueError("final pilot update schedule changed")
    if production_config["dataset"].get("protected_input_relocations") != config["dataset"].get(
        "protected_input_relocations"
    ):
        raise ValueError("final pilot relocation mapping changed")
    _phase3f_dataset_config(production_config)
    protected = verify_protected_evidence(config)
    panel_path = Path(contract["final_pilot_panel_manifest_path"])
    if _sha256_file(panel_path) != contract["final_pilot_panel_manifest_sha256"]:
        raise ValueError("final pilot panel manifest hash mismatch")
    panel = json.loads(panel_path.read_text())
    if panel.get("schedule_sha256") != contract["final_pilot_schedule_sha256"]:
        raise ValueError("final pilot schedule hash mismatch")
    if panel.get("train_sample_ids", [])[:10] != contract["first_10_update_sample_ids"]:
        raise ValueError("dynamic preflight first ten training identities differ from the final pilot")
    if panel.get("drift_audit_panel_sample_ids") != contract["audit_sample_ids_by_length"]:
        raise ValueError("dynamic preflight deterministic audit panel identities changed")
    if contract["audit_sample_ids_by_length"].get("128") != "3qoc_C":
        raise ValueError("required 3qoc_C/128 audit identity is missing")
    if int(contract["paired_seed"]) != int(production_config["pilot"]["paired_seed"]):
        raise ValueError("dynamic preflight training corruption seed base changed")
    # Hashing the deterministic identity/noise key makes every audit cell reproducible.
    audit_identity = {
        "seed": int(contract["deterministic_audit_seed"]),
        "lengths": contract["lengths"],
        "timesteps": contract["timesteps"],
        "required_cell": contract["required_cell"],
    }
    parent_table = load_coefficient_table("configs/e007_local_backbone_repair_budget_medium_coefficients_v2.json")
    output, staging = Path(contract["final_output_dir"]), Path(contract["staging_output_dir"])
    return {
        "mode": "validate_dynamic_preflight_contract",
        "status": "validated_read_only",
        "configuration_sha256": _sha256_file(path),
        "coefficient_table_sha256": table["sha256"],
        "coefficient_table_file_sha256": _sha256_file(Path(table_path)),
        "coefficient_table_rows": len(table["timesteps"]),
        "parent_v2_canonical_sha256": table["parent_canonical_sha256"],
        "parent_v2_file_sha256": table["parent_file_sha256"],
        "multiplier_policy_canonical_sha256": table["multiplier_policy_sha256"],
        "anchors": {str(key): value for key, value in expected_anchors.items()},
        "rows_0_450_unchanged": all(
            table["timesteps"][str(t)]["local_terms"] == parent_table["timesteps"][str(t)]["local_terms"]
            for t in range(451)
        ),
        "all_coefficients_finite_positive": True,
        "training_schedule_updates": 500,
        "training_schedule_first_10_sample_ids": contract["first_10_update_sample_ids"],
        "training_corruption_seeds": [int(contract["paired_seed"]) + update + 1 for update in range(10)],
        "audit_sample_ids_by_length": contract["audit_sample_ids_by_length"],
        "audit_identity_sha256": _canonical_sha(audit_identity),
        "audit_updates": list(range(11)),
        "combined_auxiliary_warning_threshold": contract.get("combined_auxiliary_warning_threshold"),
        "maximum_combined_auxiliary_v_ratio": contract.get("maximum_combined_auxiliary_v_ratio"),
        "planned_forward_count": int(contract["planned_forward_count"]),
        "planned_backward_count": int(contract["planned_backward_count"]),
        "estimated_runtime_seconds": int(contract["estimated_runtime_seconds"]),
        "protected_hashes": protected,
        "staging_created": False,
        "model_created": False,
        "checkpoint_loaded": False,
        "cuda_initialized": False,
        "forward_pass": False,
        "backward_pass": False,
        "sampling_performed": False,
        "optimizer_updates": 0,
        "final_output_exists": output.exists(),
        "staging_output_exists": staging.exists(),
        **NON_AUTHORIZING,
    }


def monitor_dynamic_preflight(config_path: str | Path) -> dict[str, Any]:
    """Read-only inspection of a dynamic-preflight staging journal."""
    config = load_config(config_path)
    contract = config["dynamic_preflight"]
    staging = Path(contract["staging_output_dir"])
    output = Path(contract["final_output_dir"])
    report = output / "report.json" if output.exists() else staging / "report.json"
    published_report = json.loads(report.read_text()) if report.exists() else None
    if output.exists():
        status = "published"
    elif published_report and published_report.get("status") == "failed_closed":
        status = "failed_closed"
    elif staging.exists():
        status = "in_progress"
    else:
        status = "not_started"
    return {
        "mode": "monitor_dynamic_preflight",
        "status": status,
        "staging_exists": staging.exists(),
        "final_output_exists": output.exists(),
        "report_sha256": _sha256_file(report) if report.exists() else None,
        "sampling_performed": False,
        **NON_AUTHORIZING,
    }


def validate_dynamic_memory_smoke_contract(config_path: str | Path) -> dict[str, Any]:
    """Read-only validation of all inputs and limits needed by the memory smoke."""
    dynamic_contract = validate_dynamic_preflight_contract(config_path)
    config = load_config(config_path)
    if int(config["selected_checkpoint"].get("optimizer_update", -1)) != 9000:
        raise ValueError("dynamic memory smoke requires the immutable step-9000 checkpoint")
    resolved = _resolve_reviewed_coordinate_source(config)
    audit_source = resolved["audit_source"]
    from protein_distance_diffusion.evaluation import e007_denoiser_sampler_localization as localization

    localization.verify_prerequisites(audit_source, full=True)
    compact, _ = localization.select_validation_panel(audit_source)
    reconstructed, _ = localization.reconstruct_authoritative_panel(audit_source, compact)
    wanted = config["dynamic_preflight"]["audit_sample_ids_by_length"]["500"]
    rows = [item for item in reconstructed if int(item["selection"]["target_length"]) == 500]
    selected = [item for item in rows if str(item["canonical_row"]["sample_id"]) == wanted]
    if len(selected) != 1:
        raise ValueError("dynamic memory smoke length-500 panel identity did not reconstruct uniquely")
    table = load_coefficient_table(config["local_objective"]["coefficient_table_path"])
    limits = config.get("memory", {})
    required_limits = ("maximum_rss_mib", "maximum_cuda_allocated_mib", "maximum_cuda_reserved_mib")
    if any(float(limits.get(key, 0)) <= 0 for key in required_limits):
        raise ValueError("dynamic memory smoke memory limits are incomplete")
    suffix = config["dynamic_preflight"]["version"].rsplit("_", 1)[-1]
    output = Path(
        f"reports/experiments/E007_matrix_sequence_cogeneration/local_backbone_repair_dynamic_memory_smoke_{suffix}"
    )
    if not output.parent.is_dir():
        raise FileNotFoundError(f"dynamic memory smoke output parent is missing: {output.parent}")
    return {
        "mode": "validate_dynamic_memory_smoke_contract",
        "status": "validated_read_only",
        "configuration_sha256": _sha256_file(Path(config_path)),
        "reviewed_source_config": str(resolved["source_config_path"]),
        "model_contract": resolved["source"]["model"],
        "checkpoint_path": str(resolved["checkpoint_path"]),
        "checkpoint_sha256": config["selected_checkpoint"]["sha256"],
        "checkpoint_optimizer_update": int(config["selected_checkpoint"]["optimizer_update"]),
        "dataset_source_config": str(config["dataset_source_config"]),
        "panel_sample_id": wanted,
        "panel_rows_at_length_500": len(rows),
        "coefficient_table_sha256": table["sha256"],
        "memory_limits_mib": limits,
        "output_dir": str(output),
        "output_exists": output.exists(),
        "model_created": False,
        "checkpoint_loaded": False,
        "cuda_initialized": False,
        "staging_created": False,
        "dynamic_preflight_contract": dynamic_contract["status"],
        **NON_AUTHORIZING,
    }


def _dynamic_warning_telemetry(audits: list[dict[str, Any]]) -> dict[str, Any]:
    """Preserve every warning identity and cosine value by boundary and globally."""

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        warnings = [warning for audit in items for warning in audit.get("warnings", [])]
        cells = [cell for audit in items for cell in audit.get("records", [])]
        cosines = [cell.get("total_v_cosine") for cell in cells]
        maximum = max(cells, key=lambda cell: cell["combined_auxiliary_ratio"], default=None)
        return {
            "warning_count": len(warnings),
            "warning_identities": [
                {key: warning[key] for key in ("update", "sample_id", "length", "timestep")} for warning in warnings
            ],
            "maximum_combined_auxiliary_v_ratio": maximum["combined_auxiliary_ratio"] if maximum else None,
            "maximum_combined_auxiliary_identity": (
                {key: maximum[key] for key in ("update", "sample_id", "length", "timestep")} if maximum else None
            ),
            "total_v_cosines": cosines,
            "total_v_cosine_min": min(cosines) if cosines and None not in cosines else None,
            "total_v_cosine_max": max(cosines) if cosines and None not in cosines else None,
        }

    return {
        "by_boundary": {str(audit["update"]): summarize([audit]) for audit in audits},
        "global": summarize(audits),
    }


def _dynamic_memory_violation(observed: Mapping[str, float], limits: Mapping[str, float]) -> bool:
    required = ("peak_rss_mib", "run_peak_cuda_allocated_mib", "run_peak_cuda_reserved_mib")
    if any(not math.isfinite(float(observed[key])) or float(observed[key]) < 0 for key in required):
        return True
    return (
        observed["peak_rss_mib"] > float(limits["maximum_rss_mib"])
        or observed["run_peak_cuda_allocated_mib"] > float(limits["maximum_cuda_allocated_mib"])
        or observed["run_peak_cuda_reserved_mib"] > float(limits["maximum_cuda_reserved_mib"])
    )


def run_dynamic_stability_preflight(config_path: str | Path, *, resume: bool = False) -> dict[str, Any]:
    """Run the ten-update v-only dynamic stability gate with exact boundary recovery."""
    config_path = Path(config_path)
    contract_report = validate_dynamic_preflight_contract(config_path)
    config = load_config(config_path)
    dynamic = config["dynamic_preflight"]
    # Reconstruct both panels and the full paired schedule before making staging state.
    from protein_distance_diffusion.evaluation import e007_denoiser_sampler_localization as localization
    from protein_distance_diffusion.training import e007_coordinate_real_pilot as phase3f

    resolved = _resolve_reviewed_coordinate_source(config)
    source = resolved["source"]
    source["successful_optimizer_updates"] = 500
    source["seed"] = int(dynamic["paired_seed"])
    authorization = phase3f._authorize(source)
    selected = phase3f._select_rows(source, authorization)
    schedule = _paired_schedule(source, selected["train"])
    schedule_hash = _canonical_sha(schedule)
    if schedule_hash != dynamic["final_pilot_schedule_sha256"]:
        raise ValueError("reconstructed final pilot schedule differs from the pinned dynamic contract")
    if [entry["sample_ids"] for entry in schedule[:10]] != dynamic["first_10_update_sample_ids"]:
        raise ValueError("reconstructed first ten update identities differ from the dynamic contract")
    audit_source = localization.load_config(config["dataset_source_config"])
    localization.verify_prerequisites(audit_source, full=True)
    compact, _selection = localization.select_validation_panel(audit_source)
    reconstructed, _canonical = localization.reconstruct_authoritative_panel(audit_source, compact)
    by_length: dict[int, dict[str, Any]] = {}
    expected_audits = dynamic["audit_sample_ids_by_length"]
    for item in reconstructed:
        length = int(item["selection"]["target_length"])
        row = item["canonical_row"]
        if str(length) in expected_audits and str(row["sample_id"]) == expected_audits[str(length)]:
            by_length[length] = row
    lengths = list(map(int, dynamic["lengths"]))
    if sorted(by_length) != sorted(lengths):
        raise ValueError("pinned deterministic audit identities did not reconstruct for every length")
    audit_panel = [(length, by_length[length]) for length in lengths]
    protected = verify_protected_evidence(config)
    table = load_coefficient_table(config["local_objective"]["coefficient_table_path"])
    output, staging = Path(dynamic["final_output_dir"]), Path(dynamic["staging_output_dir"])
    if output.exists() or (staging.exists() and not resume):
        raise FileExistsError(f"dynamic preflight output exists: {output} or {staging}")
    if resume and not staging.is_dir():
        raise FileNotFoundError("dynamic-preflight resume requires its staging directory")

    # CUDA and model setup start only after every metadata, hash, panel, and schedule gate above.
    import random
    import resource

    import numpy as np
    import torch

    from protein_distance_diffusion.models.coordinate_equivariance import coordinate_model_execution_context
    from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import EquivariantPairCoordinateUNet
    from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
    from protein_distance_diffusion.training.e007_coordinate_real_loader_smoke import prepare_coordinate_batch

    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("dynamic stability preflight requires configured CUDA")
    if not resume:
        staging.mkdir(parents=True)
        _atomic_json(
            staging / "heartbeat.json",
            {"status": "running", "stage": "initialized", "optimizer_updates": 0, **NON_AUTHORIZING},
        )
    arm_dir = staging / "v_only"
    arm_dir.mkdir(exist_ok=True)
    latest = arm_dir / "latest.pt"
    metrics_path, audits_path = arm_dir / "metrics.jsonl", arm_dir / "drift_audits.json"
    metrics = (
        [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        if metrics_path.exists()
        else []
    )
    audits = json.loads(audits_path.read_text()) if audits_path.exists() else []
    if [int(row["optimizer_update"]) for row in metrics] != list(range(1, len(metrics) + 1)):
        raise ValueError("dynamic-preflight metrics are not a complete update prefix")
    if [int(row["update"]) for row in audits] != list(range(len(audits))):
        raise ValueError("dynamic-preflight audits are not a complete boundary prefix")

    seed = int(dynamic["paired_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    telemetry = CudaMemoryTelemetry(torch.cuda, device)
    checkpoint = _resolve_reviewed_coordinate_source(config, load_checkpoint=True)["checkpoint"]
    with coordinate_model_execution_context(config["numerics"], device):
        model = EquivariantPairCoordinateUNet(**source["model"]).to(device).eval()
        model.load_state_dict(checkpoint["model"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(source["optimizer"]["learning_rate"]),
            weight_decay=float(source["optimizer"]["weight_decay"]),
            betas=tuple(source["optimizer"]["betas"]),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        diffusion = CoordinateVPDiffusion(int(config["diffusion_steps"]))
        update = samples = residues = 0
        completed_audits: set[int] = set()
        memory_history: list[dict[str, float]] = []
        if latest.exists():
            if not resume:
                raise FileExistsError(f"dynamic-preflight recovery state exists: {latest}")
            state = torch.load(latest, map_location="cpu", weights_only=False)
            _validate_recovery_state(
                state, config_path, protected, "v_only", schedule_hash, table["sha256"], audit_boundaries=range(11)
            )
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            _restore_rng_state(state["rng_state"])
            update = int(state["optimizer_update"])
            samples = int(state["samples_processed"])
            residues = int(state["valid_residues_processed"])
            completed_audits = set(map(int, state["completed_audit_updates"]))
        _reconcile_metrics_to_checkpoint(metrics_path, update)
        metrics = (
            [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
            if metrics_path.exists()
            else []
        )

        def recovery_payload() -> dict[str, Any]:
            return _recovery_payload(
                model,
                optimizer,
                scheduler,
                update,
                samples,
                residues,
                config_path,
                protected,
                "v_only",
                schedule_hash,
                table["sha256"],
                sorted(completed_audits),
                audit_boundaries=range(11),
            )

        def check_memory(*, reset_boundary_peak: bool = False) -> dict[str, float]:
            rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            memory = telemetry.end_phase() if reset_boundary_peak else telemetry.snapshot()
            limits = config["memory"]
            observed = {
                "peak_rss_mib": rss_mib,
                **memory,
                "boundary_peak_cuda_allocated_mib": memory["phase_peak_cuda_allocated_mib"],
                "boundary_peak_cuda_reserved_mib": memory["phase_peak_cuda_reserved_mib"],
            }
            if _dynamic_memory_violation(observed, limits):
                raise MemoryError(f"dynamic-preflight configured memory limit exceeded: {observed}")
            return observed

        def audit_boundary(boundary: int) -> None:
            nonlocal audits
            candidate = zero_update_drift_audit(
                model,
                optimizer,
                scheduler,
                diffusion,
                audit_panel,
                config,
                table,
                device,
                update=boundary,
                data_cursor=boundary,
                audit_timesteps=dynamic["timesteps"],
                update_identity=(
                    {"boundary": 0, "checkpoint_sha256": config["selected_checkpoint"]["sha256"]}
                    if boundary == 0
                    else schedule[boundary - 1]
                ),
            )
            if boundary < len(audits):
                if _canonical_sha(audits[boundary]) != _canonical_sha(candidate):
                    raise ValueError(f"dynamic-preflight deterministic replay mismatch at boundary {boundary}")
            else:
                audits.append(candidate)
                _atomic_json(audits_path, audits)
            if not candidate["pass"]:
                failure_report = {
                    "version": dynamic["version"],
                    "status": "failed_closed",
                    "failed_update_boundary": boundary,
                    "failed_cells": candidate["violations"],
                    "audit_updates": audits,
                    "warning_count": sum(item.get("warning_count", 0) for item in audits),
                    "warnings": [warning for item in audits for warning in item.get("warnings", [])],
                    "warning_telemetry": _dynamic_warning_telemetry(audits),
                    "optimizer_updates": update,
                    "sampling_performed": False,
                    **NON_AUTHORIZING,
                }
                _atomic_json(staging / "report.json", failure_report)
                _atomic_json(
                    staging / "heartbeat.json",
                    {"status": "failed_closed", "optimizer_update": update, **NON_AUTHORIZING},
                )
                raise FloatingPointError(f"dynamic-preflight audit gate failed at update {boundary}")
            completed_audits.add(boundary)
            memory_history.append({"audit_boundary": float(boundary), **check_memory(reset_boundary_peak=True)})
            _atomic_torch(latest, recovery_payload())

        audit_boundary(update)
        settings = dict(config["local_objective"])
        settings["coordinate_scale_angstrom"] = float(config["coordinate_scale_angstrom"])
        downsample = int(audit_source["expected_downsample_factor"])
        while update < 10:
            entry = schedule[update]
            rows = selected["train"][entry["stratum"]][entry["start"] : entry["stop"]]
            if [str(row["sample_id"]) for row in rows] != entry["sample_ids"]:
                raise ValueError("dynamic-preflight training identity contradiction")
            prepared = prepare_coordinate_batch(rows, float(config["coordinate_scale_angstrom"]), downsample)
            corruption = phase3f.make_uniform_training_corruption(
                prepared, diffusion, seed=seed + update + 1, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            model.train()
            prediction = model(
                corruption["batch"].noisy_coordinates,
                corruption["timesteps"],
                prepared["lengths"].to(device),
                corruption["mask"],
                prepared["chain_continuity_mask"].to(device),
            )["v_prediction"]
            v_loss = phase3f.uniform_coordinate_v_mse(
                prediction, corruption["batch"].coordinate_v_target, corruption["mask"]
            )
            if not bool(torch.isfinite(v_loss)):
                raise FloatingPointError("dynamic-preflight v-only training loss is non-finite")
            v_loss.backward()
            phase3f._gradient_evidence(model)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(source["optimizer"]["gradient_clip_norm"]), error_if_nonfinite=True
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            update += 1
            samples += len(rows)
            residues += int(prepared["lengths"].sum())
            metric = {
                "arm": "v_only",
                "optimizer_update": update,
                "update_identity": entry,
                "corruption_seed": seed + update,
                "timesteps": [int(value) for value in corruption["timesteps"].detach().cpu().tolist()],
                "sample_ids": list(entry["sample_ids"]),
                "v_loss": float(v_loss.detach().cpu()),
                "total_loss": float(v_loss.detach().cpu()),
                "auxiliary_losses_used": False,
                "sampling_performed": False,
            }
            _append_fsync_jsonl(metrics_path, metric)
            metrics.append(metric)
            _atomic_torch(latest, recovery_payload())
            if update in range(11):
                audit_boundary(update)
            del prediction, v_loss, corruption, prepared, rows
            import gc

            gc.collect()
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            memory = check_memory(reset_boundary_peak=True)
            memory_history.append(memory)
            _atomic_torch(latest, recovery_payload())
            _atomic_json(
                staging / "heartbeat.json",
                {
                    "status": "running",
                    "stage": "training",
                    "optimizer_update": update,
                    "memory": memory,
                    "sampling_performed": False,
                    **NON_AUTHORIZING,
                },
            )
        if update != 10 or completed_audits != set(range(11)):
            raise RuntimeError("dynamic-preflight completion boundary mismatch")
        replay_hash = _canonical_sha({"metrics": metrics, "audits": audits})
        warnings = [warning for item in audits for warning in item.get("warnings", [])]
        report = {
            "version": dynamic["version"],
            "status": "passed" if all(item["pass"] for item in audits) else "failed",
            "configuration_sha256": _sha256_file(config_path),
            "coefficient_table_sha256": table["sha256"],
            "protected_hashes": protected,
            "training_updates": metrics,
            "audit_updates": audits,
            "warning_count": len(warnings),
            "warnings": warnings,
            "warning_telemetry": _dynamic_warning_telemetry(audits),
            "replay_sha256": replay_hash,
            "memory_observations": memory_history,
            "optimizer_updates": update,
            "planned_forward_count": contract_report["planned_forward_count"],
            "planned_backward_count": contract_report["planned_backward_count"],
            "sampling_performed": False,
            "reverse_sampling_performed": False,
            "guided_sampling_performed": False,
            "trajectory_generation_performed": False,
            "evaluation_sampling_performed": False,
            "passing_authorizes_preparation_only": True,
            "authorizes_pilot_execution": False,
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "report.json", report)
        _atomic_json(
            staging / "protocol.json", {"report_sha256": _sha256_file(staging / "report.json"), **NON_AUTHORIZING}
        )
        _atomic_json(staging / "heartbeat.json", {"status": "completed", **NON_AUTHORIZING})
        staging.replace(output)
        _fsync_directory(output.parent)
        return report


def run_dense_timestep_preflight(config_path: str | Path) -> dict[str, Any]:
    """Zero-update 130-cell preflight, atomically resumable at complete-cell boundaries."""
    import resource

    import torch

    from protein_distance_diffusion.evaluation import e007_denoiser_sampler_localization as localization
    from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
    from protein_distance_diffusion.training.e007_coordinate_real_pilot import uniform_coordinate_v_mse

    config_path = Path(config_path)
    config = load_config(config_path)
    dense = config["dense_preflight"]
    preflight_timesteps = tuple(map(int, dense["timesteps"]))
    if preflight_timesteps not in (PREFLIGHT_TIMESTEPS, PREFLIGHT_V2_TIMESTEPS) or dense["structures_per_length"] != 2:
        raise ValueError("dense-preflight grid contract mismatch")
    if int(dense["optimizer_updates"]) != 0:
        raise ValueError("dense preflight must have zero optimizer updates")
    table = load_coefficient_table(config["local_objective"]["coefficient_table_path"])
    protected_before = verify_protected_evidence(config)
    v3_before = verify_v3_evidence()
    source = localization.load_config(config["dataset_source_config"])
    localization.verify_prerequisites(source, full=True)
    compact, selection = localization.select_validation_panel(source)
    reconstructed, canonical = localization.reconstruct_authoritative_panel(source, compact)
    by_length: dict[int, list[dict[str, Any]]] = {length: [] for length in config["lengths"]}
    for record in reconstructed:
        length = int(record["selection"]["target_length"])
        if length in by_length and len(by_length[length]) < 2:
            by_length[length].append(record["canonical_row"])
    if any(len(records) != 2 for records in by_length.values()):
        raise ValueError("dense-preflight immutable panel lacks two canonical structures per length")
    output = Path(dense["final_output_dir"])
    staging = Path(dense["staging_output_dir"])
    if output.exists():
        raise FileExistsError(f"dense-preflight output exists: {output}")
    staging.mkdir(parents=True, exist_ok=True)
    cells_path = staging / "cells.jsonl"
    cell_count = len(config["lengths"]) * len(preflight_timesteps) * 2
    completed: list[dict[str, Any]] = []
    if cells_path.exists():
        for line in cells_path.read_text().splitlines():
            if line.strip():
                completed.append(json.loads(line))
    expected_ids = [
        {"length": int(length), "timestep": int(timestep), "sample_id": str(row["sample_id"])}
        for length in config["lengths"]
        for row in by_length[int(length)]
        for timestep in preflight_timesteps
    ]
    actual_ids = [row["identity"] for row in completed]
    if actual_ids != expected_ids[: len(actual_ids)] or len(actual_ids) > cell_count:
        raise ValueError("dense-preflight resume journal is not a complete-cell prefix")
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("dense preflight requires configured CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    model = localization._load_model(source, device).requires_grad_(True).eval()
    diffusion = CoordinateVPDiffusion(int(config["diffusion_steps"]))
    settings = dict(config["local_objective"])
    settings["coordinate_scale_angstrom"] = float(config["coordinate_scale_angstrom"])
    resume_at = len(completed)
    sequence = [
        (int(length), row, int(t))
        for length in config["lengths"]
        for row in by_length[int(length)]
        for t in preflight_timesteps
    ]
    for cell_index, (length, row, timestep_value) in enumerate(sequence[resume_at:], start=resume_at):
        prepared = localization._prepared_reference(source, row)
        clean = prepared["coordinates"].to(device)
        mask = prepared["residue_mask"].to(device)
        continuity = prepared["chain_continuity_mask"].to(device)
        timestep = torch.tensor([timestep_value], dtype=torch.long, device=device)
        generator = torch.Generator(device=device).manual_seed(5173000 + length * 100 + timestep_value + cell_index)
        diffusion_batch = diffusion.make_training_batch(clean, mask, timesteps=timestep, generator=generator)
        model.zero_grad(set_to_none=True)
        prediction = model(
            diffusion_batch.noisy_coordinates, timestep, prepared["lengths"].to(device), mask, continuity
        )["v_prediction"]
        v_loss = uniform_coordinate_v_mse(prediction, diffusion_batch.coordinate_v_target, mask)
        predicted_x0 = diffusion.reconstruct_x0(diffusion_batch.noisy_coordinates, timestep, prediction, mask)
        raw = local_backbone_losses(predicted_x0, clean, mask, continuity, settings, per_structure=True)
        coeff = coefficient_tensor(table, timestep, device)[0].to(prediction.dtype)
        weighted = {name: raw[name] * coeff[index] for index, name in enumerate(LOCAL_TERMS)}
        parameters = tuple(model.parameters())
        # Calibration also differentiates several terms from one forward graph.
        # Keep its pre-existing within-cell graph lifetime explicit.
        v_vector = _gradient_vector(v_loss, parameters, retain_graph=True)
        term_vectors = {name: _gradient_vector(weighted[name], parameters, retain_graph=True) for name in LOCAL_TERMS}
        aux_vector = sum(term_vectors.values())
        total_vector = v_vector + aux_vector
        v_norm = float(torch.linalg.vector_norm(v_vector).detach().cpu())
        term_norms = {
            name: float(torch.linalg.vector_norm(value).detach().cpu()) for name, value in term_vectors.items()
        }
        eps = 1e-30
        term_ratios = {name: value / max(v_norm, eps) for name, value in term_norms.items()}
        term_cosines = {
            name: float(
                (
                    torch.dot(v_vector, term_vectors[name])
                    / (torch.linalg.vector_norm(term_vectors[name]) * torch.linalg.vector_norm(v_vector)).clamp_min(eps)
                )
                .detach()
                .cpu()
            )
            for name in LOCAL_TERMS
        }
        pairwise = {
            left: {
                right: float(
                    (
                        torch.dot(term_vectors[left], term_vectors[right])
                        / (
                            torch.linalg.vector_norm(term_vectors[left]) * torch.linalg.vector_norm(term_vectors[right])
                        ).clamp_min(eps)
                    )
                    .detach()
                    .cpu()
                )
                for right in LOCAL_TERMS
            }
            for left in LOCAL_TERMS
        }
        finite_counts = {
            "v": _finite_counts(v_vector),
            "raw_terms": {
                name: _finite_counts(_gradient_vector(raw[name][0], tuple(model.parameters()), retain_graph=True))
                for name in LOCAL_TERMS
            },
            **{name: _finite_counts(vec) for name, vec in term_vectors.items()},
            "combined_auxiliary": _finite_counts(aux_vector),
            "total": _finite_counts(total_vector),
        }
        identity = {"length": length, "timestep": timestep_value, "sample_id": str(row["sample_id"])}
        cell = {
            "identity": identity,
            "coefficients": {name: float(coeff[i].detach().cpu()) for i, name in enumerate(LOCAL_TERMS)},
            "safety_factor": float(table["timesteps"][str(timestep_value)].get("safety_factor", 1.0)),
            "coefficient_row_sha256": _canonical_sha(
                {name: float(coeff[i].detach().cpu()) for i, name in enumerate(LOCAL_TERMS)}
            ),
            "raw_losses": {name: float(raw[name][0].detach().cpu()) for name in LOCAL_TERMS},
            "weighted_losses": {name: float(weighted[name][0].detach().cpu()) for name in LOCAL_TERMS},
            "v_loss": float(v_loss.detach().cpu()),
            "mask_eligibility": {name: int(raw["denominators"][name][0].detach().cpu()) for name in LOCAL_TERMS},
            "finite_counts": finite_counts,
            "individual_term_v_ratios": term_ratios,
            "combined_auxiliary_v_ratio": float(torch.linalg.vector_norm(aux_vector).detach().cpu()) / max(v_norm, eps),
            "total_v_ratio": float(torch.linalg.vector_norm(total_vector).detach().cpu()) / max(v_norm, eps),
            "term_v_cosines": term_cosines,
            "pairwise_auxiliary_cosines": pairwise,
            "memory": {
                "process_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            },
        }
        cell = _json_finite(cell)
        _append_fsync_jsonl(cells_path, cell)
        completed.append(cell)
    protected_after = verify_protected_evidence(config)
    v3_after = verify_v3_evidence()
    if protected_before != protected_after or v3_before != v3_after:
        raise ValueError("dense-preflight protected hashes changed")
    if len(completed) != cell_count:
        raise ValueError("dense-preflight cell count mismatch")
    failures = []
    for cell in completed:
        gradient_summaries = [value for key, value in cell["finite_counts"].items() if key != "raw_terms"] + list(
            cell["finite_counts"]["raw_terms"].values()
        )
        if any(not value["all_finite"] for value in gradient_summaries):
            failures.append({"identity": cell["identity"], "gate": "finite"})
        if (
            cell["v_loss"] is None
            or not math.isfinite(cell["v_loss"])
            or any(
                value is None or not math.isfinite(value)
                for key in ("raw_losses", "weighted_losses")
                for value in cell[key].values()
            )
        ):
            failures.append({"identity": cell["identity"], "gate": "non_finite_loss"})
        failures.extend({"identity": cell["identity"], "gate": gate} for gate in _upper_safety_gate_failures(cell))
    activity = dense.get("activity_gates", {})
    activity_results = {}
    # At zero noise, near-zero auxiliary gradients are expected when the predicted geometry
    # is already close to its source. Zero remains subject to finiteness and upper safety gates.
    activity_exemptions = activity.get("exempt_timesteps", [activity.get("exempt_endpoint", 499)])
    for timestep_value in _lower_activity_timesteps(preflight_timesteps, activity_exemptions):
        subset = [cell for cell in completed if cell["identity"]["timestep"] == timestep_value]
        if any(
            cell["combined_auxiliary_v_ratio"] is None
            or any(value is None for value in cell["individual_term_v_ratios"].values())
            for cell in subset
        ):
            activity_results[str(timestep_value)] = {"status": "failed_non_finite_cell"}
            failures.append({"identity": {"timestep": timestep_value}, "gate": "activity_non_finite"})
            continue
        core_stats = {
            term: sorted(cell["individual_term_v_ratios"][term] for cell in subset)
            for term in activity.get("core_terms", ("adjacent", "i_plus_2", "i_plus_3", "bond_angle_cosine"))
        }
        tail_stats = {
            term: sorted(cell["individual_term_v_ratios"][term] for cell in subset)
            for term in activity.get("tail_terms", ("discontinuity", "clash"))
        }
        combined = sorted(cell["combined_auxiliary_v_ratio"] for cell in subset)

        def quantile(values: list[float]) -> float:
            return values[min(len(values) - 1, math.ceil(0.9 * len(values)) - 1)]

        activity_results[str(timestep_value)] = {
            "core_p90": {term: quantile(values) for term, values in core_stats.items()},
            "tail_p90": {term: quantile(values) for term, values in tail_stats.items()},
            "combined_median": combined[len(combined) // 2],
        }
        for term, value in activity_results[str(timestep_value)]["core_p90"].items():
            if value < float(activity.get("core_minimum", 0.002)):
                failures.append(
                    {"identity": {"timestep": timestep_value}, "gate": "core_activity", "term": term, "value": value}
                )
        for term, value in activity_results[str(timestep_value)]["tail_p90"].items():
            if value < float(activity.get("tail_minimum", 0.0001)):
                failures.append(
                    {"identity": {"timestep": timestep_value}, "gate": "tail_activity", "term": term, "value": value}
                )
        if activity_results[str(timestep_value)]["combined_median"] < float(
            activity.get("combined_median_minimum", 0.005)
        ):
            failures.append({"identity": {"timestep": timestep_value}, "gate": "combined_activity"})
    ordered = sorted(
        completed,
        key=lambda cell: (
            cell["combined_auxiliary_v_ratio"] if cell["combined_auxiliary_v_ratio"] is not None else -math.inf
        ),
        reverse=True,
    )
    worst_cells = {
        "combined_auxiliary_max": max(
            completed,
            key=lambda cell: (
                cell["combined_auxiliary_v_ratio"] if cell["combined_auxiliary_v_ratio"] is not None else -math.inf
            ),
        )["identity"],
        "total_v_min": min(
            completed, key=lambda cell: cell["total_v_ratio"] if cell["total_v_ratio"] is not None else math.inf
        )["identity"],
        "total_v_max": max(
            completed, key=lambda cell: cell["total_v_ratio"] if cell["total_v_ratio"] is not None else -math.inf
        )["identity"],
        "individual_term_max": {
            term: max(
                completed,
                key=lambda cell: (
                    cell["individual_term_v_ratios"][term]
                    if cell["individual_term_v_ratios"][term] is not None
                    else -math.inf
                ),
            )["identity"]
            for term in LOCAL_TERMS
        },
    }
    by_timestep = {}
    by_length = {}
    for dimension, values in (
        ("timestep", preflight_timesteps),
        ("length", tuple(map(int, config["lengths"]))),
    ):
        summary = by_timestep if dimension == "timestep" else by_length
        for value in values:
            subset = [cell for cell in completed if int(cell["identity"][dimension]) == value]
            summary[str(value)] = {
                "cell_count": len(subset),
                "combined_auxiliary_v_ratio": [cell["combined_auxiliary_v_ratio"] for cell in subset],
                "individual_term_v_ratios": [cell["individual_term_v_ratios"] for cell in subset],
                "total_v_ratio": [cell["total_v_ratio"] for cell in subset],
                "finite_counts": [cell["finite_counts"] for cell in subset],
                "raw_losses": [cell["raw_losses"] for cell in subset],
                "weighted_losses": [cell["weighted_losses"] for cell in subset],
                "mask_eligibility": [cell["mask_eligibility"] for cell in subset],
                "pairwise_auxiliary_cosines": [cell["pairwise_auxiliary_cosines"] for cell in subset],
            }
    schedule_report = [
        {
            "timestep": t,
            "safety_factor": float(table["timesteps"][str(t)].get("safety_factor", 1.0)),
            "v_mse": float(table["timesteps"][str(t)]["v_mse"]),
            "local_terms": dict(table["timesteps"][str(t)]["local_terms"]),
        }
        for t in range(500)
    ]
    report = {
        "version": "e007_phase3i2_dense_preflight_v2"
        if len(preflight_timesteps) == len(PREFLIGHT_V2_TIMESTEPS)
        else "e007_phase3i2_dense_preflight_v1",
        "status": "completed",
        "cell_count": cell_count,
        "optimizer_updates": 0,
        "configuration_sha256": _sha256_file(config_path),
        "coefficient_table_sha256": table["sha256"],
        "v3_hashes": v3_before,
        "protected_hashes": protected_before,
        "selection_panel_sha256": selection["record_sha256"],
        "canonical_panel_identity_sha256": canonical["identity_sha256"],
        "cells": completed,
        "coefficient_and_safety_factor_rows": schedule_report,
        "results_by_timestep": by_timestep,
        "results_by_length": by_length,
        "activity_gate_results": activity_results,
        "worst_cell_identities": {
            **worst_cells,
            "combined_auxiliary_top_five": [cell["identity"] for cell in ordered[:5]],
        },
        "gate_failures": failures,
        "pilot_authorized": False,
        **NON_AUTHORIZING,
    }
    _atomic_json(staging / "report.json", report)
    protocol = {
        "report_sha256": _sha256_file(staging / "report.json"),
        "cell_count": cell_count,
        "optimizer_updates": 0,
        "pilot_authorized": False,
        **NON_AUTHORIZING,
    }
    _atomic_json(staging / "protocol.json", protocol)
    staging.replace(output)
    _fsync_directory(output.parent)
    return report


def publish_dynamic_memory_smoke_result(
    result: dict[str, Any], config_path: str | Path, *, log_path: str | Path | None = None
) -> dict[str, Any]:
    """Validate and atomically publish a completed, non-authorizing smoke record."""
    config_path = Path(config_path)
    config = load_config(config_path)
    suffix = config["dynamic_preflight"]["version"].rsplit("_", 1)[-1]
    default_output = (
        f"reports/experiments/E007_matrix_sequence_cogeneration/local_backbone_repair_dynamic_memory_smoke_{suffix}"
    )
    output = Path(config.get("dynamic_memory_smoke", {}).get("output_dir", default_output))
    if output.as_posix() != default_output:
        raise ValueError("dynamic memory smoke must use its declared smoke-specific path")
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"dynamic memory smoke publication already exists: {output} or {staging}")
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    measurements = result.get("measurements", [])
    if result.get("status") != "passed" or len(measurements) != result.get("repetitions") or len(measurements) < 3:
        raise ValueError("dynamic memory smoke result is incomplete")
    if (
        result.get("optimizer_updates") != 0
        or result.get("checkpoint_sha256") != config["selected_checkpoint"]["sha256"]
    ):
        raise ValueError("dynamic memory smoke result provenance mismatch")
    limits = config["memory"]
    for index, item in enumerate(measurements, 1):
        if (
            item.get("repetition") != index
            or item.get("audit_state_restored") is not True
            or item.get("optimizer_updates") != 0
        ):
            raise ValueError("dynamic memory smoke repetition audit mismatch")
        for measured, limit in (
            ("peak_rss_mib", "maximum_rss_mib"),
            ("peak_cuda_allocated_mib", "maximum_cuda_allocated_mib"),
            ("peak_cuda_reserved_mib", "maximum_cuda_reserved_mib"),
        ):
            if (
                not math.isfinite(float(item[measured]))
                or float(item[measured]) < 0
                or float(item[measured]) > float(limits[limit])
            ):
                raise ValueError(f"dynamic memory smoke {measured} limit failed")
    for key in ("current_cuda_allocated_mib", "current_cuda_reserved_mib"):
        values = [float(item[key]) for item in measurements]
        if any(not math.isfinite(value) or value < 0 for value in values) or max(values) - min(values) > 64.0:
            raise ValueError(f"dynamic memory smoke {key} plateau failed")
    for kind in ("allocated", "reserved"):
        peak = max(float(item[f"peak_cuda_{kind}_mib"]) for item in measurements)
        if f"run_peak_cuda_{kind}_mib" in result and float(result[f"run_peak_cuda_{kind}_mib"]) != peak:
            raise ValueError(f"dynamic memory smoke run peak must be the maximum observed {kind} phase peak")
    if any(result.get(key) is not False for key in NON_AUTHORIZING):
        raise ValueError("dynamic memory smoke authorization mismatch")
    log_pin = None
    if log_path is not None:
        log_path = Path(log_path)
        log_pin = {"path": str(log_path), "sha256": _sha256_file(log_path)}
    report = {
        **result,
        "version": f"e007_phase3i2_dynamic_memory_smoke_{suffix}",
        "record_type": "retrospective_preserved_log" if log_pin else "direct_execution",
        "configuration_sha256": _sha256_file(config_path),
        "memory_limits_mib": limits,
        "pilot_authorized": False,
        **NON_AUTHORIZING,
    }
    staging.mkdir()
    try:
        _atomic_json(staging / "report.json", report)
        protocol = {
            "version": report["version"],
            "status": "passed",
            "report_sha256": _sha256_file(staging / "report.json"),
            "repetitions": len(measurements),
            "optimizer_updates": 0,
            "publication": "atomic_directory_rename",
            "pilot_authorized": False,
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "protocol.json", protocol)
        inventory = {
            "inputs": {
                "configuration": {"path": str(config_path), "sha256": _sha256_file(config_path)},
                "checkpoint": config["selected_checkpoint"],
                "coefficient_schedule": {
                    "path": config["local_objective"]["coefficient_table_path"],
                    "sha256": _sha256_file(Path(config["local_objective"]["coefficient_table_path"])),
                },
                **({"log": log_pin} if log_pin else {}),
            },
            "outputs": {name: _sha256_file(staging / name) for name in ("report.json", "protocol.json")},
            "pilot_authorized": False,
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "artifact_inventory.json", inventory)
        _atomic_json(
            staging / "heartbeat.json",
            {
                "status": "completed",
                "report_sha256": protocol["report_sha256"],
                "optimizer_updates": 0,
                "pilot_authorized": False,
                **NON_AUTHORIZING,
            },
        )
        staging.replace(output)
        _fsync_directory(output.parent)
    except Exception:
        # Keep failed staging available for inspection; never replace a prior publication.
        raise
    return report


def run_dynamic_memory_smoke(
    config_path: str | Path, *, repetitions: int = 3, multi_cell: bool = False
) -> dict[str, Any]:
    """Bounded no-update length-500 audit lifecycle smoke from the pinned step-9000 model."""
    if repetitions < 3 or repetitions > 5:
        raise ValueError("dynamic memory smoke repetitions must be between three and five")
    config = load_config(config_path)
    validate_dynamic_memory_smoke_contract(config_path)
    resolved = _resolve_reviewed_coordinate_source(config)
    import gc
    import resource

    import torch

    from protein_distance_diffusion.evaluation import e007_denoiser_sampler_localization as localization
    from protein_distance_diffusion.models.coordinate_equivariance import coordinate_model_execution_context
    from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import EquivariantPairCoordinateUNet
    from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion

    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("dynamic memory smoke requires configured CUDA")
    source_config = resolved["source"]
    audit_source = resolved["audit_source"]
    localization.verify_prerequisites(audit_source, full=True)
    compact, _ = localization.select_validation_panel(audit_source)
    reconstructed, _ = localization.reconstruct_authoritative_panel(audit_source, compact)
    lengths = (64, 128, 500) if multi_cell else (500,) * repetitions
    timesteps = (25, 250, 499) if multi_cell else (499,) * repetitions
    if multi_cell and repetitions != 3:
        raise ValueError("multi-cell smoke requires exactly three cells")
    rows = {}
    for length in set(lengths):
        required_id = config["dynamic_preflight"]["audit_sample_ids_by_length"][str(length)]
        matches = [
            item["canonical_row"]
            for item in reconstructed
            if int(item["selection"]["target_length"]) == length
            and str(item["canonical_row"]["sample_id"]) == required_id
        ]
        if len(matches) != 1:
            raise ValueError(f"dynamic memory smoke panel identity did not reconstruct uniquely: {length}")
        rows[length] = matches[0]
    table = load_coefficient_table(config["local_objective"]["coefficient_table_path"])
    checkpoint = _resolve_reviewed_coordinate_source(config, load_checkpoint=True)["checkpoint"]
    measurements = []
    with coordinate_model_execution_context(config["numerics"], device):
        model = EquivariantPairCoordinateUNet(**source_config["model"]).to(device).eval()
        model.load_state_dict(checkpoint["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        diffusion = CoordinateVPDiffusion(int(config["diffusion_steps"]))
        telemetry = CudaMemoryTelemetry(torch.cuda, device)
        for repetition in range(repetitions):
            length = lengths[repetition]
            timestep_value = timesteps[repetition]
            optimizer.zero_grad(set_to_none=True)
            result = zero_update_drift_audit(
                model,
                optimizer,
                scheduler,
                diffusion,
                [(length, rows[length])],
                config,
                table,
                device,
                update=0,
                data_cursor=0,
                audit_timesteps=[timestep_value],
                update_identity={
                    "smoke_repetition": repetition,
                    "checkpoint_sha256": config["selected_checkpoint"]["sha256"],
                },
            )
            if not result["pass"]:
                raise RuntimeError(
                    f"dynamic memory smoke audit failed at repetition {repetition}: {result['violations']}"
                )
            del result
            gc.collect()
            torch.cuda.empty_cache()
            memory = telemetry.end_phase()
            current_allocated = memory["current_cuda_allocated_mib"]
            current_reserved = memory["current_cuda_reserved_mib"]
            peak_allocated = memory["phase_peak_cuda_allocated_mib"]
            peak_reserved = memory["phase_peak_cuda_reserved_mib"]
            limits = config["memory"]
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            if peak_rss > float(limits["maximum_rss_mib"]):
                raise MemoryError(f"dynamic memory smoke RSS exceeded: {peak_rss:.1f} MiB")
            if peak_allocated > float(limits["maximum_cuda_allocated_mib"]):
                raise MemoryError(f"dynamic memory smoke allocated peak exceeded: {peak_allocated:.1f} MiB")
            if peak_reserved > float(limits["maximum_cuda_reserved_mib"]):
                raise MemoryError(f"dynamic memory smoke reserved peak exceeded: {peak_reserved:.1f} MiB")
            measurements.append(
                {
                    "repetition": repetition + 1,
                    "length": length,
                    "timestep": timestep_value,
                    "sample_id": str(rows[length]["sample_id"]),
                    "current_cuda_allocated_mib": current_allocated,
                    "current_cuda_reserved_mib": current_reserved,
                    "peak_cuda_allocated_mib": peak_allocated,
                    "peak_cuda_reserved_mib": peak_reserved,
                    "memory_telemetry": memory,
                    "peak_rss_mib": peak_rss,
                    "audit_state_restored": True,
                    "optimizer_updates": 0,
                }
            )
        for key in ("current_cuda_allocated_mib", "current_cuda_reserved_mib"):
            values = [item[key] for item in measurements]
            if max(values) - min(values) > 64.0:
                raise MemoryError(f"dynamic memory smoke did not reach a stable {key} plateau: {values}")
    result = {
        "version": "e007_phase3i2_dynamic_memory_smoke_v1",
        "status": "passed",
        "checkpoint_sha256": config["selected_checkpoint"]["sha256"],
        "length": 500 if not multi_cell else None,
        "smoke_kind": "multi_cell_lifecycle" if multi_cell else "length_500_memory",
        "sampling_performed": False,
        "repetitions": repetitions,
        "optimizer_updates": 0,
        "measurements": measurements,
        "run_peak_cuda_allocated_mib": max(item["peak_cuda_allocated_mib"] for item in measurements),
        "run_peak_cuda_reserved_mib": max(item["peak_cuda_reserved_mib"] for item in measurements),
        "stable_plateau_tolerance_mib": 64.0,
        **NON_AUTHORIZING,
    }
    return publish_dynamic_memory_smoke_result(result, config_path)


def plan_local_backbone_repair(config_path: str | Path) -> dict[str, Any]:
    """Validate immutable metadata only; never import torch or create output."""
    path = Path(config_path)
    config = load_config(path)
    if config.get("output_dir", "").endswith("reviewed_v6_v4"):
        validate_pilot_contract(path)
    protected = verify_protected_evidence(config)
    v3_protected = verify_v3_evidence()
    table = (
        load_coefficient_table(config["local_objective"]["coefficient_table_path"])
        if config["local_objective"].get("coefficient_table_path")
        else None
    )
    dynamic = config.get("dynamic_preflight")
    samples = int(config["pilot"]["samples_per_length_per_sampler"])
    boundaries = len(config["pilot"]["evaluation_updates"])
    guidance_variants = len(config["guidance"]["strengths"])
    sampling_records = (
        0 if dynamic else len(ARMS) * boundaries * len(config["lengths"]) * samples * (1 + guidance_variants)
    )
    selected = config["local_objective"].get("selected_coefficient_set")
    return {
        "version": VERSION,
        "mode": "plan_only",
        "status": "planned",
        "configuration_sha256": _sha256_file(path),
        "protected_evidence": protected,
        "v3_protected_evidence": v3_protected,
        "factorial_cells": [] if dynamic else [f"{arm}/{sampler}" for arm in ARMS for sampler in SAMPLERS],
        "training_arms": list(config["arms"]) if dynamic else list(ARMS),
        "samplers": [] if dynamic else list(SAMPLERS),
        "guidance_strengths_reported_separately": [] if dynamic else list(config["guidance"]["strengths"]),
        "optimizer_updates_per_arm": int(dynamic["updates"]) if dynamic else 500,
        "evaluation_updates": list(dynamic["audit_updates"])
        if dynamic
        else list(config["pilot"]["evaluation_updates"]),
        "samples_per_length_per_sampler": 0 if dynamic else samples,
        "planned_sampling_records_including_each_guidance_strength": sampling_records,
        "runtime_estimate": (
            {
                "forward_count": int(dynamic["planned_forward_count"]),
                "backward_count": int(dynamic["planned_backward_count"]),
                "estimated_wall_seconds": int(dynamic["estimated_runtime_seconds"]),
            }
            if dynamic
            else dict(config["runtime"])
        ),
        "memory_limits_mib": dict(config["memory"]),
        "calibration_cells_per_coefficient_set": (
            0 if dynamic else len(config["calibration"]["lengths"]) * len(config["calibration"]["timesteps"])
        ),
        "coefficient_candidates": config["local_objective"]["candidate_coefficient_sets"],
        "selected_coefficient_set": selected,
        "coefficient_table_sha256": table["sha256"] if table else None,
        "coefficient_table_rows": len(table["timesteps"]) if table else 0,
        "planned_dense_preflight_cells": (
            len(dynamic["audit_updates"]) * len(dynamic["lengths"]) * len(dynamic["timesteps"])
            if dynamic
            else len(config["lengths"])
            * len(config.get("dense_preflight", {}).get("timesteps", PREFLIGHT_TIMESTEPS))
            * int(config.get("dense_preflight", {}).get("structures_per_length", 2))
        ),
        "planned_forward_count": int(dynamic["planned_forward_count"]) if dynamic else None,
        "planned_backward_count": int(dynamic["planned_backward_count"]) if dynamic else None,
        "estimated_runtime_seconds": int(dynamic["estimated_runtime_seconds"]) if dynamic else None,
        "pilot_startup_ready": False if dynamic else _calibration_ready(config, raise_on_failure=False),
        "pilot_block_reason": None
        if selected is not None
        else "reviewed gradient calibration and selected coefficient set are not pinned",
        "initialization": (
            "immutable step-9000 weights; fresh v_only optimizer/scheduler; 10 updates"
            if dynamic
            else "identical step-9000 model weights; fresh identical optimizer/scheduler per arm"
        ),
        "native_handedness_unresolved": True,
        "chirality_objective_added": False,
        "model_created": False,
        "checkpoint_loaded": False,
        "cuda_initialized": False,
        "optimizer_created": False,
        "dataset_scanned": False,
        "sampling_performed": False,
        "output_created": False,
        **NON_AUTHORIZING,
    }


def _valid_offset_mask(residue_mask: Any, continuity: Any, offset: int) -> Any:
    mask = residue_mask[:, :-offset] & residue_mask[:, offset:]
    for shift in range(offset):
        mask = mask & continuity[:, shift : continuity.shape[1] - (offset - shift - 1)]
    return mask


def _per_structure_mean(values: Any, mask: Any, *, keep_structures: bool = False) -> tuple[Any, Any]:
    import torch

    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    masked_values = torch.where(weights.bool(), values, torch.zeros((), dtype=values.dtype, device=values.device))
    numerators = masked_values.reshape(values.shape[0], -1).sum(dim=1)
    denominators = weights.expand_as(values).reshape(values.shape[0], -1).sum(dim=1)
    available = denominators > 0
    means = numerators / denominators.clamp_min(1)
    zero = values.sum() * 0.0
    if keep_structures:
        return torch.where(available, means, torch.zeros_like(means)), denominators
    return (means[available].mean() if bool(available.any()) else zero), denominators


def _distance_loss(predicted: Any, target: Any, mask: Any, offset: int, normalizer: float) -> tuple[Any, Any]:
    import torch

    pred = torch.linalg.vector_norm(predicted[:, offset:] - predicted[:, :-offset], dim=-1)
    native = torch.linalg.vector_norm(target[:, offset:] - target[:, :-offset], dim=-1)
    return _per_structure_mean(((pred - native) / normalizer).square(), mask)


def _angle_cosines(coordinates: Any) -> Any:
    import torch

    left = coordinates[:, :-2] - coordinates[:, 1:-1]
    right = coordinates[:, 2:] - coordinates[:, 1:-1]
    denominator = torch.linalg.vector_norm(left, dim=-1) * torch.linalg.vector_norm(right, dim=-1)
    return (left * right).sum(dim=-1) / denominator.clamp_min(torch.finfo(coordinates.dtype).eps)


def _bounded_clash_pairs(length: int, minimum_separation: int, maximum_pairs: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for offset in range(minimum_separation, length):
        for left in range(length - offset):
            pairs.append((left, left + offset))
            if len(pairs) == maximum_pairs:
                return pairs
    return pairs


def local_backbone_losses(
    predicted_x0: Any,
    native_x0: Any,
    residue_mask: Any,
    continuity_mask: Any,
    settings: Mapping[str, Any],
    *,
    per_structure: bool = False,
) -> dict[str, Any]:
    """Compute O(3)-invariant, masked, per-protein normalized local losses."""
    import torch
    import torch.nn.functional as functional

    if predicted_x0.shape != native_x0.shape or predicted_x0.ndim != 3:
        raise ValueError("E007 Phase-3I.2 coordinate shape contradiction")
    if residue_mask.shape != predicted_x0.shape[:2] or continuity_mask.shape != (
        predicted_x0.shape[0],
        predicted_x0.shape[1] - 1,
    ):
        raise ValueError("E007 Phase-3I.2 mask shape contradiction")
    coordinate_scale = float(settings.get("coordinate_scale_angstrom", 12.22820347644835))
    scale = float(settings["distance_normalizer_angstrom"]) / coordinate_scale
    masks = {offset: _valid_offset_mask(residue_mask.bool(), continuity_mask.bool(), offset) for offset in (1, 2, 3)}
    output: dict[str, Any] = {}
    counts: dict[str, Any] = {}
    for name, offset in (("adjacent", 1), ("i_plus_2", 2), ("i_plus_3", 3)):
        if per_structure:
            pred = torch.linalg.vector_norm(predicted_x0[:, offset:] - predicted_x0[:, :-offset], dim=-1)
            native = torch.linalg.vector_norm(native_x0[:, offset:] - native_x0[:, :-offset], dim=-1)
            output[name], counts[name] = _per_structure_mean(
                ((pred - native) / scale).square(), masks[offset], keep_structures=True
            )
        else:
            output[name], counts[name] = _distance_loss(predicted_x0, native_x0, masks[offset], offset, scale)
    predicted_cosine = _angle_cosines(predicted_x0)
    native_cosine = _angle_cosines(native_x0)
    output["bond_angle_cosine"], counts["bond_angle_cosine"] = _per_structure_mean(
        (predicted_cosine - native_cosine).square(), masks[2], keep_structures=per_structure
    )
    adjacent = torch.linalg.vector_norm(predicted_x0[:, 1:] - predicted_x0[:, :-1], dim=-1)
    threshold = float(settings["discontinuity_threshold_angstrom"]) / coordinate_scale
    beta = float(settings["smooth_tail_beta"])
    tail = functional.softplus((adjacent - threshold) * beta).square() / (beta * beta)
    output["discontinuity"], counts["discontinuity"] = _per_structure_mean(
        tail, masks[1], keep_structures=per_structure
    )

    clash_values = []
    clash_masks = []
    for batch_index in range(predicted_x0.shape[0]):
        pairs = _bounded_clash_pairs(
            predicted_x0.shape[1],
            int(settings["clash_minimum_sequence_separation"]),
            int(settings["maximum_clash_pairs_per_structure"]),
        )
        if not pairs:
            continue
        left = torch.tensor([pair[0] for pair in pairs], device=predicted_x0.device)
        right = torch.tensor([pair[1] for pair in pairs], device=predicted_x0.device)
        valid = residue_mask[batch_index, left] & residue_mask[batch_index, right]
        distance = torch.linalg.vector_norm(predicted_x0[batch_index, left] - predicted_x0[batch_index, right], dim=-1)
        clash = functional.softplus(
            (float(settings["clash_threshold_angstrom"]) / coordinate_scale - distance) * beta
        ).square() / (beta * beta)
        clash_values.append(clash)
        clash_masks.append(valid)
    if clash_values:
        width = max(value.numel() for value in clash_values)
        padded_values = predicted_x0.new_zeros((len(clash_values), width))
        padded_masks = torch.zeros((len(clash_values), width), dtype=torch.bool, device=predicted_x0.device)
        for index, (values, valid) in enumerate(zip(clash_values, clash_masks, strict=True)):
            padded_values[index, : values.numel()] = values
            padded_masks[index, : values.numel()] = valid
        output["clash"], counts["clash"] = _per_structure_mean(
            padded_values, padded_masks, keep_structures=per_structure
        )
    else:
        output["clash"] = predicted_x0.sum() * 0.0
        counts["clash"] = predicted_x0.new_zeros((predicted_x0.shape[0],))
    output["denominators"] = counts
    return output


def weighted_local_objective(
    losses: Mapping[str, Any], weights: Mapping[str, Any], *, reduce_batch: bool = True
) -> tuple[Any, dict[str, Any]]:
    weighted = {
        name: losses[name] * (weights[name] if hasattr(weights[name], "shape") else float(weights[name]))
        for name in LOCAL_TERMS
    }
    per_structure_total = sum(weighted.values())
    total = (
        per_structure_total.mean() if reduce_batch and getattr(per_structure_total, "ndim", 0) else per_structure_total
    )
    return total, weighted


def local_guidance_energy(
    coordinates: Any, residue_mask: Any, continuity_mask: Any, settings: Mapping[str, Any]
) -> Any:
    """Reference-free generic C-alpha-chain energy; no native coordinates enter."""
    import torch
    import torch.nn.functional as functional

    scale = float(settings.get("coordinate_scale_angstrom", 12.22820347644835))
    masks = {offset: _valid_offset_mask(residue_mask.bool(), continuity_mask.bool(), offset) for offset in (1, 2, 3)}
    terms = []
    target_fields = (
        (1, "adjacent_target_angstrom"),
        (2, "i_plus_2_target_angstrom"),
        (3, "i_plus_3_target_angstrom"),
    )
    for offset, key in target_fields:
        distances = torch.linalg.vector_norm(coordinates[:, offset:] - coordinates[:, :-offset], dim=-1)
        term, _ = _per_structure_mean((distances - float(settings[key]) / scale).square(), masks[offset])
        terms.append(term)
    target_cosine = math.cos(math.radians(float(settings["bond_angle_degrees"])))
    angle, _ = _per_structure_mean((_angle_cosines(coordinates) - target_cosine).square(), masks[2])
    terms.append(angle)
    adjacent = torch.linalg.vector_norm(coordinates[:, 1:] - coordinates[:, :-1], dim=-1)
    discontinuity, _ = _per_structure_mean(
        functional.softplus(adjacent - float(settings["discontinuity_threshold_angstrom"]) / scale).square(),
        masks[1],
    )
    terms.append(discontinuity)
    # Reuse the bounded clash implementation against a detached placeholder; only the clash term is reference-free.
    clash_settings = {
        "distance_normalizer_angstrom": scale,
        "discontinuity_threshold_angstrom": settings["discontinuity_threshold_angstrom"],
        "clash_threshold_angstrom": settings["clash_threshold_angstrom"],
        "clash_minimum_sequence_separation": 3,
        "maximum_clash_pairs_per_structure": settings["maximum_clash_pairs_per_structure"],
        "smooth_tail_beta": 4.0,
    }
    terms.append(
        local_backbone_losses(coordinates, coordinates.detach(), residue_mask, continuity_mask, clash_settings)["clash"]
    )
    return sum(terms)


def apply_local_guidance(
    coordinates: Any,
    residue_mask: Any,
    continuity_mask: Any,
    settings: Mapping[str, Any],
    *,
    strength: float,
) -> tuple[Any, dict[str, float]]:
    """Apply bounded gradient correction while preserving centering and zero padding."""
    import torch

    from protein_distance_diffusion.training.coordinate_diffusion import center_coordinates

    if strength <= 0 or not math.isfinite(strength):
        raise ValueError("E007 Phase-3I.2 guidance strength must be finite and positive")
    current = coordinates
    initial_energy = None
    maximum_step = 0.0
    for _ in range(int(settings["correction_steps"])):
        differentiable = current.detach().requires_grad_(True)
        energy = local_guidance_energy(differentiable, residue_mask, continuity_mask, settings)
        gradient = torch.autograd.grad(energy, differentiable)[0]
        update = -strength * gradient
        limit = float(settings["maximum_displacement_angstrom"]) / float(
            settings.get("coordinate_scale_angstrom", 12.22820347644835)
        )
        norm = torch.linalg.vector_norm(update, dim=-1, keepdim=True)
        update = update * (limit / norm.clamp_min(limit)).clamp_max(1.0)
        update = update * residue_mask[..., None].to(update.dtype)
        current = center_coordinates(differentiable + update, residue_mask).detach()
        initial_energy = float(energy.detach()) if initial_energy is None else initial_energy
        maximum_step = max(maximum_step, float(torch.linalg.vector_norm(update, dim=-1).max().detach()))
    final_energy = float(local_guidance_energy(current, residue_mask, continuity_mask, settings).detach())
    return current, {
        "initial_energy": float(initial_energy),
        "final_energy": final_energy,
        "maximum_normalized_displacement": maximum_step,
    }


def gradient_profile(v_loss: Any, local_losses: Mapping[str, Any], parameters: Sequence[Any]) -> dict[str, Any]:
    """Return separate norms/cosines without mutating parameter gradients."""
    import torch

    def vector(loss: Any) -> Any:
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        pieces = [gradient.reshape(-1) for gradient in gradients if gradient is not None]
        return torch.cat(pieces) if pieces else loss.new_zeros((0,))

    def finite_summary(value: Any) -> dict[str, Any]:
        finite_mask = torch.isfinite(value)
        total_count = int(value.numel())
        finite_count = int(finite_mask.sum().item())
        return {
            "finite_count": finite_count,
            "total_count": total_count,
            "non_finite_count": total_count - finite_count,
            "all_finite": bool(torch.isfinite(value).all().item()),
            # Descriptive only: compute in Python/float64, never use as a gate.
            "finite_gradient_coverage": float(finite_count / total_count) if total_count else 1.0,
        }

    base = vector(v_loss)
    base_norm = torch.linalg.vector_norm(base)
    result: dict[str, Any] = {
        "v_gradient_norm": float(base_norm.detach()),
        "v_finite_gradient_coverage": finite_summary(base)["finite_gradient_coverage"],
        "v_finite_counts": finite_summary(base),
        "terms": {},
    }
    for name in LOCAL_TERMS:
        auxiliary = vector(local_losses[name])
        norm = torch.linalg.vector_norm(auxiliary)
        cosine = torch.dot(base, auxiliary) / (base_norm * norm).clamp_min(torch.finfo(base.dtype).eps)
        result["terms"][name] = {
            "gradient_norm": float(norm.detach()),
            "gradient_norm_to_v": float((norm / base_norm.clamp_min(torch.finfo(base.dtype).eps)).detach()),
            "cosine_with_v": float(cosine.detach()),
            **finite_summary(auxiliary),
        }
    return result


def paired_update_identity(seed: int, update: int, sample_ids: Sequence[str]) -> dict[str, Any]:
    identity = {"seed": int(seed), "update": int(update), "sample_ids": list(sample_ids)}
    identity["sha256"] = _canonical_sha(identity)
    return identity


def _calibration_ready(config: Mapping[str, Any], *, raise_on_failure: bool) -> bool:
    selected = config["local_objective"].get("selected_coefficient_set")
    calibration = config["calibration"]
    ready = (
        selected in config["local_objective"]["candidate_coefficient_sets"]
        and calibration.get("reviewed") is True
        and isinstance(calibration.get("report_path"), str)
        and isinstance(calibration.get("report_sha256"), str)
    )
    if ready:
        report_path = Path(calibration["report_path"])
        ready = report_path.is_file() and _sha256_file(report_path) == calibration["report_sha256"]
        if ready:
            report = json.loads(report_path.read_text())
            ready = (
                report.get("status") == "completed"
                and report.get("selected_coefficient_set") == selected
                and (
                    report.get("authorizes_training") is False
                    or (
                        "reviewed_v6_decision_path" in config
                        and report.get("review_required") is True
                        and report.get("calibration_authorizes_pilot") is False
                    )
                )
            )
    if not ready and raise_on_failure:
        raise ValueError("E007 Phase-3I.2 pilot requires reviewed hash-pinned gradient calibration")
    return ready


def _append_fsync_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _reconcile_metrics_to_checkpoint(path: Path, committed_update: int) -> dict[str, int]:
    """Discard only a trailing metric row that lacks a committed checkpoint."""
    if not path.exists():
        if committed_update:
            raise ValueError("E007 Phase-3I.2 checkpoint has no matching metric history")
        return {"retained": 0, "discarded_uncommitted": 0}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    updates = [int(row["optimizer_update"]) for row in rows]
    if updates != list(range(1, len(updates) + 1)):
        raise ValueError("E007 Phase-3I.2 metric history is duplicated or out of order")
    if len(rows) < committed_update:
        raise ValueError("E007 Phase-3I.2 metric history trails the committed checkpoint")
    discarded = len(rows) - committed_update
    if discarded > 1:
        raise ValueError("E007 Phase-3I.2 has multiple uncommitted metric rows")
    retained = rows[:committed_update]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in retained:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)
    return {"retained": len(retained), "discarded_uncommitted": discarded}


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        torch.save(dict(payload), stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def _rng_state() -> dict[str, Any]:
    import random

    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    import random

    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _paired_schedule(
    source: Mapping[str, Any], selected: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    from protein_distance_diffusion.training import e007_coordinate_real_pilot as phase3f

    cursors = {name: 0 for name in selected}
    schedule = []
    for update in range(1, 501):
        stratum = phase3f.update_stratum(update, source["length_strata"])
        size = int(phase3f._regime_for_stratum(stratum, source)["physical_batch_size"])
        start = cursors[stratum]
        stop = start + size
        rows = selected[stratum][start:stop]
        if len(rows) != size:
            raise RuntimeError("E007 Phase-3I.2 deterministic training schedule exhausted")
        sample_ids = [str(row["sample_id"]) for row in rows]
        identity = paired_update_identity(int(source["seed"]), update, sample_ids)
        schedule.append(
            {
                "update": update,
                "stratum": stratum,
                "start": start,
                "stop": stop,
                "sample_ids": sample_ids,
                "identity_sha256": identity["sha256"],
            }
        )
        cursors[stratum] = stop
    return schedule


def _recovery_payload(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    update: int,
    samples: int,
    residues: int,
    config_path: Path,
    protected: Mapping[str, str],
    arm: str,
    schedule_hash: str,
    coefficient_table_sha256: str = "legacy_unpinned",
    completed_audit_updates: Sequence[int] = (),
    audit_boundaries: Sequence[int] | None = None,
) -> dict[str, Any]:
    required_audits = tuple(map(int, audit_boundaries or GRADIENT_AUDIT_UPDATES))
    audit_set = set(map(int, completed_audit_updates))
    next_audit = next(
        (boundary for boundary in required_audits if boundary >= update and boundary not in audit_set), None
    )
    return {
        "version": VERSION,
        "configuration_sha256": _sha256_file(config_path),
        "protected_hashes_sha256": _canonical_sha(protected),
        "protected_hashes": dict(protected),
        "arm": arm,
        "paired_schedule_sha256": schedule_hash,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_state": _rng_state(),
        "optimizer_update": update,
        "sampler_cursor": update,
        "sample_cursor": update,
        "next_required_audit_boundary": next_audit,
        "completed_audit_updates": sorted(audit_set),
        "coefficient_table_sha256": coefficient_table_sha256,
        "samples_processed": samples,
        "valid_residues_processed": residues,
        "successful_optimizer_boundary": True,
        **NON_AUTHORIZING,
    }


def _validate_recovery_state(
    state: Mapping[str, Any],
    config_path: Path,
    protected: Mapping[str, str],
    arm: str,
    schedule_hash: str,
    coefficient_table_sha256: str | None = None,
    audit_boundaries: Sequence[int] | None = None,
) -> None:
    required_audits = tuple(map(int, audit_boundaries or GRADIENT_AUDIT_UPDATES))
    expected = {
        "version": VERSION,
        "configuration_sha256": _sha256_file(config_path),
        "protected_hashes_sha256": _canonical_sha(protected),
        "arm": arm,
        "paired_schedule_sha256": schedule_hash,
        "successful_optimizer_boundary": True,
    }
    if coefficient_table_sha256 is not None:
        expected["coefficient_table_sha256"] = coefficient_table_sha256
        expected["protected_hashes"] = dict(protected)
    contradictions = {key: (state.get(key), value) for key, value in expected.items() if state.get(key) != value}
    if contradictions:
        raise ValueError(f"E007 Phase-3I.2 recovery identity contradiction: {contradictions}")
    if "next_required_audit_boundary" in state:
        update = int(state["optimizer_update"])
        completed = set(map(int, state.get("completed_audit_updates", ())))
        next_boundary = next(
            (boundary for boundary in required_audits if boundary >= update and boundary not in completed), None
        )
        if state.get("next_required_audit_boundary") != next_boundary:
            raise ValueError("E007 Phase-3I.2 recovery next-audit boundary contradiction")


def sample_with_local_guidance(
    model: Any,
    diffusion: Any,
    *,
    length: int,
    seed: int,
    device: Any,
    guidance: Mapping[str, Any] | None,
    strength: float | None,
) -> Any:
    """Use the canonical reverse update, optionally followed by local correction."""
    import torch

    from protein_distance_diffusion.training.coordinate_diffusion import centered_coordinate_noise

    generator = torch.Generator(device=device).manual_seed(seed)
    residue_mask = torch.ones((1, length), dtype=torch.bool, device=device)
    continuity = torch.ones((1, length - 1), dtype=torch.bool, device=device)
    lengths = torch.tensor([length], dtype=torch.long, device=device)
    coordinates = centered_coordinate_noise(
        torch.empty((1, length, 3), device=device), residue_mask, generator=generator
    )
    for step in range(diffusion.timesteps - 1, -1, -1):
        timestep = torch.tensor([step], dtype=torch.long, device=device)
        with torch.no_grad():
            prediction = model(coordinates, timestep, lengths, residue_mask, continuity)["v_prediction"]
            coordinates, _, _ = diffusion.deterministic_reverse_step(coordinates, timestep, prediction, residue_mask)
        if guidance is not None:
            coordinates, _ = apply_local_guidance(
                coordinates, residue_mask, continuity, guidance, strength=float(strength)
            )
    return coordinates[0].detach()


def _bounded_evaluation(
    model: Any,
    diffusion: Any,
    validation: Mapping[str, Sequence[Mapping[str, Any]]],
    source: Mapping[str, Any],
    config: Mapping[str, Any],
    device: Any,
) -> dict[str, Any]:
    from protein_distance_diffusion.evaluation import e007_geometry_generator_capability as phase3i
    from protein_distance_diffusion.training import e007_coordinate_real_pilot as phase3f

    model.eval()
    denoising = phase3f._evaluate_panel(model, diffusion, validation, source, device, seed_offset=3916000)
    phase3i_config = phase3i.load_config("configs/e007_geometry_generator_capability_audit_v1.yaml")
    records = []
    reference_records = []
    from protein_distance_diffusion.training.e007_coordinate_real_loader_smoke import prepare_coordinate_batch

    for rows in validation.values():
        for row in rows:
            prepared = prepare_coordinate_batch(
                [dict(row)], float(config["coordinate_scale_angstrom"]), int(source["expected_downsample_factor"])
            )
            length = int(prepared["lengths"][0])
            metrics, _ = phase3i.geometry_metrics(
                prepared["coordinates"][0, :length].double().numpy() * float(config["coordinate_scale_angstrom"]),
                source="identity_30_clean_validation",
                sample_id=str(row["sample_id"]),
                length=length,
                config=phase3i_config,
            )
            reference_records.append(metrics)
    sample_count = int(config["pilot"]["samples_per_length_per_sampler"])
    guidance = dict(config["guidance"])
    guidance["coordinate_scale_angstrom"] = float(config["coordinate_scale_angstrom"])
    for length in map(int, config["lengths"]):
        for index in range(sample_count):
            seed = 3917000 + length * 100 + index
            variants = [("native_reverse", None)] + [
                ("local_guided_reverse", float(strength)) for strength in config["guidance"]["strengths"]
            ]
            for sampler, strength in variants:
                coordinates = sample_with_local_guidance(
                    model,
                    diffusion,
                    length=length,
                    seed=seed,
                    device=device,
                    guidance=guidance if sampler == "local_guided_reverse" else None,
                    strength=strength,
                )
                metrics, _ = phase3i.geometry_metrics(
                    coordinates.cpu().double().numpy() * float(config["coordinate_scale_angstrom"]),
                    source="generated",
                    sample_id=f"N{length}_{index}",
                    length=length,
                    config=phase3i_config,
                )
                records.append(
                    {
                        "sampler": sampler,
                        "guidance_strength": strength,
                        "requested_length": length,
                        "sample_index": index,
                        "seed": seed,
                        **metrics,
                    }
                )
                del coordinates
                gc.collect()
                if getattr(device, "type", None) == "cuda":
                    import torch

                    torch.cuda.empty_cache()
    hashes = [_canonical_sha(record) for record in records]
    descriptors = (
        "adjacent_distance_rmse_to_3_8_angstrom",
        "discontinuity_fraction",
        "clash_fraction",
        "radius_of_gyration_angstrom",
        "asphericity",
        "contact_density_8a",
        "contact_order_8a",
        "contact_graph_component_count_8a",
        "contact_graph_largest_component_fraction_8a",
    )
    return {
        "denoising": denoising,
        "sampling_records": records,
        "sampling_record_count": len(records),
        "duplicate_record_fraction": 1.0 - len(set(hashes)) / max(len(hashes), 1),
        "distribution_comparison": _distribution_summary(records, reference_records, descriptors),
        "descriptor_space_dispersion": _descriptor_dispersion(records, descriptors),
    }


def _distribution_summary(
    generated: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> dict[str, Any]:
    import numpy as np

    result = {}
    for field in fields:
        candidate = np.asarray([float(row[field]) for row in generated], dtype=np.float64)
        native = np.asarray([float(row[field]) for row in references], dtype=np.float64)
        result[field] = {
            "generated_quantiles": [float(x) for x in np.quantile(candidate, [0.05, 0.5, 0.95])],
            "reference_quantiles": [float(x) for x in np.quantile(native, [0.05, 0.5, 0.95])],
            "generated_mean_minus_reference_mean": float(candidate.mean() - native.mean()),
        }
    return result


def _descriptor_dispersion(records: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> dict[str, Any]:
    import numpy as np

    values = np.asarray([[float(row[field]) for field in fields] for row in records], dtype=np.float64)
    scale = values.std(axis=0)
    standardized = (values - values.mean(axis=0)) / np.where(scale > 0, scale, 1.0)
    if len(values) < 2:
        pairwise = np.asarray([], dtype=np.float64)
    else:
        pairwise = np.linalg.norm(standardized[:, None] - standardized[None, :], axis=-1)[
            np.triu_indices(len(values), 1)
        ]
    return {
        "record_count": len(records),
        "descriptor_fields": list(fields),
        "mean_pairwise_distance": float(pairwise.mean()) if pairwise.size else 0.0,
        "median_pairwise_distance": float(np.median(pairwise)) if pairwise.size else 0.0,
        "cluster_count_at_unit_distance": _greedy_cluster_count(standardized, 1.0),
    }


def _greedy_cluster_count(values: Any, threshold: float) -> int:
    import numpy as np

    representatives = []
    for value in values:
        if not any(float(np.linalg.norm(value - representative)) <= threshold for representative in representatives):
            representatives.append(value)
    return len(representatives)


def paired_bootstrap_improvement(
    left: Sequence[float], right: Sequence[float], *, seed: int, replicates: int
) -> dict[str, Any]:
    """Positive values mean the right/candidate condition is lower and better."""
    import numpy as np

    differences = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    if differences.size == 0 or differences.shape != np.asarray(right).shape:
        raise ValueError("E007 Phase-3I.2 paired bootstrap shape contradiction")
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [differences[rng.integers(0, differences.size, differences.size)].mean() for _ in range(replicates)]
    )
    return {
        "positive_means_candidate_lower_better": True,
        "mean_improvement": float(differences.mean()),
        "ci_95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "fraction_improved": float((differences > 0).mean()),
        "pair_count": int(differences.size),
    }


def classify_factorial_results(
    cells: Mapping[str, Sequence[Mapping[str, Any]]], config: Mapping[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    """Apply predeclared component-wise gates without a composite score."""
    local_metrics = (
        "adjacent_distance_rmse_to_3_8_angstrom",
        "discontinuity_fraction",
        "clash_fraction",
    )
    global_metrics = (
        "radius_of_gyration_angstrom",
        "asphericity",
        "contact_graph_component_count_8a",
    )

    criteria = config.get("publication", {}).get("local_backbone_validity_criteria")

    def locally_valid(row: Mapping[str, Any]) -> float:
        if criteria is None:
            return 0.0
        return float(
            bool(row["finite_coordinates"])
            and float(row["adjacent_distance_rmse_to_3_8_angstrom"])
            <= float(criteria["maximum_adjacent_rmse_angstrom"])
            and float(row["discontinuity_fraction"]) <= float(criteria["maximum_discontinuity_fraction"])
            and float(row["clash_fraction"]) <= float(criteria["maximum_clash_fraction"])
        )

    def compare(left_name: str, right_name: str, seed_offset: int) -> dict[str, Any]:
        left = sorted(cells[left_name], key=lambda row: (row["requested_length"], row["sample_index"]))
        right = sorted(cells[right_name], key=lambda row: (row["requested_length"], row["sample_index"]))
        if [(x["requested_length"], x["sample_index"], x["seed"]) for x in left] != [
            (x["requested_length"], x["sample_index"], x["seed"]) for x in right
        ]:
            raise ValueError("E007 Phase-3I.2 sampling pairing contradiction")
        result = {"local": {}, "global": {}, "by_length": {}}
        if criteria is not None:
            # The bootstrap helper defines positive as left minus right. Reverse the
            # arguments for a validity rate, where a larger candidate is better.
            result["local_backbone_validity"] = paired_bootstrap_improvement(
                [locally_valid(row) for row in right],
                [locally_valid(row) for row in left],
                seed=int(config["publication"]["bootstrap_seed"]) + seed_offset + 100,
                replicates=int(config["publication"]["bootstrap_replicates"]),
            )
        for index, metric in enumerate((*local_metrics, *global_metrics)):
            result["local" if metric in local_metrics else "global"][metric] = paired_bootstrap_improvement(
                [float(row[metric]) for row in left],
                [float(row[metric]) for row in right],
                seed=int(config["publication"]["bootstrap_seed"]) + seed_offset + index,
                replicates=int(config["publication"]["bootstrap_replicates"]),
            )
        for length in config["lengths"]:
            left_values = [
                float(row["adjacent_distance_rmse_to_3_8_angstrom"])
                for row in left
                if int(row["requested_length"]) == int(length)
            ]
            right_values = [
                float(row["adjacent_distance_rmse_to_3_8_angstrom"])
                for row in right
                if int(row["requested_length"]) == int(length)
            ]
            result["by_length"][str(length)] = paired_bootstrap_improvement(
                left_values,
                right_values,
                seed=int(config["publication"]["bootstrap_seed"]) + seed_offset + int(length),
                replicates=int(config["publication"]["bootstrap_replicates"]),
            )
            if criteria is not None:
                right_rows = [row for row in right if int(row["requested_length"]) == int(length)]
                left_rows = [row for row in left if int(row["requested_length"]) == int(length)]
                result["by_length"][str(length)]["local_backbone_validity"] = paired_bootstrap_improvement(
                    [locally_valid(row) for row in right_rows],
                    [locally_valid(row) for row in left_rows],
                    seed=int(config["publication"]["bootstrap_seed"]) + seed_offset + int(length) + 100,
                    replicates=int(config["publication"]["bootstrap_replicates"]),
                )
        improved_strata = sum(value["mean_improvement"] > 0 for value in result["by_length"].values())
        primary_interval = (
            result["local_backbone_validity"]
            if criteria is not None
            else result["local"]["adjacent_distance_rmse_to_3_8_angstrom"]
        )
        local_supported = primary_interval["ci_95"][0] > 0 and improved_strata >= int(
            config["publication"]["minimum_improved_length_strata"]
        )
        global_degradation = any(
            value["mean_improvement"] < 0 and value["ci_95"][1] < 0 for value in result["global"].values()
        )
        result["gate"] = {
            "local_supported": local_supported,
            "improved_length_strata": improved_strata,
            "global_degradation_detected": global_degradation,
        }
        return result

    objective = compare("v_only/native_reverse", "v_plus_local/native_reverse", 10000)
    sampler = compare("v_only/native_reverse", "v_only/local_guided_reverse", 20000)
    combined = compare("v_plus_local/native_reverse", "v_plus_local/local_guided_reverse", 30000)
    categories = []
    if objective["gate"]["local_supported"] and not objective["gate"]["global_degradation_detected"]:
        categories.append("objective_correction_supported")
    if sampler["gate"]["local_supported"] and not sampler["gate"]["global_degradation_detected"]:
        categories.append("sampler_correction_supported")
    if combined["gate"]["local_supported"] and not combined["gate"]["global_degradation_detected"]:
        categories.append("combined_correction_supported")
    if any(
        item["gate"]["local_supported"] and item["gate"]["global_degradation_detected"]
        for item in (objective, sampler, combined)
    ):
        categories.append("local_improvement_with_global_degradation")
    if any(
        objective["by_length"][str(length)]["mean_improvement"] <= 0
        and sampler["by_length"][str(length)]["mean_improvement"] <= 0
        and combined["by_length"][str(length)]["mean_improvement"] <= 0
        for length in (384, 500)
    ):
        categories.append("length_384_500_failure_persists")
    if not categories:
        categories.append("no_material_local_geometry_improvement")
    return categories, {"objective": objective, "sampler": sampler, "combined": combined, "no_scalar_score": True}


def _pilot_safeguards(final_evaluations: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    """Report each predeclared safeguard separately for the matched native sampler."""
    control = final_evaluations["v_only"]
    candidate = final_evaluations["v_plus_local"]
    expected_records = (
        len(config["lengths"])
        * int(config["pilot"]["samples_per_length_per_sampler"])
        * (1 + len(config["guidance"]["strengths"]))
    )
    sampling_completion = all(
        evaluation["sampling_record_count"] == expected_records for evaluation in (control, candidate)
    )
    native = {
        arm: sorted(
            (record for record in evaluation["sampling_records"] if record["sampler"] == "native_reverse"),
            key=lambda record: (record["requested_length"], record["sample_index"], record["seed"]),
        )
        for arm, evaluation in final_evaluations.items()
    }

    def keys(rows: Sequence[Mapping[str, Any]]) -> list[tuple[Any, Any, Any]]:
        return [(row["requested_length"], row["sample_index"], row["seed"]) for row in rows]

    if keys(native["v_only"]) != keys(native["v_plus_local"]):
        raise ValueError("pilot safeguard sampling identity contradiction")
    finite_coordinates = all(bool(row["finite_coordinates"]) for rows in native.values() for row in rows)
    control_v = float(control["denoising"]["global"]["coordinate_v_mse"])
    candidate_v = float(candidate["denoising"]["global"]["coordinate_v_mse"])
    denoising = candidate_v <= control_v
    control_diversity = float(control["descriptor_space_dispersion"]["mean_pairwise_distance"])
    candidate_diversity = float(candidate["descriptor_space_dispersion"]["mean_pairwise_distance"])
    diversity = (
        candidate_diversity
        >= control_diversity * (1 - float(config["publication"]["maximum_diversity_relative_degradation"]))
        and candidate["duplicate_record_fraction"] <= control["duplicate_record_fraction"]
    )
    chirality_shift = abs(
        sum(float(row["signed_pseudo_dihedral_positive_fraction"]) for row in native["v_plus_local"])
        / len(native["v_plus_local"])
        - sum(float(row["signed_pseudo_dihedral_positive_fraction"]) for row in native["v_only"])
        / len(native["v_only"])
    )
    chirality = chirality_shift <= float(config["publication"]["maximum_chirality_positive_fraction_shift"])
    components = {
        "denoising_objective": {"passed": denoising, "v_only_mse": control_v, "v_plus_local_mse": candidate_v},
        "diversity": {
            "passed": diversity,
            "v_only_mean_pairwise_distance": control_diversity,
            "v_plus_local_mean_pairwise_distance": candidate_diversity,
        },
        "chirality": {"passed": chirality, "positive_fraction_shift": chirality_shift},
        "finite_coordinate_rate": {"passed": finite_coordinates},
        "sampling_completion": {"passed": sampling_completion, "expected_records_per_arm": expected_records},
    }
    return {
        "components": components,
        "all_passed_excluding_global_topology": all(x["passed"] for x in components.values()),
    }


def _not_executed_payload(mode: str, config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    verify_protected_evidence(config)
    return {
        "version": VERSION,
        "mode": mode,
        "status": "execution_scaffold_ready",
        "scientific_contract": "bounded matched factorial; no scalar winner score",
        **NON_AUTHORIZING,
    }


def validate_calibration_panel(
    config_path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Verify the complete immutable calibration panel before any execution side effects."""
    from collections import Counter

    from protein_distance_diffusion.evaluation import e007_denoiser_sampler_localization as localization

    config_path = Path(config_path)
    config = load_config(config_path)
    protected = verify_protected_evidence(config)
    source = localization.load_config(config["dataset_source_config"])
    prerequisites = localization.verify_prerequisites(source, full=True)
    compact_panel, selection_diagnostics = localization.select_validation_panel(source)
    reconstructed, canonical = localization.reconstruct_authoritative_panel(source, compact_panel)
    requested = Counter(int(item["selection"]["target_length"]) for item in reconstructed)
    actual = Counter(int(item["selection"]["actual_length"]) for item in reconstructed)
    expected = list(map(int, config["calibration"]["lengths"]))
    if sorted(requested) != sorted(expected) or len(reconstructed) != sum(requested.values()):
        raise ValueError("E007 Phase-3I.2 calibration panel requested-length coverage contradiction")
    by_length: dict[int, dict[str, Any]] = {}
    for item in reconstructed:
        length = int(item["selection"]["target_length"])
        by_length.setdefault(length, item["canonical_row"])
    if sorted(by_length) != sorted(expected):
        raise ValueError("E007 Phase-3I.2 calibration panel length coverage contradiction")
    direct = relocated = 0
    for item in reconstructed:
        resolutions = (
            item["provenance"]["source_relocation_resolution"],
            item["provenance"]["npz_relocation_resolution"],
        )
        if all(value == "recorded_path" for value in resolutions):
            direct += 1
        else:
            relocated += 1
    payload = {
        "version": VERSION,
        "mode": "validate_calibration_panel",
        "status": "panel_validated_read_only",
        "configuration_sha256": _sha256_file(config_path),
        "protected_hashes": protected,
        "source_prerequisites": prerequisites,
        "selected_row_count": len(compact_panel),
        "accepted_row_count": canonical["accepted_row_count"],
        "counts_by_requested_length": {str(k): requested[k] for k in sorted(requested)},
        "counts_by_actual_length": {str(k): actual[k] for k in sorted(actual)},
        "direct_resolution_row_count": direct,
        "relocated_resolution_row_count": relocated,
        "canonical_panel_identity_sha256": canonical["identity_sha256"],
        "selection_panel_sha256": selection_diagnostics["record_sha256"],
        "dataset_scan_scope": "selected_authoritative_panel_rows_only",
        "model_created": False,
        "cuda_initialized": False,
        "optimizer_created": False,
        "forward_performed": False,
        "output_created": False,
        "staging_created": False,
        **NON_AUTHORIZING,
    }
    return payload, reconstructed, by_length


def run_validate_calibration_panel(config_path: str | Path) -> dict[str, Any]:
    """Read-only bounded calibration startup validation."""
    payload, _panel, _by_length = validate_calibration_panel(config_path)
    return payload


def run_gradient_calibration(config_path: str | Path) -> dict[str, Any]:
    """Run the bounded zero-update gradient calibration and publish atomically."""
    import resource

    from protein_distance_diffusion.evaluation import e007_denoiser_sampler_localization as localization

    config_path = Path(config_path)
    validation, _panel, by_length = validate_calibration_panel(config_path)
    config = load_config(config_path)
    protected_before = verify_protected_evidence(config)
    source = localization.load_config(config["dataset_source_config"])
    expected_lengths = list(map(int, config["calibration"]["lengths"]))
    prepared_by_length = {
        length: localization._prepared_reference(source, by_length[length]) for length in expected_lengths
    }
    output = Path(config["calibration_output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 Phase-3I.2 calibration output exists: {output} or {staging}")
    import torch

    from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
    from protein_distance_diffusion.training.e007_coordinate_real_pilot import uniform_coordinate_v_mse

    staging.mkdir(parents=True)
    heartbeat = staging / "heartbeat.json"
    _atomic_json(heartbeat, {"status": "running", "stage": "panel_validation", **NON_AUTHORIZING})
    try:
        if config["device"] != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("E007 Phase-3I.2 calibration requires configured CUDA")
        panel_evidence = validation
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)
        model = localization._load_model(source, device).requires_grad_(True).train()
        diffusion = CoordinateVPDiffusion(int(config["diffusion_steps"]))
        settings = dict(config["local_objective"])
        settings["coordinate_scale_angstrom"] = float(config["coordinate_scale_angstrom"])
        rows = []
        for length in expected_lengths:
            prepared = prepared_by_length[length]
            clean = prepared["coordinates"].to(device)
            mask = prepared["residue_mask"].to(device)
            continuity = prepared["chain_continuity_mask"].to(device)
            for timestep_value in map(int, config["calibration"]["timesteps"]):
                generator = torch.Generator(device=device).manual_seed(3915000 + length * 1000 + timestep_value)
                timestep = torch.tensor([timestep_value], dtype=torch.long, device=device)
                batch = diffusion.make_training_batch(clean, mask, timesteps=timestep, generator=generator)
                prediction = model(
                    batch.noisy_coordinates,
                    timestep,
                    prepared["lengths"].to(device),
                    mask,
                    continuity,
                )["v_prediction"]
                v_loss = uniform_coordinate_v_mse(prediction, batch.coordinate_v_target, mask)
                predicted_x0 = diffusion.reconstruct_x0(batch.noisy_coordinates, timestep, prediction, mask)
                losses = local_backbone_losses(predicted_x0, clean, mask, continuity, settings)
                profile = gradient_profile(v_loss, losses, tuple(model.parameters()))
                candidates = {}
                for name, weights in config["local_objective"]["candidate_coefficient_sets"].items():
                    auxiliary, weighted = weighted_local_objective(losses, weights)
                    total = v_loss + auxiliary
                    total_gradients = torch.autograd.grad(total, tuple(model.parameters()), retain_graph=True)
                    total_norm = torch.sqrt(
                        sum(gradient.detach().double().square().sum() for gradient in total_gradients)
                    )
                    candidates[name] = {
                        "weighted_losses": {key: float(value.detach()) for key, value in weighted.items()},
                        "weighted_auxiliary_loss": float(auxiliary.detach()),
                        "total_gradient_norm": float(total_norm),
                        "total_to_v_gradient_ratio": float(total_norm / max(profile["v_gradient_norm"], 1e-30)),
                    }
                rows.append(
                    {
                        "length": length,
                        "timestep": timestep_value,
                        "sample_id": prepared["sample_ids"][0],
                        "v_loss": float(v_loss.detach()),
                        "raw_losses": {name: float(losses[name].detach()) for name in LOCAL_TERMS},
                        "gradient_profile": profile,
                        "candidate_coefficients": candidates,
                    }
                )
                model.zero_grad(set_to_none=True)
                _atomic_json(
                    heartbeat,
                    {
                        "status": "running",
                        "stage": "gradient_cells",
                        "completed_cells": len(rows),
                        "total_cells": len(expected_lengths) * len(config["calibration"]["timesteps"]),
                        **NON_AUTHORIZING,
                    },
                )
        protected_after = verify_protected_evidence(config)
        if protected_before != protected_after:
            raise ValueError("E007 Phase-3I.2 protected evidence changed during calibration")
        report = {
            "version": VERSION,
            "mode": "gradient_calibration_smoke",
            "status": "completed",
            "configuration_sha256": _sha256_file(config_path),
            "selected_coefficient_set": None,
            "review_required": True,
            "calibration_authorizes_pilot": False,
            "cells": rows,
            "panel_evidence": panel_evidence,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
            "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
            "protected_inputs_unchanged": True,
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "report.json", report)
        protocol = {
            "version": VERSION,
            "status": "completed",
            "mode": "gradient_calibration_smoke",
            "configuration_sha256": report["configuration_sha256"],
            "report_sha256": _sha256_file(staging / "report.json"),
            "cell_count": len(rows),
            "zero_optimizer_updates": True,
            "review_required": True,
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "protocol.json", protocol)
        _atomic_json(heartbeat, {"status": "completed", "report_sha256": protocol["report_sha256"], **NON_AUTHORIZING})
        staging.replace(output)
        _fsync_directory(output.parent)
        return report
    except BaseException as error:
        _atomic_json(
            heartbeat,
            {"status": "failed", "error_type": type(error).__name__, "error_message": str(error), **NON_AUTHORIZING},
        )
        raise


def run_local_backbone_pilot(
    config_path: str | Path, *, resume: bool = False, lifecycle_smoke: bool = False
) -> dict[str, Any]:
    """Run matched arms sequentially and publish only after both complete."""
    config_path = Path(config_path)
    config = load_config(config_path)
    if not config["local_objective"].get("coefficient_table_path"):
        _calibration_ready(config, raise_on_failure=True)
    if lifecycle_smoke and resume:
        raise ValueError("lifecycle smoke cannot resume a pilot")
    validate_pilot_contract(config_path, refuse_unreviewed=not lifecycle_smoke, allow_staging_for_resume=resume)
    import random
    import resource

    import numpy as np
    import torch

    from protein_distance_diffusion.evaluation import e007_denoiser_sampler_localization as localization
    from protein_distance_diffusion.models.coordinate_equivariance import coordinate_model_execution_context
    from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import EquivariantPairCoordinateUNet
    from protein_distance_diffusion.training import e007_coordinate_real_pilot as phase3f
    from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
    from protein_distance_diffusion.training.e007_coordinate_real_loader_smoke import prepare_coordinate_batch

    telemetry: CudaMemoryTelemetry | None = None
    memory_path: Path | None = None

    def check_pilot_memory(*, end_phase: bool = False, phase: str = "boundary") -> dict[str, float]:
        if telemetry is None:
            raise RuntimeError("pilot CUDA memory telemetry was not initialized")
        memory = telemetry.end_phase() if end_phase else telemetry.snapshot()
        observed = {
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            **memory,
        }
        if not math.isfinite(observed["peak_rss_mib"]) or observed["peak_rss_mib"] < 0:
            raise MemoryError(f"invalid pilot RSS observation: {observed}")
        if end_phase and memory_path is not None:
            _append_fsync_jsonl(memory_path, {"phase": phase, **observed})
        if _dynamic_memory_violation(observed, config["memory"]):
            raise MemoryError(f"E007 Phase-3I.2 pilot memory limit exceeded: {observed}")
        return observed

    protected = verify_protected_evidence(config)
    coefficient_table = load_coefficient_table(config["local_objective"]["coefficient_table_path"])
    reviewed_source = _resolve_reviewed_coordinate_source(config)
    output = Path(config["output_dir"])
    if lifecycle_smoke:
        expected_smoke = (
            "reports/experiments/E007_matrix_sequence_cogeneration/"
            "local_backbone_repair_pilot_phase3i2_lifecycle_smoke_v3"
        )
        if config.get("lifecycle_smoke_output_dir") != expected_smoke:
            raise ValueError("lifecycle smoke requires its fresh dedicated path")
        output = Path(expected_smoke)
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or (staging.exists() and not resume):
        raise FileExistsError(f"E007 Phase-3I.2 output exists: {output} or {staging}")
    if resume and not staging.is_dir():
        raise FileNotFoundError("E007 Phase-3I.2 resume requires its staging directory")
    if not resume:
        (staging / "arms").mkdir(parents=True)
    heartbeat_path = staging / "heartbeat.json"
    _atomic_json(heartbeat_path, {"status": "running", "stage": "dataset_selection", **NON_AUTHORIZING})
    try:
        if config["device"] != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("E007 Phase-3I.2 pilot requires configured CUDA")
        source = reviewed_source["source"]
        source["successful_optimizer_updates"] = 500
        source["evaluation_updates"] = list(config["pilot"]["evaluation_updates"])
        source["sampling_updates"] = list(config["pilot"]["evaluation_updates"])
        source["checkpoint_updates"] = list(config["pilot"]["evaluation_updates"])[1:]
        source["seed"] = int(config["pilot"]["paired_seed"])
        source["timestep_evaluation_bins"] = [
            {"name": "low", "timestep": 25},
            {"name": "medium", "timestep": 250},
            {"name": "high", "timestep": 425},
            {"name": "very_high", "timestep": 499},
        ]
        authorization = phase3f._authorize(source)
        selected = phase3f._select_rows(source, authorization)
        schedule = _paired_schedule(source, selected["train"])
        schedule_hash = _canonical_sha(schedule)
        audit_source = localization.load_config(config["dataset_source_config"])
        audit_compact, audit_selection = localization.select_validation_panel(audit_source)
        audit_canonical, audit_identity = localization.reconstruct_authoritative_panel(audit_source, audit_compact)
        audit_by_length: dict[int, dict[str, Any]] = {}
        for item in audit_canonical:
            target_length = int(item["selection"]["target_length"])
            if target_length in config["lengths"]:
                audit_by_length.setdefault(target_length, item["canonical_row"])
        if sorted(audit_by_length) != sorted(map(int, config["lengths"])):
            raise ValueError("drift-audit immutable panel does not cover every configured length")
        audit_panel = [(length, audit_by_length[length]) for length in map(int, config["lengths"])]
        panel_manifest = {
            "schedule_sha256": schedule_hash,
            "train_sample_ids": [item["sample_ids"] for item in schedule],
            "validation_sample_ids": {
                name: [row["sample_id"] for row in rows] for name, rows in selected["validation"].items()
            },
            "drift_audit_panel_sample_ids": {str(length): str(row["sample_id"]) for length, row in audit_panel},
            "drift_audit_panel_identity_sha256": audit_identity["identity_sha256"],
            "drift_audit_selection_sha256": audit_selection["record_sha256"],
        }
        _atomic_json(staging / "panel_manifest.json", panel_manifest)
        source_checkpoint = _resolve_reviewed_coordinate_source(config, load_checkpoint=True)["checkpoint"]
        source_model_hash = _canonical_sha(
            [
                (name, hashlib.sha256(value.numpy().tobytes()).hexdigest())
                for name, value in source_checkpoint["model"].items()
            ]
        )
        results = []
        run_peak_allocated = run_peak_reserved = 0.0
        for arm in ARMS[:1] if lifecycle_smoke else ARMS:
            arm_dir = staging / "arms" / arm
            arm_dir.mkdir(exist_ok=True)
            memory_path = arm_dir / "memory_telemetry.jsonl"
            latest = arm_dir / "latest.pt"
            seed = int(config["pilot"]["paired_seed"])
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.cuda.manual_seed_all(seed)
            device = torch.device("cuda")
            # The next arm has a fresh phase counter; prior arm peaks remain in the run accumulator.
            telemetry = CudaMemoryTelemetry(torch.cuda, device)
            telemetry.run_allocated = run_peak_allocated
            telemetry.run_reserved = run_peak_reserved
            if resume and memory_path.exists():
                for line in memory_path.read_text().splitlines():
                    record = json.loads(line)
                    for kind in ("allocated", "reserved"):
                        phase_peak = float(record[f"phase_peak_cuda_{kind}_mib"])
                        previous_run_peak = float(record[f"run_peak_cuda_{kind}_mib"])
                        if not (
                            math.isfinite(phase_peak)
                            and math.isfinite(previous_run_peak)
                            and 0 <= phase_peak <= previous_run_peak <= telemetry.total
                        ):
                            raise MemoryError("pilot resume contains impossible prior phase memory evidence")
                        field = "run_allocated" if kind == "allocated" else "run_reserved"
                        setattr(telemetry, field, max(getattr(telemetry, field), previous_run_peak))
            with coordinate_model_execution_context(config["numerics"], device):
                model, optimizer, scheduler = _initialize_pilot_arm(
                    EquivariantPairCoordinateUNet,
                    source["model"],
                    source_checkpoint["model"],
                    source["optimizer"],
                    device,
                )
                diffusion = CoordinateVPDiffusion(int(config["diffusion_steps"]))
                update = samples_processed = residues_processed = 0
                if latest.exists():
                    if not resume:
                        raise FileExistsError(f"E007 Phase-3I.2 arm state already exists: {latest}")
                    state = torch.load(latest, map_location="cpu", weights_only=False)
                    _validate_recovery_state(
                        state, config_path, protected, arm, schedule_hash, coefficient_table["sha256"]
                    )
                    model.load_state_dict(state["model"])
                    optimizer.load_state_dict(state["optimizer"])
                    scheduler.load_state_dict(state["scheduler"])
                    _restore_rng_state(state["rng_state"])
                    update = int(state["optimizer_update"])
                    samples_processed = int(state["samples_processed"])
                    residues_processed = int(state["valid_residues_processed"])
                elif resume and (arm_dir / "summary.json").exists():
                    completed = json.loads((arm_dir / "summary.json").read_text())
                    prior_allocated = float(completed["peak_cuda_allocated_mib"])
                    prior_reserved = float(completed["peak_cuda_reserved_mib"])
                    if not (
                        math.isfinite(prior_allocated)
                        and math.isfinite(prior_reserved)
                        and 0 <= prior_allocated <= prior_reserved <= telemetry.total
                    ):
                        raise MemoryError("pilot resume contains impossible prior arm memory evidence")
                    results.append(completed)
                    run_peak_allocated = max(run_peak_allocated, prior_allocated)
                    run_peak_reserved = max(run_peak_reserved, prior_reserved)
                    continue
                # Initialization and any exact-state restore form their own phase.
                check_pilot_memory(end_phase=True, phase="arm_startup")
                metrics_path = arm_dir / "metrics.jsonl"
                _reconcile_metrics_to_checkpoint(metrics_path, update)
                evaluations: dict[str, Any] = {}
                evaluation_path = arm_dir / "evaluations.json"
                if evaluation_path.exists():
                    evaluations = json.loads(evaluation_path.read_text())
                if update == 0 and "0" not in evaluations:
                    evaluations["0"] = _bounded_evaluation(
                        model, diffusion, selected["validation"], source, config, device
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    check_pilot_memory(end_phase=True, phase="update_zero_evaluation")
                    _atomic_json(evaluation_path, evaluations)
                audit_path = arm_dir / "drift_audits.json"
                audit_records = json.loads(audit_path.read_text()) if audit_path.exists() else []
                completed_audits = {int(item["update"]) for item in audit_records}
                if len(completed_audits) != len(audit_records) or any(
                    not item.get("pass", False) for item in audit_records
                ):
                    raise ValueError("pilot resume contains failed or duplicated drift-audit records")
                if latest.exists() and not set(map(int, state.get("completed_audit_updates", ()))).issubset(
                    completed_audits
                ):
                    raise ValueError("pilot resume checkpoint claims an absent completed drift audit")

                def audit_boundary(
                    boundary: int,
                    cursor: int,
                    *,
                    seen_audits: set[int] = completed_audits,
                    audit_log: list[dict[str, Any]] = audit_records,
                    log_path: Path = audit_path,
                    audit_model: Any = model,
                    audit_optimizer: Any = optimizer,
                    audit_scheduler: Any = scheduler,
                    audit_diffusion: Any = diffusion,
                    audit_device: Any = device,
                ) -> None:
                    if boundary in seen_audits:
                        prior = next(item for item in audit_log if int(item["update"]) == boundary)
                        if not prior["pass"]:
                            raise FloatingPointError(f"previous drift audit failed at update {boundary}")
                        return
                    audit_result = zero_update_drift_audit(
                        audit_model,
                        audit_optimizer,
                        audit_scheduler,
                        audit_diffusion,
                        audit_panel,
                        config,
                        coefficient_table,
                        audit_device,
                        update=boundary,
                        data_cursor=cursor,
                        audit_timesteps=config["online_drift_monitor"]["timesteps"],
                    )
                    audit_log.append(audit_result)
                    _atomic_json(log_path, audit_log)
                    seen_audits.add(boundary)
                    if not audit_result["pass"]:
                        raise FloatingPointError(f"drift audit failed at update {boundary}; evidence recorded")
                    gc.collect()
                    torch.cuda.empty_cache()
                    check_pilot_memory(end_phase=True, phase=f"audit_{boundary}")

                if update in GRADIENT_AUDIT_UPDATES:
                    audit_boundary(update, update)
                    if update > 0:
                        recovery = _recovery_payload(
                            model,
                            optimizer,
                            scheduler,
                            update,
                            samples_processed,
                            residues_processed,
                            config_path,
                            protected,
                            arm,
                            schedule_hash,
                            coefficient_table["sha256"],
                            completed_audits,
                        )
                        _atomic_torch(latest, recovery)
                settings = dict(config["local_objective"])
                settings["coordinate_scale_angstrom"] = float(config["coordinate_scale_angstrom"])
                while update < (25 if lifecycle_smoke else 500):
                    entry = schedule[update]
                    rows = selected["train"][entry["stratum"]][entry["start"] : entry["stop"]]
                    if [row["sample_id"] for row in rows] != entry["sample_ids"]:
                        raise ValueError("E007 Phase-3I.2 paired schedule identity contradiction")
                    prepared = prepare_coordinate_batch(
                        rows, float(config["coordinate_scale_angstrom"]), int(source["expected_downsample_factor"])
                    )
                    corruption = phase3f.make_uniform_training_corruption(
                        prepared, diffusion, seed=seed + update + 1, device=device
                    )
                    optimizer.zero_grad(set_to_none=True)
                    model.train()
                    prediction = model(
                        corruption["batch"].noisy_coordinates,
                        corruption["timesteps"],
                        prepared["lengths"].to(device),
                        corruption["mask"],
                        prepared["chain_continuity_mask"].to(device),
                    )["v_prediction"]
                    total, v_loss, raw, weighted, coefficient_rows, auxiliary = _pilot_training_objective(
                        arm, prediction, corruption, prepared, diffusion, settings, coefficient_table, device
                    )
                    if not bool(torch.isfinite(total)):
                        raise FloatingPointError("E007 Phase-3I.2 non-finite training loss")
                    total.backward()
                    phase3f._gradient_evidence(model)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(source["optimizer"]["gradient_clip_norm"]), error_if_nonfinite=True
                    )
                    optimizer.step()
                    scheduler.step()
                    check_pilot_memory()
                    update += 1
                    samples_processed += len(rows)
                    residues_processed += int(prepared["lengths"].sum())
                    metric = {
                        "arm": arm,
                        "optimizer_update": update,
                        "paired_identity_sha256": entry["identity_sha256"],
                        "v_loss": float(v_loss.detach()),
                        "raw_local_losses": {
                            name: [float(value) for value in raw[name].detach().cpu().tolist()]
                            if raw and raw[name].ndim
                            else (float(raw[name].detach()) if raw else None)
                            for name in LOCAL_TERMS
                        },
                        "weighted_local_losses": {
                            name: [float(value) for value in weighted[name].detach().cpu().tolist()]
                            if weighted and weighted[name].ndim
                            else (float(weighted[name].detach()) if weighted else None)
                            for name in LOCAL_TERMS
                        },
                        "sample_local_objective_telemetry": []
                        if arm == "v_only"
                        else [
                            {
                                "sample_id": prepared["sample_ids"][sample_index],
                                "timestep": int(corruption["timesteps"][sample_index].detach().cpu()),
                                "coefficient_row": [
                                    float(value) for value in coefficient_rows[sample_index].detach().cpu().tolist()
                                ],
                                "raw_losses": {
                                    name: float(raw[name][sample_index].detach().cpu()) for name in LOCAL_TERMS
                                },
                                "weighted_losses": {
                                    name: float(weighted[name][sample_index].detach().cpu()) for name in LOCAL_TERMS
                                },
                                "eligibility_counts": {
                                    name: int(raw["denominators"][name][sample_index].detach().cpu())
                                    for name in LOCAL_TERMS
                                },
                                "reduced_batch_contribution": float(
                                    sum(weighted[name][sample_index] for name in LOCAL_TERMS).detach().cpu()
                                    / len(prepared["sample_ids"])
                                ),
                            }
                            for sample_index in range(len(prepared["sample_ids"]))
                        ],
                        "total_loss": float(total.detach()),
                    }
                    _append_fsync_jsonl(metrics_path, metric)
                    recovery = _recovery_payload(
                        model,
                        optimizer,
                        scheduler,
                        update,
                        samples_processed,
                        residues_processed,
                        config_path,
                        protected,
                        arm,
                        schedule_hash,
                        coefficient_table["sha256"],
                        completed_audits,
                    )
                    _atomic_torch(latest, recovery)
                    # The metric and recovery payload are CPU evidence. Release the
                    # training graph before audit and evaluation phase measurements.
                    optimizer.zero_grad(set_to_none=True)
                    del total, v_loss, prediction, corruption, raw, weighted, coefficient_rows, auxiliary
                    gc.collect()
                    torch.cuda.empty_cache()
                    check_pilot_memory(end_phase=True, phase=f"training_{update}")
                    if update in GRADIENT_AUDIT_UPDATES:
                        audit_boundary(update, update)
                        recovery["completed_audit_updates"] = sorted(completed_audits)
                        recovery["next_required_audit_boundary"] = next(
                            (
                                boundary
                                for boundary in GRADIENT_AUDIT_UPDATES
                                if boundary >= update and boundary not in completed_audits
                            ),
                            None,
                        )
                        _atomic_torch(latest, recovery)
                    if update in config["pilot"]["evaluation_updates"]:
                        evaluations[str(update)] = _bounded_evaluation(
                            model, diffusion, selected["validation"], source, config, device
                        )
                        gc.collect()
                        torch.cuda.empty_cache()
                        check_pilot_memory(end_phase=True, phase=f"evaluation_{update}")
                        _atomic_json(evaluation_path, evaluations)
                        _atomic_torch(arm_dir / f"step-{update:05d}.pt", recovery)
                    _atomic_json(
                        heartbeat_path,
                        {
                            "status": "running",
                            "stage": "training",
                            "arm": arm,
                            "optimizer_update": update,
                            "samples_processed": samples_processed,
                            **NON_AUTHORIZING,
                        },
                    )
                if lifecycle_smoke:
                    smoke = {
                        "status": "completed",
                        "mode": "pilot_lifecycle_smoke",
                        "optimizer_updates": update,
                        "arm": arm,
                        "completed_audit_updates": sorted(completed_audits),
                        "update_zero_evaluation_lifecycle_completed": "0" in evaluations,
                        "memory_telemetry": check_pilot_memory(end_phase=True, phase="smoke_completion"),
                        "scientific_pilot_result": False,
                        **NON_AUTHORIZING,
                    }
                    if update != 25 or not set(range(11)).issubset(completed_audits) or "0" not in evaluations:
                        raise RuntimeError("pilot lifecycle smoke boundary mismatch")
                    _atomic_json(staging / "smoke_contract.json", smoke)
                    del audit_boundary, model, optimizer, scheduler
                    gc.collect()
                    torch.cuda.empty_cache()
                    check_pilot_memory(end_phase=True, phase="smoke_cleanup")
                    _atomic_json(heartbeat_path, {"status": "completed_smoke", **NON_AUTHORIZING})
                    staging.replace(output)
                    _fsync_directory(output.parent)
                    return smoke
                summary = {
                    "arm": arm,
                    "status": "completed",
                    "optimizer_updates": update,
                    "samples_processed": samples_processed,
                    "valid_residues_processed": residues_processed,
                    "source_model_state_sha256": source_model_hash,
                    "paired_schedule_sha256": schedule_hash,
                    "evaluation_updates": sorted(map(int, evaluations)),
                    "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                    "peak_cuda_allocated_mib": telemetry.run_allocated,
                    "peak_cuda_reserved_mib": telemetry.run_reserved,
                    "memory_telemetry": check_pilot_memory(end_phase=True, phase="arm_completion"),
                    **NON_AUTHORIZING,
                }
                _atomic_json(arm_dir / "summary.json", summary)
                results.append(summary)
                del audit_boundary
                del model, optimizer, scheduler
                gc.collect()
                torch.cuda.empty_cache()
                check_pilot_memory(end_phase=True, phase="arm_cleanup")
                run_peak_allocated = telemetry.run_allocated
                run_peak_reserved = telemetry.run_reserved
        configured_arms = tuple(config["arms"])
        if len(results) != 2 or {item["arm"] for item in results} != set(configured_arms):
            raise ValueError("E007 Phase-3I.2 matched-arm completion contradiction")
        after = verify_protected_evidence(config)
        if after != protected:
            raise ValueError("E007 Phase-3I.2 protected evidence changed")
        final_evaluations = {
            arm: json.loads((staging / "arms" / arm / "evaluations.json").read_text())["500"] for arm in ARMS
        }
        final_records = {arm: evaluation["sampling_records"] for arm, evaluation in final_evaluations.items()}
        decisions_by_strength = {}
        all_categories: set[str] = set()
        for strength in map(float, config["guidance"]["strengths"]):
            cells = {}
            for arm in ARMS:
                cells[f"{arm}/native_reverse"] = [
                    row for row in final_records[arm] if row["sampler"] == "native_reverse"
                ]
                cells[f"{arm}/local_guided_reverse"] = [
                    row
                    for row in final_records[arm]
                    if row["sampler"] == "local_guided_reverse" and float(row["guidance_strength"]) == strength
                ]
            categories, evidence = classify_factorial_results(cells, config)
            decisions_by_strength[str(strength)] = {"decision_categories": categories, "evidence": evidence}
            all_categories.update(categories)
        if not all_categories:
            all_categories.add("inconclusive_bounded_pilot")
        safeguards = _pilot_safeguards(final_evaluations, config)
        global_topology_passed = all(
            not item["evidence"]["objective"]["gate"]["global_degradation_detected"]
            for item in decisions_by_strength.values()
        )
        safeguards["components"]["global_topology"] = {"passed": global_topology_passed}
        safeguards["all_passed"] = safeguards["all_passed_excluding_global_topology"] and global_topology_passed
        report = {
            "version": VERSION,
            "status": "completed_non_authorizing_bounded_pilot",
            "arm_summaries": results,
            "paired_schedule_sha256": schedule_hash,
            "decision_categories": sorted(all_categories),
            "decisions_reported_separately_by_guidance_strength": decisions_by_strength,
            "primary_comparison_supported_with_safeguards": (
                "objective_correction_supported" in all_categories and safeguards["all_passed"]
            ),
            "safeguards": safeguards,
            "no_scalar_score": True,
            "native_handedness_unresolved": True,
            "protected_inputs_unchanged": True,
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "report.json", report)
        protocol = {
            "version": VERSION,
            "status": report["status"],
            "report_sha256": _sha256_file(staging / "report.json"),
            "configuration_sha256": _sha256_file(config_path),
            "optimizer_updates_per_arm": 500,
            "factorial_cells": [f"{arm}/{sampler}" for arm in ARMS for sampler in SAMPLERS],
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "protocol.json", protocol)
        _atomic_json(
            heartbeat_path,
            {"status": "completed", "report_sha256": protocol["report_sha256"], **NON_AUTHORIZING},
        )
        staging.replace(output)
        _fsync_directory(output.parent)
        return report
    except BaseException as error:
        _atomic_json(
            heartbeat_path,
            {"status": "failed", "error_type": type(error).__name__, "error_message": str(error), **NON_AUTHORIZING},
        )
        raise


def monitor_local_backbone_pilot(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    heartbeat = (output if output.exists() else staging) / "heartbeat.json"
    if not heartbeat.is_file():
        return {"status": "not_started", "heartbeat_path": heartbeat.as_posix(), **NON_AUTHORIZING}
    return json.loads(heartbeat.read_text())
