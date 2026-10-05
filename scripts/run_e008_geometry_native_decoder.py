#!/usr/bin/env python3
"""CPU preflight, longest-length CUDA smoke, or bounded E008 decoder pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from protein_distance_diffusion.data.e007_coordinate_dataset import E007CoordinateDataset
from protein_distance_diffusion.data.rich_geometry import authorize_rich_geometry_dataset
from protein_distance_diffusion.models.e008_kinematic_decoder import (
    BackboneKinematicDecoder,
    cartesian_to_internal,
    decoder_parameter_count,
    geometry_native_losses,
    internal_to_cartesian,
)

E008_MANIFEST_COLUMNS = (
    "sample_id",
    "experimental_method",
    "conformation_class",
    "selection_class",
    "label_source",
    "source_path",
    "source_sha256",
    "exclusion_reason",
)


def _prototype_geometry_exclusion(sequence: str, coordinates, ca_mask, continuity) -> str:
    """Predeclared feasibility screen; experimental method is not a criterion."""
    xyz = np.asarray(coordinates, dtype=np.float64)
    n = len(sequence)
    reasons = []
    mask = np.asarray(ca_mask, dtype=bool)
    links = np.asarray(continuity, dtype=bool)
    if not 20 <= n <= 500:
        reasons.append("length_out_of_range")
    if mask.shape != (n,) or not mask.all():
        reasons.append("missing_or_unresolved_calpha")
    if links.shape != (max(n - 1, 0),) or not links.all():
        reasons.append("chain_break_or_ambiguous")
    if xyz.shape != (n, 3) or not np.isfinite(xyz).all():
        reasons.append("invalid_coordinates")
    if not reasons:
        bonds = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        if not np.all((bonds >= 3.6) & (bonds <= 4.0)):
            reasons.append("bond_geometry_quality")
        mean_radius = float(np.linalg.norm(xyz - xyz.mean(0), axis=1).mean())
        i, j = np.triu_indices(n, k=8)
        contacts = float(np.mean(np.linalg.norm(xyz[i] - xyz[j], axis=1) < 8.0)) if len(i) else 0.0
        if mean_radius / math.sqrt(n) > 2.5 or contacts < 0.005:
            reasons.append("prototype_geometry_screen")
    return ";".join(reasons)


def _build_manifest(cfg: dict, output_path: str | Path) -> dict:
    """Build an auditable selection manifest from authorized local sidecars only."""
    import pyarrow.parquet as pq

    auth = _authorization(cfg)
    metadata_path = cfg["real_structures"].get("local_metadata_manifest")
    metadata: dict[str, dict] = {}
    metadata_sha = None
    if metadata_path:
        metadata_path = Path(metadata_path)
        metadata_sha = _sha(metadata_path)
        expected_metadata_sha = cfg["real_structures"].get("local_metadata_sha256")
        if not expected_metadata_sha or metadata_sha != expected_metadata_sha:
            raise ValueError("local metadata must have a matching configured SHA-256 pin")
        if metadata_path.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq

            metadata_rows = pq.read_table(metadata_path, columns=["sample_id", "experimental_method"]).to_pylist()
        else:
            with metadata_path.open(newline="") as stream:
                metadata_rows = list(csv.DictReader(stream))
        for row in metadata_rows:
            sid = row.get("sample_id", "")
            if not sid or sid in metadata:
                raise ValueError("local metadata manifest has missing or duplicate sample_id")
            metadata[sid] = row
    entries: dict[str, dict] = {}
    geometric_counts: dict[str, int] = {}
    for split in ("train", "validation"):
        dataset = E007CoordinateDataset(auth, split=split)
        for locator in dataset._locators:
            table = pq.ParquetFile(locator.path).read_row_group(
                locator.row_group,
                columns=[
                    "sample_id",
                    "sequence",
                    "ca_coordinates",
                    "ca_mask",
                    "chain_continuity_mask",
                    "chain_break_mask",
                    "source_path",
                    "source_sha256",
                ],
            )
            for raw in table.to_pylist():
                sid = str(raw["sample_id"])
                if sid in entries:
                    raise ValueError(f"duplicate authorized sidecar identity: {sid}")
                m = metadata.get(sid, {})
                method = str(m.get("experimental_method") or "unknown").strip().upper() or "unknown"
                label = str(m.get("conformation_class") or "unknown").strip().lower() or "unknown"
                source = str(m.get("label_source") or "").strip()
                authoritative = str(m.get("label_authoritative") or "").strip().lower() in {"1", "true", "yes"}
                if label in {"globular", "idp"} and (not source or not authoritative):
                    label = "unknown"
                if label not in {"globular", "idp"}:
                    source = ""
                if label not in {"globular", "idp", "unknown", "ambiguous"}:
                    label, source = "ambiguous", ""
                geom = _prototype_geometry_exclusion(
                    raw["sequence"], raw["ca_coordinates"], raw["ca_mask"], raw["chain_continuity_mask"]
                )
                reasons = geom.split(";") if geom else []
                if label == "idp":
                    reasons.append("known_idp")
                if label == "ambiguous":
                    reasons.append("ambiguous_conformation_label")
                exclusion = ";".join(dict.fromkeys(reasons))
                selection = "prototype_structured" if not exclusion else "excluded"
                entries[sid] = {
                    "sample_id": sid,
                    "experimental_method": method,
                    "conformation_class": label,
                    "selection_class": selection,
                    "label_source": source,
                    "source_path": str(raw.get("source_path") or ""),
                    "source_sha256": str(raw.get("source_sha256") or ""),
                    "exclusion_reason": exclusion,
                }
                geometric_counts[selection] = geometric_counts.get(selection, 0) + 1
    rows = [entries[sid] for sid in sorted(entries)]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=E008_MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": list(E008_MANIFEST_COLUMNS),
        "manifest_path": str(out),
        "manifest_sha256": _sha(out),
        "local_metadata_path": str(metadata_path) if metadata_path else None,
        "local_metadata_sha256": metadata_sha,
        "selection_rule": (
            "20<=length<=500; complete contiguous C-alpha; all adjacent distances 3.6-4.0 A; "
            "mean radial distance/sqrt(length)<=2.5; long-range (|i-j|>=8) <8 A contact fraction>=0.005"
        ),
        "counts": geometric_counts,
        "experimental_method_counts": {
            k: sum(r["experimental_method"] == k and r["selection_class"] == "prototype_structured" for r in rows)
            for k in sorted({r["experimental_method"] for r in rows})
        },
        "conformation_class_counts": {
            k: sum(r["conformation_class"] == k and r["selection_class"] == "prototype_structured" for r in rows)
            for k in sorted({r["conformation_class"] for r in rows})
        },
        "exclusions": {
            reason: sum(reason in r["exclusion_reason"].split(";") for r in rows)
            for reason in sorted({x for r in rows for x in r["exclusion_reason"].split(";") if x})
        },
        "scope_note": (
            "prototype_structured is a decoder-feasibility selection and is not a definitive "
            "biological globular annotation"
        ),
    }
    _atomic_json(out.with_suffix(out.suffix + ".summary.json"), summary)
    return summary


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get("version") != "e008_geometry_native_decoder_v1":
        raise ValueError("E008 configuration version mismatch")
    _validate_config_schema(cfg)
    prior = cfg["frozen_prior"]
    if _sha(prior["checkpoint"]) != prior["checkpoint_sha256"]:
        raise ValueError("frozen v_only checkpoint hash mismatch")
    if _sha(prior["generator_config"]) != prior["generator_config_sha256"]:
        raise ValueError("frozen generator configuration hash mismatch")
    calibration = cfg["corruption_calibration"]
    if _sha(calibration["observed_error_summary"]) != calibration["observed_error_summary_sha256"]:
        raise ValueError("frozen-prior error-calibration report hash mismatch")
    validation = cfg["real_structures"]
    if _sha(validation["clean_validation_manifest"]) != validation["clean_validation_manifest_sha256"]:
        raise ValueError("identity-30 clean-validation manifest hash mismatch")
    return cfg


def _validate_config_schema(cfg: dict) -> None:
    """Validate every runtime field before any model or accelerator is created."""
    required = {
        "decoder": ("width", "layers", "pair_rbf_bins", "bond_length_angstrom", "max_length"),
        "frozen_prior": (
            "checkpoint",
            "checkpoint_sha256",
            "generator_config",
            "generator_config_sha256",
            "coordinate_scale_angstrom",
            "diffusion_steps",
        ),
        "real_structures": (
            "source_config",
            "train_split",
            "clean_validation_manifest",
            "clean_validation_manifest_sha256",
            "validation_cluster_split_seed",
            "development_fraction",
            "sidecar_acceptance_policy",
            "experimental_method_policy",
            "conformation_label_policy",
            "prototype_selection_policy",
            "structure_class_manifest",
        ),
        "corruption_calibration": (
            "observed_error_summary",
            "observed_error_summary_sha256",
            "minimum_sigma_angstrom",
            "maximum_sigma_angstrom",
        ),
        "milestones": (
            "real_structure_roundtrip_count",
            "tiny_overfit_structure_count",
            "tiny_overfit_updates",
            "tiny_overfit_coordinate_rmse_angstrom_max",
            "pilot_updates",
            "checkpoint_interval",
            "evaluation_lengths",
            "generated_evaluation_count",
            "generated_samples_per_length",
        ),
        "optimizer": ("learning_rate", "weight_decay", "gradient_clip_norm"),
        "loss_weights": (
            "internal_angle_torsion",
            "kabsch_coordinate",
            "i_plus_2",
            "i_plus_3",
            "long_range_pair",
            "contact_map",
            "chirality",
            "radius_of_gyration",
        ),
        "go_no_go": (
            "adjacent_distance_rmse_angstrom_max",
            "locally_valid_bond_fraction_min",
            "locally_valid_residue_fraction_improvement_min",
            "i_plus_2_rmse_relative_improvement_min",
            "i_plus_3_rmse_relative_improvement_min",
            "global_safeguards",
            "median_coordinate_displacement",
            "inference_overhead_fraction_of_500_step_sampler_max",
            "maximum_radius_of_gyration_relative_degradation",
            "maximum_long_range_contact_density_absolute_change",
            "maximum_chirality_agreement_decrease",
            "minimum_generated_diversity_retention_fraction",
        ),
    }
    missing = [key for key in ("seed", "output_dir") if key not in cfg]
    for section, keys in required.items():
        if section not in cfg or not isinstance(cfg[section], dict):
            missing.append(section)
        else:
            missing.extend(f"{section}.{key}" for key in keys if key not in cfg[section])
    if missing:
        raise ValueError(f"E008 config schema missing required fields: {', '.join(missing)}")
    threshold = cfg["milestones"]["tiny_overfit_coordinate_rmse_angstrom_max"]
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(threshold)
        or threshold <= 0
    ):
        raise ValueError("milestones.tiny_overfit_coordinate_rmse_angstrom_max must be a finite positive number")


def _model(cfg: dict, device: torch.device) -> BackboneKinematicDecoder:
    model = BackboneKinematicDecoder(**cfg["decoder"]).to(device)
    return model


def _structure_classes(path: str | None) -> dict[str, dict]:
    if not path or not Path(path).is_file():
        raise ValueError("pilot requires a generated E008 manifest or optional manual manifest")
    labels: dict[str, dict] = {}
    with open(path, newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(E008_MANIFEST_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"E008 manifest lacks required columns: {sorted(missing)}")
        for row in reader:
            sid, category = row.get("sample_id", ""), row.get("conformation_class", "unknown").lower()
            if category not in {"globular", "idp", "unknown", "ambiguous"} or sid in labels:
                raise ValueError("invalid or duplicate E008 conformation label")
            if category in {"globular", "idp"} and not row.get("label_source"):
                raise ValueError("authoritative conformation labels require label_source provenance")
            if row.get("selection_class") not in {"prototype_structured", "excluded"}:
                raise ValueError("invalid E008 selection_class")
            labels[sid] = row
    return labels


def _authorization(cfg: dict):
    source = yaml.safe_load(Path(cfg["real_structures"]["source_config"]).read_text())["dataset"]
    return authorize_rich_geometry_dataset(
        source["root"],
        expected_protocol_sha256=source["protocol_sha256"],
        expected_schema_sha256=source["schema_sha256"],
        expected_vocabulary_sha256=source["vocabulary_sha256"],
        expected_normalization_sha256=source["normalization_sha256"],
        expected_shard_inventory_sha256=source["shard_inventory_sha256"],
        protected_input_relocations=(
            cfg["real_structures"].get("protected_input_relocations") or source.get("protected_input_relocations")
        ),
    )


def _verify_manifest_provenance(auth, manifest: dict[str, dict]) -> None:
    """Bind each manifest identity and source hash back to the authorized sidecar."""
    import pyarrow.parquet as pq

    seen = {}
    protocol = json.loads((auth.root / "protocol.json").read_text())
    for shard in protocol["shards"]:
        if shard["dataset"] not in {"train", "validation"}:
            continue
        path = auth.root / shard["path"]
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            for row in parquet.read_row_group(group, columns=["sample_id", "source_path", "source_sha256"]).to_pylist():
                sid = str(row["sample_id"])
                if sid in seen:
                    raise ValueError(f"duplicate authorized sidecar identity: {sid}")
                seen[sid] = (str(row["source_path"]), str(row["source_sha256"]))
    if set(manifest) != set(seen):
        raise ValueError("E008 manifest identity set differs from authorized sidecars")
    for sid, row in manifest.items():
        if (row.get("source_path"), row.get("source_sha256")) != seen[sid]:
            raise ValueError(f"E008 source provenance mismatch for {sid}")


def _noise_calibration(cfg: dict, device: torch.device) -> dict[int, float]:
    """Calibrate independent coordinate noise to v_only update-500 errors."""
    source = cfg["corruption_calibration"]["observed_error_summary"]
    if _sha(source) != cfg["corruption_calibration"]["observed_error_summary_sha256"]:
        raise ValueError("frozen-prior observed-error summary hash mismatch")
    evaluations = json.loads(Path(source).read_text())
    strata = {64: "20-64", 128: "65-128", 256: "129-256", 384: "257-384", 500: "385-500"}
    values = evaluations["500"]["denoising"]["by_length_stratum"]
    errors = {length: float(values[name]["adjacent_distance_error_angstrom"]) for length, name in strata.items()}
    return {
        length: float(
            np.clip(
                error / math.sqrt(6.0),
                cfg["corruption_calibration"]["minimum_sigma_angstrom"],
                cfg["corruption_calibration"]["maximum_sigma_angstrom"],
            )
        )
        for length, error in errors.items()
    }


def _scan_indices(dataset: E007CoordinateDataset, labels: dict[str, dict], allowed: set[str]):
    """Index sidecar row IDs and lengths without loading coordinate payloads."""
    import pyarrow.parquet as pq

    indices = []
    start = 0
    for locator in dataset._locators:
        table = pq.ParquetFile(locator.path).read_row_group(locator.row_group, columns=["sample_id", "sequence"])
        for j, row in enumerate(table.to_pylist()):
            manifest_row = labels.get(row["sample_id"], {})
            category = manifest_row.get("conformation_class", "unknown")
            n = len(row["sequence"])
            if manifest_row.get("selection_class") == "prototype_structured" and category in allowed and 20 <= n <= 500:
                indices.append(
                    (start + j, row["sample_id"], n, category, manifest_row.get("experimental_method", "unknown"))
                )
        start += locator.count
    return indices


def _pad_pair(target: torch.Tensor, coarse: torch.Tensor):
    n = target.shape[0]
    m = torch.ones((1, n), dtype=torch.bool, device=target.device)
    return target[None], coarse[None], m


def _corrupt(target: torch.Tensor, sigma: float, generator: torch.Generator) -> torch.Tensor:
    noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator) * sigma
    noise -= noise.mean(0, keepdim=True)
    # Add a low-frequency component to reflect accumulated global drift.
    walk = torch.cumsum(noise, dim=0)
    walk = walk - walk.mean(0, keepdim=True)
    walk *= sigma * 0.12 / walk.square().mean().sqrt().clamp_min(1e-6)
    return target + noise + walk


def _panel_rows(dataset, candidates, count: int, seed: int):
    rng = random.Random(seed)
    chosen = rng.sample(candidates, min(count, len(candidates)))
    rows = []
    for index, _, _, category, method in chosen:
        row = dataset[index]
        row["conformation_class"] = category
        row["experimental_method"] = method
        rows.append(row)
    return rows


def _stratified_tiny_pool(candidates, seed: int):
    ranges = ((20, 64), (65, 128), (129, 256), (257, 384), (385, 500))
    quotas = (6, 6, 7, 6, 7)
    rng = random.Random(seed)
    selected = []
    for (lo, hi), quota in zip(ranges, quotas, strict=True):
        rows = [r for r in candidates if lo <= r[2] <= hi]
        if len(rows) < quota:
            raise ValueError(f"tiny-overfit panel underfilled at lengths {lo}-{hi}")
        selected.extend(rng.sample(rows, quota))
    return selected


@torch.no_grad()
def _evaluate_real_panel(decoder, dataset, pool, calibration, device, *, count: int, seed: int):
    rows = _panel_rows(dataset, pool, count, seed)
    g = torch.Generator(device=device).manual_seed(seed + 1)
    decoder.eval()
    records = []
    for row in rows:
        target = row["coordinates"].to(device).float()
        sigma = calibration[min(calibration, key=lambda x: abs(x - len(target)))]
        coarse = _corrupt(target, sigma, g)
        tb, cb, mask = _pad_pair(target, coarse)
        pred = decoder(cb, mask)["coordinates"]
        losses = geometry_native_losses(pred, tb, cb, mask)

        def local_validity(xyz):
            bonds = torch.linalg.vector_norm(xyz[1:] - xyz[:-1], dim=-1)
            valid = (bonds >= 3.6) & (bonds <= 4.0)
            residue_valid = valid[:-1] & valid[1:]
            return float(valid.float().mean()), float(residue_valid.float().mean())

        coarse_bond, coarse_residue = local_validity(coarse)
        pred_bond, pred_residue = local_validity(pred[0])

        def global_descriptors(xyz):
            d = torch.cdist(xyz.float()[None], xyz.float()[None])[0]
            sep = torch.arange(len(xyz), device=device)
            long_pair = (sep[:, None] - sep[None, :]).abs() >= 8
            contacts = ((d < 8.0) & long_pair).sum().float() / long_pair.sum().clamp_min(1)
            rg = torch.linalg.vector_norm(xyz - xyz.mean(0), dim=-1).square().mean().sqrt()
            return float(rg), float(contacts)

        coarse_rg, coarse_contacts = global_descriptors(coarse)
        pred_rg, pred_contacts = global_descriptors(pred[0])
        _, _, target_torsions = cartesian_to_internal(target)
        _, _, coarse_torsions = cartesian_to_internal(coarse)
        _, _, pred_torsions = cartesian_to_internal(pred[0])
        coarse_chirality = float((torch.cos(coarse_torsions - target_torsions) > 0).float().mean())
        pred_chirality = float((torch.cos(pred_torsions - target_torsions) > 0).float().mean())
        i2_target = torch.linalg.vector_norm(target[2:] - target[:-2], dim=-1)
        i3_target = torch.linalg.vector_norm(target[3:] - target[:-3], dim=-1)
        i2_coarse = torch.linalg.vector_norm(coarse[2:] - coarse[:-2], dim=-1)
        i3_coarse = torch.linalg.vector_norm(coarse[3:] - coarse[:-3], dim=-1)
        i2_pred = torch.linalg.vector_norm(pred[0, 2:] - pred[0, :-2], dim=-1)
        i3_pred = torch.linalg.vector_norm(pred[0, 3:] - pred[0, :-3], dim=-1)
        records.append(
            {
                "sample_id": row["sample_id"],
                "length": len(target),
                "conformation_class": row.get("conformation_class", "unknown"),
                "experimental_method": row.get("experimental_method", "unknown"),
                "coarse_to_target_rmse": float(
                    torch.linalg.vector_norm(coarse - target, dim=-1).square().mean().sqrt()
                ),
                "decoded_to_target_rmse": float(
                    torch.linalg.vector_norm(pred[0] - target, dim=-1).square().mean().sqrt()
                ),
                "coarse_valid_bond_fraction": coarse_bond,
                "decoded_valid_bond_fraction": pred_bond,
                "coarse_valid_residue_fraction": coarse_residue,
                "decoded_valid_residue_fraction": pred_residue,
                "coarse_radius_of_gyration": coarse_rg,
                "decoded_radius_of_gyration": pred_rg,
                "coarse_long_range_contact_density": coarse_contacts,
                "decoded_long_range_contact_density": pred_contacts,
                "coarse_chirality_agreement": coarse_chirality,
                "decoded_chirality_agreement": pred_chirality,
                "coarse_i_plus_2_rmse": float((i2_coarse - i2_target).square().mean().sqrt()),
                "decoded_i_plus_2_rmse": float((i2_pred - i2_target).square().mean().sqrt()),
                "coarse_i_plus_3_rmse": float((i3_coarse - i3_target).square().mean().sqrt()),
                "decoded_i_plus_3_rmse": float((i3_pred - i3_target).square().mean().sqrt()),
                "coarse_global_losses": {k: float(v) for k, v in geometry_native_losses(cb, tb, cb, mask).items()},
                "decoded_global_losses": {k: float(v) for k, v in losses.items()},
            }
        )
    return records


def _roundtrip_rows(rows: list[dict]) -> float:
    maximum = 0.0
    for row in rows:
        xyz = row["coordinates"].double()
        bonds, angles, torsions = cartesian_to_internal(xyz)
        rebuilt = internal_to_cartesian(xyz[:3], angles, torsions, bond_lengths=bonds)
        maximum = max(maximum, float(torch.linalg.vector_norm(rebuilt - xyz, dim=-1).max()))
    return maximum


def _atomic_checkpoint(path: Path, payload: dict) -> str:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, tmp)
    digest = _sha(tmp)
    os.replace(tmp, path)
    return digest


def _atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def _evaluate_generator(cfg: dict, decoder: BackboneKinematicDecoder, device: torch.device, output: Path):
    from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import EquivariantPairCoordinateUNet
    from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion

    source = yaml.safe_load(Path(cfg["frozen_prior"]["generator_config"]).read_text())
    prior = torch.load(cfg["frozen_prior"]["checkpoint"], map_location=device, weights_only=False)
    generator = EquivariantPairCoordinateUNet(**source["model"]).to(device).eval()
    generator.load_state_dict(prior["model"], strict=True)
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    diffusion = CoordinateVPDiffusion(int(cfg["frozen_prior"]["diffusion_steps"]))
    scale = float(cfg["frozen_prior"]["coordinate_scale_angstrom"])
    records, raw_seconds, decode_seconds, diversity_coords = [], [], [], {"coarse": {}, "decoded": {}}
    per_length = int(cfg["milestones"]["generated_samples_per_length"])
    for li, length in enumerate(cfg["milestones"]["evaluation_lengths"]):
        for si in range(per_length):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            sample = diffusion.sample(generator, length=length, seed=cfg["seed"] + li * 1000 + si, device=device)
            torch.cuda.synchronize(device)
            raw_seconds.append(time.perf_counter() - t0)
            coarse = sample["coordinates"] * scale
            mask = sample["residue_mask"]
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                repaired = decoder(coarse, mask)["coordinates"]
            torch.cuda.synchronize(device)
            decode_seconds.append(time.perf_counter() - t0)
            for name, xyz in (("coarse", coarse[0]), ("decoded", repaired[0])):
                bonds = torch.linalg.vector_norm(xyz[1:] - xyz[:-1], dim=-1)
                valid_bonds = (bonds >= 3.6) & (bonds <= 4.0)
                valid_residues = valid_bonds[:-1] & valid_bonds[1:]
                d2 = torch.linalg.vector_norm(xyz[2:] - xyz[:-2], dim=-1)
                d3 = torch.linalg.vector_norm(xyz[3:] - xyz[:-3], dim=-1)
                rg = torch.linalg.vector_norm(xyz - xyz.mean(0), dim=-1).square().mean().sqrt()
                contacts = torch.cdist(xyz.float(), xyz.float()) < 8.0
                sep = torch.arange(length, device=device)
                contacts &= (sep[:, None] - sep[None, :]).abs() >= 8
                diversity_coords[name].setdefault(length, []).append(xyz.detach().float().cpu())
                displacement = (
                    float(torch.linalg.vector_norm(xyz - coarse[0], dim=-1).median()) if name == "decoded" else 0.0
                )
                _, _, pseudo = cartesian_to_internal(xyz)
                _, _, coarse_pseudo = cartesian_to_internal(coarse[0])
                chirality_retention = float((torch.cos(pseudo - coarse_pseudo) > 0).float().mean())
                records.append(
                    {
                        "length": length,
                        "sample": si,
                        "arm": name,
                        "adjacent_rmse": float((bonds - 3.8).square().mean().sqrt()),
                        "valid_bond_fraction": float(valid_bonds.float().mean()),
                        "valid_residue_fraction": float(valid_residues.float().mean()),
                        "i_plus_2_mean_distance": float(d2.mean()),
                        "i_plus_3_mean_distance": float(d3.mean()),
                        "radius_of_gyration": float(rg),
                        "contact_density": float(
                            contacts.sum() / max(int((sep[:, None] - sep[None, :]).abs().ge(8).sum()), 1)
                        ),
                        "chirality_retention_fraction": chirality_retention,
                        "median_coordinate_displacement_angstrom": displacement,
                    }
                )
    diversity = {}
    for length in diversity_coords["coarse"]:
        summaries = {}
        for arm in ("coarse", "decoded"):
            structures = diversity_coords[arm][length]
            fingerprints = []
            for xyz in structures:
                stride = max(1, length // 50)
                landmarks = xyz[::stride][:50]
                fingerprints.append(torch.cdist(landmarks[None], landmarks[None])[0].reshape(-1))
            pair_differences = [
                torch.linalg.vector_norm(fingerprints[i] - fingerprints[j]) / math.sqrt(fingerprints[i].numel())
                for i in range(len(fingerprints))
                for j in range(i + 1, len(fingerprints))
            ]
            summaries[arm] = float(torch.stack(pair_differences).mean())
        diversity[length] = {
            "coarse_descriptor_spread": summaries["coarse"],
            "decoded_descriptor_spread": summaries["decoded"],
            "retention_fraction": summaries["decoded"] / max(summaries["coarse"], 1e-9),
        }
    return {
        "records": records,
        "mean_sampler_seconds": float(np.mean(raw_seconds)),
        "mean_decoder_seconds": float(np.mean(decode_seconds)),
        "inference_overhead_fraction": float(np.mean(decode_seconds) / max(np.mean(raw_seconds), 1e-9)),
        "diversity_by_length": diversity,
        "median_coordinate_displacement_angstrom": float(
            np.median([r["median_coordinate_displacement_angstrom"] for r in records if r["arm"] == "decoded"])
        ),
    }


def _go_no_go(cfg: dict, evaluation: dict, prospective: list[dict]) -> dict:
    thresholds = cfg["go_no_go"]
    decoded = [r for r in evaluation["records"] if r["arm"] == "decoded"]
    valid_bond = float(np.mean([r["valid_bond_fraction"] for r in decoded]))
    adjacent_rmse = float(np.mean([r["adjacent_rmse"] for r in decoded]))
    residue_delta = float(
        np.mean([r["decoded_valid_residue_fraction"] - r["coarse_valid_residue_fraction"] for r in prospective])
    )
    i2_gain = float(
        np.mean(
            [
                (r["coarse_i_plus_2_rmse"] - r["decoded_i_plus_2_rmse"]) / max(r["coarse_i_plus_2_rmse"], 1e-6)
                for r in prospective
            ]
        )
    )
    i3_gain = float(
        np.mean(
            [
                (r["coarse_i_plus_3_rmse"] - r["decoded_i_plus_3_rmse"]) / max(r["coarse_i_plus_3_rmse"], 1e-6)
                for r in prospective
            ]
        )
    )
    rg_relative = float(
        np.mean(
            [
                abs(r["decoded_radius_of_gyration"] - r["coarse_radius_of_gyration"])
                / max(r["coarse_radius_of_gyration"], 1e-6)
                for r in prospective
            ]
        )
    )
    contact_change = float(
        np.mean([r["decoded_long_range_contact_density"] - r["coarse_long_range_contact_density"] for r in prospective])
    )
    chirality_decrease = float(
        np.mean([r["coarse_chirality_agreement"] - r["decoded_chirality_agreement"] for r in prospective])
    )
    diversity_retention = min(x["retention_fraction"] for x in evaluation["diversity_by_length"].values())
    checks = {
        "adjacent_distance_rmse": adjacent_rmse <= thresholds["adjacent_distance_rmse_angstrom_max"],
        "locally_valid_bond_fraction": valid_bond >= thresholds["locally_valid_bond_fraction_min"],
        "locally_valid_residue_improvement": residue_delta
        >= thresholds["locally_valid_residue_fraction_improvement_min"],
        "i_plus_2_improved": i2_gain >= thresholds["i_plus_2_rmse_relative_improvement_min"],
        "i_plus_3_improved": i3_gain >= thresholds["i_plus_3_rmse_relative_improvement_min"],
        "radius_of_gyration": rg_relative <= thresholds["maximum_radius_of_gyration_relative_degradation"],
        "long_range_contacts": abs(contact_change) <= thresholds["maximum_long_range_contact_density_absolute_change"],
        "chirality": chirality_decrease <= thresholds["maximum_chirality_agreement_decrease"],
        "diversity": diversity_retention >= thresholds["minimum_generated_diversity_retention_fraction"],
        "inference_overhead": evaluation["inference_overhead_fraction"]
        <= thresholds["inference_overhead_fraction_of_500_step_sampler_max"],
    }
    return {
        "decision": "go" if all(checks.values()) else "no_go",
        "checks": checks,
        "effects": {
            "adjacent_rmse_angstrom": adjacent_rmse,
            "valid_bond_fraction": valid_bond,
            "valid_residue_fraction_improvement": residue_delta,
            "i_plus_2_rmse_relative_improvement": i2_gain,
            "i_plus_3_rmse_relative_improvement": i3_gain,
            "mean_radius_of_gyration_relative_change": rg_relative,
            "long_range_contact_density_change": contact_change,
            "chirality_agreement_decrease": chirality_decrease,
            "minimum_diversity_retention_fraction": diversity_retention,
            "median_coordinate_displacement_angstrom": evaluation["median_coordinate_displacement_angstrom"],
            "inference_overhead_fraction": evaluation["inference_overhead_fraction"],
        },
    }


def _pilot(cfg: dict, resume: bool, tiny_overfit_only: bool = False) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("E008 pilot requires CUDA")
    labels = _structure_classes(cfg["real_structures"].get("structure_class_manifest"))
    auth = _authorization(cfg)
    _verify_manifest_provenance(auth, labels)
    train_data = E007CoordinateDataset(auth, split="train")
    val_data = E007CoordinateDataset(auth, split="validation")
    allowed = {"globular", "unknown"}
    train_pool = _scan_indices(train_data, labels, allowed)
    val_pool = _scan_indices(val_data, labels, allowed)
    if not train_pool or not val_pool:
        raise ValueError("curated structure-class manifest has no usable train/validation rows")
    import pyarrow.parquet as pq

    clean = pq.read_table(
        cfg["real_structures"]["clean_validation_manifest"], columns=["sample_id", "cluster_id", "coordinate_accepted"]
    ).to_pylist()
    id_to_cluster = {r["sample_id"]: r["cluster_id"] for r in clean if r["coordinate_accepted"]}
    val_pool = [r for r in val_pool if r[1] in id_to_cluster]
    clusters = sorted({id_to_cluster[r[1]] for r in val_pool})
    ranked = sorted(
        clusters,
        key=lambda c: hashlib.sha256(
            f"{cfg['real_structures']['validation_cluster_split_seed']}|{c}".encode()
        ).hexdigest(),
    )
    dev_clusters = set(ranked[: int(len(ranked) * cfg["real_structures"]["development_fraction"])])
    dev_pool = [r for r in val_pool if id_to_cluster[r[1]] in dev_clusters]
    prospective_pool = [r for r in val_pool if id_to_cluster[r[1]] not in dev_clusters]
    if not dev_pool or not prospective_pool:
        raise ValueError("identity-cluster split produced an empty panel")
    device = torch.device("cuda")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    decoder = _model(cfg, device)
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=cfg["optimizer"]["learning_rate"], weight_decay=cfg["optimizer"]["weight_decay"]
    )
    output = Path(cfg["output_dir"])
    if output.exists() and not resume and any(output.iterdir()):
        raise FileExistsError("E008 output directory already contains a run; use --resume or choose a fresh output")
    output.mkdir(parents=True, exist_ok=True)
    config_sha = cfg["_config_sha256"]
    class_manifest_sha = _sha(cfg["real_structures"]["structure_class_manifest"])
    checkpoint_path = output / "checkpoint.pt"
    update = 0
    resume_state = None
    if resume:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        recorded = (output / "checkpoint.sha256").read_text().strip()
        required_state = {
            "config_sha256",
            "update",
            "decoder",
            "optimizer",
            "python_rng",
            "numpy_rng",
            "torch_rng",
            "cuda_rng",
            "corruption_rng",
            "class_manifest_sha256",
            "panel_sha256",
            "tiny_overfit_coordinate_rmse_angstrom",
            "training_metrics_sha256",
        }
        if (
            not required_state.issubset(state)
            or state["config_sha256"] != config_sha
            or state["class_manifest_sha256"] != class_manifest_sha
            or _sha(checkpoint_path) != recorded
        ):
            raise ValueError("resume checkpoint integrity/configuration mismatch")
        decoder.load_state_dict(state["decoder"])
        optimizer.load_state_dict(state["optimizer"])
        update = int(state["update"])
        resume_state = state
    roundtrip = _roundtrip_rows(
        _panel_rows(train_data, train_pool, cfg["milestones"]["real_structure_roundtrip_count"], cfg["seed"])
    )
    if roundtrip > 1e-4:
        raise ValueError(f"real-structure internal-coordinate roundtrip failed: {roundtrip}")
    calibration = _noise_calibration(cfg, device)
    metric_path = output / "training_metrics.jsonl"
    max_updates = int(cfg["milestones"]["pilot_updates"])
    tiny_updates = int(cfg["milestones"]["tiny_overfit_updates"])
    rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 11)
    train_step_total = tiny_updates + max_updates
    tiny_pool = _stratified_tiny_pool(train_pool, cfg["seed"] + 3)
    ranges = ((20, 64), (65, 128), (129, 256), (257, 384), (385, 500))
    train_by_stratum = [[r for r in train_pool if lo <= r[2] <= hi] for lo, hi in ranges]
    if any(not rows for rows in train_by_stratum):
        raise ValueError("length-balanced training pool has an empty stratum")
    tiny_cache = []
    for item in tiny_pool:
        row = train_data[item[0]]
        target = row["coordinates"].to(device).float()
        sigma = calibration[min(calibration, key=lambda x: abs(x - len(target)))]
        tiny_cache.append((row, target, _corrupt(target, sigma, rng)))
    tiny_overfit_rmse = None if resume_state is None else resume_state.get("tiny_overfit_coordinate_rmse_angstrom")
    panel_sha = hashlib.sha256(
        json.dumps(
            {
                "train": [r[1] for r in train_pool],
                "development": [r[1] for r in dev_pool],
                "prospective": [r[1] for r in prospective_pool],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if resume_state is not None:
        if resume_state["panel_sha256"] != panel_sha:
            raise ValueError("resume panel identity hash mismatch")
        random.setstate(resume_state["python_rng"])
        np.random.set_state(resume_state["numpy_rng"])
        torch.set_rng_state(resume_state["torch_rng"])
        torch.cuda.set_rng_state_all(resume_state["cuda_rng"])
        rng.set_state(resume_state["corruption_rng"])
        if update >= tiny_updates and tiny_overfit_rmse is not None:
            gate = float(cfg["milestones"]["tiny_overfit_coordinate_rmse_angstrom_max"])
            if tiny_overfit_rmse > gate:
                raise RuntimeError(f"persisted tiny-overfit gate failed: {tiny_overfit_rmse:.4f} A > {gate:.4f} A")
        if metric_path.exists():
            if _sha(metric_path) != resume_state["training_metrics_sha256"]:
                raise ValueError("resume training metrics hash does not match checkpoint boundary")
            all_rows = [json.loads(line) for line in metric_path.read_text().splitlines()]
            committed_rows = [row for row in all_rows if int(row["update"]) <= update]
            if [int(row["update"]) for row in committed_rows] != list(range(1, update + 1)):
                raise ValueError("resume training-update identity sequence is incomplete or duplicated")
            if any(
                row["stage"] != ("tiny_overfit_32" if row["update"] <= tiny_updates else "bounded_pilot")
                for row in committed_rows
            ):
                raise ValueError("resume training-stage identities mismatch checkpoint update")
            train_identities = {row[1] for row in train_pool}
            if any(row["sample_id"] not in train_identities for row in committed_rows):
                raise ValueError("resume training sample identities differ from authorized training panel")
            kept = [json.dumps(row, sort_keys=True, allow_nan=False) for row in committed_rows]
            temp_metrics = metric_path.with_suffix(".tmp")
            temp_metrics.write_text("\n".join(kept) + ("\n" if kept else ""))
            os.replace(temp_metrics, metric_path)
        elif update:
            raise ValueError("resume checkpoint has no matching training-update identity log")
    source_dataset = yaml.safe_load(Path(cfg["real_structures"]["source_config"]).read_text())["dataset"]
    _atomic_json(
        output / "run_manifest.json",
        {
            "config_sha256": config_sha,
            "checkpoint_sha256": cfg["frozen_prior"]["checkpoint_sha256"],
            "class_manifest_path": cfg["real_structures"]["structure_class_manifest"],
            "class_manifest_sha256": class_manifest_sha,
            "identity30_validation_manifest_sha256": cfg["real_structures"]["clean_validation_manifest_sha256"],
            "dataset_hash_pins": {
                k: source_dataset[k]
                for k in (
                    "protocol_sha256",
                    "schema_sha256",
                    "vocabulary_sha256",
                    "normalization_sha256",
                    "shard_inventory_sha256",
                )
            },
            "corruption_calibration_sha256": cfg["corruption_calibration"]["observed_error_summary_sha256"],
            "panel_sha256": panel_sha,
            "decoder_parameter_count": decoder_parameter_count(decoder),
            "training_split": "authorized_sidecar_train",
            "development_and_prospective_split": "disjoint_identity30_validation_clusters",
        },
    )
    _atomic_json(
        output / "panel_manifest.json",
        {
            "training_sample_ids": [r[1] for r in train_pool],
            "development_sample_ids": [r[1] for r in dev_pool],
            "prospective_sample_ids": [r[1] for r in prospective_pool],
            "idp_sample_ids_excluded": [sid for sid, row in labels.items() if row.get("conformation_class") == "idp"],
        },
    )
    update_limit = tiny_updates if tiny_overfit_only else train_step_total
    while update < update_limit:
        tiny_stage = update < tiny_updates
        if tiny_stage:
            row, target, coarse = random.choice(tiny_cache)
            sigma = calibration[min(calibration, key=lambda x: abs(x - len(target)))]
        else:
            stratum = (update - tiny_updates) % len(train_by_stratum)
            selected = random.choice(train_by_stratum[stratum])
            row = train_data[selected[0]]
            target = row["coordinates"].to(device).float()
            n = target.shape[0]
            sigma = calibration[min(calibration, key=lambda x: abs(x - n))]
            coarse = _corrupt(target, sigma, rng)
        n = target.shape[0]
        target_b, coarse_b, mask = _pad_pair(target, coarse)
        decoder.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = decoder(coarse_b, mask)["coordinates"]
        losses = geometry_native_losses(prediction, target_b, coarse_b, mask)
        weights = cfg["loss_weights"]
        loss = (
            weights["internal_angle_torsion"] * losses["internal"]
            + weights["kabsch_coordinate"] * losses["kabsch_coordinate"]
            + weights["i_plus_2"] * losses["i_plus_2"]
            + weights["i_plus_3"] * losses["i_plus_3"]
            + weights["long_range_pair"] * losses["long_range_pair"]
            + weights["contact_map"] * losses["contact_map"]
            + weights["chirality"] * losses["chirality"]
            + weights["radius_of_gyration"] * losses["radius_of_gyration"]
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite decoder loss at update {update + 1}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), cfg["optimizer"]["gradient_clip_norm"])
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite decoder gradient at update {update + 1}")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in decoder.parameters()):
            raise FloatingPointError(f"non-finite decoder parameter at update {update + 1}")
        update += 1
        record = {
            "update": update,
            "stage": "tiny_overfit_32" if tiny_stage else "bounded_pilot",
            "sample_id": row["sample_id"],
            "length": n,
            "loss": float(loss.detach()),
            "gradient_norm": float(grad_norm),
            "losses": {k: float(v.detach()) for k, v in losses.items()},
            "corruption_sigma_angstrom": sigma,
        }
        with open(metric_path, "a") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        if update == tiny_updates:
            decoder.eval()
            tiny_errors = []
            with torch.no_grad():
                for _, tiny_target, tiny_coarse in tiny_cache:
                    tb, cb, mb = _pad_pair(tiny_target, tiny_coarse)
                    tiny_errors.append(
                        float(geometry_native_losses(decoder(cb, mb)["coordinates"], tb, cb, mb)["kabsch_coordinate"])
                    )
            tiny_overfit_rmse = float(np.mean(tiny_errors))
            payload = {
                "config_sha256": config_sha,
                "update": update,
                "decoder": decoder.state_dict(),
                "class_manifest_sha256": class_manifest_sha,
                "panel_sha256": panel_sha,
                "tiny_overfit_coordinate_rmse_angstrom": tiny_overfit_rmse,
                "training_metrics_sha256": _sha(metric_path),
                "optimizer": optimizer.state_dict(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "corruption_rng": rng.get_state(),
                "training_split": "sidecar_train",
                "authorizes_frozen_prior_training": False,
            }
            digest = _atomic_checkpoint(checkpoint_path, payload)
            digest_tmp = output / f".checkpoint.sha256.{os.getpid()}.tmp"
            digest_tmp.write_text(digest + "\n")
            os.replace(digest_tmp, output / "checkpoint.sha256")
            gate = float(cfg["milestones"]["tiny_overfit_coordinate_rmse_angstrom_max"])
            _atomic_json(
                output / "tiny_overfit_result.json",
                {
                    "status": "passed" if tiny_overfit_rmse <= gate else "failed",
                    "completed_updates": update,
                    "tiny_overfit_structure_count": len(tiny_pool),
                    "coordinate_rmse_angstrom": tiny_overfit_rmse,
                    "predeclared_threshold_angstrom_max": gate,
                    "config_sha256": config_sha,
                    "checkpoint_sha256": digest,
                    "panel_sha256": panel_sha,
                    "authorizes_2000_update_pilot": tiny_overfit_rmse <= gate,
                },
            )
            if tiny_overfit_rmse > gate:
                raise RuntimeError(f"tiny-overfit gate failed: {tiny_overfit_rmse:.4f} A > {gate:.4f} A")
        if update % int(cfg["milestones"]["checkpoint_interval"]) == 0 or update == train_step_total:
            payload = {
                "config_sha256": config_sha,
                "update": update,
                "decoder": decoder.state_dict(),
                "class_manifest_sha256": class_manifest_sha,
                "panel_sha256": panel_sha,
                "tiny_overfit_coordinate_rmse_angstrom": tiny_overfit_rmse,
                "training_metrics_sha256": _sha(metric_path),
                "optimizer": optimizer.state_dict(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "corruption_rng": rng.get_state(),
                "training_split": "sidecar_train",
                "authorizes_frozen_prior_training": False,
            }
            digest = _atomic_checkpoint(checkpoint_path, payload)
            digest_tmp = output / f".checkpoint.sha256.{os.getpid()}.tmp"
            digest_tmp.write_text(digest + "\n")
            os.replace(digest_tmp, output / "checkpoint.sha256")
    if tiny_overfit_only:
        return json.loads((output / "tiny_overfit_result.json").read_text())
    # Frozen-prior samples are generated only for the required prospective evaluation.
    dev_records = _evaluate_real_panel(
        decoder, val_data, dev_pool, calibration, device, count=min(100, len(dev_pool)), seed=cfg["seed"] + 31
    )
    prospective_records = _evaluate_real_panel(
        decoder,
        val_data,
        prospective_pool,
        calibration,
        device,
        count=min(100, len(prospective_pool)),
        seed=cfg["seed"] + 97,
    )
    evaluation = _evaluate_generator(cfg, decoder.eval(), device, output)
    # This is a decoder-feasibility decision over the predeclared selection
    # cohort; prototype geometry is never reported as a biological class label.
    decision = _go_no_go(cfg, evaluation, prospective_records)
    class_eval = {
        category: [r for r in prospective_records if r["conformation_class"] == category]
        for category in ("globular", "idp", "unknown")
    }
    report = {
        "status": "completed_non_authorizing",
        "parameter_count": decoder_parameter_count(decoder),
        "architecture": "BackboneKinematicDecoder(width=48,layers=3,pair_rbf_bins=8,fixed_bond=3.8A)",
        "training_update_count": max_updates,
        "tiny_overfit_update_count": tiny_updates,
        "total_decoder_optimizer_updates": max_updates + tiny_updates,
        "frozen_prior_optimizer_updates": 0,
        "evaluation_counts": {
            "real_roundtrips": cfg["milestones"]["real_structure_roundtrip_count"],
            "tiny_overfit_structures": len(tiny_pool),
            "development": len(dev_records),
            "prospective": len(prospective_records),
            "frozen_prior_generations": len(evaluation["records"]) // 2,
        },
        "identity_cluster_counts": {"development": len(dev_clusters), "prospective": len(clusters) - len(dev_clusters)},
        "conformation_class_counts": {
            key: sum(1 for category in labels.values() if category.get("conformation_class") == key)
            for key in ("globular", "idp", "unknown")
        },
        "roundtrip_max_error_angstrom": roundtrip,
        "tiny_overfit_coordinate_rmse_angstrom": tiny_overfit_rmse,
        "development_records": dev_records,
        "prospective_records": prospective_records,
        "prospective_conformation_class_counts": {key: len(rows) for key, rows in class_eval.items()},
        "go_no_go_population": (
            "identity-disjoint prospective prototype_structured selection; not a biological globular annotation"
        ),
        "evaluation": evaluation,
        "go_no_go": decision,
        "criteria": cfg["go_no_go"],
        "frozen_prior_v_mse_change": 0.0,
        "training_numerical_stability_passed": True,
        "authorizes_production": False,
        "authorizes_downstream_generation": False,
        "authorizes_full_generative_retraining": False,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def _identity_plan(cfg: dict) -> dict:
    """Resolve exact disjoint identities without constructing a model or doing training."""
    manifest = _structure_classes(cfg["real_structures"].get("structure_class_manifest"))
    auth = _authorization(cfg)
    _verify_manifest_provenance(auth, manifest)
    train = _scan_indices(E007CoordinateDataset(auth, split="train"), manifest, {"globular", "unknown"})
    valid = _scan_indices(E007CoordinateDataset(auth, split="validation"), manifest, {"globular", "unknown"})
    import pyarrow.parquet as pq

    clean = pq.read_table(
        cfg["real_structures"]["clean_validation_manifest"], columns=["sample_id", "cluster_id", "coordinate_accepted"]
    ).to_pylist()
    id_to_cluster = {r["sample_id"]: r["cluster_id"] for r in clean if r["coordinate_accepted"]}
    identity30_excluded = sorted(r[1] for r in valid if r[1] not in id_to_cluster)
    valid = [r for r in valid if r[1] in id_to_cluster]
    clusters = sorted({id_to_cluster[r[1]] for r in valid})
    ranked = sorted(
        clusters,
        key=lambda c: hashlib.sha256(
            f"{cfg['real_structures']['validation_cluster_split_seed']}|{c}".encode()
        ).hexdigest(),
    )
    ndev = int(len(ranked) * cfg["real_structures"]["development_fraction"])
    dev_clusters = set(ranked[:ndev])
    dev = [r for r in valid if id_to_cluster[r[1]] in dev_clusters]
    prospective = [r for r in valid if id_to_cluster[r[1]] not in dev_clusters]
    groups = {"training": train, "development": dev, "prospective": prospective}
    if set(r[1] for r in train) & (set(r[1] for r in dev) | set(r[1] for r in prospective)):
        raise ValueError("E008 split identities overlap")

    def dimensions(rows):
        return {
            "count": len(rows),
            "conformation_class_counts": {k: sum(r[3] == k for r in rows) for k in ("globular", "idp", "unknown")},
            "experimental_method_counts": {k: sum(r[4] == k for r in rows) for k in sorted({r[4] for r in rows})},
            "length_stratum_counts": {
                f"{lo}-{hi}": sum(lo <= r[2] <= hi for r in rows)
                for lo, hi in ((20, 64), (65, 128), (129, 256), (257, 384), (385, 500))
            },
        }

    return {
        "manifest_sha256": _sha(cfg["real_structures"]["structure_class_manifest"]),
        "selection_class": "prototype_structured",
        "groups": {name: {"sample_ids": [r[1] for r in rows], **dimensions(rows)} for name, rows in groups.items()},
        "exclusion_count": sum(row.get("selection_class") != "prototype_structured" for row in manifest.values()),
        "identity30_validation_exclusion_count": len(identity30_excluded),
        "identity30_validation_exclusion_sample_ids": identity30_excluded,
        "identity_disjoint": True,
        "scope_note": (
            "prototype_structured is a decoder-feasibility selection, not a definitive biological globular annotation"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e008_geometry_native_decoder.yaml")
    parser.add_argument(
        "--structure-class-manifest", help="optional eight-column local E008 manifest input for pilot/validation"
    )
    parser.add_argument("--manifest-output", help="output CSV path for --build-manifest")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--cuda-smoke", action="store_true")
    modes.add_argument("--pilot", action="store_true")
    modes.add_argument("--tiny-overfit-only", action="store_true")
    modes.add_argument("--resume", action="store_true")
    modes.add_argument("--build-manifest", dest="build_manifest_mode", action="store_true")
    modes.add_argument("--validate", dest="validate_mode", action="store_true")
    args = parser.parse_args()
    if args.build_manifest_mode:
        cfg = yaml.safe_load(Path(args.config).read_text())
        if cfg.get("version") != "e008_geometry_native_decoder_v1":
            raise ValueError("E008 configuration version mismatch")
        _validate_config_schema(cfg)
        target = args.manifest_output or cfg["real_structures"].get("generated_manifest")
        if not target:
            raise ValueError("set real_structures.generated_manifest or pass --structure-class-manifest as output path")
        result = _build_manifest(cfg, target)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return
    cfg = _load(args.config)
    cfg["_config_sha256"] = _sha(args.config)
    if args.structure_class_manifest and (args.pilot or args.resume or args.validate_mode):
        cfg["real_structures"]["structure_class_manifest"] = args.structure_class_manifest
    if args.validate_mode:
        result = _identity_plan(cfg)
    elif args.plan_only:
        model = _model(cfg, torch.device("cpu"))
        result = {
            "status": "prototype_manifest_ready_for_validation",
            "parameter_count": decoder_parameter_count(model),
            "maximum_length": model.max_length,
            "pilot_requires_structure_class_manifest": not bool(cfg["real_structures"].get("structure_class_manifest")),
            "cuda_smoke_command": (
                f"python scripts/run_e008_geometry_native_decoder.py --config {args.config} --cuda-smoke"
            ),
            "manifest_command": (
                f"python scripts/run_e008_geometry_native_decoder.py --config {args.config} --build-manifest"
            ),
            "pilot_command": f"python scripts/run_e008_geometry_native_decoder.py --config {args.config} --pilot",
        }
        if cfg["real_structures"].get("structure_class_manifest"):
            result["identity_plan"] = _identity_plan(cfg)
    elif args.cuda_smoke:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for longest-length smoke")
        device = torch.device("cuda")
        model = _model(cfg, device).train()
        x = torch.randn(1, 500, 3, device=device, requires_grad=True)
        mask = torch.ones(1, 500, dtype=torch.bool, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        y = model(x, mask)["coordinates"]
        y.square().mean().backward()
        torch.cuda.synchronize(device)
        result = {
            "status": "cuda_smoke_passed",
            "parameter_count": decoder_parameter_count(model),
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "elapsed_seconds": time.perf_counter() - t0,
            "bond_rmse_angstrom": float(
                (torch.linalg.vector_norm(y[0, 1:] - y[0, :-1], dim=-1) - 3.8).square().mean().sqrt()
            ),
            "authorizes_training": False,
        }
    else:
        result = _pilot(cfg, resume=args.resume, tiny_overfit_only=args.tiny_overfit_only)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
