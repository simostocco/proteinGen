"""Read-only E007 contact-stability and topology-loss diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from protein_distance_diffusion.evaluation.e007_matrix_audit import load_candidate_npz

PLAN_VERSION = "e007_contact_stability_audit_plan_v1"
AUDIT_VERSION = "e007_contact_stability_audit_v1"
COMPARISON_METHODS = ("rank3_psd_projection", "constrained_coordinate_repair")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _atomic_parquet(path: Path, rows: list[dict[str, Any]], compression: str) -> None:
    if not rows:
        raise ValueError(f"E007 contact-stability table would be empty: {path.name}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression=compression)
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolved_file(root: Path, path_value: str | Path, *, expected_sha256: str, label: str) -> Path:
    raw = Path(path_value)
    path = raw if raw.is_absolute() else root / raw
    if not path.is_file():
        raise FileNotFoundError(f"E007 contact-stability {label} is missing: {path}")
    observed = sha256_file(path)
    if observed != str(expected_sha256).lower():
        raise ValueError(f"E007 contact-stability {label} SHA-256 contradiction: {path}")
    return path


def _contained_file(directory: Path, relative: str | Path, *, label: str) -> Path:
    base = directory.resolve()
    path = (base / Path(relative)).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"E007 contact-stability {label} path escapes its source directory") from exc
    if not path.is_file():
        raise FileNotFoundError(f"E007 contact-stability {label} is missing: {path}")
    return path


def _thresholds(config: dict[str, Any]) -> list[float]:
    settings = config["contact_thresholds"]
    start = float(settings["start_angstrom"])
    stop = float(settings["stop_angstrom"])
    step = float(settings["step_angstrom"])
    count = int(round((stop - start) / step))
    values = [round(start + index * step, 10) for index in range(count + 1)]
    if step <= 0.0 or values[-1] != stop:
        raise ValueError("E007 contact threshold sweep must exactly cover start through stop")
    return values


def _upper_mask(length: int) -> np.ndarray:
    return np.triu(np.ones((length, length), dtype=bool), k=1)


def contact_confusion(
    raw_matrix: np.ndarray,
    repaired_matrix: np.ndarray,
    *,
    threshold: float,
    mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Return complete contact confusion counts and rates for one pair mask."""
    raw = np.asarray(raw_matrix, dtype=np.float64)
    repaired = np.asarray(repaired_matrix, dtype=np.float64)
    if raw.shape != repaired.shape or raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("E007 contact comparison requires equal square matrices")
    selected = _upper_mask(raw.shape[0]) if mask is None else np.asarray(mask, dtype=bool)
    if selected.shape != raw.shape:
        raise ValueError("E007 contact mask shape contradiction")
    raw_contact = raw[selected] <= float(threshold)
    repaired_contact = repaired[selected] <= float(threshold)
    tp = int(np.sum(raw_contact & repaired_contact))
    fp = int(np.sum(~raw_contact & repaired_contact))
    fn = int(np.sum(raw_contact & ~repaired_contact))
    tn = int(np.sum(~raw_contact & ~repaired_contact))
    if tp + fp + fn == 0:
        return {
            "pair_count": int(np.sum(selected)),
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
            "true_negative": tn,
            "contact_prevalence": 0.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "jaccard": 1.0,
        }
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "pair_count": int(np.sum(selected)),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "contact_prevalence": (tp + fn) / max(tp + fp + fn + tn, 1),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "jaccard": float(tp / max(tp + fp + fn, 1)),
    }


def threshold_sweep_metrics(
    raw_matrix: np.ndarray, repaired_matrix: np.ndarray, thresholds: list[float]
) -> list[dict[str, float | int]]:
    return [
        contact_confusion(raw_matrix, repaired_matrix, threshold=value) | {"threshold": value} for value in thresholds
    ]


def ambiguity_exclusion_metrics(
    raw_matrix: np.ndarray,
    repaired_matrix: np.ndarray,
    *,
    threshold: float,
    half_widths: list[float],
) -> list[dict[str, float | int]]:
    raw = np.asarray(raw_matrix, dtype=np.float64)
    upper = _upper_mask(raw.shape[0])
    rows = []
    for width in half_widths:
        mask = upper & (np.abs(raw - threshold) > float(width))
        rows.append(
            contact_confusion(raw, repaired_matrix, threshold=threshold, mask=mask)
            | {"threshold": float(threshold), "ambiguity_half_width_angstrom": float(width)}
        )
    return rows


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray, *, ranked: bool = False) -> float:
    x = _rankdata(left) if ranked else np.asarray(left, dtype=np.float64)
    y = _rankdata(right) if ranked else np.asarray(right, dtype=np.float64)
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 1.0 if np.array_equal(x, y) else 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _band_mask(values: np.ndarray, band: dict[str, Any]) -> np.ndarray:
    mask = np.ones(values.shape, dtype=bool)
    lower, upper = band.get("lower"), band.get("upper")
    if lower is not None:
        mask &= values >= float(lower) if band.get("lower_inclusive", False) else values > float(lower)
    if upper is not None:
        mask &= values <= float(upper) if band.get("upper_inclusive", False) else values < float(upper)
    return mask


def _separation_mask(separation: np.ndarray, band: dict[str, Any]) -> np.ndarray:
    mask = separation >= int(band["minimum"])
    if band.get("maximum") is not None:
        mask &= separation <= int(band["maximum"])
    return mask


def _soft_contacts(values: np.ndarray, threshold: float, temperature: float) -> np.ndarray:
    scaled = np.clip((values - threshold) / temperature, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(scaled))


def continuous_topology_metrics(
    raw_matrix: np.ndarray,
    repaired_matrix: np.ndarray,
    *,
    thresholds: list[float],
    soft_temperature: float,
    distance_ranges: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = np.asarray(raw_matrix, dtype=np.float64)
    repaired = np.asarray(repaired_matrix, dtype=np.float64)
    mask = _upper_mask(raw.shape[0])
    raw_values, repaired_values = raw[mask], repaired[mask]
    change = repaired_values - raw_values
    absolute = np.abs(change)
    sweep = threshold_sweep_metrics(raw, repaired, thresholds)
    threshold_values = np.asarray([row["threshold"] for row in sweep], dtype=np.float64)
    f1_values = np.asarray([row["f1"] for row in sweep], dtype=np.float64)
    soft = {}
    for threshold in (6.0, 8.0, 10.0):
        left = _soft_contacts(raw_values, threshold, soft_temperature)
        right = _soft_contacts(repaired_values, threshold, soft_temperature)
        soft[f"{threshold:g}"] = {
            "mean_absolute_difference": float(np.mean(np.abs(left - right))),
            "similarity": float(1.0 - np.mean(np.abs(left - right))),
            "pearson_correlation": _correlation(left, right),
        }
    range_rows = []
    for band in distance_ranges:
        selected = _band_mask(raw_values, band)
        range_rows.append(
            {
                "summary_type": "distance_change_range",
                "band": str(band["name"]),
                "pair_count": int(np.sum(selected)),
                "mean_absolute_distance_change_angstrom": (
                    float(np.mean(absolute[selected])) if np.any(selected) else None
                ),
            }
        )
    return {
        "pairwise_distance_mae_angstrom": float(np.mean(absolute)),
        "pairwise_distance_rmse_angstrom": float(np.sqrt(np.mean(np.square(change)))),
        "distance_spearman_correlation": _correlation(raw_values, repaired_values, ranked=True),
        "distance_pearson_correlation": _correlation(raw_values, repaired_values),
        "relative_frobenius_distortion": float(np.linalg.norm(change) / max(np.linalg.norm(raw_values), 1e-12)),
        "absolute_distance_change_quantiles_angstrom": {
            f"q{int(q * 100):02d}": float(np.quantile(absolute, q)) for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
        },
        "soft_contact": soft,
        "contact_f1_threshold_auc_5_to_12": float(np.trapezoid(f1_values, threshold_values) / 7.0),
    }, range_rows


def neighbourhood_metrics(
    raw_matrix: np.ndarray,
    repaired_matrix: np.ndarray,
    *,
    k_values: list[int],
    retention_threshold: float,
    maximum_worst_examples: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    raw = np.asarray(raw_matrix, dtype=np.float64)
    repaired = np.asarray(repaired_matrix, dtype=np.float64)
    n = raw.shape[0]
    residue_rows = []
    summaries: dict[str, Any] = {}
    separation_rows = []
    for requested_k in k_values:
        k = min(int(requested_k), n - 1)
        if k <= 0:
            continue
        jaccards = []
        retained_counts: dict[str, int] = {}
        lost_counts: dict[str, int] = {}
        for index in range(n):
            raw_order = [value for value in np.argsort(raw[index], kind="mergesort") if value != index][:k]
            repaired_order = [value for value in np.argsort(repaired[index], kind="mergesort") if value != index][:k]
            raw_set, repaired_set = set(raw_order), set(repaired_order)
            retained, lost, added = raw_set & repaired_set, raw_set - repaired_set, repaired_set - raw_set
            jaccard = len(retained) / max(len(raw_set | repaired_set), 1)
            jaccards.append(jaccard)
            residue_rows.append(
                {
                    "residue_index": index,
                    "position_quartile": min(4, 4 * index // max(n, 1) + 1),
                    "requested_k": int(requested_k),
                    "effective_k": k,
                    "retained_neighbour_count": len(retained),
                    "lost_neighbour_count": len(lost),
                    "added_neighbour_count": len(added),
                    "neighbourhood_jaccard": float(jaccard),
                    "below_existing_retention_threshold": bool(jaccard < retention_threshold),
                }
            )
            for _state, neighbours, destination in (
                ("retained", retained, retained_counts),
                ("lost", lost, lost_counts),
            ):
                for neighbour in neighbours:
                    separation = abs(index - neighbour)
                    key = (
                        "1"
                        if separation == 1
                        else "2_to_4"
                        if separation <= 4
                        else "5_to_11"
                        if separation <= 11
                        else "12_to_23"
                        if separation <= 23
                        else "24_or_more"
                    )
                    destination[key] = destination.get(key, 0) + 1
        values = np.asarray(jaccards, dtype=np.float64)
        summaries[str(requested_k)] = {
            "effective_k": k,
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "quantile_05": float(np.quantile(values, 0.05)),
            "quantile_10": float(np.quantile(values, 0.10)),
            "quantile_25": float(np.quantile(values, 0.25)),
            "fraction_below_existing_threshold": float(np.mean(values < retention_threshold)),
            "worst_residues": [
                {"residue_index": int(index), "neighbourhood_jaccard": float(values[index])}
                for index in np.lexsort((np.arange(n), values))[:maximum_worst_examples]
            ],
        }
        for state, counts in (("retained", retained_counts), ("lost", lost_counts)):
            for band, count in sorted(counts.items()):
                separation_rows.append(
                    {
                        "summary_type": "neighbour_sequence_separation",
                        "requested_k": int(requested_k),
                        "transition": state,
                        "band": band,
                        "count": count,
                    }
                )
    return residue_rows, summaries, separation_rows


def _localization(
    raw: np.ndarray,
    repaired: np.ndarray,
    *,
    threshold: float,
    top_fraction: float,
    maximum_examples: int,
    localized_share_threshold: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    raw_contacts = raw <= threshold
    repaired_contacts = repaired <= threshold
    np.fill_diagonal(raw_contacts, False)
    np.fill_diagonal(repaired_contacts, False)
    lost = raw_contacts & ~repaired_contacts
    added = ~raw_contacts & repaired_contacts
    lost_per_residue = np.sum(lost, axis=1)
    added_per_residue = np.sum(added, axis=1)
    top_count = max(1, int(math.ceil(raw.shape[0] * top_fraction)))

    def _share(values: np.ndarray) -> float:
        return float(np.sum(np.sort(values)[-top_count:]) / max(np.sum(values), 1))

    def _examples(values: np.ndarray) -> list[dict[str, int]]:
        order = np.lexsort((np.arange(values.size), -values))
        return [
            {"residue_index": int(index), "change_count": int(values[index])}
            for index in order[:maximum_examples]
            if values[index] > 0
        ]

    lost_share = _share(lost_per_residue)
    added_share = _share(added_per_residue)
    return (
        {
            "lost_contact_count": int(np.sum(lost) // 2),
            "added_contact_count": int(np.sum(added) // 2),
            "top_residue_count": top_count,
            "lost_contact_top_residue_share": lost_share,
            "added_contact_top_residue_share": added_share,
            "change_distribution": (
                "localized" if max(lost_share, added_share) >= localized_share_threshold else "diffuse"
            ),
            "worst_lost_contact_residues": _examples(lost_per_residue),
            "worst_added_contact_residues": _examples(added_per_residue),
        },
        lost_per_residue,
        added_per_residue,
    )


def analyze_candidate_pair(
    raw_matrix: np.ndarray,
    repaired_matrix: np.ndarray,
    *,
    candidate_id: str,
    requested_length: int,
    method: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute bounded diagnostics for one immutable raw/repaired candidate pair."""
    raw = np.asarray(raw_matrix, dtype=np.float64)
    repaired = np.asarray(repaired_matrix, dtype=np.float64)
    if raw.shape != repaired.shape or raw.shape != (requested_length, requested_length):
        raise ValueError(f"E007 contact-stability matrix shape contradiction: {candidate_id}")
    if not np.isfinite(raw).all() or not np.isfinite(repaired).all():
        raise ValueError(f"E007 contact-stability requires finite matrices: {candidate_id}")
    thresholds = _thresholds(config)
    canonical = float(config["classification_policy"]["canonical_threshold_angstrom"])
    base = {"candidate_id": candidate_id, "requested_length": requested_length, "repair_method": method}
    threshold_rows = [
        base | {"metric_scope": "threshold_sweep"} | row for row in threshold_sweep_metrics(raw, repaired, thresholds)
    ]
    ambiguity = ambiguity_exclusion_metrics(
        raw,
        repaired,
        threshold=canonical,
        half_widths=[
            float(value) for value in config["contact_thresholds"]["ambiguity_exclusion_half_widths_angstrom"]
        ],
    )
    threshold_rows.extend(base | {"metric_scope": "ambiguity_exclusion"} | row for row in ambiguity)
    n = raw.shape[0]
    indices = np.arange(n)
    separation = np.abs(indices[:, None] - indices[None, :])
    upper = _upper_mask(n)
    for band in config["sequence_separation_bands"]:
        mask = upper & _separation_mask(separation, band)
        threshold_rows.append(
            base
            | {"metric_scope": "sequence_separation", "band": str(band["name"]), "threshold": canonical}
            | contact_confusion(raw, repaired, threshold=canonical, mask=mask)
        )
    raw_values = raw[upper]
    for band in config["raw_distance_bands_angstrom"]:
        pair_mask = np.zeros_like(upper)
        pair_mask[upper] = _band_mask(raw_values, band)
        threshold_rows.append(
            base
            | {"metric_scope": "raw_distance_band", "band": str(band["name"]), "threshold": canonical}
            | contact_confusion(raw, repaired, threshold=canonical, mask=pair_mask)
        )
    continuous, topology_rows = continuous_topology_metrics(
        raw,
        repaired,
        thresholds=thresholds,
        soft_temperature=float(config["contact_thresholds"]["soft_contact_temperature_angstrom"]),
        distance_ranges=config["distance_change_ranges_angstrom"],
    )
    neighbourhood_rows, neighbourhood_summary, separation_rows = neighbourhood_metrics(
        raw,
        repaired,
        k_values=[int(value) for value in config["neighbourhood"]["k_values"]],
        retention_threshold=float(config["neighbourhood"]["existing_retention_threshold"]),
        maximum_worst_examples=int(config["neighbourhood"]["maximum_worst_residue_examples"]),
    )
    localization, lost_per_residue, added_per_residue = _localization(
        raw,
        repaired,
        threshold=canonical,
        top_fraction=float(config["neighbourhood"]["localization_top_residue_fraction"]),
        maximum_examples=int(config["neighbourhood"]["maximum_worst_residue_examples"]),
        localized_share_threshold=float(config["neighbourhood"]["localized_change_share_threshold"]),
    )
    for row in neighbourhood_rows:
        index = int(row["residue_index"])
        row.update(base)
        row["lost_contact_count_8A"] = int(lost_per_residue[index])
        row["added_contact_count_8A"] = int(added_per_residue[index])
    for row in [*topology_rows, *separation_rows]:
        row.update(base)
    for requested_k in config["neighbourhood"]["k_values"]:
        for quartile in range(1, 5):
            selected = [
                row["neighbourhood_jaccard"]
                for row in neighbourhood_rows
                if row["requested_k"] == int(requested_k) and row["position_quartile"] == quartile
            ]
            if selected:
                topology_rows.append(
                    base
                    | {
                        "summary_type": "neighbourhood_position_quartile",
                        "requested_k": int(requested_k),
                        "position_quartile": quartile,
                        "residue_count": len(selected),
                        "mean_neighbourhood_jaccard": float(np.mean(selected)),
                    }
                )
    eight = next(row for row in threshold_rows if row["metric_scope"] == "threshold_sweep" and row["threshold"] == 8.0)
    confident = next(
        row
        for row in threshold_rows
        if row["metric_scope"] == "ambiguity_exclusion" and row["ambiguity_half_width_angstrom"] == 0.5
    )
    long_range = next(
        row
        for row in threshold_rows
        if row["metric_scope"] == "sequence_separation" and row["band"] == "separation_24_or_more"
    )
    flips = eight["false_positive"] + eight["false_negative"]
    ambiguity_mask = upper & (np.abs(raw - canonical) <= 0.5)
    raw_contact = raw <= canonical
    repaired_contact = repaired <= canonical
    flips_in_band = int(np.sum(ambiguity_mask & (raw_contact != repaired_contact)))
    candidate_row = base | {
        "contact_8A": {key: eight[key] for key in contact_confusion(raw, repaired, threshold=8.0)},
        "confident_contact_8A_excluding_0_5A": {
            key: confident[key] for key in contact_confusion(raw, repaired, threshold=8.0)
        },
        "long_range_contact_8A": {key: long_range[key] for key in contact_confusion(raw, repaired, threshold=8.0)},
        "flip_count_8A": int(flips),
        "flip_count_inside_0_5A_ambiguity_band": flips_in_band,
        "flip_fraction_inside_0_5A_ambiguity_band": (float(flips_in_band / flips) if flips else 1.0),
        "continuous": continuous,
        "neighbourhood": neighbourhood_summary,
        "localization": localization,
    }
    return candidate_row, threshold_rows, neighbourhood_rows, [*topology_rows, *separation_rows]


def build_contact_stability_plan(config_path: str | Path, *, repository_root: str | Path = ".") -> dict[str, Any]:
    """Verify both source audits and every matrix artifact without publishing output."""
    root = Path(repository_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if config.get("version") != PLAN_VERSION:
        raise ValueError(f"Unsupported E007 contact-stability plan version: {config.get('version')!r}")
    if tuple(config["comparisons"]) != COMPARISON_METHODS:
        raise ValueError("E007 contact-stability comparisons contradict the Phase-2C contract")
    _thresholds(config)
    output = root / str(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 contact-stability output or in-progress directory exists: {output}")

    phase2 = config["phase2_source"]
    phase2b = config["phase2b_source"]
    phase2_report_path = _resolved_file(
        root, phase2["report_path"], expected_sha256=phase2["report_sha256"], label="Phase-2 report"
    )
    phase2_protocol_path = _resolved_file(
        root,
        phase2["protocol_path"],
        expected_sha256=phase2["protocol_sha256"],
        label="Phase-2 protocol",
    )
    raw_manifest_path = _resolved_file(
        root,
        phase2["candidate_manifest_path"],
        expected_sha256=phase2["candidate_manifest_sha256"],
        label="Phase-2 candidate manifest",
    )
    phase2b_report_path = _resolved_file(
        root, phase2b["report_path"], expected_sha256=phase2b["report_sha256"], label="Phase-2B report"
    )
    phase2b_protocol_path = _resolved_file(
        root,
        phase2b["protocol_path"],
        expected_sha256=phase2b["protocol_sha256"],
        label="Phase-2B protocol",
    )
    metrics_path = _resolved_file(
        root,
        phase2b["matrix_metrics_path"],
        expected_sha256=phase2b["matrix_metrics_sha256"],
        label="Phase-2B matrix metrics",
    )
    repair_manifest_path = _resolved_file(
        root,
        phase2b["repair_manifest_path"],
        expected_sha256=phase2b["repair_manifest_sha256"],
        label="Phase-2B repair manifest",
    )
    phase2_report = json.loads(phase2_report_path.read_text(encoding="utf-8"))
    phase2_protocol = json.loads(phase2_protocol_path.read_text(encoding="utf-8"))
    phase2b_report = json.loads(phase2b_report_path.read_text(encoding="utf-8"))
    phase2b_protocol = json.loads(phase2b_protocol_path.read_text(encoding="utf-8"))
    if phase2_protocol.get("status") != "completed" or phase2b_protocol.get("status") != "completed":
        raise ValueError("E007 contact-stability requires completed Phase-2 and Phase-2B protocols")
    if phase2_protocol.get("report_sha256") != phase2["report_sha256"]:
        raise ValueError("E007 Phase-2 protocol/report identity contradiction")
    if phase2b_protocol.get("report_sha256") != phase2b["report_sha256"]:
        raise ValueError("E007 Phase-2B protocol/report identity contradiction")
    if phase2b_report.get("decision", {}).get("classification") != phase2b["required_decision"]:
        raise ValueError("E007 Phase-2B scientific decision contradicts the Phase-2C contract")
    for payload in (phase2_protocol, phase2b_protocol, phase2_report, phase2b_report):
        if payload.get("authorizes_training") or payload.get("authorizes_joint_training"):
            raise ValueError("E007 contact-stability source has invalid authorization semantics")
    phase2_payloads = phase2_protocol.get("published_payload_hashes", {})
    phase2b_payloads = phase2b_report.get("published_payload_hashes", {})
    if phase2_payloads.get("candidate_manifest.jsonl") != phase2["candidate_manifest_sha256"]:
        raise ValueError("E007 Phase-2 candidate-manifest publication contradiction")
    if phase2b_payloads.get("matrix_metrics.jsonl") != phase2b["matrix_metrics_sha256"]:
        raise ValueError("E007 Phase-2B matrix-metrics publication contradiction")
    if phase2b_payloads.get("repair_manifest.jsonl") != phase2b["repair_manifest_sha256"]:
        raise ValueError("E007 Phase-2B repair-manifest publication contradiction")

    raw_rows = _read_jsonl(raw_manifest_path)
    repaired_rows = _read_jsonl(repair_manifest_path)
    if len(raw_rows) != int(config["required_candidate_count"]) or len(repaired_rows) != len(raw_rows):
        raise ValueError("E007 contact-stability candidate-count contradiction")
    if len(raw_rows) > int(config["bounds"]["maximum_candidate_count"]):
        raise ValueError("E007 contact-stability candidate count exceeds its bound")
    if not bool(config["bounds"]["process_candidates_individually"]):
        raise ValueError("E007 contact-stability requires individual candidate processing")
    raw_by_id = {str(row["candidate_id"]): row for row in raw_rows}
    repaired_by_id = {str(row["candidate_id"]): row for row in repaired_rows}
    if len(raw_by_id) != len(raw_rows) or len(repaired_by_id) != len(repaired_rows):
        raise ValueError("E007 contact-stability candidate IDs must be unique")
    if set(raw_by_id) != set(repaired_by_id):
        raise ValueError("E007 raw and repaired candidate membership differs")
    phase2_directory = root / str(phase2["directory"])
    phase2b_directory = root / str(phase2b["directory"])
    inventory = []
    length_counts: dict[str, int] = {}
    for candidate_id in sorted(raw_by_id):
        raw_row, repaired_row = raw_by_id[candidate_id], repaired_by_id[candidate_id]
        raw_relative = str(raw_row["candidate_artifact_path"])
        repaired_relative = str(repaired_row["repaired_artifact_path"])
        raw_sha = str(raw_row["candidate_artifact_sha256"])
        repaired_sha = str(repaired_row["repaired_artifact_sha256"])
        raw_path = _contained_file(phase2_directory, raw_relative, label="raw candidate")
        repaired_path = _contained_file(phase2b_directory, repaired_relative, label="repaired candidate")
        if phase2_payloads.get(raw_relative) != raw_sha or sha256_file(raw_path) != raw_sha:
            raise ValueError(f"E007 raw candidate hash contradiction: {candidate_id}")
        if phase2b_payloads.get(repaired_relative) != repaired_sha or sha256_file(repaired_path) != repaired_sha:
            raise ValueError(f"E007 repaired candidate hash contradiction: {candidate_id}")
        if str(repaired_row["source_candidate_sha256"]) != raw_sha:
            raise ValueError(f"E007 repaired/raw lineage contradiction: {candidate_id}")
        raw_loaded = load_candidate_npz(raw_path)
        length = int(raw_row["requested_length"])
        if raw_loaded["candidate_id"] != candidate_id or raw_loaded["actual_valid_length"] != length:
            raise ValueError(f"E007 raw candidate provenance contradiction: {candidate_id}")
        with np.load(repaired_path, allow_pickle=False) as data:
            if str(data["candidate_id"]) != candidate_id or str(data["source_candidate_sha256"]) != raw_sha:
                raise ValueError(f"E007 repaired candidate provenance contradiction: {candidate_id}")
            for key in ("rank3_distance_matrix_angstrom", "constrained_distance_matrix_angstrom"):
                if data[key].shape != (length, length):
                    raise ValueError(f"E007 repaired candidate shape contradiction: {candidate_id}")
        length_counts[str(length)] = length_counts.get(str(length), 0) + 1
        inventory.append(
            {
                "candidate_id": candidate_id,
                "requested_length": length,
                "raw_path": raw_relative,
                "raw_sha256": raw_sha,
                "repaired_path": repaired_relative,
                "repaired_sha256": repaired_sha,
            }
        )
    if sorted(map(int, length_counts)) != sorted(map(int, config["required_lengths"])):
        raise ValueError("E007 contact-stability length strata contradict the configuration")
    metrics_rows = _read_jsonl(metrics_path)
    expected_metric_keys = {
        (candidate_id, version) for candidate_id in raw_by_id for version in ("raw_generated", *COMPARISON_METHODS)
    }
    observed_metric_keys = {(str(row["candidate_id"]), str(row["version"])) for row in metrics_rows}
    if expected_metric_keys != observed_metric_keys:
        raise ValueError("E007 Phase-2B matrix-metric membership contradiction")
    protected = {
        name: {
            "path": str(record["path"]),
            "sha256": sha256_file(_resolved_file(root, record["path"], expected_sha256=record["sha256"], label=name)),
        }
        for name, record in config["protected_inputs"].items()
    }
    return {
        "version": PLAN_VERSION,
        "status": "planned",
        "mode": "plan_only",
        "configuration_sha256": sha256_file(config_file),
        "output_dir": str(config["output_dir"]),
        "output_directory_absent": True,
        "candidate_count": len(inventory),
        "candidate_counts_by_length": length_counts,
        "candidate_inventory_sha256": _json_hash(inventory),
        "candidates": inventory,
        "comparisons": list(COMPARISON_METHODS),
        "threshold_sweep_angstrom": _thresholds(config),
        "classification_policy": config["classification_policy"],
        "protected_inputs": protected,
        "phase2_report_sha256": phase2["report_sha256"],
        "phase2_protocol_sha256": phase2["protocol_sha256"],
        "phase2b_report_sha256": phase2b["report_sha256"],
        "phase2b_protocol_sha256": phase2b["protocol_sha256"],
        "matrices_modified": False,
        "candidates_averaged": False,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "backward_performed": False,
        "authorizes_training": False,
        "authorizes_joint_training": False,
        "authorizes_sequence_conditioning": False,
        "real_diagnostic_executed": False,
    }


def _candidate_scalar_metrics(row: dict[str, Any]) -> dict[str, float]:
    return {
        "contact_f1_8A": float(row["contact_8A"]["f1"]),
        "contact_precision_8A": float(row["contact_8A"]["precision"]),
        "contact_recall_8A": float(row["contact_8A"]["recall"]),
        "confident_contact_f1_8A": float(row["confident_contact_8A_excluding_0_5A"]["f1"]),
        "long_range_contact_f1_8A": float(row["long_range_contact_8A"]["f1"]),
        "distance_spearman": float(row["continuous"]["distance_spearman_correlation"]),
        "distance_pearson": float(row["continuous"]["distance_pearson_correlation"]),
        "relative_frobenius_distortion": float(row["continuous"]["relative_frobenius_distortion"]),
        "pairwise_rmse_angstrom": float(row["continuous"]["pairwise_distance_rmse_angstrom"]),
        "neighbourhood_retention_k8": float(row["neighbourhood"]["8"]["mean"]),
        "flip_fraction_inside_0_5A": float(row["flip_fraction_inside_0_5A_ambiguity_band"]),
        "lost_contact_top_residue_share": float(row["localization"]["lost_contact_top_residue_share"]),
        "contact_f1_threshold_auc": float(row["continuous"]["contact_f1_threshold_auc_5_to_12"]),
    }


def _aggregate_candidate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    lengths: list[int | None] = [None, *sorted({int(row["requested_length"]) for row in rows})]
    for method in COMPARISON_METHODS:
        for length in lengths:
            selected = [
                row
                for row in rows
                if row["repair_method"] == method and (length is None or int(row["requested_length"]) == length)
            ]
            if not selected:
                continue
            scalar_rows = [_candidate_scalar_metrics(row) for row in selected]
            for metric in scalar_rows[0]:
                values = np.asarray([row[metric] for row in scalar_rows], dtype=np.float64)
                output.append(
                    {
                        "repair_method": method,
                        "requested_length": length,
                        "metric": metric,
                        "count": int(values.size),
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "standard_deviation": float(np.std(values)),
                        "quantile_05": float(np.quantile(values, 0.05)),
                        "quantile_25": float(np.quantile(values, 0.25)),
                        "quantile_75": float(np.quantile(values, 0.75)),
                        "quantile_95": float(np.quantile(values, 0.95)),
                        "minimum": float(np.min(values)),
                        "maximum": float(np.max(values)),
                    }
                )
    return output


def _aggregate_threshold_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    lengths: list[int | None] = [None, *sorted({int(row["requested_length"]) for row in rows})]
    for method in COMPARISON_METHODS:
        for length in lengths:
            selected = [
                row
                for row in rows
                if row["repair_method"] == method and (length is None or int(row["requested_length"]) == length)
            ]
            groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
            for row in selected:
                key = (
                    row["metric_scope"],
                    row.get("threshold"),
                    row.get("ambiguity_half_width_angstrom"),
                    row.get("band"),
                )
                groups.setdefault(key, []).append(row)
            for key, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
                tp = sum(int(row["true_positive"]) for row in group)
                fp = sum(int(row["false_positive"]) for row in group)
                fn = sum(int(row["false_negative"]) for row in group)
                tn = sum(int(row["true_negative"]) for row in group)
                no_positive_union = tp + fp + fn == 0
                precision = 1.0 if no_positive_union else tp / max(tp + fp, 1)
                recall = 1.0 if no_positive_union else tp / max(tp + fn, 1)
                output.append(
                    {
                        "repair_method": method,
                        "requested_length": length,
                        "metric_scope": key[0],
                        "threshold": key[1],
                        "ambiguity_half_width_angstrom": key[2],
                        "band": key[3],
                        "candidate_count": len(group),
                        "pair_count": tp + fp + fn + tn,
                        "true_positive": tp,
                        "false_positive": fp,
                        "false_negative": fn,
                        "true_negative": tn,
                        "contact_prevalence": (tp + fn) / max(tp + fp + fn + tn, 1),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1": float(2.0 * precision * recall / max(precision + recall, 1e-12)),
                        "jaccard": float(1.0 if no_positive_union else tp / (tp + fp + fn)),
                    }
                )
    return output


def _method_diagnostic_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictors = (
        "raw_negative_eigenmass_fraction",
        "raw_rank3_reconstruction_rmse_angstrom",
        "repair_pairwise_rmse_angstrom",
        "final_objective_total",
        "final_objective_robust_distance_fit",
        "final_objective_adjacent_bond",
        "final_objective_steric_clash",
        "iteration_count",
        "final_gradient_norm",
        "requested_length",
    )
    outcomes = {
        "contact_f1_8A": lambda row: row["contact_8A"]["f1"],
        "confident_contact_f1_8A": lambda row: row["confident_contact_8A_excluding_0_5A"]["f1"],
        "long_range_contact_f1_8A": lambda row: row["long_range_contact_8A"]["f1"],
        "neighbourhood_retention_k8": lambda row: row["neighbourhood"]["8"]["mean"],
    }
    output = []
    for method in COMPARISON_METHODS:
        selected = [row for row in rows if row["repair_method"] == method]
        for predictor in predictors:
            for outcome_name, outcome in outcomes.items():
                pairs = [
                    (float(row[predictor]), float(outcome(row)))
                    for row in selected
                    if row.get(predictor) is not None and math.isfinite(float(row[predictor]))
                ]
                if len(pairs) < 3:
                    continue
                left = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
                right = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
                output.append(
                    {
                        "repair_method": method,
                        "predictor": predictor,
                        "outcome": outcome_name,
                        "count": len(pairs),
                        "pearson_correlation": _correlation(left, right),
                        "spearman_correlation": _correlation(left, right, ranked=True),
                    }
                )
    return output


def classify_contact_stability(
    rows: list[dict[str, Any]], config: dict[str, Any], correlations: list[dict[str, Any]]
) -> dict[str, Any]:
    """Apply the predeclared Phase-2C table without changing Phase-2B decisions."""
    policy = config["classification_policy"]
    minimum_fraction = float(policy["minimum_candidate_fraction"])
    results = {}
    for method in COMPARISON_METHODS:
        selected = [row for row in rows if row["repair_method"] == method]
        instability = policy["hard_threshold_instability_only"]
        recoverable = policy["recoverable_contact_scale_distortion"]

        def _threshold_only(row: dict[str, Any], criteria: dict[str, Any] = instability) -> bool:
            metric = _candidate_scalar_metrics(row)
            return (
                metric["flip_fraction_inside_0_5A"] >= float(criteria["minimum_flip_fraction_inside_ambiguity_band"])
                and metric["confident_contact_f1_8A"] >= float(criteria["minimum_confident_contact_f1"])
                and metric["long_range_contact_f1_8A"] >= float(criteria["minimum_long_range_contact_f1"])
                and metric["distance_spearman"] >= float(criteria["minimum_distance_spearman"])
                and metric["relative_frobenius_distortion"] <= float(criteria["maximum_relative_frobenius_distortion"])
                and metric["neighbourhood_retention_k8"] >= float(criteria["minimum_neighbourhood_retention"])
            )

        def _recoverable(row: dict[str, Any], criteria: dict[str, Any] = recoverable) -> bool:
            metric = _candidate_scalar_metrics(row)
            return (
                metric["confident_contact_f1_8A"] >= float(criteria["minimum_confident_contact_f1"])
                and metric["long_range_contact_f1_8A"] >= float(criteria["minimum_long_range_contact_f1"])
                and metric["distance_spearman"] >= float(criteria["minimum_distance_spearman"])
                and metric["relative_frobenius_distortion"] <= float(criteria["maximum_relative_frobenius_distortion"])
                and metric["neighbourhood_retention_k8"] >= float(criteria["minimum_neighbourhood_retention"])
                and (
                    row["localization"]["lost_contact_count"] == 0
                    or metric["lost_contact_top_residue_share"] >= float(criteria["minimum_top_residue_change_share"])
                )
            )

        threshold_flags = [_threshold_only(row) for row in selected]
        recoverable_flags = [_recoverable(row) for row in selected]
        substantial_policy = policy["substantial_length_dependent_topology_loss"]
        long_rows = [
            row for row in selected if int(row["requested_length"]) >= int(substantial_policy["minimum_length"])
        ]
        substantial_flags = [not _recoverable(row) for row in long_rows]
        strong_pre_signal = max(
            (
                abs(float(row["spearman_correlation"]))
                for row in correlations
                if row["repair_method"] == method
                and row["predictor"] in {"raw_negative_eigenmass_fraction", "raw_rank3_reconstruction_rmse_angstrom"}
            ),
            default=0.0,
        )
        mixed_policy = policy["mixed_candidate_quality_requires_filtering"]
        good_count = int(np.sum(recoverable_flags))
        bad_count = len(recoverable_flags) - good_count
        if threshold_flags and float(np.mean(threshold_flags)) >= minimum_fraction:
            classification = "hard_threshold_instability_only"
        elif long_rows and float(np.mean(substantial_flags)) >= float(substantial_policy["minimum_failing_fraction"]):
            classification = "substantial_length_dependent_topology_loss"
        elif recoverable_flags and float(np.mean(recoverable_flags)) >= minimum_fraction:
            classification = "recoverable_contact_scale_distortion"
        elif (
            strong_pre_signal >= float(mixed_policy["minimum_absolute_pre_repair_signal_correlation"])
            and good_count >= int(mixed_policy["minimum_good_and_bad_candidate_count"])
            and bad_count >= int(mixed_policy["minimum_good_and_bad_candidate_count"])
        ):
            classification = "mixed_candidate_quality_requires_filtering"
        else:
            classification = "inconclusive_requires_scientific_review"
        results[method] = {
            "classification": classification,
            "candidate_count": len(selected),
            "threshold_instability_fraction": float(np.mean(threshold_flags)),
            "recoverable_fraction": float(np.mean(recoverable_flags)),
            "long_length_substantial_loss_fraction": (float(np.mean(substantial_flags)) if substantial_flags else None),
            "maximum_absolute_pre_repair_signal_correlation": strong_pre_signal,
        }
    return {
        "version": policy["version"],
        "primary_method": "constrained_coordinate_repair",
        "classification": results["constrained_coordinate_repair"]["classification"],
        "method_results": results,
        "phase2b_decision_changed": False,
        "authorizes_sequence_conditioning": False,
    }


def _peak_rss_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_repaired_artifact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "rank3_psd_projection": np.asarray(data["rank3_distance_matrix_angstrom"], dtype=np.float32),
            "constrained_coordinate_repair": np.asarray(data["constrained_distance_matrix_angstrom"], dtype=np.float32),
        }


def run_contact_stability_audit(
    config_path: str | Path,
    *,
    plan: dict[str, Any],
    repository_root: str | Path = ".",
) -> Path:
    """Read immutable matrices one candidate at a time and publish bounded diagnostics."""
    root = Path(repository_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if plan.get("configuration_sha256") != sha256_file(config_file):
        raise ValueError("E007 contact-stability plan/configuration identity contradiction")
    output = root / str(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 contact-stability output or in-progress directory exists: {output}")
    phase2_directory = root / str(config["phase2_source"]["directory"])
    phase2b_directory = root / str(config["phase2b_source"]["directory"])
    metrics_rows = _read_jsonl(root / str(config["phase2b_source"]["matrix_metrics_path"]))
    metric_index = {(str(row["candidate_id"]), str(row["version"])): row for row in metrics_rows}
    source_paths = {
        "phase2_report": root / str(config["phase2_source"]["report_path"]),
        "phase2_protocol": root / str(config["phase2_source"]["protocol_path"]),
        "phase2_manifest": root / str(config["phase2_source"]["candidate_manifest_path"]),
        "phase2b_report": root / str(config["phase2b_source"]["report_path"]),
        "phase2b_protocol": root / str(config["phase2b_source"]["protocol_path"]),
        "phase2b_metrics": root / str(config["phase2b_source"]["matrix_metrics_path"]),
        "phase2b_manifest": root / str(config["phase2b_source"]["repair_manifest_path"]),
    }
    source_paths.update(
        {f"protected::{name}": root / record["path"] for name, record in plan["protected_inputs"].items()}
    )
    for candidate in plan["candidates"]:
        source_paths[f"raw::{candidate['candidate_id']}"] = phase2_directory / candidate["raw_path"]
        source_paths[f"repaired::{candidate['candidate_id']}"] = phase2b_directory / candidate["repaired_path"]
    hashes_before = {name: sha256_file(path) for name, path in source_paths.items()}
    started = _utc_now()
    staging.mkdir(parents=True)
    _atomic_json(
        staging / "heartbeat.json",
        {"status": "running", "stage": "candidate_analysis", "started_utc": started, "processed": 0},
    )
    candidate_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    residue_rows: list[dict[str, Any]] = []
    topology_rows: list[dict[str, Any]] = []
    try:
        for index, candidate in enumerate(plan["candidates"]):
            candidate_id = str(candidate["candidate_id"])
            raw_path = phase2_directory / candidate["raw_path"]
            repaired_path = phase2b_directory / candidate["repaired_path"]
            if sha256_file(raw_path) != candidate["raw_sha256"]:
                raise ValueError(f"E007 raw candidate changed before analysis: {candidate_id}")
            if sha256_file(repaired_path) != candidate["repaired_sha256"]:
                raise ValueError(f"E007 repaired candidate changed before analysis: {candidate_id}")
            raw = load_candidate_npz(raw_path)["physical_matrix_angstrom"]
            repaired_matrices = _load_repaired_artifact(repaired_path)
            raw_metric = metric_index[(candidate_id, "raw_generated")]
            for method in COMPARISON_METHODS:
                row, threshold, residue, topology = analyze_candidate_pair(
                    raw,
                    repaired_matrices[method],
                    candidate_id=candidate_id,
                    requested_length=int(candidate["requested_length"]),
                    method=method,
                    config=config,
                )
                existing = metric_index[(candidate_id, method)]
                diagnostics = existing["method_diagnostics"]
                final = diagnostics.get("final_component_losses", {})
                row.update(
                    {
                        "source_raw_path": candidate["raw_path"],
                        "source_raw_sha256": candidate["raw_sha256"],
                        "source_repaired_path": candidate["repaired_path"],
                        "source_repaired_sha256": candidate["repaired_sha256"],
                        "raw_negative_eigenmass_fraction": raw_metric["quality"]["negative_eigenmass_fraction"],
                        "raw_rank3_reconstruction_rmse_angstrom": raw_metric["quality"][
                            "rank3_reconstruction_rmse_angstrom"
                        ],
                        "repair_pairwise_rmse_angstrom": existing["topology"]["pairwise_distance_rmse_angstrom"],
                        "final_objective_total": final.get("total"),
                        "final_objective_robust_distance_fit": final.get("robust_distance_fit"),
                        "final_objective_adjacent_bond": final.get("adjacent_bond"),
                        "final_objective_steric_clash": final.get("steric_clash"),
                        "termination_reason": diagnostics.get("termination_reason"),
                        "iteration_count": diagnostics.get("iterations"),
                        "final_gradient_norm": diagnostics.get("final_gradient_norm"),
                    }
                )
                candidate_rows.append(row)
                threshold_rows.extend(threshold)
                residue_rows.extend(residue)
                if len(topology) > int(config["bounds"]["maximum_topology_summary_rows_per_candidate"]):
                    raise ValueError(f"E007 topology summary bound exceeded: {candidate_id}/{method}")
                topology_rows.extend(topology)
            if _peak_rss_mib() > float(config["bounds"]["maximum_rss_mib"]):
                raise MemoryError(
                    f"E007 contact-stability RSS limit exceeded at {candidate_id}: {_peak_rss_mib():.1f} MiB"
                )
            _atomic_json(
                staging / "heartbeat.json",
                {
                    "status": "running",
                    "stage": "candidate_analysis",
                    "started_utc": started,
                    "processed": index + 1,
                    "total": len(plan["candidates"]),
                    "last_candidate_id": candidate_id,
                    "peak_rss_mib": _peak_rss_mib(),
                },
            )
        correlations = _method_diagnostic_correlations(candidate_rows)
        classification = classify_contact_stability(candidate_rows, config, correlations)
        aggregates = _aggregate_candidate_metrics(candidate_rows)
        threshold_aggregates = _aggregate_threshold_metrics(threshold_rows)
        compression = str(config["bounds"]["parquet_compression"])
        _atomic_parquet(staging / "per_candidate_metrics.parquet", candidate_rows, compression)
        _atomic_parquet(staging / "per_threshold_metrics.parquet", threshold_rows, compression)
        _atomic_parquet(staging / "per_residue_neighbourhood_metrics.parquet", residue_rows, compression)
        _atomic_parquet(staging / "topology_change_summaries.parquet", topology_rows, compression)
        hashes_after = {name: sha256_file(path) for name, path in source_paths.items()}
        if hashes_before != hashes_after:
            raise RuntimeError("E007 contact-stability protected inputs changed during analysis")
        payload_hashes = {path.name: sha256_file(path) for path in sorted(staging.glob("*.parquet"))}
        report = {
            "version": AUDIT_VERSION,
            "status": "completed_requires_scientific_review",
            "classification": classification,
            "phase2b_decision_retained": "repair_is_length_limited",
            "candidate_count": len(plan["candidates"]),
            "candidate_counts_by_length": plan["candidate_counts_by_length"],
            "comparison_methods": list(COMPARISON_METHODS),
            "threshold_sweep_angstrom": plan["threshold_sweep_angstrom"],
            "classification_policy": config["classification_policy"],
            "aggregate_metrics": aggregates,
            "aggregate_threshold_metrics": threshold_aggregates,
            "method_diagnostic_correlations": correlations,
            "table_row_counts": {
                "per_candidate_metrics": len(candidate_rows),
                "per_threshold_metrics": len(threshold_rows),
                "per_residue_neighbourhood_metrics": len(residue_rows),
                "topology_change_summaries": len(topology_rows),
            },
            "published_payload_hashes": payload_hashes,
            "protected_input_inventory_sha256_before": _json_hash(hashes_before),
            "protected_input_inventory_sha256_after": _json_hash(hashes_after),
            "protected_inputs_unchanged": True,
            "matrices_modified": False,
            "candidates_averaged": False,
            "optimizer_created": False,
            "optimizer_updates": 0,
            "backward_performed": False,
            "training_performed": False,
            "authorizes_training": False,
            "authorizes_joint_training": False,
            "authorizes_sequence_conditioning": False,
            "scientific_boundary": (
                "This diagnostic cannot authorize sequence conditioning and does not alter Phase-2B thresholds."
            ),
            "peak_rss_mib": _peak_rss_mib(),
        }
        _atomic_json(staging / "report.json", report)
        completed = _utc_now()
        protocol = {
            "version": AUDIT_VERSION,
            "status": "completed",
            "started_utc": started,
            "completed_utc": completed,
            "git_commit": _git_commit(root),
            "configuration_sha256": sha256_file(config_file),
            "phase2_report_sha256": plan["phase2_report_sha256"],
            "phase2_protocol_sha256": plan["phase2_protocol_sha256"],
            "phase2b_report_sha256": plan["phase2b_report_sha256"],
            "phase2b_protocol_sha256": plan["phase2b_protocol_sha256"],
            "report_sha256": sha256_file(staging / "report.json"),
            "protected_inputs_unchanged": True,
            "matrices_modified": False,
            "candidates_averaged": False,
            "optimizer_created": False,
            "optimizer_updates": 0,
            "backward_performed": False,
            "training_performed": False,
            "authorizes_training": False,
            "authorizes_joint_training": False,
            "authorizes_sequence_conditioning": False,
        }
        _atomic_json(staging / "protocol.json", protocol)
        _atomic_json(
            staging / "heartbeat.json",
            {
                "status": "completed",
                "completed_utc": completed,
                "processed": len(plan["candidates"]),
                "report_sha256": sha256_file(staging / "report.json"),
                "protocol_sha256": sha256_file(staging / "protocol.json"),
            },
        )
        staging.replace(output)
        return output
    except BaseException as exc:
        if staging.exists():
            _atomic_json(
                staging / "heartbeat.json",
                {
                    "status": "failed",
                    "failed_utc": _utc_now(),
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "completed_output_published": False,
                },
            )
        raise
