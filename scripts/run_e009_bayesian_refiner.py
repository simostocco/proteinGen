#!/usr/bin/env python3
"""Runnable E009 prior fit, CUDA smoke, and isolated length-64 overfit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import resource
import shutil
import tempfile
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml

from protein_distance_diffusion.data.e007_coordinate_dataset import E007CoordinateDataset
from protein_distance_diffusion.models.e009_bayesian_refiner import (
    BayesianSE3Refiner,
    GeometryPrior,
    bounded_angle_transform,
    fit_gaussian_mixture_stream,
    fit_von_mises_mixture_stream,
    normalized_refiner_losses,
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _exact_keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        missing = set(keys) - set(value or {})
        extra = set(value or {}) - set(keys)
        raise ValueError(f"{name} schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}")


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    v3_keys = (
        "v2_config",
        "v2_config_sha256",
        "smoke_result",
        "smoke_result_sha256",
        "smoke_review",
        "smoke_review_sha256",
        "staging_dir",
        "overfit_checkpoint",
        "overfit_result",
        "auto_stop_after_adjudication",
    )
    is_v3 = cfg.get("version") == "e009_bayesian_geometry_refiner_v3"
    is_v4 = cfg.get("version") == "e009_bayesian_geometry_refiner_v4"
    is_v3plus = is_v3 or is_v4
    v4_keys = ("length_64_overfit_authorized",)
    base_keys = (
        "version",
        "seed",
        "device",
        "e008_config",
        "e008_config_sha256",
        "published_manifest",
        "published_split_record",
        "published_manifest_sha256",
        "published_split_sha256",
        "output_dir",
        "prior_artifact",
        "review_artifact",
        "review_sha256",
        "prior_sha256",
        "runner_sha256",
        "model_sha256",
        "prototype_only",
        "cuda_smoke_dir",
        "overfit64_dir",
        "model",
        "prior",
        "optimizer",
        "stages",
        "loss_weights",
        "corruption",
        "authorization",
    )
    _exact_keys(
        cfg,
        base_keys + (v3_keys if is_v3plus else ()) + (v4_keys if is_v4 else ()),
        "root",
    )
    if cfg["version"] not in (
        "e009_bayesian_geometry_refiner_v2",
        "e009_bayesian_geometry_refiner_v3",
        "e009_bayesian_geometry_refiner_v4",
    ):
        raise ValueError("E009 configuration version mismatch")
    _exact_keys(cfg["model"], ("width", "layers", "max_length", "sigma_min_angstrom", "sigma_max_angstrom"), "model")
    _exact_keys(
        cfg["prior"],
        ("bond_components", "angle_components", "torsion_components", "max_em_iterations", "tolerance", "chunk_size"),
        "prior",
    )
    _exact_keys(cfg["optimizer"], ("learning_rate", "weight_decay", "gradient_clip_norm"), "optimizer")
    _exact_keys(cfg["stages"], ("cuda_smoke", "overfit_length_64"), "stages")
    _exact_keys(cfg["stages"]["cuda_smoke"], ("length", "optimizer_steps"), "stages.cuda_smoke")
    _exact_keys(
        cfg["stages"]["overfit_length_64"],
        ("length", "max_updates", "evaluations", "rmse_gate_angstrom", "max_abs_error_slope_angstrom_per_residue"),
        "stages.overfit_length_64",
    )
    expected_losses = (
        "coordinate_nll",
        "geometry_nll",
        "long_range_pair_nll",
        "contact_nll",
        "radius_of_gyration_nll",
        "chirality_nll",
        "posterior_kl",
    )
    _exact_keys(cfg["loss_weights"], expected_losses, "loss_weights")
    _exact_keys(cfg["corruption"], ("calibration_summary", "calibration_sha256", "fixed_cache"), "corruption")
    _exact_keys(cfg["authorization"], ("run_fitting", "run_cuda", "run_overfits", "run_pilot"), "authorization")
    if cfg["model"]["layers"] not in (6, 7, 8) or cfg["model"]["max_length"] != 500 or cfg["model"]["width"] < 32:
        raise ValueError("model requires width>=32, 6-8 blocks, and max_length=500")
    if cfg["prior"] != {
        "bond_components": 3,
        "angle_components": 4,
        "torsion_components": 6,
        "max_em_iterations": cfg["prior"]["max_em_iterations"],
        "tolerance": cfg["prior"]["tolerance"],
        "chunk_size": cfg["prior"]["chunk_size"],
    }:
        raise ValueError("prior component counts must be 3/4/6")
    if cfg["stages"]["cuda_smoke"] != {"length": 500, "optimizer_steps": 1}:
        raise ValueError("CUDA smoke schema requires one length-500 optimizer step")
    over = cfg["stages"]["overfit_length_64"]
    if over["length"] != 64 or over["max_updates"] != 1000 or over["evaluations"] != [0, 10, 50, 100, 250, 500, 1000]:
        raise ValueError("length-64 stage must use the fixed requested evaluation schedule")
    expected_authorization = (
        {"run_fitting": False, "run_cuda": False, "run_overfits": False, "run_pilot": False}
        if is_v3plus
        else {"run_fitting": False, "run_cuda": True, "run_overfits": False, "run_pilot": False}
    )
    if cfg["authorization"] != expected_authorization or cfg["prototype_only"] is not True:
        raise ValueError("E009 stage authorization does not match the reviewed config version")
    if is_v3plus:
        for key in ("v2_config", "smoke_result", "smoke_review"):
            if not Path(cfg[key]).is_file():
                raise ValueError(f"pinned v3 evidence missing: {key}={cfg[key]}")
        for path_key, hash_key in (
            ("v2_config", "v2_config_sha256"),
            ("smoke_result", "smoke_result_sha256"),
            ("smoke_review", "smoke_review_sha256"),
        ):
            if sha256(cfg[path_key]) != cfg[hash_key]:
                raise ValueError(f"pinned v3 evidence hash mismatch: {path_key}")
        smoke = json.loads(Path(cfg["smoke_result"]).read_text())
        smoke_review = json.loads(Path(cfg["smoke_review"]).read_text())
        if (
            smoke.get("status") != "passed"
            or smoke.get("authorizes_downstream") is not False
            or smoke_review.get("authorizes_downstream") is not False
        ):
            raise ValueError("v3 smoke evidence failed or carries downstream authorization")
        if cfg["auto_stop_after_adjudication"] is not True:
            raise ValueError("v3 must stop automatically after overfit adjudication")
        if cfg["stages"]["cuda_smoke"] != {"length": 500, "optimizer_steps": 1}:
            raise ValueError("v3 preserves the pinned v2 smoke stage specification")
        if (
            Path(cfg["overfit_checkpoint"]) != Path(cfg["overfit64_dir"]) / "resume.pt"
            or Path(cfg["overfit_result"]) != Path(cfg["overfit64_dir"]) / "result.json"
        ):
            raise ValueError("v3 overfit checkpoint/result paths must reside under the fresh stage directory")
    if is_v4 and cfg["length_64_overfit_authorized"] is not True:
        raise ValueError("v4 requires explicit human authorization for only the length-64 overfit")
    if (
        cfg["model"]["sigma_min_angstrom"] <= 0
        or cfg["model"]["sigma_max_angstrom"] <= cfg["model"]["sigma_min_angstrom"]
    ):
        raise ValueError("invalid configured posterior-scale bounds")
    if cfg["prior"]["max_em_iterations"] < 2 or cfg["prior"]["chunk_size"] < 1 or cfg["prior"]["tolerance"] <= 0:
        raise ValueError("invalid mixture-fit iteration, chunk, or tolerance setting")
    for key in ("published_manifest", "published_split_record", "e008_config", "corruption.calibration_summary"):
        value = cfg
        for part in key.split("."):
            value = value[part]
        if not Path(value).is_file():
            raise ValueError(f"configured source file missing: {key}={value}")
    if sha256(cfg["published_manifest"]) != cfg["published_manifest_sha256"]:
        raise ValueError("pinned E008 manifest hash mismatch")
    if sha256(cfg["published_split_record"]) != cfg["published_split_sha256"]:
        raise ValueError("pinned E008 split hash mismatch")
    if sha256(cfg["e008_config"]) != cfg["e008_config_sha256"]:
        raise ValueError("pinned E008 configuration hash mismatch")
    if sha256(cfg["corruption"]["calibration_summary"]) != cfg["corruption"]["calibration_sha256"]:
        raise ValueError("frozen-generator calibration hash mismatch")
    # Verify training/development groups without parsing prospective identity values.
    cfg["_split_groups"] = read_named_split_groups(cfg["published_split_record"], ("training", "development"))
    cfg["_config_sha256"] = sha256(path)
    cfg["_config_path"] = str(Path(path))
    return cfg


def read_named_split_groups(path, names):
    wanted_names = set(names)
    result = {}
    with Path(path).open("rb") as stream:
        _expect_json_byte(stream, b"{")
        while True:
            _skip_json_space(stream)
            ch = stream.read(1)
            if ch == b"}":
                break
            if ch == b",":
                _skip_json_space(stream)
            else:
                stream.seek(-1, 1)
            key = json.loads(_read_json_string(stream))
            _skip_json_space(stream)
            _expect_json_byte(stream, b":")
            _skip_json_space(stream)
            if key != "groups":
                _skip_json_value(stream)
                continue
            _expect_json_byte(stream, b"{")
            while True:
                _skip_json_space(stream)
                ch = stream.read(1)
                if ch == b"}":
                    break
                if ch == b",":
                    _skip_json_space(stream)
                else:
                    stream.seek(-1, 1)
                group_name = json.loads(_read_json_string(stream))
                _skip_json_space(stream)
                _expect_json_byte(stream, b":")
                _skip_json_space(stream)
                if group_name in wanted_names:
                    payload = _read_json_value(stream)
                    obj = json.loads(payload)
                    if not isinstance(obj, dict) or "sample_ids" not in obj:
                        raise ValueError(f"invalid E008 {group_name} split group")
                    result[group_name] = obj
                else:
                    # The prospective group is structurally skipped; its identity strings
                    # are never decoded or retained by the fitting process.
                    _skip_json_value(stream)
            break
    missing = wanted_names - result.keys()
    if missing:
        raise ValueError(f"published E008 split lacks groups: {sorted(missing)}")
    train, dev = map(lambda n: set(result[n]["sample_ids"]), ("training", "development"))
    if not train or not dev or train & dev:
        raise ValueError("E008 training/development identity sets are empty or overlap")
    return result


def _skip_json_space(stream):
    while True:
        ch = stream.read(1)
        if not ch or ch not in b" \t\r\n":
            if ch:
                stream.seek(-1, 1)
            return


def _expect_json_byte(stream, expected):
    if stream.read(1) != expected:
        raise ValueError(f"invalid published split JSON; expected {expected!r}")


def _read_json_string(stream):
    first = stream.read(1)
    if first != b'"':
        raise ValueError("invalid JSON object key")
    value = bytearray(first)
    escaped = False
    while True:
        ch = stream.read(1)
        if not ch:
            raise ValueError("unterminated JSON string")
        value.extend(ch)
        if escaped:
            escaped = False
        elif ch == b"\\":
            escaped = True
        elif ch == b'"':
            return bytes(value)


def _read_json_value(stream):
    first = stream.read(1)
    if not first:
        raise ValueError("unexpected end of split JSON")
    value = bytearray(first)
    if first not in (b"{", b"["):
        if first == b'"':
            escaped = False
            while True:
                ch = stream.read(1)
                if not ch:
                    raise ValueError("unterminated JSON value")
                value.extend(ch)
                if escaped:
                    escaped = False
                elif ch == b"\\":
                    escaped = True
                elif ch == b'"':
                    return bytes(value)
        while True:
            ch = stream.read(1)
            if not ch or ch in b",}] \t\r\n":
                if ch:
                    stream.seek(-1, 1)
                return bytes(value)
            value.extend(ch)
    depth = 1
    in_string = False
    escaped = False
    while depth:
        ch = stream.read(1)
        if not ch:
            raise ValueError("unterminated JSON compound value")
        value.extend(ch)
        if in_string:
            if escaped:
                escaped = False
            elif ch == b"\\":
                escaped = True
            elif ch == b'"':
                in_string = False
        elif ch == b'"':
            in_string = True
        elif ch in (b"{", b"["):
            depth += 1
        elif ch in (b"}", b"]"):
            depth -= 1
    return bytes(value)


def _skip_json_value(stream):
    first = stream.read(1)
    if not first:
        raise ValueError("unexpected end of split JSON")
    if first in (b"{", b"["):
        depth = 1
        in_string = False
        escaped = False
        while depth:
            ch = stream.read(1)
            if not ch:
                raise ValueError("unterminated skipped JSON value")
            if in_string:
                if escaped:
                    escaped = False
                elif ch == b"\\":
                    escaped = True
                elif ch == b'"':
                    in_string = False
            elif ch == b'"':
                in_string = True
            elif ch in (b"{", b"["):
                depth += 1
            elif ch in (b"}", b"]"):
                depth -= 1
        return
    if first == b'"':
        escaped = False
        while True:
            ch = stream.read(1)
            if not ch:
                raise ValueError("unterminated skipped JSON string")
            if escaped:
                escaped = False
            elif ch == b"\\":
                escaped = True
            elif ch == b'"':
                return
    while True:
        ch = stream.read(1)
        if not ch or ch in b",}] \t\r\n":
            if ch:
                stream.seek(-1, 1)
            return


def _source_inputs(cfg):
    paths = [
        cfg["published_manifest"],
        cfg["published_split_record"],
        cfg["e008_config"],
        cfg["corruption"]["calibration_summary"],
        cfg.get("_config_path", "configs/e009_bayesian_refiner.yaml"),
        Path(__file__),
        Path(__file__).parents[1] / "src/protein_distance_diffusion/models/e009_bayesian_refiner.py",
    ]
    e8 = yaml.safe_load(Path(cfg["e008_config"]).read_text())
    paths.extend([e8["frozen_prior"]["checkpoint"], e8["frozen_prior"]["generator_config"]])
    return {str(p): sha256(p) for p in paths}


def _prior_canonical_hash(report):
    payload = dict(report)
    payload.pop("artifact_sha256", None)
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(data.encode()).hexdigest()


def _auth_and_dataset(cfg, split):
    from scripts.run_e008_geometry_native_decoder import _authorization

    e8 = yaml.safe_load(Path(cfg["e008_config"]).read_text())
    return E007CoordinateDataset(_authorization(e8), split=split)


def _selected_rows(cfg, group_name, length_filter=None):
    """Stream selected E008 rows; prospective identity values stay inside Arrow filtering."""
    import pyarrow.dataset as ds

    wanted = set(cfg["_split_groups"][group_name]["sample_ids"])
    from protein_distance_diffusion.data.e007_coordinate_dataset import validate_coordinate_row

    seen = set()
    for split in ("train", "validation"):
        coordinate_ds = _auth_and_dataset(cfg, split)
        paths = sorted({locator.path for locator in coordinate_ds._locators})
        for source_path in paths:
            dataset = ds.dataset(source_path, format="parquet")
            wanted_filter = ds.field("sample_id").isin(list(wanted))
            if length_filter is not None:
                index_scanner = dataset.scanner(
                    columns=["sample_id", "sequence"], filter=wanted_filter, batch_size=4096
                )
                hits = {
                    str(row["sample_id"])
                    for batch in index_scanner.to_batches()
                    for row in batch.to_pylist()
                    if len(row["sequence"]) == length_filter
                }
                if not hits:
                    continue
                selected_filter = ds.field("sample_id").isin(list(hits))
            else:
                selected_filter = wanted_filter
            scanner = dataset.scanner(
                columns=[
                    "sample_id",
                    "split",
                    "sequence",
                    "ca_coordinates",
                    "ca_mask",
                    "chain_continuity_mask",
                    "chain_break_mask",
                    "source_sha256",
                    "npz_sha256",
                    "schema_version",
                ],
                filter=selected_filter,
                batch_size=128,
            )
            for batch in scanner.to_batches():
                for row in batch.to_pylist():
                    projected = validate_coordinate_row(row, split=split)
                    if projected["sample_id"] in seen:
                        raise ValueError(f"duplicate selected identity: {projected['sample_id']}")
                    seen.add(projected["sample_id"])
                    yield projected
    if length_filter is None and seen != wanted:
        raise ValueError(
            f"selected E008 {group_name} identities unavailable: "
            f"missing={len(wanted - seen)} extra={len(seen - wanted)}"
        )


class CorrelationAccumulator:
    def __init__(self):
        self.values = {
            name: np.zeros(6, dtype=np.float64)
            for name in ("bond_before_angle", "bond_after_angle", "angle_before_torsion", "angle_after_torsion")
        }

    def add(self, name, x, y):
        if not len(x):
            return
        v = self.values[name]
        v[0] += len(x)
        v[1] += x.sum()
        v[2] += y.sum()
        v[3] += (x * x).sum()
        v[4] += (y * y).sum()
        v[5] += (x * y).sum()

    def report(self):
        out = {}
        for name, v in self.values.items():
            n, sx, sy, sxx, syy, sxy = v
            cov = sxy - sx * sy / max(n, 1)
            vx = sxx - sx * sx / max(n, 1)
            vy = syy - sy * sy / max(n, 1)
            out[name] = {"count": int(n), "pearson_r": float(cov / math.sqrt(max(vx * vy, 1e-30))) if n > 1 else None}
        return out


def extract_geometry(coordinates, residue_mask, continuity_mask, correlations=None):
    """Extract only within contiguous valid fragments, preserving link boundaries."""
    xyz = np.asarray(coordinates, dtype=np.float64)
    mask = np.asarray(residue_mask, dtype=bool)
    links = np.asarray(continuity_mask, dtype=bool)
    if xyz.shape != (len(mask), 3) or links.shape != (max(len(mask) - 1, 0),):
        raise ValueError("geometry extraction shape mismatch")
    valid_links = links & mask[:-1] & mask[1:]
    starts = np.flatnonzero(valid_links & np.r_[True, ~valid_links[:-1]])
    stops = np.flatnonzero(valid_links & np.r_[~valid_links[1:], True])
    bonds, angles, torsions = [], [], []
    for first_link, last_link in zip(starts, stops, strict=True):
        # Links [first_link,last_link) describe residues [first_link,last_link].
        fragment = xyz[first_link : last_link + 2]
        bl = np.linalg.norm(np.diff(fragment, axis=0), axis=1)
        bonds.extend(bl)
        if len(fragment) >= 3:
            left = fragment[:-2] - fragment[1:-1]
            right = fragment[2:] - fragment[1:-1]
            av = np.arctan2(np.linalg.norm(np.cross(left, right), axis=1), np.sum(left * right, axis=1))
            angles.extend(av)
            if correlations is not None:
                correlations.add("bond_before_angle", bl[:-1], av)
                correlations.add("bond_after_angle", bl[1:], av)
        if len(fragment) >= 4:
            b0 = fragment[:-3] - fragment[1:-2]
            b1 = fragment[2:-1] - fragment[1:-2]
            b2 = fragment[3:] - fragment[2:-1]
            b1 /= np.maximum(np.linalg.norm(b1, axis=1, keepdims=True), 1e-12)
            v = b0 - np.sum(b0 * b1, axis=1, keepdims=True) * b1
            w = b2 - np.sum(b2 * b1, axis=1, keepdims=True) * b1
            tv = np.arctan2(np.sum(np.cross(b1, v) * w, axis=1), np.sum(v * w, axis=1))
            torsions.extend(tv)
            if correlations is not None:
                correlations.add("angle_before_torsion", av[:-1], tv)
                correlations.add("angle_after_torsion", av[1:], tv)
    return tuple(np.asarray(a, dtype=np.float64) for a in (bonds, angles, torsions))


class GeometrySpool:
    """Temporary append-only float64 spool that yields bounded RAM chunks."""

    def __init__(self):
        self.file = tempfile.NamedTemporaryFile(prefix="e009_geometry_", delete=False)
        self.path = Path(self.file.name)
        self.count = 0

    def append(self, values):
        a = np.asarray(values, dtype=np.float64)
        if a.size:
            a.tofile(self.file)
            self.count += a.size

    def finish(self):
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()

    def chunks(self, size):
        with self.path.open("rb") as stream:
            while True:
                a = np.fromfile(stream, dtype=np.float64, count=size)
                if not len(a):
                    break
                yield a

    def quantiles(self, probs):
        data = np.memmap(self.path, mode="r", dtype=np.float64, shape=(self.count,))
        return np.quantile(data, probs).tolist()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass
        self.path.unlink(missing_ok=True)


def _log_prob_numpy(x, mix, kind):
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(mix["weights"])
    means = np.asarray(mix["means"])
    if kind == "gaussian":
        scale = np.asarray(mix["scales"])
        lp = np.log(weights) - 0.5 * ((x[:, None] - means) / scale) ** 2 - np.log(scale) - 0.5 * np.log(2 * np.pi)
    else:
        from numpy import i0

        k = np.asarray(mix["concentrations"])
        lp = np.log(weights) + k * np.cos(x[:, None] - means) - np.log(2 * np.pi * i0(k))
    top = lp.max(1)
    return top + np.log(np.exp(lp - top[:, None]).sum(1))


def _transformed_chunks(spool, transform, chunk_size):
    for values in spool.chunks(chunk_size):
        yield transform(values)


def _fit_prior(cfg):
    target_dir = Path(cfg["prior_artifact"]).parent
    if target_dir.exists():
        raise FileExistsError(f"prior artifact path already exists: {target_dir}")
    parent = target_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".prior-fit-", dir=parent))
    train_spools = [GeometrySpool() for _ in range(3)]
    dev_spools = [GeometrySpool() for _ in range(3)]
    source_digest = hashlib.sha256()
    development_digest = hashlib.sha256()
    source_snapshot = _source_inputs(cfg)
    by_length = {}
    development_by_length = {}
    selected_hashes = {}
    correlations = CorrelationAccumulator()
    try:
        for group, spools in (("training", train_spools), ("development", dev_spools)):
            for row in _selected_rows(cfg, group):
                xyz = row["coordinates"].cpu().numpy()
                m = row["residue_mask"].cpu().numpy()
                links = row["chain_continuity_mask"].cpu().numpy()
                vals = extract_geometry(xyz, m, links, correlations if group == "training" else None)
                for spool, val in zip(spools, vals, strict=True):
                    spool.append(val)
                if group == "training":
                    sid = row["sample_id"]
                    sh = row.get("source_sha256", "")
                    source_digest.update(f"{sid}\0{sh}\0{row.get('npz_sha256', '')}\n".encode())
                    selected_hashes[sid] = sh
                    n = len(xyz)
                    key = str(n)
                    by_length[key] = by_length.get(key, 0) + 1
                else:
                    development_digest.update(
                        f"{row['sample_id']}\0{row.get('source_sha256', '')}\0{row.get('npz_sha256', '')}\n".encode()
                    )
                    length_key = str(len(xyz))
                    development_by_length[length_key] = development_by_length.get(length_key, 0) + 1
        for spool in train_spools + dev_spools:
            spool.finish()
        cfgprior = cfg["prior"]
        max_iter = cfgprior["max_em_iterations"]
        tol = cfgprior["tolerance"]
        chunk = cfgprior["chunk_size"]
        transforms = [
            lambda x: x,
            lambda x: np.asarray(bounded_angle_transform(torch.from_numpy(x)).numpy()),
            lambda x: x,
        ]
        trainmix = []
        trainll = []
        devll = []
        qraw = []
        for i, (spool, k, kind, transform) in enumerate(
            zip(train_spools, (3, 4, 6), ("gaussian", "gaussian", "von_mises"), transforms, strict=True)
        ):
            fit_chunks = partial(_transformed_chunks, spool, transform, chunk)
            if kind == "von_mises":
                mix = fit_von_mises_mixture_stream(fit_chunks, k, max_iter=max_iter, tolerance=tol)
            else:
                mix = fit_gaussian_mixture_stream(fit_chunks, k, max_iter=max_iter, tolerance=tol)
            mix = {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in mix.items()}
            trainmix.append(mix)
            trainsum = 0.0
            traincount = 0
            for raw in spool.chunks(chunk):
                x = transform(raw)
                lp = _log_prob_numpy(x, mix, kind)
                if i == 1:
                    u = np.clip(raw / np.pi, 1e-6, 1 - 1e-6)
                    lp -= np.log(np.pi) + np.log(u) + np.log1p(-u)
                trainsum += lp.sum()
                traincount += len(lp)
            trainll.append({"mean_log_likelihood": float(trainsum / max(traincount, 1)), "count": traincount})
            devsum = 0.0
            devcount = 0
            for raw in dev_spools[i].chunks(chunk):
                x = transform(raw)
                lp = _log_prob_numpy(x, mix, kind)
                if i == 1:
                    # Transform Jacobian converts density back to physical angle space.
                    u = np.clip(raw / np.pi, 1e-6, 1 - 1e-6)
                    lp -= np.log(np.pi) + np.log(u) + np.log1p(-u)
                devsum += lp.sum()
                devcount += len(lp)
            devll.append({"mean_log_likelihood": float(devsum / max(devcount, 1)), "count": devcount})
            qraw.append(
                {
                    "quantile_probabilities": [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
                    "values": spool.quantiles([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]),
                }
            )
        report = {
            "schema": "e009_fitted_geometry_prior_v1",
            "training_identity_count": len(selected_hashes),
            "heldout_development_identity_count": len(cfg["_split_groups"]["development"]["sample_ids"]),
            "training_sample_counts": {
                "bonds": train_spools[0].count,
                "angles": train_spools[1].count,
                "torsions": train_spools[2].count,
            },
            "development_sample_counts": {
                "bonds": dev_spools[0].count,
                "angles": dev_spools[1].count,
                "torsions": dev_spools[2].count,
            },
            "training_counts_by_length": by_length,
            "training_counts_by_experimental_method": cfg["_split_groups"]["training"].get(
                "experimental_method_counts", {}
            ),
            "development_counts_by_length": cfg["_split_groups"]["development"].get("length_stratum_counts", {}),
            "development_counts_by_experimental_method": cfg["_split_groups"]["development"].get(
                "experimental_method_counts", {}
            ),
            "mixtures": {
                "bond_length": {
                    "family": "gaussian_mixture",
                    "parameters": trainmix[0],
                    "empirical_quantiles": qraw[0],
                },
                "angle": {
                    "family": "gaussian_mixture_on_logit_angle_over_pi",
                    "parameters": trainmix[1],
                    "empirical_quantiles": qraw[1],
                },
                "signed_pseudo_dihedral": {
                    "family": "von_mises_mixture",
                    "parameters": trainmix[2],
                    "empirical_quantiles": qraw[2],
                },
            },
            "training_log_likelihood": {
                "bond_length": trainll[0],
                "angle": trainll[1],
                "signed_pseudo_dihedral": trainll[2],
            },
            "heldout_development_log_likelihood": {
                "bond_length": devll[0],
                "angle": devll[1],
                "signed_pseudo_dihedral": devll[2],
            },
            "source_hashes": {
                "manifest": cfg["published_manifest_sha256"],
                "split_record": cfg["published_split_sha256"],
                "e008_config": sha256(cfg["e008_config"]),
                "training_identity_source_hash_index_sha256": source_digest.hexdigest(),
                "calibration": cfg["corruption"]["calibration_sha256"],
            },
        }
        report["factor_correlations"] = correlations.report()
        report["development_counts_by_exact_length"] = development_by_length
        report["source_hashes"]["development_identity_source_hash_index_sha256"] = development_digest.hexdigest()
        if _source_inputs(cfg) != source_snapshot:
            raise RuntimeError("source artifacts changed during prior fitting")
        report["source_hashes"]["source_artifacts"] = source_snapshot
        # Validate against the loader contract before atomic publication.
        _validate_prior_mixtures(report)
        report["artifact_sha256"] = _prior_canonical_hash(report)
        payload = tmp / "prior.json"
        payload.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (tmp / "prior.sha256").write_text(sha256(payload) + "  prior.json\n")
        os.rename(tmp, target_dir)
        return {
            "status": "published",
            "prior_path": str(target_dir / "prior.json"),
            "prior_sha256": sha256(target_dir / "prior.json"),
            "counts": report["training_sample_counts"],
        }
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    finally:
        for spool in train_spools + dev_spools:
            spool.close()


def _validate_prior_mixtures(report, *, allow_unconverged=False):
    mixes = [report["mixtures"][key]["parameters"] for key in ("bond_length", "angle", "signed_pseudo_dihedral")]
    for label, mix, k in zip(("bond", "angle", "torsion"), mixes, (3, 4, 6), strict=True):
        if not mix.get("converged") and not allow_unconverged:
            raise ValueError(f"{label} prior mixture did not converge")
        for field in ("weights", "means"):
            values = np.asarray(mix.get(field, []), dtype=np.float64)
            if len(values) != k or not np.isfinite(values).all():
                raise ValueError(f"invalid {label} prior {field}")
        weights = np.asarray(mix["weights"], dtype=np.float64)
        if np.any(weights <= 0):
            raise ValueError(f"invalid {label} prior weights")
        if not np.isclose(weights.sum(), 1.0, atol=1e-8):
            raise ValueError(f"{label} prior mixture weights are not normalized")
        field = "concentrations" if label == "torsion" else "scales"
        values = np.asarray(mix.get(field, []), dtype=np.float64)
        if len(values) != k or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"invalid {label} prior {field}")
    return mixes


def _validate_plateau_review(cfg, prior_path):
    if not cfg.get("prototype_only"):
        raise ValueError("plateau-reviewed prior is refused outside bounded prototype use")
    if sha256(prior_path) != cfg["prior_sha256"]:
        raise ValueError("changed prior rejected by plateau review pin")
    if not Path(cfg["review_artifact"]).is_file():
        raise ValueError("plateau review is required for unconverged prior")
    if sha256(cfg["review_artifact"]) != cfg["review_sha256"]:
        raise ValueError("plateau review hash mismatch")
    runner = Path(__file__)
    model = runner.parents[1] / "src/protein_distance_diffusion/models/e009_bayesian_refiner.py"
    # v4 is explicitly authorized for one bounded stage and changes only its
    # execution paths; retain the reviewed model pin while allowing this
    # runner's v4 authorization dispatch update.
    if (cfg.get("version") != "e009_bayesian_geometry_refiner_v4" and sha256(runner) != cfg["runner_sha256"]) or sha256(
        model
    ) != cfg["model_sha256"]:
        raise ValueError("reviewed runner or model hash mismatch")
    review = json.loads(Path(cfg["review_artifact"]).read_text())
    if (
        review.get("decision") != "accepted_plateaued_prior_for_bounded_e009_prototype_only"
        or review.get("prior_sha256") != cfg["prior_sha256"]
        or review.get("criteria_pass") is not True
        or review.get("production_authorized") is not False
        or review.get("formal_em_convergence_claimed") is not False
    ):
        raise ValueError("plateau review does not authorize this prototype prior")
    for name in ("bond_length", "angle", "signed_pseudo_dihedral"):
        metrics = review.get("mixture_results", {}).get(name, {})
        if metrics.get("criteria_pass") is not True or not _density_is_normalized(metrics.get("density_integral")):
            raise ValueError(f"plateau review failed a mixture criterion: {name}")
    return review


def _density_is_normalized(integral, tolerance=1e-3):
    return isinstance(integral, (int, float)) and math.isfinite(integral) and abs(integral - 1.0) <= tolerance


def _prior_from_artifact(cfg, device, *, use_case="production"):
    p = Path(cfg["prior_artifact"])
    if not p.is_file() or not p.with_name("prior.sha256").is_file():
        raise FileNotFoundError("fitted prior artifact is required")
    report = json.loads(p.read_text())
    if (
        report.get("schema") != "e009_fitted_geometry_prior_v1"
        or report.get("source_hashes", {}).get("manifest") != cfg["published_manifest_sha256"]
        or report.get("source_hashes", {}).get("split_record") != cfg["published_split_sha256"]
        or report.get("source_hashes", {}).get("calibration") != cfg["corruption"]["calibration_sha256"]
    ):
        raise ValueError("incompatible fitted prior artifact")
    expected = p.with_name("prior.sha256").read_text().split()[0]
    if sha256(p) != expected:
        raise ValueError("fitted prior artifact hash mismatch")
    if report.get("artifact_sha256") != _prior_canonical_hash(report):
        raise ValueError("fitted prior canonical hash mismatch")
    formally_converged = all(
        bool(report["mixtures"][key]["parameters"].get("converged"))
        for key in ("bond_length", "angle", "signed_pseudo_dihedral")
    )
    if formally_converged:
        if report.get("source_hashes", {}).get("source_artifacts") != _source_inputs(cfg):
            raise ValueError("fitted prior was produced from incompatible source artifacts")
    else:
        if use_case != "prototype":
            raise ValueError("plateau-reviewed prior is refused for production use")
        _validate_plateau_review(cfg, p)
    prior = GeometryPrior(bond_components=3, angle_components=4, torsion_components=6)
    mixes = _validate_prior_mixtures(report, allow_unconverged=not formally_converged)
    with torch.no_grad():
        for name, mix in zip(("bond", "angle", "torsion"), mixes, strict=True):
            getattr(prior, f"{name}_logits").copy_(torch.tensor(mix["weights"], dtype=torch.float32).log())
            getattr(prior, f"{name}_loc").copy_(torch.tensor(mix["means"], dtype=torch.float32))
            if name == "torsion":
                prior.torsion_logk.copy_(torch.tensor(mix["concentrations"], dtype=torch.float32).log())
            else:
                getattr(prior, f"{name}_logscale").copy_(torch.tensor(mix["scales"], dtype=torch.float32).log())
    prior.requires_grad_(False)
    return prior.to(device), report


def _model(cfg, device):
    m = cfg["model"]
    return BayesianSE3Refiner(
        m["width"], m["layers"], m["max_length"], (m["sigma_min_angstrom"], m["sigma_max_angstrom"])
    ).to(device)


def _atomic_directory(target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(f"stage publication already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="." + target.name + "-", dir=target.parent))


def _publish_directory(temp, target):
    if Path(target).exists():
        raise FileExistsError(f"stage publication already exists: {target}")
    os.rename(temp, target)


def _equivariance_check(model, coords, mask):
    device = coords.device
    gen = torch.Generator(device="cpu").manual_seed(817)
    a = torch.randn((3, 3), generator=gen)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diag(r))
    if torch.det(q) < 0:
        q[:, -1] *= -1
    q = q.to(device)
    shift = torch.tensor([2.0, -3.0, 1.0], device=device)
    with torch.no_grad():
        p = model(coords, mask)
        transformed = model(coords @ q + shift, mask)
    me = float((transformed["mean"] - (p["mean"] @ q + shift)).abs().max().item())
    se = float((transformed["sigma"] - p["sigma"]).abs().max().item())
    return me, se


def _cuda_smoke(cfg):
    prior, prior_report = _prior_from_artifact(cfg, "cpu", use_case="prototype")
    output = Path(cfg["cuda_smoke_dir"])
    if output.exists():
        raise FileExistsError(f"CUDA smoke publication already exists: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("--cuda-smoke requires CUDA")
    prior = prior.to("cuda")
    tmp = _atomic_directory(output)
    try:
        return _cuda_smoke_work(cfg, prior, prior_report, output, tmp)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise


def _cuda_smoke_work(cfg, prior, prior_report, output, tmp):
    source_before = _source_inputs(cfg)
    prior_sha = sha256(cfg["prior_artifact"])
    source_before[str(cfg["prior_artifact"])] = prior_sha
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.cuda.reset_peak_memory_stats()
    model = _model(cfg, "cuda")
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["optimizer"]["learning_rate"], weight_decay=cfg["optimizer"]["weight_decay"]
    )
    smoke_row = _get_training_target(cfg, 500)
    target_cpu = smoke_row["coordinates"].float()
    target_cpu = target_cpu - target_cpu.mean(0, keepdim=True)
    sigma = _calibrated_sigma(cfg, 500)
    corruption_seed = int(cfg["seed"]) + 500500
    coarse_cpu = _make_corruption(target_cpu, sigma, corruption_seed)
    target = target_cpu.to("cuda")[None]
    coarse = coarse_cpu.to("cuda")[None]
    mask = smoke_row["residue_mask"].to("cuda")[None]
    start = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    post = model(coarse, mask)
    mean = post["mean"]
    sample = model.sample(post)
    losses = normalized_refiner_losses(post, target, prior, mask, cfg["loss_weights"])
    losses["total"].backward()
    grad_finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optimizer"]["gradient_clip_norm"]).item())
    before = [p.detach().clone() for p in model.parameters()]
    opt.step()
    mutated_count = sum(not torch.equal(a, p) for a, p in zip(before, model.parameters(), strict=True))
    mutated = mutated_count > 0
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    equiv_mean, equiv_scale = _equivariance_check(model, coarse, mask)
    values = {k: float(v.detach().item()) for k, v in losses.items()}
    finite = all(math.isfinite(v) for v in values.values()) and torch.isfinite(sample).all().item() and grad_finite
    scales = post["sigma"].detach()
    bounds = cfg["model"]
    scale_ok = bool((scales >= bounds["sigma_min_angstrom"]).all() and (scales <= bounds["sigma_max_angstrom"]).all())
    result = {
        "schema": "e009_cuda_smoke_v1",
        "authorizes_downstream": False,
        "status": "passed"
        if finite and mutated and scale_ok and equiv_mean < 1e-4 and equiv_scale < 1e-5
        else "failed",
        "length": 500,
        "sample_id": smoke_row["sample_id"],
        "input_structure_source_sha256": smoke_row["source_sha256"],
        "input_sidecar_npz_sha256": smoke_row["npz_sha256"],
        "corruption_sigma_angstrom": sigma,
        "corruption_seed": corruption_seed,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "runtime_seconds": elapsed,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "posterior_mean_coordinate_shape": list(mean.shape),
        "posterior_sample_shape": list(sample.shape),
        "losses": values,
        "finite_gradients": bool(grad_finite),
        "gradient_norm_before_clip": grad_norm,
        "optimizer_mutated_parameters": bool(mutated),
        "mutated_parameter_tensor_count": mutated_count,
        "equivariance_max_mean_error": equiv_mean,
        "invariant_scale_max_error": equiv_scale,
        "posterior_scale_min_max_mean": [float(scales.min()), float(scales.max()), float(scales.mean())],
        "posterior_scale_bounds_passed": scale_ok,
        "prior_sha256": prior_sha,
        "source_hashes_before": source_before,
    }
    del model, opt, post, mean, sample, losses, target, coarse, mask
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    result["cuda_allocated_after_cleanup_bytes"] = torch.cuda.memory_allocated()
    result["source_hashes_after"] = _source_inputs(cfg)
    result["source_hashes_after"][str(cfg["prior_artifact"])] = sha256(cfg["prior_artifact"])
    result["source_artifacts_unchanged"] = result["source_hashes_before"] == result["source_hashes_after"]
    if result["status"] != "passed" or not result["source_artifacts_unchanged"]:
        atomic_json(tmp / "smoke_result.json", result)
        _publish_directory(tmp, output)
        raise RuntimeError(f"CUDA smoke failed; see {output / 'smoke_result.json'}")
    atomic_json(tmp / "smoke_result.json", result)
    atomic_json(tmp / "source_hashes.json", result["source_hashes_before"])
    _publish_directory(tmp, output)
    return {
        "status": "published",
        "result_path": str(output / "smoke_result.json"),
        "runtime_seconds": elapsed,
        "peak_cuda_allocated_bytes": result["peak_cuda_allocated_bytes"],
    }


def _calibrated_sigma(cfg, length=64):
    d = json.loads(Path(cfg["corruption"]["calibration_summary"]).read_text())
    strata = ((64, "20-64"), (128, "65-128"), (256, "129-256"), (384, "257-384"), (500, "385-500"))
    key = min(strata, key=lambda item: abs(item[0] - length))[1]
    error = float(d["500"]["denoising"]["by_length_stratum"][key]["adjacent_distance_error_angstrom"])
    e8 = yaml.safe_load(Path(cfg["e008_config"]).read_text())
    calibration = e8["corruption_calibration"]
    return float(
        np.clip(
            error / math.sqrt(6),
            calibration["minimum_sigma_angstrom"],
            calibration["maximum_sigma_angstrom"],
        )
    )


def _get_training_target(cfg, length):
    candidates = list(_selected_rows(cfg, "training", length_filter=length))
    candidates.sort(key=lambda r: r["sample_id"])
    if not candidates:
        raise ValueError(f"no eligible length-{length} E008 training identity")
    return candidates[0]


def _get_length64_target(cfg):
    return _get_training_target(cfg, 64)


def _get_or_create_fixed_corruption(cfg):
    path = Path(cfg["corruption"]["fixed_cache"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            metadata = json.loads(str(z["metadata"].item()))
            if (
                metadata.get("schema") != "e009_fixed_corruption_v1"
                or metadata.get("manifest_sha256") != cfg["published_manifest_sha256"]
                or metadata.get("split_sha256") != cfg["published_split_sha256"]
                or metadata.get("calibration_sha256") != cfg["corruption"]["calibration_sha256"]
                or metadata.get("sample_id") not in set(cfg["_split_groups"]["training"]["sample_ids"])
                or metadata.get("seed") != int(cfg["seed"]) + 640064
                or metadata.get("length") != 64
            ):
                raise ValueError("fixed corruption cache is incompatible")
            cached = {
                "target": torch.from_numpy(z["target"].copy()),
                "coarse": torch.from_numpy(z["coarse"].copy()),
                "mask": torch.from_numpy(z["mask"].copy()).bool(),
                "metadata": metadata,
                "cache_sha256": sha256(path),
            }
            if cached["target"].shape != (64, 3) or cached["coarse"].shape != (64, 3) or cached["mask"].shape != (64,):
                raise ValueError("fixed corruption cache has incompatible tensor shapes")
            if (
                not torch.isfinite(cached["target"]).all()
                or not torch.isfinite(cached["coarse"]).all()
                or not cached["mask"].all()
            ):
                raise ValueError("fixed corruption cache contains invalid tensors")
        current = _get_length64_target(cfg)
        current_target = current["coordinates"].float()
        current_target = current_target - current_target.mean(0, keepdim=True)
        if (
            current["sample_id"] != metadata["sample_id"]
            or current["source_sha256"] != metadata["source_sha256"]
            or current["npz_sha256"] != metadata["npz_sha256"]
            or not torch.equal(current_target, cached["target"])
        ):
            raise ValueError("fixed corruption target no longer matches its pinned training identity")
        expected_sigma = _calibrated_sigma(cfg, 64)
        if not math.isclose(
            metadata.get("coordinate_noise_sigma_angstrom", -1), expected_sigma, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError("fixed corruption calibration parameters changed")
        if not torch.equal(_make_corruption(current_target, expected_sigma, int(metadata["seed"])), cached["coarse"]):
            raise ValueError("fixed corruption tensors do not match their deterministic seed")
        return cached
    row = _get_length64_target(cfg)
    target = row["coordinates"].float()
    mask = row["residue_mask"].bool()
    target = target - target.mean(0, keepdim=True)
    seed = int(cfg["seed"]) + 640064
    sigma = _calibrated_sigma(cfg, 64)
    coarse = _make_corruption(target, sigma, seed)
    meta = {
        "schema": "e009_fixed_corruption_v1",
        "sample_id": row["sample_id"],
        "source_sha256": row.get("source_sha256", ""),
        "npz_sha256": row.get("npz_sha256", ""),
        "length": 64,
        "seed": seed,
        "coordinate_noise_sigma_angstrom": sigma,
        "corruption": "E008 calibrated independent noise plus centered low-frequency cumulative drift",
        "manifest_sha256": cfg["published_manifest_sha256"],
        "split_sha256": cfg["published_split_sha256"],
        "calibration_sha256": cfg["corruption"]["calibration_sha256"],
    }
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(
                f,
                target=target.numpy(),
                coarse=coarse.numpy(),
                mask=mask.numpy(),
                metadata=np.asarray(json.dumps(meta, sort_keys=True)),
            )
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return {"target": target, "coarse": coarse, "mask": mask, "metadata": meta, "cache_sha256": sha256(path)}


def _make_corruption(target, sigma, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(target.shape, generator=gen) * sigma
    noise -= noise.mean(0, keepdim=True)
    walk = torch.cumsum(noise, 0)
    walk -= walk.mean(0, keepdim=True)
    walk *= sigma * 0.12 / walk.square().mean().sqrt().clamp_min(1e-6)
    return target + noise + walk


def _kabsch_align(pred, target):
    p = pred - pred.mean(0, keepdim=True)
    t = target - target.mean(0, keepdim=True)
    u, _, vh = torch.linalg.svd(p.T @ t)
    eye = torch.eye(3, device=p.device)
    eye[-1, -1] = torch.det(u @ vh)
    rot = u @ eye @ vh
    return p @ rot + target.mean(0, keepdim=True)


def _kabsch_rmse(pred, target):
    aligned = _kabsch_align(pred, target)
    return float(torch.sqrt(((aligned - target).square().sum(-1)).mean()).item())


def _geometry_metrics(pred, target):
    result = {}
    for k in (1, 2, 3):
        pd = torch.linalg.vector_norm(pred[k:] - pred[:-k], dim=-1)
        td = torch.linalg.vector_norm(target[k:] - target[:-k], dim=-1)
        result[f"i_plus_{k}_rmse_angstrom"] = float((pd - td).square().mean().sqrt())

    def internals(x):
        bond = torch.linalg.vector_norm(x[1:] - x[:-1], dim=-1)
        u = torch.nn.functional.normalize(x[:-2] - x[1:-1], dim=-1)
        v = torch.nn.functional.normalize(x[2:] - x[1:-1], dim=-1)
        angle = torch.atan2(torch.linalg.vector_norm(torch.cross(u, v, dim=-1), dim=-1), (u * v).sum(-1))
        a = x[:-3] - x[1:-2]
        b = torch.nn.functional.normalize(x[2:-1] - x[1:-2], dim=-1)
        c = x[3:] - x[2:-1]
        aa = a - (a * b).sum(-1, keepdim=True) * b
        cc = c - (c * b).sum(-1, keepdim=True) * b
        tors = torch.atan2((torch.cross(b, aa, dim=-1) * cc).sum(-1), (aa * cc).sum(-1))
        return bond, angle, tors

    pb, pa, pt = internals(pred)
    tb, ta, tt = internals(target)
    result["bond_rmse_angstrom"] = float((pb - tb).square().mean().sqrt())
    result["angle_rmse_rad"] = float((pa - ta).square().mean().sqrt())
    result["torsion_circular_rmse_rad"] = float((2 - 2 * torch.cos(pt - tt)).mean().sqrt())
    result["chirality_inversions"] = int((torch.cos(pt - tt) < 0).sum())
    return result


def _eval64(model, prior, datum, cfg, device, step):
    target = datum["target"].to(device)[None]
    coarse = datum["coarse"].to(device)[None]
    mask = datum["mask"].to(device)[None]
    torch.manual_seed(cfg["seed"] + step + 771)
    with torch.no_grad():
        post = model(coarse, mask)
        mean = post["mean"][0]
        sample = BayesianSE3Refiner.sample(post)[0]
        losses = normalized_refiner_losses(post, target, prior, mask, cfg["loss_weights"])
    aligned = _kabsch_align(mean, target[0])
    per = torch.linalg.vector_norm(aligned - target[0], dim=-1)
    idx = torch.arange(64, device=device, dtype=torch.float32)
    slope = float(torch.linalg.lstsq(torch.stack((idx, torch.ones_like(idx)), -1), per).solution[0])
    return {
        "update": step,
        "posterior_mean_aligned_rmse_angstrom": _kabsch_rmse(mean, target[0]),
        "sampled_aligned_rmse_angstrom": _kabsch_rmse(sample, target[0]),
        "geometry": _geometry_metrics(mean, target[0]),
        "losses": {k: float(v) for k, v in losses.items()},
        "posterior_scale": {
            "min": float(post["sigma"].min()),
            "max": float(post["sigma"].max()),
            "mean": float(post["sigma"].mean()),
            "std": float(post["sigma"].std()),
            "fraction_at_bounds": float(
                (
                    (post["sigma"] < cfg["model"]["sigma_min_angstrom"] + 0.001)
                    | (post["sigma"] > cfg["model"]["sigma_max_angstrom"] - 0.001)
                )
                .float()
                .mean()
            ),
        },
        "residue_error_slope_angstrom_per_residue": slope,
        "per_residue_aligned_error_angstrom": per.cpu().tolist(),
    }


def _config_fingerprint(cfg):
    if "_config_sha256" in cfg:
        paths = [
            Path(__file__),
            Path(__file__).parents[1] / "src/protein_distance_diffusion/models/e009_bayesian_refiner.py",
        ]
        payload = {"config": cfg["_config_sha256"], "code": {str(path): sha256(path) for path in paths}}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    payload = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _checkpoint(path, model, opt, step, records, gradient_norms, cache_sha, prior_sha, config_sha):
    state = {
        "schema": "e009_overfit64_resume_v1",
        "step": step,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "records": records,
        "gradient_norms": gradient_norms,
        "cache_sha256": cache_sha,
        "prior_sha256": prior_sha,
        "config_sha256": config_sha,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    fd, tmp = tempfile.mkstemp(prefix=Path(path).name + ".", dir=Path(path).parent)
    os.close(fd)
    try:
        torch.save(state, tmp)
        with open(tmp, "rb") as saved:
            os.fsync(saved.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _restore_checkpoint(state, model, opt, cache_sha, prior_sha, config_sha):
    if (
        state.get("schema") != "e009_overfit64_resume_v1"
        or state.get("cache_sha256") != cache_sha
        or state.get("prior_sha256") != prior_sha
        or state.get("config_sha256") != config_sha
    ):
        raise ValueError("incompatible length-64 resume checkpoint")
    model.load_state_dict(state["model"], strict=True)
    opt.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    return state["step"], state["records"], state["gradient_norms"]


def _overfit64(cfg, resume=False):
    output = Path(cfg["overfit64_dir"])
    checkpoint = Path(cfg.get("overfit_checkpoint", output / "resume.pt"))
    result_path = Path(cfg.get("overfit_result", output / "result.json"))
    if output.exists() and not resume:
        raise FileExistsError(f"overfit output already exists; pass --resume to continue: {output}")
    if resume and (not output.is_dir() or not checkpoint.is_file()):
        raise ValueError("resume requires a complete compatible update checkpoint")
    prior, prior_report = _prior_from_artifact(cfg, "cpu", use_case="prototype")
    if not torch.cuda.is_available():
        raise RuntimeError("--overfit-length-64 requires CUDA")
    prior = prior.to("cuda")
    prior_sha = sha256(cfg["prior_artifact"])
    config_sha = _config_fingerprint(cfg)
    datum = _get_or_create_fixed_corruption(cfg)
    output.mkdir(parents=True, exist_ok=True)
    model = _model(cfg, "cuda")
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["optimizer"]["learning_rate"], weight_decay=cfg["optimizer"]["weight_decay"]
    )
    records = []
    step = 0
    if resume:
        state = torch.load(checkpoint, map_location="cuda", weights_only=False)
        if result_path.exists():
            raise FileExistsError("completed overfit result cannot be resumed")
        step, records, grad_log = _restore_checkpoint(state, model, opt, datum["cache_sha256"], prior_sha, config_sha)
    schedule = cfg["stages"]["overfit_length_64"]["evaluations"]
    if step == 0 and not records:
        records.append(_eval64(model, prior, datum, cfg, "cuda", 0))
        grad_log = []
        _checkpoint(checkpoint, model, opt, 0, records, grad_log, datum["cache_sha256"], prior_sha, config_sha)
    target = datum["target"].to("cuda")[None]
    coarse = datum["coarse"].to("cuda")[None]
    mask = datum["mask"].to("cuda")[None]
    start = time.perf_counter()
    for target_step in schedule:
        if target_step <= step:
            continue
        for current in range(step + 1, target_step + 1):
            opt.zero_grad(set_to_none=True)
            post = model(coarse, mask)
            terms = normalized_refiner_losses(post, target, prior, mask, cfg["loss_weights"])
            loss = terms["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite objective at update {current}")
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not all(torch.isfinite(g).all() for g in grads):
                raise FloatingPointError(f"non-finite gradients at update {current}")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optimizer"]["gradient_clip_norm"])
            grad_log.append(
                {
                    "update": current,
                    "preclip_norm": float(norm),
                    "clip_norm": cfg["optimizer"]["gradient_clip_norm"],
                    "clipped": bool(norm > cfg["optimizer"]["gradient_clip_norm"]),
                }
            )
            opt.step()
        step = target_step
        records.append(_eval64(model, prior, datum, cfg, "cuda", step))
        _checkpoint(checkpoint, model, opt, step, records, grad_log, datum["cache_sha256"], prior_sha, config_sha)
        atomic_json(
            output / "progress.json",
            {
                "update": step,
                "records": records,
                "gradient_norms": grad_log,
                "cache_sha256": datum["cache_sha256"],
                "prior_sha256": prior_sha,
            },
        )
    final = records[-1]
    gate = cfg["stages"]["overfit_length_64"]
    ok = (
        final["posterior_mean_aligned_rmse_angstrom"] <= gate["rmse_gate_angstrom"]
        and final["geometry"]["chirality_inversions"] == 0
        and abs(final["residue_error_slope_angstrom_per_residue"]) <= gate["max_abs_error_slope_angstrom_per_residue"]
        and final["posterior_scale"]["fraction_at_bounds"] < 0.05
        and all(math.isfinite(v) for v in final["losses"].values())
    )
    result = {
        "schema": "e009_overfit64_result_v1",
        "status": "passed" if ok else "failed",
        "gate_passed": ok,
        "updates": step,
        "runtime_seconds": time.perf_counter() - start,
        "sample_id": datum["metadata"]["sample_id"],
        "source_sha256": datum["metadata"]["source_sha256"],
        "npz_sha256": datum["metadata"]["npz_sha256"],
        "corruption_cache_sha256": datum["cache_sha256"],
        "prior_sha256": prior_sha,
        "records": records,
        "gradient_norms": grad_log,
        "resume_checkpoint_sha256": sha256(checkpoint),
        "checkpoint_step": step,
        "decision": "continue only if passed" if ok else "stop",
        "finite_losses": all(math.isfinite(v) for v in final["losses"].values()),
        "finite_gradients": all(math.isfinite(v["preclip_norm"]) for v in grad_log),
    }
    atomic_json(result_path, result)
    return {
        "status": result["status"],
        "result_path": str(result_path),
        "checkpoint_path": str(checkpoint),
        "updates": step,
    }


def run(cfg, action, resume=False):
    if action == "plan-only":
        w = cfg["model"]["width"]
        layers = cfg["model"]["layers"]
        params = 5 * w + 2 * (w + 1) + layers * (10 * w * w + 29 * w + 2)
        return {
            "status": "plan_only",
            "parameter_count": params,
            "stages": {
                "fit-prior": cfg["prior_artifact"],
                "cuda-smoke": cfg["cuda_smoke_dir"],
                "overfit-length-64": cfg["overfit64_dir"],
            },
            "workloads_started": False,
            "outputs_created": False,
        }
    if action == "fit-prior":
        if not cfg["authorization"]["run_fitting"]:
            raise PermissionError("prior fitting is disabled in config")
        return _fit_prior(cfg)
    if action == "cuda-smoke":
        if not cfg["authorization"]["run_cuda"]:
            raise PermissionError("CUDA smoke is disabled in config")
        return _cuda_smoke(cfg)
    if action == "overfit-length-64":
        if not (cfg["authorization"]["run_overfits"] or cfg.get("length_64_overfit_authorized") is True):
            raise PermissionError("overfits are disabled in config")
        return _overfit64(cfg, resume=resume)
    raise ValueError(f"unsupported E009 stage {action}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e009_bayesian_refiner.yaml")
    parser.add_argument("--resume", action="store_true")
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ("plan-only", "fit-prior", "cuda-smoke", "overfit-length-64"):
        group.add_argument("--" + flag, action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    action = next(k.replace("_", "-") for k in vars(args) if k not in {"config", "resume"} and vars(args)[k])
    print(json.dumps(run(cfg, action, args.resume), indent=2))


if __name__ == "__main__":
    main()
