import json
import math

import numpy as np
import pytest
import torch

from protein_distance_diffusion.models.e009_bayesian_refiner import (
    GeometryPrior,
    bounded_angle_transform,
    fit_gaussian_mixture_stream,
    fit_von_mises_mixture_stream,
)
from scripts import run_e009_bayesian_refiner as e009


def _chunk_factory(values, size):
    return lambda: (values[i : i + size] for i in range(0, len(values), size))


def test_deterministic_streaming_gaussian_fit_and_heldout_separation():
    rng = np.random.default_rng(51)
    train = np.r_[rng.normal(-1.2, 0.15, 500), rng.normal(1.1, 0.25, 500)]
    dev = rng.normal(4.0, 0.2, 400)
    a = fit_gaussian_mixture_stream(_chunk_factory(train, 37), 2, max_iter=100)
    b = fit_gaussian_mixture_stream(_chunk_factory(train, 113), 2, max_iter=100)
    c = fit_gaussian_mixture_stream(_chunk_factory(train, 64), 2, max_iter=100)
    assert np.allclose(a["means"], b["means"], atol=1e-10)
    assert np.allclose(a["weights"], c["weights"], atol=1e-10)
    assert a["converged"] and np.isfinite(a["scales"]).all()
    assert a["training_log_likelihood"] > np.mean(e009._log_prob_numpy(dev, a, "gaussian"))


def test_bounded_angle_transform_and_von_mises_normalization():
    edges = torch.tensor([0.0, 1e-12, math.pi / 2, math.pi - 1e-12, math.pi])
    transformed = bounded_angle_transform(edges)
    assert torch.isfinite(transformed).all()
    assert transformed[0] < transformed[-1]
    angles = np.linspace(-math.pi, math.pi, 3000, endpoint=False)
    mix = fit_von_mises_mixture_stream(_chunk_factory(angles, 77), 6, max_iter=40)
    mix_chunked = fit_von_mises_mixture_stream(_chunk_factory(angles, 239), 6, max_iter=40)
    assert np.allclose(mix["means"], mix_chunked["means"], atol=1e-10)
    assert np.allclose(mix["concentrations"], mix_chunked["concentrations"], atol=1e-10)
    prior = GeometryPrior(bond_components=3, angle_components=4, torsion_components=6)
    with torch.no_grad():
        prior.torsion_logits.copy_(torch.tensor(mix["weights"]).log())
        prior.torsion_loc.copy_(torch.tensor(mix["means"]))
        prior.torsion_logk.copy_(torch.tensor(mix["concentrations"]).log())
    grid = torch.tensor(angles, dtype=torch.float32)
    logp = prior.log_prob(torch.ones_like(grid), torch.ones_like(grid), grid)["torsion"]
    integral = torch.exp(logp).mean() * (2 * math.pi)
    assert torch.isfinite(logp).all()
    assert integral.item() == pytest.approx(1.0, abs=0.02)


def test_geometry_extraction_honors_mask_and_chain_boundaries():
    x = np.stack((np.arange(8.0), np.zeros(8), np.zeros(8)), axis=-1)
    mask = np.ones(8, dtype=bool)
    links = np.array([1, 1, 0, 1, 1, 1, 1], dtype=bool)
    bonds, angles, torsions = e009.extract_geometry(x, mask, links)
    assert len(bonds) == 6
    assert len(angles) == 4
    assert len(torsions) == 2
    mask[5] = False
    bonds2, angles2, torsions2 = e009.extract_geometry(x, mask, links)
    assert len(bonds2) == 4
    assert len(angles2) == 1
    assert len(torsions2) == 0


def test_split_parser_returns_only_training_and_development(tmp_path):
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps(
            {
                "groups": {
                    "training": {"sample_ids": ["train"]},
                    "development": {"sample_ids": ["dev"]},
                    "prospective": {"sample_ids": ["must_not_be_loaded"]},
                }
            }
        )
    )
    groups = e009.read_named_split_groups(path, ("training", "development"))
    assert set(groups) == {"training", "development"}


def test_fixed_corruption_cache_is_deterministic_and_identity_bound(monkeypatch, tmp_path):
    target = torch.arange(192, dtype=torch.float32).reshape(64, 3) / 10
    row = {
        "sample_id": "training-64",
        "source_sha256": "source-pin",
        "npz_sha256": "sidecar-pin",
        "coordinates": target,
        "residue_mask": torch.ones(64, dtype=torch.bool),
    }
    monkeypatch.setattr(e009, "_get_length64_target", lambda cfg: row)
    monkeypatch.setattr(e009, "_calibrated_sigma", lambda cfg, length=64: 0.4)
    cache = tmp_path / "fixed.npz"
    cfg = {
        "corruption": {
            "fixed_cache": str(cache),
            "calibration_sha256": "cal",
        },
        "seed": 17,
        "published_manifest_sha256": "manifest",
        "published_split_sha256": "split",
        "_split_groups": {"training": {"sample_ids": ["training-64"]}},
    }
    first = e009._get_or_create_fixed_corruption(cfg)
    second = e009._get_or_create_fixed_corruption(cfg)
    assert first["cache_sha256"] == second["cache_sha256"]
    assert torch.equal(first["target"], second["target"])
    assert torch.equal(first["coarse"], second["coarse"])
    assert first["metadata"]["sample_id"] == "training-64"


def test_plan_only_and_real_handler_dispatch(monkeypatch):
    cfg = {
        "model": {"width": 128, "layers": 7},
        "prior_artifact": "prior",
        "cuda_smoke_dir": "smoke",
        "overfit64_dir": "overfit",
        "authorization": {"run_fitting": True, "run_cuda": True, "run_overfits": True},
    }
    initialized_before = torch.cuda.is_initialized()
    monkeypatch.setattr(e009, "BayesianSE3Refiner", lambda *args, **kwargs: pytest.fail("plan-only constructed model"))
    plan = e009.run(cfg, "plan-only")
    assert plan["parameter_count"] == 1_173_776
    assert plan["outputs_created"] is False
    assert torch.cuda.is_initialized() is initialized_before
    calls = []
    monkeypatch.setattr(e009, "_fit_prior", lambda c: calls.append("fit") or {"status": "fit"})
    monkeypatch.setattr(e009, "_cuda_smoke", lambda c: calls.append("smoke") or {"status": "smoke"})
    monkeypatch.setattr(
        e009, "_overfit64", lambda c, resume=False: calls.append(("overfit", resume)) or {"status": "overfit"}
    )
    assert e009.run(cfg, "fit-prior")["status"] == "fit"
    assert e009.run(cfg, "cuda-smoke")["status"] == "smoke"
    assert e009.run(cfg, "overfit-length-64", resume=True)["status"] == "overfit"
    assert calls == ["fit", "smoke", ("overfit", True)]


def test_atomic_directory_publication_refuses_overwrite(tmp_path):
    target = tmp_path / "published"
    staged = e009._atomic_directory(target)
    (staged / "record.json").write_text("{}")
    e009._publish_directory(staged, target)
    assert (target / "record.json").exists()
    with pytest.raises(FileExistsError):
        e009._atomic_directory(target)


def test_resume_checkpoint_replays_same_next_update(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    torch.manual_seed(112)
    model_a = torch.nn.Linear(3, 2)
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
    x = torch.randn(4, 3)
    loss = model_a(x).square().mean()
    loss.backward()
    opt_a.step()
    opt_a.zero_grad(set_to_none=True)
    ckpt = tmp_path / "resume.pt"
    e009._checkpoint(ckpt, model_a, opt_a, 10, [], [], "cache", "prior", "config")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    next_input = torch.randn(4, 3)
    next_loss = model_a(next_input).square().mean()
    next_loss.backward()
    opt_a.step()
    model_b = torch.nn.Linear(3, 2)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)
    e009._restore_checkpoint(state, model_b, opt_b, "cache", "prior", "config")
    same_input = torch.randn(4, 3)
    assert torch.equal(next_input, same_input)
    loss_b = model_b(same_input).square().mean()
    loss_b.backward()
    opt_b.step()
    for a, b in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert torch.equal(a, b)
    with pytest.raises(ValueError, match="incompatible"):
        e009._restore_checkpoint(state, model_b, opt_b, "other-cache", "prior", "config")


def test_missing_or_incompatible_prior_is_refused(tmp_path):
    cfg = {
        "prior_artifact": str(tmp_path / "missing" / "prior.json"),
        "published_manifest_sha256": "m",
        "published_split_sha256": "s",
        "corruption": {"calibration_sha256": "c"},
    }
    with pytest.raises(FileNotFoundError, match="fitted prior"):
        e009._prior_from_artifact(cfg, "cpu")
    p = tmp_path / "bad" / "prior.json"
    p.parent.mkdir()
    p.write_text("{}")
    p.with_name("prior.sha256").write_text(e009.sha256(p) + "  prior.json\n")
    cfg["prior_artifact"] = str(p)
    with pytest.raises(ValueError, match="incompatible"):
        e009._prior_from_artifact(cfg, "cpu")


def test_prior_fit_parameters_round_trip_through_artifact_loader(monkeypatch, tmp_path):
    mixtures = {}
    for key, k, family in (
        ("bond_length", 3, "gaussian_mixture"),
        ("angle", 4, "gaussian_mixture_on_logit_angle_over_pi"),
        ("signed_pseudo_dihedral", 6, "von_mises_mixture"),
    ):
        params = {
            "weights": [1 / k] * k,
            "means": [float(i) for i in range(k)],
            "converged": True,
            "scales" if key != "signed_pseudo_dihedral" else "concentrations": [1.0] * k,
        }
        mixtures[key] = {"family": family, "parameters": params}
    report = {
        "schema": "e009_fitted_geometry_prior_v1",
        "source_hashes": {
            "manifest": "m",
            "split_record": "s",
            "calibration": "c",
            "source_artifacts": {"pin": "hash"},
        },
        "mixtures": mixtures,
    }
    report["artifact_sha256"] = e009._prior_canonical_hash(report)
    target = tmp_path / "prior_v2" / "prior.json"
    target.parent.mkdir()
    target.write_text(json.dumps(report))
    target.with_name("prior.sha256").write_text(e009.sha256(target) + "  prior.json\n")
    cfg = {
        "prior_artifact": str(target),
        "published_manifest_sha256": "m",
        "published_split_sha256": "s",
        "corruption": {"calibration_sha256": "c"},
    }
    monkeypatch.setattr(e009, "_source_inputs", lambda _cfg: {"pin": "hash"})
    prior, loaded = e009._prior_from_artifact(cfg, "cpu")
    assert loaded["artifact_sha256"] == report["artifact_sha256"]
    assert (prior.bond_logits.numel(), prior.angle_logits.numel(), prior.torsion_logits.numel()) == (3, 4, 6)


def _write_plateau_fixture(tmp_path, monkeypatch):
    from pathlib import Path

    runner = Path(e009.__file__)
    model_file = runner.parents[1] / "src/protein_distance_diffusion/models/e009_bayesian_refiner.py"
    mixtures = {}
    for key, k in (("bond_length", 3), ("angle", 4), ("signed_pseudo_dihedral", 6)):
        par = {
            "weights": [1 / k] * k,
            "means": [float(i) for i in range(k)],
            "converged": False,
            "history": [0.0, 0.0],
            "iterations": 2,
            "scales" if key != "signed_pseudo_dihedral" else "concentrations": [1.0] * k,
        }
        mixtures[key] = {"family": "fixture", "parameters": par}
    report = {
        "schema": "e009_fitted_geometry_prior_v1",
        "source_hashes": {"manifest": "m", "split_record": "s", "calibration": "c"},
        "mixtures": mixtures,
    }
    report["artifact_sha256"] = e009._prior_canonical_hash(report)
    prior_file = tmp_path / "prior_v1" / "prior.json"
    prior_file.parent.mkdir()
    prior_file.write_text(json.dumps(report))
    prior_file.with_name("prior.sha256").write_text(e009.sha256(prior_file) + "  prior.json\n")
    review = {
        "decision": "accepted_plateaued_prior_for_bounded_e009_prototype_only",
        "prior_sha256": e009.sha256(prior_file),
        "criteria_pass": True,
        "production_authorized": False,
        "formal_em_convergence_claimed": False,
        "mixture_results": {name: {"criteria_pass": True, "density_integral": 1.0} for name in mixtures},
    }
    review_file = tmp_path / "plateau_review.json"
    review_file.write_text(json.dumps(review))
    cfg = {
        "prior_artifact": str(prior_file),
        "published_manifest_sha256": "m",
        "published_split_sha256": "s",
        "corruption": {"calibration_sha256": "c"},
        "review_artifact": str(review_file),
        "review_sha256": e009.sha256(review_file),
        "prior_sha256": e009.sha256(prior_file),
        "runner_sha256": e009.sha256(runner),
        "model_sha256": e009.sha256(model_file),
        "prototype_only": True,
    }
    return cfg, prior_file, review_file


def test_plateau_review_accepts_exact_pinned_prior(tmp_path):
    cfg, _, _ = _write_plateau_fixture(tmp_path, pytest.MonkeyPatch())
    prior, _ = e009._prior_from_artifact(cfg, "cpu", use_case="prototype")
    assert prior.bond_logits.numel() == 3


def test_plateau_review_rejects_changed_prior(tmp_path):
    cfg, prior_file, _ = _write_plateau_fixture(tmp_path, pytest.MonkeyPatch())
    prior_file.write_text(prior_file.read_text() + " ")
    prior_file.with_name("prior.sha256").write_text(e009.sha256(prior_file) + "  prior.json\n")
    with pytest.raises(ValueError, match="changed prior"):
        e009._prior_from_artifact(cfg, "cpu", use_case="prototype")


def test_plateau_review_rejects_failed_criterion(tmp_path):
    cfg, _, review_file = _write_plateau_fixture(tmp_path, pytest.MonkeyPatch())
    review = json.loads(review_file.read_text())
    review["criteria_pass"] = False
    review_file.write_text(json.dumps(review))
    cfg["review_sha256"] = e009.sha256(review_file)
    with pytest.raises(ValueError, match="does not authorize"):
        e009._prior_from_artifact(cfg, "cpu", use_case="prototype")


def test_plateau_review_rejects_non_normalized_density():
    assert not e009._density_is_normalized(0.9)
    assert e009._density_is_normalized(1.0001)


def test_plateau_loader_rejects_review_with_non_normalized_density(tmp_path):
    cfg, _, review_file = _write_plateau_fixture(tmp_path, pytest.MonkeyPatch())
    review = json.loads(review_file.read_text())
    review["mixture_results"]["angle"]["density_integral"] = 0.9
    review_file.write_text(json.dumps(review))
    cfg["review_sha256"] = e009.sha256(review_file)
    with pytest.raises(ValueError, match="failed a mixture criterion"):
        e009._prior_from_artifact(cfg, "cpu", use_case="prototype")


def test_unconverged_prior_requires_review(tmp_path):
    cfg, _, _ = _write_plateau_fixture(tmp_path, pytest.MonkeyPatch())
    cfg["review_artifact"] = str(tmp_path / "missing-review.json")
    with pytest.raises(ValueError, match="review is required"):
        e009._prior_from_artifact(cfg, "cpu", use_case="prototype")


def test_plateau_review_refuses_production_use(tmp_path):
    cfg, _, _ = _write_plateau_fixture(tmp_path, pytest.MonkeyPatch())
    with pytest.raises(ValueError, match="production use"):
        e009._prior_from_artifact(cfg, "cpu", use_case="production")
