"""Matched full-data exposure and execution invariants."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

if not Path(__file__).resolve().is_relative_to("/mnt/d"):
    pytest.skip("D-resident scientific control only", allow_module_level=True)

import yaml
from e007_fullscale_runtime import CONFIG, MODEL_CONFIG, RUN, expected_exposures

from protein_distance_diffusion.models.e014_e007_control import E007ScaleupControl
from protein_distance_diffusion.training.e014_runtime import epoch_schedule


def test_resume_cursor_requires_exact_once_epoch():
    lengths = [20, 64, 65, 128, 129, 256, 384, 500]
    schedule, counts = expected_exposures(lengths, 2, 3, [4, 2, 1, 1, 1], 14014)
    assert schedule == epoch_schedule(lengths, [4, 2, 1, 1, 1], 14014, 2)
    assert sorted(i for batch in schedule for i in batch) == list(range(len(lengths)))
    seen = {i for batch in schedule[:3] for i in batch}
    assert all(counts[i] == (2 if i in seen else 1) for i in range(len(lengths)))
    assert np.all(expected_exposures(lengths, 2, len(schedule), [4, 2, 1, 1, 1], 14014)[1] == 2)


def test_historical_representation_and_frozen_optimizer():
    config = yaml.safe_load(MODEL_CONFIG.read_text())
    model = E007ScaleupControl(**config["model"])
    assert sum(p.numel() for p in model.parameters()) == 7586505
    assert model.core.rbf_bins == 16
    assert not hasattr(model, "local_head")
    shared = yaml.safe_load(CONFIG.read_text())
    assert shared["optimizer"] == dict(family="AdamW", learning_rate=0.0002, weight_decay=0.01, gradient_clip=1.0)
    assert shared["batch_sizes"] == [4, 2, 1, 1, 1]


def test_memory_profile_is_actual_optimizer_lifecycle():
    profile = json.loads((RUN / "compute_profile.json").read_text())
    assert profile["passed"] and profile["optimizer_included"]
    assert [r["N"] for r in profile["results"]] == [64, 128, 256, 500]
    assert all(
        r["optimizer_states_initialized"] > 0 and r["safety_headroom_bytes"] >= 512 * 2**20 for r in profile["results"]
    )


def test_runtime_storage_and_allocator():
    for key in ("TMPDIR", "TEMP", "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "MPLCONFIGDIR"):
        assert Path(os.environ[key]).resolve().is_relative_to(RUN)
    assert os.environ["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"


def test_invalid_resume_cursor_refused():
    with pytest.raises(AssertionError):
        expected_exposures([64, 500], 2, 99, [4, 2, 1, 1, 1], 14014)
