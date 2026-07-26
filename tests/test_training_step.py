"""CPU training and sampling smoke test."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import protein_distance_diffusion.training.trainer as trainer_module
from protein_distance_diffusion.data.collate import make_pair_mask, make_sequence_separation
from protein_distance_diffusion.data.preprocess import ProteinSample, save_processed_sample
from protein_distance_diffusion.data.statistics import write_normalization
from protein_distance_diffusion.diffusion.gaussian import GaussianDiffusion
from protein_distance_diffusion.diffusion.sampling import sample_ddpm
from protein_distance_diffusion.diffusion.schedules import cosine_beta_schedule
from protein_distance_diffusion.models.unet import DistanceUNet
from protein_distance_diffusion.training.checkpointing import load_checkpoint, save_checkpoint
from protein_distance_diffusion.training.trainer import _optimizer_boundary_step, train_from_config


class _FakeScaler:
    def __init__(self, *, scale: float = 8.0, overflow: bool = False) -> None:
        self.scale_value = scale
        self.overflow = overflow
        self.unscale_called = False
        self.step_called = False
        self.update_called = False

    def get_scale(self) -> float:
        return self.scale_value

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer
        self.unscale_called = True
        for parameter in self._parameters:
            if parameter.grad is not None:
                parameter.grad.div_(self.scale_value)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self.step_called = True
        if not self.overflow:
            optimizer.step()

    def update(self) -> None:
        self.update_called = True
        if self.overflow:
            self.scale_value /= 2.0

    def state_dict(self) -> dict[str, float]:
        return {"scale": self.scale_value}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.scale_value = float(state["scale"])

    def attach(self, model: torch.nn.Module) -> _FakeScaler:
        self._parameters = list(model.parameters())
        return self


def _write_split(tmp_path: Path, name: str, sample_ids: list[str]) -> Path:
    rows = []
    for idx, sample_id in enumerate(sample_ids):
        n = 8
        coords = np.stack([np.arange(n), np.zeros(n) + idx, np.zeros(n)], axis=1).astype(np.float32)
        sample = ProteinSample(
            sample_id=sample_id,
            pdb_id=sample_id[:4].upper(),
            chain_id="A",
            sequence="ACDEFGHI",
            residue_ids=[str(i) for i in range(n)],
            ca_coordinates=coords,
            metadata={"experimental_method": "synthetic", "resolution_angstrom": 1.0},
        )
        rows.append(save_processed_sample(sample, tmp_path / "samples"))
    path = tmp_path / f"{name}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _tiny_training_config(tmp_path: Path, *, output_name: str = "out") -> dict:
    train_manifest = _write_split(tmp_path, "train", ["trn1", "trn2"])
    val_manifest = _write_split(tmp_path, "validation", ["val1"])
    norm = tmp_path / "normalization.json"
    write_normalization(train_manifest, norm)
    return {
        "train_manifest": str(train_manifest),
        "validation_manifest": str(val_manifest),
        "normalization_file": str(norm),
        "output_dir": str(tmp_path / output_name),
        "seed": 123,
        "device": "cpu",
        "batch_size": 1,
        "epochs": 1,
        "gradient_accumulation_steps": 1,
        "mixed_precision": False,
        "amp_dtype": "float16",
        "diffusion_steps": 4,
        "learning_rate": 1e-3,
        "checkpoint_every_steps": 1,
        "logging": {
            "progress_bar": False,
            "tensorboard": False,
            "log_every_steps": 1,
            "validation_every_epochs": 1,
            "image_every_epochs": 99,
        },
        "model": {
            "base_channels": 4,
            "channel_multipliers": [1, 2],
            "residual_blocks_per_level": 1,
            "attention_heads": 1,
            "time_embedding_dim": 16,
            "length_embedding_dim": 16,
            "max_length": 16,
        },
    }


def test_cpu_training_checkpoint_and_reproducible_sampling(tmp_path: Path) -> None:
    """A tiny CPU run trains, validates, checkpoints, reloads, and preserves metadata."""
    config = _tiny_training_config(tmp_path)
    ckpt_path = train_from_config(config)
    ckpt = load_checkpoint(ckpt_path)
    assert ckpt["global_step"] == 2
    assert ckpt["amp_overflows_total"] == 0
    assert ckpt["amp_overflows_consecutive"] == 0
    assert ckpt["next_epoch"] == 1
    assert "rng_state" in ckpt
    assert ckpt["normalization"]["mode"] == "scale"
    assert Path(tmp_path / "out" / "logs" / "train.jsonl").exists()
    assert Path(tmp_path / "out" / "checkpoints" / "last.pt").exists()
    assert Path(tmp_path / "out" / "checkpoints" / "latest.pt").exists()
    assert Path(tmp_path / "out" / "checkpoints" / "best_validation.pt").exists()
    assert Path(tmp_path / "out" / "checkpoints" / "step_00000001.pt").exists()
    assert json.loads(Path(config["normalization_file"]).read_text())["scale"] > 0
    assert isinstance(ckpt["model"], dict)
    model_cfg = dict(config["model"])
    model_cfg["channel_multipliers"] = tuple(model_cfg["channel_multipliers"])
    model = DistanceUNet(**model_cfg)
    model.load_state_dict(ckpt["ema"])
    lengths = torch.tensor([8])
    mask = make_pair_mask(lengths, 8)
    sep = make_sequence_separation(lengths, 8)
    diffusion = GaussianDiffusion(cosine_beta_schedule(4))
    sample_a = sample_ddpm(
        model,
        diffusion,
        lengths=lengths,
        pair_mask=mask,
        sequence_separation=sep,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(9),
    )
    sample_b = sample_ddpm(
        model,
        diffusion,
        lengths=lengths,
        pair_mask=mask,
        sequence_separation=sep,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(9),
    )
    assert torch.allclose(sample_a, sample_b)


def test_keyboard_interrupt_writes_interrupted_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A KeyboardInterrupt produces a resumable interrupted checkpoint."""
    config = _tiny_training_config(tmp_path, output_name="interrupted")

    def interrupt_loss(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        raise KeyboardInterrupt

    monkeypatch.setattr(trainer_module, "masked_upper_triangular_loss", interrupt_loss)
    ckpt_path = train_from_config(config)
    assert ckpt_path.name == "interrupted.pt"
    ckpt = load_checkpoint(ckpt_path)
    assert ckpt["global_step"] == 0
    assert ckpt["next_epoch"] == 0
    assert "optimizer" in ckpt
    assert "rng_state" in ckpt
    assert "grad_scaler" in ckpt


def test_resume_restores_checkpoint_state(tmp_path: Path) -> None:
    """Resuming continues from the saved epoch and global step."""
    config = _tiny_training_config(tmp_path, output_name="resume")
    first_ckpt = train_from_config(config)
    resumed = dict(config)
    resumed["epochs"] = 2
    resumed["resume_from"] = str(first_ckpt)
    second_ckpt = train_from_config(resumed)
    ckpt = load_checkpoint(second_ckpt)
    assert ckpt["epoch"] == 1
    assert ckpt["next_epoch"] == 2
    assert ckpt["global_step"] == 4


def test_v_training_checkpoint_records_parameterization(tmp_path: Path) -> None:
    """A v-prediction training run stores the canonical parameterization in checkpoints."""
    config = _tiny_training_config(tmp_path, output_name="v-run")
    config["prediction_parameterization"] = "v"
    ckpt_path = train_from_config(config)
    ckpt = load_checkpoint(ckpt_path)
    assert ckpt["config"]["prediction_parameterization"] == "v"
    assert ckpt["config"]["prediction_type"] == "v"


def test_resume_rejects_prediction_parameterization_mismatch(tmp_path: Path) -> None:
    """Resume fails clearly when a v config points at an epsilon checkpoint."""
    config = _tiny_training_config(tmp_path, output_name="epsilon-run")
    ckpt_path = train_from_config(config)
    resumed = _tiny_training_config(tmp_path, output_name="v-mismatch")
    resumed["prediction_parameterization"] = "v"
    resumed["resume_from"] = str(ckpt_path)
    with pytest.raises(ValueError, match="mismatch"):
        train_from_config(resumed)


def test_tensorboard_logging_when_available(tmp_path: Path) -> None:
    """TensorBoard event files are written when SummaryWriter is enabled."""
    pytest.importorskip("tensorboard")
    config = _tiny_training_config(tmp_path, output_name="tensorboard")
    config["logging"] = {
        "progress_bar": False,
        "tensorboard": True,
        "tensorboard_dir": str(tmp_path / "tensorboard-events"),
        "log_every_steps": 1,
        "validation_every_epochs": 1,
        "image_every_epochs": 1,
    }
    train_from_config(config)
    assert list((tmp_path / "tensorboard-events").glob("events.out.tfevents.*"))


def test_training_uses_standard_shuffled_dataloader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Training uses PyTorch DataLoader batch_size/shuffle rather than a custom batch sampler."""
    original_loader = trainer_module.DataLoader
    calls = []

    def recording_loader(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "DataLoader", recording_loader)
    train_from_config(_tiny_training_config(tmp_path, output_name="loader"))
    train_kwargs, validation_kwargs = calls[:2]
    assert train_kwargs["batch_size"] == 1
    assert train_kwargs["shuffle"] is True or train_kwargs["sampler"].__class__.__name__ == "_NumpyFreeRandomSampler"
    assert "batch_sampler" not in train_kwargs
    assert validation_kwargs["shuffle"] is False


def test_no_batch_matrix_budget_references_remain() -> None:
    """The obsolete matrix-budget sampler/config knob is removed from active source and docs."""
    root = Path(__file__).resolve().parents[1]
    checked_paths = [
        root / "configs" / "train.yaml",
        root / "configs" / "train_full.yaml",
        root / "README.md",
        root / "src" / "protein_distance_diffusion" / "training" / "trainer.py",
    ]
    for path in checked_paths:
        assert "batch_matrix_budget" not in path.read_text()
    assert not (root / "src" / "protein_distance_diffusion" / "data" / "samplers.py").exists()


def test_amp_requested_on_cpu_is_disabled_but_checkpoint_compatible(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """AMP is disabled cleanly on CPU and checkpoints still include the scaler slot."""
    config = _tiny_training_config(tmp_path, output_name="cpu-amp")
    config["mixed_precision"] = True
    ckpt_path = train_from_config(config)
    captured = capsys.readouterr()
    assert "Mixed precision requested but disabled" in captured.out
    ckpt = load_checkpoint(ckpt_path)
    assert ckpt["grad_scaler"] is None


def test_amp_dtype_validation(tmp_path: Path) -> None:
    """Invalid AMP dtype values fail before training starts."""
    config = _tiny_training_config(tmp_path, output_name="bad-amp")
    config["amp_dtype"] = "float32"
    with pytest.raises(ValueError, match="amp_dtype"):
        train_from_config(config)


def test_gradient_accumulation_steps_once_per_boundary(tmp_path: Path) -> None:
    """Two microbatches with accumulation=2 produce one optimizer/global step."""
    config = _tiny_training_config(tmp_path, output_name="accum")
    config["gradient_accumulation_steps"] = 2
    ckpt = load_checkpoint(train_from_config(config))
    assert ckpt["global_step"] == 1
    assert ckpt["optimizer_step"] == 1
    assert ckpt["microbatch_in_accumulation"] == 0
    assert ckpt["amp_overflows_total"] == 0


def test_ema_updates_only_at_accumulation_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """EMA updates once for two accumulated microbatches."""
    calls = {"count": 0}
    original_update = trainer_module.EMA.update

    def recording_update(self, model):  # type: ignore[no-untyped-def]
        calls["count"] += 1
        return original_update(self, model)

    monkeypatch.setattr(trainer_module.EMA, "update", recording_update)
    config = _tiny_training_config(tmp_path, output_name="ema-accum")
    config["gradient_accumulation_steps"] = 2
    train_from_config(config)
    assert calls["count"] == 1


def test_amp_unscales_before_gradient_clipping() -> None:
    """The AMP path unscales gradients before clipping."""
    source = inspect.getsource(trainer_module._optimizer_boundary_step)
    assert source.index("scaler.unscale_(optimizer)") < source.index("clip_grad_norm_")
    assert "error_if_nonfinite=False" in source


def test_finite_gradients_are_clipped_and_logged_preclip() -> None:
    """Full-precision gradients are clipped to threshold while reporting the pre-clipping norm."""
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ema = trainer_module.EMA(model, decay=0.9)
    parameter = next(model.parameters())
    parameter.grad = torch.tensor([[3.0, 4.0]])
    result = _optimizer_boundary_step(
        model=model,
        optimizer=optimizer,
        ema=ema,
        scaler=None,
        grad_clip=1.0,
        epoch=0,
        microbatch=1,
        global_step=0,
        loss_value=1.0,
        accumulation_window=[{"sample_ids": ["s"], "lengths": [2], "timesteps": [0]}],
        amp_overflows_total=0,
        amp_overflows_consecutive=0,
        max_consecutive_amp_overflows=20,
    )
    assert result["grad_norm_preclip"] == pytest.approx(5.0)
    assert result["global_step"] == 1
    assert torch.linalg.vector_norm(parameter.grad if parameter.grad is not None else torch.zeros(1)) == 0


def test_fp16_overflow_skips_update_ema_step_and_clears_gradients() -> None:
    """A scaler overflow reduces scale and skips optimizer/EMA/global-step updates."""
    model = torch.nn.Linear(1, 1, bias=False)
    before = next(model.parameters()).detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    ema = trainer_module.EMA(model, decay=0.9)
    ema_before = {key: value.clone() for key, value in ema.state_dict().items()}
    next(model.parameters()).grad = torch.tensor([[float("inf")]])
    scaler = _FakeScaler(scale=8.0, overflow=True).attach(model)
    result = _optimizer_boundary_step(
        model=model,
        optimizer=optimizer,
        ema=ema,
        scaler=scaler,
        grad_clip=1.0,
        epoch=0,
        microbatch=1,
        global_step=0,
        loss_value=1.0,
        accumulation_window=[{"sample_ids": ["s"], "lengths": [2], "timesteps": [0]}],
        amp_overflows_total=0,
        amp_overflows_consecutive=0,
        max_consecutive_amp_overflows=20,
    )
    assert result["update_skipped"] is True
    assert result["global_step"] == 0
    assert result["amp_overflows_total"] == 1
    assert result["amp_overflows_consecutive"] == 1
    assert result["amp_scale_old"] == 8.0
    assert result["amp_scale_new"] == 4.0
    assert torch.equal(next(model.parameters()).detach(), before)
    assert all(torch.equal(value, ema_before[key]) for key, value in ema.state_dict().items())
    assert next(model.parameters()).grad is None


def test_successful_fp16_update_resets_overflow_counter() -> None:
    """A successful scaler update advances the optimizer step and resets consecutive overflows."""
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ema = trainer_module.EMA(model, decay=0.9)
    next(model.parameters()).grad = torch.tensor([[8.0]])
    scaler = _FakeScaler(scale=8.0, overflow=False).attach(model)
    result = _optimizer_boundary_step(
        model=model,
        optimizer=optimizer,
        ema=ema,
        scaler=scaler,
        grad_clip=1.0,
        epoch=0,
        microbatch=1,
        global_step=3,
        loss_value=1.0,
        accumulation_window=[{"sample_ids": ["s"], "lengths": [2], "timesteps": [0]}],
        amp_overflows_total=5,
        amp_overflows_consecutive=2,
        max_consecutive_amp_overflows=20,
    )
    assert result["update_skipped"] is False
    assert result["global_step"] == 4
    assert result["amp_overflows_total"] == 5
    assert result["amp_overflows_consecutive"] == 0
    assert result["grad_norm_preclip"] == pytest.approx(1.0)


def test_scaler_and_overflow_counters_restore_from_checkpoint(tmp_path: Path) -> None:
    """Resume restores GradScaler state and AMP overflow counters."""
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ema = trainer_module.EMA(model, decay=0.9)
    scaler = _FakeScaler(scale=16.0).attach(model)
    norm = tmp_path / "normalization.json"
    norm.write_text(json.dumps({"mode": "scale", "scale": 1.0}))
    config = {"normalization_file": str(norm)}
    payload = trainer_module._checkpoint_payload(
        model=model,
        ema=ema,
        optimizer=optimizer,
        scaler=scaler,
        epoch=2,
        next_epoch=3,
        global_step=17,
        amp_overflows_total=5,
        amp_overflows_consecutive=2,
        config=config,
        normalization={"mode": "scale", "scale": 1.0},
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, payload)

    restored_model = torch.nn.Linear(1, 1, bias=False)
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1)
    restored_ema = trainer_module.EMA(restored_model, decay=0.9)
    restored_scaler = _FakeScaler(scale=1.0).attach(restored_model)
    start_epoch, global_step, total, consecutive = trainer_module._restore_training_state(
        resume_from=path,
        model=restored_model,
        ema=restored_ema,
        optimizer=restored_optimizer,
        scaler=restored_scaler,
        device=torch.device("cpu"),
    )
    assert (start_epoch, global_step, total, consecutive) == (3, 17, 5, 2)
    assert restored_scaler.get_scale() == 16.0


def test_repeated_fp16_overflow_reaches_threshold() -> None:
    """Repeated scaler overflows eventually fail with sample diagnostics."""
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ema = trainer_module.EMA(model, decay=0.9)
    next(model.parameters()).grad = torch.tensor([[float("inf")]])
    scaler = _FakeScaler(scale=8.0, overflow=True).attach(model)
    with pytest.raises(FloatingPointError, match="sample_ids=\\['s'\\]"):
        _optimizer_boundary_step(
            model=model,
            optimizer=optimizer,
            ema=ema,
            scaler=scaler,
            grad_clip=1.0,
            epoch=0,
            microbatch=1,
            global_step=0,
            loss_value=1.0,
            accumulation_window=[{"sample_ids": ["s"], "lengths": [2], "timesteps": [0]}],
            amp_overflows_total=1,
            amp_overflows_consecutive=1,
            max_consecutive_amp_overflows=2,
        )


def test_non_amp_nonfinite_gradient_is_fatal() -> None:
    """BF16/full-precision paths without GradScaler fail immediately on non-finite gradients."""
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ema = trainer_module.EMA(model, decay=0.9)
    next(model.parameters()).grad = torch.tensor([[float("inf")]])
    with pytest.raises(FloatingPointError, match="Non-finite gradient norm"):
        _optimizer_boundary_step(
            model=model,
            optimizer=optimizer,
            ema=ema,
            scaler=None,
            grad_clip=1.0,
            epoch=0,
            microbatch=1,
            global_step=0,
            loss_value=1.0,
            accumulation_window=[{"sample_ids": ["s"], "lengths": [2], "timesteps": [0]}],
            amp_overflows_total=0,
            amp_overflows_consecutive=0,
            max_consecutive_amp_overflows=20,
        )


def test_max_optimizer_steps_exits_cleanly_with_checkpoint(tmp_path: Path) -> None:
    """Preflight mode stops after the requested optimizer step count and saves last.pt."""
    config = _tiny_training_config(tmp_path, output_name="preflight")
    config["epochs"] = 5
    config["max_optimizer_steps"] = 1
    ckpt_path = train_from_config(config)
    ckpt = load_checkpoint(ckpt_path)
    assert ckpt_path.name == "last.pt"
    assert ckpt["global_step"] == 1
    assert Path(tmp_path / "preflight" / "checkpoints" / "latest.pt").exists()


def test_nonfinite_loss_stops_without_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-finite losses raise a clear diagnostic before checkpointing."""
    config = _tiny_training_config(tmp_path, output_name="nan")

    def nan_loss(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        return torch.tensor(float("nan"), requires_grad=True)

    monkeypatch.setattr(trainer_module, "masked_upper_triangular_loss", nan_loss)
    with pytest.raises(FloatingPointError, match="Non-finite training loss"):
        train_from_config(config)
    assert not Path(tmp_path / "nan" / "checkpoints" / "last.pt").exists()


def test_optional_cuda_amp_smoke(tmp_path: Path) -> None:
    """Optional CUDA smoke test for AMP; skipped on CPU-only machines."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    config = _tiny_training_config(tmp_path, output_name="cuda-amp")
    config["device"] = "cuda"
    config["mixed_precision"] = True
    config["amp_dtype"] = "float16"
    config["max_optimizer_steps"] = 1
    ckpt = load_checkpoint(train_from_config(config))
    assert ckpt["grad_scaler"] is not None
    assert ckpt["global_step"] == 1
