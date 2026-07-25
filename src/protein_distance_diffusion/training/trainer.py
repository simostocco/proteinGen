"""Training loop for conditional distance-map diffusion."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from protein_distance_diffusion.data.collate import collate_distance_maps
from protein_distance_diffusion.data.dataset import DistanceMapDataset
from protein_distance_diffusion.data.samplers import LengthBucketBatchSampler, MatrixBudgetBatchSampler
from protein_distance_diffusion.diffusion.gaussian import GaussianDiffusion, masked_upper_triangular_loss
from protein_distance_diffusion.diffusion.sampling import sample_ddpm
from protein_distance_diffusion.diffusion.schedules import cosine_beta_schedule
from protein_distance_diffusion.models.unet import DistanceUNet
from protein_distance_diffusion.training.checkpointing import load_checkpoint, save_checkpoint
from protein_distance_diffusion.training.ema import EMA
from protein_distance_diffusion.training.logging import JsonlLogger
from protein_distance_diffusion.utils.reproducibility import collect_runtime_versions, seed_everything


def _patch_torch_pytree_compatibility() -> None:
    """Patch older PyTorch pytree API names used by optional imports.

    Args:
        None.

    Returns:
        None.

    Notes:
        Some environments import `transformers` through Torch Dynamo while constructing
        optimizers. Older PyTorch exposes `_register_pytree_node` but not
        `register_pytree_node`; this shim keeps AdamW usable without making
        Transformers a project dependency.
    """
    try:
        import torch.utils._pytree as pytree
    except Exception:
        return
    if hasattr(pytree, "register_pytree_node") or not hasattr(pytree, "_register_pytree_node"):
        return

    def register_pytree_node(node_type, flatten_fn, unflatten_fn, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        return pytree._register_pytree_node(node_type, flatten_fn, unflatten_fn)

    pytree.register_pytree_node = register_pytree_node


def build_model_from_config(config: dict[str, Any]) -> DistanceUNet:
    """Build the U-Net from a configuration mapping.

    Args:
        config: Model config.

    Returns:
        DistanceUNet instance.
    """
    cfg = dict(config)
    if "channel_multipliers" in cfg:
        cfg["channel_multipliers"] = tuple(cfg["channel_multipliers"])
    return DistanceUNet(**cfg)


def _summary_writer(log_config: dict[str, Any]):
    if not bool(log_config.get("tensorboard", False)):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        msg = "TensorBoard logging is enabled, but the 'tensorboard' package is not installed."
        raise RuntimeError(msg) from exc
    return SummaryWriter(log_dir=str(log_config["tensorboard_dir"]))


def _gpu_memory_gib(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0, 0.0
    idx = device.index if device.index is not None else torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(idx) / 1024**3
    reserved = torch.cuda.memory_reserved(idx) / 1024**3
    return allocated, reserved


def _gpu_name(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return "none"
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_name(idx)


def _model_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    next_epoch: int,
    global_step: int,
    config: dict[str, Any],
    normalization: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "next_epoch": next_epoch,
        "global_step": global_step,
        "rng_state": _rng_state(),
        "config": config,
        "normalization": normalization,
        "runtime": collect_runtime_versions().__dict__,
    }


def _log_run_metadata(
    writer: Any,
    *,
    config: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    if writer is None:
        return
    writer.add_text("config/yaml", f"```yaml\n{yaml.safe_dump(config, sort_keys=True)}```", 0)
    writer.add_text("model/parameter_count", str(_model_parameter_count(model)), 0)
    writer.add_text("runtime/pytorch_version", torch.__version__, 0)
    writer.add_text("runtime/cuda_version", str(torch.version.cuda), 0)
    writer.add_text("runtime/gpu_name", _gpu_name(device), 0)


def _log_tensorboard_images(
    writer: Any,
    *,
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    batch: dict[str, torch.Tensor] | None,
    device: torch.device,
    epoch: int,
) -> None:
    if writer is None or batch is None:
        return
    was_training = model.training
    model.eval()
    lengths = batch["lengths"][:1].to(device)
    pair_mask = batch["pair_masks"][:1].to(device)
    sep = batch["sequence_separation"][:1].to(device)
    generated = sample_ddpm(
        model,
        diffusion,
        lengths=lengths,
        pair_mask=pair_mask,
        sequence_separation=sep,
        device=device,
        generator=torch.Generator(device=device).manual_seed(epoch),
    )[0]
    image = generated.detach().cpu()
    image = image - image.min()
    denom = image.max().clamp_min(1e-8)
    writer.add_image("samples/generated_distance_matrix", np.array((image / denom).tolist(), dtype=np.float32), epoch)
    if was_training:
        model.train()


def _restore_training_state(
    *,
    resume_from: str | Path | None,
    model: torch.nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int]:
    if resume_from is None:
        return 0, 0
    checkpoint = load_checkpoint(resume_from, map_location=device)
    model.load_state_dict(checkpoint["model"])
    ema.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    _restore_rng_state(checkpoint.get("rng_state"))
    return int(checkpoint.get("next_epoch", int(checkpoint["epoch"]) + 1)), int(checkpoint["global_step"])


def train_from_config(config: dict[str, Any]) -> Path:
    """Run training from a resolved configuration.

    Args:
        config: Training configuration.

    Returns:
        Path to the latest checkpoint.
    """
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    normalization = json.loads(Path(config["normalization_file"]).read_text())
    train_ds = DistanceMapDataset(config["train_manifest"], normalization)
    val_ds = DistanceMapDataset(config["validation_manifest"], normalization)
    downsample_stages = len(config["model"].get("channel_multipliers", [1, 2, 4, 8])) - 1
    train_lengths = train_ds.frame["length"].astype(int).tolist()
    if config.get("batch_matrix_budget") is not None:
        train_sampler = MatrixBudgetBatchSampler(
            train_lengths,
            batch_matrix_budget=int(config["batch_matrix_budget"]),
            shuffle=bool(config.get("shuffle", True)),
            seed=seed,
        )
    else:
        train_sampler = LengthBucketBatchSampler(
            train_lengths,
            batch_size=int(config.get("batch_size", 2)),
            boundaries=list(config.get("length_boundaries", [64, 96, 128])),
            shuffle=bool(config.get("shuffle", True)),
            seed=seed,
        )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=int(config.get("num_workers", 0)),
        collate_fn=lambda x: collate_distance_maps(x, downsample_stages=downsample_stages),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config.get("batch_size", 2)),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: collate_distance_maps(x, downsample_stages=downsample_stages),
    )
    model = build_model_from_config(config["model"]).to(device)
    diffusion = GaussianDiffusion(cosine_beta_schedule(int(config.get("diffusion_steps", 100)))).to(device)
    _patch_torch_pytree_compatibility()
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-4)))
    ema = EMA(model, decay=float(config.get("ema_decay", 0.999)))
    out = Path(config.get("output_dir", "outputs/experiment"))
    logger = JsonlLogger(out / "logs" / "train.jsonl")
    log_config = dict(config.get("logging", {}))
    log_config.setdefault("progress_bar", False)
    log_config.setdefault("tensorboard", False)
    log_config.setdefault("tensorboard_dir", str(out / "tensorboard"))
    log_config.setdefault("log_every_steps", 10)
    log_config.setdefault("validation_every_epochs", 1)
    log_config.setdefault("image_every_epochs", 5)
    writer = _summary_writer(log_config)
    _log_run_metadata(writer, config=config, model=model, device=device)
    start_epoch, global_step = _restore_training_state(
        resume_from=config.get("resume_from"),
        model=model,
        ema=ema,
        optimizer=opt,
        device=device,
    )
    checkpoint_dir = out / "checkpoints"
    last = checkpoint_dir / "last.pt"
    latest = checkpoint_dir / "latest.pt"
    best = checkpoint_dir / "best_validation.pt"
    epochs = int(config.get("epochs", 1))
    best_validation = float("inf")
    checkpoint_every_steps = int(config.get("checkpoint_every_steps", 0))
    log_every_steps = max(1, int(log_config["log_every_steps"]))
    validate_every_epochs = max(1, int(log_config["validation_every_epochs"]))
    image_every_epochs = max(1, int(log_config["image_every_epochs"]))
    progress_enabled = bool(log_config["progress_bar"])
    last_batch: dict[str, torch.Tensor] | None = None
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_loss = 0.0
            epoch_samples = 0
            running_loss = 0.0
            epoch_bar = tqdm(
                train_loader,
                desc=f"epoch {epoch + 1}/{epochs}",
                disable=not progress_enabled,
                dynamic_ncols=True,
                mininterval=0.5,
            )
            for batch_index, batch in enumerate(epoch_bar, start=1):
                start = time.time()
                last_batch = batch
                clean = batch["distance_matrices"].to(device)
                pair_mask = batch["pair_masks"].to(device)
                lengths = batch["lengths"].to(device)
                sep = batch["sequence_separation"].to(device)
                t = torch.randint(0, diffusion.timesteps, (clean.shape[0],), device=device)
                noisy, eps = diffusion.q_sample(clean, t, pair_mask)
                eps_hat = model(noisy, t, lengths, sep, pair_mask)
                loss = masked_upper_triangular_loss(eps, eps_hat, pair_mask)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
                opt.step()
                ema.update(model)
                global_step += 1
                batch_size = int(clean.shape[0])
                loss_value = float(loss.detach().cpu())
                running_loss += (loss_value - running_loss) / batch_index
                epoch_loss += loss_value * batch_size
                epoch_samples += batch_size
                elapsed = time.time() - start
                samples_per_second = batch_size / max(elapsed, 1e-8)
                allocated_gib, reserved_gib = _gpu_memory_gib(device)
                lr = float(opt.param_groups[0]["lr"])
                min_length = int(lengths.min().detach().cpu())
                max_length = int(lengths.max().detach().cpu())
                metrics = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "training_loss": loss_value,
                    "running_loss": running_loss,
                    "learning_rate": lr,
                    "gradient_norm": float(grad_norm),
                    "average_N": float(lengths.float().mean().cpu()),
                    "min_length": min_length,
                    "max_length": max_length,
                    "samples_per_second": samples_per_second,
                    "gpu_allocated_gib": allocated_gib,
                    "gpu_reserved_gib": reserved_gib,
                    "wall_clock_seconds": elapsed,
                }
                logger.log(metrics)
                if writer is not None and global_step % log_every_steps == 0:
                    writer.add_scalar("train/loss_step", loss_value, global_step)
                    writer.add_scalar("optimizer/learning_rate", lr, global_step)
                    writer.add_scalar("optimizer/gradient_norm", float(grad_norm), global_step)
                    writer.add_scalar("performance/samples_per_second", samples_per_second, global_step)
                    writer.add_scalar("gpu/allocated_gib", allocated_gib, global_step)
                    writer.add_scalar("gpu/reserved_gib", reserved_gib, global_step)
                    writer.add_scalar("data/min_length", min_length, global_step)
                    writer.add_scalar("data/max_length", max_length, global_step)
                epoch_bar.set_postfix(
                    {
                        "step": global_step,
                        "loss": f"{loss_value:.3e}",
                        "avg": f"{running_loss:.3e}",
                        "lr": f"{lr:.2e}",
                        "gpu": f"{allocated_gib:.2f}GiB",
                    }
                )
                if checkpoint_every_steps > 0 and global_step % checkpoint_every_steps == 0:
                    payload = _checkpoint_payload(
                        model=model,
                        ema=ema,
                        optimizer=opt,
                        epoch=epoch,
                        next_epoch=epoch,
                        global_step=global_step,
                        config=config,
                        normalization=normalization,
                    )
                    save_checkpoint(checkpoint_dir / f"step_{global_step:08d}.pt", payload)
                    save_checkpoint(last, payload)
                    save_checkpoint(latest, payload)
            mean_epoch_loss = epoch_loss / max(epoch_samples, 1)
            logger.log({"epoch": epoch, "global_step": global_step, "training_epoch_loss": mean_epoch_loss})
            if writer is not None:
                writer.add_scalar("train/loss_epoch", mean_epoch_loss, epoch)
            val_loss = float("nan")
            if (epoch + 1) % validate_every_epochs == 0:
                val_loss = validate_loss(
                    model,
                    diffusion,
                    val_loader,
                    device=device,
                    seed=seed,
                    progress_bar=progress_enabled,
                    epoch=epoch,
                )
                logger.log({"epoch": epoch, "global_step": global_step, "validation_loss": val_loss})
                if writer is not None:
                    writer.add_scalar("validation/loss", val_loss, epoch)
                if val_loss < best_validation:
                    best_validation = val_loss
                    payload = _checkpoint_payload(
                        model=model,
                        ema=ema,
                        optimizer=opt,
                        epoch=epoch,
                        next_epoch=epoch + 1,
                        global_step=global_step,
                        config=config,
                        normalization=normalization,
                    )
                    save_checkpoint(best, payload)
            if (epoch + 1) % image_every_epochs == 0:
                _log_tensorboard_images(
                    writer,
                    model=model,
                    diffusion=diffusion,
                    batch=last_batch,
                    device=device,
                    epoch=epoch,
                )
            payload = _checkpoint_payload(
                model=model,
                ema=ema,
                optimizer=opt,
                epoch=epoch,
                next_epoch=epoch + 1,
                global_step=global_step,
                config=config,
                normalization=normalization,
            )
            save_checkpoint(last, payload)
            save_checkpoint(latest, payload)
    except KeyboardInterrupt:
        interrupted = checkpoint_dir / "interrupted.pt"
        payload = _checkpoint_payload(
            model=model,
            ema=ema,
            optimizer=opt,
            epoch=max(start_epoch, min(epochs - 1, locals().get("epoch", start_epoch))),
            next_epoch=max(start_epoch, min(epochs - 1, locals().get("epoch", start_epoch))),
            global_step=global_step,
            config=config,
            normalization=normalization,
        )
        save_checkpoint(interrupted, payload)
        print(f"Interrupted training checkpoint saved to {interrupted}")
        return interrupted
    finally:
        if hasattr(logger, "close"):
            logger.close()
        if writer is not None:
            writer.flush()
            writer.close()
    return last


@torch.no_grad()
def validate_loss(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    *,
    device: torch.device,
    seed: int,
    progress_bar: bool = False,
    epoch: int | None = None,
) -> float:
    """Compute deterministic validation diffusion loss.

    Args:
        model: Epsilon-prediction model.
        diffusion: Diffusion process.
        loader: Validation DataLoader.
        device: Target device.
        seed: Fixed validation seed.

    Returns:
        Mean validation loss.
    """
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    losses = []
    desc = "validation" if epoch is None else f"validation {epoch + 1}"
    iterator = tqdm(loader, desc=desc, disable=not progress_bar, dynamic_ncols=True, mininterval=0.5)
    for batch in iterator:
        clean = batch["distance_matrices"].to(device)
        pair_mask = batch["pair_masks"].to(device)
        lengths = batch["lengths"].to(device)
        sep = batch["sequence_separation"].to(device)
        t = torch.randint(0, diffusion.timesteps, (clean.shape[0],), device=device, generator=gen)
        noisy, eps = diffusion.q_sample(clean, t, pair_mask, generator=gen)
        eps_hat = model(noisy, t, lengths, sep, pair_mask)
        loss = masked_upper_triangular_loss(eps, eps_hat, pair_mask).cpu()
        losses.append(loss)
        iterator.set_postfix({"loss": f"{float(loss):.3e}"})
    return float(torch.stack(losses).mean()) if losses else float("nan")
