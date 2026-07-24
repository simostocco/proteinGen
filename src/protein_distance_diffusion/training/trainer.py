"""Training loop for conditional distance-map diffusion."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from protein_distance_diffusion.data.collate import collate_distance_maps
from protein_distance_diffusion.data.dataset import DistanceMapDataset
from protein_distance_diffusion.data.samplers import LengthBucketBatchSampler
from protein_distance_diffusion.diffusion.gaussian import GaussianDiffusion, masked_upper_triangular_loss
from protein_distance_diffusion.diffusion.schedules import cosine_beta_schedule
from protein_distance_diffusion.models.unet import DistanceUNet
from protein_distance_diffusion.training.checkpointing import save_checkpoint
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
    train_sampler = LengthBucketBatchSampler(
        train_ds.frame["length"].astype(int).tolist(),
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
    global_step = 0
    latest = out / "checkpoints" / "latest.pt"
    epochs = int(config.get("epochs", 1))
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            start = time.time()
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
            logger.log(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "training_loss": float(loss.detach().cpu()),
                    "learning_rate": opt.param_groups[0]["lr"],
                    "gradient_norm": float(grad_norm),
                    "average_N": float(lengths.float().mean().cpu()),
                    "wall_clock_seconds": time.time() - start,
                }
            )
        val_loss = validate_loss(model, diffusion, val_loader, device=device, seed=seed)
        logger.log({"epoch": epoch, "global_step": global_step, "validation_loss": val_loss})
        payload = {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": config,
            "normalization": normalization,
            "runtime": collect_runtime_versions().__dict__,
        }
        save_checkpoint(latest, payload)
    return latest


@torch.no_grad()
def validate_loss(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    *,
    device: torch.device,
    seed: int,
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
    for batch in loader:
        clean = batch["distance_matrices"].to(device)
        pair_mask = batch["pair_masks"].to(device)
        lengths = batch["lengths"].to(device)
        sep = batch["sequence_separation"].to(device)
        t = torch.randint(0, diffusion.timesteps, (clean.shape[0],), device=device, generator=gen)
        noisy, eps = diffusion.q_sample(clean, t, pair_mask, generator=gen)
        eps_hat = model(noisy, t, lengths, sep, pair_mask)
        losses.append(masked_upper_triangular_loss(eps, eps_hat, pair_mask).cpu())
    return float(torch.stack(losses).mean()) if losses else float("nan")
