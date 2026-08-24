"""Training loop for the structure-unconditioned sequence Transformer."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

from protein_sequence_generation.checkpointing import assert_resume_compatible, load_checkpoint, save_checkpoint
from protein_sequence_generation.collate import collate_sequences
from protein_sequence_generation.dataset import ProteinSequenceDataset
from protein_sequence_generation.metrics import language_model_metrics, sequence_cross_entropy
from protein_sequence_generation.model import ProteinSequenceTransformer, parameter_count
from protein_sequence_generation.utils import (
    restore_rng_state,
    rng_state,
    runtime_versions,
    seed_everything,
    sha256_file,
)
from protein_sequence_generation.vocabulary import ProteinVocabulary


class _TorchOnlyRandomSampler(Sampler[int]):
    """Uniform random sampler for environments where PyTorch RandomSampler calls Tensor.numpy()."""

    def __init__(self, length: int, *, seed: int) -> None:
        self.length = int(length)
        self.seed = int(seed)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed)
        yield from torch.randperm(self.length, generator=generator).tolist()

    def __len__(self) -> int:
        return self.length


def _torch_random_sampler_works() -> bool:
    try:
        torch.empty(0).numpy()
    except RuntimeError as exc:
        if "Numpy is not available" in str(exc):
            return False
        raise
    return True


def _summary_writer(enabled: bool, log_dir: str | Path):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise RuntimeError("TensorBoard is enabled but tensorboard is not installed") from exc
    return SummaryWriter(log_dir=str(log_dir))


def _amp_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError("amp_dtype must be bfloat16 or float16")


def _autocast(enabled: bool, dtype: torch.dtype):
    return torch.amp.autocast("cuda", dtype=dtype) if enabled else nullcontext()


def _make_scaler(enabled: bool, dtype: torch.dtype):
    return torch.amp.GradScaler("cuda", enabled=True) if enabled and dtype is torch.float16 else None


def _lr_lambda(*, warmup_steps: int, total_steps: int):
    def schedule(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return schedule


def _patch_torch_pytree_compatibility() -> None:
    """Patch older PyTorch pytree API names touched by optional imports."""
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


def _checkpoint_payload(
    *,
    model: ProteinSequenceTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: Any,
    epoch: int,
    microbatch_step: int,
    global_step: int,
    best_validation: float,
    epochs_without_improvement: int,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    vocabulary: ProteinVocabulary,
    train_manifest: str | Path,
    validation_manifest: str | Path,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "grad_scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "microbatch_step": microbatch_step,
        "global_step": global_step,
        "best_validation_metric": best_validation,
        "early_stopping": {"epochs_without_improvement": epochs_without_improvement},
        "model_config": model_config,
        "training_config": training_config,
        "vocabulary": {"tokens": list(vocabulary.tokens)},
        "manifests": {"train": str(train_manifest), "validation": str(validation_manifest)},
        "manifest_hashes": {"train": sha256_file(train_manifest), "validation": sha256_file(validation_manifest)},
        "rng_state": rng_state(),
        "runtime": runtime_versions(),
    }


@torch.no_grad()
def validate(
    model: ProteinSequenceTransformer,
    loader: DataLoader,
    *,
    device: torch.device,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Run validation without gradients."""
    model.eval()
    totals = {"token_nll_sum": 0.0, "sequence_loss_sum": 0.0, "tokens": 0, "sequences": 0, "correct": 0}
    for batch in tqdm(loader, desc="validation", dynamic_ncols=True, disable=not show_progress):
        input_ids = batch["input_ids"].to(device)
        targets = batch["target_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lengths = batch["lengths"].to(device)
        logits = model(input_ids=input_ids, lengths=lengths, attention_mask=mask)
        metrics = language_model_metrics(logits, targets, mask)
        token_count = int(metrics["valid_token_count"])
        sequence_count = int(metrics["sequence_count"])
        totals["token_nll_sum"] += float(metrics["token_nll"]) * token_count
        totals["sequence_loss_sum"] += float(metrics["sequence_loss"]) * sequence_count
        totals["tokens"] += token_count
        totals["sequences"] += sequence_count
        totals["correct"] += float(metrics["token_accuracy"]) * token_count
    token_nll = totals["token_nll_sum"] / max(totals["tokens"], 1)
    sequence_loss = totals["sequence_loss_sum"] / max(totals["sequences"], 1)
    return {
        "token_nll": token_nll,
        "perplexity": float(np.exp(token_nll)),
        "sequence_loss": sequence_loss,
        "token_accuracy": totals["correct"] / max(totals["tokens"], 1),
        "valid_token_count": int(totals["tokens"]),
        "sequence_count": int(totals["sequences"]),
    }


def train_from_config(
    config: dict[str, Any],
    *,
    resume_from: str | Path | None = None,
    max_optimizer_steps: int | None = None,
    unsafe_resume_override: bool = False,
) -> Path:
    """Train the sequence Transformer from config and return the latest checkpoint path."""
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    vocabulary = ProteinVocabulary()
    data_cfg = config["data"]
    model_cfg = dict(config["model"])
    model_cfg.setdefault("vocabulary_size", len(vocabulary.tokens))
    train_manifest = Path(data_cfg["train_manifest"])
    validation_manifest = Path(data_cfg["validation_manifest"])
    min_length = int(data_cfg.get("min_length", 20))
    max_length = int(data_cfg.get("max_length", model_cfg.get("max_length", 500)))
    train_dataset = ProteinSequenceDataset(
        train_manifest, vocabulary=vocabulary, min_length=min_length, max_length=max_length
    )
    val_dataset = ProteinSequenceDataset(
        validation_manifest, vocabulary=vocabulary, min_length=min_length, max_length=max_length
    )
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = ProteinSequenceTransformer(model_cfg).to(device)
    out = Path(config.get("output_dir", "outputs/sequence_baseline"))
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jsonl = (log_dir / "train.jsonl").open("a")
    writer = _summary_writer(bool(config.get("tensorboard", True)), config.get("tensorboard_dir", out / "tensorboard"))
    if writer is not None:
        writer.add_text("config/json", json.dumps(config, indent=2, sort_keys=True), 0)
        writer.add_text("model/parameter_count", str(parameter_count(model)), 0)
    _patch_torch_pytree_compatibility()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        betas=(float(config.get("adam_beta1", 0.9)), float(config.get("adam_beta2", 0.95))),
    )
    batch_size = int(config.get("batch_size", 8))
    accumulation = int(config.get("gradient_accumulation_steps", 8))
    epochs = int(config.get("epochs", 20))
    estimated_steps = math.ceil(len(train_dataset) / batch_size / accumulation) * epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_lr_lambda(
            warmup_steps=int(float(config.get("warmup_fraction", 0.05)) * estimated_steps),
            total_steps=estimated_steps,
        ),
    )
    amp_dtype = _amp_dtype(str(config.get("amp_dtype", "bfloat16")))
    amp_enabled = bool(config.get("mixed_precision", True)) and device.type == "cuda"
    if amp_enabled and amp_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("BF16 AMP requested but unsupported; falling back to FP16 GradScaler.")
        amp_dtype = torch.float16
    elif bool(config.get("mixed_precision", True)) and device.type != "cuda":
        print("Mixed precision requested but disabled because device is not CUDA.")
    scaler = _make_scaler(amp_enabled, amp_dtype)
    train_loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": int(config.get("num_workers", 2)),
        "collate_fn": lambda items: collate_sequences(items, vocabulary=vocabulary),
    }
    if _torch_random_sampler_works():
        train_loader_kwargs["shuffle"] = True
    else:
        train_loader_kwargs["shuffle"] = False
        train_loader_kwargs["sampler"] = _TorchOnlyRandomSampler(len(train_dataset), seed=seed)
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda items: collate_sequences(items, vocabulary=vocabulary),
    )
    checkpoint_dir = out / "checkpoints"
    last_path = checkpoint_dir / "last.pt"
    best_path = checkpoint_dir / "best_validation.pt"
    start_epoch = 0
    global_step = 0
    best_validation = float("inf")
    epochs_without_improvement = 0
    if resume_from is None:
        resume_from = config.get("resume_from")
    if resume_from:
        checkpoint = load_checkpoint(resume_from, map_location=device)
        assert_resume_compatible(
            checkpoint,
            model_config=model_cfg,
            vocabulary=vocabulary,
            train_manifest=train_manifest,
            validation_manifest=validation_manifest,
            unsafe_override=unsafe_resume_override,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler is not None and checkpoint.get("grad_scaler") is not None:
            scaler.load_state_dict(checkpoint["grad_scaler"])
        restore_rng_state(checkpoint.get("rng_state"))
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_validation = float(checkpoint.get("best_validation_metric", best_validation))
        epochs_without_improvement = int(checkpoint.get("early_stopping", {}).get("epochs_without_improvement", 0))
    grad_clip = float(config.get("grad_clip", 1.0))
    checkpoint_every = int(config.get("checkpoint_every_optimizer_steps", 2000))
    log_every = int(config.get("log_every_optimizer_steps", 50))
    patience = int(config.get("early_stopping_patience", 5))
    max_overflows = int(config.get("max_consecutive_amp_overflows", 20))
    consecutive_overflows = 0
    validation_metric = str(config.get("early_stopping_metric", "token_nll"))
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            bar = tqdm(train_loader, desc=f"sequence epoch {epoch + 1}/{epochs}", dynamic_ncols=True)
            running = 0.0
            for microbatch, batch in enumerate(bar, start=1):
                start = time.time()
                input_ids = batch["input_ids"].to(device)
                targets = batch["target_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                lengths = batch["lengths"].to(device)
                with _autocast(amp_enabled, amp_dtype):
                    logits = model(input_ids=input_ids, lengths=lengths, attention_mask=mask)
                    loss = sequence_cross_entropy(
                        logits,
                        targets,
                        mask,
                        reduction=str(config.get("loss_reduction", "sequence_mean")),
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite sequence training loss")
                if scaler is not None:
                    scaler.scale(loss / accumulation).backward()
                else:
                    (loss / accumulation).backward()
                should_step = microbatch % accumulation == 0 or microbatch == len(train_loader)
                grad_norm = float("nan")
                overflow = False
                if should_step:
                    old_scale = float(scaler.get_scale()) if scaler is not None else None
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=False)
                    grad_norm = float(norm.detach().cpu() if isinstance(norm, torch.Tensor) else norm)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                        new_scale = float(scaler.get_scale())
                        overflow = new_scale < float(old_scale)
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if overflow:
                        consecutive_overflows += 1
                        if consecutive_overflows >= max_overflows:
                            raise FloatingPointError("Maximum consecutive AMP overflows reached")
                    else:
                        consecutive_overflows = 0
                        scheduler.step()
                        global_step += 1
                loss_value = float(loss.detach().cpu())
                running += (loss_value - running) / microbatch
                tokens_per_second = int(mask.sum().detach().cpu()) / max(time.time() - start, 1e-8)
                row = {
                    "epoch": epoch,
                    "microbatch": microbatch,
                    "global_step": global_step,
                    "loss": loss_value,
                    "running_loss": running,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm": grad_norm,
                    "amp_overflow": overflow,
                    "tokens_per_second": tokens_per_second,
                }
                jsonl.write(json.dumps(row, sort_keys=True) + "\n")
                jsonl.flush()
                if writer is not None and should_step and global_step % log_every == 0:
                    writer.add_scalar("train/loss_step", loss_value, global_step)
                    writer.add_scalar("optimizer/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                    writer.add_scalar("optimizer/gradient_norm", grad_norm, global_step)
                    writer.add_scalar("performance/tokens_per_second", tokens_per_second, global_step)
                bar.set_postfix({"loss": f"{loss_value:.3e}", "step": global_step, "tok/s": f"{tokens_per_second:.0f}"})
                if should_step and global_step > 0 and checkpoint_every > 0 and global_step % checkpoint_every == 0:
                    payload = _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        microbatch_step=0,
                        global_step=global_step,
                        best_validation=best_validation,
                        epochs_without_improvement=epochs_without_improvement,
                        model_config=model_cfg,
                        training_config=config,
                        vocabulary=vocabulary,
                        train_manifest=train_manifest,
                        validation_manifest=validation_manifest,
                    )
                    save_checkpoint(checkpoint_dir / f"step_{global_step:08d}.pt", payload)
                    save_checkpoint(last_path, payload)
                if max_optimizer_steps is not None and global_step >= max_optimizer_steps:
                    payload = _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        microbatch_step=0,
                        global_step=global_step,
                        best_validation=best_validation,
                        epochs_without_improvement=epochs_without_improvement,
                        model_config=model_cfg,
                        training_config=config,
                        vocabulary=vocabulary,
                        train_manifest=train_manifest,
                        validation_manifest=validation_manifest,
                    )
                    save_checkpoint(last_path, payload)
                    return last_path
            if (epoch + 1) % int(config.get("validation_every_epochs", 1)) == 0:
                val = validate(model, val_loader, device=device, show_progress=bool(config.get("progress_bar", True)))
                metric = float(val[validation_metric])
                if writer is not None:
                    writer.add_scalar("validation/token_nll", val["token_nll"], epoch)
                    writer.add_scalar("validation/perplexity", val["perplexity"], epoch)
                    writer.add_scalar("validation/sequence_loss", val["sequence_loss"], epoch)
                if metric < best_validation:
                    best_validation = metric
                    epochs_without_improvement = 0
                    payload = _checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch + 1,
                        microbatch_step=0,
                        global_step=global_step,
                        best_validation=best_validation,
                        epochs_without_improvement=epochs_without_improvement,
                        model_config=model_cfg,
                        training_config=config,
                        vocabulary=vocabulary,
                        train_manifest=train_manifest,
                        validation_manifest=validation_manifest,
                    )
                    save_checkpoint(best_path, payload)
                else:
                    epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break
            payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                microbatch_step=0,
                global_step=global_step,
                best_validation=best_validation,
                epochs_without_improvement=epochs_without_improvement,
                model_config=model_cfg,
                training_config=config,
                vocabulary=vocabulary,
                train_manifest=train_manifest,
                validation_manifest=validation_manifest,
            )
            save_checkpoint(last_path, payload)
    except KeyboardInterrupt:
        interrupted = checkpoint_dir / "interrupted.pt"
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=locals().get("epoch", start_epoch),
            microbatch_step=0,
            global_step=global_step,
            best_validation=best_validation,
            epochs_without_improvement=epochs_without_improvement,
            model_config=model_cfg,
            training_config=config,
            vocabulary=vocabulary,
            train_manifest=train_manifest,
            validation_manifest=validation_manifest,
        )
        save_checkpoint(interrupted, payload)
        print(f"Interrupted sequence checkpoint saved to {interrupted}")
        return interrupted
    finally:
        jsonl.close()
        if writer is not None:
            writer.flush()
            writer.close()
    return last_path
