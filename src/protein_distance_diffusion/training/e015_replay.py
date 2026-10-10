"""Strict scientific CUDA replay fingerprints; performance timers are excluded."""

import hashlib

import numpy as np
import torch

from protein_distance_diffusion.training.e014_runtime import rng_state


def fingerprint(value):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
            packed = torch.empty(tensor.numel(), dtype=tensor.dtype).copy_(tensor.reshape(-1))
            digest.update(packed.view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(f"array:{item.dtype}:{item.shape}:".encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict:")
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(f"{type(item).__name__}:{len(item)}:".encode())
            for child in item:
                visit(child)
        else:
            digest.update(f"{type(item).__name__}:{repr(item)};".encode())

    visit(value)
    return digest.hexdigest()


def replay_snapshot(engine, observations, gradients):
    return dict(
        observations=fingerprint(observations),
        gradient_telemetry=fingerprint(gradients),
        inference_outputs_and_losses=engine.replay_microbatches.copy(),
        combined_gradient=engine.replay_combined_gradient,
        clipped_gradient=engine.replay_clipped_gradient,
        model=fingerprint(engine.model.state_dict()),
        ema=fingerprint(engine.ema.state_dict()),
        optimizer=fingerprint(engine.optimizer.state_dict()),
        scaler=fingerprint(engine.scaler.state_dict()),
        engine_rng=fingerprint(engine.rng_state()),
        global_rng=fingerprint(rng_state()),
    )
