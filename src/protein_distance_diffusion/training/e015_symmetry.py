"""Symmetry checks use FP32 geometric transforms outside model autocast."""

import contextlib

import torch


def error(actual, expected):
    actual, expected = actual.float(), expected.float()
    absolute = float((actual - expected).norm())
    return dict(
        absolute_error=absolute,
        relative_error=absolute / max(float(expected.norm()), 1e-12),
        reference_norm=float(expected.norm()),
    )


def autocast(precision, device):
    return (
        contextlib.nullcontext()
        if precision == "fp32"
        else torch.autocast(device, dtype=torch.bfloat16 if precision == "bf16" else torch.float16)
    )


def symmetry(model, x, t, lengths, mask, previous, available, precision, seed=7715):
    generator = torch.Generator().manual_seed(seed)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    rotation[:, 0] *= torch.det(rotation)
    rotation = rotation.to(x.device)
    rotated = x.float() @ rotation.T + 3
    rotated_previous = None if previous is None else previous.float() @ rotation.T - 4
    translated = x.float() + 3
    translated_previous = None if previous is None else previous.float() - 4
    was_training = model.training
    model.eval()
    with torch.no_grad(), autocast(precision, x.device.type):
        original, components = model.with_components(x, t, lengths, mask, previous, available)
        changed = model(rotated, t, lengths, mask, rotated_previous, available)
        shifted = model(translated, t, lengths, mask, translated_previous, available)
    model.train(was_training)
    return dict(
        rotation=error(changed, original.float() @ rotation.T),
        translation=error(shifted, original),
        branch_norms={
            key: float(components[0][key].norm()) for key in ["global_v", "chiral_effective", "local_effective"]
        },
        output_norm=float(original.norm()),
        input_transform_precision="fp32",
        precision=precision,
        seed=seed,
    )


def require_symmetry(result, precision):
    tolerance = 5e-5 if precision == "fp32" else 0.03
    for key in ("rotation", "translation"):
        if result[key]["relative_error"] > tolerance:
            raise RuntimeError(f"{precision} {key} error {result[key]['relative_error']}>{tolerance}")
    return tolerance
