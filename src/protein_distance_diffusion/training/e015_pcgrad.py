"""One-sided FP32 conflict projection; accumulation happens after projection."""

import torch


def combine_gradients(gv, gs, eps=1e-12):
    if len(gv) != len(gs):
        raise ValueError("Gradient lists differ")
    pairs = [(v.float(), s.float()) for v, s in zip(gv, gs, strict=True) if v is not None and s is not None]
    dot = sum((v * s).sum() for v, s in pairs)
    norm = sum(v.float().square().sum() for v in gv if v is not None)
    snorm = sum(s.float().square().sum() for s in gs if s is not None)
    if not isinstance(norm, torch.Tensor):
        raise ValueError("Velocity objective has no gradients")
    dot = torch.as_tensor(dot, device=norm.device)
    snorm = torch.as_tensor(snorm, device=norm.device)
    coefficient = dot.clamp_max(0) / (norm + eps)
    result = []
    projected_snorm = norm.new_zeros(())
    for v, s in zip(gv, gs, strict=True):
        if v is None and s is None:
            result.append(None)
            continue
        vf = None if v is None else v.float()
        sf = None if s is None else s.float()
        if vf is not None and sf is not None:
            sf = sf - coefficient * vf
        if sf is not None:
            projected_snorm += sf.square().sum()
        result.append(vf if sf is None else (sf if vf is None else vf + sf))
    return result, {
        "conflict": bool(dot < 0),
        "cosine": float(dot / (norm * snorm).sqrt().clamp_min(eps)),
        "v_norm": float(norm.sqrt()),
        "struct_norm": float(snorm.sqrt()),
        "projected_struct_norm": float(projected_snorm.sqrt()),
    }


def backward_objectives(vloss, sloss, parameters, accumulation_weight=1.0):
    parameters = tuple(p for p in parameters if p.requires_grad)
    gv = torch.autograd.grad(vloss.float(), parameters, retain_graph=True, allow_unused=True)
    gs = torch.autograd.grad(sloss.float(), parameters, allow_unused=True)
    gradients, telemetry = combine_gradients(gv, gs)
    for p, g in zip(parameters, gradients, strict=True):
        if g is not None:
            if not torch.isfinite(g).all():
                raise FloatingPointError("Nonfinite PCGrad gradient")
            if p.grad is None:
                p.grad = g.to(p.dtype).mul_(accumulation_weight)
            else:
                p.grad.add_(g.to(p.dtype), alpha=accumulation_weight)
    return telemetry
