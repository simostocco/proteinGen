import os, hashlib
import torch
from ..contracts import d_path


def save_atomic(path, payload):
    p = d_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    part = p.with_name(p.name + ".part")
    with part.open("wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    h = hashlib.sha256(part.read_bytes()).hexdigest()
    os.replace(part, p)
    receipt = p.with_name(p.name + ".sha256")
    tmp = receipt.with_name(receipt.name + ".part")
    tmp.write_text(h + "\n")
    os.replace(tmp, receipt)
    return h


def load_checked(path):
    p = d_path(path)
    if hashlib.sha256(p.read_bytes()).hexdigest() != p.with_name(p.name + ".sha256").read_text().strip():
        raise ValueError("checkpoint hash mismatch")
    return torch.load(p, map_location="cpu", weights_only=True)
