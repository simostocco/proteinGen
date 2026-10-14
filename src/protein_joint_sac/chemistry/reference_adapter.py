"""Hash-bound reuse of the original fourteen approved chemistry definitions."""

import hashlib, importlib.util, json
from pathlib import Path
import numpy as np
from ..contracts import ChemistryRegistryRef, BuildState

ROOT = Path("/mnt/d/Simone/proteinGen_data/e016_chemical_features_inference_safe_v1_20261014")
EXPECTED_REGISTRY = "70214400325dc2872635ef9b744bcb7656b834b6a7488b2a2d401dd44793254e"
EXPECTED_IMPLEMENTATION = "fa6029ec1f6593204fefffe2d47619a8cb5683f542b0869b0f5b0478e3c34519"
_reference = None


def load():
    global _reference
    if _reference is not None:
        return _reference
    for p, h in [(ROOT / "feature_registry.json", EXPECTED_REGISTRY), (ROOT / "features.py", EXPECTED_IMPLEMENTATION)]:
        if hashlib.sha256(p.read_bytes()).hexdigest() != h:
            raise ValueError("chemistry source hash mismatch")
    frozen = json.loads((ROOT / "REGISTRY_FROZEN.json").read_text())
    registry = json.loads((ROOT / "feature_registry.json").read_text())
    names = tuple(frozen["allowed_online_reference_features"])
    if len(names) != 14 or set(names) != {
        r["name"]
        for r in registry
        if r["actor_allowed_in_active_CA_profile"]
        and r["implementation_status"] in ["IMPLEMENTED", "IMPLEMENTED_REFERENCE"]
    }:
        raise ValueError("allowlist mismatch")
    spec = importlib.util.spec_from_file_location("e016_chemistry_immutable_reference", ROOT / "features.py")
    module = importlib.util.module_from_spec(spec)
    import sys, os

    # Preserve new layer's D-only environment; historical module sets its own paths on import.
    before = os.environ.copy()
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    os.environ.clear()
    os.environ.update(before)
    module.APPROVED_NAMES = names
    _reference = module
    return module


def ref():
    return ChemistryRegistryRef(str(ROOT), EXPECTED_REGISTRY, EXPECTED_IMPLEMENTATION)


def reference_features(state, mode="INFERENCE"):
    if type(state) != BuildState:
        raise ValueError("closed BuildState only")
    m = load()
    coords = tuple(tuple(x.tolist()) if valid else None for x, valid in zip(state.ca.detach().cpu(), state.valid.cpu()))
    p = m.PrefixState(state.residues, coords, state.N, state.t)
    return m.actor_preprocess(p, mode=mode, plan=None)


def chemistry_table():
    m = load()
    return np.asarray([m.LUT[a] for a in m.AA], dtype=np.float64)
