"""Legacy diagnostic names route to the authoritative inference-gauge prefix path."""
import torch
from ..contracts import BuildState
from ..chemistry.reference_adapter import load
from .certified_unified import canonical_teacher_prefix, canonical_native_prefix


def _coordinates(observation, t):
    m = load()
    prefix = m.experimental_prefix_for_BC(observation, t=t, requested_N=len(observation["canonical_full_sequence"]))
    if any(x is None for x in prefix.ca):
        raise ValueError("missing full-polymer prefix CA; fragments cannot become natural-start chains")
    return prefix.ca


def diagnostic_prefix(observation, t, source_sha256):
    # Retain masked reference-feature diagnostics; never use this branch as a joint BC target.
    m=load(); seq=observation["canonical_full_sequence"]
    prefix=m.experimental_prefix_for_BC(observation,t=t,requested_N=len(seq))
    if any(x is None for x in prefix.ca):
        ca=torch.tensor([x if x is not None else (0.,0.,0.) for x in prefix.ca],dtype=torch.float64).reshape(t,3)
        return BuildState(prefix.residues,ca,len(seq),torch.tensor([x is not None for x in prefix.ca],dtype=torch.bool))
    return canonical_native_prefix(seq, prefix.ca, t)


def diagnostic_target(observation, t, source_sha256):
    return canonical_teacher_prefix(observation["canonical_full_sequence"], _coordinates(observation, t+1), t, source_sha256)
