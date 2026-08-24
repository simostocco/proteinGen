"""MDAnalysis-backed legacy PDB/ENT parser."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from protein_distance_diffusion.constants import DEFAULT_RESIDUE_MAPPINGS, STANDARD_AA3_TO_1
from protein_distance_diffusion.data.preprocess import ProteinSample


class StructureParseError(RuntimeError):
    """Raised when a legacy PDB file cannot be converted to samples."""


def parse_pdb_file(
    path: str | Path,
    *,
    min_length: int | None = 20,
    max_length: int = 500,
    chain_id: str | None = None,
    residue_mappings: dict[str, str] | None = None,
    **_: Any,
) -> list[ProteinSample]:
    """Parse a PDB/ENT file through MDAnalysis and extract one C-alpha per residue."""
    if min_length is not None and min_length <= 0:
        raise ValueError("min_length must be a positive integer or None")
    try:
        import MDAnalysis as mda
    except ModuleNotFoundError as exc:
        raise StructureParseError("MDAnalysis is required to parse PDB files") from exc

    src = Path(path)
    try:
        universe = mda.Universe(str(src))
    except Exception as exc:
        raise StructureParseError(f"Failed to parse PDB: {exc}") from exc
    mappings = {**DEFAULT_RESIDUE_MAPPINGS, **(residue_mappings or {})}
    pdb_id = src.name.split(".")[0].upper()
    samples: list[ProteinSample] = []
    chain_ids = sorted(
        {str(segid or chain) for segid, chain in zip(universe.atoms.segids, universe.atoms.chainIDs, strict=False)}
    )
    if chain_id is not None:
        chain_ids = [chain_id]
    for cid in chain_ids:
        residues = universe.select_atoms(f"(segid {cid} or chainID {cid}) and protein").residues
        sequence: list[str] = []
        residue_ids: list[str] = []
        coords: list[np.ndarray] = []
        missing_ca = False
        for residue in residues:
            resname = mappings.get(str(residue.resname).upper(), str(residue.resname).upper())
            if resname not in STANDARD_AA3_TO_1:
                continue
            ca = residue.atoms.select_atoms("name CA")
            if len(ca) != 1:
                missing_ca = True
                break
            sequence.append(STANDARD_AA3_TO_1[resname])
            residue_ids.append(str(residue.resid))
            coords.append(np.asarray(ca.positions[0], dtype=np.float32))
        if missing_ca:
            raise StructureParseError(f"Chain {cid} has missing C-alpha atoms")
        if not sequence:
            continue
        if min_length is not None and len(sequence) < min_length:
            continue
        if len(sequence) > max_length:
            continue
        samples.append(
            ProteinSample(
                sample_id=f"{pdb_id.lower()}_{cid}",
                pdb_id=pdb_id,
                chain_id=cid,
                sequence="".join(sequence),
                residue_ids=residue_ids,
                ca_coordinates=np.stack(coords).astype(np.float32),
                metadata={
                    "source_file": str(src),
                    "structure_format": "pdb",
                    "experimental_method": None,
                    "resolution_angstrom": None,
                    "model_number": 1,
                    "original_chain_length": len(sequence),
                },
            )
        )
    if not samples:
        raise StructureParseError("No accepted chains found")
    return samples
