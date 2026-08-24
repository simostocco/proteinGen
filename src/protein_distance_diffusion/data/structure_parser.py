"""Structure parser backend selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from protein_distance_diffusion.data.mdanalysis_parser import parse_pdb_file
from protein_distance_diffusion.data.mmcif_parser import parse_mmcif_file_with_rejections
from protein_distance_diffusion.data.preprocess import ProteinSample, StructureRejection

SUPPORTED_SUFFIXES = (".cif", ".mmcif", ".cif.gz", ".mmcif.gz", ".pdb", ".ent")


def detect_structure_format(path: str | Path) -> str:
    """Return ``mmcif`` or ``pdb`` from a supported structure filename."""
    name = str(path).lower()
    if name.endswith((".cif", ".mmcif", ".cif.gz", ".mmcif.gz")):
        return "mmcif"
    if name.endswith((".pdb", ".ent")):
        return "pdb"
    raise ValueError(f"Unsupported structure format: {path}")


def select_backend(path: str | Path, backend: str = "auto") -> str:
    """Select a concrete parser backend."""
    normalized = str(backend).lower()
    if normalized == "auto":
        return "gemmi" if detect_structure_format(path) == "mmcif" else "mdanalysis"
    if normalized in {"gemmi", "mdanalysis"}:
        return normalized
    raise ValueError("backend must be one of auto, gemmi, mdanalysis")


def iter_supported_structure_files(source_dir: str | Path) -> list[Path]:
    """Return supported structure files below ``source_dir`` in deterministic order."""
    root = Path(source_dir)
    files = [path for path in root.rglob("*") if path.is_file() and str(path).lower().endswith(SUPPORTED_SUFFIXES)]
    return sorted(files)


def parse_structure_file_with_rejections(
    path: str | Path,
    *,
    backend: str = "auto",
    **kwargs: Any,
) -> tuple[list[ProteinSample], list[StructureRejection]]:
    """Parse one structure file using the requested backend."""
    concrete = select_backend(path, backend)
    if concrete == "gemmi":
        return parse_mmcif_file_with_rejections(path, **kwargs)
    if detect_structure_format(path) != "pdb":
        raise ValueError("MDAnalysis backend supports only legacy PDB/ENT input")
    try:
        return parse_pdb_file(path, **kwargs), []
    except Exception as exc:
        return [], [
            StructureRejection(
                source_file=str(path),
                reason="parse_error",
                message=str(exc),
                pdb_id=Path(path).stem.upper(),
                chain_id=kwargs.get("chain_id"),
            )
        ]


def parse_structure_file(path: str | Path, *, backend: str = "auto", **kwargs: Any) -> list[ProteinSample]:
    """Parse one structure file and return accepted chain samples."""
    samples, _ = parse_structure_file_with_rejections(path, backend=backend, **kwargs)
    return samples
