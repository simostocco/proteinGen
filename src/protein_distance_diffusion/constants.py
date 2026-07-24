"""Shared biochemical constants for protein distance-map preprocessing."""

from __future__ import annotations

STANDARD_AA3_TO_1: dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
AA_TO_TOKEN: dict[str, int] = {aa: i + 1 for i, aa in enumerate(sorted(set(STANDARD_AA3_TO_1.values())))}
TOKEN_TO_AA: dict[int, str] = {value: key for key, value in AA_TO_TOKEN.items()}

# Explicit, conservative mappings for common unambiguous protein modifications.
DEFAULT_RESIDUE_MAPPINGS: dict[str, str] = {"MSE": "MET"}

DEFAULT_LENGTH_BOUNDARIES: tuple[int, ...] = (64, 96, 128)
