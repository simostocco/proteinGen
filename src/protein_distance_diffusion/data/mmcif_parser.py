"""Gemmi-backed mmCIF parser."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from protein_distance_diffusion.constants import DEFAULT_RESIDUE_MAPPINGS, STANDARD_AA3_TO_1
from protein_distance_diffusion.data.preprocess import ProteinSample, StructureRejection


class StructureParseError(RuntimeError):
    """Raised for unrecoverable structure parsing failures."""


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().strip("'\"")
    return None if text in {"", ".", "?"} else text


def _float_or_none(value: Any) -> float | None:
    text = _clean(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _validate_min_length(min_length: int | None) -> None:
    if min_length is not None and int(min_length) <= 0:
        raise ValueError("min_length must be a positive integer or None")


def _metadata(block: Any) -> tuple[str | None, float | None]:
    method = _clean(block.find_value("_exptl.method"))
    resolution = _float_or_none(block.find_value("_refine.ls_d_res_high"))
    if resolution is None:
        resolution = _float_or_none(block.find_value("_em_3d_reconstruction.resolution"))
    return method, resolution


def _atom_site(block: Any) -> dict[str, list[str]]:
    category = block.find_mmcif_category("_atom_site.")
    if not category:
        raise StructureParseError("mmCIF file has no _atom_site category")
    tags = [str(tag) for tag in category.tags]
    rows = [[str(value) for value in row] for row in category]
    if not tags or not rows:
        raise StructureParseError("mmCIF file has no _atom_site category")
    if any(len(row) != len(tags) for row in rows):
        raise StructureParseError("mmCIF _atom_site rows have inconsistent column counts")
    result: dict[str, list[str]] = {}
    for column_index, tag in enumerate(tags):
        key = tag.removeprefix("_atom_site.")
        result[key] = [row[column_index] for row in rows]
    lengths = {len(values) for values in result.values()}
    if len(lengths) != 1:
        raise StructureParseError("mmCIF _atom_site columns have inconsistent lengths")
    return result


def _col(site: dict[str, list[str]], *names: str, default: str = "?") -> list[str]:
    for name in names:
        if name in site:
            return site[name]
    first = next(iter(site.values()))
    return [default] * len(first)


def _passes_metadata(
    method: str | None,
    resolution: float | None,
    *,
    allowed_methods: list[str] | None,
    max_xray_resolution_angstrom: float | None,
    max_cryoem_resolution_angstrom: float | None,
) -> tuple[bool, str | None]:
    if allowed_methods is not None and method not in allowed_methods:
        return False, "experimental_method_not_allowed"
    method_upper = (method or "").upper()
    if "X-RAY" in method_upper and max_xray_resolution_angstrom is not None:
        if resolution is None or resolution > max_xray_resolution_angstrom:
            return False, "xray_resolution_above_maximum"
    if ("ELECTRON MICROSCOPY" in method_upper or "CRYO" in method_upper) and max_cryoem_resolution_angstrom is not None:
        if resolution is None or resolution > max_cryoem_resolution_angstrom:
            return False, "cryoem_resolution_above_maximum"
    return True, None


def _sample_id(pdb_id: str, chain_id: str) -> str:
    return f"{pdb_id.lower()}_{chain_id}"


def parse_mmcif_file_with_rejections(
    path: str | Path,
    *,
    min_length: int | None = 20,
    max_length: int = 500,
    chain_id: str | None = None,
    residue_mappings: dict[str, str] | None = None,
    allowed_methods: list[str] | None = None,
    max_xray_resolution_angstrom: float | None = None,
    max_cryoem_resolution_angstrom: float | None = None,
    strict_contiguous_ca: bool = True,
) -> tuple[list[ProteinSample], list[StructureRejection]]:
    """Parse an mmCIF/mmCIF.gz file into chain-level samples and rejections."""
    _validate_min_length(min_length)
    try:
        import gemmi
    except ModuleNotFoundError as exc:
        raise StructureParseError("Gemmi is required to parse mmCIF files") from exc

    src = Path(path)
    try:
        block = gemmi.cif.read_file(str(src)).sole_block()
    except Exception as exc:
        raise StructureParseError(f"Failed to parse mmCIF: {exc}") from exc
    pdb_id = str(block.name or src.stem).upper()
    if pdb_id.lower().endswith(".cif"):
        pdb_id = pdb_id[:-4]
    method, resolution = _metadata(block)
    ok, reason = _passes_metadata(
        method,
        resolution,
        allowed_methods=allowed_methods,
        max_xray_resolution_angstrom=max_xray_resolution_angstrom,
        max_cryoem_resolution_angstrom=max_cryoem_resolution_angstrom,
    )
    if not ok:
        return [], [
            StructureRejection(
                source_file=str(src),
                pdb_id=pdb_id,
                chain_id=None,
                reason=reason or "metadata_rejected",
                message=f"Structure metadata rejected: method={method}, resolution={resolution}",
            )
        ]

    site = _atom_site(block)
    atom_names = _col(site, "label_atom_id", "auth_atom_id")
    residues = _col(site, "label_comp_id", "auth_comp_id")
    chains = _col(site, "auth_asym_id", "label_asym_id")
    label_seq = _col(site, "label_seq_id", "auth_seq_id")
    auth_seq = _col(site, "auth_seq_id", "label_seq_id")
    insertion = _col(site, "pdbx_PDB_ins_code")
    xs = _col(site, "Cartn_x")
    ys = _col(site, "Cartn_y")
    zs = _col(site, "Cartn_z")
    models = _col(site, "pdbx_PDB_model_num", default="1")
    rows_by_chain: dict[str, list[tuple[int, str, str, str, np.ndarray]]] = defaultdict(list)
    residues_by_chain: dict[str, list[str]] = defaultdict(list)
    mappings = {**DEFAULT_RESIDUE_MAPPINGS, **(residue_mappings or {})}
    for idx, atom in enumerate(atom_names):
        if _clean(models[idx]) not in {None, "1"}:
            continue
        cid = _clean(chains[idx]) or ""
        if chain_id is not None and cid != chain_id:
            continue
        resname = (_clean(residues[idx]) or "").upper()
        residues_by_chain[cid].append(resname)
        atom_name = (_clean(atom) or "").upper()
        if atom_name != "CA":
            continue
        ins = _clean(insertion[idx])
        resid = _clean(label_seq[idx]) or _clean(auth_seq[idx])
        if resid is None:
            continue
        try:
            order = int(float(resid))
        except ValueError:
            continue
        coord = np.asarray([float(xs[idx]), float(ys[idx]), float(zs[idx])], dtype=np.float32)
        rows_by_chain[cid].append((order, str(_clean(auth_seq[idx]) or resid), ins or "", resname, coord))

    samples: list[ProteinSample] = []
    rejections: list[StructureRejection] = []
    for cid in sorted(residues_by_chain):
        rows = sorted(rows_by_chain[cid], key=lambda item: item[0])
        if not rows:
            mapped = [mappings.get(resname, resname) for resname in residues_by_chain[cid]]
            has_unsupported_residues = any(resname not in STANDARD_AA3_TO_1 for resname in mapped)
            reason = "unsupported_residues" if has_unsupported_residues else "missing_ca"
            message = (
                "Chain contains unsupported residues" if has_unsupported_residues else "Chain has no C-alpha atoms"
            )
            rejections.append(StructureRejection(str(src), reason, message, pdb_id, cid))
            continue
        if any(ins for _, _, ins, _, _ in rows):
            rejections.append(
                StructureRejection(str(src), "insertion_codes", "Insertion codes are unsupported", pdb_id, cid)
            )
            continue
        if len({order for order, *_ in rows}) != len(rows):
            rejections.append(
                StructureRejection(str(src), "duplicate_ca", "Multiple C-alpha atoms for a residue", pdb_id, cid)
            )
            continue
        if strict_contiguous_ca and rows and [r[0] for r in rows] != list(range(rows[0][0], rows[0][0] + len(rows))):
            rejections.append(
                StructureRejection(
                    str(src), "non_contiguous_coordinates", "C-alpha residue ids are not contiguous", pdb_id, cid
                )
            )
            continue
        mapped = [mappings.get(resname, resname) for _, _, _, resname, _ in rows]
        if any(resname not in STANDARD_AA3_TO_1 for resname in mapped):
            rejections.append(
                StructureRejection(str(src), "unsupported_residues", "Chain contains unsupported residues", pdb_id, cid)
            )
            continue
        sequence = "".join(STANDARD_AA3_TO_1[resname] for resname in mapped)
        if min_length is not None and len(sequence) < min_length:
            rejections.append(
                StructureRejection(
                    str(src),
                    "length_below_minimum",
                    f"Chain length {len(sequence)} < min_length={min_length}",
                    pdb_id,
                    cid,
                )
            )
            continue
        if len(sequence) > max_length:
            rejections.append(
                StructureRejection(
                    str(src),
                    "length_above_maximum",
                    f"Chain length {len(sequence)} > max_length={max_length}",
                    pdb_id,
                    cid,
                )
            )
            continue
        samples.append(
            ProteinSample(
                sample_id=_sample_id(pdb_id, cid),
                pdb_id=pdb_id,
                chain_id=cid,
                sequence=sequence,
                residue_ids=[auth for _, auth, _, _, _ in rows],
                ca_coordinates=np.stack([coord for *_, coord in rows]).astype(np.float32),
                metadata={
                    "source_file": str(src),
                    "structure_format": "mmcif",
                    "experimental_method": method,
                    "resolution_angstrom": resolution,
                    "model_number": 1,
                    "original_chain_length": len(sequence),
                },
            )
        )
    return samples, rejections


def parse_mmcif_file(path: str | Path, **kwargs: Any) -> list[ProteinSample]:
    """Parse an mmCIF file and return accepted samples."""
    samples, _ = parse_mmcif_file_with_rejections(path, **kwargs)
    return samples
