"""Gemmi-backed mmCIF parser."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from protein_distance_diffusion.constants import DEFAULT_RESIDUE_MAPPINGS, STANDARD_AA3_TO_1
from protein_distance_diffusion.data.preprocess import ProteinSample, StructureRejection


class StructureParseError(RuntimeError):
    """Raised for unrecoverable structure parsing failures."""


@dataclass(frozen=True)
class _ResidueKey:
    chain_id: str
    order: int
    auth_seq_id: str
    insertion_code: str
    resname: str

    @property
    def residue_id(self) -> str:
        return f"{self.auth_seq_id}{self.insertion_code}" if self.insertion_code else self.auth_seq_id


@dataclass(frozen=True)
class _CACandidate:
    coord: np.ndarray
    altloc: str | None
    occupancy: float | None
    source_row: int


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
    return _table_to_columns(category, required=True)


def _optional_category(block: Any, prefix: str) -> dict[str, list[str]]:
    category = block.find_mmcif_category(prefix)
    if not category:
        return {}
    return _table_to_columns(category, required=False, prefix=prefix)


def _table_to_columns(category: Any, *, required: bool, prefix: str = "_atom_site.") -> dict[str, list[str]]:
    tags = [str(tag) for tag in category.tags]
    rows = [[str(value) for value in row] for row in category]
    if not tags or not rows:
        if required:
            raise StructureParseError("mmCIF file has no _atom_site category")
        return {}
    if any(len(row) != len(tags) for row in rows):
        if required:
            raise StructureParseError("mmCIF _atom_site rows have inconsistent column counts")
        raise StructureParseError(f"mmCIF {prefix} rows have inconsistent column counts")
    result: dict[str, list[str]] = {}
    for column_index, tag in enumerate(tags):
        key = tag.removeprefix(prefix)
        result[key] = [row[column_index] for row in rows]
    lengths = {len(values) for values in result.values()}
    if len(lengths) != 1:
        if required:
            raise StructureParseError("mmCIF _atom_site columns have inconsistent lengths")
        raise StructureParseError(f"mmCIF {prefix} columns have inconsistent lengths")
    return result


def _normalize_altloc(value: Any) -> str | None:
    return _clean(value)


def _occupancy(value: Any) -> float | None:
    return _float_or_none(value)


def _is_polymer_atom(group: str | None, resname: str, mappings: dict[str, str]) -> bool:
    if group == "ATOM":
        return True
    return resname in mappings or resname in STANDARD_AA3_TO_1


def _mapped_resname(resname: str, mappings: dict[str, str]) -> str:
    return mappings.get(resname, resname)


def _scheme_residues(
    block: Any,
    *,
    chain_id: str | None,
    mappings: dict[str, str],
) -> dict[str, dict[tuple[int, str, str], _ResidueKey]]:
    scheme = _optional_category(block, "_pdbx_poly_seq_scheme.")
    if not scheme:
        return {}
    chains = _col(scheme, "pdb_strand_id", "auth_asym_id", "asym_id")
    seq_ids = _col(scheme, "seq_id")
    auth_seq_ids = _col(scheme, "auth_seq_num", "pdb_seq_num", "seq_id")
    insertion = _col(scheme, "pdb_ins_code")
    residues = _col(scheme, "mon_id")
    by_chain: dict[str, dict[tuple[int, str, str], _ResidueKey]] = defaultdict(dict)
    for idx, chain in enumerate(chains):
        cid = _clean(chain) or ""
        if chain_id is not None and cid != chain_id:
            continue
        resname = (_clean(residues[idx]) or "").upper()
        mapped = _mapped_resname(resname, mappings)
        if mapped not in STANDARD_AA3_TO_1:
            continue
        seq_text = _clean(seq_ids[idx]) or _clean(auth_seq_ids[idx])
        if seq_text is None:
            continue
        try:
            order = int(float(seq_text))
        except ValueError:
            continue
        auth_seq = str(_clean(auth_seq_ids[idx]) or seq_text)
        ins = _clean(insertion[idx]) or ""
        by_chain[cid][(order, auth_seq, ins)] = _ResidueKey(cid, order, auth_seq, ins, mapped)
    return by_chain


def _choose_ca_candidate(candidates: list[_CACandidate]) -> _CACandidate | str:
    no_alt = [candidate for candidate in candidates if candidate.altloc is None]
    if len(no_alt) == 1:
        return no_alt[0]
    if len(no_alt) > 1:
        return "duplicate_ca"
    seen_altlocs = [candidate.altloc for candidate in candidates]
    if len(set(seen_altlocs)) != len(seen_altlocs):
        return "duplicate_ca"

    def sort_key(candidate: _CACandidate) -> tuple[float, int, str, int]:
        occupancy = (
            candidate.occupancy if candidate.occupancy is not None and np.isfinite(candidate.occupancy) else -1.0
        )
        return (-float(occupancy), 0 if candidate.altloc == "A" else 1, candidate.altloc or "", candidate.source_row)

    return sorted(candidates, key=sort_key)[0]


def _missing_ca_rejection(src: Path, pdb_id: str, cid: str, missing: list[_ResidueKey]) -> StructureRejection:
    residues = ", ".join(residue.residue_id for residue in missing[:5])
    if len(missing) > 5:
        residues += ", ..."
    return StructureRejection(
        str(src),
        "missing_calpha",
        f"Chain {cid} is missing C-alpha atoms for {len(missing)} residue(s): {residues}",
        pdb_id,
        cid,
    )


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
    occupancies = _col(site, "occupancy")
    altlocs = _col(site, "label_alt_id")
    groups = _col(site, "group_PDB", default="ATOM")
    models = _col(site, "pdbx_PDB_model_num", default="1")
    mappings = {**DEFAULT_RESIDUE_MAPPINGS, **(residue_mappings or {})}

    residues_by_chain = _scheme_residues(block, chain_id=chain_id, mappings=mappings)
    ca_candidates: dict[str, dict[tuple[int, str, str], list[_CACandidate]]] = defaultdict(lambda: defaultdict(list))
    unsupported_by_chain: dict[str, set[str]] = defaultdict(set)
    for idx, atom in enumerate(atom_names):
        if _clean(models[idx]) not in {None, "1"}:
            continue
        cid = _clean(chains[idx]) or ""
        if chain_id is not None and cid != chain_id:
            continue
        resname = (_clean(residues[idx]) or "").upper()
        resid = _clean(label_seq[idx]) or _clean(auth_seq[idx])
        if resid is None:
            continue
        try:
            order = int(float(resid))
        except ValueError:
            continue
        group = (_clean(groups[idx]) or "ATOM").upper()
        if not _is_polymer_atom(group, resname, mappings):
            continue
        auth = str(_clean(auth_seq[idx]) or resid)
        ins = _clean(insertion[idx]) or ""
        mapped = _mapped_resname(resname, mappings)
        key = (order, auth, ins)
        if mapped not in STANDARD_AA3_TO_1:
            unsupported_by_chain[cid].add(resname)
            residues_by_chain.setdefault(cid, {})
            continue
        residues_by_chain.setdefault(cid, {})
        residues_by_chain[cid].setdefault(key, _ResidueKey(cid, order, auth, ins, mapped))
        atom_name = (_clean(atom) or "").upper()
        if atom_name != "CA":
            continue
        try:
            coord = np.asarray([float(xs[idx]), float(ys[idx]), float(zs[idx])], dtype=np.float32)
        except ValueError:
            ca_candidates[cid][key].append(
                _CACandidate(
                    coord=np.asarray([np.nan, np.nan, np.nan], dtype=np.float32),
                    altloc=_normalize_altloc(altlocs[idx]),
                    occupancy=_occupancy(occupancies[idx]),
                    source_row=idx,
                )
            )
            continue
        ca_candidates[cid][key].append(
            _CACandidate(
                coord=coord,
                altloc=_normalize_altloc(altlocs[idx]),
                occupancy=_occupancy(occupancies[idx]),
                source_row=idx,
            )
        )

    samples: list[ProteinSample] = []
    rejections: list[StructureRejection] = []
    for cid in sorted(set(residues_by_chain) | set(unsupported_by_chain)):
        residues_for_chain = sorted(residues_by_chain.get(cid, {}).values(), key=lambda item: item.order)
        if unsupported_by_chain.get(cid):
            unsupported = ", ".join(sorted(unsupported_by_chain[cid])[:5])
            rejections.append(
                StructureRejection(
                    str(src),
                    "unsupported_residues",
                    f"Chain contains unsupported residues: {unsupported}",
                    pdb_id,
                    cid,
                )
            )
            continue
        if not residues_for_chain:
            rejections.append(
                StructureRejection(
                    str(src),
                    "unsupported_residues",
                    "Chain contains unsupported residues",
                    pdb_id,
                    cid,
                )
            )
            continue
        if any(residue.insertion_code for residue in residues_for_chain):
            rejections.append(
                StructureRejection(str(src), "insertion_codes", "Insertion codes are unsupported", pdb_id, cid)
            )
            continue
        if (
            strict_contiguous_ca
            and residues_for_chain
            and [r.order for r in residues_for_chain]
            != list(range(residues_for_chain[0].order, residues_for_chain[0].order + len(residues_for_chain)))
        ):
            rejections.append(
                StructureRejection(
                    str(src), "non_contiguous_coordinates", "C-alpha residue ids are not contiguous", pdb_id, cid
                )
            )
            continue

        selected_coords: list[np.ndarray] = []
        selected_altlocs: list[str | None] = []
        selected_occupancies: list[float | None] = []
        missing: list[_ResidueKey] = []
        duplicate = False
        nonfinite = False
        for residue in residues_for_chain:
            key = (residue.order, residue.auth_seq_id, residue.insertion_code)
            candidates = ca_candidates.get(cid, {}).get(key, [])
            if not candidates:
                missing.append(residue)
                continue
            choice = _choose_ca_candidate(candidates)
            if choice == "duplicate_ca":
                duplicate = True
                break
            if not np.isfinite(choice.coord).all():
                nonfinite = True
                break
            selected_coords.append(choice.coord)
            selected_altlocs.append(choice.altloc)
            selected_occupancies.append(choice.occupancy)
        if missing:
            rejections.append(_missing_ca_rejection(src, pdb_id, cid, missing))
            continue
        if duplicate:
            rejections.append(
                StructureRejection(str(src), "duplicate_ca", "Multiple C-alpha atoms for a residue", pdb_id, cid)
            )
            continue
        if nonfinite:
            rejections.append(
                StructureRejection(
                    str(src),
                    "nonfinite_coordinates",
                    "Chain contains non-finite C-alpha coordinates",
                    pdb_id,
                    cid,
                )
            )
            continue

        sequence = "".join(STANDARD_AA3_TO_1[residue.resname] for residue in residues_for_chain)
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
                residue_ids=[residue.residue_id for residue in residues_for_chain],
                ca_coordinates=np.stack(selected_coords).astype(np.float32),
                metadata={
                    "source_file": str(src),
                    "structure_format": "mmcif",
                    "experimental_method": method,
                    "resolution_angstrom": resolution,
                    "model_number": 1,
                    "original_chain_length": len(sequence),
                    "selected_altlocs": selected_altlocs,
                    "selected_occupancies": selected_occupancies,
                    "missing_occupancy_policy": "ranked_below_explicit_finite_occupancy",
                },
            )
        )
    return samples, rejections


def parse_mmcif_file(path: str | Path, **kwargs: Any) -> list[ProteinSample]:
    """Parse an mmCIF file and return accepted samples."""
    samples, _ = parse_mmcif_file_with_rejections(path, **kwargs)
    return samples
