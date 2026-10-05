"""Read-only sequence provenance and geometry-pairing audit helpers."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from protein_distance_diffusion.constants import AA_TO_TOKEN, DEFAULT_RESIDUE_MAPPINGS, STANDARD_AA3_TO_1

CANONICAL_AA = frozenset(STANDARD_AA3_TO_1.values())


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def canonical_sequence(sequence: str) -> bool:
    return bool(sequence) and set(sequence) <= CANONICAL_AA


def _clean(value: Any) -> str | None:
    text = str(value).strip().strip("'\"") if value is not None else ""
    return None if text in {"", ".", "?"} else text


def _columns(block: Any, prefix: str) -> dict[str, list[str]]:
    table = block.find_mmcif_category(prefix)
    if not table:
        return {}
    tags = [str(tag).removeprefix(prefix) for tag in table.tags]
    rows = [[str(value) for value in row] for row in table]
    if not tags or not rows:
        return {}
    if any(len(row) != len(tags) for row in rows):
        raise ValueError(f"{prefix} contains rows with inconsistent widths")
    return {tag: [row[index] for row in rows] for index, tag in enumerate(tags)}


def _column(columns: dict[str, list[str]], *names: str, default: str = "?") -> list[str]:
    for name in names:
        if name in columns:
            return columns[name]
    if not columns:
        return []
    return [default] * len(next(iter(columns.values())))


def _mapped_token(resname: str) -> tuple[str, str]:
    name = resname.upper()
    mapped = DEFAULT_RESIDUE_MAPPINGS.get(name, name)
    if mapped in STANDARD_AA3_TO_1:
        return STANDARD_AA3_TO_1[mapped], "modified" if mapped != name else "canonical"
    return "X", "unknown"


def _ordered_residues(entries: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(entries.values(), key=lambda item: (item["order"], item["first_row"]))


def _legacy_inspect_mmcif(path: str | Path) -> tuple[list[dict[str, Any]], Counter[tuple[str, str, str]]]:
    """Inspect chain/model sequence evidence without accepting or rewriting samples."""
    try:
        import gemmi
    except ModuleNotFoundError as exc:
        raise RuntimeError("Gemmi is required for the raw sequence-readiness audit") from exc

    source = Path(path)
    source_hash_before = sha256_file(source)
    block = gemmi.cif.read_file(str(source)).sole_block()
    pdb_id = str(block.name or source.name.split(".")[0]).upper()
    method = _clean(block.find_value("_exptl.method"))
    is_nmr = "NMR" in (method or "").upper()
    token_counts: Counter[tuple[str, str, str]] = Counter()

    scheme = _columns(block, "_pdbx_poly_seq_scheme.")
    scheme_by_chain: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    if scheme:
        chains = _column(scheme, "pdb_strand_id", "auth_asym_id", "asym_id")
        seq_ids = _column(scheme, "seq_id", "auth_seq_num")
        auth_ids = _column(scheme, "auth_seq_num", "pdb_seq_num", "seq_id")
        insertions = _column(scheme, "pdb_ins_code")
        residues = _column(scheme, "mon_id")
        for index, chain in enumerate(chains):
            chain_id = _clean(chain) or ""
            label_id = _clean(seq_ids[index]) or str(index + 1)
            auth_id = _clean(auth_ids[index]) or label_id
            insertion = _clean(insertions[index]) or ""
            token, token_kind = _mapped_token(_clean(residues[index]) or "")
            token_counts[("scheme", (_clean(residues[index]) or "").upper(), token_kind)] += 1
            try:
                order = float(label_id)
            except ValueError:
                order = float(index + 1)
            scheme_by_chain[chain_id][(label_id, insertion)] = {
                "label_id": label_id,
                "auth_id": auth_id,
                "insertion": insertion,
                "token": token,
                "token_kind": token_kind,
                "resname": (_clean(residues[index]) or "").upper(),
                "order": order,
                "first_row": index,
            }

    atom = _columns(block, "_atom_site.")
    if not atom:
        raise ValueError("mmCIF file has no _atom_site category")
    atom_names = _column(atom, "label_atom_id", "auth_atom_id")
    chains = _column(atom, "auth_asym_id", "label_asym_id")
    models = _column(atom, "pdbx_PDB_model_num", default="1")
    label_ids = _column(atom, "label_seq_id", "auth_seq_id")
    auth_ids = _column(atom, "auth_seq_id", "label_seq_id")
    insertions = _column(atom, "pdbx_PDB_ins_code")
    residues = _column(atom, "label_comp_id", "auth_comp_id")
    altlocs = _column(atom, "label_alt_id")
    occupancies = _column(atom, "occupancy")
    atom_by_chain_model: dict[tuple[str, str], dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    ca_positions_by_chain_model: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    ca_candidates: dict[tuple[str, str, str, str], list[tuple[str | None, float | None, int]]] = defaultdict(list)
    altlocs_by_position: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    ambiguous_positions: set[tuple[str, str, str, str]] = set()
    for index, atom_name in enumerate(atom_names):
        chain_id = _clean(chains[index]) or ""
        model_id = _clean(models[index]) or "1"
        label_id = _clean(label_ids[index]) or _clean(auth_ids[index])
        if label_id is None:
            continue
        auth_id = _clean(auth_ids[index]) or label_id
        insertion = _clean(insertions[index]) or ""
        resname = (_clean(residues[index]) or "").upper()
        token, token_kind = _mapped_token(resname)
        position = (label_id, insertion)
        key = (chain_id, model_id, label_id, insertion)
        existing = atom_by_chain_model[(chain_id, model_id)].get(position)
        if existing is not None and existing["resname"] != resname:
            ambiguous_positions.add(key)
        elif existing is None:
            try:
                order = float(label_id)
            except ValueError:
                order = float(index + 1)
            atom_by_chain_model[(chain_id, model_id)][position] = {
                "label_id": label_id,
                "auth_id": auth_id,
                "insertion": insertion,
                "token": token,
                "token_kind": token_kind,
                "resname": resname,
                "order": order,
                "first_row": index,
            }
            token_counts[("atom", resname, token_kind)] += 1
        if (_clean(atom_name) or "").upper() != "CA":
            continue
        ca_positions_by_chain_model[(chain_id, model_id)].add(position)
        altloc = _clean(altlocs[index])
        if altloc is not None:
            altlocs_by_position[key].add(altloc)
        occupancy_text = _clean(occupancies[index])
        try:
            occupancy = float(occupancy_text) if occupancy_text is not None else None
        except ValueError:
            occupancy = None
        ca_candidates[key].append((altloc, occupancy, index))

    chain_ids = sorted(set(scheme_by_chain) | {chain for chain, _ in atom_by_chain_model})
    model_ids_by_chain = {
        chain: sorted(model for candidate_chain, model in atom_by_chain_model if candidate_chain == chain)
        for chain in chain_ids
    }
    rows = []
    for chain_id in chain_ids:
        scheme_residues = _ordered_residues(scheme_by_chain.get(chain_id, {}))
        scheme_sequence = "".join(item["token"] for item in scheme_residues)
        model_sequences = {
            model_id: "".join(
                item["token"] for item in _ordered_residues(atom_by_chain_model.get((chain_id, model_id), {}))
            )
            for model_id in model_ids_by_chain[chain_id]
        }
        sequence_consistent = len(set(model_sequences.values())) <= 1
        for model_id in model_ids_by_chain[chain_id] or ["1"]:
            atom_residues = _ordered_residues(atom_by_chain_model.get((chain_id, model_id), {}))
            atom_positions = {(item["label_id"], item["insertion"]) for item in atom_residues}
            ca_positions = ca_positions_by_chain_model.get((chain_id, model_id), set())
            alignment_residues = scheme_residues or atom_residues
            missing_residue_mask = [
                (item["label_id"], item["insertion"]) not in atom_positions for item in alignment_residues
            ]
            missing_calpha_mask = [
                (item["label_id"], item["insertion"]) not in ca_positions for item in alignment_residues
            ]

            def selected_altloc(
                item: dict[str, Any], selected_chain: str = chain_id, selected_model: str = model_id
            ) -> str | None:
                key = (selected_chain, selected_model, item["label_id"], item["insertion"])
                candidates = ca_candidates.get(key, [])
                if not candidates:
                    return None
                return sorted(
                    candidates,
                    key=lambda candidate: (
                        0 if candidate[0] is None else 1,
                        -(candidate[1] if candidate[1] is not None and np.isfinite(candidate[1]) else -1.0),
                        0 if candidate[0] == "A" else 1,
                        candidate[0] or "",
                        candidate[2],
                    ),
                )[0][0]

            unknown = Counter(
                item["resname"] for item in scheme_residues + atom_residues if item["token_kind"] == "unknown"
            )
            modified = Counter(
                item["resname"] for item in scheme_residues + atom_residues if item["token_kind"] == "modified"
            )
            rows.append(
                {
                    "source_file": str(source),
                    "source_sha256": source_hash_before,
                    "pdb_id": pdb_id,
                    "chain_id": chain_id,
                    "model_id": str(model_id),
                    "experimental_method": method,
                    "is_nmr": is_nmr,
                    "chain_count": len(chain_ids),
                    "model_count": len(model_ids_by_chain[chain_id]),
                    "seqres_sequence": scheme_sequence or None,
                    "scheme_sequence": scheme_sequence or None,
                    "atom_sequence": model_sequences.get(model_id) or None,
                    "scheme_length": len(scheme_residues),
                    "atom_calpha_length": len(atom_residues),
                    "missing_residue_count": int(sum(missing_residue_mask)),
                    "missing_calpha_count": int(sum(missing_calpha_mask)),
                    "label_sequence_ids": json.dumps([item["label_id"] for item in alignment_residues]),
                    "auth_sequence_ids": json.dumps([item["auth_id"] for item in alignment_residues]),
                    "insertion_codes": json.dumps([item["insertion"] for item in alignment_residues]),
                    "residue_names": json.dumps([item["resname"] for item in alignment_residues]),
                    "residue_tokens": json.dumps([item["token"] for item in alignment_residues]),
                    "missing_residue_mask": json.dumps(missing_residue_mask),
                    "missing_calpha_mask": json.dumps(missing_calpha_mask),
                    "selected_altlocs": json.dumps([selected_altloc(item) for item in alignment_residues]),
                    "insertion_code_count": sum(bool(item["insertion"]) for item in scheme_residues + atom_residues),
                    "alternate_location_position_count": sum(
                        bool(values)
                        for (chain, model, _label, _insertion), values in altlocs_by_position.items()
                        if chain == chain_id and model == model_id
                    ),
                    "ambiguous_residue_position_count": sum(
                        1
                        for chain, model, _label, _insertion in ambiguous_positions
                        if chain == chain_id and model == model_id
                    ),
                    "unknown_residue_counts": json.dumps(dict(sorted(unknown.items()))),
                    "modified_residue_counts": json.dumps(dict(sorted(modified.items()))),
                    "sequence_consistent_across_models": sequence_consistent,
                }
            )
    if sha256_file(source) != source_hash_before:
        raise RuntimeError(f"Raw structure changed during inspection: {source}")
    return rows, token_counts


def inspect_mmcif(
    path: str | Path,
) -> tuple[list[dict[str, Any]], Counter[tuple[str, str, str, str]]]:
    """Inspect only declared protein-polymer chains and coordinate models."""
    from protein_distance_diffusion.evaluation.raw_sequence_readiness import inspect_protein_mmcif

    return inspect_protein_mmcif(path)


def infer_sequence_source(sequence: str, raw_row: dict[str, Any] | None) -> str:
    """Classify provenance conservatively from raw chain evidence."""
    if raw_row is None:
        return "unknown"
    scheme = str(raw_row.get("seqres_sequence") or raw_row.get("scheme_sequence") or "")
    atom = str(raw_row.get("atom_sequence") or "")
    if scheme and sequence and sequence in scheme:
        return "SEQRES-derived"
    if atom and sequence and sequence in atom:
        return "ATOM-derived"
    return "unknown"


def sequence_evidence_occurrences(sequence: str, raw_row: dict[str, Any] | None, source: str) -> int:
    """Count exact contiguous occurrences in the selected raw evidence."""
    if raw_row is None or not sequence:
        return 0
    field = {"SEQRES-derived": "seqres_sequence", "ATOM-derived": "atom_sequence"}.get(source)
    if field is None:
        return 0
    evidence = str(raw_row.get(field) or raw_row.get("scheme_sequence") or "")
    return sum(evidence.startswith(sequence, offset) for offset in range(len(evidence) - len(sequence) + 1))


def _npz_array_shape(path: Path, key: str) -> tuple[int, ...] | None:
    """Read an NPY member shape without decompressing its array payload."""
    member = f"{key}.npy"
    with zipfile.ZipFile(path) as archive:
        if member not in archive.namelist():
            return None
        with archive.open(member) as handle:
            version = np.lib.format.read_magic(handle)
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, _fortran_order, _dtype = reader(handle)
    return tuple(int(value) for value in shape)


def inspect_processed_npz(path: str | Path, *, include_geometry: bool = False) -> dict[str, Any]:
    """Read sequence/geometry metadata from one processed NPZ without mutation."""
    source = Path(path)
    stat_before = source.stat()
    digest = sha256_file(source)
    matrix_shape = _npz_array_shape(source, "distance_matrix")
    with np.load(source, allow_pickle=False) as data:
        sequence = str(data["sequence"]) if "sequence" in data else ""
        sequence_tokens = np.asarray(data["sequence_tokens"]).tolist() if "sequence_tokens" in data else []
        residue_ids = [str(item) for item in data["residue_ids"].tolist()] if "residue_ids" in data else []
        residue_mask = np.asarray(data["residue_mask"], dtype=bool) if "residue_mask" in data else np.asarray([])
        coordinates = np.asarray(data["ca_coordinates"]) if "ca_coordinates" in data else np.asarray([])
        metadata = json.loads(str(data["metadata"])) if "metadata" in data else {}
        expected_tokens = [AA_TO_TOKEN[amino_acid] for amino_acid in sequence] if canonical_sequence(sequence) else []
        result = {
            "npz_sequence": sequence,
            "sequence_tokens": sequence_tokens,
            "sequence_tokens_match_sequence": sequence_tokens == expected_tokens,
            "matrix_rows": matrix_shape[0] if matrix_shape is not None and len(matrix_shape) == 2 else None,
            "matrix_columns": matrix_shape[1] if matrix_shape is not None and len(matrix_shape) == 2 else None,
            "matrix_is_square": bool(
                matrix_shape is not None and len(matrix_shape) == 2 and matrix_shape[0] == matrix_shape[1]
            ),
            "residue_ids": residue_ids,
            "residue_id_count": len(residue_ids),
            "residue_mask_count": int(residue_mask.size),
            "residue_mask_true_count": int(residue_mask.sum()),
            "coordinate_count": (
                int(coordinates.shape[0]) if coordinates.ndim == 2 and coordinates.shape[1] == 3 else None
            ),
            "coordinates_have_n_by_3_shape": bool(coordinates.ndim == 2 and coordinates.shape[1] == 3),
            "retained_insertion_codes": metadata.get("retained_insertion_codes"),
            "selected_altlocs": metadata.get("selected_altlocs"),
            "retained_start_label_seq_id": metadata.get("retained_start_label_seq_id"),
            "retained_end_label_seq_id": metadata.get("retained_end_label_seq_id"),
            "trimmed_n_terminal_residues": metadata.get("trimmed_n_terminal_residues"),
            "trimmed_c_terminal_residues": metadata.get("trimmed_c_terminal_residues"),
            "original_chain_length": metadata.get("original_chain_length"),
            "terminal_trimming_applied": metadata.get("terminal_trimming_applied"),
            "model_id_npz": str(metadata.get("model_number", "")),
            "source_file_npz": metadata.get("source_file"),
            "matrix_path_sha256": digest,
        }
        if include_geometry:
            result["distance_matrix"] = np.asarray(data["distance_matrix"], dtype=np.float32)
            result["ca_coordinates"] = np.asarray(coordinates, dtype=np.float32)
    stat_after = source.stat()
    if (stat_before.st_size, stat_before.st_mtime_ns) != (stat_after.st_size, stat_after.st_mtime_ns):
        raise RuntimeError(f"Processed NPZ changed during inspection: {source}")
    return result
