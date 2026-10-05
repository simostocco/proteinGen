"""Entity-aware raw mmCIF sequence provenance inspection."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from protein_distance_diffusion.constants import DEFAULT_RESIDUE_MAPPINGS, STANDARD_AA3_TO_1

WATER_COMPONENTS = {"DOD", "HOH"}
ION_COMPONENTS = {
    "CA",
    "CD",
    "CL",
    "CO",
    "CU",
    "FE",
    "K",
    "MG",
    "MN",
    "NA",
    "NI",
    "ZN",
}
RESIDUE_MAPPING_VERSION = "protein_distance_diffusion_default_v1"


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


def _numeric_order(value: str, fallback: int) -> tuple[float, int]:
    try:
        return float(value), fallback
    except ValueError:
        return float(fallback), fallback


def _component_class(resname: str) -> str:
    if resname in WATER_COMPONENTS:
        return "water"
    if resname in ION_COMPONENTS:
        return "ion"
    return "ligand"


def _select_ca(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str, bool]:
    """Reproduce the preprocessing blank-altloc/occupancy/A/name ordering."""
    no_alt = [candidate for candidate in candidates if candidate["altloc"] is None]
    if len(no_alt) == 1:
        return no_alt[0], "selected_blank_altloc", False
    if len(no_alt) > 1:
        return None, "ambiguous_duplicate_blank_altloc", False
    altlocs = [candidate["altloc"] for candidate in candidates]
    if len(set(altlocs)) != len(altlocs):
        return None, "ambiguous_duplicate_altloc", False

    def key(candidate: dict[str, Any]) -> tuple[float, int, str, int]:
        occupancy = candidate["occupancy"]
        rank = occupancy if occupancy is not None and np.isfinite(occupancy) else -1.0
        return (-float(rank), 0 if candidate["altloc"] == "A" else 1, candidate["altloc"] or "", candidate["row"])

    ordered = sorted(candidates, key=key)
    finite_occupancies = [
        candidate["occupancy"]
        for candidate in candidates
        if candidate["occupancy"] is not None and np.isfinite(candidate["occupancy"])
    ]
    deterministic_tie = len(finite_occupancies) > 1 and len(set(finite_occupancies)) < len(finite_occupancies)
    return ordered[0], "selected_ranked_altloc", deterministic_tie


def inspect_protein_mmcif(
    path: str | Path,
) -> tuple[list[dict[str, Any]], Counter[tuple[str, str, str, str]]]:
    """Inspect protein-polymer chains and models without accepting samples."""
    try:
        import gemmi
    except ModuleNotFoundError as exc:
        raise RuntimeError("Gemmi is required for the raw sequence-readiness audit") from exc

    from protein_distance_diffusion.evaluation.sequence_readiness import sha256_file

    source = Path(path)
    source_hash = sha256_file(source)
    block = gemmi.cif.read_file(str(source)).sole_block()
    pdb_id = str(block.name or source.name.split(".")[0]).upper()
    method = _clean(block.find_value("_exptl.method"))
    is_nmr = "NMR" in (method or "").upper()

    entity_poly = _columns(block, "_entity_poly.")
    entity_polymer_types = {
        str(entity_id): str(polymer_type)
        for entity_id, polymer_type in zip(
            _column(entity_poly, "entity_id"), _column(entity_poly, "type"), strict=False
        )
        if _clean(entity_id) is not None
    }
    entity_poly_sequences = {
        str(entity_id): "".join(str(sequence).split()).upper()
        for entity_id, sequence in zip(
            _column(entity_poly, "entity_id"),
            _column(entity_poly, "pdbx_seq_one_letter_code_can"),
            strict=False,
        )
        if _clean(entity_id) is not None and _clean(sequence) is not None
    }
    protein_entities = {
        str(entity_id)
        for entity_id, polymer_type in zip(
            _column(entity_poly, "entity_id"), _column(entity_poly, "type"), strict=False
        )
        if _clean(entity_id) is not None and "polypeptide" in str(polymer_type).lower()
    }
    has_entity_contract = bool(entity_poly)
    struct_asym = _columns(block, "_struct_asym.")
    struct_entity = {
        str(label): str(entity)
        for label, entity in zip(_column(struct_asym, "id"), _column(struct_asym, "entity_id"), strict=False)
        if _clean(label) is not None and _clean(entity) is not None
    }
    protein_label_chains = {label for label, entity in struct_entity.items() if entity in protein_entities}

    counts: Counter[tuple[str, str, str, str]] = Counter()
    scheme = _columns(block, "_pdbx_poly_seq_scheme.")
    scheme_by_chain: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    scheme_authors: dict[str, set[str]] = defaultdict(set)
    if scheme:
        label_chains = _column(scheme, "asym_id")
        author_chains = _column(scheme, "pdb_strand_id", "auth_asym_id")
        entity_ids = _column(scheme, "entity_id")
        label_ids = _column(scheme, "seq_id", "auth_seq_num")
        auth_ids = _column(scheme, "auth_seq_num", "pdb_seq_num", "seq_id")
        insertions = _column(scheme, "pdb_ins_code")
        residues = _column(scheme, "mon_id")
        for index, raw_label_chain in enumerate(label_chains):
            author_chain = _clean(author_chains[index])
            label_chain = _clean(raw_label_chain) or author_chain
            if label_chain is None:
                continue
            entity_id = _clean(entity_ids[index]) or struct_entity.get(label_chain)
            if has_entity_contract and entity_id not in protein_entities:
                continue
            resname = (_clean(residues[index]) or "").upper()
            token, token_kind = _mapped_token(resname)
            if not has_entity_contract and token_kind == "unknown":
                continue
            label_id = _clean(label_ids[index]) or _clean(auth_ids[index])
            if label_id is None:
                continue
            auth_id = _clean(auth_ids[index]) or label_id
            insertion = _clean(insertions[index]) or ""
            protein_label_chains.add(label_chain)
            if author_chain:
                scheme_authors[label_chain].add(author_chain)
            key = (label_id, insertion)
            existing = scheme_by_chain[label_chain].get(key)
            if existing is None:
                scheme_by_chain[label_chain][key] = {
                    "label_id": label_id,
                    "auth_id": auth_id,
                    "insertion": insertion,
                    "resname": resname,
                    "token": token,
                    "token_kind": token_kind,
                    "order": _numeric_order(label_id, index + 1),
                }
                counts[("evidence_occurrence", "scheme", resname, token_kind)] += 1
                counts[("biological_residue", "declared_polymer", resname, token_kind)] += 1

    atom = _columns(block, "_atom_site.")
    if not atom:
        raise ValueError("mmCIF file has no _atom_site category")
    atom_names = _column(atom, "label_atom_id", "auth_atom_id")
    label_chains = _column(atom, "label_asym_id", "auth_asym_id")
    author_chains = _column(atom, "auth_asym_id", "label_asym_id")
    entity_ids = _column(atom, "label_entity_id")
    models = _column(atom, "pdbx_PDB_model_num", default="1")
    label_ids = _column(atom, "label_seq_id", "auth_seq_id")
    auth_ids = _column(atom, "auth_seq_id", "label_seq_id")
    insertions = _column(atom, "pdbx_PDB_ins_code")
    residues = _column(atom, "label_comp_id", "auth_comp_id")
    groups = _column(atom, "group_PDB", default="ATOM")
    altlocs = _column(atom, "label_alt_id")
    occupancies = _column(atom, "occupancy")
    xs = _column(atom, "Cartn_x")
    ys = _column(atom, "Cartn_y")
    zs = _column(atom, "Cartn_z")

    atom_by_chain_model: dict[tuple[str, str], dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    author_by_chain_model: dict[tuple[str, str], set[str]] = defaultdict(set)
    ca_candidates: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    nonpolymer_positions: set[tuple[str, str, str, str, str, str]] = set()
    nonpolymer_components: Counter[str] = Counter()
    source_models: set[str] = set()
    for index, atom_name in enumerate(atom_names):
        author_chain = _clean(author_chains[index]) or ""
        label_chain = _clean(label_chains[index]) or author_chain
        model_id = _clean(models[index]) or "1"
        entity_id = _clean(entity_ids[index]) or struct_entity.get(label_chain)
        resname = (_clean(residues[index]) or "").upper()
        label_id = _clean(label_ids[index])
        auth_id = _clean(auth_ids[index])
        insertion = _clean(insertions[index]) or ""
        group = (_clean(groups[index]) or "ATOM").upper()
        token, token_kind = _mapped_token(resname)
        if has_entity_contract:
            is_protein = entity_id in protein_entities or label_chain in protein_label_chains
        else:
            is_protein = token_kind != "unknown" and (group == "ATOM" or resname in DEFAULT_RESIDUE_MAPPINGS)
        if not is_protein or label_id is None:
            component_key = (model_id, label_chain, author_chain, auth_id or "", insertion, resname)
            if component_key not in nonpolymer_positions:
                nonpolymer_positions.add(component_key)
                nonpolymer_components[resname or "UNKNOWN"] += 1
            continue
        protein_label_chains.add(label_chain)
        source_models.add(model_id)
        if author_chain:
            author_by_chain_model[(label_chain, model_id)].add(author_chain)
        position = (label_id, insertion)
        existing = atom_by_chain_model[(label_chain, model_id)].get(position)
        if existing is None:
            atom_by_chain_model[(label_chain, model_id)][position] = {
                "label_id": label_id,
                "auth_id": auth_id or label_id,
                "insertion": insertion,
                "resname": resname,
                "token": token,
                "token_kind": token_kind,
                "order": _numeric_order(label_id, index + 1),
            }
            counts[("evidence_occurrence", "atom", resname, token_kind)] += 1
            counts[("biological_residue", "coordinate_model", resname, token_kind)] += 1
        elif existing["resname"] != resname:
            existing["ambiguous_resname"] = True
        if (_clean(atom_name) or "").upper() != "CA":
            continue
        try:
            coord = [float(xs[index]), float(ys[index]), float(zs[index])]
        except ValueError:
            coord = [float("nan")] * 3
        occupancy_text = _clean(occupancies[index])
        try:
            occupancy = float(occupancy_text) if occupancy_text is not None else None
        except ValueError:
            occupancy = None
        ca_candidates[(label_chain, model_id, label_id, insertion)].append(
            {
                "altloc": _clean(altlocs[index]),
                "occupancy": occupancy,
                "row": index,
                "coord": coord,
            }
        )

    source_model_ids = sorted(source_models, key=lambda value: _numeric_order(value, 0))
    source_model_count = len(source_model_ids)
    rows: list[dict[str, Any]] = []
    chain_model_sequences: dict[str, dict[str, str]] = defaultdict(dict)
    for label_chain in sorted(protein_label_chains):
        scheme_residues = sorted(scheme_by_chain.get(label_chain, {}).values(), key=lambda item: item["order"])
        present_models = sorted(
            {model for chain, model in atom_by_chain_model if chain == label_chain},
            key=lambda value: _numeric_order(value, 0),
        )
        emitted_models = present_models or ["1"]
        for model_id in emitted_models:
            atom_residues = sorted(
                atom_by_chain_model.get((label_chain, model_id), {}).values(), key=lambda item: item["order"]
            )
            model_polymer_sequence = "".join(item["token"] for item in atom_residues)
            alignment_residues = scheme_residues or atom_residues
            atom_positions = {(item["label_id"], item["insertion"]) for item in atom_residues}
            missing_residue_mask = [
                (item["label_id"], item["insertion"]) not in atom_positions for item in alignment_residues
            ]
            selected_altlocs: list[str | None] = []
            selected_coords: list[list[float] | None] = []
            selected_occupancies: list[float | None] = []
            missing_calpha_mask: list[bool] = []
            multi_altloc_positions = 0
            ambiguous_altloc_positions = 0
            deterministic_ties = 0
            altloc_outcomes: Counter[str] = Counter()
            for residue in alignment_residues:
                candidates = ca_candidates.get((label_chain, model_id, residue["label_id"], residue["insertion"]), [])
                if len(candidates) > 1:
                    multi_altloc_positions += 1
                selected, outcome, tied = _select_ca(candidates) if candidates else (None, "missing", False)
                altloc_outcomes[outcome] += 1
                ambiguous_altloc_positions += int(outcome.startswith("ambiguous"))
                deterministic_ties += int(tied)
                selected_altlocs.append(selected["altloc"] if selected else None)
                selected_coords.append(selected["coord"] if selected else None)
                selected_occupancies.append(selected["occupancy"] if selected else None)
                missing_calpha_mask.append(selected is None)
            atom_sequence = "".join(
                residue["token"]
                for residue, missing_calpha in zip(alignment_residues, missing_calpha_mask, strict=True)
                if not missing_calpha
            )
            if present_models:
                chain_model_sequences[label_chain][model_id] = model_polymer_sequence
            atom_authors = author_by_chain_model.get((label_chain, model_id), set())
            scheme_author_set = scheme_authors.get(label_chain, set())
            author_evidence = atom_authors or scheme_author_set
            mapping_reasons = []
            if len(atom_authors) > 1:
                mapping_reasons.append("inconsistent_atom_label_to_author_mapping")
            if len(scheme_author_set) > 1:
                mapping_reasons.append("multiple_scheme_author_chain_ids")
            rows.append(
                {
                    "source_file": str(source),
                    "source_sha256": source_hash,
                    "pdb_id": pdb_id,
                    "chain_id": label_chain,
                    "label_asym_id": label_chain,
                    "auth_asym_id": next(iter(author_evidence)) if len(author_evidence) == 1 else None,
                    "entity_id": struct_entity.get(label_chain),
                    "polymer_type": entity_polymer_types.get(struct_entity.get(label_chain, "")),
                    "entity_poly_sequence": entity_poly_sequences.get(struct_entity.get(label_chain, "")),
                    "scheme_author_chain_id": (next(iter(scheme_author_set)) if len(scheme_author_set) == 1 else None),
                    "chain_mapping_status": "ambiguous" if mapping_reasons else "unambiguous",
                    "chain_mapping_ambiguity_reason": ";".join(mapping_reasons) or None,
                    "model_id": model_id,
                    "coordinate_model_present": bool(present_models),
                    "synthetic_model_row": not bool(present_models),
                    "experimental_method": method,
                    "residue_mapping_version": RESIDUE_MAPPING_VERSION,
                    "residue_mappings": json.dumps(dict(sorted(DEFAULT_RESIDUE_MAPPINGS.items()))),
                    "is_nmr": is_nmr,
                    "chain_count": len(protein_label_chains),
                    "model_count": source_model_count,
                    "source_model_ids": json.dumps(source_model_ids),
                    "seqres_sequence": "".join(item["token"] for item in scheme_residues) or None,
                    "scheme_sequence": "".join(item["token"] for item in scheme_residues) or None,
                    "atom_sequence": atom_sequence or None,
                    "coordinate_model_polymer_sequence": model_polymer_sequence or None,
                    "scheme_length": len(scheme_residues),
                    "atom_calpha_length": sum(not missing for missing in missing_calpha_mask),
                    "missing_residue_count": sum(missing_residue_mask) if present_models else None,
                    "missing_calpha_count": sum(missing_calpha_mask) if present_models else None,
                    "model_specific_residue_omission_count": (sum(missing_residue_mask) if present_models else None),
                    "label_sequence_ids": json.dumps([item["label_id"] for item in alignment_residues]),
                    "auth_sequence_ids": json.dumps([item["auth_id"] for item in alignment_residues]),
                    "insertion_codes": json.dumps([item["insertion"] for item in alignment_residues]),
                    "residue_names": json.dumps([item["resname"] for item in alignment_residues]),
                    "residue_tokens": json.dumps([item["token"] for item in alignment_residues]),
                    "atom_label_sequence_ids": json.dumps([item["label_id"] for item in atom_residues]),
                    "atom_auth_sequence_ids": json.dumps([item["auth_id"] for item in atom_residues]),
                    "atom_insertion_codes": json.dumps([item["insertion"] for item in atom_residues]),
                    "missing_residue_mask": json.dumps(missing_residue_mask) if present_models else None,
                    "missing_calpha_mask": json.dumps(missing_calpha_mask) if present_models else None,
                    "selected_altlocs": json.dumps(selected_altlocs),
                    "selected_occupancies": json.dumps(selected_occupancies),
                    "selected_calpha_coordinates": json.dumps(selected_coords),
                    "multiple_altloc_candidate_position_count": multi_altloc_positions,
                    "ambiguous_altloc_position_count": ambiguous_altloc_positions,
                    "deterministic_altloc_tie_count": deterministic_ties,
                    "altloc_outcome_counts": json.dumps(dict(sorted(altloc_outcomes.items()))),
                    "unknown_residue_counts": json.dumps(
                        dict(Counter(item["resname"] for item in alignment_residues if item["token_kind"] == "unknown"))
                    ),
                    "modified_residue_counts": json.dumps(
                        dict(
                            Counter(item["resname"] for item in alignment_residues if item["token_kind"] == "modified")
                        )
                    ),
                    "nonpolymer_component_counts": json.dumps(dict(sorted(nonpolymer_components.items()))),
                    "water_count": sum(
                        count
                        for component, count in nonpolymer_components.items()
                        if _component_class(component) == "water"
                    ),
                    "ion_count": sum(
                        count
                        for component, count in nonpolymer_components.items()
                        if _component_class(component) == "ion"
                    ),
                    "ligand_count": sum(
                        count
                        for component, count in nonpolymer_components.items()
                        if _component_class(component) == "ligand"
                    ),
                }
            )

    source_global_consistent = source_model_count < 2
    if source_model_count >= 2:
        source_global_consistent = all(
            set(model_sequences) == set(source_model_ids) and len(set(model_sequences.values())) == 1
            for model_sequences in chain_model_sequences.values()
        )
    for row in rows:
        model_sequences = chain_model_sequences.get(row["label_asym_id"], {})
        row["sequence_consistent_across_models"] = len(set(model_sequences.values())) <= 1
        row["chain_sequence_consistent_across_models"] = len(set(model_sequences.values())) <= 1
        row["source_sequence_consistent_across_models"] = source_global_consistent
        row["chain_missing_from_models"] = json.dumps(
            sorted(set(source_model_ids) - set(model_sequences), key=lambda value: _numeric_order(value, 0))
        )

    if sha256_file(source) != source_hash:
        raise RuntimeError(f"Raw structure changed during inspection: {source}")
    return rows, counts
