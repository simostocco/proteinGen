"""Gemmi mmCIF parser regression tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from protein_distance_diffusion.data.mmcif_parser import _atom_site, parse_mmcif_file, parse_mmcif_file_with_rejections


def _write_poly_gly_mmcif(path: Path, *, length: int) -> None:
    """Write a synthetic contiguous poly-glycine mmCIF with C-alpha atoms."""
    rows = []
    atom_id = 1
    for resid in range(1, length + 1):
        x = float(resid)
        rows.append(f"ATOM {atom_id} C CA . GLY A 1 {resid} ? {x:.3f} 0.000 0.000 1.00 20.00 ? {resid} GLY A CA 1")
        atom_id += 1
    path.write_text(
        "\n".join(
            [
                "data_LONG",
                "#",
                "_exptl.method 'X-RAY DIFFRACTION'",
                "_refine.ls_d_res_high 1.50",
                "#",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.id",
                "_atom_site.type_symbol",
                "_atom_site.label_atom_id",
                "_atom_site.label_alt_id",
                "_atom_site.label_comp_id",
                "_atom_site.label_asym_id",
                "_atom_site.label_entity_id",
                "_atom_site.label_seq_id",
                "_atom_site.pdbx_PDB_ins_code",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                "_atom_site.occupancy",
                "_atom_site.B_iso_or_equiv",
                "_atom_site.pdbx_formal_charge",
                "_atom_site.auth_seq_id",
                "_atom_site.auth_comp_id",
                "_atom_site.auth_asym_id",
                "_atom_site.auth_atom_id",
                "_atom_site.pdbx_PDB_model_num",
                *rows,
                "#",
            ]
        )
        + "\n"
    )


def _write_dna_then_protein_mmcif(path: Path) -> None:
    """Write a minimized 10FZ-like file with a non-protein chain before a valid protein chain."""
    path.write_text(
        "\n".join(
            [
                "data_MIXD",
                "#",
                "_exptl.method 'ELECTRON MICROSCOPY'",
                "_em_3d_reconstruction.resolution 2.50",
                "#",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.id",
                "_atom_site.type_symbol",
                "_atom_site.label_atom_id",
                "_atom_site.label_alt_id",
                "_atom_site.label_comp_id",
                "_atom_site.label_asym_id",
                "_atom_site.label_entity_id",
                "_atom_site.label_seq_id",
                "_atom_site.pdbx_PDB_ins_code",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                "_atom_site.occupancy",
                "_atom_site.B_iso_or_equiv",
                "_atom_site.pdbx_formal_charge",
                "_atom_site.auth_seq_id",
                "_atom_site.auth_comp_id",
                "_atom_site.auth_asym_id",
                "_atom_site.auth_atom_id",
                "_atom_site.pdbx_PDB_model_num",
                "ATOM 1 P P . DA A 1 1 ? 0.000 0.000 0.000 1.00 20.00 ? 1 DA A P 1",
                "ATOM 2 P P . DC A 1 2 ? 1.000 0.000 0.000 1.00 20.00 ? 2 DC A P 1",
                "ATOM 3 C CA . GLY B 2 1 ? 0.000 0.000 0.000 1.00 20.00 ? 1 GLY B CA 1",
                "ATOM 4 C CA . ALA B 2 2 ? 3.000 4.000 0.000 1.00 20.00 ? 2 ALA B CA 1",
                "#",
            ]
        )
        + "\n"
    )


def _write_custom_mmcif(path: Path, rows: list[str], *, scheme_rows: list[str] | None = None) -> None:
    lines = [
        "data_CSTM",
        "#",
        "_exptl.method 'X-RAY DIFFRACTION'",
        "_refine.ls_d_res_high 1.50",
        "#",
    ]
    if scheme_rows is not None:
        lines.extend(
            [
                "loop_",
                "_pdbx_poly_seq_scheme.asym_id",
                "_pdbx_poly_seq_scheme.entity_id",
                "_pdbx_poly_seq_scheme.seq_id",
                "_pdbx_poly_seq_scheme.mon_id",
                "_pdbx_poly_seq_scheme.pdb_strand_id",
                "_pdbx_poly_seq_scheme.auth_seq_num",
                "_pdbx_poly_seq_scheme.pdb_ins_code",
                *scheme_rows,
                "#",
            ]
        )
    lines.extend(
        [
            "loop_",
            "_atom_site.group_PDB",
            "_atom_site.id",
            "_atom_site.type_symbol",
            "_atom_site.label_atom_id",
            "_atom_site.label_alt_id",
            "_atom_site.label_comp_id",
            "_atom_site.label_asym_id",
            "_atom_site.label_entity_id",
            "_atom_site.label_seq_id",
            "_atom_site.pdbx_PDB_ins_code",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "_atom_site.occupancy",
            "_atom_site.B_iso_or_equiv",
            "_atom_site.pdbx_formal_charge",
            "_atom_site.auth_seq_id",
            "_atom_site.auth_comp_id",
            "_atom_site.auth_asym_id",
            "_atom_site.auth_atom_id",
            "_atom_site.pdbx_PDB_model_num",
            *rows,
            "#",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _atom_row(
    atom_id: int,
    *,
    atom: str = "CA",
    altloc: str = ".",
    resname: str = "GLY",
    chain: str = "A",
    seq: int = 1,
    x: str | float = 0.0,
    occupancy: str | float = 1.0,
    group: str = "ATOM",
) -> str:
    element = atom[0]
    return (
        f"{group} {atom_id} {element} {atom} {altloc} {resname} {chain} 1 {seq} ? "
        f"{x} 0.000 0.000 {occupancy} 20.00 ? {seq} {resname} {chain} {atom} 1"
    )


def test_atom_site_adapter_reads_real_gemmi_table() -> None:
    """Real Gemmi tables are converted to bare atom_site column names with aligned rows."""
    import gemmi

    fixture = Path(__file__).parent / "fixtures" / "two_residue_xray.cif"
    block = gemmi.cif.read_file(str(fixture)).sole_block()
    site = _atom_site(block)
    assert "label_atom_id" in site
    assert "_atom_site.label_atom_id" not in site
    assert site["label_atom_id"][:2] == ["N", "CA"]
    assert site["pdbx_PDB_ins_code"][:2] == ["?", "?"]
    assert len({len(values) for values in site.values()}) == 1


@pytest.mark.skipif(importlib.util.find_spec("gemmi") is None, reason="Gemmi is not installed")
def test_mmcif_missing_terminal_calpha_with_other_atoms_is_rejected(tmp_path: Path) -> None:
    fixture = tmp_path / "missing_terminal.cif"
    _write_custom_mmcif(fixture, [_atom_row(1, seq=1), _atom_row(2, atom="N", seq=2)])
    samples, rejections = parse_mmcif_file_with_rejections(fixture, min_length=None)
    assert samples == []
    assert [(rejection.reason, rejection.chain_id) for rejection in rejections] == [("missing_calpha", "A")]
    assert "2" in rejections[0].message


@pytest.mark.skipif(importlib.util.find_spec("gemmi") is None, reason="Gemmi is not installed")
def test_mmcif_missing_internal_calpha_is_rejected(tmp_path: Path) -> None:
    fixture = tmp_path / "missing_internal.cif"
    _write_custom_mmcif(fixture, [_atom_row(1, seq=1), _atom_row(2, atom="N", seq=2), _atom_row(3, seq=3)])
    samples, rejections = parse_mmcif_file_with_rejections(fixture, min_length=None)
    assert samples == []
    assert rejections[0].reason == "missing_calpha"
    assert "2" in rejections[0].message


@pytest.mark.skipif(importlib.util.find_spec("gemmi") is None, reason="Gemmi is not installed")
def test_mmcif_polymer_scheme_detects_absent_terminal_residue(tmp_path: Path) -> None:
    fixture = tmp_path / "scheme_missing.cif"
    _write_custom_mmcif(
        fixture,
        [_atom_row(1, resname="ALA", seq=1)],
        scheme_rows=["A 1 1 ALA A 1 ?", "A 1 2 GLY A 2 ?"],
    )
    samples, rejections = parse_mmcif_file_with_rejections(fixture, min_length=None)
    assert samples == []
    assert rejections[0].reason == "missing_calpha"
    assert "2" in rejections[0].message


@pytest.mark.skipif(importlib.util.find_spec("gemmi") is None, reason="Gemmi is not installed")
def test_mmcif_valid_complete_chain_ligand_water_ignored_and_mse_mapped(tmp_path: Path) -> None:
    fixture = tmp_path / "valid_mse_ligand_water.cif"
    _write_custom_mmcif(
        fixture,
        [
            _atom_row(1, resname="MSE", seq=1, group="HETATM"),
            _atom_row(2, resname="ALA", seq=2),
            _atom_row(3, atom="O", resname="HOH", seq=900, group="HETATM"),
            _atom_row(4, atom="P", resname="ATP", seq=901, group="HETATM"),
        ],
    )
    samples = parse_mmcif_file(fixture, min_length=None, residue_mappings={"MSE": "MET"})
    assert len(samples) == 1
    assert samples[0].sequence == "MA"
    assert samples[0].metadata["original_chain_length"] == 2


@pytest.mark.skipif(importlib.util.find_spec("gemmi") is None, reason="Gemmi is not installed")
def test_mmcif_multiple_chains_keep_valid_chain_when_other_missing_ca(tmp_path: Path) -> None:
    fixture = tmp_path / "multi_chain_missing.cif"
    _write_custom_mmcif(
        fixture,
        [_atom_row(1, atom="N", chain="A", seq=1), _atom_row(2, chain="B", seq=1), _atom_row(3, chain="B", seq=2)],
    )
    samples, rejections = parse_mmcif_file_with_rejections(fixture, min_length=None)
    assert [sample.chain_id for sample in samples] == ["B"]
    assert [(rejection.chain_id, rejection.reason) for rejection in rejections] == [("A", "missing_calpha")]


@pytest.mark.skipif(importlib.util.find_spec("gemmi") is None, reason="Gemmi is not installed")
@pytest.mark.parametrize(
    ("rows", "expected_x"),
    [
        ([_atom_row(1, altloc="A", x=1.0, occupancy=0.8), _atom_row(2, altloc="B", x=2.0, occupancy=0.3)], 1.0),
        ([_atom_row(1, altloc="A", x=1.0, occupancy=0.3), _atom_row(2, altloc="B", x=2.0, occupancy=0.8)], 2.0),
        ([_atom_row(1, altloc="A", x=1.0, occupancy=0.5), _atom_row(2, altloc="B", x=2.0, occupancy=0.5)], 1.0),
        ([_atom_row(1, altloc=".", x=1.0, occupancy=0.1), _atom_row(2, altloc="A", x=2.0, occupancy=1.0)], 1.0),
        ([_atom_row(1, altloc="A", x=1.0, occupancy="?"), _atom_row(2, altloc="B", x=2.0, occupancy=0.1)], 2.0),
    ],
)
def test_mmcif_altloc_occupancy_selection_is_deterministic(tmp_path: Path, rows: list[str], expected_x: float) -> None:
    fixture = tmp_path / "altloc.cif"
    _write_custom_mmcif(fixture, rows)
    first = parse_mmcif_file(fixture, min_length=None)[0]
    second = parse_mmcif_file(fixture, min_length=None)[0]
    assert first.ca_coordinates[0, 0] == pytest.approx(expected_x)
    np.testing.assert_allclose(first.ca_coordinates, second.ca_coordinates)


@pytest.mark.skipif(importlib.util.find_spec("gemmi") is None, reason="Gemmi is not installed")
def test_mmcif_duplicate_identical_altloc_rejected(tmp_path: Path) -> None:
    fixture = tmp_path / "duplicate_altloc.cif"
    _write_custom_mmcif(
        fixture,
        [_atom_row(1, altloc="A", x=1.0, occupancy=0.5), _atom_row(2, altloc="A", x=2.0, occupancy=0.4)],
    )
    samples, rejections = parse_mmcif_file_with_rejections(fixture, min_length=None)
    assert samples == []
    assert rejections[0].reason == "duplicate_ca"


@pytest.mark.skipif(importlib.util.find_spec("gemmi") is None, reason="Gemmi is not installed")
def test_mmcif_nonfinite_coordinates_rejected(tmp_path: Path) -> None:
    fixture = tmp_path / "nonfinite.cif"
    _write_custom_mmcif(fixture, [_atom_row(1, x="nan")])
    samples, rejections = parse_mmcif_file_with_rejections(fixture, min_length=None)
    assert samples == []
    assert rejections[0].reason == "nonfinite_coordinates"


@pytest.mark.skipif(importlib.util.find_spec("gemmi") is None, reason="Gemmi is not installed")
def test_mmcif_missing_optional_metadata_does_not_crash() -> None:
    """Missing `_exptl.method` metadata is optional and must not crash Gemmi parsing."""
    fixture = Path(__file__).parent / "fixtures" / "two_residue_missing_metadata.cif"
    samples = parse_mmcif_file(fixture, min_length=None, max_length=500)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.sequence == "GA"
    assert sample.metadata["experimental_method"] is None
    assert sample.ca_coordinates.shape == (2, 3)
    assert np.isclose(np.linalg.norm(sample.ca_coordinates[0] - sample.ca_coordinates[1]), 5.0)


@pytest.mark.skipif(importlib.util.find_spec("gemmi") is None, reason="Gemmi is not installed")
def test_mmcif_min_length_none_disables_lower_bound() -> None:
    """`min_length=None` accepts short fixtures while a positive integer lower bound filters them."""
    fixture = Path(__file__).parent / "fixtures" / "two_residue_missing_metadata.cif"
    assert len(parse_mmcif_file(fixture, min_length=None, max_length=500)) == 1
    assert parse_mmcif_file(fixture, min_length=3, max_length=500) == []
    with pytest.raises(ValueError, match="min_length"):
        parse_mmcif_file(fixture, min_length=0, max_length=500)


@pytest.mark.skipif(importlib.util.find_spec("gemmi") is None, reason="Gemmi is not installed")
def test_mmcif_xray_metadata_and_resolution_filters() -> None:
    """X-ray method and resolution are extracted and method-specific thresholds apply."""
    fixture = Path(__file__).parent / "fixtures" / "two_residue_xray.cif"
    samples = parse_mmcif_file(
        fixture,
        min_length=None,
        allowed_methods=["X-RAY DIFFRACTION"],
        max_xray_resolution_angstrom=2.0,
    )
    assert len(samples) == 1
    assert samples[0].metadata["experimental_method"] == "X-RAY DIFFRACTION"
    assert samples[0].metadata["resolution_angstrom"] == 1.7
    assert (
        parse_mmcif_file(
            fixture,
            min_length=None,
            allowed_methods=["X-RAY DIFFRACTION"],
            max_xray_resolution_angstrom=1.0,
        )
        == []
    )


@pytest.mark.skipif(importlib.util.find_spec("gemmi") is None, reason="Gemmi is not installed")
def test_mmcif_nmr_rejection_and_optional_method_filter() -> None:
    """NMR is rejected by an exact allowed-method list and accepted when filtering is disabled."""
    fixture = Path(__file__).parent / "fixtures" / "two_residue_nmr.cif"
    assert (
        parse_mmcif_file(
            fixture,
            min_length=None,
            allowed_methods=["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"],
        )
        == []
    )
    samples = parse_mmcif_file(fixture, min_length=None, allowed_methods=None)
    assert len(samples) == 1
    assert samples[0].metadata["experimental_method"] == "SOLUTION NMR"
    assert samples[0].metadata["resolution_angstrom"] is None


@pytest.mark.skipif(importlib.util.find_spec("gemmi") is None, reason="Gemmi is not installed")
@pytest.mark.parametrize(("length", "accepted"), [(129, True), (500, True), (501, False)])
def test_mmcif_max_length_500_boundary(tmp_path: Path, length: int, accepted: bool) -> None:
    """The preprocessing parser accepts lengths 129 and 500, and rejects length 501."""
    fixture = tmp_path / f"poly_gly_{length}.cif"
    _write_poly_gly_mmcif(fixture, length=length)
    samples = parse_mmcif_file(
        fixture,
        min_length=None,
        max_length=500,
        allowed_methods=["X-RAY DIFFRACTION"],
    )
    if accepted:
        assert len(samples) == 1
        assert len(samples[0].sequence) == length
        assert samples[0].metadata["original_chain_length"] == length
    else:
        assert samples == []


@pytest.mark.skipif(importlib.util.find_spec("gemmi") is None, reason="Gemmi is not installed")
def test_mmcif_nonprotein_chain_does_not_discard_valid_chain(tmp_path: Path) -> None:
    """A non-protein chain before a valid protein chain is recorded but does not abort the entry."""
    fixture = tmp_path / "dna_then_protein.cif"
    _write_dna_then_protein_mmcif(fixture)
    samples, rejections = parse_mmcif_file_with_rejections(
        fixture,
        min_length=None,
        max_length=500,
        allowed_methods=["ELECTRON MICROSCOPY"],
        max_cryoem_resolution_angstrom=4.0,
    )
    assert [sample.chain_id for sample in samples] == ["B"]
    assert samples[0].sequence == "GA"
    assert [(rejection.chain_id, rejection.reason) for rejection in rejections] == [("A", "unsupported_residues")]


@pytest.mark.skipif(importlib.util.find_spec("gemmi") is None, reason="Gemmi is not installed")
def test_preprocess_records_max_length_rejection_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Preprocessing records a clear rejection reason for length > max_length."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_poly_gly_mmcif(raw_dir / "too_long.cif", length=501)
    summary_path = tmp_path / "processed" / "preprocess_summary.json"
    config_path = tmp_path / "preprocess.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"source_dir: {raw_dir}",
                f"samples_dir: {tmp_path / 'processed' / 'samples'}",
                f"manifest_path: {tmp_path / 'processed' / 'manifest.parquet'}",
                f"summary_path: {summary_path}",
                "backend: auto",
                "chain_id:",
                "min_length:",
                "max_length: 500",
                "allowed_methods:",
                "  - X-RAY DIFFRACTION",
                "max_xray_resolution_angstrom: 3.0",
                "max_cryoem_resolution_angstrom: 4.0",
                "residue_mappings:",
                "  MSE: MET",
            ]
        )
        + "\n"
    )
    script = Path(__file__).parents[1] / "scripts" / "preprocess_pdb.py"
    spec = importlib.util.spec_from_file_location("preprocess_pdb", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    monkeypatch.setattr("sys.argv", ["preprocess_pdb.py", "--config", str(config_path)])
    module.main()
    summary = json.loads(summary_path.read_text())
    assert summary["accepted_samples"] == 0
    assert summary["rejection_counts"] == {"length_above_maximum": 1}
    assert "max_length=500" in summary["rejections"][0]["message"]
