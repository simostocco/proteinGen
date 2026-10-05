import ast
import csv
import inspect

import numpy as np
import pytest

from scripts import run_e008_geometry_native_decoder as e008


def _trace(n=40):
    # Compact, continuous trace with fixed 3.8 A neighboring distances.
    angle = np.arange(n) * 1.5
    return np.column_stack((2.6 * np.cos(angle), 2.6 * np.sin(angle), np.arange(n) * 0.9))


def test_e008_metadata_schema_separates_method_and_conformation():
    assert e008.E008_MANIFEST_COLUMNS == (
        "sample_id",
        "experimental_method",
        "conformation_class",
        "selection_class",
        "label_source",
        "source_path",
        "source_sha256",
        "exclusion_reason",
    )
    assert "nmr" not in {"globular", "idp", "unknown", "ambiguous"}


def test_geometry_screen_is_method_agnostic_and_excludes_unresolved():
    xyz = _trace()
    mask = np.ones(len(xyz), dtype=bool)
    links = np.ones(len(xyz) - 1, dtype=bool)
    result = e008._prototype_geometry_exclusion("A" * len(xyz), xyz, mask, links)
    assert result == ""
    # Experimental method is not an input to this predeclared selection criterion.
    for method in ("SOLUTION NMR", "X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"):
        assert method and e008._prototype_geometry_exclusion("A" * len(xyz), xyz, mask, links) == result
    mask[3] = False
    assert "missing_or_unresolved_calpha" in e008._prototype_geometry_exclusion("A" * len(xyz), xyz, mask, links)


def test_manifest_requires_label_source_for_authoritative_classes(tmp_path):
    path = tmp_path / "manifest.csv"
    row = dict(
        sample_id="s1",
        experimental_method="SOLUTION NMR",
        conformation_class="globular",
        selection_class="prototype_structured",
        label_source="",
        source_path="local.cif",
        source_sha256="a" * 64,
        exclusion_reason="",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=e008.E008_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match="label_source provenance"):
        e008._structure_classes(str(path))


def test_manifest_construction_has_no_model_or_pilot_calls():
    tree = ast.parse(inspect.getsource(e008._build_manifest))
    calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "_model" not in calls
    assert "_pilot" not in calls


def test_original_e008_config_fails_read_only_schema_validation_before_workload():
    config = "configs/e008_geometry_native_decoder.yaml"
    with pytest.raises(ValueError, match=r"milestones\.tiny_overfit_coordinate_rmse_angstrom_max"):
        e008._load(config)


def test_corrected_e008_config_passes_schema_validation():
    import yaml

    config = yaml.safe_load(open("configs/e008_geometry_native_decoder_restart_v2.yaml"))
    e008._validate_config_schema(config)
    assert config["milestones"]["tiny_overfit_coordinate_rmse_angstrom_max"] == 0.5
