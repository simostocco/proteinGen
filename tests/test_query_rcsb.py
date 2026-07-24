"""RCSB pilot query tests with mocked HTTP responses."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest


def _load_query_module():
    """Load `scripts/query_rcsb.py` as a test module."""
    script = Path(__file__).parents[1] / "scripts" / "query_rcsb.py"
    spec = importlib.util.spec_from_file_location("query_rcsb", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_query_contains_required_filters() -> None:
    """The query targets experimental protein polymer entities with length <= 500."""
    module = _load_query_module()
    query = module.build_query(start=0, rows=25)
    assert query["return_type"] == "polymer_entity"
    assert query["request_options"]["results_content_type"] == ["experimental"]
    assert query["request_options"]["paginate"] == {"start": 0, "rows": 25}
    assert "sort" not in query["request_options"]
    nodes = query["query"]["nodes"]
    attributes = {node["parameters"]["attribute"]: node["parameters"] for node in nodes}
    assert "rcsb_polymer_entity.type" not in attributes
    assert attributes["entity_poly.rcsb_entity_polymer_type"]["value"] == "Protein"
    assert attributes["exptl.method"]["value"] == ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"]
    assert attributes["rcsb_entry_info.structure_determination_methodology"]["value"] == "experimental"
    assert attributes["entity_poly.rcsb_sample_sequence_length"]["operator"] == "less_or_equal"
    assert attributes["entity_poly.rcsb_sample_sequence_length"]["value"] == 500


def test_fetch_candidates_paginates_and_deduplicates_entries() -> None:
    """Polymer entity hits are converted to unique PDB entry IDs across pages."""
    module = _load_query_module()
    calls = []
    pages = [
        {
            "total_count": 5,
            "result_set": [
                {"identifier": "1ABC_1"},
                {"identifier": "1ABC_2"},
                {"identifier": "2DEF_1"},
            ],
        },
        {
            "total_count": 5,
            "result_set": [
                {"identifier": "3GHI_1"},
                {"identifier": "4JKL_1"},
            ],
        },
    ]

    def fake_post(url, payload):
        calls.append((url, payload))
        return pages.pop(0)

    candidates, query = module.fetch_candidate_entry_ids(candidate_limit=4, page_size=3, post_fn=fake_post)
    assert candidates == ["1ABC", "2DEF", "3GHI", "4JKL"]
    assert query["request_options"]["paginate"] == {"start": 0, "rows": 3}
    assert calls[1][1]["request_options"]["paginate"] == {"start": 3, "rows": 3}


def test_seeded_selection_is_deterministic_and_not_prefix_biased() -> None:
    """Selection uses seeded sampling rather than taking the first IDs."""
    module = _load_query_module()
    candidates = ["1AAA", "2BBB", "3CCC", "4DDD", "5EEE"]
    selected_a = module.select_entries(candidates, num_entries=3, seed=11)
    selected_b = module.select_entries(candidates, num_entries=3, seed=11)
    assert selected_a == selected_b
    assert selected_a != candidates[:3]


def test_empty_response_raises_clear_error() -> None:
    """Empty RCSB responses fail clearly."""
    module = _load_query_module()

    def fake_post(url, payload):
        return {"total_count": 0, "result_set": []}

    with pytest.raises(module.RcsbQueryError, match="no candidate"):
        module.fetch_candidate_entry_ids(candidate_limit=5, post_fn=fake_post)


def test_write_outputs_and_metadata(tmp_path: Path) -> None:
    """The script writes one PDB ID per line and metadata JSON with query details."""
    module = _load_query_module()
    output = tmp_path / "pdb_ids.txt"
    metadata = module.write_outputs(
        output=output,
        selected_entries=["1ABC", "2DEF"],
        candidates=["1ABC", "2DEF", "3GHI"],
        query_json={"query": {"example": True}},
        seed=7,
    )
    assert output.read_text().splitlines() == ["1ABC", "2DEF"]
    data = json.loads(metadata.read_text())
    assert data["seed"] == 7
    assert data["candidate_count"] == 3
    assert data["selected_count"] == 2
    assert data["query_json"] == {"query": {"example": True}}


@pytest.mark.skipif(
    os.environ.get("RUN_RCSB_INTEGRATION_TESTS") != "1",
    reason="Set RUN_RCSB_INTEGRATION_TESTS=1 to query the real RCSB Search API",
)
def test_real_rcsb_smoke_query_candidate_limit_10() -> None:
    """Optional real API smoke test for the official query attributes."""
    module = _load_query_module()
    candidates, query = module.fetch_candidate_entry_ids(candidate_limit=10, page_size=10)
    assert len(candidates) == 10
    assert query["return_type"] == "polymer_entity"
