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
    assert attributes["exptl.method"]["value"] == ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY", "SOLUTION NMR"]
    assert attributes["rcsb_entry_info.structure_determination_methodology"]["value"] == "experimental"
    assert attributes["entity_poly.rcsb_sample_sequence_length"]["operator"] == "less_or_equal"
    assert attributes["entity_poly.rcsb_sample_sequence_length"]["value"] == 500


def test_build_query_supports_release_date_cutoff() -> None:
    """A fixed release-date cutoff can be added for reproducibility."""
    module = _load_query_module()
    query = module.build_query(start=0, rows=25, release_date_cutoff="2026-07-24")
    nodes = query["query"]["nodes"]
    attributes = {node["parameters"]["attribute"]: node["parameters"] for node in nodes}
    assert attributes["rcsb_accession_info.initial_release_date"] == {
        "attribute": "rcsb_accession_info.initial_release_date",
        "operator": "less_or_equal",
        "value": "2026-07-24",
    }


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

    def fake_post(url, payload, **kwargs):
        del kwargs
        calls.append((url, payload))
        return pages.pop(0)

    candidates, query, total_count = module.fetch_candidate_entry_ids(candidate_limit=4, page_size=3, post_fn=fake_post)
    assert candidates == ["1ABC", "2DEF", "3GHI", "4JKL"]
    assert total_count == 5
    assert query["request_options"]["paginate"] == {"start": 0, "rows": 3}
    assert calls[1][1]["request_options"]["paginate"] == {"start": 3, "rows": 3}


def test_fetch_total_count_does_not_page_through_ids() -> None:
    """Count-only mode only needs one request."""
    module = _load_query_module()
    calls = []

    def fake_post(url, payload, **kwargs):
        del kwargs
        calls.append((url, payload))
        return {"total_count": 123, "result_set": [{"identifier": "1ABC_1"}]}

    total_count, query = module.fetch_total_count(post_fn=fake_post)
    assert total_count == 123
    assert query["request_options"]["paginate"] == {"start": 0, "rows": 1}
    assert len(calls) == 1


def test_fetch_all_matches_saves_progress_and_can_resume(tmp_path: Path) -> None:
    """All-match pagination checkpoints each page and resumes from the saved offset."""
    module = _load_query_module()
    progress = tmp_path / "pdb_ids.txt.query_state.json"
    pages = [
        {
            "total_count": 8,
            "result_set": [
                {"identifier": "1ABC_1"},
                {"identifier": "1ABC_2"},
                {"identifier": "2DEF_1"},
            ],
        },
        {
            "total_count": 8,
            "result_set": [
                {"identifier": "2DEF_2"},
                {"identifier": "3GHI_1"},
                {"identifier": "4JKL_1"},
            ],
        },
    ]

    def first_post(url, payload, **kwargs):
        del url
        del kwargs
        if not pages:
            raise KeyboardInterrupt
        return pages.pop(0)

    with pytest.raises(module.RcsbQueryInterrupted) as exc_info:
        module.fetch_all_entry_ids(
            page_size=3,
            progress_path=progress,
            output_path=tmp_path / "pdb_ids.txt",
            post_fn=first_post,
            show_progress=False,
        )
    state = json.loads(progress.read_text())
    assert state["next_start"] == 6
    assert state["total_count"] == 8
    assert state["processed_hit_count"] == 6
    assert state["unique_pdb_entry_count"] == 4
    assert state["entry_ids"] == ["1ABC", "2DEF", "3GHI", "4JKL"]
    assert state["unique_pdb_ids_collected"] == ["1ABC", "2DEF", "3GHI", "4JKL"]
    assert state["query_hash"] == module.stable_query_hash(module.build_query(start=0, rows=3))
    assert exc_info.value.processed_hit_count == 6

    resumed_calls = []

    def resumed_post(url, payload, **kwargs):
        del url, kwargs
        resumed_calls.append(payload["request_options"]["paginate"])
        return {
            "total_count": 8,
            "result_set": [
                {"identifier": "4JKL_2"},
                {"identifier": "3GHI_1"},
                {"identifier": "5MNO_1"},
            ],
        }

    entries, _, total_count = module.fetch_all_entry_ids(
        page_size=3,
        progress_path=progress,
        output_path=tmp_path / "pdb_ids.txt",
        post_fn=resumed_post,
        show_progress=False,
    )
    assert resumed_calls == [{"start": 6, "rows": 3}]
    assert entries == ["1ABC", "2DEF", "3GHI", "4JKL", "5MNO"]
    assert total_count == 8


def test_fetch_all_matches_refuses_query_mismatch(tmp_path: Path) -> None:
    """Resume state is tied to the exact filter query hash."""
    module = _load_query_module()
    progress = tmp_path / "pdb_ids.txt.query_state.json"
    old_query = module.build_query(start=0, rows=3, release_date_cutoff="2020-01-01")
    module._write_progress(
        progress,
        query_json=old_query,
        page_size=3,
        next_start=3,
        total_count=9,
        processed_hit_count=3,
        entry_ids=["1ABC"],
    )

    with pytest.raises(module.RcsbQueryError, match="different query"):
        module.fetch_all_entry_ids(
            page_size=3,
            progress_path=progress,
            release_date_cutoff="2026-07-24",
            show_progress=False,
            post_fn=lambda url, payload, **kwargs: {"total_count": 9, "result_set": []},
        )


def test_restart_discards_previous_query_state(tmp_path: Path) -> None:
    """Restart removes old state and begins from the first page."""
    module = _load_query_module()
    progress = tmp_path / "pdb_ids.txt.query_state.json"
    old_query = module.build_query(start=0, rows=3, release_date_cutoff="2020-01-01")
    module._write_progress(
        progress,
        query_json=old_query,
        page_size=3,
        next_start=3,
        total_count=3,
        processed_hit_count=3,
        entry_ids=["1ABC"],
    )

    def fake_post(url, payload, **kwargs):
        del url, kwargs
        assert payload["request_options"]["paginate"] == {"start": 0, "rows": 3}
        return {"total_count": 1, "result_set": [{"identifier": "2DEF_1"}]}

    entries, _, total_count = module.fetch_all_entry_ids(
        page_size=3,
        progress_path=progress,
        restart=True,
        release_date_cutoff="2026-07-24",
        show_progress=False,
        post_fn=fake_post,
    )
    assert entries == ["2DEF"]
    assert total_count == 1
    assert json.loads(progress.read_text())["entry_ids"] == ["2DEF"]


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

    def fake_post(url, payload, **kwargs):
        del kwargs
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
        total_polymer_entity_hits=9,
        page_size=100,
    )
    assert output.read_text().splitlines() == ["1ABC", "2DEF"]
    data = json.loads(metadata.read_text())
    assert data["seed"] == 7
    assert data["candidate_count"] == 3
    assert data["selected_count"] == 2
    assert data["total_polymer_entity_hits"] == 9
    assert data["unique_pdb_entry_count"] == 3
    assert data["pagination_size"] == 100
    assert data["id_file_sha256"] == module.sha256_file(output)
    assert data["query_json"] == {"query": {"example": True}}


@pytest.mark.skipif(
    os.environ.get("RUN_RCSB_INTEGRATION_TESTS") != "1",
    reason="Set RUN_RCSB_INTEGRATION_TESTS=1 to query the real RCSB Search API",
)
def test_real_rcsb_smoke_query_candidate_limit_10() -> None:
    """Optional real API smoke test for the official query attributes."""
    module = _load_query_module()
    candidates, query, total_count = module.fetch_candidate_entry_ids(candidate_limit=10, page_size=10)
    assert len(candidates) == 10
    assert total_count >= 10
    assert query["return_type"] == "polymer_entity"
