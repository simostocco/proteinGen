#!/usr/bin/env python3
"""Query the RCSB Search API v2 for a reproducible pilot PDB ID list."""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from protein_distance_diffusion.utils.io import write_json

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DEFAULT_PAGE_SIZE = 100


class RcsbQueryError(RuntimeError):
    """Raised when the RCSB Search API query cannot produce usable results."""


def build_query(*, start: int, rows: int) -> dict[str, Any]:
    """Build the RCSB Search API v2 query for pilot protein entities.

    Args:
        start: Zero-based result offset.
        rows: Page size.

    Returns:
        JSON-serializable RCSB query dictionary.
    """
    terminal_nodes: list[dict[str, Any]] = [
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "entity_poly.rcsb_entity_polymer_type",
                "operator": "exact_match",
                "value": "Protein",
            },
        },
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "exptl.method",
                "operator": "in",
                "value": ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"],
            },
        },
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.structure_determination_methodology",
                "operator": "exact_match",
                "value": "experimental",
            },
        },
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "entity_poly.rcsb_sample_sequence_length",
                "operator": "less_or_equal",
                "value": 500,
            },
        },
    ]
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": terminal_nodes,
        },
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": start, "rows": rows},
            "results_content_type": ["experimental"],
        },
    }


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    retries: int = 3,
    backoff_seconds: float = 1.0,
) -> dict[str, Any]:
    """POST JSON to an HTTP endpoint with retry handling.

    Args:
        url: Endpoint URL.
        payload: JSON payload.
        retries: Number of attempts for transient URL/HTTP failures.
        backoff_seconds: Base exponential-backoff sleep.

    Returns:
        Parsed JSON response.

    Raises:
        RcsbQueryError: If all attempts fail or the response is invalid JSON.
    """
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 official RCSB endpoint
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            last_error = RcsbQueryError(f"HTTP {exc.code} {exc.reason}: {body_text}")
            if 400 <= exc.code < 500 and exc.code != 429:
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        if attempt < retries - 1:
            time.sleep(backoff_seconds * (2**attempt))
    raise RcsbQueryError(f"RCSB query failed after {retries} attempt(s): {last_error}")


def entry_id_from_polymer_entity(identifier: str) -> str:
    """Convert an RCSB polymer-entity identifier into a PDB entry ID.

    Args:
        identifier: Polymer entity identifier, commonly `1ABC_1`.

    Returns:
        Uppercase four-character PDB entry ID.
    """
    return identifier.split("_", 1)[0].upper()


def fetch_candidate_entry_ids(
    *,
    candidate_limit: int,
    page_size: int = DEFAULT_PAGE_SIZE,
    post_fn=post_json,
) -> tuple[list[str], dict[str, Any]]:
    """Fetch unique candidate entry IDs from polymer-entity RCSB search results.

    Args:
        candidate_limit: Desired number of unique entry candidates.
        page_size: RCSB pagination page size.
        post_fn: Injectable HTTP function for tests.

    Returns:
        Tuple of unique PDB IDs and the first query JSON.

    Raises:
        ValueError: If limits are invalid.
        RcsbQueryError: If no results are returned.
    """
    if candidate_limit <= 0:
        raise ValueError("candidate_limit must be positive")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    start = 0
    candidates: list[str] = []
    seen: set[str] = set()
    first_query = build_query(start=0, rows=page_size)
    total_count: int | None = None
    while len(candidates) < candidate_limit:
        query = build_query(start=start, rows=page_size)
        response = post_fn(RCSB_SEARCH_URL, query)
        result_set = response.get("result_set")
        if not isinstance(result_set, list):
            raise RcsbQueryError("RCSB response is missing a list `result_set`")
        total_count_value = response.get("total_count")
        if isinstance(total_count_value, int):
            total_count = total_count_value
        if not result_set:
            break
        for item in result_set:
            identifier = item.get("identifier") if isinstance(item, dict) else None
            if not isinstance(identifier, str):
                continue
            entry_id = entry_id_from_polymer_entity(identifier)
            if entry_id not in seen:
                seen.add(entry_id)
                candidates.append(entry_id)
                if len(candidates) >= candidate_limit:
                    break
        start += page_size
        if total_count is not None and start >= total_count:
            break
    if not candidates:
        raise RcsbQueryError("RCSB query returned no candidate PDB entries")
    return sorted(candidates), first_query


def select_entries(candidates: list[str], *, num_entries: int, seed: int) -> list[str]:
    """Select a deterministic seeded subset from candidate PDB IDs.

    Args:
        candidates: Unique candidate PDB IDs.
        num_entries: Number of final entries requested.
        seed: Random seed.

    Returns:
        Deterministically selected PDB IDs sorted for stable output.

    Raises:
        ValueError: If `num_entries` is invalid or exceeds candidate count.
    """
    if num_entries <= 0:
        raise ValueError("num_entries must be positive")
    if num_entries > len(candidates):
        raise ValueError(f"Requested {num_entries} entries but only {len(candidates)} unique candidates were retrieved")
    rng = random.Random(seed)
    selected = rng.sample(candidates, k=num_entries)
    return sorted(selected)


def write_outputs(
    *,
    output: str | Path,
    selected_entries: list[str],
    candidates: list[str],
    query_json: dict[str, Any],
    seed: int,
) -> Path:
    """Write selected PDB IDs and metadata.

    Args:
        output: Text file destination.
        selected_entries: Final selected PDB IDs.
        candidates: Unique candidate PDB IDs before sampling.
        query_json: Exact first-page query JSON.
        seed: Sampling seed.

    Returns:
        Metadata JSON path.
    """
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(selected_entries) + "\n")
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    write_json(
        metadata_path,
        {
            "retrieval_date_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 Python 3.10 test env
            "seed": seed,
            "candidate_count": len(candidates),
            "selected_count": len(selected_entries),
            "query_url": RCSB_SEARCH_URL,
            "query_json": query_json,
            "selected_entries": selected_entries,
        },
    )
    return metadata_path


def main() -> None:
    """Run the RCSB pilot query CLI."""
    parser = argparse.ArgumentParser(description="Query RCSB Search API v2 for pilot protein PDB IDs.")
    parser.add_argument("--output", required=True, type=Path, help="Output text file, one PDB ID per line.")
    parser.add_argument("--num-entries", type=int, default=100, help="Number of final PDB entries to write.")
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=1000,
        help="Number of unique candidate entries to retrieve before seeded selection.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic candidate selection.")
    args = parser.parse_args()
    if args.candidate_limit < args.num_entries:
        raise ValueError("--candidate-limit must be greater than or equal to --num-entries")
    candidates, query_json = fetch_candidate_entry_ids(candidate_limit=args.candidate_limit)
    selected = select_entries(candidates, num_entries=args.num_entries, seed=args.seed)
    metadata = write_outputs(
        output=args.output,
        selected_entries=selected,
        candidates=candidates,
        query_json=query_json,
        seed=args.seed,
    )
    print(f"Wrote {len(selected)} PDB IDs to {args.output}")
    print(f"Wrote query metadata to {metadata}")


if __name__ == "__main__":
    main()
