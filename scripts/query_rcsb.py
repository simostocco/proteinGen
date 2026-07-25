#!/usr/bin/env python3
"""Query the RCSB Search API v2 for a reproducible pilot PDB ID list."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from protein_distance_diffusion.utils.io import write_json

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DEFAULT_PAGE_SIZE = 1000
MAX_PAGE_SIZE = 10_000


class RcsbQueryError(RuntimeError):
    """Raised when the RCSB Search API query cannot produce usable results."""


class RcsbQueryInterrupted(RuntimeError):
    """Raised after a query interruption has been checkpointed."""

    def __init__(self, *, state_path: Path, processed_hit_count: int, resume_command: str) -> None:
        self.state_path = state_path
        self.processed_hit_count = processed_hit_count
        self.resume_command = resume_command
        super().__init__(f"RCSB query interrupted after {processed_hit_count} polymer-entity hits")


def build_query(*, start: int, rows: int, release_date_cutoff: str | None = None) -> dict[str, Any]:
    """Build the RCSB Search API v2 query for pilot protein entities.

    Args:
        start: Zero-based result offset.
        rows: Page size.
        release_date_cutoff: Optional inclusive initial release-date cutoff, as YYYY-MM-DD.

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
    if release_date_cutoff is not None:
        terminal_nodes.append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_accession_info.initial_release_date",
                    "operator": "less_or_equal",
                    "value": release_date_cutoff,
                },
            }
        )
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
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """POST JSON to an HTTP endpoint with retry handling.

    Args:
        url: Endpoint URL.
        payload: JSON payload.
        retries: Number of attempts for transient URL/HTTP failures.
        backoff_seconds: Base exponential-backoff sleep.
        timeout_seconds: HTTP timeout per attempt.

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
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 official RCSB endpoint
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


def sha256_file(path: str | Path) -> str:
    """Compute the SHA256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_query_hash(query_json: dict[str, Any]) -> str:
    """Hash the resume-compatible query fingerprint."""
    payload = json.dumps(_progress_query_fingerprint(query_json), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    """Atomically write UTF-8 text."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    tmp.write_text(text)
    tmp.replace(dst)


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically write a JSON object."""
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_entry_ids(response: dict[str, Any]) -> list[str]:
    """Parse polymer-entity search hits into unique PDB entry IDs preserving order."""
    result_set = response.get("result_set")
    if not isinstance(result_set, list):
        raise RcsbQueryError("RCSB response is missing a list `result_set`")
    entries: list[str] = []
    seen: set[str] = set()
    for item in result_set:
        identifier = item.get("identifier") if isinstance(item, dict) else None
        if not isinstance(identifier, str):
            continue
        entry_id = entry_id_from_polymer_entity(identifier)
        if entry_id not in seen:
            seen.add(entry_id)
            entries.append(entry_id)
    return entries


def total_count_from_response(response: dict[str, Any]) -> int:
    """Extract `total_count` from an RCSB response."""
    total_count = response.get("total_count")
    if not isinstance(total_count, int):
        raise RcsbQueryError("RCSB response is missing integer `total_count`")
    return total_count


def fetch_total_count(
    *,
    page_size: int = 1,
    release_date_cutoff: str | None = None,
    post_fn=post_json,
) -> tuple[int, dict[str, Any]]:
    """Fetch only the total polymer-entity hit count."""
    query = build_query(start=0, rows=page_size, release_date_cutoff=release_date_cutoff)
    response = post_fn(RCSB_SEARCH_URL, query)
    return total_count_from_response(response), query


def _progress_query_fingerprint(query_json: dict[str, Any]) -> dict[str, Any]:
    """Return the query shape relevant for resume compatibility."""
    payload = json.loads(json.dumps(query_json))
    payload.get("request_options", {}).pop("paginate", None)
    return payload


def _write_progress(
    path: Path,
    *,
    query_json: dict[str, Any],
    page_size: int,
    next_start: int,
    total_count: int | None,
    processed_hit_count: int,
    entry_ids: list[str],
) -> None:
    atomic_write_json(
        path,
        {
            "query_hash": stable_query_hash(query_json),
            "query_fingerprint": _progress_query_fingerprint(query_json),
            "page_size": page_size,
            "next_start": next_start,
            "total_count": total_count,
            "total_polymer_entity_hits": total_count,
            "processed_hit_count": processed_hit_count,
            "entry_ids": entry_ids,
            "unique_pdb_ids_collected": entry_ids,
            "unique_pdb_entry_count": len(entry_ids),
            "retrieval_timestamp_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 Python 3.10 test env
        },
    )


def _load_progress(path: Path, *, query_json: dict[str, Any], page_size: int) -> tuple[int, int | None, int, list[str]]:
    data = json.loads(path.read_text())
    expected_hash = stable_query_hash(query_json)
    saved_hash = data.get("query_hash")
    if saved_hash is None and data.get("query_fingerprint") == _progress_query_fingerprint(query_json):
        saved_hash = expected_hash
    if saved_hash != expected_hash:
        raise RcsbQueryError(f"Progress file {path} was created for a different query")
    if int(data.get("page_size", 0)) != page_size:
        raise RcsbQueryError(f"Progress file {path} was created with a different page size")
    ids = data.get("entry_ids")
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise RcsbQueryError(f"Progress file {path} is missing entry_ids")
    ids = list(dict.fromkeys(ids))
    total = data.get("total_count", data.get("total_polymer_entity_hits"))
    processed = data.get("processed_hit_count", data.get("next_start", 0))
    return int(data.get("next_start", 0)), total if isinstance(total, int) else None, int(processed), ids


def _resume_command(
    *,
    output_path: Path,
    page_size: int,
    release_date_cutoff: str | None,
) -> str:
    parts = [
        "python",
        "scripts/query_rcsb.py",
        "--output",
        str(output_path),
        "--all-matches",
        "--page-size",
        str(page_size),
        "--resume",
    ]
    if release_date_cutoff is not None:
        parts.extend(["--release-date-cutoff", release_date_cutoff])
    return " ".join(parts)


def fetch_all_entry_ids(
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    progress_path: str | Path | None = None,
    output_path: str | Path | None = None,
    resume: bool = True,
    restart: bool = False,
    release_date_cutoff: str | None = None,
    retries: int = 3,
    backoff_seconds: float = 1.0,
    timeout_seconds: float = 30.0,
    show_progress: bool = True,
    post_fn=post_json,
) -> tuple[list[str], dict[str, Any], int]:
    """Fetch every matching polymer entity and return unique PDB entry IDs."""
    if page_size <= 0 or page_size > MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {MAX_PAGE_SIZE}")
    first_query = build_query(start=0, rows=page_size, release_date_cutoff=release_date_cutoff)
    output = Path(output_path) if output_path is not None else None
    progress = Path(progress_path) if progress_path is not None else None
    start = 0
    total_count: int | None = None
    processed_hit_count = 0
    entries: list[str] = []
    seen: set[str] = set()
    if progress is not None and restart and progress.exists():
        progress.unlink()
    if progress is not None and resume and progress.exists():
        start, total_count, processed_hit_count, entries = _load_progress(
            progress, query_json=first_query, page_size=page_size
        )
        seen = set(entries)
        if total_count is not None:
            print(f"total_count={total_count}", flush=True)
    progress_bar = tqdm(
        total=total_count,
        initial=processed_hit_count if total_count is not None else 0,
        desc="RCSB polymer entities",
        unit="hit",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    pages_completed = processed_hit_count // page_size
    try:
        while total_count is None or start < total_count:
            query = build_query(start=start, rows=page_size, release_date_cutoff=release_date_cutoff)
            response = post_fn(
                RCSB_SEARCH_URL,
                query,
                retries=retries,
                backoff_seconds=backoff_seconds,
                timeout_seconds=timeout_seconds,
            )
            result_set = response.get("result_set")
            if not isinstance(result_set, list):
                raise RcsbQueryError("RCSB response is missing a list `result_set`")
            page_entries = parse_entry_ids(response)
            if total_count is None:
                total_count = total_count_from_response(response)
                print(f"total_count={total_count}", flush=True)
                progress_bar.reset(total=total_count)
                progress_bar.update(processed_hit_count)
            else:
                total_count = total_count_from_response(response)
                progress_bar.total = total_count
            if not result_set:
                break
            for entry_id in page_entries:
                if entry_id not in seen:
                    seen.add(entry_id)
                    entries.append(entry_id)
            processed_hit_count += len(result_set)
            start += len(result_set)
            pages_completed += 1
            current_page = pages_completed
            progress_bar.update(len(result_set))
            progress_bar.set_postfix(
                {
                    "pages": pages_completed,
                    "current_page": current_page,
                    "unique_pdb": len(entries),
                }
            )
            if progress is not None:
                _write_progress(
                    progress,
                    query_json=first_query,
                    page_size=page_size,
                    next_start=start,
                    total_count=total_count,
                    processed_hit_count=processed_hit_count,
                    entry_ids=entries,
                )
    except KeyboardInterrupt as exc:
        if progress is not None:
            _write_progress(
                progress,
                query_json=first_query,
                page_size=page_size,
                next_start=start,
                total_count=total_count,
                processed_hit_count=processed_hit_count,
                entry_ids=entries,
            )
        progress_bar.close()
        resume_command = _resume_command(
            output_path=output or Path("data/full/pdb_ids.txt"),
            page_size=page_size,
            release_date_cutoff=release_date_cutoff,
        )
        raise RcsbQueryInterrupted(
            state_path=progress or Path("<no state path>"),
            processed_hit_count=processed_hit_count,
            resume_command=resume_command,
        ) from exc
    finally:
        progress_bar.close()
    if not entries:
        raise RcsbQueryError("RCSB query returned no PDB entries")
    return sorted(entries), first_query, int(total_count or 0)


def fetch_candidate_entry_ids(
    *,
    candidate_limit: int,
    page_size: int = DEFAULT_PAGE_SIZE,
    release_date_cutoff: str | None = None,
    post_fn=post_json,
) -> tuple[list[str], dict[str, Any], int]:
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
    first_query = build_query(start=0, rows=page_size, release_date_cutoff=release_date_cutoff)
    total_count: int | None = None
    while len(candidates) < candidate_limit:
        query = build_query(start=start, rows=page_size, release_date_cutoff=release_date_cutoff)
        response = post_fn(RCSB_SEARCH_URL, query)
        result_set = response.get("result_set")
        page_entries = parse_entry_ids(response)
        total_count = total_count_from_response(response)
        if not result_set:
            break
        for entry_id in page_entries:
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
    return sorted(candidates), first_query, int(total_count or 0)


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
    seed: int | None,
    total_polymer_entity_hits: int | None = None,
    page_size: int | None = None,
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
    atomic_write_text(output_path, "\n".join(selected_entries) + "\n")
    file_sha256 = sha256_file(output_path)
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    write_json(
        metadata_path,
        {
            "retrieval_date_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 Python 3.10 test env
            "seed": seed,
            "total_polymer_entity_hits": total_polymer_entity_hits,
            "candidate_count": len(candidates),
            "unique_pdb_entry_count": len(candidates),
            "selected_count": len(selected_entries),
            "pagination_size": page_size,
            "id_file_sha256": file_sha256,
            "query_url": RCSB_SEARCH_URL,
            "query_json": query_json,
            "selected_entries": selected_entries,
        },
    )
    return metadata_path


def write_count_only(
    *,
    output: str | Path,
    total_count: int,
    query_json: dict[str, Any],
    page_size: int,
) -> Path:
    """Write count-only output and metadata."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{total_count}\n")
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    write_json(
        metadata_path,
        {
            "retrieval_date_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 Python 3.10 test env
            "total_polymer_entity_hits": total_count,
            "unique_pdb_entry_count": None,
            "pagination_size": page_size,
            "id_file_sha256": sha256_file(output_path),
            "query_url": RCSB_SEARCH_URL,
            "query_json": query_json,
        },
    )
    return metadata_path


def main() -> None:
    """Run the RCSB pilot query CLI."""
    parser = argparse.ArgumentParser(description="Query RCSB Search API v2 for filtered protein PDB IDs.")
    parser.add_argument("--output", required=True, type=Path, help="Output text file, one PDB ID per line.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--num-entries", type=int, default=None, help="Number of final PDB entries to write.")
    mode.add_argument("--all-matches", action="store_true", help="Write all matching unique PDB entry IDs.")
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only query and save total_count; do not retrieve IDs.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=1000,
        help="Number of unique candidate entries to retrieve before seeded selection.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help=f"RCSB page size. --all-matches may use up to {MAX_PAGE_SIZE}.",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=None,
        help="Deprecated alias for the --all-matches state path. Defaults to OUTPUT.query_state.json.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume a compatible --all-matches state file. Enabled by default.",
    )
    parser.add_argument("--restart", action="store_true", help="Discard previous --all-matches query state.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic candidate selection.")
    parser.add_argument("--retries", type=int, default=3, help="HTTP retry attempts per RCSB page.")
    parser.add_argument("--backoff-seconds", type=float, default=1.0, help="Base exponential-backoff delay.")
    parser.add_argument("--timeout-seconds", type=float, default=30.0, help="HTTP timeout per request.")
    parser.add_argument(
        "--release-date-cutoff",
        default=None,
        help="Optional inclusive initial release-date cutoff for reproducibility, e.g. 2026-07-24.",
    )
    args = parser.parse_args()
    num_entries = 100 if args.num_entries is None and not args.all_matches and not args.count_only else args.num_entries
    if args.count_only:
        total_count, query_json = fetch_total_count(
            page_size=1,
            release_date_cutoff=args.release_date_cutoff,
        )
        metadata = write_count_only(
            output=args.output,
            total_count=total_count,
            query_json=query_json,
            page_size=1,
        )
        print(f"total_count={total_count}")
        print(f"Wrote count metadata to {metadata}")
        return
    if args.all_matches:
        progress = args.progress or args.output.with_suffix(args.output.suffix + ".query_state.json")
        try:
            entries, query_json, total_count = fetch_all_entry_ids(
                page_size=args.page_size,
                progress_path=progress,
                output_path=args.output,
                resume=args.resume,
                restart=args.restart,
                release_date_cutoff=args.release_date_cutoff,
                retries=args.retries,
                backoff_seconds=args.backoff_seconds,
                timeout_seconds=args.timeout_seconds,
            )
        except RcsbQueryInterrupted as exc:
            print(f"Interrupted after {exc.processed_hit_count} polymer-entity hits.", file=sys.stderr)
            print(f"State saved to {exc.state_path}", file=sys.stderr)
            print(f"Resume with: {exc.resume_command}", file=sys.stderr)
            raise SystemExit(130) from None
        metadata = write_outputs(
            output=args.output,
            selected_entries=entries,
            candidates=entries,
            query_json=query_json,
            seed=None,
            total_polymer_entity_hits=total_count,
            page_size=args.page_size,
        )
        print(f"Wrote {len(entries)} unique PDB IDs to {args.output}")
        print(f"Wrote query metadata to {metadata}")
        print(f"Progress checkpoint: {progress}")
        return
    if num_entries is None:
        raise ValueError("--num-entries is required unless --all-matches or --count-only is used")
    if args.candidate_limit < num_entries:
        raise ValueError("--candidate-limit must be greater than or equal to --num-entries")
    candidates, query_json, total_count = fetch_candidate_entry_ids(
        candidate_limit=args.candidate_limit,
        page_size=args.page_size,
        release_date_cutoff=args.release_date_cutoff,
    )
    selected = select_entries(candidates, num_entries=num_entries, seed=args.seed)
    metadata = write_outputs(
        output=args.output,
        selected_entries=selected,
        candidates=candidates,
        query_json=query_json,
        seed=args.seed,
        total_polymer_entity_hits=total_count,
        page_size=args.page_size,
    )
    print(f"Wrote {len(selected)} PDB IDs to {args.output}")
    print(f"Wrote query metadata to {metadata}")


if __name__ == "__main__":
    main()
