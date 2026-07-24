#!/usr/bin/env python3
"""Download requested PDB mmCIF files."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_distance_diffusion.data.download import download_mmcif_ids


def main() -> None:
    """Run the download CLI."""
    parser = argparse.ArgumentParser(description="Download selected PDB mmCIF files without bulk acquisition.")
    parser.add_argument("--ids-file", required=True, type=Path, help="Text/CSV file containing PDB identifiers.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Local mmCIF cache directory.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Download manifest path. Defaults to OUTPUT_DIR/../download_manifest.csv.",
    )
    parser.add_argument("--force", action="store_true", help="Redownload files even if cached.")
    parser.add_argument("--max-entries", type=int, default=None, help="Optional small development limit.")
    parser.add_argument("--delay-seconds", type=float, default=0.0, help="Optional polite delay between downloads.")
    parser.add_argument("--dry-run", action="store_true", help="Do not access the network.")
    args = parser.parse_args()
    ids = [line.split(",")[0].strip() for line in args.ids_file.read_text().splitlines() if line.strip()]
    manifest = download_mmcif_ids(
        ids,
        args.output_dir,
        force=args.force,
        max_entries=args.max_entries,
        dry_run=args.dry_run,
        delay_seconds=args.delay_seconds,
    )
    manifest_path = args.manifest or (args.output_dir.parent / "download_manifest.csv")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
