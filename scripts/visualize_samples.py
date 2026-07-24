#!/usr/bin/env python3
"""Visualize processed protein distance-map samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import seaborn as sns  # noqa: E402
except ImportError:
    sns = None

from protein_distance_diffusion.data.preprocess import load_manifest
from protein_distance_diffusion.evaluation.contact_maps import binary_contact_map, offdiagonal_pair_values


def _load_sample(path: str | Path) -> dict[str, object]:
    """Load a processed `.npz` sample file."""
    data = np.load(path, allow_pickle=False)
    return {
        "sample_id": str(data["sample_id"]),
        "pdb_id": str(data["pdb_id"]),
        "chain_id": str(data["chain_id"]),
        "sequence": str(data["sequence"]),
        "ca_coordinates": data["ca_coordinates"].astype(np.float32),
        "distance_matrix": data["distance_matrix"].astype(np.float32),
        "metadata": json.loads(str(data["metadata"])),
    }


def select_rows(
    frame: pd.DataFrame,
    *,
    num_samples: int,
    seed: int,
    pdb_id: str | None,
    chain_id: str | None,
    split: str | None,
    min_length: int | None,
    max_length: int | None,
) -> pd.DataFrame:
    """Select manifest rows for deterministic visualization."""
    selected = frame.copy()
    if pdb_id:
        selected = selected[selected["pdb_id"].astype(str).str.upper() == pdb_id.upper()]
    if chain_id:
        selected = selected[selected["chain_id"].astype(str) == chain_id]
    if split and "split" in selected:
        selected = selected[selected["split"].astype(str) == split]
    if min_length is not None:
        selected = selected[selected["length"] >= min_length]
    if max_length is not None:
        selected = selected[selected["length"] <= max_length]
    if selected.empty:
        raise ValueError("No manifest rows matched the visualization filters")
    return selected.sample(n=min(num_samples, len(selected)), random_state=seed).sort_values("sample_id")


def plot_sample(
    sample: dict[str, object],
    output: str | Path,
    *,
    contact_threshold: float,
    distance_vmax: float | None,
    exclude_near_diagonal: int,
) -> None:
    """Write a four-panel PNG for one processed sample."""
    coords = sample["ca_coordinates"]
    matrix = sample["distance_matrix"]
    metadata = sample["metadata"]
    assert isinstance(coords, np.ndarray)
    assert isinstance(matrix, np.ndarray)
    assert isinstance(metadata, dict)
    contacts = binary_contact_map(
        matrix,
        threshold_angstrom=contact_threshold,
        exclude_near_diagonal=exclude_near_diagonal,
    )
    pair_values = offdiagonal_pair_values(matrix, exclude_near_diagonal=exclude_near_diagonal)
    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.plot(coords[:, 0], coords[:, 1], coords[:, 2], marker="o", markersize=2, linewidth=1)
    ax1.set_title("C-alpha backbone trace")
    ax1.set_xlabel("x (A)")
    ax1.set_ylabel("y (A)")
    ax1.set_zlabel("z (A)")
    ax2 = fig.add_subplot(2, 2, 2)
    if sns is not None:
        sns.heatmap(matrix, ax=ax2, cmap="viridis", square=True, vmin=0.0, vmax=distance_vmax)
    else:
        im = ax2.imshow(matrix, cmap="viridis", vmin=0.0, vmax=distance_vmax)
        fig.colorbar(im, ax=ax2)
    ax2.set_title("Continuous distance matrix (A)")
    ax3 = fig.add_subplot(2, 2, 3)
    if sns is not None:
        sns.heatmap(contacts.astype(int), ax=ax3, cmap="mako", square=True, cbar=False)
    else:
        ax3.imshow(contacts.astype(int), cmap="gray_r")
    ax3.set_title(f"Binary contacts: D < {contact_threshold:g} A")
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.hist(pair_values, bins=30, color="#386cb0", edgecolor="white")
    ax4.set_xlabel("Pairwise C-alpha distance (A)")
    ax4.set_ylabel("Count")
    ax4.set_title("Non-diagonal distance histogram")
    resolution = metadata.get("resolution_angstrom")
    method = metadata.get("experimental_method", "unknown")
    fig.suptitle(
        f"{sample['pdb_id']} chain {sample['chain_id']} | N={matrix.shape[0]} | {method} | "
        f"resolution={resolution} | min={pair_values.min(initial=0):.2f} A | "
        f"max={pair_values.max(initial=0):.2f} A | contact={contact_threshold:g} A",
        fontsize=12,
    )
    fig.tight_layout()
    dst = Path(output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=140)
    plt.close(fig)


def main() -> None:
    """Run sample visualization."""
    parser = argparse.ArgumentParser(description="Visualize processed distance-map samples from a manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contact-threshold", type=float, default=8.0)
    parser.add_argument("--pdb-id", default=None)
    parser.add_argument("--chain-id", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--min-length", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--distance-vmax", type=float, default=None)
    parser.add_argument("--exclude-near-diagonal", type=int, default=0)
    args = parser.parse_args()
    frame = load_manifest(args.manifest)
    rows = select_rows(
        frame,
        num_samples=args.num_samples,
        seed=args.seed,
        pdb_id=args.pdb_id,
        chain_id=args.chain_id,
        split=args.split,
        min_length=args.min_length,
        max_length=args.max_length,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    links = []
    for row in rows.itertuples(index=False):
        sample = _load_sample(row.path)
        png = args.output_dir / f"{sample['sample_id']}.png"
        plot_sample(
            sample,
            png,
            contact_threshold=args.contact_threshold,
            distance_vmax=args.distance_vmax,
            exclude_near_diagonal=args.exclude_near_diagonal,
        )
        links.append(f'<li><a href="{png.name}">{sample["sample_id"]}</a></li>')
    (args.output_dir / "index.html").write_text("<html><body><ul>\n" + "\n".join(links) + "\n</ul></body></html>\n")
    print(f"Wrote {len(rows)} visualization(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
