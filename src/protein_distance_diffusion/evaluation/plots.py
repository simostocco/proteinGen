"""Plotting helpers for generated and real distance maps."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import seaborn as sns  # noqa: E402
except ImportError:
    sns = None


def save_heatmap(matrix, path: str | Path, *, title: str | None = None) -> None:
    """Save a distance-matrix heatmap.

    Args:
        matrix: Square array in angstrom.
        path: Output image path.
        title: Optional plot title.

    Returns:
        None.
    """
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    if sns is not None:
        sns.heatmap(matrix, ax=ax, cmap="viridis", cbar=True, square=True)
    else:
        im = ax.imshow(matrix, cmap="viridis")
        fig.colorbar(im, ax=ax)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(dst, dpi=120)
    plt.close(fig)
