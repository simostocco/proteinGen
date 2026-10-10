import pytest
import torch

from protein_distance_diffusion.models.e015_multiscale_parity_denoiser import (
    E015MultiscaleParityDenoiser,
    graph,
    local_geometry,
    stable_knn,
)


def rotation():
    torch.manual_seed(123)
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    q[:, 0] *= torch.det(q)
    return q


@pytest.mark.parametrize("kind", ["random", "symmetric", "equal", "near_tied"])
def test_stable_knn_proper_rotations_translations_and_ties(kind):
    torch.manual_seed(12)
    if kind == "random":
        x = torch.randn(32, 3)
    elif kind == "symmetric":
        x = torch.tensor([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0], [0, 0, 1.0], [0, 0, -1.0]])
    else:
        x = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], [-1.0, 0, 0], [0, -1.0, 0]])
        if kind == "near_tied":
            x[1, 0] += 1e-7
    q = rotation()
    for k in (1, 2, 3):
        expected = stable_knn(x, k)
        assert torch.equal(expected, stable_knn(x @ q.T, k))
        assert torch.equal(expected, stable_knn(x @ q.T + 0.75, k))
    if kind in ("equal", "near_tied"):
        assert stable_knn(x, 1)[0, 0] == 1


def test_geometry_islands_remain_fp32_under_cpu_autocast():
    x = torch.randn(20, 3).cumsum(0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        features, frame, reliability = local_geometry(x)
        i, j, edges, directions = graph(x, x * 0.8, 4, 3)
    assert all(z.dtype == torch.float32 for z in (features, frame, reliability, edges, directions))
    assert i.dtype == j.dtype == torch.long


def test_accepted_parameter_count_is_unchanged():
    model = E015MultiscaleParityDenoiser()
    assert sum(p.numel() for p in model.parameters()) == 33128139
