import torch

from protein_distance_diffusion.training.e015_replay import fingerprint


def test_fingerprint_is_order_independent_and_bit_sensitive():
    x = torch.tensor([1.0, 2.0])
    assert fingerprint({"x": x, "n": 2}) == fingerprint({"n": 2, "x": x.clone()})
    changed = x.clone()
    changed[0] = torch.nextafter(changed[0], changed.new_tensor(float("inf")))
    assert fingerprint(x) != fingerprint(changed)
    assert fingerprint(torch.tensor(1.0)) == fingerprint(torch.tensor(1.0))


def test_fingerprint_handles_strided_singleton_storage():
    strided = torch.arange(12, dtype=torch.float32)[6:7:6]
    assert fingerprint(strided) == fingerprint(torch.tensor([6.0]))
