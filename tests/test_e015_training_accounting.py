import numpy as np

from protein_distance_diffusion.evaluation.e015_evaluation import paired_scalar_comparison
from protein_distance_diffusion.training.e015_trainer import identity_schedule, plan_update


def test_complete_epoch_microbatch_splits_never_duplicate_or_skip():
    rows = [dict(length=n) for n in (20, 23, 64, 65, 128, 129, 256, 257, 384, 385, 500)]
    flat, ends = identity_schedule(rows, [4, 2, 1, 1, 1], 15, 1)
    assert sorted(flat) == list(range(len(rows)))
    cursor = 0
    seen = []
    for stop in (3, 6, 9, 11):
        while cursor < stop:
            groups, cursor, _ = plan_update(rows, flat, ends, cursor, stop, 100)
            seen.extend(i for group in groups for i in group)
    assert seen == flat and len(set(seen)) == len(rows)


def test_paired_bootstrap_equal_protein_means_and_known_delta():
    old = []
    new = []
    for i, length in enumerate((20, 500, 40, 400)):
        for t in (5, 50):
            a = dict(
                sample_id=str(i),
                length=length,
                length_stratum="20-64" if length < 65 else "385-500",
                timestep=t,
                metric=10.0 + i,
            )
            old.append(a)
            new.append(dict(a, metric=a["metric"] - 2))
    result = paired_scalar_comparison(new, old, "metric", (5, 50))
    assert result["paired_proteins"] == 4 and result["difference"] == -2
    assert np.allclose(result["ci95"], [-2, -2])


def test_smoke_namespace_is_shared_across100_and1000_but_changes_with_source():
    from protein_distance_diffusion.training.e015_trainer import smoke_folder

    contract = {"source": "fixed", "batching_hash": "same"}
    assert smoke_folder(contract) == smoke_folder(dict(contract))
    assert smoke_folder(contract) != smoke_folder(dict(contract, source="changed"))
