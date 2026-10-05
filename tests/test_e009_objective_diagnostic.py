import torch

from scripts.e009_objective_diagnostic import (
    ARM_CONFIG,
    _eligible_counts,
    _normalization_review,
    _reductions,
    _validate_normalized_arm_protocol,
)


def test_objective_reduction_audit_covers_all_components_and_counts():
    mask = torch.ones((1, 64), dtype=torch.bool)
    counts = _eligible_counts(mask)
    assert counts == {
        "coordinate_nll": 192,
        "bond_prior_nll": 63,
        "angle_prior_nll": 62,
        "torsion_prior_nll": 61,
        "long_range_pair_nll": 3192,
        "contact_nll": 3192,
        "radius_of_gyration_nll": 1,
        "chirality_nll": 61,
        "posterior_kl": 64,
    }
    assert set(counts) == set(_reductions())


def test_mixed_historical_reductions_fail_closed_and_arm_b_declares_conversion():
    review = _normalization_review()
    assert review["existing_objective"]["fail_closed"] is True
    assert review["existing_objective"]["compatible_reductions"] is False
    assert review["arm_b"]["compatible_reductions"] is True
    assert "multiply xyz-scalar mean by 3" in review["arm_b"]["explicit_conversions"]["coordinate_nll"]
    assert ARM_CONFIG["max_updates_per_arm"] == 1000
    assert ARM_CONFIG["authorizes_downstream"] is False
    assert set(_validate_normalized_arm_protocol()) == set(_reductions())
