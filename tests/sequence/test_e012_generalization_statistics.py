"""Independent statistics contract for disjoint-panel and paired-identity inference."""

import numpy as np

from protein_sequence_generation.e012 import paired_interval
from scripts.report_e012_generalization import gap_interval, interval
from scripts.verify_e012_generalization import reproduce, reproduce_gap


def test_historical_paired_bootstrap_numerical_parity():
    values = np.random.default_rng(99).normal(size=37)
    canonical = paired_interval(values)
    actual = interval(values)
    assert actual["delta"] == canonical["delta"]
    assert np.allclose(actual["ci95"], canonical["ci95"], atol=1e-12, rtol=0)
    mean, ci = reproduce(values)
    assert mean == actual["delta"]
    assert np.allclose(ci, actual["ci95"], atol=1e-12, rtol=0)


def test_disjoint_panel_gap_does_not_invent_pairing():
    values = [-1.0, 1.0]
    actual = gap_interval(values, values)
    assert actual["delta"] == 0
    assert actual["ci95"][0] < 0 < actual["ci95"][1]
    mean, ci = reproduce_gap(values, values)
    assert mean == 0 and np.array_equal(ci, actual["ci95"])
    constant = gap_interval([1.0] * 10, [3.0] * 10)
    assert constant["delta"] == 2 and constant["ci95"] == [2, 2]
