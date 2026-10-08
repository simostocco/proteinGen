"""Analytic checks of the convex projection, independent of protein labels."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_e010_constraint_projected_v1 as experiment
import numpy as np


def test_coupled_halfspace_and_ball():
    desired = np.array([[[.1, .1, 0.]]])
    action, info = experiment.convex_projection(
        desired, np.array([-.04]), np.array([[1., 0., 0.]]), np.array([[True]]), np.zeros_like(desired))
    # Both boundaries bind; projecting separately once does not give this point.
    expected = np.array([[[.04, np.sqrt(.1**2 - .04**2), 0.]]])
    assert info['converged']
    assert np.allclose(action, expected, atol=2e-9, rtol=0.)
    assert np.linalg.norm(action, axis=-1).max() <= .1


def test_eliminated_eligibility_and_relinearized_offset():
    desired = np.array([[[.08, 0., 0.], [.08, 0., 0.]]])
    offset = np.array([[[.02, 0., 0.], [0., 0., 0.]]])
    # c=.02 at offset=.02 linearizes total x<=0; ineligible second residue stays zero.
    action, info = experiment.convex_projection(
        desired, np.array([.02]), np.array([[1., 0., 0., 1., 0., 0.]]), np.array([[True, False]]), offset)
    assert info['converged']
    assert np.allclose(action, 0., atol=1e-12)
    assert (action[:, 1] == 0).all()


def test_infeasible_linearization_is_explicitly_budget_limited():
    desired = np.array([[[.1, 0., 0.]]])
    action, info = experiment.convex_projection(
        desired, np.array([.2]), np.array([[1., 0., 0.]]), np.array([[True]]), np.zeros_like(desired))
    assert not info['converged']
    assert info['sweeps'] == experiment.CONFIG['maximum_convex_sweeps']
    assert info['original_linear_violation'] >= .099
    assert np.linalg.norm(action, axis=-1).max() <= .1
