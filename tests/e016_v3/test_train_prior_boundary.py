import importlib.util
from pathlib import Path
import numpy as np
import pytest
p=Path(__file__).resolve().parents[2]/'scripts/e016_v3/fit_certified_train_priors.py'
# tests/e016_v3 -> source root is parents[2]
spec=importlib.util.spec_from_file_location('train_prior_reference',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_no_heldout_fitting():
 good={'split':'TRAIN','role':'TRAINING','eligible':True};m.assert_train([good])
 for change in [{'split':'TEST'},{'split':'VALIDATION'},{'role':'NATIVE_AF_CALIBRATION'},{'eligible':False}]:
  with pytest.raises(ValueError):m.assert_train([good|change])
def test_prior_internal_matches_reference_geometry():
 import torch
 from protein_joint_sac.geometry.ca_builder import replay,mask_for_step
 from protein_joint_sac.contracts import BuildAction
 actions=tuple(BuildAction('A',torch.tensor([3.8 if i>=1 else 0.,1.9 if i>=2 else 0.,.4 if i>=3 else 0.],dtype=torch.float64),mask_for_step(i)) for i in range(20))
 state=replay(20,actions);d,t,tau=m.internal(state.ca.numpy())
 np.testing.assert_allclose(d,3.8,atol=1e-12);np.testing.assert_allclose(t,1.9,atol=1e-12);np.testing.assert_allclose(tau,.4,atol=1e-12)
