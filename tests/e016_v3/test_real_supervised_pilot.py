import math
from dataclasses import replace
import numpy as np
import pytest,torch
from protein_joint_sac.rl.bc_training import BCConfig,BCTrainer,Population
from protein_joint_sac.data.engineering_fixtures import records
from protein_joint_sac.geometry.ca_builder import inverse
from protein_joint_sac.data.accepted_supervised import CoordinateReader

def test_component_normalized_loss_and_bootstrap():
 c=BCConfig(width=16,heads=2,layers=1,geometry_normalization='valid_component_mean');p=Population(records(),'ENGINEERING_FIXTURE');t=BCTrainer(c,p)
 for step in [0,1,2,3,12]:
  loss,m=t.loss(0,step);assert m['geometry_components']==sum([step>=1,step>=2,step>=3]);assert float(loss)==pytest.approx(m['aa_nll']+m['geometry_nll_sum']/max(1,m['geometry_components']),abs=1e-5)

def test_unadmitted_real_scope_rejected():
 with pytest.raises(ValueError,match='factory'):Population(records(),'LIMITED_PDB_PILOT')

def test_config_normalization_and_weight_decay_validation():
 for kwargs in [dict(geometry_normalization='hidden'),dict(weight_decay=-1),dict(weight_decay=float('nan'))]:
  with pytest.raises(ValueError):BCConfig(**kwargs)

def test_prior_vectorized_geometry_matches_builder():
 import importlib.util
 from pathlib import Path
 path=Path(__file__).parents[2]/'scripts/e016_v3/real_pdb_pilot.py';spec=importlib.util.spec_from_file_location('pilot',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 x=np.asarray(records()[0]['coordinates']);d,theta,tau=m.internal(x)
 for i in range(3,len(x)):
  expected=inverse(torch.tensor(x[:i+1],dtype=torch.float64))[0].numpy();np.testing.assert_allclose([d[i-1],theta[i-2],tau[i-3]],expected,atol=1e-10)

def test_modified_or_degenerate_not_silently_normalized():
 c=BCConfig(width=16,heads=2,layers=1)
 r=records();r[0]['sequence']='X'+r[0]['sequence'][1:]
 with pytest.raises(ValueError):Population(r,'ENGINEERING_FIXTURE')

def test_sequence_no_coordinate_reader_and_chemistry_disabled():
 p=Population(records(),'ENGINEERING_FIXTURE');p.coordinates=lambda *a:(_ for _ in ()).throw(AssertionError('no coordinates'))
 c=BCConfig(mode='sequence',chemistry=False,width=16,heads=2,layers=1);t=BCTrainer(c,p);loss,m=t.loss(0,12);assert math.isfinite(float(loss));assert m['geometry_components']==0
