import numpy as np
import torch
import pytest
from protein_distance_diffusion.training import e010_sequential_teacher_v16c as teacher
from scripts import run_e010_sequential_teacher_v16d as run
from scripts.recover_e010_conditioning_v9b import synthetic


@pytest.fixture(scope='module',autouse=True)
def all_derivatives_before_integration_solver():
    torch.set_num_threads(1)
    for n in (12,32,500):
        assert teacher.validate(synthetic(n))['baseline_feasible']


def test_one_step_execution_safety_and_serialization(tmp_path):
    torch.set_num_threads(1)
    b=synthetic(12);context=teacher.Context(b)
    path=tmp_path/'action.npy'
    a,log=teacher.solve_step(context,b['pg'],lambda v:np.save(path,v))
    assert np.array_equal(np.load(path),a)
    assert np.linalg.norm(a,axis=-1).max()<=.1
    assert context.safety(b['pg']+torch.from_numpy(a))['passes']
    assert log['accepted_normalized_local']<=1
    assert teacher.OneStep(context,b['pg']).n==36


def test_metadata_reconstruction_and_distribution():
    torch.set_num_threads(1)
    b=synthetic(12);context=teacher.Context(b);step=teacher.OneStep(context,b['pg'])
    a=np.zeros(tuple(b['pg'].shape))
    rows=run.metric_rows(b,dict(sample_id='synthetic',condition=50,stratum='20–64'),
                          [b['pg'].numpy(),b['pg'].numpy()],[a],[step.eligible.numpy()])
    assert rows[1]['safety']['passes'] and rows[1]['exact_stored_step_max']==0
    assert run.distribution(a,step.eligible.numpy())['rms']==0
    assert teacher.validate(b)['directional_checks'][0]['epsilon']==1e-4
