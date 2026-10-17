import numpy as np
import pytest
from partial_finalize import metrics

def helix(n=64):
 t=np.arange(n);return np.column_stack([2.3*np.cos(t*1.7),2.3*np.sin(t*1.7),1.5*t])

def test_assessment_uses_same_rigid_invariant_ca_metrics():
 x=helix();q=np.linalg.qr(np.random.default_rng(42).normal(size=(3,3)))[0];a=metrics(x);b=metrics(x@q+np.array([31.,-42.,8.]));assert a['N']==64
 for k in ['Rg_A','Rg_over_N_cuberoot','nonlocal_clashes_below2A_seqsep_ge3','long_contacts_below8A_seqsep_ge24','torsion_resultant']:assert b[k]==pytest.approx(a[k],abs=1e-10)
 assert a['angle_histogram']==b['angle_histogram'];assert a['torsion_histogram']==b['torsion_histogram']

def test_assessment_rejects_nonfinite_complete_trace():
 x=helix();x[8]=np.nan
 with pytest.raises(ValueError):metrics(x)
