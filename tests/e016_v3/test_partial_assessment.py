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

def test_signed_chirality_metric_reverses_under_reflection():
 x=helix();a=metrics(x);x[:,0]*=-1;b=metrics(x)
 assert b['signed_triple_product_mean']==pytest.approx(-a['signed_triple_product_mean'],abs=1e-12)
 assert b['torsion_circular_mean_rad']==pytest.approx(-a['torsion_circular_mean_rad'],abs=1e-12)

def test_generation_panel_recovers_cached_work_without_model(tmp_path):
 import partial_finalize as f
 f.write(tmp_path/'panel.json',[{'N':N,'seed':2026101700+100*N+j,'sample_index':j,'checkpoint_sha256':'fixture-sha','status':'GENERATION_GEOMETRY_FAILURE'} for N in f.LENGTHS for j in range(32) if (N,j)!=(500,31)])
 reused={('fixture-sha',500,31):{'N':500,'seed':2026101700+50000+31,'sample_index':31,'checkpoint_sha256':'fixture-sha','status':'GENERATION_GEOMETRY_FAILURE'}}
 old=f.R;f.R=tmp_path
 try:
  items=f.sample_panel('fixture',{},None,'fixture-sha',tmp_path/'panel.json',None,set(),reused);assert len(items)==160;assert items[-1]['prediction_cache_reused']
  items[0]['seed']=0;f.write(tmp_path/'panel.json',items)
  with pytest.raises(ValueError,match='generation cache'):f.sample_panel('fixture',{},None,'fixture-sha',tmp_path/'panel.json',None,set(),{})
 finally:f.R=old
