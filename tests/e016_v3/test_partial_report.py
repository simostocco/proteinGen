import pytest
from partial_report import paired_delta,selected_update

def test_paired_likelihood_comparison_preserves_opposing_components():
 a={'per_identity_AA_geometry_mean':{'a':[2.,-1.],'b':[3.,0.]}};b={'per_identity_AA_geometry_mean':{'a':[1.,0.],'b':[2.,1.]}}
 result=paired_delta(a,b);assert result['right_minus_left_AA_geometry_objective']==pytest.approx([-1.,1.,0.]);assert result['identity_bootstrap_95pct']==[[-1.,1.,0.],[-1.,1.,0.]]
 with pytest.raises(ValueError,match='panels differ'):paired_delta(a,{'per_identity_AA_geometry_mean':{'a':[1.,0.]}})

def test_original_parent_is_additional_update_zero(tmp_path):
 assert selected_update(tmp_path,'J1-FULL',{'selected_checkpoint':str(tmp_path/'old/J1/checkpoint_10000.pt')})==0
 assert selected_update(tmp_path,'J1-FULL',{'selected_checkpoint':str(tmp_path/'runs/J1-FULL/checkpoint_05000.pt')})==5000
