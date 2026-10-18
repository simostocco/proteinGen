import copy
import numpy as np
import pytest
from j1_sequence_pause import describe,validate_counters

def state(u=8450):
 return {'additional_update':u,'optimizer':{'state':{0:{'step':30000+u}}},'scheduler':{'last_epoch':u},'sampler':{'cursor':u}}

def test_paused_recovery_ceiling_and_counter_contract():
 st=state();done={'update':8450};receipt={'verified_received_update':6000,'maximum_additional_update':8500}
 assert validate_counters(st,done,receipt)==(8450,{38450})
 for mutate in [
  lambda s:s.update(additional_update=8501),
  lambda s:s['optimizer']['state'][0].update(step=30000),
  lambda s:s['scheduler'].update(last_epoch=8449),
  lambda s:s['sampler'].update(cursor=8449)]:
  bad=copy.deepcopy(st);mutate(bad)
  with pytest.raises(ValueError):validate_counters(bad,done,receipt)

def test_distribution_summary_does_not_convert_missing_to_zero():
 out=describe([1.,2.,3.,np.nan,np.inf])
 assert out['n']==3 and out['mean']==2. and out['zero_fraction']==0.
 assert out['nonfinite_count']==2 and out['n_total']==5
 assert describe([])=={'n':0}


def test_bfloat16_audit_conversion_preserves_values():
 import torch
 out=describe(torch.tensor([.5,1.,2.],dtype=torch.bfloat16))
 assert out['n']==3 and out['nonfinite_count']==0
 assert out['mean']==pytest.approx(3.5/3)
