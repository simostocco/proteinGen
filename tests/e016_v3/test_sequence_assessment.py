import copy
from j1_sequence_assess import check_candidate
def row():
 return {'complete_valid':32,'unique_sequences':32,'clashes_per100_residues':{'mean':2.},'Rg_over_N_cuberoot':{'mean':4.},'mean_pair_sequence_Hamming_fraction':.95,'mean_pair_Rg_normalized_distance_fingerprint_RMS':.6}
def test_inherited_clash_guard():
 a={'64':row()};b=copy.deepcopy(a);n={'TRAIN':{'64':{'rows':[{'metrics':{'Rg_over_N_cuberoot':4.}}]}}}
 assert check_candidate(a,b,n)==[]
 a['64']['clashes_per100_residues']['mean']=2.76
 assert '64: clash regression' in check_candidate(a,b,n)
def test_empty_generation_is_rejected_without_fake_metrics():
 b={'64':row()};a={'64':{'complete_valid':0}};n={'TRAIN':{}}
 assert check_candidate(a,b,n)==['64: incomplete']

def test_numpy_eligibility_is_json_safe():
 import json,numpy as np
 from j1_sequence_first import selection_flag
 value=selection_flag(-.01,.05,True,np.float64(2.),{'local_delta':.2,'severe_clash_mean':6.})
 assert type(value) is bool
 assert json.loads(json.dumps({'eligible':value}))=={'eligible':True}
