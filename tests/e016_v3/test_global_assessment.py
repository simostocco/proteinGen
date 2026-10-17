import copy,json
import numpy as np
from j1_global_finalize import metric_summary,paired_ci
from partial_finalize import metrics
from j1_global import OLD
def samples():
 rows=json.load(open(OLD/'runs/J1-FULL/SELECTED_GENERATION.json'))
 out=[copy.deepcopy(v) for v in rows if v['N']==64 and v['sample_index']<3]
 for v in out:v['metrics']=v['independent_CA_metrics']
 return out
def test_shape_diversity_is_not_just_radius():
 a=samples();b=copy.deepcopy(a)
 for v in b:
  v['generated_CA_A']=(np.array(v['generated_CA_A'])*2).tolist();v['metrics']=metrics(v['generated_CA_A'])
 x=metric_summary(a)['64'];y=metric_summary(b)['64']
 np.testing.assert_allclose(x['mean_pair_Rg_normalized_distance_fingerprint_RMS'],y['mean_pair_Rg_normalized_distance_fingerprint_RMS'],rtol=1e-12)
 np.testing.assert_allclose(2*x['mean_pair_distance_fingerprint_RMS_A'],y['mean_pair_distance_fingerprint_RMS_A'],rtol=1e-12)
def test_small_or_failed_generation_panels_do_not_create_NaN():
 s=metric_summary(samples()[:1]);assert s['64']['mean_pair_sequence_Hamming_fraction'] is None
 json.dumps(s,allow_nan=False);json.dumps(metric_summary([]),allow_nan=False)
def test_paired_identical_checkpoint_comparison_is_zero():
 x=paired_ci([0.]*32);assert x['mean']==0 and x['bootstrap_95pct']==[0.,0.]
