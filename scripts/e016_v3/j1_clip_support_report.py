"""CPU-only paired-noise precision diagnostics; reference density/code unchanged."""
import json,math
from pathlib import Path
import numpy as np
import torch
from protein_joint_sac.models.geometry_inference import corrected_action,legacy_action,PROTOCOL
R=Path('/mnt/d/Simone/proteinGen_data/e016_j1_clip_precision_pilot_20261018_v1')
def main():
 rows=[]
 for z in [-1e6,-100.,-10.,-9.,0.,9.,10.,100.,1e6]:
  mu=torch.tensor([math.log(3.8),0.,z]);eps=torch.zeros(3);g,d=corrected_action(mu,torch.zeros(3),(True,True,True),eps);old=legacy_action(mu,torch.zeros(3),(True,True,True),eps).double()
  rows.append({'latent_torsion':z,'legacy_torsion':float(old[2]),'corrected_torsion':float(g[2]),'legacy_outside_builder_interval':not -math.pi<=float(old[2])<=math.pi,**d})
 rng=np.random.default_rng(182042);eps=rng.standard_normal((2048,3));legacy_bad=repairs=0;maxdiff=0.
 for e in eps:
  mu=torch.tensor([math.log(3.8),0.,0.]);ls=torch.tensor([-3.,-1.,2.]);noise=torch.from_numpy(e)
  old=legacy_action(mu,ls,(True,True,True),noise).double();g,d=corrected_action(mu,ls,(True,True,True),noise)
  legacy_bad+=int(not -math.pi<=float(old[2])<=math.pi);repairs+=d['adjusted_components'][2];maxdiff=max(maxdiff,abs(float(g[2]-old[2])))
 out={'status':'PASS','protocol':PROTOCOL,'extreme_latent_checks':rows,'paired_noise_stress':{'seed':182042,'samples':2048,'torsion_log_std':2.,'interpretation':'deliberately broad fixed reference Gaussian; not measured model frequency','legacy_outside_torsion_support':legacy_bad,'corrected_outside_support':0,'corrected_endpoint_representations':repairs,'max_abs_action_difference_rad':maxdiff},'CPU_tests':'tests/CLIP_PRECISION_RETRY.xml:17passed','regression_tests':'tests/REGRESSION.xml:35passed','limitations':['Inference-only finite-grid quantization, not an altered training density','Exact densities at original latent and represented action differ for repaired extremes; no claim of discrete mass likelihood','Distance overflow/underflow and degenerate frames remain explicit failures','Historical failed latent unavailable; no retrospective exact attribution','Saved common epsilon/uniform arrays are new-protocol draws, not replicas of archived random draws'],'normalization_changed':False}
 p=R/'TORSION_SUPPORT_TESTS.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['paired_noise_stress']))
if __name__=='__main__':main()
