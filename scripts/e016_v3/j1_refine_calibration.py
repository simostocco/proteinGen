"""Higher-order quadrature and seam/tail sensitivity; no fitting or model changes."""
import json,math
import numpy as np
import j1_torsion_diagnostic as d
cal=json.load(open(d.R/'TORSION_CALIBRATION.json'));tails={}
for tag in d.CP:
 for role in ['TRAIN','VALIDATION']:
  z=np.load(d.R/'arrays'/f'{tag}_{role}.npz');ok=z['mask'][:,2];mu=z['p'][ok,2].astype(float);sig=np.exp(np.clip(z['p'][ok,5].astype(float),-5,2));tau=z['g'][ok,2];out=[]
  for order in [128,256]:
   nodes,w=np.polynomial.hermite.hermgauss(order);w=w/math.sqrt(math.pi);a=np.empty(len(mu));circ=a.copy()
   for off in range(0,len(mu),1024):
    x=math.pi*np.tanh(mu[off:off+1024,None]+math.sqrt(2)*sig[off:off+1024,None]*nodes);a[off:off+1024]=x@w;circ[off:off+1024]=np.arctan2(np.sin(x)@w,np.cos(x)@w)
   out.append((a,circ))
  delta=lambda x:np.arctan2(np.sin(x),np.cos(x))
  cal[tag][role]['physical_moment_refinement']={'quadrature_order':256,'arithmetic_mean_abs_circular_error_rad':d.st(abs(delta(out[-1][0]-tau))),'circular_mean_abs_error_rad':d.st(abs(delta(out[-1][1]-tau))),'order256_vs128_circular_mean_abs_difference_rad':d.st(abs(delta(out[-1][1]-out[0][1]))),'note':'quadrature is approximate, especially broad conditionals; CDF interval/mass diagnostics are analytic and do not depend on quadrature'}
for role in ['TRAIN','VALIDATION']:
 a=np.load(d.R/'arrays'/f'pilot500_{role}.npz');b=np.load(d.R/'arrays'/f'pilot2500_{role}.npz');ok=a['mask'][:,2];v=b['nll'][ok,2]-a['nll'][ok,2];tau=b['g'][ok,2];seam=abs(tau)>math.pi-math.radians(10);q={}
 for f in [.001,.01,.05]:
  k=math.ceil(len(v)*f);ix=np.argsort(v)[-k:];q[str(f)]={'targets':k,'delta_contribution_to_mean':float(v[ix].sum()/len(v)),'remaining_mean_delta_analysis_only':float((v.sum()-v[ix].sum())/(len(v)-k)),'native_seam10deg_fraction':float(seam[ix].mean())}
 tails[role]={'torsion_delta_pilot2500_minus_pilot500':float(v.mean()),'native_seam10deg_fraction':float(seam.mean()),'seam_delta_contribution_to_mean':float(v[seam].sum()/len(v)),'other_delta_contribution_to_mean':float(v[~seam].sum()/len(v)),'tails':q,'known_anomalous_observation_decisions':int(b['anomaly'].sum()),'known_anomaly_note':'observation-level flag, not proof target geometry contains the anomalous coordinate pair'}
d.write('TORSION_CALIBRATION.json',cal);d.write('TORSION_TAIL_SEAM_SENSITIVITY.json',tails);print('CALIBRATION_REFINEMENT_COMPLETE')
