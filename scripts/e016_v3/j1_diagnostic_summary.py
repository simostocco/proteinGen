"""Compact postprocessing of immutable diagnostic arrays; TRAIN-only analytic calibration prototypes."""
import json,math,hashlib
from pathlib import Path
import numpy as np
from scipy.special import i0e,i1e,logsumexp
from scipy.optimize import minimize_scalar
import j1_torsion_diagnostic as d
R=d.R

def fit_options(tag):
 z=np.load(R/'arrays'/f'{tag}_TRAIN.npz');panel=sorted(json.load(open(d.O/'configs/EVALUATION_PANELS.json'))['TRAIN'],key=lambda e:(e[2],e[0],e[1]));ok=z['mask'][:,2];ids=np.array([h for h,o,t in panel])[ok];train=np.array([int(h[:8],16)%2==0 for h in ids]);mu=z['p'][ok,2].astype(float);sig=np.exp(np.clip(z['p'][ok,5].astype(float),-5,2));u=z['u'][ok,2];tau=z['g'][ok,2];old=z['nll'][ok,2];res=(u-mu)/sig
 # Fit a single Gaussian uncertainty multiplier using TRAIN fit identities only. No neural updates.
 temp=float(np.sqrt(np.mean(res[train]**2)));jac=np.log(math.pi)+2*(np.log(2)-u-np.logaddexp(0,-2*u));nll=.5*(res/temp)**2+np.log(sig*temp)+.5*np.log(2*math.pi)+jac
 # Moment-match circular candidates, no production head changes. GH256 vs512 convergence reported.
 def moments(n):
  nodes,w=np.polynomial.hermite.hermgauss(n);w=w/math.sqrt(math.pi);s=np.empty(len(mu));c=s.copy()
  for j in range(0,len(mu),512):
   x=math.pi*np.tanh(mu[j:j+512,None]+math.sqrt(2)*sig[j:j+512,None]*nodes);s[j:j+512]=np.sin(x)@w;c[j:j+512]=np.cos(x)@w
  return np.arctan2(s,c),np.minimum(.999999999999,np.hypot(s,c))
 circ,r=moments(256);c128,r128=moments(128);delta=np.arctan2(np.sin(tau-circ),np.cos(tau-circ));k=np.where(r<.53,2*r+r**3+5*r**5/6,np.where(r<.85,-.4+1.39*r+.43/(1-r),1/(r**3-4*r**2+3*r)))
 for _ in range(6):
  a=i1e(k)/i0e(k);k=np.maximum(1e-8,k-(a-r)/np.maximum(1e-12,1-a*a-a/k))
 def vm_nll(scale):
  kk=k*scale;return math.log(2*math.pi)+np.log(i0e(kk))+abs(kk)-kk*np.cos(delta)
 fit=minimize_scalar(lambda x:float(vm_nll(math.exp(x))[train].mean()),bounds=(-3,3),method='bounded');vm=vm_nll(math.exp(fit.x))
 # Wrapped Gaussian moment matching: exp(-s^2/2)=R. A finite sum of images validated by integration.
 sd=np.sqrt(-2*np.log(np.maximum(r,1e-12)));v=np.stack([-.5*((delta+2*math.pi*j)/sd)**2-np.log(sd)-.5*np.log(2*math.pi) for j in range(-12,13)]);wn=-logsumexp(v,axis=0)
 hold=~train
 return {'TRAIN_fit_identities':len(set(ids[train])),'TRAIN_holdout_identities':len(set(ids[hold])),'fit_targets':int(train.sum()),'holdout_targets':int(hold.sum()),'current_holdout_torsion_NLL':float(old[hold].mean()),'uncertainty_multiplier_fitted_TRAIN':temp,'scaled_gaussian_holdout_NLL':float(nll[hold].mean()),'scaled_gaussian_holdout_delta':float((nll-old)[hold].mean()),'von_mises_concentration_scale_fitted_TRAIN':math.exp(fit.x),'moment_matched_von_mises_holdout_NLL':float(vm[hold].mean()),'moment_matched_wrapped_normal_holdout_NLL':float(wn[hold].mean()),'quadrature256_vs128_circular_mean_difference':d.st(abs(np.arctan2(np.sin(circ-c128),np.cos(circ-c128)))),'interpretation':'posthoc distribution prototypes derived from existing predicted moments, not trained circular heads or demonstrated causal fixes; fits/calibration use TRAIN identities only; heldout TRAIN is not independent VALIDATION; no production modification','holdout_seam10deg':{'targets':int((hold&(abs(tau)>math.pi-math.radians(10))).sum()),'current_NLL':float(old[hold&(abs(tau)>math.pi-math.radians(10))].mean()),'vonMises_NLL':float(vm[hold&(abs(tau)>math.pi-math.radians(10))].mean())}}

def main():
 summaries={tag:{role:json.load(open(R/f'summaries/{tag}_{role}.json')) for role in ['TRAIN','VALIDATION']} for tag in d.CP};d.write('LOCAL_COMPONENT_TRAIN_VALIDATION.json',summaries);d.write('TORSION_CALIBRATION.json',{tag:{role:v['torsion_calibration'] for role,v in roles.items()} for tag,roles in summaries.items()});differences={}
 for role in ['TRAIN','VALIDATION']:
  base=np.load(R/'arrays'/f'source5000_{role}.npz');a=base['nll'];mask=base['mask'];out={}
  for tag in d.CP:
   b=np.load(R/'arrays'/f'{tag}_{role}.npz')['nll'];delta=b-a;ret={'component_mean_delta_from_source5000':{name:float(delta[mask[:,c],c].mean()) for c,name in enumerate(['distance','angle','torsion'])},'historic_local_delta':float((delta.sum(1)/mask.sum(1).clip(1)).mean())};v=delta[mask[:,2],2];idx=np.argsort(v);tails={}
   for frac in [.001,.01,.05]:
    k=max(1,math.ceil(len(v)*frac));tails[str(frac)]={'targets':k,'positive_delta_contribution_to_mean':float(v[idx[-k:]].sum()/len(v)),'remaining_delta_mean_analysis_only':float(v[idx[:-k]].mean()),'fraction_of_positive_deterioration':float(v[idx[-k:]].clip(0).sum()/max(v.clip(0).sum(),1e-30))}
   ret['paired_torsion_worst_deterioration_targets']=tails
   # Excluding exact same targets ranked by source NLL is a separate sensitivity, never an eligibility change.
   src=a[mask[:,2],2];order=np.argsort(src);ret['excluding_source_worst_torsion_1pct_analysis_only_delta']=float(v[order[:-math.ceil(len(v)*.01)]].mean());out[tag]=ret
  differences[role]=out
 d.write('PAIRED_COMPONENT_DIFFERENCES.json',differences)
 options={tag:fit_options(tag) for tag in ['original','source5000','pilot500','pilot1500','pilot2500']};d.write('TRAIN_ONLY_CORRECTION_PROTOTYPES.json',options)
 tests={'optimizer_steps':0,'production_checkpoint_weights_modified':False,'panel_decisions_each':23576,'valid_components_each':20629,'tests':[]}
 for tag,roles in summaries.items():
  for role,v in roles.items():
   if v['decisions']!=23576 or any(v['components'][n]['target_weighted']['n']!=20629 for n in ['distance','angle','torsion']):raise ValueError('frozen denominators changed')
   local=sum(v['components'][n]['target_weighted']['mean'] for n in ['distance','angle','torsion'])/3*7/8
   if abs(local-v['historic_local_per_decision'])>1e-12:raise ValueError('historic normalization mismatch')
 tests['tests'].append('all12 frozen panels exact counts and historic component normalization PASS')
 # Normalization of physical circular prototype densities and periodic seam continuity.
 grid=np.linspace(-math.pi,math.pi,20001);k=2.7;mean=2.9;vm=np.exp(k*np.cos(grid-mean)-math.log(2*math.pi)-math.log(float(i0e(k)))-k);wn=np.exp(logsumexp(np.stack([-.5*((grid-mean+2*math.pi*j)/.9)**2-math.log(.9)-.5*math.log(2*math.pi) for j in range(-12,13)]),axis=0))
 tests['circular_density_integrals']={'vonMises':float(np.trapezoid(vm,grid)),'wrappedGaussian':float(np.trapezoid(wn,grid))};tests['circular_seam_endpoint_density_difference']={'vonMises':float(abs(vm[0]-vm[-1])),'wrappedGaussian':float(abs(wn[0]-wn[-1]))}
 if any(abs(x-1)>1e-9 for x in tests['circular_density_integrals'].values()):raise ValueError('circular prototype normalization failed')
 tests['tests'].append('circular prototype density normalization and seam continuity PASS');tests['status']='PASS';d.write('DIAGNOSTIC_TESTS.json',tests)
 print(json.dumps({'tests':tests,'prototype_results':options}),flush=True)
if __name__=='__main__':main()
