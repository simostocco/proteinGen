"""C-alpha auxiliary descriptors, separate from sequence confidence and rewards."""
import numpy as np
from .sequence_rewards import mapped_structure_scores
def geometry_features(x):
 x=np.asarray(x,float)
 if x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all():raise ValueError('hard failure: nonfinite/malformed proposed backbone')
 n=len(x);b=np.diff(x,axis=0);d=np.linalg.norm(b,axis=1);cross=np.cross(b[:-1],b[1:]);angles=np.arctan2(np.linalg.norm(cross,axis=1),-(b[:-1]*b[1:]).sum(1));degen=int(np.sum(d<1e-10)+np.sum(np.linalg.norm(cross,axis=1)<1e-10));tors=[]
 for i in range(3,n):
  a,c,e=x[i-3:i];z=x[i];u=e-c;u=u/max(np.linalg.norm(u),1e-30);v=-(c-a);v=v-v@u*u;w=z-e;w=w-w@u*u;vn=np.linalg.norm(v);wn=np.linalg.norm(w)
  if vn>=1e-10 and wn>=1e-10:tors.append(float(np.arctan2(np.cross(u,v/vn)@(w/wn),(v/vn)@(w/wn))))
 dist=np.linalg.norm(x[:,None]-x[None,:],axis=-1);sep=np.abs(np.arange(n)[:,None]-np.arange(n)[None,:]);pairs=np.triu(sep>=3,1);long=np.triu(sep>=24,1);clashes=int(np.sum((dist<2)&pairs));rg=float(np.sqrt(np.mean(np.sum((x-x.mean(0))**2,1))))
 return {'N':n,'hard_geometry_valid':degen==0,'degenerate_local_frames_or_bonds':degen,'adjacent_distance_mean_A':float(d.mean()),'adjacent_distance_min_A':float(d.min()),'adjacent_distance_max_A':float(d.max()),'virtual_angle_mean_rad':float(angles.mean()),'signed_torsion_positive_fraction':float(np.mean(np.array(tors)>0)) if tors else None,'nonlocal_pairs_below2A':clashes,'eligible_nonlocal_pairs':int(pairs.sum()),'clashes_per100_residues':100*clashes/n,'clashes_per_eligible_pair':clashes/max(1,int(pairs.sum())),'Rg_A':rg,'Rg_over_N_cuberoot':rg/n**(1/3),'long_contact_fraction':float(np.mean(dist[long]<8)) if long.any() else None,'local_torsion_valid_count':len(tors),'physical_role':'C-alpha proxies, not atomic chirality, foldability or stability'}
def confidence_aware_agreement(proposed,predicted,plddt):
 raw=mapped_structure_scores(proposed,predicted);mask=np.asarray(plddt)>=70
 return {'all_positions_descriptive':raw,'confidence_qualified':mapped_structure_scores(proposed,predicted,mask) if mask.sum()>=4 else None,'confident_residue_count':int(mask.sum()),'low_confidence_AF_not_treated_as_native_truth':True,'AA_sequence_reward_depends_on_this':False}
