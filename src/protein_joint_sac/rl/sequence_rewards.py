"""Separate sequence features and geometry evidence; uncalibrated is not deployable."""
import hashlib,math
from collections import Counter
import numpy as np

def sequence_features(sequence,plddt,ptm,pae,predicted_ca):
    n=len(sequence);p=np.asarray(plddt,dtype=float);x=np.asarray(predicted_ca,dtype=float)
    if p.shape!=(n,) or x.shape!=(n,3) or not np.isfinite(p).all() or not np.isfinite(x).all() or np.any((p<0)|(p>100)):raise ValueError('full finite mapped oracle outputs required')
    counts=Counter(sequence);freq=np.array(list(counts.values()),float)/n;entropy=float(-(freq*np.log(freq)).sum());pairs=np.abs(np.arange(n)[:,None]-np.arange(n)[None,:])>=min(24,max(4,n//4))
    if ptm is None or pae is None:global_conf=None;pae_good=None;pae_mean=None
    else:
        a=np.asarray(pae,float)
        if a.shape!=(n,n) or not np.isfinite(a).all() or (a<0).any() or not 0<=ptm<=1:raise ValueError('invalid genuine pTM/PAE')
        pae_good=float(np.mean(a[pairs]<10));pae_mean=float(np.mean(a[pairs]));global_conf=float(ptm)*pae_good
    runs=[];v=1
    for a,b in zip(sequence,sequence[1:]):
        if a==b:v+=1
        else:runs.append(v);v=1
    runs.append(v);rg=float(np.sqrt(((x-x.mean(0))**2).sum(1).mean()));d=np.linalg.norm(x[:,None]-x[None,:],axis=-1);pair=np.triu(pairs,1)
    return {'length':n,'sequence_sha256':hashlib.sha256(sequence.encode()).hexdigest(),'mean_pLDDT':float(p.mean()),'pLDDT_q10':float(np.quantile(p,.1)),'fraction_pLDDT_above70':float(np.mean(p>=70)),'ptm':ptm,'long_range_PAE_good_fraction':pae_good,'long_range_PAE_mean_A':pae_mean,'global_confidence_product':global_conf,'sequence_confidence_product':None if global_conf is None else global_conf*float(p.mean())/100,'composition_entropy_nats':entropy,'max_residue_fraction':float(max(freq)),'longest_identical_run':max(runs),'tripeptide_unique_fraction':len(set(sequence[i:i+3] for i in range(n-2)))/(n-2),'predicted_Rg_A':rg,'predicted_long_contact_fraction':float(np.mean(d[pair]<8)),'interpretation':'Confidence is not experimental stability; variants are not proven negative controls.'}

def calibrated_sequence_reward(features,calibration):
    if not calibration.get('GO'):raise PermissionError('AF reward NO-GO or not calibrated')
    if features['global_confidence_product'] is None:raise ValueError('missing global metrics, no fabrication')
    # TRAIN-fitted affine scales and weights are required, never silently supplied.
    if not calibration.get('weights') or any(not math.isfinite(calibration['scale'][k]) or calibration['scale'][k]<=0 for k in calibration['weights']):raise PermissionError('reward coefficients are not sealed')
    z={k:(features[k]-calibration['center'][k])/calibration['scale'][k] for k in calibration['weights']}
    if not all(math.isfinite(v) for v in z.values()):raise ValueError('nonfinite reward features')
    return float(np.clip(sum(calibration['weights'][k]*z[k] for k in z),0.,1.))

def combine_rewards(sequence_reward,geometry_reward,weight):
    if not 0<=weight<=.1:raise ValueError('conservative auxiliary geometry budget')
    if not math.isfinite(sequence_reward) or not math.isfinite(geometry_reward) or not -1<=geometry_reward<=1:raise ValueError('finite separate rewards')
    # Never multiply the primary sequence reward by proposal agreement.
    return {'sequence':sequence_reward,'geometry':geometry_reward,'geometry_weight':weight,'total':sequence_reward+weight*geometry_reward}

def mapped_structure_scores(native,predicted,observed=None):
    """Same residue identities/positions. Robust TM-style alignment, not US-align."""
    x=np.asarray(native,float);y=np.asarray(predicted,float);N=len(y);mask=np.isfinite(x).all(1) if observed is None else np.asarray(observed,bool)
    if x.shape!=y.shape or x.shape!=(N,3) or mask.sum()<4 or not np.isfinite(y).all():raise ValueError('mapped structural coverage')
    x=x[mask];y=y[mask];d0=max(.5,1.24*max(1,N-15)**(1/3)-1.8)
    def fit(weights):
        weights=weights/weights.sum();a=(x*weights[:,None]).sum(0);b=(y*weights[:,None]).sum(0);u,_,vh=np.linalg.svd(((x-a)*weights[:,None]).T@(y-b));r=u@np.diag([1,1,np.linalg.det(u@vh)])@vh;e=np.linalg.norm((x-a)@r+b-y,axis=1);return e
    e0=fit(np.ones(len(x)));best=e0
    # Multi-start robust fixed correspondence alignment retains true residue map.
    seeds=[np.ones(len(x))]
    for length in sorted(set([min(8,len(x)),min(24,len(x)),max(4,len(x)//2)])):
        for start in np.linspace(0,max(0,len(x)-length),min(8,len(x)),dtype=int):
            w=np.full(len(x),1e-8);w[start:start+length]=1;seeds.append(w)
    for w in seeds:
        for _ in range(12):
            e=fit(w);score=(1/(1+(e/d0)**2)).sum()
            if score>(1/(1+(best/d0)**2)).sum():best=e
            w=1/(1+(e/d0)**2)**2+1e-10
    dx=np.linalg.norm(x[:,None]-x[None,:],axis=-1);dy=np.linalg.norm(y[:,None]-y[None,:],axis=-1);idx=np.flatnonzero(mask);pairs=np.triu(np.abs(idx[:,None]-idx[None,:])>=12,1);cx=dx<8;cy=dy<8;union=((cx|cy)&pairs).sum()
    return {'full_length':N,'observed_positions':int(mask.sum()),'coverage':float(mask.mean()),'Kabsch_RMSD_A':float(np.sqrt(np.mean(e0**2))),'fixed_map_Kabsch_TM_similarity_fullN':float((1/(1+(e0/d0)**2)).sum()/N),'robust_fixed_map_TM_similarity_fullN':float((1/(1+(best/d0)**2)).sum()/N),'observed_only_TM_similarity':float((1/(1+(best/d0)**2)).mean()),'alignment_method':'multi-start iteratively reweighted rigid alignment under authoritative residue map; not US-align optimized sequence alignment','long_contact_Jaccard':float(((cx&cy)&pairs).sum()/union) if union else None,'long_distance_RMSE_A':float(np.sqrt(np.mean((dx[pairs]-dy[pairs])**2))) if pairs.any() else None}

def fit_reward_gate(train_rows,validation_rows):
    native=[r for r in train_rows if r.get('kind')=='native' and r.get('status')=='SUCCEEDED'];vals=[r for r in validation_rows if r.get('kind')=='native' and r.get('status')=='SUCCEEDED']
    fails=[]
    if len(native)<6:fails.append('fewer than six TRAIN native successes')
    if len(vals)<2:fails.append('fewer than two VALIDATION audit successes')
    if native:
        tm=np.array([r['recovery']['robust_fixed_map_TM_similarity_fullN'] for r in native]);conf=np.array([r['features']['ptm'] if r['features']['ptm'] is not None else np.nan for r in native],float);order=lambda v:np.array([sum(v<z)+(sum(v==z)-1)/2 for z in v],float);rho=float(np.corrcoef(order(tm),order(conf))[0,1]) if np.isfinite(conf).all() and np.ptp(conf)>0 and np.ptp(tm)>0 else None
        if np.median(tm)<.6 or np.mean(tm>=.5)<2/3:fails.append('TRAIN native recovery below sealed median.60 and >=2/3 at.50')
        if rho is None or rho<.4:fails.append('TRAIN pTM/recovery rank correlation below.40 or undefined')
    else:tm=np.array([]);rho=None
    if vals and (np.median([r['recovery']['robust_fixed_map_TM_similarity_fullN'] for r in vals])<.5):fails.append('disjoint VALIDATION native audit median below.50')
    if any(r['features']['global_confidence_product'] is None for r in native+vals):fails.append('required global confidence missing')
    # This gate is necessary, not sufficient: no fit/deployable coefficients unless
    # the small native controls also separate from declared exploitation probes.
    variants=[r for r in train_rows if r.get('kind')=='variant' and r.get('status')=='SUCCEEDED']
    if native and variants and all(r['features']['global_confidence_product'] is not None for r in native+variants):
        score=lambda r:r['features']['mean_pLDDT']/100*r['features']['global_confidence_product']
        if max(score(r) for r in variants)>=float(np.median([score(r) for r in native])):fails.append('variant exploitation probe reaches native median confidence-product')
    fit=None
    if len(native)>=6 and all(r['features']['sequence_confidence_product'] is not None for r in native):
        ordered=sorted(native,key=lambda r:r['sequence_sha256']);fitted=ordered[:4];hold=ordered[4:];x=np.array([r['features']['sequence_confidence_product'] for r in fitted]);y=np.array([r['recovery']['robust_fixed_map_TM_similarity_fullN'] for r in fitted]);beta=float(max(0.,x@y/max(float(x@x),1e-12)));pred=lambda r:float(np.clip(beta*r['features']['sequence_confidence_product'],0.,1.));hold_mae=float(np.mean([abs(pred(r)-r['recovery']['robust_fixed_map_TM_similarity_fullN']) for r in hold]));val_mae=float(np.mean([abs(pred(r)-r['recovery']['robust_fixed_map_TM_similarity_fullN']) for r in vals])) if vals and all(r['features']['sequence_confidence_product'] is not None for r in vals) else None
        fit={'coefficient':beta,'TRAIN_fit_ids':[r['sequence_sha256'] for r in fitted],'TRAIN_holdout_ids':[r['sequence_sha256'] for r in hold],'TRAIN_holdout_MAE':hold_mae,'VALIDATION_audit_MAE':val_mae,'formula':'clip(beta * mean_pLDDT/100 * pTM * fraction(long-range PAE<10A),0,1)','fit_excludes_VALIDATION':True}
        if hold_mae>.15:fails.append('held-out TRAIN confidence calibration MAE>.15')
        if val_mae is not None and val_mae>.20:fails.append('disjoint VALIDATION calibration audit MAE>.20')
    else:fails.append('insufficient valid TRAIN controls to fit reward coefficients')
    return {'GO':not fails,'candidate_TRAIN_only_fit':fit,'failures':fails,'TRAIN_native_successes':len(native),'VALIDATION_native_successes':len(vals),'TRAIN_median_native_TM':float(np.median(tm)) if len(tm) else None,'TRAIN_pTM_recovery_rank_correlation':rho,'fit_on_VALIDATION':False,'weights':{'sequence_confidence_product':fit['coefficient']} if fit and not fails else None,'center':{'sequence_confidence_product':0.},'scale':{'sequence_confidence_product':1.},'output_range':[0.,1.],'calibration_scope':'small protocol-specific native controls, not family independence or experimental thermodynamic quality'}
