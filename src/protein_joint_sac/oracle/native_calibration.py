"""Offline native-panel preparation/scoring; no inference and no calibrated reward weights."""
import hashlib,math
import torch
from ..contracts import OracleStatus


def prepare_controls(panel):
    rows=[]
    for c in panel['controls']:
        seq=c['full_deposited_canonical_sequence'];missing=c['missing_CA_positions']
        if hashlib.sha256(seq.encode()).hexdigest()!=c['sequence_sha256'] or len(seq)!=c['full_length']:raise ValueError('native control identity')
        if len(set(missing))!=len(missing) or any(not 1<=i<=len(seq) for i in missing):raise ValueError('native missing position map')
        rows.append({'id':c['id'],'sequence_sha256':c['sequence_sha256'],'full_length':len(seq),'observed_CA_length':c['observed_CA_length'],'source_sha256':c['source_sha256'],'role':'NATIVE_REWARD_DEVELOPMENT_ONLY_PENDING_APPROVED_ROLE_POLICY','reference_export':'REQUIRED_BEFORE_SCORING','quality_acceptance':'UNRESOLVED_NOT_THERMODYNAMIC_LABEL'})
    return {'controls':rows,'real_AF_calls':0,'length500_resource_evidence':'NOT_ESTABLISHED_BY_447_CONTROL','calibration_pass':False}


def score_native(control,reference_coordinates,result):
    if result.status!=OracleStatus.SUCCEEDED or result.sequence_sha256!=control['sequence_sha256']:raise ValueError('matching completed native prediction')
    n=control['full_length']
    if len(reference_coordinates)!=n or result.predicted_ca is None or len(result.predicted_ca)!=n:raise ValueError('full residue mapping')
    mask=torch.tensor([x is not None for x in reference_coordinates])
    if {i+1 for i,x in enumerate(reference_coordinates) if x is None}!=set(control['missing_CA_positions']) or mask.sum()<3:raise ValueError('reference missing mask')
    # Select observed coordinates directly; no interpolation or missing-atom coordinates.
    x=torch.tensor([x for x in reference_coordinates if x is not None],dtype=torch.float64)
    y=torch.tensor(result.predicted_ca,dtype=torch.float64)[mask]
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('finite mapped coordinates')
    x=x-x.mean(0);y=y-y.mean(0);u,_,vh=torch.linalg.svd(x.T@y);d=torch.eye(3,dtype=x.dtype);d[-1,-1]=torch.det(u@vh);err=torch.linalg.vector_norm(x@(u@d@vh)-y,dim=-1);d0=max(.5,1.24*(n-15)**(1/3)-1.8)
    return {'full_length':n,'observed_reference_positions':int(mask.sum()),'coordinate_coverage':float(mask.sum()/n),'observed_CA_Kabsch_RMSD_A':float(err.square().mean().sqrt()),'full_length_normalized_fixed_mapping_TM_similarity':float((1/(1+(err/d0).square())).sum()/n),'optimized_TM_score':None,'plddt':result.plddt,'ptm':result.ptm,'PAE_available':result.pae is not None,'role':'NATIVE_DIAGNOSTIC_ONLY','foldability_label':None,'calibration_pass':False}
