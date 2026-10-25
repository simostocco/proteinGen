"""Exact necessary one-step feasibility; no sampling/search in the production shield.

UNKNOWN is permitted, never interpreted as infeasibility. A full valid trajectory
cannot traverse a prefix with more forced following contacts than its final slack.
The 1e-8 A guard is conservative for float64 coordinates in the qualified 1e5 A box.
Exact rational squared-distance comparisons avoid rounding at the certificate edge.
"""
import copy,math,runpy
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/d/Simone/proteinGen_data/e017_v94r3_one_step_feasibility_20261025_v1')
F=runpy.run_path(str(ROOT/'source/scripts/e017_v94r/packing.py'))
prefix_budget=F['prefix_budget'];UPPER=F['UPPER'];OLD=F['OLD']
DMAX=float(OLD['LIMITS']['adjacent_high_A']);GUARD_A=1e-8;COORD_BOX_A=1e5
CUTOFF=Fraction(8)-Fraction.from_float(DMAX)-Fraction.from_float(GUARD_A)

def coordinate_certificate(x,contacts=None,N=64):
    x=np.asarray(x,dtype=np.float64);n=len(x)
    out=dict(schema='EXACT_ONE_STEP_CONTACT_FEASIBILITY_1',status='UNKNOWN',reject=False,
        prefix_length=n,final_N=N,forced_next_contacts=None,contacts=contacts,
        maximum_pairs=115,maximum_valid_bond_A=DMAX,numerical_guard_A=GUARD_A,
        uses_native_future_coordinates=False,proves_property_reachability=False)
    if N!=64:out['reason']='unqualified_final_length';return out
    if x.shape!=(n,3) or not np.isfinite(x).all():out['reason']='nonfinite_or_invalid_shape';return out
    if n>=N:
        out.update(status='TERMINAL_NO_EXTENSION_REQUIRED',reason='never test a nonexistent following residue');return out
    if n==0:out['reason']='empty_prefix';return out
    if np.abs(x).max()>COORD_BOX_A:out['reason']='outside_qualified_numerical_box';return out
    if contacts is None:
        d=np.linalg.norm(x[:,None]-x[None,:],axis=-1);ii=np.arange(n)
        contacts=int(((d<8)&(ii[:,None]-ii[None,:]>=8)).sum());out['contacts']=contacts
    eligible=max(0,n-7);slack=115-contacts;out['slack_pairs']=slack
    if slack<0:
        # This is the existing packing filter's job, not a new one-step theorem.
        out['reason']='existing_prefix_invalid_under_old_packing_contract';return out
    if eligible<=slack:out.update(maximum_possible_next_contacts=eligible,reason='bound_cannot_exceed_remaining_budget');return out
    forced=[];cut=float(CUTOFF)
    # Broad float prescreen is conservative; each included contact is then
    # certified by exact arithmetic on the represented Cartesian coordinates.
    for j in np.where(np.linalg.norm(x[:eligible]-x[-1],axis=-1)<cut+1e-7)[0]:
        squared=sum((Fraction.from_float(float(a))-Fraction.from_float(float(b)))**2 for a,b in zip(x[j],x[-1]))
        if squared<CUTOFF*CUTOFF:forced.append(int(j))
    out.update(forced_next_contacts=len(forced),forced_old_residues_0based=forced,
        final_contact_degree_next_lower_bound=2*(contacts+len(forced))/64)
    if contacts+len(forced)>115:
        out.update(status='INFEASIBLE_CERTIFIED',reject=True,reason='no_next_valid_bond_can_preserve_final_contact_budget')
    else:out['reason']='necessary_bound_inconclusive'
    return out

def feasibility(state,next_aa=None,witness_geometry=None):
    result=coordinate_certificate(state.ca.detach().cpu().numpy(),N=state.N)
    if result['reject'] or result['status']=='TERMINAL_NO_EXTENSION_REQUIRED':return result
    if witness_geometry is not None and next_aa is not None:
        after,check=F['extend_checked'](state,next_aa,witness_geometry)
        if after is not None and check['accepted']:
            result.update(status='FEASIBLE_WITNESS',witness_geometry=list(witness_geometry),
                reason='one valid immediate extension; no eventual completion claim')
    return result

def extend_checked(state,aa,geometry):
    after,check=F['extend_checked'](state,aa,geometry)
    check=copy.deepcopy(check)
    if after is not None and check['accepted']:
        c=coordinate_certificate(after.ca.detach().cpu().numpy(),contacts=check['packing_after']['contacts'],N=after.N)
        check['one_step_feasibility']=c
        if c['reject']:
            check['accepted']=False;check['reasons'].append('certified_one_step_dead_end')
    return after,check

def proof_contract():
    return dict(N=64,base=F['proof_contract'](),maximum_valid_bond_A=DMAX,
        inequality='Cprefix + count(j<=n-8: distance(x[j],x[-1])+Dmax+guard <8) >115',
        strict=True,terminal_prefix_exempt=True,unknown_is_not_rejected=True,
        guarantee='excluded prefixes cannot be extended once while preserving original final contact bound and local bond limits',
        constraints_not_completely_solved=['clashes','global Rg','regional properties','longer-horizon reachability'],
        native_future_coordinates=False,neural_networks_added=0,production_searches=0,proposal_budget=8,
        finite_filter_likelihood='all raw proposals, rejected and accepted; fixed-AA categorical terms zero')
