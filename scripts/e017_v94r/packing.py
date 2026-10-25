"""Exact necessary final contact budget; append-only fixed-N C-alpha chains."""
import copy,math,runpy
from pathlib import Path
import numpy as np
BASE=Path('/mnt/d/Simone/proteinGen_data')
OLD=runpy.run_path(str(BASE/'e017_v92_prefix_feasibility_20261024_v1/source/scripts/e017_v92/common.py'))
UPPER=OLD['SPEC']['packing_limits']['64'][1]

def prefix_budget(state):
    assert state.N==64,'this qualified production arm is N64 only'
    x=state.ca.numpy();n=len(x);assert n<=64
    if not np.isfinite(x).all():return dict(finite=False,over_budget=True,contacts=None)
    ix=np.arange(n);d=np.linalg.norm(x[:,None]-x[None,:],axis=-1)
    C=int(((d<8)&(ix[:,None]-ix[None,:]>=8)).sum())
    # Do not use2*C/n: the final denominator is fixed at64, and old positions
    # never move when a suffix is appended. Each undirected edge counts twice.
    lower=2*C/64
    return dict(finite=True,contacts=C,final_N=64,final_contact_degree_lower_bound=lower,
        original_final_upper=UPPER,over_budget=lower>UPPER,maximum_pairs=115,
        slack_pairs=115-C,heuristic_risk=min(1.,C/115),
        risk_is_reachability_proof=False,current_prefix_length=n)

def extend_checked(state,aa,geometry):
    before=prefix_budget(state)
    if before['over_budget']:
        return None,dict(accepted=False,reasons=['existing_prefix_provably_over_final_contact_budget'],packing_before=before)
    after,check=OLD['extend_checked'](state,aa,geometry)
    check=copy.deepcopy(check);check['packing_before']=before
    if after is not None:
        budget=prefix_budget(after);check['packing_after']=budget
        if budget['over_budget']:
            check['accepted']=False;check['reasons'].append('provably_exceeds_final_contact_budget')
    return after,check

def proof_contract():
    return dict(radius_A=8.,strict_radius=True,minimum_sequence_separation=8,final_N=64,
        upper=UPPER,maximum_undirected_pairs=115,reject_from_pairs=116,
        invariant='old residue coordinates are immutable under suffix append; Cprefix<=Cfinal',
        necessary_not_sufficient=True,does_not_prove_future_property_success=True,
        unchanged_rules=['local supports','clash budget','packing interval','Rg interval','reward','property goals'],
        changed_rule='finite8 proposal acceptance additionally requires2*Cprefix/64<=original packing upper')
