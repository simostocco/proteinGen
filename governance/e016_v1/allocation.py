"""Typed diagnostic grouping and deterministic task-aware allocation. Never certifies a release."""
from dataclasses import dataclass
from collections import Counter
from hashlib import sha256
import json
from .framework import KINDS,MODES

@dataclass(frozen=True)
class Evidence:
    a:str
    b:str
    kind:str
    source_sha256:str
    completeness:str
    interpretation:str="UNRESOLVED"
    def __post_init__(self):
        if self.kind not in KINDS or self.completeness not in ["COMPLETE_PREDICATE","RETAINED_FOREST_ONLY","INCOMPLETE_ALIGNMENT_COVERAGE"]:raise ValueError("typed relationship evidence")
        if len(self.source_sha256)!=64 or set(self.source_sha256)-set('0123456789abcdef'):raise ValueError('evidence SHA')


def build_groups(identities,evidence,modes,*,scope,approval=None):
    if scope not in ['SYNTHETIC_ONLY','PROVISIONAL_REAL_DIAGNOSTIC']:raise ValueError('scope')
    if scope!='SYNTHETIC_ONLY':raise ValueError('reviewed biological predicates and evidence approval are unavailable in preparation v1; real diagnostic execution remains blocked')
    ids=sorted(identities);parent={i:i for i in ids}
    if len(ids)!=len(set(ids)):raise ValueError('duplicate identity')
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    context=[]
    for e in evidence:
        if e.a not in parent or e.b not in parent:raise ValueError('ownership')
        mode=modes.get(e.kind,'UNRESOLVED')
        if mode not in MODES or mode=='UNRESOLVED':raise ValueError('unreviewed relation')
        if e.kind in ['exact_identity','observation_ownership'] and mode!='HARD_CLOSURE':raise ValueError('identity invariant')
        if e.completeness!='COMPLETE_PREDICATE':raise ValueError('forest/incomplete evidence cannot prove counterfactual closure')
        if mode=='HARD_CLOSURE':
            a,b=find(e.a),find(e.b)
            if a!=b:parent[max(a,b)]=min(a,b)
        else:context.append(e)
    groups={}
    for i in ids:groups.setdefault(find(i),[]).append(i)
    return tuple(tuple(v) for _,v in sorted(groups.items())),context


@dataclass(frozen=True)
class Group:
    identity:str
    sequences:int
    joint_observations:int
    joint_identities:int
    length_counts:tuple=()
    method_counts:tuple=()
    joint_length_counts:tuple=()
    joint_method_counts:tuple=()
    @property
    def joint_groups(self):return int(self.joint_identities>0)
    def __post_init__(self):
        if min(self.sequences,self.joint_observations,self.joint_identities)<0 or self.joint_identities>self.sequences:raise ValueError('group counts')


def allocate(groups,reservations,acceptance,*,seed=42,ratios=(.8,.1,.1),scope='SYNTHETIC_ONLY',context_restrictions=()):
    # Real execution remains gated pending a reviewed schema beyond preparation v1.
    if scope!='SYNTHETIC_ONLY':raise ValueError('real allocation requires separately approved policy implementation')
    if len(ratios)!=3 or abs(sum(ratios)-1)>1e-12 or min(ratios)<=0:raise ValueError('ratios')
    required={'groups','joint_groups','sequences','joint_observations','joint_identities'}
    if set(acceptance)!=required or any(type(v) is not int or v<1 for v in acceptance.values()) or acceptance['groups']<2 or acceptance['joint_groups']<2:raise ValueError('explicit meaningful acceptance')
    gs={g.identity:g for g in groups}
    if len(gs)!=len(groups):raise ValueError('duplicate groups')
    fixed={}
    for group,split,role in reservations:
        if group not in gs or split not in ['TRAIN','VALIDATION','TEST','CALIBRATION']:raise ValueError('reservation')
        if role=='REWARD_DEVELOPMENT' and split!='CALIBRATION':return {'status':'INFEASIBLE_RESERVATION_ROLE','certified':False}
        if group in fixed and fixed[group]!=split:return {'status':'INFEASIBLE_RESERVATION_CONFLICT','certified':False}
        fixed[group]=split
    ordinary=[g for g in groups if fixed.get(g.identity)!='CALIBRATION'];n=sum(g.sequences for g in ordinary)
    available={k:sum(getattr(g,k) for g in ordinary) if k!='groups' else sum(g.sequences>0 for g in ordinary) for k in required}
    if any(available[k]<2*acceptance[k] for k in required):return {'status':'INFEASIBLE_NECESSARY_EVALUATION_COUNTS','available':available,'certified':False}
    assignment=dict(fixed);splits=['TRAIN','VALIDATION','TEST']
    def counts(split):
        selected=[gs[i] for i,v in assignment.items() if v==split and gs[i].sequences>0]
        return {'groups':len(selected),**{k:sum(getattr(g,k) for g in selected) for k in required-{'groups'}}}
    order=sorted((g for g in ordinary if g.identity not in fixed),key=lambda g:(-g.sequences,sha256(f'{seed}|{g.identity}'.encode()).hexdigest()))
    for g in order:
        choices=[]
        for k,split in enumerate(splits):
            c=counts(split)
            # Explicit evaluation deficits get priority; no singleton repair or group breaking.
            benefit=sum(min(acceptance[x]-c[x],1 if x=='groups' else getattr(g,x))/acceptance[x] for x in required if c[x]<acceptance[x]) if split!='TRAIN' else 0
            delta=((c['sequences']+g.sequences-ratios[k]*n)**2-(c['sequences']-ratios[k]*n)**2)/max(n,1)
            choices.append((-benefit,delta,k,split))
        assignment[g.identity]=min(choices)[3]
    achieved={split:counts(split) for split in splits}
    strata={}
    for split in splits:
        lengths=Counter();methods=Counter();joint_lengths=Counter();joint_methods=Counter()
        for identity,assigned in assignment.items():
            if assigned==split:
                lengths.update(dict(gs[identity].length_counts));methods.update(dict(gs[identity].method_counts));joint_lengths.update(dict(gs[identity].joint_length_counts));joint_methods.update(dict(gs[identity].joint_method_counts))
        strata[split]={'length_counts':dict(lengths),'method_counts':dict(methods),'joint_length_counts':dict(joint_lengths),'joint_method_counts':dict(joint_methods)}
    context_conflicts=[]
    for a,b,rule in context_restrictions:
        if a not in assignment or b not in assignment:raise ValueError('context owner missing')
        if assignment[a]!=assignment[b]:context_conflicts.append({'a':a,'b':b,'restriction':rule})
    violations={sp:[k for k in required if achieved[sp][k]<acceptance[k]] for sp in ['VALIDATION','TEST']}
    if not achieved['TRAIN']['sequences']:violations['TRAIN']=['EMPTY_TRAIN']
    if context_conflicts:violations['CONTEXT']=['EXPLICIT_CONTEXT_RESTRICTION_VIOLATED']
    return {'status':'DIAGNOSTIC_PASS_NOT_CERTIFICATION' if not any(violations.values()) else 'NO_VALID_ALLOCATION_FOUND','assignments':assignment,'counts':achieved,'strata':strata,'context_conflicts':context_conflicts,'coverage_acceptance':'UNRESOLVED_NOT_CERTIFIED','actual_ratios':{sp:achieved[sp]['sequences']/n for sp in splits},'denominator':'ordinary identities after explicit calibration reservation (synthetic only)','violations':violations,'certified':False}


def relationship_inventory(root):
    """Read existing summaries only; never reconstructs relationships or aligns sequences."""
    from pathlib import Path
    r=Path(root);freeze=json.load(open(r/'homology/POOL_FREEZE.json'));screen=json.load(open(r/'homology/GLOBAL_SCREEN_COMPLETE.json'))
    return {'nodes':freeze['node_sequences'],'source_and_containment_attempts':freeze['exact_source_edge_counts'],'alignment_counts':screen['counts'],'existing_alignment_scope':screen['edge_storage'],'same_entry':'RECOVERABLE_FROM_SOURCE_ENTITY_RECORDS_NOT_EXPANDED','containment':'RECOVERABLE_FROM_FULL_SEQUENCES_AND_FRAGMENT_OWNERS_NOT_EXPANDED','similarity':'INCOMPLETE_FOR_REMOVING_EXISTING_CLOSURE_CONSTRAINTS','missing_alignment_candidates':screen['counts']['already_connected_pairs_skipped'],'recovery':{'procedure':'fresh all-node queries against frozen global target, without prior component skipping, followed by approved predicates; preserve native per-batch outputs','automatic_execution':False,'memory_and_scratch_peak':'UNKNOWN_REQUIRES_BOUNDED_RESOURCE_PILOT','existing_prefilter_pairs':screen['counts']['prefilter_pairs_seen']},'real_counterfactual_ready':False}
