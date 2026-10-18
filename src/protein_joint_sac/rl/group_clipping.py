"""Explicit sequence-first gradient transform; no whole-model clipping."""
import math
import torch
VERSION='sequence_first_groups_v1'

def groups(model):
    out={'S':[],'A':[],'G':[]};seen=set();names={k:[] for k in out}
    for n,p in model.named_parameters(remove_duplicate=False):
        if not p.requires_grad:continue
        if id(p) in seen:raise ValueError('aliased trainable parameter: '+n)
        seen.add(id(p))
        if n.startswith('encoder.'):k='S'
        elif n.startswith('aa_head.'):k='A'
        elif n.startswith(('geom.','selected_aa.')):k='G'
        else:raise ValueError('unclassified trainable parameter: '+n)
        out[k].append((n,p));names[k].append(n)
    if any(not v for v in out.values()):raise ValueError('empty gradient group')
    return out,names

def norm(vectors):
    vals=[v.detach().float().square().sum() for v in vectors if v is not None]
    if not vals:return 0.
    n=float(torch.stack(vals).sum().sqrt())
    if not math.isfinite(n):raise FloatingPointError('nonfinite gradient; emergency stop')
    return n

def combine(model,aa,limits=None,ratio=.5):
    gs,_=groups(model)
    for n,p in model.named_parameters():
        for v in [p.grad,aa.get(n)]:
            if v is not None and not bool(torch.isfinite(v).all()):raise FloatingPointError('nonfinite gradient '+n)
    an=norm([aa[n] for n,p in gs['S']]);gn=norm([p.grad for n,p in gs['S']])
    alpha=min(1.,ratio*an/max(gn,1e-30))
    dot=sum(float((aa[n].float()*p.grad.float()).sum()) for n,p in gs['S'] if aa[n] is not None and p.grad is not None)
    aa_head=norm([aa[n] for n,p in gs['A']])
    if norm([p.grad for n,p in gs['A']])!=0:raise ValueError('auxiliary gradient bypass into AA-only head')
    with torch.no_grad():
        for n,p in model.named_parameters():
            a=aa[n]
            if p.grad is not None:p.grad.mul_(alpha)
            if a is not None:
                if p.grad is None:p.grad=a.clone()
                else:p.grad.add_(a)
    before={k:norm([p.grad for n,p in v]) for k,v in gs.items()}
    factors={k:(min(1.,limits[k]/max(before[k],1e-30)) if limits else 1.) for k in gs}
    with torch.no_grad():
        for k,v in gs.items():
            for n,p in v:
                if p.grad is not None:p.grad.mul_(factors[k])
    after={k:norm([p.grad for n,p in v]) for k,v in gs.items()}
    return {'AA_shared_L2':an,'weighted_geometry_shared_raw_L2':gn,'geometry_shared_scale':alpha,
            'geometry_shared_capped_L2':alpha*gn,'geometry_to_AA_shared_ratio':alpha*gn/max(an,1e-30),
            'AA_geometry_shared_cosine':dot/max(an*gn,1e-30),'AA_head_only_L2':aa_head,
            'group_before':before,'group_after':after,'group_scale':factors,
            'AA_shared_after_S_clip_L2':an*factors['S'],
            'old_whole_model_norm1_scale':min(1.,1/max(math.sqrt(sum(x*x for x in before.values())),1e-30)),
            'policy':VERSION,'whole_model_clip_applied':False}
