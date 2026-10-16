"""C-alpha assessment only: no atomistic or thermodynamic labels."""
import math,time,resource
import torch
from ..contracts import AA,BuildState,GeneratedProtein
from ..geometry.ca_builder import append,inverse,action_from_positions,replay


def trace_metrics(protein):
    x=protein.state.ca.detach().cpu().double();n=len(x);b=x[1:]-x[:-1];d=torch.linalg.vector_norm(b,dim=-1)
    g=[inverse(x[:i+1])[0] for i in range(3,n)]
    torsion=torch.stack([a[2] for a in g]) if g else torch.empty(0)
    angles=torch.stack([inverse(x[:i+1])[0][1] for i in range(2,n)])
    dist=torch.cdist(x,x);sep=(torch.arange(n)[:,None]-torch.arange(n)[None,:]).abs();nonlocal_pair=torch.triu(sep>=3,diagonal=1)
    def stats(v):return {"mean":float(v.mean()),"min":float(v.min()),"max":float(v.max())} if len(v) else None
    reconstructed=replay(n,tuple(action_from_positions(a,x[:i+1]) for i,a in enumerate(protein.sequence)))
    roundtrip=float((reconstructed.ca-x).abs().max())
    return {"geometry_roundtrip_max_error_A":roundtrip,"length":n,"adjacent_CA_A":stats(d),"virtual_angle_rad":stats(angles),"signed_torsion_rad":stats(torsion),"positive_torsion_fraction":float((torsion>0).double().mean()) if len(torsion) else None,"degenerate_bonds":int((d<1e-10).sum()),"CA_nonlocal_clash_pairs_below_2A":int(((dist<2)&nonlocal_pair).sum()),"nonlocal_contacts_below_8A":int(((dist<8)&nonlocal_pair).sum()),"radius_of_gyration_A":float((x-x.mean(0)).square().sum(-1).mean().sqrt()),"assessment":"CA_PROXIES_ONLY_NOT_FOLDING_EVIDENCE"}


@torch.no_grad()
def generate(model,N,mode):
    device=next(model.parameters()).device
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    start=time.monotonic();state=BuildState.empty(N,device=device);seq=[]
    for _ in range(N):
        if mode=="joint":
            action,_,_=model.sample(state);state=append(state,action)
        else:
            i=int(torch.distributions.Categorical(logits=model.logits(state)).sample());seq.append(AA[i]);t=len(seq)
            state=BuildState(tuple(seq),torch.zeros(t,3,dtype=torch.float64,device=device),N,torch.zeros(t,dtype=torch.bool,device=device))
    if device.type=="cuda":torch.cuda.synchronize()
    elapsed=time.monotonic()-start
    sequence="".join(state.residues)
    return {"sequence":sequence,"seconds":elapsed,"residues_per_second":N/max(elapsed,1e-9),"CPU_RSS_highwater":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"CUDA_peak_allocated":torch.cuda.max_memory_allocated() if device.type=="cuda" else None,"geometry":trace_metrics(GeneratedProtein(state)) if mode=="joint" else None}


def sequence_diversity(sequences,reference_sequences=()):
    n=len(sequences);unique=set(sequences)
    return {"candidates":n,"unique":len(unique),"exact_duplicate_fraction":1-len(unique)/n if n else None,"exact_reference_overlap":len(unique&set(reference_sequences)),"nearest_homology_novelty":"NOT_MEASURED_NO_NEW_ALIGNMENTS"}
