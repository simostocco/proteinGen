"""Artificial 20-residue engineering traces, never a biological dataset or holdout."""
import hashlib,torch
from ..contracts import AA,BuildAction
from ..geometry.ca_builder import replay,mask_for_step

def records():
    rows=[]
    for j in range(6):
        seq=AA[j:]+AA[:j]
        actions=tuple(BuildAction(a,torch.tensor([3.8 if i else 0.,1.9 if i>=2 else 0.,.4+j*.03 if i>=3 else 0.],dtype=torch.float64),mask_for_step(i)) for i,a in enumerate(seq))
        rows.append({"observation_id":f"toy-observation-{j}","sequence":seq,"sequence_hash":hashlib.sha256(seq.encode()).hexdigest(),"source_sha":hashlib.sha256(f"toy-source-{j}".encode()).hexdigest(),"homology_group":f"toy-group-{j}","split":"TRAIN" if j<4 else "VALIDATION","role":"TRAINING" if j<4 else "DEVELOPMENT","coordinates":replay(20,actions).ca.tolist(),"label_positions":list(range(1,21)),"fragment":False})
    return rows
