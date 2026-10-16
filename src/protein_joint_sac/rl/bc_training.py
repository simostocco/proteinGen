"""Supervised prefix trainer. Real data require certified governance admission; fixtures never certify data."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import defaultdict
from pathlib import Path
from functools import lru_cache
import hashlib, json, math, random, shutil, platform
import numpy as np
import torch
import pyarrow.parquet as pq
from ..contracts import AA, BuildState, GEOMETRY_VERSION, digest, d_path
from ..chemistry.reference_adapter import ref as chemistry_ref
from ..models.dense_geometry_attention import ModelConfig
from ..models.joint_actor import JointActor
from ..models.sequence_only_actor import SequenceOnlyActor
from ..data.certified_unified import verify_release, canonical_teacher_prefix, native_coordinates
from .checkpointing import save_atomic, load_checked


def code_hash():
    root = Path(__file__).resolve().parents[1]
    files={str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(root.rglob("*.py"))}
    entry=root.parents[1]/"scripts/e016_v3/train_bc.py"
    files["ENTRYPOINT_train_bc.py"]=hashlib.sha256(entry.read_bytes()).hexdigest()
    return digest(files)


def rng_state():
    a=np.random.get_state()
    return {"python":random.getstate(),"numpy":(a[0],a[1].tolist(),a[2],a[3],a[4]),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(s):
    random.setstate(s["python"])
    n=s["numpy"]; np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(s["torch"])
    if s["cuda"]:
        if not torch.cuda.is_available(): raise ValueError("CUDA RNG cannot resume on unavailable device")
        torch.cuda.set_rng_state_all(s["cuda"])


@dataclass(frozen=True)
class BCConfig:
    version: str = "bc-prefix-v1"
    mode: str = "joint"
    width: int = 128
    heads: int = 4
    layers: int = 4
    chemistry: bool = True
    seed: int = 42
    batch_size: int = 2
    learning_rate: float = 0.0003
    aa_weight: float = 1.0
    geometry_weight: float = 1.0
    clip_norm: float = 1.0
    scheduler_updates: int = 100
    device: str = "cpu"
    scope: str = "ENGINEERING_FIXTURE"
    precision: str = "FP32_MODEL_FP64_GEOMETRY"
    population_view: str = "MATCHED_JOINT"
    validation_identity_limit: int = 32
    def __post_init__(self):
        if self.version!="bc-prefix-v1" or self.mode not in ["sequence","joint"] or self.scope not in ["ENGINEERING_FIXTURE","CERTIFIED_PILOT"]: raise ValueError("explicit validated BC scope/mode")
        if self.validation_identity_limit<1 or self.batch_size<1 or self.scheduler_updates<1 or self.device not in ["cpu","cuda"] or self.precision!="FP32_MODEL_FP64_GEOMETRY": raise ValueError("BC resource/precision config")
        if any(not math.isfinite(v) or v<=0 for v in [self.learning_rate,self.aa_weight,self.geometry_weight,self.clip_norm]): raise ValueError("finite positive BC scales")
        if self.population_view not in ["MATCHED_JOINT","ALL_SEQUENCES"] or self.mode=="joint" and self.population_view!="MATCHED_JOINT":raise ValueError("explicit comparison population")
        ModelConfig(width=self.width,heads=self.heads,layers=self.layers,chemistry=self.chemistry)


_CERTIFIED_ADMISSION = object()


class Population:
    def __init__(self, rows, scope, membership=None, *, _admission=None):
        if scope=="CERTIFIED_PILOT" and _admission is not _CERTIFIED_ADMISSION:raise ValueError("use verified certified population factory")
        if scope not in ["ENGINEERING_FIXTURE","CERTIFIED_PILOT"]: raise ValueError("population scope")
        self.rows=tuple(sorted(rows,key=lambda r:r["observation_id"]))
        self.scope=scope; self.groups=defaultdict(lambda:defaultdict(list))
        seen=set(); seq_splits={}; group_splits={}
        for i,r in enumerate(self.rows):
            seq=r["sequence"]; h=hashlib.sha256(seq.encode()).hexdigest()
            if not 20<=len(seq)<=500 or set(seq)-set(AA) or r["sequence_hash"]!=h: raise ValueError("canonical complete sequence identity")
            if r["observation_id"] in seen: raise ValueError("duplicate observation")
            seen.add(r["observation_id"])
            if r["split"] not in ["TRAIN","VALIDATION"] or r["role"]!={"TRAIN":"TRAINING","VALIDATION":"DEVELOPMENT"}[r["split"]]: raise ValueError("no TEST/calibration in BC population")
            for mapping,key in [(seq_splits,h),(group_splits,r["homology_group"])]:
                if key in mapping and mapping[key]!=r["split"]: raise ValueError("identity/group crosses splits")
                mapping[key]=r["split"]
            if r.get("fragment",False) or r.get("label_positions",list(range(1,len(seq)+1)))!=list(range(1,len(seq)+1)): raise ValueError("fragment/full-polymer correspondence")
            if scope=="ENGINEERING_FIXTURE" and not r["observation_id"].startswith("toy-"): raise ValueError("real record cannot impersonate fixture")
            self.groups[r["split"]][h].append(i)
        if not self.groups["TRAIN"] or not self.groups["VALIDATION"]: raise ValueError("TRAIN/VALIDATION required")
        identities=[{k:v for k,v in r.items() if k!="coordinates"}|({"coordinates_sha":digest(torch.as_tensor(r["coordinates"]))} if "coordinates" in r else {}) for r in self.rows]
        self.membership=digest({"rows":identities,"release":membership,"scope":scope})

    @classmethod
    def certified(cls, reference, mode, view="MATCHED_JOINT"):
        if view not in ["MATCHED_JOINT","ALL_SEQUENCES"] or mode=="joint" and view!="MATCHED_JOINT":raise ValueError("comparison population")
        m=verify_release(reference)
        gate=m.get("governance_acceptance",{})
        required=["approved_policy","complete_relationship_evidence","independent_cross_split_pass","evaluation_acceptance_pass","control_roles_pass","mapping_pass"]
        if not all(gate.get(k) is True for k in required) or not gate.get("policy_sha256") or not gate.get("acceptance_sha256"): raise ValueError("reviewed governance acceptance required; no implicit upgrade")
        for k in ["policy_sha256","acceptance_sha256"]:
            from ..contracts import sha
            sha(gate[k])
        root=Path(reference.root); rows=[]
        name="joint_bc_eligibility.parquet" if view=="MATCHED_JOINT" else "unique_sequences.parquet"
        for b in pq.ParquetFile(root/name).iter_batches(batch_size=256,use_threads=False):
            for r in b.to_pylist():
                if r["split"] not in ["TRAIN","VALIDATION"] or r["role"] not in ["TRAINING","DEVELOPMENT"]: continue
                if view=="MATCHED_JOINT" and not r["eligible"] or view=="ALL_SEQUENCES" and not r["lm_eligible"]: continue
                if view=="ALL_SEQUENCES":r={**r,"observation_id":"sequence:"+r["sequence_id"],"source_sha":reference.manifest_sha256}
                rows.append(r)
        return cls(rows,"CERTIFIED_PILOT",asdict(reference),_admission=_CERTIFIED_ADMISSION)

    @lru_cache(maxsize=8)
    def coordinates(self,i):
        r=self.rows[i]
        x=torch.as_tensor(r["coordinates"] if "coordinates" in r else native_coordinates(r),dtype=torch.float64)
        if x.shape!=(len(r["sequence"]),3) or not torch.isfinite(x).all(): raise ValueError("full mapped finite CA required")
        return x

    def target(self,i,t,mode,device):
        r=self.rows[i]; seq=r["sequence"]
        if not 0<=t<len(seq): raise ValueError("prefix boundary")
        if mode=="sequence":
            # Coordinates are absent, not measured zero atoms. No native geometry read.
            state=BuildState(tuple(seq[:t]),torch.zeros(t,3,dtype=torch.float64),len(seq),torch.zeros(t,dtype=torch.bool))
            return state.to(device), AA.index(seq[t]), None
        target=canonical_teacher_prefix(seq,self.coordinates(i),t,r["source_sha"])
        return target.state.to(device),AA.index(target.next_action.residue),target.next_action.geometry.to(device)


class PrefixSampler:
    def __init__(self,population,seed):
        self.population=population;self.rng=random.Random(seed);self.cursor=0;self.exposure=0
        self.identities=sorted(population.groups["TRAIN"])
    def draw(self,batch):
        out=[]
        for _ in range(batch):
            h=self.rng.choice(self.identities);i=self.rng.choice(self.population.groups["TRAIN"][h]);t=self.rng.randrange(len(self.population.rows[i]["sequence"]))
            out.append((i,t));self.cursor+=1;self.exposure+=t+1
        return out
    def state_dict(self):return {"rng":self.rng.getstate(),"cursor":self.cursor,"exposure":self.exposure,"identities":self.identities}
    def load_state_dict(self,s):
        if s["identities"]!=self.identities:raise ValueError("sampler membership changed")
        self.rng.setstate(s["rng"]);self.cursor=s["cursor"];self.exposure=s["exposure"]


class BCTrainer:
    def __init__(self, config, population):
        if config.scope!=population.scope: raise ValueError("fixture/real scope mismatch")
        self.config=config;self.population=population
        random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
        if config.device=="cuda":torch.cuda.manual_seed_all(config.seed)
        c=ModelConfig(width=config.width,heads=config.heads,layers=config.layers,chemistry=config.chemistry)
        self.model=(JointActor(c) if config.mode=="joint" else SequenceOnlyActor(c)).to(config.device)
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=config.learning_rate)
        self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,config.scheduler_updates)
        self.sampler=PrefixSampler(population,config.seed+1);self.update=0;self.best_validation=float("inf")
        self.lineage={"config":asdict(config),"membership":population.membership,"geometry_version":GEOMETRY_VERSION,"chemistry":asdict(chemistry_ref()),"code_sha256":code_hash(),"normalizers":None,"scaler":None,"runtime":{"torch":str(torch.__version__),"numpy":np.__version__,"cuda":torch.version.cuda,"python":platform.python_version()}}
    def loss(self,i,t):
        state,label,g=self.population.target(i,t,self.config.mode,self.config.device)
        h=self.model.encode(state); aa=torch.nn.functional.cross_entropy(self.model.aa_head(h)[None],torch.tensor([label],device=h.device))
        geom=aa*0;components=0
        if g is not None:
            geom=-self.model.geometry_distribution(state,AA[label],h).log_prob(g)
            components=sum((t>=1,t>=2,t>=3))
        total=self.config.aa_weight*aa+self.config.geometry_weight*geom
        if not torch.isfinite(total):raise FloatingPointError("nonfinite BC loss")
        return total,{"aa_nll":float(aa.detach()),"geometry_nll_sum":float(geom.detach()),"geometry_components":components}
    def step(self):
        if self.update>=self.config.scheduler_updates: raise ValueError("configured update budget exhausted")
        if self.config.scope=="CERTIFIED_PILOT" and self.update>=100:raise ValueError("real pilot ceiling100; production protocol not authorized")
        self.model.train(); self.optimizer.zero_grad(set_to_none=True); samples=self.sampler.draw(self.config.batch_size);metrics=[]
        # Literal variable-length prefixes; gradient accumulation avoids future padding/pair features.
        for i,t in samples:
            loss,m=self.loss(i,t);(loss/len(samples)).backward();metrics.append(m)
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.config.clip_norm,error_if_nonfinite=True)
        self.optimizer.step();self.scheduler.step();self.update+=1
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):raise FloatingPointError("nonfinite parameters")
        return {"update":self.update,"samples":samples,"aa_nll":sum(m["aa_nll"] for m in metrics)/len(metrics),"geometry_nll_sum":sum(m["geometry_nll_sum"] for m in metrics)/len(metrics),"grad_norm":float(norm),"exposure":self.sampler.exposure}
    @torch.no_grad()
    def validate(self,prefixes_per_observation=4):
        if prefixes_per_observation<1:raise ValueError("validation coverage")
        self.model.eval(); aa=geom=mass=0.;n=0;dof=0
        all_groups=self.population.groups["VALIDATION"]
        groups={h:all_groups[h] for h in sorted(all_groups)[:self.config.validation_identity_limit]}
        for h,indices in sorted(groups.items()):
            for i in indices:
                L=len(self.population.rows[i]["sequence"]);ts=sorted(set(np.linspace(0,L-1,min(prefixes_per_observation,L),dtype=int).tolist()));w=1/(len(groups)*len(indices)*len(ts))
                for t in ts:
                    _,m=self.loss(i,t);aa+=w*m["aa_nll"];geom+=w*m["geometry_nll_sum"];mass+=w;n+=1;dof+=m["geometry_components"]
        score=self.config.aa_weight*aa+self.config.geometry_weight*geom;self.best_validation=min(score,self.best_validation)
        return {"aa_nll":aa,"aa_perplexity":math.exp(min(aa,700)),"geometry_nll_sum":geom,"selection_loss":score,"best_selection_loss":self.best_validation,"validation_identities":len(groups),"available_validation_identities":len(all_groups),"prefixes":n,"geometry_components":dof,"weight_mass":mass,"weighting":"uniform identity / observation / selected prefix","scope":self.config.scope}
    def checkpoint(self,path):
        path=d_path(path)
        if path.exists():raise ValueError("immutable checkpoint path exists")
        if shutil.disk_usage(path.parent).free<10*2**30+2**20:raise RuntimeError("disk reserve")
        return save_atomic(path,{"lineage":self.lineage,"model":self.model.state_dict(),"optimizer":self.optimizer.state_dict(),"scheduler":self.scheduler.state_dict(),"sampler":self.sampler.state_dict(),"rng":rng_state(),"update":self.update,"best_validation":self.best_validation})
    def resume(self,path):
        s=load_checked(path)
        if s["lineage"]!=self.lineage:raise ValueError("config/code/data/chemistry/geometry resume mismatch")
        self.model.load_state_dict(s["model"],strict=True);self.optimizer.load_state_dict(s["optimizer"]);self.scheduler.load_state_dict(s["scheduler"]);self.sampler.load_state_dict(s["sampler"]);self.update=s["update"];self.best_validation=s["best_validation"];restore_rng(s["rng"])
