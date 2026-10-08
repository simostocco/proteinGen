"""D:-only, identity-held-out observability of frozen local teacher actions."""
import os
import sys
from pathlib import Path

REPO=Path('/mnt/d/Simone/proteinGen')
ROOT=REPO/'outputs/e010_student_direction_observability_v1'


def storage():
    for k in ('TMPDIR','TMP','TEMP'):os.environ[k]=str(ROOT/'tmp')
    for k in ('XDG_CACHE_HOME','MPLCONFIGDIR','CUDA_CACHE_PATH','TORCH_HOME','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR'):os.environ[k]=str(ROOT/'cache')
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    sys.dont_write_bytecode=True


storage()
import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import time
import numpy as np
import torch
import e010_nonprivileged_student_v1 as old
storage()
sys.path.insert(0,str(ROOT/'third_party'))
AA='ACDEFGHIKLMNPQRSTVWY'
PARENT=REPO/'outputs/e010_student_feasibility_margin_v1'
PROTOCOL=dict(experiment='e010_student_direction_observability_v1',parent_commit='1de4619',
    student_commit='fc76211',teacher_commit='6d1ed30',seed=104081,
    representations='exact float32 41-channel input; float64 frame; exact pre-head 64-dimensional hidden state; full two-block +/-6-residue input window reconstructed from archived channels',
    units='residue-centered equivariant decisions, aggregated by protein/state; extract every actual eligible teacher transition and explicit logical K16 zero tails',
    sampling='8 query residues and 4 reference residues per logical state selected by SHA256(identity,condition,step,residue), without looking at actions; equal states/identities; reference uses frozen TRAIN split only',
    identity_holdout='exclude query identity, split group and sequence cluster in all searches, including TRAIN leave-group-out queries; validation/test never in reference index',
    neighbor_k=8,candidate_k=256,hnsw_M=32,efConstruction=128,efSearch=512,
    distance='squared Euclidean on TRAIN-panel standardized complete 13x41 receptive input or standardized frozen latent; standard deviation floor 1e-3',
    index='deterministic single-thread CPU HNSW insertion, fixed seed; ANN audit against exact Flat on 512 SHA-ranked queries; raise efSearch to 1024 then 2048 if recall@8 below 0.90; never tune using labels',
    contexts=['condition','residue_identity','sequence_local','sequence_condition','step','coarse_scalar_latent','coarse_output_context'],
    sequence_local='Hamming distance over +/-3 primary sequence, padding token; sequence_condition adds exact condition preference',
    coarse_scalar_latent='frozen Phase4B global-model last-block scalar latent, computed only from archived denoiser prediction and mask, 416 dimensions; no target or cache error/confidence metrics',
    coarse_output_context='local-frame initial global correction P0-source, accumulated P_t-P0, and local source features; all available outputs/history, no X',
    B_control='rerank fixed geometric/latent candidates by context distance; select 8 distinct reference identities; for each chosen pair take a different-identity control with nearest geometric distance; teacher labels/margins never enter matching',
    geometry_control='report selected/control distance mismatch; main controlled effects only pairs with relative squared-distance mismatch <=0.10; report coverage and effects by base-distance quartile',
    direction='cosine of original archived teacher vectors in corresponding local frames; norms <=1e-8 have undefined direction, reported separately and excluded from directional statistics',
    variance='unit-vector prediction SSE against mean neighbor unit vector; context explained fraction=1-SSE_context/SSE_distance_matched_control; direction dispersion=1-||mean neighbor unit vectors||^2, not Shannon entropy',
    gate='validation only: context improves median cosine >=0.05 versus distance-matched control, reduces opposition rate >=20% relative, explains >=3% directional SSE, matched coverage >=0.75, bootstrap identity paired-mean-cosine 95% lower bound >0; positive mean-cosine effect in >=4/5 length bins and all 3 conditions',
    bootstrap_identity_replicates=1000,test_never_selects_context=True,
    phase_C='only if gate passes; same width64/blocks2 and 118723 trainable parameters; original identity-first sampler/AdamW/cap/update ladder; geometry baseline, condition, sequence, sequence+condition, and single winning coarse family if informative; fixed context embeddings injected before existing blocks, no added trainable capacity; raw validation repair early stopping and raw K16 safety primary, no X shield',
    prohibited_context=['X','target_distances','teacher_gradient','privileged_constraints','denoiser_error_rmse','normalized_x0_hat_rmse','sequence_hash_or_identity_as_network_feature'],
    no_SLSQP=True,no_new_teacher_labels=True,no_marginized_training=True,no_full_corpus=True,no_capacity_increase=True,workers=8)


def write(path,obj):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(temporary,path)


def rank(value):return hashlib.sha256(f'{PROTOCOL["seed"]}|{value}'.encode()).hexdigest()


def rows():return json.loads((old.PARENT/'selection.json').read_text())['identities']


def prepare():
    for d in ('','tmp','cache','logs','representations','panel','neighbors','code'):(ROOT/d).mkdir(parents=True,exist_ok=True)
    completion=json.loads((PARENT/'COMPLETED.json').read_text())
    assert completion['status']=='complete' and not completion['phase_C_run']
    for p,digest in completion['files'].items():assert old.sha(p)==digest,p
    for p,digest in completion['source_sha256'].items():assert old.sha(p)==digest,p
    assert old.sha(old.ROOT/'COMPLETED.json')=='8113a463cd66b2baa63433a3f473ffc11bda96b977ee2336c74185257a53d601'
    if (ROOT/'protocol.json').exists():assert json.loads((ROOT/'protocol.json').read_text())==PROTOCOL
    else:write(ROOT/'protocol.json',PROTOCOL)
    manifest=dict(parent_commit='1de4619',parent_completion_sha256=old.sha(PARENT/'COMPLETED.json'),
        parent_files_verified=len(completion['files']),student_model_sha256=old.sha(old.ROOT/'models/imitation_best.pt'),
        teacher_completion_sha256=old.sha(old.PARENT/'COMPLETED.json'),splits_sha256=old.sha(old.ROOT/'splits.json'),
        split_identity_sha256=json.loads((old.ROOT/'splits.json').read_text())['identity_list_sha256'])
    write(ROOT/'parent_freeze.json',manifest)
    print(json.dumps(dict(recovery='clean at 1de4619; completed feasibility pilot verified and frozen; no marginized training; inference context excludes cache target-error metrics',**manifest)),flush=True)


def initialize():
    old.initialize();storage();torch.set_num_threads(1)


def init_worker():
    global STUDENT,COARSE
    initialize()
    STUDENT=old.Student().eval()
    STUDENT.load_state_dict(torch.load(old.ROOT/'models/imitation_best.pt',map_location='cpu',weights_only=False)['model'])
    from protein_distance_diffusion.models.e010_hybrid_local import load_frozen_hybrid
    path=REPO/'reports/experiments/E010_global_equivariant_expressivity/phase4b_real_denoiser_v1/phase4b_training_v1.final/latest.pt'
    COARSE=load_frozen_hybrid(path).global_model.eval()


@torch.no_grad()
def latent(p,mask):
    rep=old.local_representation(p,mask);valid=rep['eligible']
    h=STUDENT.local.input(rep['features'].float())*valid[...,None]
    for block in STUDENT.local.blocks:h=block(h,valid)
    return rep,h


@torch.no_grad()
def coarse_features(c):
    captured=[]
    handle=COARSE.blocks[-1].register_forward_hook(lambda module,args,result:captured.append(result[0].detach().numpy().copy()))
    try:
        torch.set_num_threads(2)
        output=COARSE(torch.from_numpy(c['source'].astype(np.float32))[None],torch.from_numpy(c['mask'][None]))
    finally:
        handle.remove();torch.set_num_threads(1)
    assert np.array_equal(output['prediction'].numpy()[0],c['P0'].astype(np.float32)), 'Frozen coarse P0 parity changed'
    return captured[0][0]


@torch.no_grad()
def margins(ctx,endpoint,d):
    cs=ctx.constraints(endpoint).numpy();bounds=np.zeros_like(cs);bounds[:2]=1e-8;slack=bounds-cs
    n=endpoint.shape[1];out=np.zeros((n,5));out[:,:2]=slack[:2]
    out[:,2:4]=np.inf
    for name,(lo,hi) in ctx.quartets.slices.items():
        values=slack[lo+2:hi+2]
        if name=='signed':positions=np.flatnonzero(ctx.quartets.correct.numpy()[0]);column=2;width=4
        elif name=='unsigned_assessability':positions=np.flatnonzero(ctx.quartets.inverted.numpy()[0]);column=3;width=4
        elif name=='frame_assessability':positions=np.arange(n-3);column=3;width=4
        else:positions=np.arange(n-1);column=3;width=2
        for offset in range(width):np.minimum.at(out[:,column],positions+offset,values)
    # No signed correct quartet affecting a residue is an absent constraint, not an infinite numeric artifact.
    out[np.isinf(out)]=1e30
    out[:,4]=.1-np.linalg.norm(d[0],axis=-1)
    assert ctx.safety(endpoint)['passes']
    return out


@torch.no_grad()
def extract(row):
    path=ROOT/'representations'/f'identity_{row["index"]:04d}.npz';marker=path.with_suffix('.json')
    if marker.exists():
        meta=json.loads(marker.read_text());assert meta['archive_sha256']==old.sha(path)
        assert meta['protocol_sha256']==old.sha(ROOT/'protocol.json');return str(marker)
    start=time.perf_counter();item=old.load_identity(row);arrays={};counts=[];max_error=0.;inverse_count=0
    source=REPO/row['source_path'];assert old.sha(source)==row['source_sha256']
    with np.load(source,allow_pickle=False) as z:sequence=str(z['sequence'].item())
    assert len(sequence)==row['length']
    token=np.array([AA.index(a) if a in AA else 20 for a in sequence],dtype=np.uint8)
    arrays['sequence_tokens']=token
    for ci,c in enumerate(item['conditions']):
        prefix=str(c['condition']);b={k:torch.from_numpy(c[v][None]) for k,v in [('pg','P0'),('source','source'),('target','target'),('mask','mask')]}
        ctx=old.pilot.Context(b);coarse=coarse_features(c)
        source_rep=old.local_representation(torch.from_numpy(c['source'][None]),b['mask'])['features'].numpy()[0].astype(np.float32)
        F=[];H=[];frames=[];elig=[];teacher=[];safety=[];coarse_context=[];indices=[]
        # Deduplicate exact zero tails, while preserving every logical transition and its label.
        unique=min(len(c['states']),16)
        for step in range(unique):
            p,mask,d,_=old.sample(item,ci,step);pt=torch.from_numpy(p[None]);rep,h=latent(pt,b['mask'])
            frame=rep['frame'].numpy()[0];local=np.einsum('nji,nj->ni',frame,d)
            reconstructed=np.einsum('nij,nj->ni',frame,local)
            bad=np.max(np.abs(reconstructed-d),axis=-1)>=1e-12
            if bad.any():
                assert rep['eligible'].numpy()[0][bad].all()
                local[bad]=np.linalg.solve(frame[bad],d[bad,...,None])[...,0];inverse_count+=int(bad.sum())
                reconstructed=np.einsum('nij,nj->ni',frame,local)
            error=float(np.abs(reconstructed-d).max());max_error=max(max_error,error);assert error<1e-12,(row['sample_id'],step,error)
            raw=STUDENT(pt,b['mask']).numpy()[0]
            hidden_u=STUDENT.local.head(h).double();hidden_d=torch.einsum('bnij,bnj->bni',rep['frame'],hidden_u)*rep['eligible'][...,None]
            hidden_d*=torch.clamp(.1/hidden_d.norm(dim=-1,keepdim=True).clamp_min(1e-30),max=1.)
            assert np.array_equal(raw,hidden_d.numpy()[0]), 'Penultimate representation does not reproduce frozen student'
            F.append(rep['features'].numpy()[0].astype(np.float32));H.append(h.numpy()[0]);frames.append(frame)
            elig.append(rep['eligible'].numpy()[0]);teacher.append(local);safety.append(margins(ctx,pt+torch.from_numpy(d[None]),d[None]))
            correction=np.einsum('nji,nj->ni',frame,c['P0']-c['source'])
            progress=np.einsum('nji,nj->ni',frame,p-c['P0'])
            coarse_context.append(np.concatenate([correction,progress,source_rep],axis=-1).astype(np.float32))
        for step in range(16):indices.append(min(step,unique-1))
        # If final actual stored action was nonzero and the final P state is included, its target is the true zero tail.
        arrays.update({f'{prefix}_features':np.stack(F),f'{prefix}_latent':np.stack(H),f'{prefix}_frame':np.stack(frames),
            f'{prefix}_eligible':np.stack(elig),f'{prefix}_teacher_local':np.stack(teacher),f'{prefix}_margins':np.stack(safety),
            f'{prefix}_coarse_scalar':coarse,f'{prefix}_coarse_output_context':np.stack(coarse_context),
            f'{prefix}_logical_state_index':np.asarray(indices,dtype=np.uint8)})
        counts.append(dict(condition=c['condition'],actual_actions=len(c['delta']),unique_states=unique,
            eligible_residue_transitions=sum(int(elig[i].sum()) for i in indices),nonzero_actions=sum(bool(np.any(teacher[i])) for i in indices)))
    old.frozen.old.atomic_npz(path,**arrays)
    write(marker,dict(identity=row['sample_id'],index=row['index'],length=row['length'],length_bin=row['stratum'],
        sequence=sequence,sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),source_archive_sha256=row['source_sha256'],
        archive_sha256=old.sha(path),protocol_sha256=old.sha(ROOT/'protocol.json'),conditions=counts,
        maximum_teacher_frame_roundtrip_error=max_error,numerical_frame_inverse_corrections=inverse_count,
        coarse_P0_bitwise_parity=True,frozen_head_bitwise_parity=True,wall_seconds=time.perf_counter()-start))
    return str(marker)


def run_extract():
    prepare();start=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=PROTOCOL['workers'],mp_context=multiprocessing.get_context('spawn'),initializer=init_worker) as pool:
        for i,p in enumerate(pool.map(extract,rows()),1):
            if i%20==0:print(json.dumps(dict(extracted=i,total=2000,wall_seconds=time.perf_counter()-start,last=p)),flush=True)
    write(ROOT/'EXTRACTION_COMPLETED.json',dict(identities=2000,protocol_sha256=old.sha(ROOT/'protocol.json'),wall_seconds=time.perf_counter()-start))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','extract']);args=parser.parse_args()
    if args.command=='prepare':prepare()
    else:run_extract()
