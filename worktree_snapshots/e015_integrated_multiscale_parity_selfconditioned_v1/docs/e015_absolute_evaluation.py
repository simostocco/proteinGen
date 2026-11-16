"""Read-only E015 absolute audit. CPU now; guarded CUDA generation later.
All outputs/caches remain in the dedicated D: evaluation folder.
No Runtime initialization, training, signaling, or production metadata writes.
"""
import os, sys, json, time, hashlib, math, shutil, argparse, tempfile, fcntl
from pathlib import Path
WORK=Path('/mnt/d/Simone/proteinGen/.e015_worktree')
ROOT=Path('/mnt/d/Simone/proteinGen/e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1')
SHA='2986d44ed22aa1c2fd0dea2fe57fa1b50c0808893fd3608142240841e08cbc03'
OUT=ROOT/'evaluations/absolute_generation_u3811_2986d44e_v1'
MODE=sys.argv[1] if len(sys.argv)>1 else 'cpu'
OUT.mkdir(parents=True,exist_ok=True)
for key in ('TMPDIR','TMP','TEMP','XDG_CACHE_HOME','MPLCONFIGDIR','TORCH_HOME','NUMBA_CACHE_DIR','CUDA_CACHE_PATH','TRITON_CACHE_DIR'):
    p=OUT/'runtime'/key.lower();p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
os.environ['PYTHONDONTWRITEBYTECODE']='1';tempfile.tempdir=os.environ['TMPDIR']
if MODE!='generate':os.environ['CUDA_VISIBLE_DEVICES']=''
os.nice(19)
sys.path.insert(0,str(WORK/'src'))
import numpy as np, torch, yaml, pyarrow as pa, pyarrow.parquet as pq
from protein_distance_diffusion.training.e015_engine import Engine
from protein_distance_diffusion.training.e015_runtime import Runtime
from protein_distance_diffusion.training.e014_runtime import canonical_hash, sha, noise_seed
from protein_distance_diffusion.data.e014_population import CoordinateCache
from protein_distance_diffusion.evaluation.e014_evaluation import coordinate_batch, extended_metrics
from protein_distance_diffusion.evaluation.e015_evaluation import prediction
from protein_distance_diffusion.evaluation.reverse_field_diagnostic import paired_state, vector_metrics
from protein_distance_diffusion.evaluation.marginal_distribution_diagnostic import descriptor, NAMES, ReferenceStandardizer, Comparison, bandwidth, projections, distances
from protein_distance_diffusion.training.coordinate_diffusion import centered_coordinate_noise
from protein_distance_diffusion.models.e013_unconditional_equivariant_denoiser import local_geometry
GRID=[5,50,250,400,490];STRATA=['20-64','65-128','129-256','257-384','385-500'];LENGTHS=[48,96,192,320,448]
CP=['initialization','epoch_1_fraction_0.25','epoch_1_fraction_0.50']
PRIMARY=['v_mse','aligned_rmsd_angstrom','local_mse_angstrom2','torsion_angular_mae_radians','triple_inversion_fraction','clash_fraction']
SCALE=12.22820347644835
CFG_PATH=WORK/'configs/e015_integrated_multiscale_parity_selfconditioned_v1.yaml'
cfg=yaml.safe_load(CFG_PATH.read_text());cal=json.loads((ROOT/'calibration/frozen.json').read_text())
rt=object.__new__(Runtime);rt.config_path=CFG_PATH;rt.cfg=cfg
CONTRACT=rt.contract(cal)
assert CONTRACT==json.loads((ROOT/'preflight/full_certification.json').read_text())['calibration_contract']
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
for name in ('samples','plots','success_gallery','median_gallery','failure_gallery','logs'): (OUT/name).mkdir(exist_ok=True)

def clean(v):
    if isinstance(v,np.generic):v=v.item()
    if isinstance(v,Path):return str(v)
    if isinstance(v,float) and not math.isfinite(v):return None
    if isinstance(v,dict):return {str(k):clean(a) for k,a in v.items()}
    if isinstance(v,(tuple,list,np.ndarray)):return [clean(a) for a in v]
    return v

def write(name,v):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.partial')
    q.write_text(json.dumps(clean(v),separators=(',',':'),allow_nan=False)+'\n');q.replace(p)
def read(name):return json.loads((OUT/name).read_text())
def table(name,rows):
    p=OUT/name;q=p.with_name(p.name+'.partial')
    t=pa.Table.from_pylist([clean(r) for r in rows]) if rows else pa.table({'status':pa.array([],pa.string())})
    pq.write_table(t,q,compression='zstd');assert pq.read_metadata(q).num_rows==len(rows);q.replace(p)
def progress(phase,**kw):
    r=dict(phase=phase,time_unix=time.time(),mode=MODE,**kw);write('logs/progress.json',r);print(json.dumps(r),flush=True)
def model_engine():
    e=Engine(cfg,cal,device='cpu',precision='fp32');assert sum(p.numel() for p in e.model.parameters() if p.requires_grad)==33128139
    return e

def finite(obj):
    if isinstance(obj,torch.Tensor):return not obj.is_floating_point() or bool(torch.isfinite(obj).all())
    if isinstance(obj,dict):return all(finite(v) for v in obj.values())
    if isinstance(obj,(list,tuple)):return all(finite(v) for v in obj)
    return not isinstance(obj,float) or math.isfinite(obj)

def pinned_load(path):
    with Path(path).open('rb') as f:
        b=os.fstat(f.fileno());h=hashlib.sha256()
        while True:
            block=f.read(8*1024**2)
            if not block:break
            h.update(block)
        f.seek(0);s=torch.load(f,map_location='cpu',weights_only=False);a=os.fstat(f.fileno())
    assert (a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_ino,b.st_size,b.st_mtime_ns)
    return s,h.hexdigest()

def verify(s,e):
    assert {k:v for k,v in s['contract'].items() if k!='batching_hash'}==CONTRACT
    assert s['contract']['batching_hash']==canonical_hash(json.loads((ROOT/'batch_autotuning.json').read_text()))
    assert s['population_hash']==cfg['population_hash'] and s['architecture_hash']==canonical_hash(cfg['model'])
    e.model.load_state_dict(s['model'],strict=True);e.ema.load_state_dict(s['ema'],strict=True)
    e.optimizer.load_state_dict(s['optimizer']);e.scaler.load_state_dict(s['scaler'])
    ps=list(e.model.parameters()); ids=[i for g in s['optimizer']['param_groups'] for i in g['params']]
    assert len(ids)==len(ps)
    for pid,p in zip(ids,ps,strict=True):
        for k,v in s['optimizer']['state'].get(pid,{}).items():
            if k in ('exp_avg','exp_avg_sq'):assert tuple(v.shape)==tuple(p.shape)
    assert finite({k:s[k] for k in ('model','ema','optimizer','rng','engine_rng')})
    pos=s['position'];seen=np.asarray(s['exposures']);expected=np.full(len(seen),pos['epoch']-1,dtype=np.uint8)
    expected[np.asarray(s['epoch_schedule'][:pos['cursor']],dtype=np.int64)]+=1
    assert np.array_equal(seen,expected) and int(seen.sum())==pos['exposures']
    assert len(set(s['epoch_schedule']))==230132 and len(seen)==230132
    assert len(s['engine_rng']['actual_t_hist'])==500
    e.optimizer.state.clear()

def inventory():
    e=model_engine();rows=[]
    for label in CP:
        path=ROOT/'production/checkpoints'/f'{label}.pt';s,h=pinned_load(path);verify(s,e)
        if label==CP[-1]:assert h==SHA and s['position']==dict(epoch=1,cursor=115066,update=3811,exposures=115066)
        rows.append(dict(label=label,source_path=str(path),snapshot_path=None,sha256=h,position=s['position'],epoch_fraction=s['position']['cursor']/230132,
            raw_available=True,ema_available=True,optimizer_key_present=True,optimizer_state_count=len(s['optimizer']['state']),
            strict_cpu_raw='PASS',strict_cpu_ema='PASS',optimizer_load_and_shapes='PASS',finite='PASS',exposure_audit='PASS',
            config_hash=s['contract']['config_hash'],train_population_hash=s['population_hash'],contract=s['contract']))
        del s;progress('checkpoint_verified',label=label,update=rows[-1]['position']['update'])
    # Read an immutable inode and atomically publish a bounded contextual snapshot.
    committed=json.loads((ROOT/'COMMITTED.json').read_text())
    if Path(committed['path']).name.startswith('recovery_'):
        target=OUT/'snapshots'/f"secondary_u{committed['position']['update']}_{committed['sha256'][:8]}.pt";target.parent.mkdir(exist_ok=True)
        partial=target.with_name(target.name+'.partial');h=hashlib.sha256()
        with open(committed['path'],'rb') as source,partial.open('wb') as dest:
            b=os.fstat(source.fileno())
            while True:
                block=source.read(8*1024**2)
                if not block:break
                h.update(block);dest.write(block)
            dest.flush();os.fsync(dest.fileno());a=os.fstat(source.fileno())
        assert (a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_ino,b.st_size,b.st_mtime_ns)
        assert h.hexdigest()==committed['sha256'];partial.replace(target)
        s,h=pinned_load(target);verify(s,e);assert s['position']==committed['position']
        rows.append(dict(label='secondary_context_only',source_path=committed['path'],snapshot_path=str(target),sha256=h,position=s['position'],
            config_hash=s['contract']['config_hash'],train_population_hash=s['population_hash'],raw_available=True,ema_available=True,
            strict_cpu_raw='PASS',strict_cpu_ema='PASS',optimizer_load_and_shapes='PASS',finite='PASS',exposure_audit='PASS',evaluated=False))
        del s
    write('checkpoint_inventory.json',dict(primary_label=CP[-1],selection='User-predeclared exact 50%, never performance-selected',checkpoints=rows,committed_at_snapshot=committed))
    return rows

def key(namespace,row):return hashlib.sha256(f"{namespace}|{row['sample_id']}".encode()).hexdigest()
def panels():
    data=Path(cfg['population_root']);metadata=json.loads((data/'population/population.json').read_text())
    for path,digest in metadata['manifest_hashes'].items():assert sha(path)==digest
    train=pq.read_table(data/'population/train.parquet',use_threads=False).to_pylist();dev=pq.read_table(data/'population/development.parquet',use_threads=False).to_pylist()
    assert canonical_hash(sorted(r['sample_id'] for r in train))==cfg['population_hash']
    assert canonical_hash([r['sample_id'] for r in dev])==cfg['development_hash']
    assert not {r['sample_id'] for r in train}&{r['sample_id'] for r in dev}
    assert not {r['cluster_id'] for r in train}&{r['cluster_id'] for r in dev}
    assert not {r['sample_id'].split('_')[0].lower() for r in train}&{r['sample_id'].split('_')[0].lower() for r in dev}
    groups={}
    for r in train:groups.setdefault(r['length'],[]).append(r)
    for values in groups.values():values.sort(key=lambda r:key('e015-absolute-train-v1',r))
    selected=[];used=set()
    for st in STRATA:
        held=sorted(sorted([r for r in dev if r['length_stratum']==st],key=lambda r:key('e015-absolute-heldout-v1',r))[:20],key=lambda r:(r['length'],r['sample_id']))
        for j,r in enumerate(held):
            matched=next(x for x in groups[r['length']] if x['sample_id'] not in used);used.add(matched['sample_id'])
            for split,row in [('heldout',r),('train',matched)]:
                selected.append(dict(row,eval_split=split,pair_id=f'{st}_{j:02}',pair_index=j,execute_cpu_subset=j%2==0,raw_subset=j in (0,10)))
    seeds={f"{r['pair_id']}|{t}":noise_seed('e015-absolute-matched-noise-v1',r['pair_id'],t) for r in selected for t in GRID}
    definition=dict(full_panel_count=dict(train=100,heldout=100),executed_cpu_panel_count=dict(train=50,heldout=50),raw_subset_count=dict(train=10,heldout=10),
        reason='CPU-only while production occupies GPU. Declared bounded 50+50 first stage, not a silent sample reduction.',
        selection='SHA256 selects 20 heldout per stratum; sort by length; exact-length deterministic TRAIN match; execute even pair indices, raw pair indices0/10. No performance-based selection.',
        noise='Identical CPU generators and seeds for each exact-length TRAIN/heldout pair, reused at every checkpoint and weight set.',seeds=seeds,rows=selected,
        train_overlap=0,pdb_entry_overlap=0,cluster_overlap=0,length_pair_max_difference=0,train_population_hash=cfg['population_hash'],development_hash=cfg['development_hash'],
        timesteps=GRID,generation_lengths=LENGTHS,generation_primary_seeds=[15015000+i for i in range(8)],generation_secondary_seeds=[15015000,15015001],
        generation_primary_selection='Lower equal-protein, equal-timestep v-MSE on the fixed raw/EMA subset; ties choose EMA; report all regimes, no checkpoint selection.',
        gallery_score='Freeze before generation: RMS standardized deviation from matched real-reference median across adjacent mean/std, clash fraction, Rg, signed-triple mean, covariance eigenvalues; scale=max(reference IQR/1.349,1e-8); lower better. Best/median/worst per length, no visual selection.',
        classification='Paired reconstruction STRONG: all inherited E014 geometry gates and positive next-step gain CI; MODERATE: global gates and gain pass but bond gate fails; FAILED: nonfinite or direct-X0 collapse fraction>0.05; otherwise WEAK. These labels apply to implied-X0 reconstruction, not noisy-population generation.',
        inherited_thresholds=dict(adjacent_valid_range_angstrom=[3.4,4.2],adjacent_valid_fraction_minimum=.9,clash_fraction_maximum=.01,direct_x0_collapse_fraction_maximum=.05),
        bootstrap_replicates=2000,units=dict(coordinates='Angstrom',v='normalized coordinate-v',next_step_mse='normalized coordinate component squared'))
    write('panel_definitions.json',definition)
    cache=CoordinateCache(data,verify_payload=False)
    # Validate frozen small cache-index files; production has already verified bulk coordinates.
    for path,digest in cache.report['cache_hashes'].items():
        if not path.endswith('coordinates.f32'):assert sha(path)==digest
    write('logs/input_provenance.json',dict(manifest_hashes=metadata['manifest_hashes'],cache_ready_sha256=sha(cache.root/'READY.json'),cache_payload='Read-only verified-production cache; selected finite coordinates checked; full multi-GB bulk checksum not redundantly scanned',population_hash=cfg['population_hash']))
    return selected,train,dev,cache

def intrinsic(x):
    x=np.asarray(x,dtype=np.float64);n=len(x);z=x-x.mean(0);eig=np.linalg.eigvalsh(z.T@z/n)[::-1];trace=eig.sum()
    ds=np.linalg.norm(z[:,None]-z[None,:],axis=-1);sep=np.abs(np.arange(n)[:,None]-np.arange(n)[None,:]);non=sep>3;lr=sep>=12
    adj=np.linalg.norm(np.diff(z,axis=0),axis=-1);b=np.diff(z,axis=0);a=-b[:-1];c=b[1:];den=np.linalg.norm(a,axis=-1)*np.linalg.norm(c,axis=-1)
    valid=den>1e-12;angles=np.arccos(np.clip((a*c).sum(-1)[valid]/den[valid],-1,1))
    features,_,_=local_geometry(torch.tensor(z,dtype=torch.float32));f=features.numpy();assess=f[:,29]>0;phi=np.arctan2(f[assess,25],f[assess,26]);q=f[assess,27]
    contact=(ds<8)&lr;ci,cj=np.where(np.triu(contact,1))
    result=dict(length=n,finite_fraction=float(np.isfinite(x).mean()),centroid_norm_angstrom=float(np.linalg.norm(x.mean(0))),
        radius_of_gyration_angstrom=float(np.sqrt(max(trace,0))),end_to_end_angstrom=float(np.linalg.norm(z[-1]-z[0])),
        covariance_eigenvalues_angstrom2=eig.tolist(),covariance_trace_angstrom2=float(trace),anisotropy_largest_fraction=float(eig[0]/max(trace,1e-12)),
        adjacent_mean_angstrom=float(adj.mean()),adjacent_std_angstrom=float(adj.std()),adjacent_min_angstrom=float(adj.min()),adjacent_max_angstrom=float(adj.max()),
        adjacent_valid_fraction=float(((adj>=3.4)&(adj<=4.2)).mean()),clash_count=int(((ds<3)&non).sum()),clash_pair_count=int(non.sum()),
        clash_fraction=float(((ds<3)&non).sum()/max(non.sum(),1)),long_range_contact_density=float(contact.sum()/max(lr.sum(),1)),
        long_range_contact_order=float((cj-ci).mean()/n) if len(ci) else None,
        exact_duplicate_coordinate_pairs=int(((ds==0)&(sep>0)).sum()//2),adjacent_out_of_frozen_range_count=int(((adj<3.4)|(adj>4.2)).sum()),
        torsion_assessable_fraction=float(assess.mean()),torsion_assessable_count=int(assess.sum()),signed_triple_mean=float(q.mean()) if len(q) else None,
        negative_signed_triple_fraction=float((q<0).mean()) if len(q) else None,
        bond_angle_mean_radians=float(angles.mean()) if len(angles) else None,bond_angle_std_radians=float(angles.std()) if len(angles) else None,
        pseudo_torsion_quantiles_radians=np.quantile(phi,[.05,.25,.5,.75,.95]).tolist() if len(phi) else None,
        signed_triple_quantiles=np.quantile(q,[.05,.25,.5,.75,.95]).tolist() if len(q) else None,
        pair_distance_quantiles_angstrom=np.quantile(ds[np.triu_indices(n,1)],[.05,.25,.5,.75,.95]).tolist())
    for k in (1,2,3):
        v=np.linalg.norm(z[k:]-z[:-k],axis=-1);result[f'offset_{k}_distance_quantiles_angstrom']=np.quantile(v,[.05,.25,.5,.75,.95]).tolist()
    result['descriptor']=descriptor(z).tolist();return result

@torch.no_grad()
def measure(e,model,row,cache):
    x,n,m=coordinate_batch([row],cache,SCALE,device='cpu');assert torch.isfinite(x).all()
    for t0 in GRID:
        t=torch.tensor([t0]);seed=noise_seed('e015-absolute-matched-noise-v1',row['pair_id'],t0)
        g=torch.Generator().manual_seed(seed);batch=e.diffusion.make_training_batch(x,m,timesteps=t,generator=g)
        v,comp=prediction(model,e.diffusion,batch.noisy_coordinates,t,n,m)
        nxt,xhat,_=e.diffusion.deterministic_reverse_step(batch.noisy_coordinates,t,v,m)
        oracle=paired_state(e.diffusion,x,batch.coordinate_noise,t-1,m)
        out=extended_metrics(xhat[0],x[0],v[0],batch.coordinate_v_target[0],SCALE,comp[0]);out.update(vector_metrics(v,batch.coordinate_v_target,m,'v'))
        assert out['finite'] and torch.isfinite(nxt).all()
        noop=float((batch.noisy_coordinates-oracle).square()[m].mean());step=float((nxt-oracle).square()[m].mean())
        a=xhat[0]*SCALE;target=x[0]*SCALE;ia=intrinsic(a.numpy())
        # Store compact scalar structure diagnostics, not 34-channel descriptors on every reconstruction.
        out.update({k:v0 for k,v0 in ia.items() if k not in ('descriptor','length','torsion_assessable_count')})
        out.update(centered_component_mse_angstrom2=float((a-target).square().mean()),centered_coordinate_rmse_angstrom=float((a-target).square().mean().sqrt()),
            centered_euclidean_rmsd_angstrom=float((a-target).square().sum(-1).mean().sqrt()),
            v_predicted_rms=out['v_predicted_norm']/math.sqrt(3*row['length']),v_target_rms=out['v_target_norm']/math.sqrt(3*row['length']),
            next_step_coordinate_mse_normalized=step,next_step_coordinate_mse_angstrom2=step*SCALE**2,noop_next_step_mse_normalized=noop,
            step_gain=1-step/noop if noop>0 else None,covariance_trace_ratio=out['radius_ratio']**2,
            torsion_angular_mae_degrees=out['torsion_angular_mae_radians']*180/math.pi if out['torsion_angular_mae_radians'] is not None else None,
            direct_x0_collapse_count=int(out['collapsed']),noisy_next_rg_ratio_to_oracle=float(nxt.square().sum(-1).mean().sqrt()/oracle.square().sum(-1).mean().sqrt()),
            sample_id=row['sample_id'],split=row['eval_split'],pair_id=row['pair_id'],length=row['length'],length_stratum=row['length_stratum'],timestep=t0,noise_seed=seed)
        for k in (1,2,3):
            dp=(a[k:]-a[:-k]).norm(dim=-1);dt=(target[k:]-target[:-k]).norm(dim=-1)
            out[f'offset_{k}_rmse_angstrom']=out[f'offset_{k}_mse_angstrom2']**.5
            out[f'offset_{k}_absolute_error_quantiles_angstrom']=torch.quantile((dp-dt).abs(),torch.tensor([.05,.25,.5,.75,.95])).tolist()
        # Explicit complementary pseudo-torsion components from existing local_geometry.
        fp,_,_=local_geometry(xhat[0]);ft,_,_=local_geometry(x[0]);ass=(fp[:,29]>0)&(ft[:,29]>0)
        out['torsion_sin_mse']=float((fp[:,25]-ft[:,25]).square()[ass].mean()) if ass.any() else None
        out['torsion_cos_mse']=float((fp[:,26]-ft[:,26]).square()[ass].mean()) if ass.any() else None
        yield out

def cpu():
    if not (OUT/'checkpoint_inventory.json').exists():inventory()
    panel,train,dev,cache=panels();e=model_engine()
    path=OUT/'absolute_reconstruction_metrics.parquet';rows=pq.read_table(path,use_threads=False).to_pylist() if path.exists() else []
    done={(r['checkpoint'],r['weights'],r['sample_id'],r['timestep']) for r in rows}
    for cp in CP:
        record=next(r for r in read('checkpoint_inventory.json')['checkpoints'] if r['label']==cp)
        saved,h=pinned_load(record['source_path']);assert h==record['sha256'];verify(saved,e)
        for weights,model in [('ema',e.ema),('raw',e.model)]:
            model.load_state_dict(saved[weights if weights=='ema' else 'model'],strict=True);model.eval()
            subset=[r for r in panel if r['execute_cpu_subset'] and (weights=='ema' or r['raw_subset'])]
            for i,row in enumerate(subset):
                if all((cp,weights,row['sample_id'],t) in done for t in GRID):continue
                start=time.monotonic()
                for r in measure(e,model,row,cache):
                    ident=(cp,weights,row['sample_id'],r['timestep'])
                    if ident not in done:rows.append(dict(r,checkpoint=cp,weights=weights));done.add(ident)
                table('absolute_reconstruction_metrics.parquet',rows)
                progress('cpu_reconstruction',checkpoint=cp,weights=weights,identities=i+1,total=len(subset),conditions=len(rows),protein_seconds=time.monotonic()-start)
        del saved
    assert len(rows)==1800
    real=[]
    for row in panel:
        x=cache.get(row['sample_id']);x=x-x.mean(0);real.append(dict(intrinsic(x),sample_id=row['sample_id'],split=row['eval_split'],length_stratum=row['length_stratum']))
    table('clean_reference_metrics.parquet',real)
    write('logs/cpu_complete.json',dict(conditions=len(rows),cuda_initialized=torch.cuda.is_initialized(),production_contract_unchanged=rt.contract(cal)==CONTRACT))
    assert not torch.cuda.is_initialized();report()

# Equal-protein paired/stratified bootstrap, never resample residues or timesteps.
def estimate(a,b,strata,paired=True):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);strata=np.asarray(strata);valid=np.isfinite(a)&np.isfinite(b)
    a=a[valid];b=b[valid];strata=strata[valid]
    if not len(a):return dict(n=0)
    rng=np.random.default_rng(15015);ad=np.zeros(2000);bd=np.zeros(2000)
    for st in sorted(set(strata)):
        ix=np.flatnonzero(strata==st);draw=rng.choice(ix,(2000,len(ix)),replace=True)
        ad+=a[draw].sum(1)/len(a)
        drawb=draw if paired else rng.choice(ix,(2000,len(ix)),replace=True)
        bd+=b[drawb].sum(1)/len(b)
    ratio=bd/ad
    return dict(n=len(a),old_mean=float(a.mean()),new_mean=float(b.mean()),absolute_difference=float((b-a).mean()),
        relative_difference=float((b.mean()-a.mean())/a.mean()) if abs(a.mean())>1e-12 else None,
        ratio=float(b.mean()/a.mean()) if abs(a.mean())>1e-12 else None,old_ci95=np.quantile(ad,[.025,.975]).tolist(),new_ci95=np.quantile(bd,[.025,.975]).tolist(),
        difference_ci95=np.quantile(bd-ad,[.025,.975]).tolist(),ratio_ci95=np.quantile(ratio,[.025,.975]).tolist() if np.isfinite(ratio).all() else None,
        bootstrap_replicates=2000,paired=paired)

def groupmean(rows):
    result={}
    keys=sorted({k for r in rows for k,v in r.items() if isinstance(v,(int,float,bool)) and k not in ('length','noise_seed','timestep')})
    for k in keys:
        v=[float(r[k]) for r in rows if r.get(k) is not None and isinstance(r[k],(int,float,bool)) and math.isfinite(float(r[k]))]
        if v:result[k]=dict(mean=float(np.mean(v)),median=float(np.median(v)),n=len(v),std=float(np.std(v)))
    return result

def report():
    rows=pq.read_table(OUT/'absolute_reconstruction_metrics.parquet',use_threads=False).to_pylist();means=[];gaps=[];trajectory=[];rawema=[]
    for cp in CP:
        for t in [None,*GRID]:
            for st in ['all',*STRATA]:
                use=[r for r in rows if r['checkpoint']==cp and r['weights']=='ema' and (t is None or r['timestep']==t) and (st=='all' or r['length_stratum']==st)]
                if not use:continue
                for split in ('train','heldout'):
                    group=[r for r in use if r['split']==split];summary=groupmean(group)
                    for k,v in summary.items():means.append(dict(checkpoint=cp,timestep=t,length_stratum=st,split=split,metric=k,**v))
                for k in PRIMARY:
                    # Average timesteps inside each protein for aggregate, retaining equal protein weighting.
                    pairs={}
                    for r in use:pairs.setdefault(r['pair_id'],{}).setdefault(r['split'],[]).append(r)
                    valid=[p for p in pairs.values() if all(p.get(s) for s in ('train','heldout'))]
                    a=[np.mean([r[k] for r in p['train'] if r.get(k) is not None]) for p in valid];b=[np.mean([r[k] for r in p['heldout'] if r.get(k) is not None]) for p in valid]
                    strata=[p['train'][0]['length_stratum'] for p in valid]
                    est=estimate(a,b,strata,paired=True);gaps.append(dict(checkpoint=cp,timestep=t,length_stratum=st,metric=k,train_mean=est['old_mean'],heldout_mean=est['new_mean'],gap=est['absolute_difference'],**est))
    table('train_vs_heldout_metrics.parquet',means);write('generalization_bootstrap.json',dict(rows=gaps,interpretation='Paired exact-length pairs and shared noise; identity panels independent in split membership. Timesteps averaged within proteins for aggregate. Small gaps do not imply good absolute quality.',replicates=2000))
    numeric=[k for k,v in rows[0].items() if isinstance(v,(float,int,bool)) and k not in ('length','noise_seed','timestep')]
    index={(r['checkpoint'],r['weights'],r['sample_id'],r['timestep']):r for r in rows}
    for old,new in [(CP[0],CP[1]),(CP[1],CP[2]),(CP[0],CP[2])]:
        for split in ('train','heldout'):
            for t in GRID:
                for st in ['all',*STRATA]:
                    selected=[r for r in rows if r['checkpoint']==old and r['weights']=='ema' and r['split']==split and r['timestep']==t and (st=='all' or r['length_stratum']==st)]
                    for k in numeric:
                        valid=[(r,index[(new,'ema',r['sample_id'],t)]) for r in selected if r.get(k) is not None and index[(new,'ema',r['sample_id'],t)].get(k) is not None]
                        if not valid:continue
                        est=estimate([a[k] for a,b in valid],[b[k] for a,b in valid],[a['length_stratum'] for a,b in valid])
                        trajectory.append(dict(old_checkpoint=old,new_checkpoint=new,split=split,timestep=t,length_stratum=st,metric=k,**est))
    table('checkpoint_trajectory_metrics.parquet',trajectory)
    for cp in CP:
        for t in [None,*GRID]:
            for split in ('train','heldout'):
                raw=[r for r in rows if r['checkpoint']==cp and r['weights']=='raw' and r['split']==split and (t is None or r['timestep']==t)]
                for k in PRIMARY:
                    grouped={}
                    for r in raw:grouped.setdefault(r['sample_id'],[]).append(r)
                    groups=list(grouped.values());a=[np.mean([index[(cp,'ema',r['sample_id'],r['timestep'])][k] for r in group]) for group in groups];b=[np.mean([r[k] for r in group]) for group in groups]
                    est=estimate(a,b,[group[0]['length_stratum'] for group in groups]);rawema.append(dict(checkpoint=cp,timestep=t,split=split,metric=k,ema_mean=est['old_mean'],raw_mean=est['new_mean'],raw_minus_ema=est['absolute_difference'],**est))
    table('raw_vs_ema_metrics.parquet',rawema)
    primary=CP[-1]
    rv=[r for r in rawema if r['checkpoint']==primary and r['timestep'] is None and r['metric']=='v_mse'];raw=float(np.mean([r['raw_mean'] for r in rv]));ema=float(np.mean([r['ema_mean'] for r in rv]));choice='ema' if ema<=raw else 'raw'
    write('sampler_weight_selection.json',dict(primary_weights=choice,raw_v_mse=raw,ema_v_mse=ema,selection='Predeclared equal-protein, equal-timestep raw/EMA matched subset objective; no checkpoint performance selection',rawema_panel=20,ema_initial_coefficient=.9999**3811))
    primarygaps=[r for r in gaps if r['checkpoint']==primary and r['timestep'] is None and r['length_stratum']=='all']
    # Advisory labels based on evidence, not protocol amendments or training actions.
    vgap=next(r for r in primarygaps if r['metric']=='v_mse')
    half=[r for r in rows if r['checkpoint']==primary and r['weights']=='ema' and r['split']=='heldout']
    init=[r for r in rows if r['checkpoint']==CP[0] and r['weights']=='ema' and r['split']=='heldout']
    late=[r for r in trajectory if r['old_checkpoint']==CP[1] and r['new_checkpoint']==primary and r['split']=='heldout' and r['length_stratum']=='all']
    vimprove=np.mean([r['v_mse'] for r in half])<np.mean([r['v_mse'] for r in init])
    # Poor mid/high-noise geometry prevents a strong-generator claim even with a tiny generalization gap.
    general='UNDERFIT' if vgap['difference_ci95'][0]<=0<=vgap['difference_ci95'][1] and np.mean([r['adjacent_valid_fraction'] for r in half])<.9 else 'INCONCLUSIVE'
    verdict=dict(denoising='PROMISING_BUT_IMMATURE' if vimprove else 'WEAK',generation='INCONCLUSIVE',generalization=general,continued_training='CONTINUE_BUT_WATCH',
        primary_checkpoint=primary,primary_update=3811,primary_exposures=115066,primary_sha256=SHA,
        full_de_novo_samples=0,generation_reason='QUEUED: production trainer active, no verified supervisor evaluation-pause mechanism. No CUDA generation/evaluation run concurrently.',
        sampler_weights=choice,raw_ema_comparison_v_mse=dict(raw=raw,ema=ema),production_modified=False,cuda_used=False,all_outputs_on_d=True)
    write('verdict.json',verdict)
    # Per-regime classifications use inherited gates and bootstrap positive gain.
    regimes=[]
    for t in GRID:
        for st in STRATA:
            group=[r for r in half if r['timestep']==t and r['length_stratum']==st]
            avg=groupmean(group);collapse=avg['collapsed']['mean'];bond=avg['adjacent_valid_fraction']['mean'];clash=avg['clash_fraction']['mean']
            gain=estimate([0]*len(group),[r['step_gain'] for r in group],[st]*len(group))['difference_ci95']
            label='FAILED' if collapse>.05 or not all(r['finite'] for r in group) else 'STRONG' if bond>=.9 and clash<=.01 and gain[0]>0 else 'MODERATE' if clash<=.01 and gain[0]>0 else 'WEAK'
            regimes.append(dict(timestep=t,length_stratum=st,label=label,adjacent_valid_fraction=bond,clash_fraction=clash,direct_x0_collapse_fraction=collapse,step_gain_ci95=gain))
    write('regime_assessment.json',dict(scope='Absolute target-conditioned reconstruction only; inherited E014 thresholds; labels are not generation verdicts',rows=regimes))
    generation_pending()
    plots(rows,rawema,gaps)
    report_markdown(rows,gaps,rawema,trajectory,verdict,regimes)
    manifest('CPU_COMPLETE_GPU_QUEUED')
    progress('CPU_COMPLETE_GPU_QUEUED',conditions=len(rows),sampler_weights=choice,verdict=verdict)

def generation_pending():
    for name in ('generation_manifest.parquet','generation_metrics.parquet','trajectory_metrics.parquet'):
        if not (OUT/name).exists():table(name,[])
    for name in ('generation_distribution_metrics.json','diversity_novelty_metrics.json'):
        if not (OUT/name).exists():write(name,dict(status='QUEUED_NOT_MEASURED',reason='Active production; no supported graceful supervisor pause'))
    write('GPU_QUEUE.json',dict(status='QUEUED',command=f"/home/simostocco/miniforge3/envs/proteingen/bin/python -B {OUT}/e015_absolute_evaluation.py generate",primary_sha256=SHA,
        primary_samples=40,secondary_samples=10,lengths=LENGTHS,steps=500,frames=[499,450,400,350,300,250,200,150,100,50,0],safety='Guard refuses any active E015 supervisor/trainer or E007 trainer, requires <1.5GiB external GPU use, and nonblocking existing GPU_WORKER.lock. No signaling/pause/resume.',
        authorization='Full t0 diagnostic authorized by user only after production independently stops. Production config remains unchanged.'))

def manifest(status):
    files=[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size) for p in OUT.rglob('*') if p.is_file() and not p.name.endswith('.partial')]
    total=sum(p['bytes'] for p in files);assert total<5*1024**3
    write('MANIFEST.json',dict(status=status,primary_sha256=SHA,config_hash=CONTRACT['config_hash'],population_hash=cfg['population_hash'],
        source_hashes=CONTRACT['source_hashes'],files=files,total_bytes=total,disk_target_bytes=5*1024**3,all_paths_on_d=True,
        production_metadata_written=False,protocol_modified=False,scientific_source_modified=False,prospective_opened=False))

def plots(rows,rawema,gaps):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    for metric,ylabel in [('v_mse','Normalized v-MSE'),('aligned_rmsd_angstrom','Kabsch RMSD (Å)'),('local_mse_angstrom2','Local distance RMSE (Å)'),('torsion_angular_mae_degrees','Torsion error (degrees)'),('triple_inversion_fraction','Inversion fraction'),('clash_fraction','Clash fraction')]:
        fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        for ax,split in zip(axes,('train','heldout')):
            for cp in CP:
                vals=[np.mean([r[metric] for r in rows if r['weights']=='ema' and r['checkpoint']==cp and r['split']==split and r['timestep']==t]) for t in GRID]
                if metric=='local_mse_angstrom2':vals=np.sqrt(vals)
                ax.plot(GRID,vals,marker='o',label=cp)
            ax.set(title=split.upper(),xlabel='Diffusion timestep',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.savefig(OUT/'plots'/f'absolute_{metric}.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for split in ('train','heldout'):
        subset=[r for r in rawema if r['checkpoint']==CP[-1] and r['split']==split and r['metric']=='v_mse' and r['timestep'] is not None]
        for kind in ('raw','ema'):axes[0].plot(GRID,[next(r[f'{kind}_mean'] for r in subset if r['timestep']==t) for t in GRID],marker='o',label=f'{split} {kind}')
    g=[r for r in gaps if r['checkpoint']==CP[-1] and r['metric']=='v_mse' and r['length_stratum']=='all' and r['timestep'] is not None]
    y=[r['gap'] for r in g];err=np.array([[r['gap']-r['difference_ci95'][0],r['difference_ci95'][1]-r['gap']] for r in g]).T
    axes[1].errorbar(GRID,y,yerr=err,marker='o');axes[1].axhline(0,color='gray',linewidth=.8)
    axes[0].set(title='Matched raw versus EMA subset',xlabel='t',ylabel='v-MSE');axes[0].legend(fontsize=7)
    axes[1].set(title='Heldout − TRAIN paired gap (CI95)',xlabel='t',ylabel='v-MSE gap')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(OUT/'plots/generalization_raw_ema.png',dpi=160);plt.close(fig)

def report_markdown(rows,gaps,rawema,trajectory,verdict,regimes):
    lines=['# E015 absolute reconstruction and queued de novo evaluation','',f"**Denoising: {verdict['denoising']} · Generation: INCONCLUSIVE · Generalization: {verdict['generalization']} · Continued training: CONTINUE_BUT_WATCH**",'',
        f"Primary: exact epoch1 **50%**, update3811 / exposures115066, SHA256 `{SHA}`. Checkpoint selection was fixed by the user before any metrics. Raw/EMA/optimizer/config/RNG/exposure state strictly verified on CPU; no production state restored or mutated.",'',
        '## Scope and safety','',
        'All three checkpoints evaluated on 50 TRAIN + 50 frozen-development identities, 10 exact-length pairs per stratum, five fixed timesteps; EMA main panel. Full100+100 panels are frozen in panel_definitions.json, with the reduction explicitly declared before evaluation. Raw/EMA comparison uses20identities (10+10), two exact-length pairs per stratum, all checkpoints/timesteps. Same CPU Gaussian draws in each length-matched pair and across checkpoints. No heldout identity/PDB entry/cluster occurs in TRAIN. No performance-based sample selection, no prospective data.', '',
        'Production is active and the inspected supervisor has no verified evaluation-pause request mechanism. Signal-based stop handling is not used. No second CUDA model was created; full de novo generation is queued. Empty generation Parquets and queued JSON files represent **zero measurements**, not zero error. No claim about full t0 sample quality/diversity/novelty can be made yet.', '',
        'Primary checkpoint is immutable and recomputed SHA256 matches the expected digest. COMMITTED-referenced rolling recovery was atomically copied/hashed/strictly verified solely as a contextual snapshot; it does not replace the exact50% primary and is not performance-selected.', '',
        '## Absolute EMA errors and matched generalization gaps','',
        '| t | Metric | TRAIN mean | Heldout mean | Heldout−TRAIN | Gap CI95 | Heldout/TRAIN |', '|---:|---|---:|---:|---:|---|---:|']
    for t in GRID:
        for k in PRIMARY:
            r=next(r for r in gaps if r['checkpoint']==CP[-1] and r['timestep']==t and r['length_stratum']=='all' and r['metric']==k)
            lines.append(f"| {t} | {k} | {r['train_mean']:.6g} | {r['heldout_mean']:.6g} | {r['gap']:+.6g} | [{r['difference_ci95'][0]:.5g}, {r['difference_ci95'][1]:.5g}] | {r['ratio']:.5g} |")
    lines+=['','v-MSE is normalized; Kabsch RMSD is Å; local distance MSE is Å²; torsion MAE is radians; inversion/clash are fractions. Equal protein weighting;2,000 paired length-stratified draws; aggregate averages timesteps within proteins. Gaps and their uncertainty by every timestep/stratum/checkpoint are in generalization_bootstrap.json. Small gaps do not imply protein-like absolute geometry.', '',
        '## Physical reconstruction quality (heldout, primary EMA)','', '| t | Component RMSE Å | Kabsch RMSD Å | Adjacent mean±std Å | Adjacent valid fraction | Local RMSE Å | Torsion MAE deg | Inversion fraction | Clash fraction | X0 Rg ratio | X0 collapse fraction | Noisy next Rg/oracle |', '|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for t in GRID:
        group=[r for r in rows if r['checkpoint']==CP[-1] and r['weights']=='ema' and r['split']=='heldout' and r['timestep']==t];g=groupmean(group)
        val=lambda k:g[k]['mean']
        lines.append(f"| {t} | {val('centered_coordinate_rmse_angstrom'):.4g} | {val('aligned_rmsd_angstrom'):.4g} | {val('adjacent_mean_angstrom'):.4g} ± {val('adjacent_std_angstrom'):.4g} | {val('adjacent_valid_fraction'):.3f} | {math.sqrt(val('local_mse_angstrom2')):.4g} | {val('torsion_angular_mae_degrees'):.4g} | {val('triple_inversion_fraction'):.4f} | {val('clash_fraction'):.4f} | {val('radius_ratio'):.4f} | {val('collapsed'):.3f} | {val('noisy_next_rg_ratio_to_oracle'):.6f} |")
    lines+=['','Component RMSE is sqrt(mean per xyz component error); centered Euclidean RMSD uses sum xyz and differs by sqrt3. Local RMSE above is sqrt(equal-protein mean local-distance MSE). Covariance-trace ratio is squared Rg ratio per protein, then averaged. Frozen clash definition counts directed pairs with chain separation>3 and distance<3Å. Direct-X0 collapse means Rg<0.25clean-referenceRg. This is explicitly **not collapse of the noisy diffusion population**. Physical bonds use inherited3.4–4.2Å validity range.', '',
        'Bond angle target-error is not part of the frozen existing evaluator; bond-angle distributions are included descriptively. Pseudo-torsion sin/cos MSE, radians/degrees, normalized signed-triple MSE, assessability denominators, inversion counts, L1/L2/L3 RMSE/MSE/absolute-error quantiles, covariance eigenvalues, end-to-end distance, contacts/order and next-step gain appear in absolute_reconstruction_metrics.parquet.', '',
        '## Learning trajectory (heldout EMA)','', '| t | Metric | Initial | 25% | 50% | 25%→50% difference | Difference CI95 |', '|---:|---|---:|---:|---:|---:|---|']
    for t in GRID:
        for k in PRIMARY:
            early=next(r for r in trajectory if r['old_checkpoint']==CP[0] and r['new_checkpoint']==CP[1] and r['split']=='heldout' and r['timestep']==t and r['length_stratum']=='all' and r['metric']==k)
            late=next(r for r in trajectory if r['old_checkpoint']==CP[1] and r['new_checkpoint']==CP[2] and r['split']=='heldout' and r['timestep']==t and r['length_stratum']=='all' and r['metric']==k)
            lines.append(f"| {t} | {k} | {early['old_mean']:.6g} | {late['old_mean']:.6g} | {late['new_mean']:.6g} | {late['absolute_difference']:+.5g} | [{late['difference_ci95'][0]:.5g}, {late['difference_ci95'][1]:.5g}] |")
    lines+=['','Every recorded numeric metric has paired checkpoint absolute/relative differences and intervals, separately by split/timestep/length, in checkpoint_trajectory_metrics.parquet. Absolute plots are in plots/. No aggregate masks a high-noise regression.', '',
        '## Raw versus EMA','',f"Predeclared fixed-subset v-MSE selects **{verdict['sampler_weights']}** as the future primary sampler. Aggregate raw v-MSE={verdict['raw_ema_comparison_v_mse']['raw']:.6g}; EMA={verdict['raw_ema_comparison_v_mse']['ema']:.6g}. This selects weights only, not a checkpoint. EMA initialization coefficient is {.9999**3811:.2%}; raw versus EMA per-timestep geometry can disagree.",'', '| Split | t | Metric | EMA | Raw | Raw−EMA | CI95 |', '|---|---:|---|---:|---:|---:|---|']
    for r in rawema:
        if r['checkpoint']==CP[-1] and r['metric'] in ('v_mse','aligned_rmsd_angstrom','local_mse_angstrom2') and r['timestep'] is not None:
            lines.append(f"| {r['split']} | {r['timestep']} | {r['metric']} | {r['ema_mean']:.6g} | {r['raw_mean']:.6g} | {r['raw_minus_ema']:+.5g} | [{r['difference_ci95'][0]:.5g}, {r['difference_ci95'][1]:.5g}] |")
    lines+=['','## Strength and weakness map','', '| Length stratum | '+ ' | '.join(str(t) for t in GRID)+' |', '|---|'+ '|'.join(['---']*5)+'|']
    for st in STRATA:lines.append('| '+st+' | '+' | '.join(next(r['label'] for r in regimes if r['length_stratum']==st and r['timestep']==t) for t in GRID)+' |')
    lines+=['','Regime labels apply to absolute paired clean reconstruction using predeclared inherited geometry gates and positive-gain intervals, not to unconditional generation. Strongest/weakest lengths are assessed within noise regimes because mixing lengths/noise is confounded; full numerical strata values accompany this map. De novo memorization/mode-collapse/packing claims remain inconclusive until queued samples exist.', '',
        '## Queued unconditional experiment','',
        '40primary de novo samples (8seeds × N48/96/192/320/448) plus10secondary-weight samples;500exact deterministic production reverse steps499→0, generated self-conditioning only. No target/sequence/correction. Sparse frames499/450/400/350/300/250/200/150/100/50/0. Final CA-only NPZ/PDB, per-sample intrinsic metrics, two-view traces, distance maps/profiles, trajectory plots/montages, descriptor real-real floors, diversity and sampled novelty, numerically scored best/median/worst galleries are prepared in the evaluator.', '',
        '**Current generated samples:0. Finite fraction, intrinsic quality, descriptor distance, diversity, nearest TRAIN similarity, memorization and generation-collapse evidence are unmeasured. Galleries are currently empty by design; do not interpret denoising reconstruction images as generated proteins.**', '',
        'Run only after production independently stops and GPU safety guards pass:', '', '```bash',f'/home/simostocco/miniforge3/envs/proteingen/bin/python -B {OUT}/e015_absolute_evaluation.py generate','```','',
        'No pause occurred; no resume action is required or authorized by this evaluation. No weights/optimizer/scheduler/sampler/architecture/schedules/protocol or production metadata changed. All diagnostic files/cache/temp directories are onD. The production full_t0_sampling_authorized:false field remains unchanged: this separate owner-authorized diagnostic is not a protocol amendment. Overall answer: E015 shows denoising learning, but useful unconditional C-alpha generation is **not yet established**; poor high-noise clean reconstruction alone cannot prove poor de novo samples.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')

# GPU phase implemented below; always guard against host process namespace isolation.
def gpu_guard(memory_check=True):
    proc=Path('/proc')
    if not (proc/'28874/cmdline').exists():
        # Require a host-visible PID namespace, not a sandbox with only its own process.
        if len([p for p in proc.iterdir() if p.name.isdigit()])<20:raise RuntimeError('Host process namespace unavailable; run approved command outside sandbox to prove GPU isolation')
    active=[]
    for p in proc.iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:parts=(p/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if parts and Path(parts[0].decode(errors='ignore')).name.startswith('python') and any(any(s in a for s in (b'run_e015_integrated.py',b'supervise_e015_training.py',b'run_e007')) for a in parts[1:]):active.append(int(p.name))
    if active:raise RuntimeError(f'Production/trainer processes active: {active}; GPU evaluation refused; no signals sent')
    import subprocess
    memory=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    if memory_check and int(memory.stdout.strip())>=1536:raise RuntimeError('External GPU usage >=1.5GiB; evaluation refused')
    return dict(active_trainers=active,gpu_external_mib=int(memory.stdout.strip()),time_unix=time.time())

def generate():
    safety=gpu_guard()
    if not (OUT/'logs/cpu_complete.json').exists():raise RuntimeError('Complete frozen CPU comparison first')
    # Reuse the existing lease inode without writing its contents or Runtime metadata.
    lock=(ROOT/'GPU_WORKER.lock').open('rb')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('E015 GPU lease occupied; evaluation refused') from None
    write('logs/gpu_safety_at_start.json',safety)
    e=model_engine();s,h=pinned_load(ROOT/'production/checkpoints/epoch_1_fraction_0.50.pt');assert h==SHA;verify(s,e)
    chosen=read('sampler_weight_selection.json')['primary_weights'];panel,train,dev,cache=panels()
    selection=read('panel_definitions.json');generation_rows=[];traj_rows=[];samples=[]
    existing=OUT/'generation_manifest.parquet'
    if existing.exists():
        cached=pq.read_table(existing,use_threads=False).to_pylist()
        if cached and 'sample_key' in cached[0]:samples=cached
    if (OUT/'generation_metrics.parquet').exists():
        cached=pq.read_table(OUT/'generation_metrics.parquet',use_threads=False).to_pylist()
        if cached and 'sample_key' in cached[0]:generation_rows=cached
    if (OUT/'trajectory_metrics.parquet').exists():
        cached=pq.read_table(OUT/'trajectory_metrics.parquet',use_threads=False).to_pylist()
        if cached and 'sample_key' in cached[0]:traj_rows=cached
    done={r['sample_key'] for r in samples};frames={499,450,400,350,300,250,200,150,100,50,0}
    schedule=list(range(499,-1,-1));assert len(schedule)==cfg['diffusion_steps']==500
    # Production evaluation prefixes use FP32 inference outside autocast. Extend
    # their exact deterministic reverse update/context recurrence through t0.
    try:
        for weights,count in [(chosen,8),('raw' if chosen=='ema' else 'ema',2)]:
            model=e.ema if weights=='ema' else e.model
            model.load_state_dict(s['ema' if weights=='ema' else 'model'],strict=True);model.to('cuda').eval()
            for n in LENGTHS:
                for i in range(count):
                    seed=selection['generation_primary_seeds'][i];name=f'{weights}_n{n}_s{seed}'
                    if name in done:continue
                    # Never use target coordinates, sequences, or a correction.
                    gpu_guard(memory_check=False);mask=torch.ones(1,n,dtype=torch.bool,device='cuda');length=torch.tensor([n],device='cuda')
                    g=torch.Generator(device='cuda').manual_seed(seed)
                    current=centered_coordinate_noise(torch.empty(1,n,3,device='cuda'),mask,generator=g);context=None
                    trajectory={};local=[];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.monotonic()
                    with torch.no_grad():
                        for t0 in schedule:
                            t=torch.tensor([t0],device='cuda');available=torch.tensor([context is not None],device='cuda')
                            v,comp=model.with_components(current,t,length,mask,context,available)
                            if t0 in frames and t0!=0:
                                xyz=(current[0]*SCALE).cpu().numpy();trajectory[t0]=xyz
                                stat=intrinsic(xyz);local.append(dict(stat,sample_key=name,timestep=t0,
                                    polar_rms=float(comp[0]['global_v'].square().mean().sqrt()),chiral_rms=float(comp[0]['chiral_effective'].square().mean().sqrt()),
                                    local_rms=float(comp[0]['local_effective'].square().mean().sqrt()),gate_mean=float(comp[0]['gate'].mean())))
                            current,context,_=e.diffusion.deterministic_reverse_step(current,t,v,mask);context=context.detach()
                            if not torch.isfinite(current).all():raise FloatingPointError(f'Nonfinite generated trajectory {name} at{t0}')
                        xyz=(current[0]*SCALE).cpu().numpy();trajectory[0]=xyz
                    torch.cuda.synchronize();seconds=time.monotonic()-started
                    stat=intrinsic(xyz);local.append(dict(stat,sample_key=name,timestep=0,
                        polar_rms=float(comp[0]['global_v'].square().mean().sqrt()),chiral_rms=float(comp[0]['chiral_effective'].square().mean().sqrt()),
                        local_rms=float(comp[0]['local_effective'].square().mean().sqrt()),gate_mean=float(comp[0]['gate'].mean())))
                    folder=OUT/'samples'/name;folder.mkdir(exist_ok=True)
                    np.savez_compressed(folder/'final_coordinates.npz',coordinates_angstrom=xyz)
                    np.savez_compressed(folder/'sparse_trajectory.npz',**{f't{t}':a for t,a in trajectory.items()})
                    distance=np.linalg.norm(xyz[:,None]-xyz[None,:],axis=-1).astype('float32');np.savez_compressed(folder/'distance_matrix.npz',distance_angstrom=distance)
                    pdb=['REMARK   1 C-ALPHA TRACE ONLY; NOT AN ALL-ATOM STRUCTURE',
                         'REMARK   2 UNK LABELS ARE PLACEHOLDERS; NO AMINO-ACID IDENTITIES INFERRED',
                         f'REMARK   3 E015 {weights.upper()} N={n} SEED={seed} SHA256={SHA}']
                    for j,(a,b,c) in enumerate(xyz,1):pdb.append(f'ATOM  {j:5d}  CA  UNK A{j:4d}    {a:8.3f}{b:8.3f}{c:8.3f}  1.00  0.00           C')
                    pdb.extend(['TER','END']);(folder/'trace_ca_only.pdb').write_text('\n'.join(pdb)+'\n')
                    meta=dict(sample_key=name,requested_length=n,seed=seed,weights=weights,primary=weights==chosen,checkpoint_sha256=SHA,
                        config_hash=CONTRACT['config_hash'],sampler='Existing CoordinateVPDiffusion.deterministic_reverse_step, exact E015 generated-context recurrence,500steps499..0',
                        inference_precision='CUDA FP32, as in existing E015 prefix evaluator; FP32 geometry islands retained',
                        terminal_frame='t0 final clean estimate after terminal reverse update; t499 initial centered Gaussian; intermediate labels are pre-update states',
                        runtime_seconds=seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),path=str(folder))
                    write(str(folder.relative_to(OUT)/'metrics.json'),dict(meta=meta,metrics=stat))
                    samples.append(meta);generation_rows.append(dict(stat,**{k:meta[k] for k in ('sample_key','requested_length','seed','weights','primary')}));traj_rows+=local
                    table('generation_manifest.parquet',samples);table('generation_metrics.parquet',generation_rows);table('trajectory_metrics.parquet',traj_rows)
                    sample_images(meta,stat,trajectory)
                    progress('generation_sample_complete',sample=name,seconds=seconds,completed=len(samples),target=50)
                    done.add(name);manifest('GENERATION_IN_PROGRESS')
            model.to('cpu');torch.cuda.empty_cache()
        assert len(samples)==50 and sum(r['primary'] for r in samples)==40
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    generation_analysis(samples,generation_rows,dev,train,cache)

def sample_images(meta,stat,trajectory):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    x=trajectory[0];n=len(x);directory=Path(meta['path']);segments=np.stack([x[:-1],x[1:]],1)
    fig=plt.figure(figsize=(10,4.5),constrained_layout=True)
    caption=f"{meta['weights']} N{n}, seed{meta['seed']} | CA only | adjacent {stat['adjacent_mean_angstrom']:.2f}±{stat['adjacent_std_angstrom']:.2f}Å, clash {stat['clash_fraction']:.3f}, Rg{stat['radius_of_gyration_angstrom']:.2f}Å"
    fig.suptitle(caption,fontsize=9)
    for i,(az,el) in enumerate([(35,25),(125,20)],1):
        ax=fig.add_subplot(1,2,i,projection='3d');line=Line3DCollection(segments,cmap='viridis',norm=colors.Normalize(0,n-1),linewidth=1.4);line.set_array(np.arange(n-1));ax.add_collection3d(line)
        ax.scatter(*x[0],color='purple',s=20,label='N');ax.scatter(*x[-1],color='gold',s=20,label='C');limits=(x.min(0),x.max(0));center=(limits[0]+limits[1])/2;span=max((limits[1]-limits[0]).max(),1)/2
        ax.set(xlim=(center[0]-span,center[0]+span),ylim=(center[1]-span,center[1]+span),zlim=(center[2]-span,center[2]+span),xlabel='x Å',ylabel='y Å',zlabel='z Å');ax.set_box_aspect([1,1,1]);ax.view_init(el,az)
    fig.savefig(directory/'trace_two_views.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True);d=np.linalg.norm(x[:,None]-x[None,:],axis=-1)
    img=axes[0].imshow(d,cmap='magma',origin='lower');fig.colorbar(img,ax=axes[0],label='CA distance Å');axes[0].set(title='Distance map',xlabel='Residue',ylabel='Residue')
    for k in (1,2,3):axes[1].plot(np.arange(n-k),np.linalg.norm(x[k:]-x[:-k],axis=-1),label=f'i,i+{k}',linewidth=.8)
    axes[1].axhspan(3.4,4.2,alpha=.15,color='gray',label='Adjacent valid range');axes[1].set(title='Local distance profiles',xlabel='Residue',ylabel='Å');axes[1].legend(fontsize=7);fig.suptitle(caption,fontsize=9)
    fig.savefig(directory/'distance_map_profiles.png',dpi=160);plt.close(fig)
    ts=sorted(trajectory,reverse=True);stats=[intrinsic(trajectory[t]) for t in ts]
    fig,axes=plt.subplots(1,3,figsize=(11,3),constrained_layout=True)
    for ax,key,label in zip(axes,['radius_of_gyration_angstrom','covariance_trace_angstrom2','adjacent_mean_angstrom'],['Rg Å','Covariance trace Å²','Adjacent distance Å']):
        ax.plot(ts,[s[key] for s in stats],marker='o',linewidth=1);ax.invert_xaxis();ax.set(xlabel='Reverse timestep',ylabel=label);ax.grid(alpha=.2)
    fig.suptitle(caption,fontsize=9);fig.savefig(directory/'trajectory_geometry.png',dpi=160);plt.close(fig)
    # Predeclared first seed per length/weight: sparse montage, not selected visually.
    if meta['seed']==15015000:
        fig=plt.figure(figsize=(12,7),constrained_layout=True)
        for i,t in enumerate(ts,1):
            a=trajectory[t];ax=fig.add_subplot(3,4,i,projection='3d');ax.plot(*a.T,linewidth=.6);ax.set_title(f't={t}',fontsize=9);ax.set_box_aspect([1,1,1]);ax.set_axis_off();ax.view_init(25,35)
        fig.suptitle(f"Predeclared trajectory montage: {meta['sample_key']}, C-alpha only",fontsize=10);fig.savefig(directory/'trajectory_montage.png',dpi=160);plt.close(fig)

def score_vector(r):
    return np.array([r['adjacent_mean_angstrom'],r['adjacent_std_angstrom'],r['clash_fraction'],r['radius_of_gyration_angstrom'],r['signed_triple_mean'],*r['covariance_eigenvalues_angstrom2']],dtype=float)

def aligned_rmsd(a,b):
    assert len(a)==len(b);a=a-a.mean(0);b=b-b.mean(0);u,_,vh=np.linalg.svd(a.T@b);fix=np.diag([1,1,np.linalg.det(u@vh)])
    return float(np.sqrt(np.mean(np.sum((a@(u@fix@vh)-b)**2,axis=-1))))

def generation_analysis(samples,generated,dev,train,cache):
    choice=read('sampler_weight_selection.json')['primary_weights'];references=[];distribution=[];diversity=[];novelty=[];scored=[];gallery=[]
    for n,st in zip(LENGTHS,STRATA,strict=True):
        held=sorted([r for r in dev if r['length_stratum']==st],key=lambda r:(abs(r['length']-n),key('e015-generation-reference-v1',r)))[:16]
        trains=sorted([r for r in train if r['length_stratum']==st],key=lambda r:(abs(r['length']-n),key('e015-generation-novelty-v1',r)))[:20]
        hr=[dict(intrinsic(cache.get(r['sample_id'])),sample_id=r['sample_id'],split='heldout',requested_length=n,length_stratum=st) for r in held]
        tr=[dict(intrinsic(cache.get(r['sample_id'])),sample_id=r['sample_id'],split='train',requested_length=n,length_stratum=st) for r in trains];references+=hr+tr
        a=np.array([r['descriptor'] for r in hr[::2]]);b=np.array([r['descriptor'] for r in hr[1::2]])
        standard=ReferenceStandardizer(a,b);h=bandwidth(standard.transform(a),standard.transform(b));p=projections(len(standard.columns));spectrum=[i for i,c in enumerate(standard.columns) if NAMES[c] in ('eig1','eig2','eig3','trace','anisotropy1','anisotropy3')]
        prim=[r for r in generated if r['requested_length']==n and r['primary']];assert len(prim)==8
        raw=np.array([r['descriptor'] for r in prim]);gen=Comparison(standard.transform(raw),standard.transform(b),raw,b,h,p,spectrum);floor=Comparison(standard.transform(a),standard.transform(b),a,b,h,p,spectrum)
        gm,fm=gen.metrics(),floor.metrics();rng=np.random.default_rng(15015);gd={k:[] for k in gm};fd={k:[] for k in fm}
        for _ in range(2000):
            ix=rng.integers(0,8,8);g,f=gen.metrics(ix),floor.metrics(ix)
            for k in gm:gd[k].append(g[k]);fd[k].append(f[k])
        comparison=[dict(metric=k,generated_vs_real=gm[k],real_vs_real_floor=fm[k],ratio_to_floor=gm[k]/fm[k] if fm[k]>1e-12 else None,
            excess=gm[k]-fm[k],generated_ci95=np.quantile(gd[k],[.025,.975]).tolist(),floor_ci95=np.quantile(fd[k],[.025,.975]).tolist(),
            excess_ci95=np.quantile(np.array(gd[k])-fd[k],[.025,.975]).tolist()) for k in gm]
        distribution.append(dict(length=n,primary_weights=choice,n=8,metrics=comparison,descriptor_columns=[NAMES[c] for c in standard.columns],reference_ids=[r['sample_id'] for r in held],reference_lengths=[r['length'] for r in held],reference_floor='Disjoint interleaved closest-length frozen heldout references8+8',bootstrap_replicates=2000))
        coordinate={r['sample_key']:np.load(OUT/'samples'/r['sample_key']/'final_coordinates.npz')['coordinates_angstrom'].astype(float) for r in prim}
        pairs=[];neighbors={k:[] for k in coordinate}
        names=list(coordinate)
        for i,k in enumerate(names):
            for j in range(i+1,len(names)):
                other=names[j];aa,bb=coordinate[k],coordinate[other];rms=aligned_rmsd(aa,bb);da=np.linalg.norm(aa[:,None]-aa[None,:],axis=-1);db=np.linalg.norm(bb[:,None]-bb[None,:],axis=-1)
                dmr=float(np.sqrt(np.mean((da-db)**2)));ds=float(distances(standard.transform(np.array([prim[i]['descriptor']])),standard.transform(np.array([prim[j]['descriptor']])))[0,0])
                pairs.append(dict(a=k,b=other,kabsch_rmsd_angstrom=rms,distance_matrix_rmse_angstrom=dmr,descriptor_distance=ds));neighbors[k].append((rms,other));neighbors[other].append((rms,k))
        diversity.append(dict(length=n,pair_count=len(pairs),median_kabsch_rmsd_angstrom=float(np.median([r['kabsch_rmsd_angstrom'] for r in pairs])),median_distance_matrix_rmse_angstrom=float(np.median([r['distance_matrix_rmse_angstrom'] for r in pairs])),
            exact_duplicate_pair_count=sum(r['kabsch_rmsd_angstrom']<1e-8 for r in pairs),duplicate_definition='Numerical identity1e-8Å; descriptive, no unsupported near-duplicate cutoff',
            nearest_generated={k:dict(kabsch_rmsd_angstrom=min(v)[0],sample_key=min(v)[1]) for k,v in neighbors.items()},pairs=pairs))
        ref_vectors=np.array([score_vector(r) for r in hr]);median=np.median(ref_vectors,0);scale=np.maximum((np.quantile(ref_vectors,.75,axis=0)-np.quantile(ref_vectors,.25,axis=0))/1.349,1e-8)
        for r in generated:
            if r['requested_length']!=n:continue
            vector=score_vector(r);score=float(np.sqrt(np.mean(((vector-median)/scale)**2))) if np.isfinite(vector).all() else 1e300;percentiles=((ref_vectors<=vector).mean(0)).tolist()
            scored.append(dict(sample_key=r['sample_key'],requested_length=n,primary=r['primary'],gallery_score=score,matched_reference_percentiles=percentiles,
                percentile_columns=['adjacent_mean','adjacent_std','clash_fraction','Rg','signed_triple_mean','eig1','eig2','eig3']))
            d=standard.transform(np.array([r['descriptor']]))
            nearest={}
            for label,rr in [('train',tr),('heldout',hr)]:
                ds=distances(d,standard.transform(np.array([q['descriptor'] for q in rr])))[0];j=int(ds.argmin());q=rr[j]
                nearest[label]=dict(sample_id=q['sample_id'],length=q['length'],standardized_descriptor_distance=float(ds[j]),searched_identities=len(rr),metric='Validated standardized34D invariant descriptor Euclidean/sqrt(d); sampled novelty, not a full structural search')
                if q['length']==n:nearest[label]['equal_length_kabsch_rmsd_angstrom']=aligned_rmsd(np.load(OUT/'samples'/r['sample_key']/'final_coordinates.npz')['coordinates_angstrom'],cache.get(q['sample_id']))
            novelty.append(dict(sample_key=r['sample_key'],requested_length=n,nearest=nearest))
        ranked=sorted([s for s in scored if s['requested_length']==n and s['primary']],key=lambda s:(s['gallery_score'],s['sample_key']))
        for category,index in [('success_gallery',0),('median_gallery',len(ranked)//2),('failure_gallery',-1)]:
            selected=ranked[index];folder=OUT/category/selected['sample_key'];folder.mkdir(exist_ok=True)
            for image in ('trace_two_views.png','distance_map_profiles.png','trajectory_geometry.png'):shutil.copyfile(OUT/'samples'/selected['sample_key']/image,folder/image)
            write(str(folder.relative_to(OUT)/'selection.json'),dict(selected,category=category,selection='Frozen reference-standardized numerical score, independent of visual inspection; best means relative best, not proven successful generation'))
            gallery.append(dict(category=category,**selected))
    table('generation_reference_metrics.parquet',references);table('generation_scores.parquet',scored)
    write('generation_distribution_metrics.json',dict(status='COMPLETE',by_length=distribution,scope='Invariant descriptor distribution only, not equality of full coordinate distributions',secondary='Two-seed secondary panel descriptive only; no calibrated population claim'))
    write('diversity_novelty_metrics.json',dict(status='COMPLETE',diversity=diversity,sampled_novelty=novelty,tm_score='Not used: no verified implementation required for this bounded analysis',galleries=gallery,
        interpretation='Nearest descriptors and sampled equal-length Kabsch cannot prove or disprove memorization across230kTRAIN; extreme distance from both references may indicate artifact. Numerical duplicate absence does not rule out broader mode collapse.'))
    write('GPU_QUEUE.json',dict(status='COMPLETE',primary_samples=40,secondary_samples=10,primary_weights=choice,checkpoint_sha256=SHA))
    finite_rate=float(np.mean([r['finite_fraction']==1 for r in generated]));valid_bond=float(np.mean([r['adjacent_valid_fraction'] for r in generated if r['primary']]));clash=float(np.mean([r['clash_fraction'] for r in generated if r['primary']]))
    # Population floor intervals guide a descriptor claim; not a full-distribution proof.
    excess=[q['excess_ci95'] for d in distribution for q in d['metrics'] if q['metric']=='energy']
    if finite_rate<1 or all(r['exact_duplicate_coordinate_pairs']>0 for r in generated):genverdict='COLLAPSED_OR_UNSTABLE'
    elif valid_bond>=.9 and clash<=.01 and all(q[0]<=0<=q[1] for q in excess):genverdict='EVIDENCE_OF_USEFUL_GENERATION'
    else:genverdict='BOUNDED_BUT_NOT_PROTEIN_LIKE'
    verdict=read('verdict.json');verdict.update(generation=genverdict,full_de_novo_samples=50,primary_de_novo_samples=40,secondary_de_novo_samples=10,
        generation_reason='Full500-step unconditional diagnostic complete; no paired target RMSD reported as generation quality',generation_finite_fraction=finite_rate,
        generation_adjacent_valid_fraction=valid_bond,generation_clash_fraction=clash,cuda_used=True,production_pause_used=False,production_modified=False)
    write('verdict.json',verdict);generation_plots(generated,distribution,diversity)
    text=['\n## Completed unconditional de novo evaluation','',f"**Generation verdict: {genverdict}**.40primary {choice} samples plus10secondary, exact500steps; finite fraction{finite_rate:.3f}; primary adjacent-valid fraction{valid_bond:.3f}; clash fraction{clash:.4f}.",'',
        'Every sample has CA-only numerical/PDB/sparse-trajectory artifacts and two-view trace, distance map/profiles, trajectory plot. Predeclared first seeds have montages. Best/median/worst galleries use the frozen numerical reference-standardized score; a best gallery is not proof of successful generation.', '',
        '| N | Adjacent mean Å | Rg Å | Adjacent valid | Clash fraction | Median pairwise Kabsch Å | Energy distance | Real-real floor |', '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d,v in zip(distribution,diversity,strict=True):
        group=[r for r in generated if r['requested_length']==d['length'] and r['primary']];en=next(q for q in d['metrics'] if q['metric']=='energy')
        text.append(f"| {d['length']} | {np.mean([r['adjacent_mean_angstrom'] for r in group]):.4g} | {np.mean([r['radius_of_gyration_angstrom'] for r in group]):.4g} | {np.mean([r['adjacent_valid_fraction'] for r in group]):.3f} | {np.mean([r['clash_fraction'] for r in group]):.4f} | {v['median_kabsch_rmsd_angstrom']:.4g} | {en['generated_vs_real']:.4g} | {en['real_vs_real_floor']:.4g} |")
    text+=['','Descriptor floors use disjoint matched-length real references and2,000bootstrap draws. They measure selected invariant summaries, not chemical stability or all-atom folding. C-alpha traces contain no inferred amino-acid identities. Sampled nearest TRAIN/heldout references and per-length diversity are in diversity_novelty_metrics.json. No uncontrolled fullTRAINsearch.', '',
        'GPU phase started only after host-visible safety checks found no active trainer/supervisor and externalGPUuse<1.5GiB; existing E015 lease acquired without productionmetadata writes. No pause/signals/resume occurred. All artifacts remained onD.']
    with (OUT/'REPORT.md').open('a') as stream:stream.write('\n'.join(text)+'\n')
    manifest('COMPLETE');progress('COMPLETE',verdict=verdict)

def generation_plots(rows,distribution,diversity):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    primary=[r for r in rows if r['primary']]
    for ax,key,label in [(axes[0,0],'adjacent_valid_fraction','Adjacent valid fraction'),(axes[0,1],'clash_fraction','Clash fraction'),(axes[1,0],'radius_of_gyration_angstrom','Rg Å')]:
        ax.boxplot([[r[key] for r in primary if r['requested_length']==n] for n in LENGTHS],tick_labels=LENGTHS);ax.set(xlabel='Length',ylabel=label);ax.grid(alpha=.2)
    axes[1,1].plot(LENGTHS,[r['median_kabsch_rmsd_angstrom'] for r in diversity],marker='o');axes[1,1].set(xlabel='Length',ylabel='Median generated-pair Kabsch RMSD Å');axes[1,1].grid(alpha=.2)
    fig.savefig(OUT/'plots/generation_quality_diversity.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
    for key,label in [('generated_vs_real','Generated vs real'),('real_vs_real_floor','Real vs real')]:ax.plot(LENGTHS,[next(q[key] for q in d['metrics'] if q['metric']=='energy') for d in distribution],marker='o',label=label)
    ax.set(xlabel='Length',ylabel='Standardized descriptor energy distance',title='Descriptor population comparison');ax.legend();ax.grid(alpha=.2)
    fig.savefig(OUT/'plots/generation_descriptor_floor.png',dpi=160);plt.close(fig)

def validate():
    from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
    from protein_distance_diffusion.evaluation.reverse_field_diagnostic import reverse_identity
    d=CoordinateVPDiffusion(500);g=torch.Generator().manual_seed(15015);mask=torch.ones(1,48,dtype=torch.bool)
    x=torch.randn(1,48,3,generator=g);x=x-x.mean(1,keepdim=True);eps=centered_coordinate_noise(x,mask,generator=g)
    errors=[reverse_identity(d,x,eps,torch.tensor([t]),mask) for t in (1,5,50,250,400,490,499)]
    sample=intrinsic((x[0]*SCALE).numpy());assert sample['length']==48 and sample['finite_fraction']==1
    paired=estimate([1,2,3,4],[2,3,4,5],['a','a','b','b']);assert abs(paired['absolute_difference']-1)<1e-12 and paired['difference_ci95']==[1.,1.]
    try:
        gpu_guard();guard='Host independently free; guard passed; validation did not initialize CUDA'
    except RuntimeError as exc:guard=str(exc)
    assert not torch.cuda.is_initialized()
    write('logs/evaluator_validation.json',dict(existing_reverse_identity_max_error=max(errors),bootstrap_known_paired_difference=paired,
        intrinsic_fixture_finite=True,gpu_guard_result=guard,cuda_initialized=False,fixture='Synthetic validation only; not a de novo generated sample; no fixture artifact counted as generation'))
    progress('EVALUATOR_VALIDATED',guard=guard)

@torch.no_grad()
def reconstruction_visuals():
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    panel,_,_,cache=panels();choice=read('sampler_weight_selection.json')['primary_weights'];e=model_engine()
    saved,h=pinned_load(ROOT/'production/checkpoints/epoch_1_fraction_0.50.pt');assert h==SHA;verify(saved,e)
    model=e.ema if choice=='ema' else e.model;model.load_state_dict(saved['ema' if choice=='ema' else 'model'],strict=True);model.eval();del saved
    chosen=[next(r for r in panel if r['eval_split']=='heldout' and r['length_stratum']==s and r['pair_index']==0) for s in STRATA]
    records=[]
    for row in chosen:
        x,n,m=coordinate_batch([row],cache,SCALE,device='cpu');arrays={'clean_target_angstrom':(x[0]*SCALE).numpy()};metrics=[]
        for t0 in GRID:
            t=torch.tensor([t0]);seed=noise_seed('e015-absolute-matched-noise-v1',row['pair_id'],t0);g=torch.Generator().manual_seed(seed)
            batch=e.diffusion.make_training_batch(x,m,timesteps=t,generator=g);v,comp=prediction(model,e.diffusion,batch.noisy_coordinates,t,n,m)
            xhat=e.diffusion.reconstruct_x0(batch.noisy_coordinates,t,v,m);arrays[f'reconstruction_t{t0}_angstrom']=(xhat[0]*SCALE).numpy()
            stat=extended_metrics(xhat[0],x[0],v[0],batch.coordinate_v_target[0],SCALE,comp[0]);metrics.append(dict(timestep=t0,**stat))
        directory=OUT/'reconstruction_examples'/row['sample_id'];directory.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(directory/'target_conditioned_coordinates.npz',**arrays)
        write(str(directory.relative_to(OUT)/'metrics.json'),dict(identity=row['sample_id'],weights=choice,selection='Predeclared pair_index0 heldout per stratum, not performance selection',metrics=metrics,unconditional=False))
        fig=plt.figure(figsize=(13,7),constrained_layout=True)
        for i,(label,a) in enumerate(arrays.items(),1):
            ax=fig.add_subplot(2,3,i,projection='3d');ax.plot(*a.T,linewidth=.7);ax.scatter(*a[0],color='purple',s=10);ax.scatter(*a[-1],color='gold',s=10)
            ax.set_box_aspect([1,1,1]);ax.view_init(25,35)
            if i==1:title='Clean target'
            else:
                stat=metrics[i-2];title=f"t{stat['timestep']} RMSD{stat['aligned_rmsd_angstrom']:.2f}Å\nadj{stat['adjacent_mean_angstrom']:.2f}Å clash{stat['clash_fraction']:.3f}"
            ax.set_title(title,fontsize=9);ax.set(xlabel='x Å',ylabel='y Å',zlabel='z Å')
        fig.suptitle(f"TARGET-CONDITIONED reconstruction · {row['sample_id']} · N{row['length']} · {choice} · NOT de novo generation",fontsize=11)
        fig.savefig(directory/'reconstruction_montage.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
        for ax,k in zip(axes,(1,2,3)):
            for label,a in arrays.items():ax.plot(np.linalg.norm(a[k:]-a[:-k],axis=-1),linewidth=.7,label='Target' if label.startswith('clean') else label.split('_')[1])
            ax.set(title=f'i,i+{k}',xlabel='Residue',ylabel='Distance Å');ax.grid(alpha=.2)
        axes[-1].legend(fontsize=7);fig.suptitle(f'Target-conditioned local profiles: {row["sample_id"]}');fig.savefig(directory/'local_distance_profiles.png',dpi=160);plt.close(fig)
        records.append(dict(sample_id=row['sample_id'],length=row['length'],weights=choice,path=str(directory)))
        progress('reconstruction_visual_complete',identity=row['sample_id'])
    write('reconstruction_visual_manifest.json',dict(rows=records,scope='Five predeclared heldout target-conditioned examples; not unconditional samples',cuda_initialized=torch.cuda.is_initialized()))
    assert not torch.cuda.is_initialized();manifest('CPU_COMPLETE_GPU_QUEUED')

if __name__=='__main__':
    if MODE=='cpu':cpu()
    elif MODE=='report':report()
    elif MODE=='generate':generate()
    elif MODE=='validate':validate()
    elif MODE=='visuals':reconstruction_visuals()
    elif MODE=='inventory':inventory();panels();manifest('INVENTORY_VERIFIED');progress('INVENTORY_VERIFIED')
    else:raise ValueError(MODE)
