"""Frozen-student offline failure geometry and deployable global scaling pilot."""
import os
import sys
from pathlib import Path

REPO = Path('/mnt/d/Simone/proteinGen')
ROOT = REPO / 'outputs/e010_student_feasibility_margin_v1'


def storage():
    for key in ('TMPDIR', 'TMP', 'TEMP'):
        os.environ[key] = str(ROOT / 'tmp')
    for key in ('XDG_CACHE_HOME', 'MPLCONFIGDIR', 'CUDA_CACHE_PATH', 'TORCH_HOME', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR'):
        os.environ[key] = str(ROOT / 'cache')
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    sys.dont_write_bytecode = True


storage()
import argparse
import concurrent.futures
import json
import multiprocessing
import time
import numpy as np
import torch
import e010_nonprivileged_student_v1 as old
storage()

PARENT = old.ROOT
ALPHAS = [1., .9, .8, .7, .6, .5, .4, .3, .2]
PROTOCOL = dict(experiment='e010_student_feasibility_margin_v1', frozen_commit='fc76211',
    parent_completion_sha256='8113a463cd66b2baa63433a3f473ffc11bda96b977ee2336c74185257a53d601',
    architecture='unchanged 118723-parameter Phase4D width64 blocks2', splits='reuse frozen 1400/300/300 identity/group-isolated TRAIN pilot splits',
    alphas=ALPHAS, alpha_selection='global constants; no per-sample selection or X-based inference shield',
    evaluation='all 16 logical teacher states including zero tails; recursive own-state K16; original baseline-anchored exact gates',
    retention='ratio of summed mean-local-RMSE reductions to archived original privileged teacher reductions',
    geometry='exact normalized inequality slack by family at start, teacher endpoint, student endpoint; action error norm/cosine; central directional derivative at teacher endpoint along student-minus-teacher error with h=1e-4',
    risk_ratio='maximum positive directional slack consumption / max(teacher normalized slack,1e-12); first-order diagnostic, not exact distance or KKT proof',
    risk_bins=[0., .25, .5, 1., 2., 10., 1e300], near_boundary_normalized_thresholds=[1e-8,1e-5,.001,.01],
    phase_C_gate='validation alpha=0.5 must reduce recursive failed-state rate at least 50% relative alpha=1 while retaining at least 25% of original teacher repair; test never opens gate',
    phase_C_buffers=[.001,.01,.05],
    phase_C_buffer_definition='each original normalized global/quartet inequality slack >= buffer (global allowance 1e-8 included); eligible residue action norm <= 0.10*(1-buffer); eligibility equalities stay exactly zero; exact telemetry gates also pass',
    phase_C_search='largest passing rho on frozen grid {1,.99,...,0}; no monotonicity assumption; if none passes retain exact zero with explicit unbufferable flag, never claim positive buffer',
    phase_C_selection='choose smallest buffered level meeting >=90% nonzero TRAIN target coverage and >=50% original TRAIN one-step repair; otherwise stop without training; do not tune on test',
    phase_C_training='same capacity, action imitation only, identity-first sampling, original AdamW settings; at most 6000 updates, min3000, validation every1000, patience2; early stop on raw validation K16 repair retention with safety recorded separately; no X network input',
    no_SLSQP=True, no_new_teacher=True, no_full_corpus=True, no_capacity_increase=True,
    storage='all generated artifacts and caches on D:', workers=8)


def write(path, obj):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def initialize():
    old.initialize()
    storage()
    torch.set_num_threads(1)


def prepare():
    for name in ('', 'tmp', 'cache', 'logs', 'records', 'geometry', 'actions'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    assert old.sha(PARENT / 'COMPLETED.json') == PROTOCOL['parent_completion_sha256']
    completion = json.loads((PARENT / 'COMPLETED.json').read_text())
    for path, digest in completion['files'].items():
        assert old.sha(path) == digest, path
    manifest = dict(commit='fc76211', files=completion['files'], completion_sha256=old.sha(PARENT/'COMPLETED.json'),
        model_sha256=old.sha(PARENT/'models/imitation_best.pt'), splits_sha256=old.sha(PARENT/'splits.json'),
        identity_list_sha256=json.loads((PARENT/'splits.json').read_text())['identity_list_sha256'])
    if (ROOT/'protocol.json').exists():
        assert json.loads((ROOT/'protocol.json').read_text()) == PROTOCOL
    else:
        write(ROOT/'protocol.json', PROTOCOL)
        write(ROOT/'parent_freeze.json', manifest)
    print(json.dumps(dict(recovery='clean frozen fc76211; all archived student files verified; no training or new teacher computation',
        parent_files=len(completion['files']), protocol_sha256=old.sha(ROOT/'protocol.json'), splits=manifest['identity_list_sha256'])), flush=True)


def init_worker():
    global MODEL
    initialize()
    MODEL = old.Student().eval()
    MODEL.load_state_dict(torch.load(PARENT/'models/imitation_best.pt', map_location='cpu', weights_only=False)['model'])


def family_slices(ctx):
    return dict(aligned=(0,1), continuous_chirality=(1,2), **{k:(lo+2,hi+2) for k,(lo,hi) in ctx.quartets.slices.items()})


@torch.no_grad()
def score(ctx, p, d):
    a = old.action_metrics(ctx,p,d,dict(policy_seconds=0.))
    c = ctx.constraints(p+torch.from_numpy(d)).numpy()
    bounds = np.zeros_like(c); bounds[:2] = 1e-8
    slack = bounds-c
    a['failures'] = {k:bool(np.any(slack[lo:hi]<0)) for k,(lo,hi) in family_slices(ctx).items()}
    a['failures'].update(new_inversion=a['after']['safety']['quartets']['new_inversions']>0,
        assessability=a['after']['safety']['quartets']['assessability_lost']>0,
        cap=a['cap_violations']>0, eligibility=a['eligibility_violations']>0)
    a['hard_failed'] = not a['after']['safety']['passes'] or a['cap_violations']>0 or a['eligibility_violations']>0
    return a, slack


def worker(task):
    row, split = task
    destination = ROOT/'records'/f'{split}_{row["index"]:04d}.json'
    if destination.exists():
        return str(destination)
    started = time.perf_counter()
    item = old.load_identity(row)
    archive = PARENT/'evaluation'/f'imitation_{split}'/f'identity_{row["index"]:04d}.npz'
    saved = np.load(archive,allow_pickle=False)
    records = []; geometry_arrays = {}; action_arrays = {}
    for ci,c in enumerate(item['conditions']):
        b = {key:torch.from_numpy(c[name][None]) for key,name in [('pg','P0'),('source','source'),('target','target'),('mask','mask')]}
        ctx = old.pilot.Context(b)
        baseline = old.metrics(ctx,b['pg'])
        geometries = []; one = {str(alpha):[] for alpha in ALPHAS}
        snapshots = []; derivatives = []
        for step in range(16):
            state,mask,teacher,_ = old.sample(item,ci,step)
            p = torch.from_numpy(state[None]); td = teacher[None]
            raw = saved[f'{c["condition"]}_raw_one_delta'][step]
            before = old.metrics(ctx,p)
            ta, tm = score(ctx,p,td)
            _, sm = score(ctx,p,np.zeros_like(td))
            error = raw-td
            h = 1e-4
            with torch.no_grad():
                endpoint = p+torch.from_numpy(td)
                deriv = (ctx.constraints(endpoint+torch.from_numpy(h*error))-ctx.constraints(endpoint-torch.from_numpy(h*error))).numpy()/(2*h)
            ratio = np.maximum(deriv,0)/np.maximum(tm,1e-12)
            comparison = old.frozen.old.comparison(raw,td,old.local_representation(p,b['mask'])['eligible'].numpy())
            geo = dict(step=step+1, **comparison, error_l2=float(np.linalg.norm(error)), error_rms=float(np.sqrt(np.mean(error**2))),
                teacher_nonzero=bool(np.any(td)), teacher_cap_slack=float(.1-np.linalg.norm(td,axis=-1).max()),
                teacher_safe=not ta['hard_failed'], directional_risk_ratio=float(ratio.max()), families={})
            for family,(lo,hi) in family_slices(ctx).items():
                if hi>lo:
                    geo['families'][family]=dict(start_min=float(sm[lo:hi].min()),teacher_min=float(tm[lo:hi].min()),
                        directional_risk_ratio=float(ratio[lo:hi].max()))
            for alpha in ALPHAS:
                a, rm = score(ctx,p,raw*alpha)
                a.update(step=step+1, before=before,teacher_after=ta['after'])
                one[str(alpha)].append(a)
                if alpha==1.:
                    geo['raw_failed']=a['hard_failed']; geo['raw_failures']=a['failures']
                    for family,(lo,hi) in family_slices(ctx).items():
                        if hi>lo:geo['families'][family]['raw_min']=float(rm[lo:hi].min())
                    snapshots.append(np.stack([sm,tm,rm])); derivatives.append(deriv)
            geometries.append(geo)
        geometry_arrays[f'{c["condition"]}_slack_start_teacher_student']=np.stack(snapshots)
        geometry_arrays[f'{c["condition"]}_error_directional_derivative']=np.stack(derivatives)
        rollouts = {}
        for alpha in ALPHAS:
            p=b['pg'].clone(); steps=[]; ds=[]
            for step in range(16):
                tick=time.perf_counter()
                if alpha==1.:
                    d=saved[f'{c["condition"]}_raw_rollout_delta'][step]
                    policy_seconds=0.
                else:
                    d,policy=old.predict(MODEL,p,b['mask']); d=d*alpha
                    policy_seconds=policy['policy_seconds']
                a,_=score(ctx,p,d)
                a.update(step=step+1,policy_seconds=policy_seconds,wall_seconds=time.perf_counter()-tick)
                steps.append(a); ds.append(d); p=p+torch.from_numpy(d)
            rollouts[str(alpha)]=dict(final=steps[-1]['after'],steps=steps)
            action_arrays[f'{c["condition"]}_alpha_{alpha}_delta']=np.stack(ds)
        records.append(dict(identity=row['sample_id'],condition=c['condition'],length=row['length'],length_bin=row['stratum'],split=split,
            baseline=baseline,teacher_final=c['teacher']['final'],geometry=geometries,one_step=one,rollouts=rollouts,
            constraint_slices=family_slices(ctx)))
    for folder,arrays in [('geometry',geometry_arrays),('actions',action_arrays)]:
        old.frozen.old.atomic_npz(ROOT/folder/f'{split}_{row["index"]:04d}.npz',**arrays)
    write(destination,dict(protocol_sha256=old.sha(ROOT/'protocol.json'),model_sha256=old.sha(PARENT/'models/imitation_best.pt'),
        wall_seconds=time.perf_counter()-started,rows=records))
    return str(destination)


def run():
    prepare()
    splits=json.loads((PARENT/'splits.json').read_text())['splits']
    rows=json.loads((old.PARENT/'selection.json').read_text())['identities']
    tasks=[(r,s) for s in ('validation','test') for r in rows if r['sample_id'] in set(splits[s])]
    started=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=PROTOCOL['workers'],mp_context=multiprocessing.get_context('spawn'),initializer=init_worker) as pool:
        for i,result in enumerate(pool.map(worker,tasks),1):
            if i%10==0:print(json.dumps(dict(completed=i,total=len(tasks),wall_seconds=time.perf_counter()-started,last=result)),flush=True)
    write(ROOT/'PHASE_AB_COMPLETED.json',dict(identities=len(tasks),wall_seconds=time.perf_counter()-started,protocol_sha256=old.sha(ROOT/'protocol.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run'])
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    else:run()
