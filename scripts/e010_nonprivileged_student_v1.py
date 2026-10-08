"""Identity-held-out distillation of the frozen D:-resident structural pilot."""
import os
import sys
from pathlib import Path

REPO = Path('/mnt/d/Simone/proteinGen')
ROOT = REPO / 'outputs/e010_nonprivileged_structural_student_pilot_v1'
PARENT = REPO / 'outputs/e010_structural_teacher_diversity_scaleout_v1'


def storage():
    for key in ('TMPDIR', 'TMP', 'TEMP'):
        os.environ[key] = str(ROOT / 'tmp')
    for key in ('XDG_CACHE_HOME', 'MPLCONFIGDIR', 'CUDA_CACHE_PATH', 'TORCH_HOME', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR'):
        os.environ[key] = str(ROOT / 'cache')
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    sys.dont_write_bytecode = True


storage()
sys.path.insert(0, str(REPO / 'scripts'))
import run_e010_constraint_projected_v1 as frozen
storage()
import argparse
import collections
import concurrent.futures
import hashlib
import json
import multiprocessing
import time

import numpy as np
import torch
from torch import nn
from protein_distance_diffusion.models.e010_hybrid_local import LocalGeometryBranch, local_representation

pilot = frozen.pilot
STRATA = ['20-64', '65-128', '129-256', '257-384', '385-500']
CONFIG = dict(experiment='non_privileged_structural_student_pilot_v1', parent_commit='6d1ed30', seed=610301,
    parent_completion_sha256='f81c030b4b99f2c259b4d57ca28f9f7665e94d14ad631ff8e2d96bd42f519454',
    input='41 existing Phase4D P_t-only invariant/parity-sensitive channels, local equivariant frame and mask',
    no_target_input=True, no_target_features=True, no_gradient_input=True, no_privileged_constraint_input=True,
    no_sequence=True, no_condition_input=True, no_step_input=True,
    architecture='existing LocalGeometryBranch(width=64,blocks=2), differentiable per-residue 0.10 A cap',
    width=64, blocks=2, K=16, action_cap=.1, batch_size=8, maximum_updates=6000,
    minimum_updates=3000, validation_every=1000, patience_checks=2, retention_min_delta=.002,
    optimizer='AdamW', lr=3e-4, weight_decay=1e-4, clip_norm=1.,
    arms={'imitation': {'imitation':1., 'local':0.}, 'imitation_local': {'imitation':1., 'local':.1}},
    imitation_loss='mean over eligible residues within protein of squared Cartesian action error / 0.10^2; then mean over sampled identities',
    auxiliary_local='mean frozen offsets 1/2/3 distance MSE at P_t+delta / same-state local MSE; target only inside loss; clamp denominator at 1e-12',
    sampling='uniform TRAIN identity with replacement, then uniform condition, then uniform logical step 0..15 including frozen exact zero tail',
    splits='deterministic whole split-group greedy length balancing toward 70/15/15; stronger than identity isolation',
    validation_monitor='12 identities per length bin, SHA256 rank within validation; shielded recursive K16 ratio of summed mean-local-RMSE repair against archived teacher',
    final_selection='highest full-validation shielded rollout retention among early-stopped arms; test never used for selection',
    shield='frozen projection_2 exact acceptance/backtracking block: scales 1,.5,.25,.125; retain lowest strict-improving local objective exact-safe candidate; exact zero fallback; no new projection solve',
    shield_requires_X=True, shield_thresholds_changed=False,
    inference='CPU float64 P_t geometry/frame; float32 CPU network; float64 action and exact closed-ball ULP guard',
    training='CUDA float64 P_t geometry/frame and loss; float32 network; deterministic, TF32 off',
    near_zero_threshold=1e-8, eval_workers=8,
    success=dict(heldout_rollout_RMSE_retention_min=.75, raw_rollout_RMSE_retention_min=.50,
        shield_intervention_rate_max=.50, shield_zero_rejection_rate_max=.10,
        every_length_condition_RMSE_retention_min=.50, shielded_hard_violations_max=0,
        CPU_network_geometry_median_ms_max=5., CPU_shielded_p95_ms_max=100.),
    capacity='no automatic larger model; optional same-capacity 32-state memorization diagnostic, 1000 updates, never selected',
    no_SLSQP=True, no_teacher_rerun=True, no_full_corpus=True)
DATA = None
EVAL_MODEL = None


def sha(path):
    return pilot.sha(path)


def rank(value):
    return hashlib.sha256(f"{CONFIG['experiment']}|{value}".encode()).hexdigest()


def dirs():
    assert str(ROOT.resolve()).startswith('/mnt/d/')
    for folder in ('', 'tmp', 'cache', 'logs', 'models', 'evaluation'):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)


def initialize():
    frozen.old.initialize()
    storage()
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Repeated subprocess imports may set defaults; no SLSQP or new teacher calls.
    def forbidden(*args, **kwargs):
        raise RuntimeError('Teacher recomputation and optimization solves forbidden in student pilot')
    frozen.teacher = forbidden


class Student(nn.Module):
    """The forward signature cannot accept target X, context, gradients or labels."""
    def __init__(self):
        super().__init__()
        self.local = LocalGeometryBranch(width=CONFIG['width'], blocks=CONFIG['blocks'])

    def forward(self, p, mask):
        out = local_representation(p.detach(), mask)
        valid = out['eligible']
        h = self.local.input(out['features'].float()) * valid[...,None]
        for block in self.local.blocks:
            h = block(h, valid)
        u = self.local.head(h).double() * valid[...,None]
        d = torch.einsum('bnij,bnj->bni',out['frame'],u)
        norm = d.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        return d * torch.clamp(.1 / norm, max=1.) * out['eligible'][..., None]


def prepare():
    dirs()
    assert sha(PARENT / 'COMPLETED.json') == CONFIG['parent_completion_sha256']
    completion = json.loads((PARENT / 'COMPLETED.json').read_text())
    for path, digest in completion['files'].items():
        assert sha(path) == digest, path
    parent_results = json.loads((PARENT / 'results.json').read_text())
    assert sha(REPO / 'scripts/e010_diversity_scaleout_v1.py') == parent_results['runner_sha256']
    assert sha(REPO / 'scripts/report_e010_diversity_scaleout_v1.py') == parent_results['report_source_sha256']
    parent_freeze = json.loads((PARENT / 'parent_freeze.json').read_text())
    assert sha(frozen.__file__) == parent_freeze['teacher_source_sha256']
    selected = json.loads((PARENT / 'selection.json').read_text())['identities']
    import pandas as pd
    metadata = pd.read_parquet(REPO / 'data/full/splits/split_assignments.parquet')
    allowed = set(metadata[metadata.split == 'train'].sample_id)
    assert len(selected) == len({r['sample_id'] for r in selected}) == 2000
    assert all(r['sample_id'] in allowed for r in selected)
    groups = collections.defaultdict(list)
    for row in selected:
        groups[row['split_group_id']].append(row)
    fractions = {'train':.70, 'validation':.15, 'test':.15}
    counts = {s:collections.Counter() for s in fractions}
    assignments = {}
    # Deterministic largest-groups first; minimize normalized length/global deficit.
    for group, rows in sorted(groups.items(), key=lambda kv:(-len(kv[1]), rank(kv[0]))):
        bins = collections.Counter(r['stratum'] for r in rows)
        costs = {}
        for split, fraction in fractions.items():
            cost = 0.
            for s, f in fractions.items():
                for length in STRATA:
                    n = counts[s][length] + (bins[length] if s == split else 0)
                    cost += (n-400*f)**2 / (400*f)
                n = sum(counts[s].values()) + (len(rows) if s == split else 0)
                cost += (n-2000*f)**2 / (2000*f)
            costs[split] = cost
        chosen = min(costs, key=lambda s:(costs[s], rank(f'{group}|{s}')))
        counts[chosen].update(bins)
        for row in rows:
            assignments[row['sample_id']] = chosen
    splits = {s:sorted(r['sample_id'] for r in selected if assignments[r['sample_id']] == s) for s in fractions}
    monitors = []
    for length in STRATA:
        candidates = [r['sample_id'] for r in selected if assignments[r['sample_id']] == 'validation' and r['stratum'] == length]
        assert len(candidates) >= 12
        monitors.extend(sorted(candidates, key=lambda sid:rank(f'monitor|{sid}'))[:12])
    manifest = dict(splits=splits, assignments=assignments, counts={s:len(v) for s,v in splits.items()},
        by_length={s:dict(v) for s,v in counts.items()}, monitor=sorted(monitors),
        identity_list_sha256={s:hashlib.sha256(('\n'.join(v)+'\n').encode()).hexdigest() for s,v in splits.items()},
        split_groups={s:sorted({r['split_group_id'] for r in selected if assignments[r['sample_id']] == s}) for s in fractions},
        parent_selection_sha256=sha(PARENT / 'selection.json'), source_sha256=sha(__file__), protocol=CONFIG)
    for s in fractions:
        for t in fractions:
            if s != t:
                assert not (set(manifest['split_groups'][s]) & set(manifest['split_groups'][t]))
    for name, value in [('protocol.json', CONFIG), ('splits.json',manifest), ('parent_freeze.json',dict(
        commit='6d1ed30', parent_completion_sha256=CONFIG['parent_completion_sha256'],
        teacher_sha256=sha(frozen.__file__), corpus_freeze_sha256=sha(PARENT / 'CORPUS_FROZEN.json'),
        unchanged=True, files_verified=len(completion['files'])) )]:
        path = ROOT / name
        if path.exists():
            assert json.loads(path.read_text()) == value, path
        else:
            pilot.atomic_json(path, value)
    print('PREPARED '+json.dumps(dict(counts=manifest['counts'], by_length=manifest['by_length'])), flush=True)


def selection():
    assert json.loads((ROOT / 'protocol.json').read_text()) == CONFIG
    return json.loads((ROOT / 'splits.json').read_text())


def load_identity(row):
    with np.load(row['input_archive'], allow_pickle=False) as z:
        inputs = {k:z[k].copy() for k in z.files}
    path = PARENT / 'trajectories' / f"identity_{row['index']:04d}.npz"
    with np.load(path, allow_pickle=False) as z:
        actions = {k:z[k].copy() for k in z.files}
    records = json.loads(path.with_suffix('.json').read_text())
    conditions = []
    for ci, condition in enumerate(inputs['conditions']):
        lo, hi = actions['offsets'][ci:ci+2]
        ds = actions['delta'][lo:hi]
        states = [inputs['P0'][ci].astype(np.float64)]
        for d in ds:
            states.append(states[-1] + d[0])
        conditions.append(dict(P0=states[0], states=states, delta=ds, target=inputs['target'].astype(np.float64),
            mask=inputs['mask'], source=inputs['source'][ci].astype(np.float64), condition=int(condition),
            teacher=records['conditions'][str(condition)]))
    return dict(row=row, conditions=conditions)


def dataset():
    global DATA
    if DATA is None:
        rows = json.loads((PARENT / 'selection.json').read_text())['identities']
        DATA = {r['sample_id']:load_identity(r) for r in rows}
    return DATA


def sample(item, ci, step):
    c = item['conditions'][ci]
    p = c['states'][min(step, len(c['states'])-1)]
    d = c['delta'][step, 0] if step < len(c['delta']) else np.zeros_like(p)
    return p, c['mask'], d, c['target']


def batch(samples, device):
    n = max(len(v[0]) for v in samples)
    p = np.zeros((len(samples), n, 3)); d = p.copy(); target = p.copy()
    mask = np.zeros((len(samples), n), dtype=bool)
    for i, (v, m, action, y) in enumerate(samples):
        p[i,:len(v)] = v; mask[i,:len(v)] = m
        d[i,:len(v)] = action; target[i,:len(v)] = y
    return [torch.from_numpy(a).to(device) for a in (p, mask, d, target)]


def local_mse(p, y, mask):
    out = []
    for k in (1,2,3):
        valid = mask[:,k:] & mask[:,:-k]
        error = ((p[:,k:]-p[:,:-k]).norm(dim=-1) - (y[:,k:]-y[:,:-k]).norm(dim=-1)).square()
        out.append((error*valid).sum(1) / valid.sum(1).clamp_min(1))
    return torch.stack(out).mean(0)


def action_loss(model, p, mask, label):
    d = model(p, mask)
    eligible = local_representation(p, mask)['eligible']
    loss = (((d-label).square().sum(-1)*eligible).sum(1) / eligible.sum(1).clamp_min(1) / .1**2).mean()
    energy = ((label.square().sum(-1)*eligible).sum(1) / eligible.sum(1).clamp_min(1) / .1**2).mean()
    return loss, energy, d


@torch.no_grad()
def probe(model, identities, device):
    samples = [sample(DATA[sid], ci, step) for sid in identities for ci in range(3) for step in (0,7,15)]
    errors = []; energies = []
    for start in range(0,len(samples),8):
        p,m,d,_ = batch(samples[start:start+8], device)
        error, energy, _ = action_loss(model,p,m,d)
        errors.append(float(error)*len(p)); energies.append(float(energy)*len(p))
    return dict(normalized_imitation=sum(errors)/len(samples), zero_predictor_error=sum(energies)/len(samples),
        error_to_zero_baseline=sum(errors)/sum(energies), samples=len(samples))


def save_model(path, model, **extra):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()}, config=CONFIG,
            splits_sha256=sha(ROOT / 'splits.json'), **extra), f)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


@torch.no_grad()
def predict(model, current, mask):
    start = time.perf_counter()
    d = model(current, mask).numpy()
    # Frozen deterministic closed-ball handling belongs to cap enforcement, not X shielding.
    d, guard = pilot.project(d)
    return d, dict(policy_seconds=time.perf_counter()-start, guard=guard)


@torch.no_grad()
def shield(ctx, current, proposal):
    """Literal frozen projection_2 candidate acceptance; no teacher solve or new rule."""
    before = best = float(ctx.values(current)['local'])
    incumbent = np.zeros_like(proposal)
    chosen = 0.
    trials = []
    for scale in frozen.CONFIG['backtracking']:
        action, _ = pilot.project(proposal*scale)
        p = current + torch.from_numpy(action)
        safe = ctx.safety(p)
        objective = float(ctx.values(p)['local'])
        trials.append(dict(scale=scale, passes=safe['passes'], objective=objective))
        if safe['passes'] and objective < best:
            incumbent, best, chosen = action, objective, scale
    assert ctx.safety(current+torch.from_numpy(incumbent))['passes']
    return incumbent, dict(scale=chosen, intervention=not np.array_equal(incumbent,proposal),
        rejected_nonzero=bool(np.any(proposal) and not np.any(incumbent)), trials=trials, objective_before=before, objective_after=best)


def metrics(ctx,p):
    return pilot.scalar_metrics(ctx,p)


def action_metrics(ctx, p, d, policy, cert=None):
    after = metrics(ctx,p+torch.from_numpy(d))
    eligible = local_representation(p,ctx.b['mask'])['eligible'].numpy()
    norms = np.linalg.norm(d,axis=-1)
    return dict(after=after, action_rms=float(np.sqrt(np.square(d[eligible]).sum(-1).mean())),
        action_max=float(norms.max()), cap_violations=int((norms > .1).sum()),
        eligibility_violations=int(np.any(d!=0,axis=-1)[~eligible].sum()),
        exact_zero=not bool(np.any(d)), near_zero=bool(norms.max() <= CONFIG['near_zero_threshold']),
        policy_seconds=policy['policy_seconds'], shield=cert)


def init_evaluator(path):
    global EVAL_MODEL
    initialize()
    EVAL_MODEL = Student().eval()
    payload = torch.load(path,map_location='cpu',weights_only=False)
    assert payload['splits_sha256'] == sha(ROOT / 'splits.json')
    EVAL_MODEL.load_state_dict(payload['model'])


def evaluate_identity(task):
    row, quick, archive_folder = task
    item = load_identity(row)
    results = []
    arrays = {}
    for ci,c in enumerate(item['conditions']):
        b = {key:torch.from_numpy(c[name][None]) for key,name in [('pg','P0'),('source','source'),('target','target'),('mask','mask')]}
        ctx = pilot.Context(b)
        baseline = metrics(ctx,b['pg'])
        one = []
        one_arrays = {'raw':[], 'shielded':[]}
        if not quick:
            for step in range(16):
                state,mask,gold,_ = sample(item,ci,step)
                p = torch.from_numpy(state[None])
                before = metrics(ctx,p)
                raw, policy = predict(EVAL_MODEL,p,b['mask'])
                shield_begin=time.perf_counter()
                safe, cert = shield(ctx,p,raw)
                shield_seconds=time.perf_counter()-shield_begin
                for name,d in [('raw',raw),('shielded',safe)]:
                    one_arrays[name].append(d)
                    a = action_metrics(ctx,p,d,policy,cert if name=='shielded' else None)
                    comparison = frozen.old.comparison(d,gold[None],local_representation(p,b['mask'])['eligible'].numpy())
                    one.append(dict(step=step+1,mode=name,before=before,
                        inference_seconds=policy['policy_seconds']+(shield_seconds if name=='shielded' else 0.),
                        teacher_after=c['teacher']['horizons'][str(step+1)] if str(step+1) in c['teacher']['horizons'] else
                            metrics(ctx,torch.from_numpy(c['states'][min(step+1,len(c['states'])-1)][None])),
                        **comparison, **a))
        rollouts = {}
        for mode in (['shielded'] if quick else ['raw','shielded']):
            p = b['pg'].clone(); steps = []; ds = []
            for step in range(16):
                start = time.perf_counter()
                raw, policy = predict(EVAL_MODEL,p,b['mask'])
                d, cert = shield(ctx,p,raw) if mode == 'shielded' else (raw,None)
                inference_seconds = time.perf_counter()-start
                a = action_metrics(ctx,p,d,policy,cert)
                if mode == 'shielded':
                    assert a['after']['safety']['passes'] and a['cap_violations']==a['eligibility_violations']==0
                ds.append(d); p = p + torch.from_numpy(d)
                steps.append(dict(step=step+1,inference_seconds=inference_seconds,**a))
            rollouts[mode] = dict(final=steps[-1]['after'],steps=steps)
            if not quick:
                arrays[f'{c["condition"]}_{mode}_rollout_delta']=np.stack(ds)
                arrays[f'{c["condition"]}_{mode}_one_delta']=np.stack(one_arrays[mode])
        results.append(dict(identity=row['sample_id'],condition=c['condition'],length=row['length'],length_bin=row['stratum'],
            baseline=baseline,teacher_final=c['teacher']['final'],teacher_steps=c['teacher']['steps'],one_step=one,rollouts=rollouts))
    if not quick:
        destination=Path(archive_folder)/f'identity_{row["index"]:04d}.npz'
        frozen.old.atomic_npz(destination,**arrays)
        # Independent reconstruction of all final evaluation shielded rollouts.
        with np.load(destination,allow_pickle=False) as saved:
            for c in item['conditions']:
                b={key:torch.from_numpy(c[name][None]) for key,name in [('pg','P0'),('source','source'),('target','target'),('mask','mask')]}
                ctx=pilot.Context(b); p=b['pg'].clone()
                for d in saved[f'{c["condition"]}_shielded_rollout_delta']:
                    frozen.old.checks(dict(ctx=ctx,b=b),p,d)
                    p=p+torch.from_numpy(d)
        for result in results:
            result.update(action_archive=str(destination),action_archive_sha256=sha(destination),
                shielded_archive_reconstruction_passed=True)
    return results


def summary(rows, mode):
    out = []
    for row in rows:
        roll = row['rollouts'][mode]
        base = row['baseline']; teacher = row['teacher_final']; final = roll['final']
        out.append(dict(student_rmse=base['mean_local_rmse']-final['mean_local_rmse'],
            teacher_rmse=base['mean_local_rmse']-teacher['mean_local_rmse'],
            student_mse=base['local_mse']-final['local_mse'],teacher_mse=base['local_mse']-teacher['local_mse'],
            relative_mse=(base['local_mse']-final['local_mse'])/base['local_mse']))
    steps = [s for r in rows for s in r['rollouts'][mode]['steps']]
    return dict(identities=len({r['identity'] for r in rows}),trajectories=len(rows),
        RMSE_retention=sum(v['student_rmse'] for v in out)/sum(v['teacher_rmse'] for v in out),
        MSE_retention=sum(v['student_mse'] for v in out)/sum(v['teacher_mse'] for v in out),
        mean_relative_MSE_repair=float(np.mean([v['relative_mse'] for v in out])),
        safety_failed_steps=sum(not s['after']['safety']['passes'] for s in steps),
        new_inversions_steps=sum(s['after']['safety']['quartets']['new_inversions'] for s in steps),
        assessability_loss_steps=sum(s['after']['safety']['quartets']['assessability_lost'] for s in steps),
        cap_violations=sum(s['cap_violations'] for s in steps),eligibility_violations=sum(s['eligibility_violations'] for s in steps),
        exact_zero_rate=float(np.mean([s['exact_zero'] for s in steps])),near_zero_rate=float(np.mean([s['near_zero'] for s in steps])),
        intervention_rate=float(np.mean([s['shield']['intervention'] for s in steps])) if mode=='shielded' else 0.,
        rejection_rate=float(np.mean([s['shield']['rejected_nonzero'] for s in steps])) if mode=='shielded' else 0.,
        policy_seconds=frozen.old.statistics([s['policy_seconds'] for s in steps]),
        inference_seconds=frozen.old.statistics([s['inference_seconds'] for s in steps]))


def evaluate(path, identities, destination, quick=False):
    if destination.exists():
        result = json.loads(destination.read_text())
        assert result['model_sha256']==sha(path) and result['identities']==identities and result['quick']==quick
        return result
    rows_by_id = {r['sample_id']:r for r in json.loads((PARENT / 'selection.json').read_text())['identities']}
    started = time.perf_counter()
    archive_folder=destination.with_suffix('')
    if not quick:archive_folder.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG['eval_workers'],
        mp_context=multiprocessing.get_context('spawn'),initializer=init_evaluator,initargs=(str(path),)) as pool:
        rows = []
        for i,result in enumerate(pool.map(evaluate_identity,[(rows_by_id[sid],quick,str(archive_folder)) for sid in identities])):
            rows.extend(result)
            if (i+1)%50==0:
                print(f'EVAL {destination.name} {i+1}/{len(identities)}',flush=True)
    result = dict(identities=identities,quick=quick,model_sha256=sha(path),rows=rows,
        summary={m:summary(rows,m) for m in (['shielded'] if quick else ['raw','shielded'])},wall_seconds=time.perf_counter()-started)
    pilot.atomic_json(destination,result)
    return result


def train(arm):
    initialize(); dirs(); split = selection(); dataset()
    assert torch.cuda.is_available()
    torch.manual_seed(CONFIG['seed']); torch.cuda.manual_seed_all(CONFIG['seed'])
    rng = np.random.default_rng(CONFIG['seed']); device = torch.device('cuda')
    model = Student().to(device); optimizer = torch.optim.AdamW(model.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
    train_ids = split['splits']['train']
    train_probe = sorted(train_ids,key=lambda sid:rank(f'trainprobe|{sid}'))[:60]
    monitor = split['monitor']; history=[]; best=-float('inf'); stale=0; best_update=None
    start=time.perf_counter(); gpu_seconds=0.; arm_config=CONFIG['arms'][arm]
    for update in range(1,CONFIG['maximum_updates']+1):
        selected = rng.choice(train_ids,size=CONFIG['batch_size'],replace=True)
        samples = [sample(DATA[sid],int(rng.integers(3)),int(rng.integers(16))) for sid in selected]
        p,m,label,target = batch(samples,device)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); begin=time.perf_counter()
        imitation, energy, delta = action_loss(model,p,m,label)
        auxiliary = (local_mse(p+delta,target,m)/local_mse(p,target,m).detach().clamp_min(1e-12)).mean() if arm_config['local'] else p.new_tensor(0.)
        loss = imitation + arm_config['local']*auxiliary
        loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(),CONFIG['clip_norm'])
        assert torch.isfinite(norm); optimizer.step(); torch.cuda.synchronize()
        gpu_seconds += time.perf_counter()-begin
        if update%100==0:
            print(f'TRAIN {arm} {update} imitation={float(imitation):.6f} auxiliary={float(auxiliary):.6f}',flush=True)
        if update%CONFIG['validation_every']==0:
            path = ROOT / 'models' / f'{arm}_{update:05d}.pt'
            save_model(path,model,arm=arm,update=update)
            model.eval()
            train_loss=probe(model,train_probe,device); val_loss=probe(model,monitor,device)
            evaluation=evaluate(path,monitor,ROOT/'evaluation'/f'{arm}_{update:05d}_monitor.json',quick=True)
            retention=evaluation['summary']['shielded']['RMSE_retention']
            entry=dict(update=update,validation_retention=retention,train_probe=train_loss,validation_probe=val_loss,
                model_sha256=sha(path),validation_sha256=sha(ROOT/'evaluation'/f'{arm}_{update:05d}_monitor.json'))
            history.append(entry)
            if retention > best+CONFIG['retention_min_delta']:
                best=retention; best_update=update; stale=0
                save_model(ROOT/'models'/f'{arm}_best.pt',model,arm=arm,update=update)
            else:
                stale+=1
            pilot.atomic_json(ROOT/f'{arm}_training_progress.json',dict(history=history,best_update=best_update))
            print(f'VALIDATION {arm} {update} retention={retention:.6f} best={best:.6f} stale={stale}',flush=True)
            if update>=CONFIG['minimum_updates'] and stale>=CONFIG['patience_checks']:
                break
            model.train()
    pilot.atomic_json(ROOT/f'{arm}_training.json',dict(arm=arm,history=history,best_update=best_update,
        updates=update,wall_seconds=time.perf_counter()-start,GPU_training_seconds=gpu_seconds,
        parameters=sum(p.numel() for p in model.parameters()),CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        split_sha256=sha(ROOT/'splits.json'),source_sha256=sha(__file__),early_stopped=update<CONFIG['maximum_updates']))


def test():
    initialize();dirs();torch.manual_seed(CONFIG['seed'])
    rows=json.loads((PARENT/'selection.json').read_text())['identities']
    item=load_identity(rows[0]);p,m,label,target=batch([sample(item,0,0)],'cpu')
    model=Student().eval()
    nn.init.normal_(model.local.head.weight,0,.001)
    first=model(p,m)
    q,_=torch.linalg.qr(torch.tensor([[.2,.3,.8],[.6,.9,.1],[.4,.1,.7]],dtype=torch.float64))
    if torch.det(q)<0:q[:,0]*=-1
    rotated=model(p@q+torch.tensor([5.,2.,-3.]),m)
    assert torch.allclose(rotated,first@q,atol=1e-7,rtol=1e-5)
    assert first.norm(dim=-1).max()<=.1+1e-14
    # Forward uses exactly two P/mask arguments; two arbitrary targets change losses only.
    changed_target=target+torch.randn_like(target)
    assert torch.equal(first,model(p,m))
    assert local_mse(p,changed_target,m).mean()!=local_mse(p,target,m).mean()
    raw,_=predict(model,p,m)
    b={k:torch.from_numpy(item['conditions'][0][v][None]) for k,v in [('pg','P0'),('source','source'),('target','target'),('mask','mask')]}
    ctx=pilot.Context(b);safe,cert=shield(ctx,p,raw)
    frozen.old.checks(dict(ctx=ctx,b=b),p,safe)
    repeat,_=shield(ctx,p,raw);assert np.array_equal(repeat,safe)
    pilot.atomic_json(ROOT/'tests.json',dict(passed=True,SO3_and_translation_equivariance=True,
        target_not_in_forward=True,exact_cap_and_eligibility=True,deterministic_frozen_shield=True,
        teacher_forbidden=True,SLSQP_forbidden=True,source_sha256=sha(__file__)))


def diagnose():
    """Same-capacity memorization probe; never replaces a selected pilot model."""
    initialize();dataset();split=selection();device=torch.device('cuda')
    chosen=json.loads((ROOT/'selected_model.json').read_text())['arm']
    metadata=json.loads((ROOT/f'{chosen}_training.json').read_text())
    best=next(r for r in metadata['history'] if r['update']==metadata['best_update'])
    if best['train_probe']['error_to_zero_baseline']<.5:
        pilot.atomic_json(ROOT/'capacity_diagnostic.json',dict(run=False,
            reason='selected model explains more than half of training-probe action energy; no demonstrated severe training underfit',
            larger_model_trained=False))
        return
    rng=np.random.default_rng(CONFIG['seed']+1)
    selected=[]
    for length in STRATA:
        ids=[sid for sid in split['splits']['train'] if DATA[sid]['row']['stratum']==length]
        selected.extend(sorted(ids,key=lambda sid:rank(f'memorization|{sid}'))[:6])
    selected.extend(sorted(set(split['splits']['train'])-set(selected),key=lambda sid:rank(f'memorization|{sid}'))[:2])
    specification=[dict(identity=sid,condition_index=int(rng.integers(3)),step=int(rng.integers(16))) for sid in selected]
    samples=[sample(DATA[r['identity']],r['condition_index'],r['step']) for r in specification]
    model=Student().to(device)
    model.load_state_dict(torch.load(ROOT/'models'/f'{chosen}_best.pt',map_location='cpu',weights_only=False)['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
    history=[];start=time.perf_counter()
    def measure(update):
        with torch.no_grad():
            p,m,d,_=batch(samples,device)
            error,energy,_=action_loss(model,p,m,d)
        history.append(dict(update=update,error=float(error),zero_baseline=float(energy),error_to_zero_baseline=float(error/energy)))
    measure(0)
    for update in range(1,1001):
        p,m,d,_=batch([samples[i] for i in rng.integers(len(samples),size=8)],device)
        optimizer.zero_grad(set_to_none=True);error,_,_=action_loss(model,p,m,d)
        error.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if update%250==0:
            measure(update);print('MEMORIZATION '+json.dumps(history[-1]),flush=True)
    save_model(ROOT/'models'/'same_capacity_memorization_only.pt',model,never_selected=True)
    pilot.atomic_json(ROOT/'capacity_diagnostic.json',dict(run=True,history=history,states=specification,
        wall_seconds=time.perf_counter()-start,selected=False,larger_model_trained=False,
        explanation='training-only fixed-state memorization tests optimization/capacity, not held-out generalization or observability'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','test','train','evaluate','diagnose'])
    parser.add_argument('--arm',choices=list(CONFIG['arms']));args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='test':test()
    elif args.mode=='train':
        assert json.loads((ROOT/'tests.json').read_text())['passed'];train(args.arm)
    elif args.mode=='diagnose':diagnose()
    else:
        split=selection()
        for arm in CONFIG['arms']:
            path=ROOT/'models'/f'{arm}_best.pt'
            evaluate(path,split['splits']['validation'],ROOT/'evaluation'/f'{arm}_validation.json')
        chosen=max(CONFIG['arms'],key=lambda arm:json.loads((ROOT/'evaluation'/f'{arm}_validation.json').read_text())['summary']['shielded']['RMSE_retention'])
        pilot.atomic_json(ROOT/'selected_model.json',dict(arm=chosen,selection='full validation shielded rollout RMSE retention',test_used=False))
        evaluate(ROOT/'models'/f'{chosen}_best.pt',split['splits']['test'],ROOT/'evaluation'/f'{chosen}_test.json')


if __name__=='__main__':main()
