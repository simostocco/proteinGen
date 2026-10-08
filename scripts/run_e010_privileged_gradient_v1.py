"""Frozen CPU-float64 privileged gradient benchmark; no optimization solver or NN."""
import os
from pathlib import Path

BASE = Path('/mnt/d/Simone/proteinGen_data/e010_phase4d')
ROOT = BASE/'fast_privileged_gradient_teacher_v1'
PILOT = BASE/'fast_privileged_teacher_pilot_v1'
for key, value in {'TMPDIR':ROOT/'tmp','TMP':ROOT/'tmp','TEMP':ROOT/'tmp',
                   'XDG_CACHE_HOME':ROOT/'cache','MPLCONFIGDIR':ROOT/'cache/matplotlib',
                   'CUDA_CACHE_PATH':ROOT/'cache/cuda'}.items(): os.environ[key]=str(value)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(PILOT/'code'))
import argparse
import concurrent.futures
import json
import subprocess
import tempfile
import time
import numpy as np
import torch
import pilot

GRID = [.01,.025,.05,.075,.10]
METHODS = [f'normalized_{m:.3f}' for m in GRID]+['best_grid','projected_2','projected_4']
CONFIG = dict(experiment='fast_privileged_gradient_teacher_v1',dtype='CPU float64',
              magnitudes_angstrom=GRID,methods=METHODS,normalization='per eligible residue; exact zero gradient gives zero direction',
              gradient='autograd of the exact frozen Context.values(P)[local], unnormalized',
              grid_selection='lowest frozen local MSE; exact tie keeps incumbent; ascending magnitude order',
              refinement='recompute gradient at every accepted internal state; add normalized descent; project TOTAL delta about teacher-input origin to 0.10 A',
              refinement_eligibility='increment on intersection of original and current eligibility; total action retains original eligibility',
              shield='unchanged pilot.shield; baseline Context anchored to original corruption P0; shield each cumulative proposal relative to teacher-input origin',
              scales=pilot.CONFIG['shield_scales'],step_action_cap_angstrom=.10,
              rollout_horizon=16,matched_rollout_horizon='exact durable gold prefix length',
              benchmark='all 278 gold states, all 20 trajectory baselines; no tuning or training',
              safety='frozen aligned RMSD 1.01 baseline, continuous chirality baseline, exact quartet feasibility, no new inversions, no assessability loss',
              refinement_acceptance='never increase internal local objective; retain last feasible incumbent or initial exact zero',
              latency='perf_counter CPU wall time, includes gradient/mask/projection/shield/selection; excludes reporting; refinement includes all internal searches',
              shared_computation='fixed proposals and best-grid share first gradient; PG2/PG4 share prefix, reported timings include complete equivalent inference path',
              workers=4,no_new_gold_solves=True,no_neural_training=True,no_full_corpus=True,no_final_student=True)


def initialize():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64);torch.use_deterministic_algorithms(True)
    import scipy.optimize as optimize
    def forbidden(*a,**k): raise RuntimeError('Optimization solvers are forbidden in gradient teacher')
    optimize.minimize=forbidden
    for module in list(sys.modules.values()):
        if getattr(module,'__name__','').startswith('protein_distance_diffusion'):
            for name in ('minimize','solve_step'):
                if hasattr(module,name):setattr(module,name,forbidden)


def atomic_npz(path, **arrays):
    if path.exists():
        with np.load(path,allow_pickle=False) as existing:
            assert set(existing.files)==set(arrays)
            assert all(np.array_equal(existing[k],v) for k,v in arrays.items()),path
        return
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.tmp')
    with os.fdopen(fd,'wb') as stream:
        np.savez(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    os.rename(tmp,path);pilot.fsync_dir(path.parent)


def direction(item, current, original_eligible):
    p=current.detach().clone().requires_grad_(True)
    with pilot.float64_defaults():
        value=item['ctx'].values(p)['local']
        gradient,=torch.autograd.grad(value,p)
    assert gradient.dtype==torch.float64 and torch.isfinite(gradient).all()
    eligible=pilot.local_representation(current,item['b']['mask'])['eligible'].numpy() & original_eligible
    g=gradient.detach().numpy();norm=np.linalg.norm(g,axis=-1,keepdims=True)
    descent=np.divide(-g,norm,out=np.zeros_like(g),where=norm>0)*eligible[...,None]
    return descent,g


def all_methods(item, origin):
    start=time.perf_counter()
    eligible=pilot.local_representation(origin,item['b']['mask'])['eligible'].numpy()
    descent,gradient=direction(item,origin,eligible)
    setup=time.perf_counter()-start
    proposals=[]
    for magnitude in GRID:
        begin=time.perf_counter()
        action,certificate=pilot.shield(item['ctx'],origin,descent*magnitude)
        proposals.append((action,certificate,time.perf_counter()-begin))
    result={METHODS[i]:dict(action=a,certificate=c,seconds=setup+secs,gradient_evaluations=1) for i,(a,c,secs) in enumerate(proposals)}
    incumbent=np.zeros_like(gradient)
    best=float(item['ctx'].values(origin)['local'])
    winner=None
    for index,(a,c,_) in enumerate(proposals):
        if c['objective_after']<best: incumbent=a;best=c['objective_after'];winner=index
    if winner is None:
        _,cert=pilot.shield(item['ctx'],origin,incumbent)
    else:cert=proposals[winner][1]
    cumulative=time.perf_counter()-start
    result['best_grid']=dict(action=incumbent.copy(),certificate=cert,seconds=cumulative,gradient_evaluations=1)
    for internal in range(2,5):
        begin=time.perf_counter()
        current=origin+torch.from_numpy(incumbent)
        descent,_=direction(item,current,eligible)
        for magnitude in GRID:
            candidate,_=pilot.project(incumbent+descent*magnitude)
            action,certificate=pilot.shield(item['ctx'],origin,candidate)
            if certificate['objective_after']<best:
                next_action=action;next_cert=certificate;best=certificate['objective_after']
            else:continue
            # All candidates in this substep use the same current gradient and incumbent.
            if 'selected' not in locals() or certificate['objective_after']<selected[0]:
                selected=(certificate['objective_after'],next_action,next_cert)
        if 'selected' in locals():
            _,incumbent,cert=selected;del selected
        cumulative+=time.perf_counter()-begin
        if internal in (2,4):
            result[f'projected_{internal}']=dict(action=incumbent.copy(),certificate=cert,seconds=cumulative,gradient_evaluations=internal)
    return result,gradient


def comparison(action,gold,eligible):
    a=action[eligible].reshape(-1);b=gold[eligible].reshape(-1)
    norms=np.linalg.norm(a)*np.linalg.norm(b)
    residue_a=np.linalg.norm(action,axis=-1);residue_b=np.linalg.norm(gold,axis=-1)
    valid=eligible&(residue_a>1e-8)&(residue_b>1e-8)
    return dict(action_cosine=float(np.dot(a,b)/norms) if norms>0 else None,
                action_rms_difference=float(np.sqrt(np.mean(np.sum((action-gold)[eligible]**2,axis=-1)))),
                residue_cosine_mean=float(np.mean(np.sum(action*gold,axis=-1)[valid]/(residue_a[valid]*residue_b[valid]))) if valid.any() else None,
                nonzero_cosine_residues=int(valid.sum()))


def checks(item,origin,action):
    eligible=pilot.local_representation(origin,item['b']['mask'])['eligible'].numpy()
    norm=np.linalg.norm(action,axis=-1)
    after=pilot.scalar_metrics(item['ctx'],origin+torch.from_numpy(action))
    assert after['safety']['passes']
    assert (action[~eligible]==0).all() and norm.max()<=.1
    assert float(item['ctx'].values(origin+torch.from_numpy(action))['local'])<=float(item['ctx'].values(origin)['local'])
    return dict(after=after,action_max=float(norm.max()),action_cap_violations=int((norm>.1).sum()),
                eligibility_violations=int(np.any(action!=0,axis=-1)[~eligible].sum()),
                new_inversions=after['safety']['quartets']['new_inversions'],
                assessability_loss=after['safety']['quartets']['assessability_lost'])


def worker(index):
    initialize();data,frozen=pilot.load();item=data[index];arrays=item['arrays'];record=item['row']['record'];n=len(arrays['delta'])
    destination=ROOT/'trajectories'/f'trajectory_{index:02d}.json'
    if destination.exists():
        validate_trajectory(index,data);return
    common=dict(index=index,identity=record['sample_id'],condition=record['condition'],length=int(item['b']['mask'].sum()),length_bin=record['stratum'])
    rows=[];diagnostics=[];saved={m:[] for m in METHODS};gradients=[]
    for step in range(n):
        origin=torch.from_numpy(arrays['states'][step]);before=pilot.scalar_metrics(item['ctx'],origin)
        gold_after=pilot.scalar_metrics(item['ctx'],torch.from_numpy(arrays['states'][step+1]));gold=arrays['delta'][step];eligible=arrays['eligibility'][step]
        methods,gradient=all_methods(item,origin);gradients.append(gradient)
        diag=comparison(gold,-gradient,eligible)
        diagnostics.append(dict(common,step=step,gold_negative_gradient_cosine=diag['action_cosine'],gold_negative_gradient_residue_cosine=diag['residue_cosine_mean'],gradient_norm=float(np.linalg.norm(gradient[eligible])),gold_action_norm=float(np.linalg.norm(gold[eligible])),gold_exact_zero=bool((gold==0).all())))
        for method,output in methods.items():
            a=output['action'];saved[method].append(a);checked=checks(item,origin,a)
            gold_repair=before['mean_local_rmse']-gold_after['mean_local_rmse'];repair=before['mean_local_rmse']-checked['after']['mean_local_rmse']
            rows.append(dict(common,step=step,method=method,before=before,gold_after=gold_after,
                             gold_repair=gold_repair,repair=repair,repair_retention=repair/gold_repair if gold_repair>1e-12 else None,
                             local_mse_gain=before['local_mse']-checked['after']['local_mse'],
                             latency_seconds=output['seconds'],gradient_evaluations=output['gradient_evaluations'],
                             shield_scale=output['certificate']['scale'],shield_checks=output['certificate']['checks'],
                             exact_zero=bool((a==0).all()),**comparison(a,gold,eligible),**checked))
        print(f'GOLD_STATE {index} {step+1}/{n}',flush=True)
    rollouts=[];roll_arrays={}
    baseline=pilot.scalar_metrics(item['ctx'],item['b']['pg'])
    gold_final=pilot.scalar_metrics(item['ctx'],torch.from_numpy(arrays['states'][-1]))
    gold_repair=baseline['mean_local_rmse']-gold_final['mean_local_rmse']
    for method in METHODS:
        states=[arrays['pg'].copy()];actions=[];step_rows=[]
        for step in range(16):
            origin=torch.from_numpy(states[-1]);before=pilot.scalar_metrics(item['ctx'],origin)
            # Run just this inference path; all_methods shares candidates only within one call.
            output=one_method(item,origin,method)
            a=output['action'];checked=checks(item,origin,a)
            states.append(states[-1]+a);actions.append(a)
            step_rows.append(dict(common,method=method,step=step,before=before,latency_seconds=output['seconds'],
                                  shield_scale=output['certificate']['scale'],exact_zero=bool((a==0).all()),**checked))
        matched=step_rows[n-1]['after'];final=step_rows[-1]['after'];repair=baseline['mean_local_rmse']-matched['mean_local_rmse']
        rollouts.append(dict(common,method=method,matched_gold_horizon=n,baseline=baseline,gold_final=gold_final,
                             matched_final=matched,P16=final,gold_repair=gold_repair,repair=repair,
                             repair_retention=repair/gold_repair if gold_repair>1e-12 else None,steps=step_rows))
        roll_arrays[f'{method}_states']=np.stack(states);roll_arrays[f'{method}_delta']=np.stack(actions)
        print(f'ROLLOUT {index} {method}',flush=True)
    archive=destination.with_suffix('.npz')
    atomic_npz(archive,gradient=np.stack(gradients),**{f'{m}_one_step_delta':np.stack(saved[m]) for m in METHODS},**roll_arrays)
    pilot.atomic_json(destination,dict(config_sha256=pilot.sha(ROOT/'protocol.json'),gold_manifest_sha256=frozen['manifest_sha256'],
                                      archive_sha256=pilot.sha(archive),one_step=rows,rollouts=rollouts,gradient_diagnostic=diagnostics))
    validate_trajectory(index,data)


def one_method(item,origin,method):
    """Identical inference rules, skipping unused paths for rollout timing."""
    start=time.perf_counter();eligible=pilot.local_representation(origin,item['b']['mask'])['eligible'].numpy()
    descent,_=direction(item,origin,eligible)
    if method.startswith('normalized_'):
        action,cert=pilot.shield(item['ctx'],origin,descent*float(method.split('_')[1]))
        return dict(action=action,certificate=cert,seconds=time.perf_counter()-start)
    iterations={'best_grid':1,'projected_2':2,'projected_4':4}[method]
    incumbent=np.zeros_like(origin.numpy());best=float(item['ctx'].values(origin)['local']);_,cert=pilot.shield(item['ctx'],origin,incumbent)
    for substep in range(iterations):
        if substep:descent,_=direction(item,origin+torch.from_numpy(incumbent),eligible)
        candidates=[]
        for magnitude in GRID:
            proposal,_=pilot.project(incumbent+descent*magnitude)
            a,c=pilot.shield(item['ctx'],origin,proposal)
            candidates.append((a,c))
        for a,c in candidates:
            if c['objective_after']<best:incumbent=a;cert=c;best=c['objective_after']
    return dict(action=incumbent,certificate=cert,seconds=time.perf_counter()-start)


def validate_trajectory(index,data):
    path=ROOT/'trajectories'/f'trajectory_{index:02d}.json';result=json.loads(path.read_text());item=data[index]
    assert result['config_sha256']==pilot.sha(ROOT/'protocol.json')
    assert result['archive_sha256']==pilot.sha(path.with_suffix('.npz'))
    assert result['gold_manifest_sha256']==pilot.sha(PILOT/'gold/transitions.json')
    n=len(item['arrays']['delta'])
    assert len(result['one_step'])==n*len(METHODS) and len(result['rollouts'])==len(METHODS)
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as archive:
        assert all(np.isfinite(archive[k]).all() for k in archive.files)
        for m in METHODS:
            for t,a in enumerate(archive[f'{m}_one_step_delta']):checks(item,torch.from_numpy(item['arrays']['states'][t]),a)
            states=archive[f'{m}_states'];delta=archive[f'{m}_delta']
            assert len(states)==17 and np.array_equal(states[0],item['arrays']['pg'])
            for t,a in enumerate(delta):
                assert np.array_equal(states[t]+a,states[t+1]);checks(item,torch.from_numpy(states[t]),a)
    return result


def statistics(values):
    a=np.asarray([x for x in values if x is not None],dtype=float)
    return dict(count=len(a),min=float(a.min()),p10=float(np.quantile(a,.1)),median=float(np.median(a)),mean=float(a.mean()),p90=float(np.quantile(a,.9)),max=float(a.max())) if len(a) else dict(count=0)


def aggregate(rows,gold_lookup=None):
    gold=sum(r['gold_repair'] for r in rows);fast=sum(r['repair'] for r in rows)
    lat=[r['latency_seconds'] for r in rows] if rows and 'latency_seconds' in rows[0] else [s['latency_seconds'] for r in rows for s in r['steps']]
    matched=[]
    if gold_lookup:
        for r in rows:
            source=gold_lookup[(r['index'],r['step'])]
            seconds=(source.get('solver_metadata') or {}).get('runtime_seconds') or (source.get('timing') or {}).get('solver_wall_seconds')
            if seconds:matched.append((seconds,r['latency_seconds']))
    states=rows if rows and 'after' in rows[0] else [s for r in rows for s in r['steps']]
    return dict(count=len(rows),local_repair_sum=fast,gold_local_repair_sum=gold,repair_retention=fast/gold if gold>1e-12 else None,
                cosine=statistics([r.get('action_cosine') for r in rows]),action_rms_difference=statistics([r.get('action_rms_difference') for r in rows]),
                aligned_rmsd=statistics([r['after']['aligned_rmsd'] for r in states]),
                max_aligned_rmsd_ratio=max(r['after']['aligned_rmsd']/r['before']['aligned_rmsd'] for r in states),
                new_inversions=sum(r['new_inversions'] for r in states),assessability_loss=sum(r['assessability_loss'] for r in states),
                action_cap_violations=sum(r['action_cap_violations'] for r in states),eligibility_violations=sum(r['eligibility_violations'] for r in states),
                all_safety_passed=all(r['after']['safety']['passes'] for r in states),
                zero_action_fraction=float(np.mean([r['exact_zero'] for r in states])),latency_seconds=statistics(lat),
                archived_matching_steps=len(matched),speedup=sum(v[0] for v in matched)/sum(v[1] for v in matched) if matched else None)


def finish(data):
    outputs=[validate_trajectory(i,data) for i in sorted(data)]
    one=sum([v['one_step'] for v in outputs],[]);roll=sum([v['rollouts'] for v in outputs],[]);diag=sum([v['gradient_diagnostic'] for v in outputs],[])
    assert len(diag)==278 and len(one)==278*len(METHODS) and len(roll)==20*len(METHODS)
    manifest=json.loads((PILOT/'gold/transitions.json').read_text())['transitions'];lookup={(r['trajectory_index'],r['step']):r for r in manifest}
    summary={m:dict(one_step=aggregate([r for r in one if r['method']==m],lookup),rollout=aggregate([r for r in roll if r['method']==m])) for m in METHODS}
    grouped={}
    for dimension in ('identity','condition','length_bin','length','step'):
        grouped[dimension]={}
        for key in sorted({str(r[dimension]) for r in one}):
            grouped[dimension][key]={m:aggregate([r for r in one if str(r[dimension])==key and r['method']==m],lookup) for m in METHODS}
    rollout_groups={dimension:{key:{m:aggregate([r for r in roll if str(r[dimension])==key and r['method']==m]) for m in METHODS} for key in sorted({str(r[dimension]) for r in roll})} for dimension in ('identity','condition','length_bin','length')}
    cosines=[r['gold_negative_gradient_cosine'] for r in diag if r['gold_negative_gradient_cosine'] is not None]
    diagnostic=dict(cosine_distribution=statistics(cosines),positive_cosine_fraction=float(np.mean(np.array(cosines)>0)),
                    assessed=len(cosines),excluded_zero_or_degenerate=278-len(cosines),gold_exact_zero_actions=sum(r['gold_exact_zero'] for r in diag),
                    residue_cosine_distribution=statistics([r['gold_negative_gradient_residue_cosine'] for r in diag]),
                    by_identity={identity:statistics([r['gold_negative_gradient_cosine'] for r in diag if r['identity']==identity]) for identity in sorted({r['identity'] for r in diag})})
    pilot.atomic_json(ROOT/'results.json',dict(summary=summary,by=grouped,rollout_by=rollout_groups,gradient_diagnostic=diagnostic,
                                              gradient_diagnostic_rows=diag,retention_aggregation='sum absolute mean-local-RMSE repair / matching gold sum',no_new_gold_solves=True))
    write_report(summary,diagnostic,grouped)
    files={str(p):pilot.sha(p) for p in ROOT.rglob('*') if p.is_file() and '/tmp/' not in str(p) and '/cache/' not in str(p) and '/logs/' not in str(p) and p.name!='COMPLETED.json'}
    pilot.atomic_json(ROOT/'COMPLETED.json',dict(status='complete',files=files,gold_states=278,trajectories=20,methods=METHODS,no_new_gold_solves=True))


def write_report(summary,diagnostic,grouped):
    lines=['# Fast privileged gradient teacher v1\n',
           'Frozen benchmark: 15 complete + 5 partial trajectories, 278 accepted gold transitions, 7 identities. '
           'All methods evaluated on all gold states and on 16-step rollouts from all 20 original corruptions. '
           'Rollout retention uses exactly the available gold horizon (including partial prefixes). No parameters fitted, no neural training, no new solver calls.\n',
           '| Method | One-step retention % | Rollout retention % | Median latency ms | Archived speedup | Action cosine | RMS difference Å |\n|---|---:|---:|---:|---:|---:|---:|']
    for m,v in summary.items():
        a=v['one_step'];r=v['rollout'];cos=a['cosine'].get('mean')
        lines.append(f"| {m} | {100*a['repair_retention']:.3f} | {100*r['repair_retention']:.3f} | {1000*a['latency_seconds']['median']:.3f} | {a['speedup']:.1f}x | {cos if cos is not None else 'N/A'} | {a['action_rms_difference']['mean']:.6f} |")
    lines+=['\nGold action versus **unnormalized** negative local gradient, concatenated over eligible residues: '+json.dumps(diagnostic,indent=2)+'\n',
            'Cosine near +1 supports steepest descent; positive cosine alone establishes descent alignment, not the absence of constraint-induced/preconditioned changes. Zero gold actions are excluded from cosine statistics and counted explicitly. '
            'Action cosine versus each fast method likewise excludes zero-vector pairs; this can be selective under shielding.\n',
            'Safety: the historical CPU float64 shield is called unchanged. Aligned RMSD <=1.01 original-corruption baseline and continuous chirality nonworsening use the frozen numerical tolerance; quartet constraints are exact. '
            'Each cumulative refined correction is projected to the exact NumPy 0.10 Å closed ball. Every stored action/state was independently rechecked for gates, cap and eligibility.\n',
            'Latency is CPU wall time including gradient and shield, excluding metric reporting. Four single-threaded workers ran concurrently. '
            'Speedup is sum archived matched per-step SLSQP runtime / sum corresponding inference latency; no solver benchmarks were rerun. '
            'All variants and grids were frozen before measurement. Aggregate retention is a ratio of repair sums, not the mean of per-case ratios.\n',
            'Detailed metrics by identity, condition, exact length, length bin and structural step are in results.json. '
            'Per-state certificates, aligned RMSD, inversion/assessability counts, actions and rollout arrays are in trajectories/. '
            'protocol.json and provenance.json pin the method and gold/source lineage; COMPLETED.json pins results.\n']
    best=max(summary,key=lambda m:summary[m]['rollout']['repair_retention']);value=summary[best]['rollout']['repair_retention']
    lines.append(f'Best matched rollout method: {best}, {100*value:.3f}% gold repair retention; residual gap to 80%: {max(0,80-100*value):.3f} percentage points. '
                 'No additional complexity, full structural corpus or final student was launched.\n')
    path=ROOT/'REPORT.md';fd,tmp=tempfile.mkstemp(dir=ROOT,suffix='.tmp')
    with os.fdopen(fd,'w') as stream:stream.write('\n'.join(lines));stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path);pilot.fsync_dir(ROOT)


def launch_one(index):
    with (ROOT/'logs'/f'trajectory_{index:02d}.log').open('a',buffering=1) as log:
        result=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',str(index)],stdout=log,stderr=log,cwd='/mnt/d/Simone/proteinGen',check=False)
    if result.returncode:raise RuntimeError(f'Trajectory {index} failed; see D: log')
    print(f'TRAJECTORY_COMPLETE {index}',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);parser.add_argument('--test',action='store_true');args=parser.parse_args()
    if args.worker is not None:worker(args.worker);return
    initialize()
    for path in (ROOT,ROOT/'tmp',ROOT/'cache',ROOT/'logs',ROOT/'trajectories'):path.mkdir(exist_ok=True)
    negative=json.loads((PILOT/'FAST_PT_V1_FROZEN_NEGATIVE.json').read_text())
    for path,digest in negative['files'].items():assert pilot.sha(path)==digest,path
    data,frozen=pilot.load()
    for path,digest in frozen['source_sha256'].items():assert pilot.sha(path)==digest,path
    protocol=ROOT/'protocol.json'
    if protocol.exists():assert json.loads(protocol.read_text())==CONFIG
    else:pilot.atomic_json(protocol,CONFIG)
    provenance=ROOT/'provenance.json'
    if not provenance.exists():
        contract=json.loads((BASE/'v16d_acceleration_audit_20261007/inputs/execution_contract.json').read_text())
        sources={str(Path(__file__).resolve()):pilot.sha(__file__),str(PILOT/'code/pilot.py'):pilot.sha(PILOT/'code/pilot.py')}
        for name,module in list(sys.modules.items()):
            path=getattr(module,'__file__',None)
            if name.startswith('protein_distance_diffusion') and path:
                assert contract['protected_sha256'][path]==pilot.sha(path),path
                sources[path]=pilot.sha(path)
        pilot.atomic_json(provenance,dict(gold_manifest_sha256=frozen['manifest_sha256'],FAST_PT_v1_negative_freeze_sha256=pilot.sha(PILOT/'FAST_PT_V1_FROZEN_NEGATIVE.json'),source_sha256=sources,python=sys.version,torch=torch.__version__,cpu_model=Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0],config_sha256=pilot.sha(protocol)))
    else:
        for path,digest in json.loads(provenance.read_text())['source_sha256'].items():assert pilot.sha(path)==digest,path
    test(data)
    if args.test:return
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG['workers']) as pool:
        for completed in concurrent.futures.as_completed([pool.submit(launch_one,index) for index in sorted(data)]):completed.result()
    finish(data)
    print('GRADIENT_TEACHER_COMPLETE',flush=True)


def test(data):
    item=data[0];origin=item['b']['pg'];eligible=pilot.local_representation(origin,item['b']['mask'])['eligible'].numpy()
    descent,g=direction(item,origin,eligible)
    # Independent central directional derivative of exact frozen local objective.
    direction_tensor=torch.from_numpy(descent);epsilon=1e-5
    finite_difference=float((item['ctx'].values(origin+epsilon*direction_tensor)['local']-item['ctx'].values(origin-epsilon*direction_tensor)['local'])/(2*epsilon))
    analytic=float((torch.from_numpy(g)*direction_tensor).sum())
    assert np.isclose(finite_difference,analytic,rtol=1e-6,atol=1e-8)
    outputs,_=all_methods(item,origin)
    for method in METHODS:
        independent=one_method(item,origin,method)
        assert np.array_equal(outputs[method]['action'],independent['action']),method
        checks(item,origin,independent['action'])
    assert outputs['projected_4']['certificate']['objective_after']<=outputs['projected_2']['certificate']['objective_after']<=outputs['best_grid']['certificate']['objective_after']
    pilot.atomic_json(ROOT/'tests.json',dict(passed=True,finite_difference=finite_difference,analytic=analytic,
                                            identical_shared_and_independent_paths=True,safety_and_cap=True,solver_calls_forbidden=True))


if __name__=='__main__':main()
