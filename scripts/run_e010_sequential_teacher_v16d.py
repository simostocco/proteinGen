"""Register, preflight, generate and independently reproduce privileged teacher states."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import platform
import resource
import subprocess
import time
from pathlib import Path
import numpy as np
import scipy
import torch
import yaml
from protein_distance_diffusion.training import e010_sequential_teacher_v16c as teacher
from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins, file_hash
from protein_distance_diffusion.training.e010_recurrent_capacity import batch
from scripts.run_e010_phase4d_recurrent_capacity_v3 import ROOT, load_cache, write
from scripts.recover_e010_conditioning_v9b import synthetic
from scripts.run_e010_slsqp_v14 import available_ram
from protein_distance_diffusion.training.e010_slsqp_v14 import required_arrays

OUT=ROOT/'reports/experiments/E010_global_equivariant_expressivity/phase4d_hybrid_local_global_v1/sequential_teacher_v16d'
CONFIG=ROOT/'configs/e010_phase4d_sequential_teacher_v16d.yaml'
ADDED=['src/protein_distance_diffusion/training/e010_sequential_teacher_v16c.py',
       'scripts/run_e010_sequential_teacher_v16d.py','scripts/report_e010_sequential_teacher_v16d.py',
       'configs/e010_phase4d_sequential_teacher_v16d.yaml','docs/e010_phase4d_sequential_teacher_v16d.md',
       'tests/test_e010_sequential_teacher_v16d.py']


def initialize():
    torch.set_num_threads(1)
    assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))


def cache_data(cache,i):
    return {k:v.double() if v.is_floating_point() else v for k,v in batch(cache,[i],torch.device('cpu')).items()}


def register():
    initialize()
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='e010-phase4d-hybrid-local-global'
    config=yaml.safe_load(CONFIG.read_text())
    cache,manifest=load_cache()
    files=set(subprocess.check_output(['git','ls-files'],text=True).splitlines()+ADDED)
    pins={str(ROOT/f):file_hash(ROOT/f) for f in files}
    rows=[]
    for i in range(60):
        b=cache_data(cache,i)
        c=teacher.Context(b)
        assert c.safety(b['pg'])['passes']
        rows.append(dict(index=i,record=cache['records'][i],input_sha256={k:teacher.digest(v.numpy()) for k,v in b.items()},
                         quartets=c.quartets.counts,baseline_feasible=True))
    assert sum(r['quartets']['assessable'] for r in rows)==13029
    write(OUT/'execution_contract.json',dict(config=config,protected_sha256=pins,
        protected_input_sha256=manifest['protected_input_sha256'],cache_sha256=manifest['cache_sha256'],
        panel=rows,python=platform.python_version(),scipy=scipy.__version__,cpu_affinity=len(os.sched_getaffinity(0)),
        available_ram_bytes=available_ram(),neural_training=False,cuda=False,
        settings=teacher.SETTINGS,materiality=teacher.MATERIALITY))
    print('registered 60 feasible baselines',flush=True)


def frozen():
    initialize()
    contract=json.loads((OUT/'execution_contract.json').read_text())
    assert yaml.safe_load(CONFIG.read_text())==contract['config']
    assert_file_pins(contract['protected_sha256'])
    assert_file_pins(contract['protected_input_sha256'])
    return contract


def distribution(a, eligible):
    norms=np.linalg.norm(a,axis=-1)[eligible]
    return dict(rms=float(np.sqrt(np.mean(norms**2))),maximum=float(norms.max()),
                fractions={str(v):float((norms>v).mean()) for v in (.04,.06,.08,.095)})


def one_preflight(n):
    initialize()
    b=synthetic(n)
    validation=teacher.validate(b)
    c=teacher.Context(b); step=teacher.OneStep(c,b['pg'])
    required=required_arrays(step.n,len(step.cfun(np.zeros(step.n)))+step.n//3)
    started=time.perf_counter()
    raw=OUT/'untracked_states'/'preflight'/f'length_{n}.npz'
    raw.parent.mkdir(parents=True,exist_ok=True)
    assert not raw.exists()
    def save(a):
        np.savez(raw,delta=a,prediction=b['pg'].numpy()+a,pg=b['pg'].numpy(),target=b['target'].numpy(),mask=b['mask'].numpy())
    a,log=teacher.solve_step(c,b['pg'],save)
    safety=c.safety(b['pg']+torch.from_numpy(a))
    assert safety['passes'] and log['accepted_normalized_local']<=log['start_normalized_local']
    return dict(length=n,validation=validation,solver=log,safety=safety,action_sha256=teacher.digest(a),
                correction_distribution=distribution(a,step.eligible.numpy()),state_file_sha256=file_hash(raw),
                state_file=str(raw.relative_to(ROOT)),required_dense_arrays=required,wall_seconds=time.perf_counter()-started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


def calibration_job(_):
    initialize()
    b=synthetic(32); a,log=teacher.solve_step(teacher.Context(b),b['pg'])
    return dict(action_sha256=teacher.digest(a),iterations=log['iterations'],peak_rss_bytes=log['peak_rss_bytes'])


def preflight():
    contract=frozen()
    # All fixed derivative preflights pass before any optimizer invocation.
    derivatives=[dict(length=n,validation=teacher.validate(synthetic(n))) for n in (12,32,500)]
    write(OUT/'derivative_validation.json',dict(primary_h=1e-4,checks=derivatives,tolerances_unchanged=True))
    write(OUT/'preflight_attempt.json',dict(lengths=[12,32,500],scientific_panel=False))
    records=[]
    for n in (12,32,500):
        # Isolated process makes each peak-RSS measurement independent.
        with concurrent.futures.ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
            row=pool.submit(one_preflight,n).result(timeout=contract['config']['resource_wall_cap_seconds'])
        write(OUT/'preflight'/f'length_{n}.json',row)
        records.append(row)
        print('preflight',n,row['solver']['iterations'],row['solver']['success'],row['peak_rss_bytes'],flush=True)
        assert row['peak_rss_bytes']<contract['config']['resource_peak_budget_gib']*2**30
    cal=[]
    memory=available_ram()
    for workers in contract['config']['calibration_workers']:
        if workers>len(os.sched_getaffinity(0)) or workers*records[-1]['peak_rss_bytes']>.7*memory:
            continue
        start=time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            jobs=list(pool.map(calibration_job,range(contract['config']['calibration_jobs'])))
        assert len({(r['action_sha256'],r['iterations']) for r in jobs})==1
        assert jobs[0]['action_sha256']==records[1]['action_sha256']
        row=dict(workers=workers,wall_seconds=time.perf_counter()-start,results=jobs)
        cal.append(row); write(OUT/'preflight'/f'calibration_{workers}.json',row)
        print('calibration',workers,row['wall_seconds'],flush=True)
    chosen=min(cal,key=lambda r:(r['wall_seconds'],r['workers']))['workers']
    write(OUT/'preflight_complete.json',dict(panel_permitted=True,records=records,
          calibration=cal,workers=chosen,worker_memory_budget_bytes=records[-1]['peak_rss_bytes']))
    print('frozen workers',chosen,flush=True)


def metric_rows(b,record,states,actions,eligibility):
    context=teacher.Context(b)
    tr=dict(steps=[dict(delta=torch.from_numpy(a),eligible=torch.from_numpy(e)) for a,e in zip(actions,eligibility)])
    rows=[]
    for t,p in enumerate(states):
        p=torch.from_numpy(p)
        row=teacher.metrics.metric_row(p,b,record,tr,t)
        row.update(step=t,safety=context.safety(p),coordinate_sha256=teacher.digest(p.numpy()))
        if t:
            norms=np.linalg.norm(actions[t-1],axis=-1)[b['mask'].numpy()]
            elig=eligibility[t-1][b['mask'].numpy()]
            row.update(action_sha256=teacher.digest(actions[t-1]),
                       exact_stored_step_max=float(norms.max()),
                       correction_fractions={str(s):float((norms[elig]>s).mean()) if elig.any() else 0.
                                             for s in (.04,.06,.08,.095)},
                       eligible_corrections=int(elig.sum()))
            if t>1:
                prev=actions[t-2].reshape(-1,3); curr=actions[t-1].reshape(-1,3)
                pn,cn=np.linalg.norm(prev,axis=1),np.linalg.norm(curr,axis=1)
                defined=(pn>1e-12)&(cn>1e-12)
                cs=np.sum(prev[defined]*curr[defined],axis=1)/(pn[defined]*cn[defined])
                row['consecutive_cosines']=np.clip(cs,-1,1).tolist()
        rows.append(row)
    return rows


def trajectory_job(payload):
    initialize()
    i,record,arrays=payload
    b={k:torch.from_numpy(v.copy()) for k,v in arrays.items()}
    context=teacher.Context(b)
    current=b['pg'].clone()
    states=[current.numpy().copy()]; actions=[]; eligible=[]; frames=[]; logs=[]
    start=time.perf_counter()
    private=OUT/'untracked_states'/f'example_{i:02d}'
    private.mkdir(parents=True,exist_ok=False)
    for t in range(teacher.K_MAX):
        step=teacher.OneStep(context,current)
        def save_action(a):
            np.save(private/f'action_{t:02d}.npy',a)
        a,log=teacher.solve_step(context,current,save_action)
        current=current+torch.from_numpy(a)
        assert context.safety(current)['passes']
        actions.append(a); eligible.append(step.eligible.numpy().copy()); frames.append(step.frame.numpy().copy())
        states.append(current.numpy().copy()); logs.append(log)
        np.save(private/f'state_{t+1:02d}.npy',states[-1])
        print('teacher',i,'step',t+1,'normalized_local',log['accepted_normalized_local'],flush=True)
    arrays=dict(states=np.stack(states),delta=np.stack(actions),eligibility=np.stack(eligible),frames=np.stack(frames),
                mask=b['mask'].numpy(),target=b['target'].numpy())
    arrays['local_actions']=np.einsum('tbnji,tbnj->tbni',arrays['frames'],arrays['delta'])
    path=private/'trajectory.npz'
    np.savez(path,**arrays)
    # Raw complete trajectory exists before optional metric aggregation.
    rows=metric_rows(b,record,states,actions,eligible)
    row=dict(index=i,record=record,states=rows,solver_steps=logs,trajectory_file=str(path.relative_to(ROOT)),
             trajectory_sha256=file_hash(path),array_sha256={k:teacher.digest(v) for k,v in arrays.items()},
             runtime_seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(OUT/'examples'/f'example_{i:02d}.json',row)
    return dict(index=i,complete=True,runtime_seconds=row['runtime_seconds'],peak_rss_bytes=row['peak_rss_bytes'])


def run():
    contract=frozen(); pre=json.loads((OUT/'preflight_complete.json').read_text())
    assert pre['panel_permitted']
    write(OUT/'panel_attempt.json',dict(examples=60,steps=16,workers=pre['workers'],zero_initialization=True,once_only=True))
    cache,manifest=load_cache(); payloads=[]
    for i in range(60):
        b=cache_data(cache,i)
        arrays={k:v.numpy() for k,v in b.items()}
        assert {k:teacher.digest(v) for k,v in arrays.items()}==contract['panel'][i]['input_sha256']
        payloads.append((i,cache['records'][i],arrays))
    started=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=pre['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        rows=list(pool.map(trajectory_job,payloads))
    assert_file_pins(contract['protected_sha256'])
    assert_file_pins(contract['protected_input_sha256'])
    write(OUT/'run_complete.json',dict(examples=rows,wall_seconds=time.perf_counter()-started,workers=pre['workers']))


def reproduce():
    contract=frozen(); cache,_=load_cache(); transitions=[]
    for i in range(60):
        b=cache_data(cache,i); record=cache['records'][i]
        row=json.loads((OUT/'examples'/f'example_{i:02d}.json').read_text())
        path=ROOT/row['trajectory_file']
        assert file_hash(path)==row['trajectory_sha256']
        with np.load(path) as saved:
            arrays={k:saved[k].copy() for k in saved.files}
        assert {k:teacher.digest(v) for k,v in arrays.items()}==row['array_sha256']
        assert np.array_equal(arrays['states'][0],b['pg'].numpy())
        assert np.array_equal(arrays['target'],b['target'].numpy())
        current=b['pg'].clone(); c=teacher.Context(b)
        for t in range(16):
            step=teacher.OneStep(c,current); a=arrays['delta'][t]
            assert np.array_equal(step.eligible.numpy(),arrays['eligibility'][t])
            assert np.array_equal(step.frame.numpy(),arrays['frames'][t])
            assert np.array_equal(teacher.project(a)[0],a)
            assert (a[~arrays['eligibility'][t]]==0).all()
            u=np.einsum('bnji,bnj->bni',arrays['frames'][t],a)
            assert np.array_equal(u,arrays['local_actions'][t])
            assert np.allclose(np.einsum('bnij,bnj->bni',arrays['frames'][t],u),a,atol=1e-14,rtol=1e-12)
            old=float(c.values(current)['local']); current=current+torch.from_numpy(a)
            assert np.array_equal(current.numpy(),arrays['states'][t+1])
            assert float(c.values(current)['local'])<=old and c.safety(current)['passes']
            transitions.append(dict(trajectory_id=i,sample_id=record['sample_id'],condition=record['condition'],step=t,
                local_action_sha256=teacher.digest(arrays['local_actions'][t]),source_trajectory_sha256=row['trajectory_sha256'],
                state_sha256=teacher.digest(arrays['states'][t]),action_sha256=teacher.digest(a),
                eligibility_sha256=teacher.digest(arrays['eligibility'][t]),eligible=int(arrays['eligibility'][t].sum()),
                zero_action=bool((a==0).all()),teacher_safety=True,trajectory_file=row['trajectory_file']))
        recomputed=metric_rows(b,record,arrays['states'],arrays['delta'],arrays['eligibility'])
        assert json.loads(json.dumps(recomputed))==row['states']
        write(OUT/'reproduction'/f'example_{i:02d}.json',dict(index=i,exact=True,optimization_invocations=0))
        print('reproduced',i,flush=True)
    assert_file_pins(contract['protected_sha256'])
    write(OUT/'reproduction_complete.json',dict(examples=60,exact=True,optimizer_invocations=0))
    # Publish only after report establishes TEACH-A/B/C.
    write(OUT/'transition_candidates.json',dict(slots=960,transitions=transitions,authorized_for_distillation=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['register','preflight','run','reproduce'])
    globals()[p.parse_args().mode]()
