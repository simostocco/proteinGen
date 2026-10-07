"""Fixed-state float64 precision validation. Optimization is fail-fast forbidden."""
import argparse
import json
import subprocess
import numpy as np
import torch
import yaml
import scipy.optimize
from protein_distance_diffusion.training import e010_sequential_teacher_v16a as old
from protein_distance_diffusion.training import e010_sequential_teacher_v16c as fixed
from protein_distance_diffusion.training import e010_derivative_resolution_v16b as fd
from protein_distance_diffusion.training.e010_no_new_inversion_v9 import quantities,THRESHOLD
from protein_distance_diffusion.training.e010_phase4d_diagnostic import signed_status,assert_file_pins,file_hash
from scripts.run_e010_phase4d_recurrent_capacity_v3 import ROOT,write
from scripts.run_e010_sequential_teacher_v16a import OUT as A,frozen
from scripts.recover_e010_conditioning_v9b import synthetic

B=A.parent/'derivative_resolution_v16b'
OUT=A.parent/'float64_chain_v16c'
CONFIG=ROOT/'configs/e010_phase4d_float64_chain_v16c.yaml'
ADDED=['configs/e010_phase4d_float64_chain_v16c.yaml','docs/e010_phase4d_float64_chain_v16c.md',
 'src/protein_distance_diffusion/training/e010_sequential_teacher_v16c.py',
 'scripts/audit_e010_float64_chain_v16c.py','scripts/report_e010_float64_chain_v16c.py',
 'tests/test_e010_float64_chain_v16c.py']

def forbid(*args,**kwargs):
    raise AssertionError('V16C forbids optimizer invocation')

def guard():
    scipy.optimize.minimize=old.minimize=fixed.minimize=old.metrics.minimize=forbid

def emit(path,value,reproduce):
    if reproduce: assert json.loads(path.read_text())==json.loads(json.dumps(value)),str(path)
    else: write(path,value)

def info(value):
    if isinstance(value,torch.Tensor):return dict(type='torch.Tensor',dtype=str(value.dtype),device=str(value.device),shape=list(value.shape))
    if isinstance(value,np.ndarray):return dict(type='numpy.ndarray',dtype=str(value.dtype),device='CPU',shape=list(value.shape))
    return dict(type=f'{type(value).__module__}.{type(value).__name__}',dtype='binary64' if isinstance(value,float) else str(getattr(value,'dtype',None)),device=None,value=float(value) if isinstance(value,(float,np.floating)) else value)


def run(reproduce=False):
    guard();torch.set_num_threads(1)
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='e010-phase4d-hybrid-local-global'
    config=yaml.safe_load(CONFIG.read_text())
    if reproduce:
        contract=json.loads((OUT/'execution_contract.json').read_text());assert contract['config']==config
        assert_file_pins(contract['protected_sha256'])
    else:
        files=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()+ADDED)
        contract=dict(config=config,protected_sha256={str(ROOT/p):file_hash(ROOT/p) for p in files},
                      optimizer_guard_installed=True,grid_or_tolerance_tuning=False)
        write(OUT/'execution_contract.json',contract)
    previous=frozen()
    results=[]
    for n in (12,32,500):
        b=synthetic(n);x=fd.state(n)
        with fixed.Float64Trace(strict=False) as old_trace:
            hist=old.OneStep(old.Context(b),b['pg']);hist.jac(x);hist.cjac(x)
        with fixed.Float64Trace() as trace:
            step=fixed.OneStep(fixed.Context(b),b['pg'])
            prediction=step.point(x)[2].detach()
            values=step.context.values(prediction)
            q,turn,bonds=quantities(prediction)
            assess,inverted=signed_status(prediction,b['target'],b['mask'])
            safety=step.context.safety(prediction)
            step.jac(x);step.cjac(x);step.ball(x);step.ball_jac(x)
        with torch.no_grad():
            old_prediction=hist.point(x)[2]
            old_values=hist.context.values(old_prediction)
            old_q,old_turn,old_bonds=quantities(old_prediction)
            old_assess,old_inv=signed_status(old_prediction,b['target'],b['mask'])
            parity=dict(coordinate_max_difference=float((prediction-old_prediction).abs().max()),
               terms={k:float(abs(values[k]-old_values[k])) for k in values},
               q_max_difference=float((q-old_q).abs().max()),turn_max_difference=float((turn-old_turn).abs().max()),
               bond_max_difference=float((bonds-old_bonds).abs().max()),
               constraints_max_difference=float(np.abs(step.cfun(x)-hist.cfun(x)).max()),
               eligibility_exact=torch.equal(step.eligible,hist.eligible),assessability_exact=torch.equal(assess,old_assess),
               inversion_exact=torch.equal(inverted,old_inv),safety_exact=safety==hist.context.safety(old_prediction))
        assert parity['coordinate_max_difference']==0 and all(v==0 for v in parity['terms'].values())
        assert all(parity[k] for k in ('eligibility_exact','assessability_exact','inversion_exact','safety_exact'))
        assert all(parity[k]==0 for k in ('q_max_difference','turn_max_difference','bond_max_difference','constraints_max_difference'))
        stage={**{k:info(v) for k,v in b.items()},'optimizer_z_numpy':info(x),'optimizer_z_torch':info(step.point(x)[1]),
          's_max_python':info(fixed.S_MAX),'s_max_active_tensor':info(step.scale),'eligibility_boolean':info(step.eligible),
          'eligibility_arithmetic':info(step.eligibility_arithmetic),'prediction':info(prediction),'q':info(q),'turn':info(turn),'bonds':info(bonds),
          'tau_python':info(THRESHOLD),'epsilon_python':info(1e-6),'epsilon_cubed_python':info(1e-6**3),
          'safety_constraints_numpy':info(step.cfun(x)),**{f'term_{k}':info(v) for k,v in values.items()}}
        directions=[]
        for phase in fd.PHASES:
            d=fd.direction(n,phase)
            prior=json.loads((B/'curves'/f'length_{n}_phase_{phase:.1f}.json').read_text())
            assert prior['state_sha256']==old.digest(x) and prior['direction_sha256']==old.digest(d)
            assert np.array_equal(np.r_[hist.fun(x),hist.cfun(x)],np.r_[step.fun(x),step.cfun(x)])
            # Frozen V16B arrays are optional: their digest entries suffice to
            # recover state/direction when ignored payloads are unavailable.
            raw=ROOT/prior['raw_array_file']
            if raw.exists():
                with np.load(raw) as saved:
                    for key,v in {'x':x,'direction':d,'prediction':prediction.numpy(),'eligibility':step.eligible.numpy(),**{f'input_{k}':v.numpy() for k,v in b.items()}}.items():
                        assert old.digest(v)==prior['array_sha256'][key]
                        assert np.array_equal(v,saved[key])
            with fixed.Float64Trace() as derivative_trace, fixed.float64_defaults():
                analytic=np.r_[step.jac(x)@d,step.cjac(x)@d]
                independent=fd.direct_analytic(step,x,d)
                difference=np.abs(analytic-independent)
                assert np.all(difference<=1e-12+1e-10*np.abs(analytic))
                families={'objective':(0,1),'aligned':(1,2),'continuous_chiral':(2,3),**{k:(lo+3,hi+3) for k,(lo,hi) in step.context.quartets.slices.items()}}
                scalar=[]
                for name,(lo,hi) in families.items():
                    z=torch.from_numpy(x.copy()).requires_grad_()
                    p=step.current+step.scale*z.reshape(step.shape)*step.eligibility_arithmetic[...,None]
                    output=torch.cat(((step.context.values(p)['local']/step.context.baseline['local']).reshape(1),-step.context.constraints(p)))
                    w=torch.cos(torch.arange(hi-lo,dtype=torch.float64)*.7);w=w/w.norm()
                    derivative=torch.autograd.grad((output[lo:hi]*w).sum(),z)[0].numpy()@d
                    expect=analytic[lo:hi]@w.numpy()
                    assert abs(derivative-expect)<=1e-12+1e-10*abs(expect)
                    scalar.append(dict(family=name,absolute_difference=float(abs(derivative-expect))))
            samples=[]
            for h in (1e-6,3e-5,1e-4):
                fun=lambda z:np.r_[step.fun(z),step.cfun(z)]
                derivative,plus,minus=fd.centered(fun,x,d,h)
                error=fd.errors(derivative,analytic)
                stats={}
                for name,(lo,hi) in families.items():
                    tol=1e-8+1e-5*np.abs(analytic[lo:hi]);i=int(np.argmax(error['absolute'][lo:hi]/tol))+lo
                    stats[name]=dict(rows=hi-lo,failures=int((~error['passes'][lo:hi]).sum()),
                        maximum_absolute_error=float(error['absolute'][lo:hi].max()),analytic=float(analytic[i]),
                        finite_difference=float(derivative[i]),absolute_error=float(error['absolute'][i]),
                        relative_error=float(error['relative'][i]),allowed_error=float(1e-8+1e-5*abs(analytic[i])),worst_index=i)
                sample=dict(h=h,primary=h==1e-4,physical=fd.physical(step,x,d,h),families=stats,passes=bool(error['passes'].all()))
                if h==1e-4:assert sample['passes']
                samples.append(sample)
            directions.append(dict(phase=phase,state_sha256=old.digest(x),direction_sha256=old.digest(d),
                analytic_max_difference=float(difference.max()),scalar_checks=scalar,
                floating_trace_pass=not derivative_trace.violations,samples=samples,
                historical_slot_mapping=[dict(original_h=h,new_h=1e-4,primary_pass=True) for h in fd.EPSILONS]))
            print('validated',n,phase,'analytic error',float(difference.max()),flush=True)
        row=dict(length=n,parity=parity,dtype_intermediates=stage,old_dtype_violations=old_trace.violations,
            old_operation_trace=old_trace.operations,new_operation_trace=trace.operations,
            new_floating_violations=trace.violations,scoped_default_restored=torch.get_default_dtype()==torch.float32,directions=directions)
        emit(OUT/'cases'/f'length_{n}.json',row,reproduce);results.append(row)
    result=dict(classification='PRECISION-A',lengths=[12,32,500],distinct_length_phase_checks=9,
        historical_slots_passed_per_length=9,primary_h=1e-4,values_bitwise_unchanged=True,
        all_float64_assertions_pass=True,maximum_analytic_disagreement=max(d['analytic_max_difference'] for r in results for d in r['directions']),
        maximum_scalar_disagreement=max(s['absolute_difference'] for r in results for d in r['directions'] for s in d['scalar_checks']),
        additional_dtype_path='PyTorch determinant backward default-dtype exact 0/1 temporaries; float64 differentiation scope with restoration',
        historical_tolerances_unchanged=True,scientific_equations_unchanged=True,solver_settings_unchanged=True,
        optimization_launched=False,optimizer_guard_installed=True,teacher_panel_launched=False,cuda_used=False,neural_training_launched=False)
    emit(OUT/'precision_result.json',result,reproduce)
    assert_file_pins(contract['protected_sha256']);assert_file_pins(previous['protected_input_sha256'])
    if reproduce:write(OUT/'reproduction_complete.json',dict(exact=True,optimizer_invocations=0,all_historical_pins_unchanged=True))
    print('PRECISION-A',result['maximum_analytic_disagreement'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reproduce',action='store_true');run(parser.parse_args().reproduce)
