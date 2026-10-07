import json
import numpy as np
import torch
from protein_distance_diffusion.training import e010_sequential_teacher_v16a as teacher
from scripts.recover_e010_conditioning_v9b import synthetic
from scripts.run_e010_sequential_teacher_v16a import OUT, frozen
from scripts.run_e010_phase4d_recurrent_capacity_v3 import write
from protein_distance_diffusion.training.e010_slsqp_v14 import required_arrays
from protein_distance_diffusion.training.e010_phase4d_diagnostic import file_hash

# This analysis-only reproducer performs no optimizer invocation.
original_write = write
def write(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(value)), str(path)
    else:
        original_write(path, value)

torch.set_num_threads(1)
frozen()
b=synthetic(500); context=teacher.Context(b); step=teacher.OneStep(context,b['pg'])
x=np.sin(np.arange(step.n)*.7)*.03
families={'aligned':(0,1),'continuous_chiral':(1,2),**{k:(lo+2,hi+2) for k,(lo,hi) in context.quartets.slices.items()}}
checks=[]
for phase in (0.,.7,1.3):
    d=np.cos(np.arange(step.n)+phase); d/=np.linalg.norm(d)
    derivative=step.cjac(x)@d
    gf=float(step.jac(x)@d)
    for eps in (1e-6,1e-5,1e-4):
        fd=(step.cfun(x+eps*d)-step.cfun(x-eps*d))/(2*eps)
        fdf=(step.fun(x+eps*d)-step.fun(x-eps*d))/(2*eps)
        e=np.abs(fd-derivative); tolerance=1e-8+1e-5*np.abs(derivative)
        families_rows={}
        for key,(lo,hi) in families.items():
            fail=e[lo:hi]>tolerance[lo:hi]
            worst=int(np.argmax(e[lo:hi]/tolerance[lo:hi]))+lo
            families_rows[key]=dict(rows=hi-lo,failures=int(fail.sum()),maximum_absolute_error=float(e[lo:hi].max()),
                worst_tolerance_ratio=float((e[lo:hi]/tolerance[lo:hi]).max()),worst_constraint_index=worst,
                analytic_derivative=float(derivative[worst]),finite_difference=float(fd[worst]),
                allowed_error=float(tolerance[worst]))
        checks.append(dict(phase=phase,epsilon=eps,objective_error=abs(fdf-gf),
                           objective_pass=abs(fdf-gf)<=1e-8+1e-5*abs(gf),
                           constraint_failures=int((e>tolerance).sum()),families=families_rows))
rows=[json.loads((OUT/'preflight'/f'length_{n}.json').read_text()) for n in (12,32)]
write(OUT/'preflight'/'length_500_derivative_failure.json',dict(length=500,variable_count=1500,
    required_dense_arrays=required_arrays(step.n,len(step.cfun(np.zeros(step.n)))+500),
    baseline_feasible=context.safety(b['pg'])['passes'],fixed_checks=checks,
    optimizer_invocations=0,resource_smoke_completed=False,solver_settings_changed=False,tolerances_changed=False))
result=dict(classification='TEACH-E',reason='length-500 analytic constraint Jacobian finite-difference preflight did not pass frozen tolerances',
    panel_launched=False,teacher_frozen_for_distillation=False,trajectories_complete=0,transition_slots=0,
    teacher_solver='SciPy SLSQP',scipy='1.18.1',settings=teacher.SETTINGS,parallel_workers=None,
    K_max=16,s_max_angstrom=.1,materiality=teacher.MATERIALITY,zero_action_count=None,
    small_preflights=[dict(length=r['length'],iterations=r['solver']['iterations'],success=r['solver']['success'],
       safe=r['safety']['passes'],action_sha256=r['action_sha256'],normalized_local=r['solver']['accepted_normalized_local'],
       runtime_seconds=r['wall_seconds'],peak_rss_bytes=r['peak_rss_bytes']) for r in rows],
    length_500_derivative_checks=len(checks),length_500_failed_checks=sum(c['constraint_failures']>0 for c in checks),
    maximum_constraint_tolerance_ratio=max(v['worst_tolerance_ratio'] for c in checks for v in c['families'].values()),
    maximum_inward_guard_angstrom=max(r['solver']['maximum_inward_guard_angstrom'] for r in rows),
    neural_training_launched=False,cuda_used=False,no_package_changes=True,
    scientific_condition_gains=None,headroom_use=None,scientific_safety=None,
    tests='42 focused CPU tests passed; scientific derivative preflight failed at length 500',
    recommended_next_experiment='fixed-state float64 finite-difference resolution audit of the length-500 one-step constraint Jacobian, preserving the recorded tolerances and solver settings')
write(OUT/'teacher_result.json',result)
write(OUT/'validation_record.json',dict(cpu_tests_passed=42,
    test_log_sha256=file_hash(OUT/'cpu_tests.log'),
    supplemental_acceptance_tests_sha256=file_hash('tests/test_e010_sequential_teacher_v16a_acceptance.py'),
    optimizer_invocations_scientific_panel=0,preflight_optimizer_invocations=2,
    length_500_optimizer_invocations=0,historical_artifact_pins_verified=True,protected_inputs_verified=True))
print(json.dumps(result,indent=2))
