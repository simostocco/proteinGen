"""Pure numerical audit on immutable synthetic check states; never optimize."""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
import yaml
from protein_distance_diffusion.training import e010_derivative_resolution_v16b as fd
from protein_distance_diffusion.training.e010_phase4d_diagnostic import file_hash, assert_file_pins
from scripts.run_e010_sequential_teacher_v16a import OUT as OLD, frozen
from scripts.run_e010_phase4d_recurrent_capacity_v3 import ROOT, write
from scripts.recover_e010_conditioning_v9b import synthetic

OUT=OLD.parent/'derivative_resolution_v16b'
CONFIG=ROOT/'configs/e010_phase4d_derivative_resolution_v16b.yaml'
ADDED=['configs/e010_phase4d_derivative_resolution_v16b.yaml','docs/e010_phase4d_derivative_resolution_v16b.md',
       'src/protein_distance_diffusion/training/e010_derivative_resolution_v16b.py',
       'scripts/audit_e010_derivative_resolution_v16b.py','tests/test_e010_derivative_resolution_v16b.py']


def emit(path,value,reproduce):
    canonical=json.loads(json.dumps(value))
    if reproduce:
        assert json.loads(path.read_text())==canonical,str(path)
    else:
        write(path,value)


def audit(reproduce=False):
    torch.set_num_threads(1)
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='e010-phase4d-hybrid-local-global'
    config=yaml.safe_load(CONFIG.read_text())
    assert tuple(config['epsilon_grid'])==fd.GRID
    if reproduce:
        registration=json.loads((OUT/'analysis_continuation_contract.json').read_text())
        assert config==registration['config']; assert_file_pins(registration['protected_sha256'])
    else:
        pins={str(ROOT/p):file_hash(ROOT/p) for p in set(subprocess.check_output(['git','ls-files'],text=True,cwd=ROOT).splitlines()+ADDED)}
        registration=dict(config=config,protected_sha256=pins,grid_frozen_before_execution=True,
                          optimizer_invocations=0,quality_metrics_used=False)
        registration.update(reason='Record failed analytic agreement without suppressing it; continue fixed-state resolution analysis only',
                            original_registration_sha256=file_hash(OUT/'execution_contract.json'),
                            tolerances_or_grid_changed=False,scientific_implementation_changed=False)
        write(OUT/'analysis_continuation_contract.json',registration)
    historical_contract=frozen()
    old_failure=json.loads((OLD/'preflight/length_500_derivative_failure.json').read_text())
    recovered=[]; curves=[]; old_checks=[]
    for n in (12,32,500):
        b=synthetic(n); context=fd.historical.Context(b); step=fd.historical.OneStep(context,b['pg'])
        x=fd.state(n)
        p=step.point(x)[2].detach().numpy()
        recovery=dict(length=n,inputs_sha256={k:fd.historical.digest(v.numpy()) for k,v in b.items()},
             state_sha256=fd.historical.digest(x),prediction_sha256=fd.historical.digest(p),
             eligibility_sha256=fd.historical.digest(step.eligible.numpy()),variable_count=step.n,
             baseline_exact_feasible=context.safety(b['pg'])['passes'],
             current_coordinate_abs_max=float(np.abs(p).max()),historical_state_hash_available=False,
             historical_geometry_implementation_exact=True)
        # V16A did not serialize the derivative-state coordinate hash. Generated
        # state is verified through exact reproduction of every saved comparison.
        reconstructed=[]
        for phase in fd.PHASES:
            d=fd.direction(n,phase)
            row,arrays=fd.audit_direction(step,x,d,phase)
            arrays.update({f'input_{k}':v.numpy() for k,v in b.items()})
            arrays.update(prediction=p,eligibility=step.eligible.numpy())
            name=f'length_{n}_phase_{phase:.1f}'
            raw=OUT/'untracked_arrays'/f'{name}.npz'
            digests={k:fd.historical.digest(v) for k,v in arrays.items()}
            if reproduce:
                with np.load(raw) as saved:
                    assert {k:fd.historical.digest(saved[k]) for k in saved.files}==digests
            else:
                raw.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(raw,**arrays)
            row.update(length=n,array_sha256=digests,raw_array_file=str(raw.relative_to(ROOT)))
            emit(OUT/'curves'/f'{name}.json',row,reproduce)
            curves.append(row); old_checks+=row['historical_checks'] if n==500 else []
            for check in row['historical_checks']:
                families={}
                for family,stats in check['vector_constraint_families'].items():
                    lo,hi=row['families'][family]
                    families[family]=dict(rows=hi-lo,failures=stats['failures'],maximum_absolute_error=stats['maximum_absolute_error'],
                         worst_tolerance_ratio=stats['maximum_tolerance_ratio'],worst_constraint_index=stats['worst_index']-1,
                         analytic_derivative=stats['analytic'],finite_difference=stats['fd'],allowed_error=stats['tolerance'])
                reconstructed.append(dict(phase=phase,epsilon=check['historical_h'],objective_error=check['scalar_objective']['absolute_error'],
                     objective_pass=check['scalar_objective']['failures']==0,constraint_failures=check['vector_failures'],families=families))
            print('audited',n,phase,'stable',row['stable_grid_values'],flush=True)
        if n==500:
            assert reconstructed==old_failure['fixed_checks'],'V16A exact recovery mismatch'
        else:
            old=json.loads((OLD/'preflight'/f'length_{n}.json').read_text())['validation']['directional_checks']
            minimal=[dict(phase=r['phase'],epsilon=r['epsilon'],objective_error=r['objective_error'],
                          constraint_max_error=max(v['maximum_absolute_error'] for v in r['families'].values())) for r in reconstructed]
            assert minimal==old,'V16A small control recovery mismatch'
        recovery['all_available_historical_derivative_evidence_exact']=True
        recovered.append(recovery)
    common=sorted(set.intersection(*(set(c['stable_grid_values']) for c in curves)))
    analytic_agrees=all(c['analytic_paths_pass'] and all(v['passes'] for v in c['scalar_analytic_checks'].values()) for c in curves)
    classification=('FD-A' if analytic_agrees else 'FD-D') if common else ('FD-C' if analytic_agrees else 'FD-B')
    result=dict(classification=classification,exact_historical_recovery=True,recovery=recovered,
         historical_length_500_checks=old_checks,
         historical_failed_checks=[dict(phase=r['phase'],h=r['historical_h'],failures=r['vector_failures']) for r in old_checks if not r['passes_all']],
         common_stable_grid_values=common,
         analytic_paths_all_pass=analytic_agrees,
         chain_bias_factor=float(np.float32(.1))/.1,
         chain_factor_model_max_error=max(c['chain_factor_model_max_error'] for c in curves),
         analytic_max_disagreement=max(c['analytic_max_disagreement'] for c in curves),
         scalar_analytic_max_disagreement=max(s['absolute_difference'] for c in curves for s in c['scalar_analytic_checks'].values()),
         by_length={str(n):[dict(phase=c['phase'],stable_grid_values=c['stable_grid_values'],proposed_future_h=c['proposed_future_h']) for c in curves if c['length']==n] for n in (12,32,500)},
         epsilon_grid=list(fd.GRID),absolute_tolerance=fd.ATOL,relative_tolerance=fd.RTOL,
         proposed_future_rule=config['proposed_rule'],historical_tolerance_changed=False,
         optimization_launched=False,scientific_panel_launched=False,cuda_used=False,neural_training_launched=False,
         teacher_quality_metrics_used=False,scientific_interpretation=False)
    emit(OUT/'audit_result.json',result,reproduce)
    assert_file_pins(registration['protected_sha256'])
    assert_file_pins(historical_contract['protected_input_sha256'])
    if reproduce:
        write(OUT/'reproduction_complete.json',dict(exact=True,curves=9,optimizer_invocations=0,all_original_pins_unchanged=True))
    print(classification,'common stable window',common,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--reproduce',action='store_true')
    audit(parser.parse_args().reproduce)
