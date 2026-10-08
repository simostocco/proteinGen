"""Recover the D-resident, fixed-protocol E010 pilot without gold solves."""
import os
from pathlib import Path

ROOT = Path('/mnt/d/Simone/proteinGen_data/e010_phase4d/fast_privileged_teacher_pilot_v1')
for key, value in {
    'TMPDIR': ROOT / 'tmp', 'TMP': ROOT / 'tmp', 'TEMP': ROOT / 'tmp',
    'XDG_CACHE_HOME': ROOT / 'cache', 'MPLCONFIGDIR': ROOT / 'cache/matplotlib',
    'TORCHINDUCTOR_CACHE_DIR': ROOT / 'cache/torchinductor',
    'TORCH_EXTENSIONS_DIR': ROOT / 'cache/torch_extensions',
    'TRITON_CACHE_DIR': ROOT / 'cache/triton', 'NUMBA_CACHE_DIR': ROOT / 'cache/numba',
    'CUDA_CACHE_PATH': ROOT / 'cache/cuda',
}.items():
    os.environ[key] = str(value)
os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_LAUNCH_BLOCKING='1')
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'code'))
import json
import inspect
import time
import tempfile
import numpy as np
import torch
import pilot

RECOVERY = ROOT / 'recovery_20261008'


def atomic_torch(path, payload):
    """Never replace a durable checkpoint; publish a hash marker last."""
    assert not path.exists()
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix='.tmp')
    with os.fdopen(fd, 'wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(temporary, path)
    pilot.fsync_dir(path.parent)
    pilot.atomic_json(path.with_suffix('.valid.json'), {'sha256': pilot.sha(path)})


def save_training(fold, update, model, opt, rng, history, clipped):
    path = RECOVERY / f'fold_{fold}_update_{update:04d}.pt'
    if path.exists():
        if path.with_suffix('.valid.json').exists():
            raise RuntimeError(f'Refusing to replace {path}')
        # A checkpoint without its final commit marker is not reusable.
        path.rename(path.with_name(path.name+f'.uncommitted_{time.time_ns()}'))
    atomic_torch(path, dict(update=update, model=model.state_dict(), optimizer=opt.state_dict(),
                           rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                           cuda_rng=torch.cuda.get_rng_state_all(), history=history, clipped=clipped,
                           config=pilot.CONFIG, gold_manifest_sha256=pilot.sha(ROOT/'gold/transitions.json')))


def atomic_rollout(destination, states, delta):
    arrays = dict(states=np.stack(states), delta=np.stack(delta))
    if destination.exists():
        with np.load(destination, allow_pickle=False) as existing:
            assert all(np.array_equal(existing[key], value) for key, value in arrays.items()), destination
        return
    fd, tmp = tempfile.mkstemp(dir=destination.parent, suffix='.tmp')
    with os.fdopen(fd, 'wb') as stream:
        np.savez(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    os.rename(tmp, destination); pilot.fsync_dir(destination.parent)


def resumable_evaluate(model, data, held, device):
    source = inspect.getsource(pilot.evaluate_fold)
    source = source.replace('def evaluate_fold(', 'def recovered_evaluate_fold(', 1)
    needle = "  with destination.open('xb') as f:np.savez(f,states=np.stack(states),delta=np.stack(actions));f.flush();os.fsync(f.fileno())"
    assert needle in source
    source = source.replace(needle, '  atomic_rollout(destination,states,actions)')
    namespace = dict(vars(pilot), atomic_rollout=atomic_rollout)
    exec(compile(source, str(Path(__file__)), 'exec'), namespace)
    return namespace['recovered_evaluate_fold'](model, data, held, device)


def resumable_train(data, held, device, fold):
    # Retain the exact historical sampler, loss, optimizer and update count.
    source = inspect.getsource(pilot.train_fold)
    source = source.replace('def train_fold(', 'def recovered_train_fold(', 1)
    needle = " for update in range(CONFIG['updates_per_fold']):"
    assert needle in source
    source = source.replace(needle, """ resume_update=0
 candidates=sorted(RECOVERY.glob(f'fold_{fold}_update_*.valid.json'),reverse=True)
 for marker in candidates:
  path=marker.with_suffix('').with_suffix('.pt')
  try:
   assert sha(path)==json.loads(marker.read_text())['sha256']
   checkpoint=torch.load(path,map_location=device,weights_only=False)
   assert checkpoint['config']==CONFIG and checkpoint['gold_manifest_sha256']==sha(ROOT/'gold/transitions.json')
   assert all(torch.isfinite(v).all() for v in checkpoint['model'].values())
  except (AssertionError,OSError,ValueError,RuntimeError):continue
  model.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer'])
  rng.bit_generator.state=checkpoint['rng'];torch.set_rng_state(checkpoint['torch_rng'].cpu())
  torch.cuda.set_rng_state_all([v.cpu() for v in checkpoint['cuda_rng']])
  resume_update=checkpoint['update'];history=checkpoint['history'];clipped=checkpoint['clipped'];break
 for update in range(resume_update,CONFIG['updates_per_fold']):""")
    needle = " torch.cuda.synchronize();wall=time.perf_counter()-start"
    assert needle in source
    source = source.replace(needle, "  if (update+1)%100==0:save_training(fold,update+1,model,opt,rng,history,clipped)\n" + needle)
    needle = " with path.open('xb') as out:torch.save(dict(model=model.cpu().state_dict(),config=CONFIG,held_identity=held,training_identities=train_ids,gold_manifest_sha256=sha(ROOT/'gold/transitions.json')),out);out.flush();os.fsync(out.fileno())"
    assert needle in source
    source = source.replace(needle, " atomic_torch(path,dict(model=model.cpu().state_dict(),config=CONFIG,held_identity=held,training_identities=train_ids,gold_manifest_sha256=sha(ROOT/'gold/transitions.json')))")
    namespace = dict(vars(pilot), RECOVERY=RECOVERY, save_training=save_training, atomic_torch=atomic_torch)
    exec(compile(source, str(Path(__file__)), 'exec'), namespace)
    return namespace['recovered_train_fold'](data, held, device, fold)


def validate_result(result, data, held, checkpoint):
    assert result['training']['held_identity'] == held
    assert result['training']['model_sha256'] == pilot.sha(checkpoint)
    expected = {i for i, item in data.items() if item['row']['record']['sample_id'] == held}
    assert {r['index'] for r in result['heldout_rollouts']} == expected
    assert len(result['heldout_transitions']) == sum(len(data[i]['arrays']['delta']) for i in expected)
    for row in result['heldout_rollouts']:
        path = ROOT / f"heldout_rollout_{row['index']:02d}.npz"
        assert pilot.sha(path) == row['rollout_sha256']
        item = data[row['index']]
        with np.load(path, allow_pickle=False) as arrays:
            assert arrays['states'].shape[0] == 17 and arrays['delta'].shape[0] == 16
            assert np.isfinite(arrays['states']).all() and np.isfinite(arrays['delta']).all()
            assert np.array_equal(arrays['states'][0], item['arrays']['pg'])
            for step, delta in enumerate(arrays['delta']):
                assert np.array_equal(arrays['states'][step] + delta, arrays['states'][step+1])
                assert np.linalg.norm(delta, axis=-1).max() <= .1
                p = torch.from_numpy(arrays['states'][step+1])
                assert item['ctx'].safety(p)['passes']
                values = pilot.scalar_metrics(item['ctx'], p)
                assert abs(values['mean_local_rmse'] - row['states'][step+1]['mean_local_rmse']) < 1e-12


def main():
    for path in (RECOVERY, ROOT/'tmp'):
        path.mkdir(exist_ok=True)
    with (RECOVERY/'resume.log').open('a', buffering=1) as log:
        sys.stdout = log
        sys.stderr = log
        pilot.initialize()
        frozen = json.loads((ROOT/'gold/FROZEN.json').read_text())
        for group in ('files', 'source_sha256'):
            for path, digest in frozen[group].items():
                assert pilot.sha(path) == digest, path
        assert pilot.sha(pilot.ROOT.parent/'v16d_acceleration_audit_20261007/inputs/frozen_panel.npz') == frozen['cache_sha256']
        assert json.loads((ROOT/'protocol.json').read_text()) == pilot.CONFIG
        data, _ = pilot.load()
        for item in data.values():
            a = item['arrays']
            for step, delta in enumerate(a['delta']):
                assert np.array_equal(a['states'][step]+delta, a['states'][step+1])
                assert np.linalg.norm(delta, axis=-1).max() <= .1
                assert (delta[~a['eligibility'][step]] == 0).all()
            assert all(item['ctx'].safety(torch.from_numpy(p))['passes'] for p in a['states'])
        identities = sorted({v['row']['record']['sample_id'] for v in data.values()})
        recovered = []
        for fold, held in enumerate(identities):
            path = ROOT/'models'/f'loio_fold_{fold}.pt'
            if path.exists():
                c = torch.load(path, map_location='cpu', weights_only=False)
                assert c['config'] == pilot.CONFIG and c['held_identity'] == held
                assert c['training_identities'] == sorted(set(identities)-{held})
                assert c['gold_manifest_sha256'] == frozen['manifest_sha256']
                model = pilot.PrivilegedPolicy()
                model.load_state_dict(c['model'], strict=True)
                assert all(torch.isfinite(v).all() for v in c['model'].values())
                recovered.append(dict(fold=fold, identity=held, model_sha256=pilot.sha(path),
                                      optimizer_saved=False, fixed_final_updates=800))
            result_path = ROOT/f'fold_{fold}_results.json'
            if result_path.exists():
                validate_result(json.loads(result_path.read_text()), data, held, path)
        report = dict(frozen_hashes_verified=sum(len(frozen[k]) for k in ('files','source_sha256')),
                      gold_audit=json.loads((ROOT/'gold/audit.json').read_text()), recovered_models=recovered,
                      completed_folds=[i for i in range(7) if (ROOT/f'fold_{i}_results.json').exists()],
                      cuda_available=torch.cuda.is_available(), recovered_code_sha256=pilot.sha(ROOT/'code/pilot.py'),
                      recovery_code_sha256=pilot.sha(__file__), source_dependencies={name:pilot.sha(module.__file__) for name,module in list(sys.modules.items()) if name.startswith('protein_distance_diffusion') and getattr(module,'__file__',None)},
                      no_new_gold_solves=True)
        pilot.atomic_json(RECOVERY/'recovered_state.json', report)
        print('RECOVERY_VALIDATED '+json.dumps({k:v for k,v in report.items() if k not in ('gold_audit','source_dependencies')}), flush=True)
        assert torch.cuda.is_available(), 'GPU unavailable; validated recovery is durable, no training launched'
        import protein_distance_diffusion.training.e010_sequential_teacher_v16c as gold
        def forbidden(*args, **kwargs):
            raise RuntimeError('New gold solves are forbidden')
        gold.minimize = forbidden
        gold.solve_step = forbidden
        device = torch.device('cuda')
        all_results = []
        for fold, held in enumerate(identities):
            result_path = ROOT/f'fold_{fold}_results.json'
            if result_path.exists():
                all_results.append(json.loads(result_path.read_text()))
                continue
            checkpoint = ROOT/'models'/f'loio_fold_{fold}.pt'
            if checkpoint.exists():
                c = torch.load(checkpoint, map_location='cpu', weights_only=False)
                model = pilot.PrivilegedPolicy().to(device)
                model.load_state_dict(c['model'])
                model.eval()
                meta = dict(held_identity=held, training_identities=c['training_identities'], updates=800,
                            model_sha256=pilot.sha(checkpoint), recovered_final_model=True,
                            historical_training_log=str(ROOT/'logs/pilot_training.log'))
            else:
                model, meta = resumable_train(data, held, device, fold)
            trans, roll = resumable_evaluate(model, data, held, device)
            result = dict(training=meta, heldout_transitions=trans, heldout_rollouts=roll)
            validate_result(result, data, held, checkpoint)
            pilot.atomic_json(result_path, result)
            all_results.append(result)
            print('FOLD_COMPLETE '+json.dumps(dict(fold=fold, held=held)), flush=True)
            del model
            torch.cuda.empty_cache()
        results = dict(config=pilot.CONFIG, folds=[r['training'] for r in all_results],
                       heldout_transitions=sum([r['heldout_transitions'] for r in all_results], []),
                       heldout_rollouts=sum([r['heldout_rollouts'] for r in all_results], []),
                       gold_manifest_sha256=frozen['manifest_sha256'], no_new_gold_solves=True, no_final_student=True)
        if not (ROOT/'pilot_results.json').exists():
            pilot.atomic_json(ROOT/'pilot_results.json', results)
        summarize(results)
        print('RECOVERED_PILOT_COMPLETE', flush=True)


def summarize(results):
    trans = results['heldout_transitions']; rolls = results['heldout_rollouts']
    gold_gain = sum(r['gold_repair'] for r in trans)
    fast_gain = sum(r['fast_repair'] for r in trans)
    gold_roll = sum(r['baseline']['mean_local_rmse']-r['gold_final']['mean_local_rmse'] for r in rolls)
    fast_roll = sum(r['baseline']['mean_local_rmse']-r['fast_matched_final']['mean_local_rmse'] for r in rolls)
    latencies = [r['latency_seconds'] for r in trans]
    manifest = json.loads((ROOT/'gold/transitions.json').read_text())['transitions']
    matched = []
    lookup = {(v['trajectory_index'],v['step']):v for v in manifest}
    for fast in trans:
        gold = lookup[(fast['index'],fast['step'])]
        timing = gold.get('timing') or {}
        wall = (gold.get('solver_metadata') or {}).get('runtime_seconds') or timing.get('solver_wall_seconds')
        if wall: matched.append((wall, fast['latency_seconds']))
    summary = dict(one_step_repair_retention=fast_gain/gold_gain, matched_rollout_repair_retention=fast_roll/gold_roll,
                   action_rms_error_mean=float(np.mean([r['action_rms_error'] for r in trans])),
                   action_cosine_mean=float(np.mean([r['action_cosine'] for r in trans if r['action_cosine'] is not None])),
                   zero_action_fraction=float(np.mean([r['zero_action'] for r in trans])),
                   max_action_norm=max(r['action_max'] for r in trans),
                   shielded_latency_median_seconds=float(np.median(latencies)),
                   policy_latency_median_seconds=float(np.median([r['policy_seconds'] for r in trans])),
                   matched_timing_pairs=len(matched), measured_slsqp_speedup=(sum(v[0] for v in matched)/sum(v[1] for v in matched) if matched else None),
                   all_safety_passed=all(r['after']['safety']['passes'] for r in trans) and all(s['safety']['passes'] for r in rolls for s in r['states']),
                   new_inversions=max(s['safety']['quartets']['new_inversions'] for r in rolls for s in r['states']),
                   assessability_lost=max(s['safety']['quartets']['assessability_lost'] for r in rolls for s in r['states']),
                   fold_rollout_retention={r['identity']:[v['repair_retention'] for v in rolls if v['identity']==r['identity']] for r in rolls},
                   success=False, no_new_gold_solves=True)
    summary.update(
        matched_rollout_fast_local_rmse_repair_sum=fast_roll,
        matched_rollout_gold_local_rmse_repair_sum=gold_roll,
        retention_aggregation='Sum of absolute local RMSE repair across held-out cases divided by corresponding sum for gold; not mean of individual retention ratios',
        cosine_scope='Mean of transition cosines with nonzero gold and shielded actions; zero/near-zero pairs excluded',
        one_step_fast_local_rmse_repair_sum=fast_gain, one_step_gold_local_rmse_repair_sum=gold_gain,
        mean_one_step_local_rmse_gain_percent=float(np.mean([100*(1-r['after']['mean_local_rmse']/r['before']['mean_local_rmse']) for r in trans])),
        mean_matched_rollout_gain_percent=float(np.mean([r['fast_matched_gain_percent'] for r in rolls])),
        mean_gold_matched_rollout_gain_percent=float(np.mean([r['gold_gain_percent'] for r in rolls])),
        mean_P16_gain_percent=float(np.mean([100*(1-r['fast_P16']['mean_local_rmse']/r['baseline']['mean_local_rmse']) for r in rolls])),
        max_aligned_rmsd_ratio_to_frozen_baseline=max(s['aligned_rmsd']/r['baseline']['aligned_rmsd'] for r in rolls for s in r['states']),
        one_step_new_inversions=max(r['after']['safety']['quartets']['new_inversions'] for r in trans),
        one_step_assessability_lost=max(r['after']['safety']['quartets']['assessability_lost'] for r in trans),
        rollout_action_cap_passed=True,
        near_zero_actions=sum(r['zero_action'] or r['action_max']<=1e-8 for r in trans),
        shield_scale_counts={str(k):sum(r['shield_scale']==k for r in trans) for k in sorted({r['shield_scale'] for r in trans})},
        timing_comparison='Archived per-step solver runtime versus matching shielded policy wall time; excludes rows without durable timing',
        remaining_work='Pilot fails retention target; needs a new, separately frozen training/shield experiment before scaling' if fast_roll/gold_roll<.8 else 'Pilot meets repair target',
    )
    summary['success'] = summary['matched_rollout_repair_retention'] >= .8 and summary['all_safety_passed']
    pilot.atomic_json(RECOVERY/'summary.json',summary)
    audit = json.loads((ROOT/'gold/audit.json').read_text())
    step_table = '| Step (zero-based) | Transitions | Mean gold MSE gain % | Zero action fraction |\n|---|---:|---:|---:|\n'
    for step, row in sorted(audit['marginal_gain_by_step'].items(),key=lambda item:int(item[0])):
        step_table += f"| {step} | {row['count']} | {row['mean_step_mse_gain_percent']:.4f} | {row['zero_fraction']:.4f} |\n"
    text = ('# E010 recovered privileged teacher pilot\n\n'
            f"Recovered {audit['completed_trajectories']} completed and {audit['partial_trajectories']} partial trajectories; "
            f"{audit['total_accepted_transitions']} transitions across {audit['unique_identities']} identities. "
            'All frozen file and source hashes verified. No gold solves ran. Six final model checkpoints reused; final fold trained under the original fixed protocol.\n\n'
            'Seven protein-identity-held-out folds, fixed 800 updates per fold. This teacher sees target geometry; it is not the deployable student.\n\n'
            f"Gold transitions by corruption condition: {json.dumps(audit['transitions_by']['condition'],sort_keys=True)}. "
            f"By length bin: {json.dumps(audit['transitions_by']['length_bin'],sort_keys=True)}. "
            f"By identity: {json.dumps(audit['transitions_by']['identity'],sort_keys=True)}.\n\n"
            f"Gold mean endpoint local RMSE improvement: {audit['endpoint_rmse_gains']['mean']:.4f}%; "
            f"mean per-step local MSE improvement: {audit['step_mse_gains']['mean']:.4f}%. "
            f"Action RMS mean/median/max: {audit['action_rms']['mean']:.6f}/{audit['action_rms']['median']:.6f}/{audit['action_rms']['maximum']:.6f} Å; "
            f"maximum per-residue norm: {audit['action_max']['maximum']:.6f} Å. "
            f"Exact zero/near-zero actions: {audit['zero_actions']}/{audit['near_zero_actions']}; near-zero threshold 1e-8 Å. "
            'Every gold accepted state passes frozen safety. Gold masks, exact actions/states/targets, objective values, safety metrics, solver metadata and source hashes remain in the immutable manifest and NPZ files.\n\n'
            +step_table+'\n'
            '```json\n'+json.dumps(summary,indent=2)+'\n```\n\n'
            'Gold coverage, action statistics, gain versus step, and safety: ../gold/audit.json. '
            'Full one-step and rollout metrics: ../pilot_results.json. Recovery provenance: recovered_state.json.\n\n'
            'Speedup compares archived SLSQP timings with measured shielded neural inference; no new solver benchmarks. '
            'Historical and recovered folds were measured in separate processes. '
            'The retention result determines whether this pilot is ready for later teacher-data generation. '
            'No final student or full corpus was generated.\n')
    path=RECOVERY/'REPORT.md'
    fd,tmp=tempfile.mkstemp(dir=RECOVERY,suffix='.tmp')
    with os.fdopen(fd,'w') as stream:
        stream.write(text);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path);pilot.fsync_dir(RECOVERY)


def final_validate_and_summarize():
    pilot.initialize()
    data, frozen = pilot.load()
    results = json.loads((ROOT/'pilot_results.json').read_text())
    identities = sorted({v['row']['record']['sample_id'] for v in data.values()})
    assert len(results['folds']) == len(identities) == 7
    assert len(results['heldout_transitions']) == 278
    contract = json.loads((ROOT.parent/'v16d_acceleration_audit_20261007/inputs/execution_contract.json').read_text())
    code_checks = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith('protein_distance_diffusion') and path:
            code_checks[path] = pilot.sha(path)
            assert contract['protected_sha256'][path] == code_checks[path], path
    for fold, held in enumerate(identities):
        result = json.loads((ROOT/f'fold_{fold}_results.json').read_text())
        checkpoint = ROOT/'models'/f'loio_fold_{fold}.pt'
        c = torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert c['held_identity'] == held and c['training_identities'] == sorted(set(identities)-{held})
        assert c['config'] == pilot.CONFIG and c['gold_manifest_sha256'] == frozen['manifest_sha256']
        assert all(torch.isfinite(v).all() for v in c['model'].values())
        validate_result(result,data,held,checkpoint)
    summarize(results)
    files = {str(path):pilot.sha(path) for path in ROOT.rglob('*') if path.is_file() and '/cache/' not in str(path) and '/tmp/' not in str(path)}
    files.pop(str(RECOVERY/'FINAL_VALIDATED.json'), None)
    pilot.atomic_json(RECOVERY/'FINAL_VALIDATED.json',dict(files=files,source_code_pins_verified=code_checks,
                     recovery_source_sha256=pilot.sha(__file__),folds=7,transitions=278,rollouts=20,
                     gold_manifest_sha256=frozen['manifest_sha256'],no_new_gold_solves=True))


if __name__ == '__main__':
    if '--summarize' in sys.argv:
        final_validate_and_summarize()
    else:
        main()
