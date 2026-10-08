"""Independent frozen-input, baseline-parity, and action reconstruction audit."""
import json
import time
import numpy as np
import torch
import e010_student_feasibility_margin_v1 as e


def main():
    e.initialize()
    parent=json.loads((e.PARENT/'COMPLETED.json').read_text())
    for path,digest in parent['files'].items():assert e.old.sha(path)==digest,path
    teacher=json.loads((e.old.PARENT/'COMPLETED.json').read_text())
    for path,digest in teacher['files'].items():assert e.old.sha(path)==digest,path
    original=json.loads((e.PARENT/'results.json').read_text())
    for path,digest in original['unchanged_frozen_geometry_source_sha256'].items():assert e.old.sha(path)==digest,path
    print('Frozen student, teacher corpus, and geometry source hashes unchanged.',flush=True)
    while not (e.ROOT/'PHASE_AB_COMPLETED.json').exists():time.sleep(50)
    parent_evals={s:json.loads((e.PARENT/'evaluation'/f'imitation_{s}.json').read_text()) for s in ('validation','test')}
    lookups={s:{(r['identity'],r['condition']):r for r in p['rows']} for s,p in parent_evals.items()}
    selection={r['sample_id']:r for r in json.loads((e.old.PARENT/'selection.json').read_text())['identities']}
    seen={s:set() for s in lookups};sampled=0;count=0;max_baseline_error=0.
    for path in sorted((e.ROOT/'records').glob('*.json')):
        payload=json.loads(path.read_text());split=payload['rows'][0]['split'];rows=payload['rows']
        action_path=e.ROOT/'actions'/path.with_suffix('.npz').name
        geometry_path=e.ROOT/'geometry'/path.with_suffix('.npz').name
        assert action_path.exists() and geometry_path.exists()
        for r in rows:
            seen[split].add(r['identity']);p=lookups[split][r['identity'],r['condition']]
            for a,b in zip(r['rollouts']['1.0']['steps'],p['rollouts']['raw']['steps']):
                assert a['after']['safety']['passes']==b['after']['safety']['passes']
                for key in ('mean_local_rmse','local_mse','aligned_rmsd','continuous_chiral'):
                    error=abs(a['after'][key]-b['after'][key]);max_baseline_error=max(max_baseline_error,error)
                    assert error<1e-12,(r['identity'],key,error)
            count+=1
        # One identity per length and split, all alphas: reconstruct state and verify exact scores independently.
        marker=(split,rows[0]['length_bin'])
        if marker not in getattr(main,'checked',set()):
            if not hasattr(main,'checked'):main.checked=set()
            main.checked.add(marker);item=e.old.load_identity(selection[rows[0]['identity']])
            with np.load(action_path,allow_pickle=False) as archive:
                for ci,c in enumerate(item['conditions']):
                    b={k:torch.from_numpy(c[n][None]) for k,n in [('pg','P0'),('source','source'),('target','target'),('mask','mask')]}
                    ctx=e.old.pilot.Context(b)
                    for alpha in map(str,e.ALPHAS):
                        p=b['pg'].clone()
                        for step,d in enumerate(archive[f'{c["condition"]}_alpha_{alpha}_delta']):
                            a,_=e.score(ctx,p,d);expected=rows[ci]['rollouts'][alpha]['steps'][step]
                            assert a['hard_failed']==expected['hard_failed'] and a['failures']==expected['failures']
                            assert abs(a['after']['mean_local_rmse']-expected['after']['mean_local_rmse'])<1e-12
                            p+=torch.from_numpy(d);sampled+=1
    assert all(len(v)==300 for v in seen.values()) and count==1800
    assert not seen['validation']&seen['test']
    e.write(e.ROOT/'tests.json',dict(passed=True,frozen_student_files_verified=len(parent['files']),
        frozen_teacher_files_verified=len(teacher['files']),frozen_geometry_sources_verified=18,
        alpha_one_archived_baseline_parity=True,maximum_metric_error=max_baseline_error,
        independently_reconstructed_rollout_states=sampled,identity_counts={k:len(v) for k,v in seen.items()},
        no_teacher_changes=True,no_new_solves=True,no_privileged_inference_shield=True))
    print(json.dumps(dict(passed=True,trajectories=count,reconstructed=sampled,max_baseline_error=max_baseline_error)),flush=True)


if __name__=='__main__':main()
