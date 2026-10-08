"""Independent raw-control reconstruction and archived baseline parity."""
import concurrent.futures
import json
import multiprocessing
import time
import numpy as np
import torch
import train_e010_student_direction_observability_v1 as t
e=t.e


def check_identity(task):
    variant,split,row,deep=task
    path=t.ROOT/'evaluation'/f'{variant}_{split}'/f'identity_{row["index"]:04d}.json'
    rows=json.loads(path.read_text());archive=path.with_suffix('.npz')
    digest=t.old.sha(archive);assert all(r['action_archive_sha256']==digest for r in rows)
    count=0
    with np.load(archive,allow_pickle=False) as saved:
        if variant=='A':
            original=t.old.ROOT/'evaluation'/f'imitation_{split}'/f'identity_{row["index"]:04d}.npz'
            with np.load(original,allow_pickle=False) as gold:
                for name in saved.files:assert np.array_equal(saved[name],gold[name]),(row['sample_id'],name)
        if not deep:return dict(archives=1,baseline_archives=int(variant=='A'),independent_states=0)
        item=t.identity_data(row)
        payload=torch.load(t.ROOT/'models'/f'{variant}_best.pt',map_location='cpu',weights_only=False)
        t.MODEL=t.ContextStudent(variant).eval();t.MODEL.load_state_dict(payload['model'])
        for ci,c in enumerate(item['conditions']):
            result=next(r for r in rows if r['condition']==c['condition'])
            b={key:torch.from_numpy(c[name][None]) for key,name in [('pg','P0'),('source','source'),('target','target'),('mask','mask')]}
            ctx=t.old.pilot.Context(b);available=torch.from_numpy(t.context(item,ci,variant)[None]);p=b['pg'].clone()
            for i,delta in enumerate(saved[f'{c["condition"]}_raw_rollout_delta']):
                valid=t.old.local_representation(p,b['mask'])['eligible'].numpy()
                assert np.all(delta[~valid]==0) and np.linalg.norm(delta,axis=-1).max()<=.1
                with torch.no_grad():predicted,_=t.predict(p,b['mask'],available)
                assert np.array_equal(predicted,delta)
                p=p+torch.from_numpy(delta);after=t.old.metrics(ctx,p);record=result['rollouts']['raw']['steps'][i]
                assert np.isclose(after['mean_local_rmse'],record['after']['mean_local_rmse'],rtol=1e-12,atol=1e-10)
                assert np.isclose(after['aligned_rmsd'],record['after']['aligned_rmsd'],rtol=1e-12,atol=1e-10)
                assert after['safety']['passes']==record['after']['safety']['passes']
                for key in ('new_inversions','assessability_lost'):assert after['safety']['quartets'][key]==record['after']['safety']['quartets'][key]
                count+=1
            for i in (0,3,7,15):
                state,mask,_,_=t.old.sample(item,ci,i);p=torch.from_numpy(state[None]);delta=saved[f'{c["condition"]}_raw_one_delta'][i]
                with torch.no_grad():predicted,_=t.predict(p,b['mask'],available)
                assert np.array_equal(delta,predicted)
                after=t.old.metrics(ctx,p+torch.from_numpy(delta));record=result['one_step'][i]
                assert after['safety']['passes']==record['after']['safety']['passes']
                assert np.isclose(after['mean_local_rmse'],record['after']['mean_local_rmse'],rtol=1e-12,atol=1e-10);count+=1
    return dict(archives=1,baseline_archives=int(variant=='A'),independent_states=count)


def main():
    while not (t.ROOT/'RUNS_COMPLETED.json').exists():time.sleep(40)
    splits=json.loads((t.old.ROOT/'splits.json').read_text())['splits'];rows=e.rows();by_id={r['sample_id']:r for r in rows}
    tasks=[]
    for variant in t.VARIANTS:
        assert json.loads((t.ROOT/f'{variant}_training.json').read_text())['parameters']==118723
        for split in ('validation','test'):
            deep=set()
            for length in t.old.STRATA:
                candidates=[sid for sid in splits[split] if by_id[sid]['stratum']==length]
                deep.update(sorted(candidates,key=lambda sid:e.rank(f'independentC|{split}|{sid}'))[:2])
            tasks.extend((variant,split,by_id[sid],sid in deep) for sid in splits[split])
    total=dict(archives=0,baseline_archives=0,independent_states=0)
    with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn'),initializer=e.initialize) as pool:
        for i,result in enumerate(pool.map(check_identity,tasks),1):
            for key in total:total[key]+=result[key]
            if i%100==0:print(json.dumps(dict(checked=i,total=len(tasks))),flush=True)
    assert total==dict(archives=3000,baseline_archives=600,independent_states=6000)
    e.write(t.ROOT/'tests.json',dict(passed=True,**total,raw_prediction_reconstruction=True,
        baseline_all_one_step_and_rollout_actions_bitwise_equal_fc76211=True,
        all_length_condition_step_and_split_cells_independently_covered=True,
        original_safety_metrics_reproduced=True,all_variants_same_capacity=True,no_X_based_inference_shield=True))
    print(json.dumps(dict(passed=True,**total)),flush=True)


if __name__=='__main__':main()
