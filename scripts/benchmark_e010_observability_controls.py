"""Isolated, identical-fixture CPU latency for the five frozen controls."""
import json
import time
import numpy as np
import torch
import train_e010_student_direction_observability_v1 as t
e=t.e


def main():
    while not (t.ROOT/'tests.json').exists():time.sleep(40)
    assert json.loads((t.ROOT/'tests.json').read_text())['passed'];e.initialize()
    split=json.loads((t.old.ROOT/'splits.json').read_text())['splits']['validation'];rows=e.rows();by_id={r['sample_id']:r for r in rows};selected=[]
    for length in t.old.STRATA:
        candidates=[sid for sid in split if by_id[sid]['stratum']==length]
        selected+=sorted(candidates,key=lambda sid:e.rank(f'latency|{sid}'))[:2]
    items=[t.identity_data(by_id[sid]) for sid in selected];timings={};preparation=[]
    transform=json.loads((e.ROOT/'panel/transforms.json').read_text())['coarse_output_context'];mean=np.asarray(transform['mean'][6:],np.float32);scale=np.asarray(transform['scale'][6:],np.float32)
    for item in items:
        for ci,c in enumerate(item['conditions']):
            source=torch.from_numpy(c['source'][None]);mask=torch.from_numpy(c['mask'][None]);start=time.perf_counter()
            with torch.no_grad():features=t.old.local_representation(source,mask)['features'][0].float().numpy();available=(features-mean)/scale
            preparation.append(time.perf_counter()-start);assert np.array_equal(available,t.context(item,ci,'E'))
    for variant in t.VARIANTS:
        payload=torch.load(t.ROOT/'models'/f'{variant}_best.pt',map_location='cpu',weights_only=False)
        t.MODEL=t.ContextStudent(variant).eval();t.MODEL.load_state_dict(payload['model']);values=[]
        for item in items:
            for ci,c in enumerate(item['conditions']):
                available=torch.from_numpy(t.context(item,ci,variant)[None]);mask=torch.from_numpy(c['mask'][None])
                for step in (0,7):
                    p=torch.from_numpy(t.old.sample(item,ci,step)[0][None])
                    for _ in range(2):t.predict(p,mask,available)
                    for _ in range(10):
                        _,info=t.predict(p,mask,available);values.append(info['policy_seconds'])
        timings[variant]=t.old.frozen.old.statistics(values)
    e.write(t.ROOT/'latency_benchmark.json',dict(device='CPU',torch_threads=1,identities=selected,length_bins=list(t.old.STRATA),
        conditions=[50,250,450],logical_steps=[1,8],warmup=2,repeats=10,samples_per_variant=600,
        scope='current-state local geometry, fixed context projection, network and closed-ball cap guard; exact safety/X scoring excluded',
        source_feature_preparation_once_per_trajectory_seconds=t.old.frozen.old.statistics(preparation),
        timings_seconds=timings,no_target_network_input=True,isolated_from_training_and_raw_evaluation=True))
    print(json.dumps(dict(completed=True,timings=timings)),flush=True)


if __name__=='__main__':main()
