"""Descriptive action-norm sensitivity; never used for context/model selection."""
import collections
import json
import numpy as np
import analyze_e010_student_direction_observability_v1 as a
e=a.e


def main():
    meta=a.open_array('meta');values=collections.defaultdict(list)
    for row in e.rows():
        path=e.ROOT/'neighbors'/f'identity_{row["index"]:04d}.npz'
        with np.load(path,allow_pickle=False) as z:
            split={0:'train',1:'validation',2:'test'}[int(meta[z['query_rows'][0],6])]
            if split=='train':continue
            for representation in ('geometry','latent'):
                p=z[f'{representation}_baseline_pairs'];values[(split,representation)].append(p[:,:,[0,5,7]].reshape(-1,3))
    result=dict(descriptive_posthoc=True,no_gate_or_training_change=True,
        reason='Check whether angular disagreement is concentrated in tiny teacher vectors; norm strata are outcomes only, not matching features.',
        summaries=[])
    for (split,representation),parts in values.items():
        data=np.concatenate(parts);valid=np.isfinite(data[:,0]);data=data[valid]
        for threshold in (1e-8,.001,.01,.03):
            chosen=data[(data[:,1]>threshold)&(data[:,2]>threshold)]
            weight=chosen[:,1]*chosen[:,2]
            result['summaries'].append(dict(split=split,representation=representation,both_action_norms_above_A=threshold,
                pairs=len(chosen),fraction_of_defined_pairs=len(chosen)/len(data),
                median_cosine=float(np.median(chosen[:,0])) if len(chosen) else None,
                opposition_rate=float(np.mean(chosen[:,0]<0)) if len(chosen) else None,
                action_norm_weighted_mean_cosine=float(np.average(chosen[:,0],weights=weight)) if len(chosen) else None,
                mean_absolute_action_norm_difference=float(np.abs(chosen[:,1]-chosen[:,2]).mean()) if len(chosen) else None))
    e.write(e.ROOT/'action_norm_sensitivity.json',result)
    print(json.dumps(dict(completed=True,summaries=len(result['summaries']))),flush=True)


if __name__=='__main__':main()
