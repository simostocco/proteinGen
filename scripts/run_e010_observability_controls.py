"""Run only the five authorized fixed-capacity observability controls."""
import json
import subprocess
import sys
import time
import torch
import train_e010_student_direction_observability_v1 as t
t.e.storage()


def run(mode,variant,split=None):
    command=[sys.executable,'-B',str(t.e.REPO/'scripts/train_e010_student_direction_observability_v1.py'),mode,'--variant',variant]
    if split:command+=['--split',split]
    with (t.e.ROOT/'logs'/f'{mode}_{variant}_{split or "controls"}.log').open('a') as log:
        subprocess.run(command,cwd=t.e.REPO,stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    t.prepare();assert json.loads((t.ROOT/'input_tests.json').read_text())['passed']
    while not (t.ROOT/'A_training.json').exists():time.sleep(40)
    baseline=torch.load(t.old.ROOT/'models/imitation_best.pt',map_location='cpu',weights_only=False)['model']
    reproduction=torch.load(t.ROOT/'models/A_01000.pt',map_location='cpu',weights_only=False)['model']
    maximum=max(float((value-reproduction[name]).abs().max()) for name,value in baseline.items())
    t.e.write(t.ROOT/'baseline_reproduction.json',dict(first_1000_updates_bitwise=maximum==0.,maximum_parameter_difference=maximum,
        original_model_sha256=t.old.sha(t.old.ROOT/'models/imitation_best.pt'),reproduced_model_sha256=t.old.sha(t.ROOT/'models/A_01000.pt')))
    assert maximum==0., 'Baseline mismatch: stop controlled ladder and diagnose before comparing added context.'
    for variant in ('B','C','D','E'):
        if not (t.ROOT/f'{variant}_training.json').exists():run('train',variant)
        print(json.dumps(dict(training_complete=variant)),flush=True)
    for variant in t.VARIANTS:
        for split in ('validation','test'):
            run('evaluate',variant,split);print(json.dumps(dict(raw_evaluation_complete=variant,split=split)),flush=True)
    t.e.write(t.ROOT/'RUNS_COMPLETED.json',dict(variants=list(t.VARIANTS),raw_evaluation_splits=['validation','test'],no_X_inference_shield=True))


if __name__=='__main__':main()
