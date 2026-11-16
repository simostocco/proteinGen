"""Publish compact teacher records while retaining full traces outside Git.

Reads completed, independently reproduced trajectories; never optimizes or
changes a teacher state, action, metric, or acceptance decision.
"""
import json
import hashlib
import numpy as np
from scripts.run_e010_sequential_teacher_v16d import OUT, ROOT, write
from protein_distance_diffusion.training.e010_phase4d_diagnostic import file_hash


def publish():
    assert json.loads((OUT/'run_complete.json').read_text())['workers'] > 0
    reproduced = json.loads((OUT/'reproduction_complete.json').read_text())
    assert reproduced['exact'] and reproduced['examples'] == 60 and reproduced['optimizer_invocations'] == 0
    result = json.loads((OUT/'teacher_result.json').read_text())
    manifest_path = OUT/'frozen_transition_manifest.json'
    transitions = json.loads(manifest_path.read_text())['transitions'] if manifest_path.exists() else []
    if result['teacher_frozen_for_distillation']:
        assert len(transitions) == 960
        assert len({(t['trajectory_id'],t['step']) for t in transitions}) == 960
    artifacts = []
    for i in range(60):
        path = OUT/'examples'/f'example_{i:02d}.json'
        row = json.loads(path.read_text())
        assert row['index'] == i and len(row['states']) == 17 and len(row['solver_steps']) == 16
        raw_path = ROOT/row['trajectory_file']
        assert file_hash(raw_path) == row['trajectory_sha256']
        original_hash = file_hash(path)
        compact = json.loads(json.dumps(row))
        compact['full_record_file'] = str(path.relative_to(ROOT))
        compact['full_record_sha256'] = original_hash
        for state in compact['states']:
            cosines = state.pop('consecutive_cosines', None)
            if cosines is not None:
                state['consecutive_cosine_summary'] = dict(count=len(cosines),
                    quantiles=np.quantile(cosines,[0,.1,.5,.9,1]).tolist() if cosines else None)
        for step in compact['solver_steps']:
            history = step.pop('history')
            encoded = json.dumps(history,sort_keys=True,separators=(',',':')).encode()
            step['history_length'] = len(history)
            step['history_sha256'] = hashlib.sha256(encoded).hexdigest()
        compact_path = OUT/'compact_examples'/f'example_{i:02d}.json'
        write(compact_path, compact)
        for transition in (t for t in transitions if t['trajectory_id'] == i):
            step = transition['step']
            assert transition['source_trajectory_sha256'] == row['trajectory_sha256']
            assert transition['state_sha256'] == row['states'][step]['coordinate_sha256']
            assert transition['action_sha256'] == row['states'][step+1]['action_sha256']
            assert transition['teacher_safety'] and row['states'][step+1]['safety']['passes']
        assert file_hash(path) == original_hash
        artifacts.append(dict(index=i,full_record_file=str(path.relative_to(ROOT)),
            full_record_sha256=original_hash,full_record_bytes=path.stat().st_size,
            trajectory_file=row['trajectory_file'],trajectory_sha256=row['trajectory_sha256'],
            trajectory_bytes=raw_path.stat().st_size,compact_record_file=str(compact_path.relative_to(ROOT)),
            compact_record_sha256=file_hash(compact_path)))
    write(OUT/'publication_manifest.json',dict(examples=60,artifacts=artifacts,
        teacher_frozen_for_distillation=result['teacher_frozen_for_distillation'],
        transition_manifest_sha256=file_hash(manifest_path) if manifest_path.exists() else None,
        full_solver_traces_and_trajectory_arrays_tracked_in_git=False,
        optimization_invocations=0,original_records_unmodified=True))
    print('Published 60 compact records and verified immutable transition/artifact links.')

if __name__ == '__main__':
    publish()
