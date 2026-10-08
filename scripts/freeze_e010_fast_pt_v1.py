"""Seal FAST-PT-v1 as a completed negative pilot without modifying its results."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path('/mnt/d/Simone/proteinGen_data/e010_phase4d/fast_privileged_teacher_pilot_v1')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    final = json.loads((ROOT/'recovery_20261008/FINAL_VALIDATED.json').read_text())
    for path, expected in final['files'].items():
        assert sha(path) == expected, path
    summary = json.loads((ROOT/'recovery_20261008/summary.json').read_text())
    audit = json.loads((ROOT/'gold/audit.json').read_text())
    assert (audit['completed_trajectories'], audit['partial_trajectories'],
            audit['total_accepted_transitions'], audit['unique_identities']) == (15, 5, 278, 7)
    assert summary['all_safety_passed'] and not summary['success']
    marker = dict(experiment='FAST-PT-v1', status='completed_negative_pilot',
                  architecture_frozen=True, do_not_enlarge_or_retrain=True,
                  gold_audit=audit, results=summary,
                  identity_held_out_folds=7, no_new_gold_solves=True,
                  files=final['files'], final_validation_sha256=sha(ROOT/'recovery_20261008/FINAL_VALIDATED.json'))
    destination = ROOT/'FAST_PT_V1_FROZEN_NEGATIVE.json'
    if destination.exists():
        assert json.loads(destination.read_text()) == marker
    else:
        fd, tmp = tempfile.mkstemp(dir=ROOT, suffix='.tmp')
        with os.fdopen(fd, 'w') as stream:
            json.dump(marker, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.rename(tmp, destination)
        directory = os.open(ROOT, os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    print(destination)


if __name__ == '__main__':
    main()
