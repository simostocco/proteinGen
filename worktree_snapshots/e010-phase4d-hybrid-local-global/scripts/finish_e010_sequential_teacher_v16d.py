"""Wait for the once-only panel, then reproduce and publish without optimization."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/experiments/E010_global_equivariant_expressivity/phase4d_hybrid_local_global_v1/sequential_teacher_v16d'


def write_once(name, value):
    with (OUT/name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    assert all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
    branch = subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    assert branch == 'e010-phase4d-hybrid-local-global'
    write_once('postprocessing_registered.json',dict(optimization_invocations=0,
        stages=['independent_reproduction','teacher_evaluation','compact_publication'],pid=os.getpid()))
    while not (OUT/'run_complete.json').exists():
        log = Path('/tmp/e010_v16d_panel.log').read_text()
        if 'Traceback' in log:
            raise RuntimeError('Teacher panel exception; postprocessing stopped without optimization.')
        time.sleep(5)
    commands = [
        [sys.executable,'scripts/run_e010_sequential_teacher_v16d.py','reproduce'],
        [sys.executable,'scripts/report_e010_sequential_teacher_v16d.py'],
        [sys.executable,'scripts/publish_e010_sequential_teacher_v16d.py'],
    ]
    started = time.perf_counter()
    for command in commands:
        subprocess.run(command,cwd=ROOT,env=os.environ.copy(),check=True)
    certificate = json.loads((OUT/'reproduction_complete.json').read_text())
    assert certificate['exact'] and certificate['examples'] == 60 and certificate['optimizer_invocations'] == 0
    result = json.loads((OUT/'teacher_result.json').read_text())
    write_once('postprocessing_complete.json',dict(optimization_invocations=0,
        examples=60,classification=result['classification'],runtime_seconds=time.perf_counter()-started))
    print('Read-only reproduction and conditional teacher publication complete.',flush=True)

if __name__ == '__main__':
    main()
