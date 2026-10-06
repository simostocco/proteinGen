"""Wait for ranged transfer, then run the frozen CPU-only preparation stages."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

REPO=Path('/home/simostocco/proteinGen-causal-rope')
ROOT=Path('/mnt/d/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2')

def save(obj):
    obj['timestamp']=datetime.now(timezone.utc).isoformat()
    obj['training_launched']=False
    p=ROOT/'stats/build_status.json'
    tmp=p.with_suffix('.partial')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    tmp.replace(p)

def main():
    plan=json.loads((ROOT/'stats/resource_plan.json').read_text())
    floor=plan['starting_free_bytes']*3//10
    minimum=shutil.disk_usage(ROOT).free
    started=datetime.now(timezone.utc).isoformat()
    transport=ROOT/'checksums/download_transport.json'
    while not transport.exists() and not (ROOT/'manifests/raw.complete.json').exists():
        free=shutil.disk_usage(ROOT).free
        minimum=min(minimum,free)
        if free<floor:
            save({'status':'STOPPED','classification':'DATA2-E','reason':'D-disk runtime reserve breached before preprocessing'})
            raise RuntimeError('Disk reserve breached; no preprocessing')
        folder=ROOT/'tmp/download_ranges'
        downloaded=sum(p.stat().st_size for p in folder.glob('*.bin'))
        save({'status':'WAITING_FOR_SOURCE','download_bytes_observed':downloaded,
            'expected_archive_bytes':8780552383,'free_bytes':free,
            'classification':None,'minimum_free_sampled_bytes':minimum})
        time.sleep(10)
    free=shutil.disk_usage(ROOT).free
    minimum=min(minimum,free)
    with (ROOT/'stats/telemetry.jsonl').open('a') as f:
        f.write(json.dumps({'stage':'download_transport','start_observed':started,
            'end_observed':datetime.now(timezone.utc).isoformat(),
            'free_before_bytes':plan['starting_free_bytes'],'free_after_bytes':free,
            'minimum_free_sampled_bytes':minimum,
            'measured_peak_consumption_bytes':plan['starting_free_bytes']-minimum,
            'measurement':'Filesystem free sampled while waiting; starting source-transfer portion not continuously observed. Includes unrelated D: activity.',
            'cpu_threads':4,'parent_maxrss_kib':None,'children_maxrss_kib':None,'training':False},sort_keys=True)+'\n')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONDONTWRITEBYTECODE='1')
    command=[sys.executable,'-B',str(REPO/'scripts/prepare_e012_sequence_corpus_v2b.py'),'all']
    save({'status':'BUILD_RUNNING','command':command,'classification':None,
        'processing_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()})
    log=ROOT/'stats/build.log'
    with log.open('a') as f:
        code=subprocess.call(command,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
    if code:
        tail=log.read_text()[-10000:]
        match=re.search(r'DATA2-[CDEF]',tail)
        save({'status':'STAGE_FAILED','exit_code':code,'classification':match.group() if match else None,
            'log':str(log),'reason':tail.splitlines()[-1] if tail.splitlines() else 'unknown'})
        raise SystemExit(code)
    handoff=json.loads((REPO/'reports/experiments/E012_causal_rope_sequence/data_v2b_uniref50_lowdisk/chatgpt_handoff.json').read_text())
    save({'status':'FROZEN_CORPUS_READY','classification':handoff['classification'],
        'n':handoff['final_corpus_n'],'log':str(log)})
    print(json.dumps({'status':'FROZEN_CORPUS_READY','classification':handoff['classification'],'n':handoff['final_corpus_n']}))

if __name__=='__main__':
    main()
