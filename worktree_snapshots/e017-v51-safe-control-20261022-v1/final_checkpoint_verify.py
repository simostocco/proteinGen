"""Read-only CPU verification of existing complete checkpoints; no replay step."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent/'src'))
from protein_joint_sac.crisp_v3.recovery import sha, state_digest
from final_scientific_analysis import R, read, write, VERSION

torch.set_num_threads(1)
out = dict(schema=VERSION, verification='fresh SHA256 + CPU deserialize + complete semantic digest; no model construction, GPU load, or optimizer step', arms={})
for arm in 'ABCD':
    index=read(R/f'checkpoints/{arm}/INDEX.json')
    stop=read(R/f'results/{arm}_STOP.json')
    finished=read(R/f'results/{arm}_FINISHED.json')
    assert finished['update']==stop['exact_update']==250
    assert stop['complete_verified_recovery']==index['latest']
    out['arms'][arm]=dict(stop=stop, finished=finished, states={})
    for role in ['latest','best']:
        rec=index[role]
        p=Path(rec['path'])
        assert sha(p)==rec['sha256'] and p.stat().st_size==rec['bytes']
        st=torch.load(p,map_location='cpu',weights_only=False)
        assert state_digest(st)==rec['state_digest']
        required=['actor','optimizer','scheduler','sampler','rng','dual','arm','update','best','monitor','stop_reason','config','config_hash','source','input_hashes','parent','parent_SHA','J1_tensor_SHA','immutable_nonactor_system_parent','replay','stage']
        assert all(k in st for k in required)
        assert st['update']==rec['update'] and st['arm']==arm and st['scheduler']['last_epoch']==st['update']
        adam_steps=sorted({int(x['step'].item()) for x in st['optimizer']['state'].values()})
        assert adam_steps==[st['update']]
        assert st['sampler']['cursor']==st['update']*st['config']['batch']
        assert st['source']==read(R/'SEALED_CONFIG.json')['source']
        assert st['input_hashes']==read(R/'INPUT_MANIFEST.json')
        assert st['stop_reason']==(stop['reason'] if role=='latest' else None)
        rng=st['rng']
        assert all(v is not None for v in rng.values())
        timestamps={k:Path(v).stat().st_mtime_ns for k,v in dict(checkpoint=p,index=R/f'checkpoints/{arm}/INDEX.json',stop=R/f'results/{arm}_STOP.json',finished=R/f'results/{arm}_FINISHED.json').items()}
        assert timestamps['checkpoint']<=timestamps['index']<=timestamps['stop']<=timestamps['finished']
        order=st['sampler']['order'];cursor=st['sampler']['cursor']
        tasks=read(R.parent/'e017_v5_goal_observability_20261022_v1/manifests/TRAIN.json')['tasks']
        pool=[t for t in tasks if t['H']==4]
        nxt=pool[order[cursor]] if cursor<len(order) else None
        out['arms'][arm]['states'][role]=dict(record=rec,SHA256_verified=True,CPU_deserialized=True,
            complete_semantic_digest_verified=True,required_fields=required,optimizer_parameter_states=len(st['optimizer']['state']),
            optimizer_step_values=adam_steps,scheduler=st['scheduler'],rng_keys=list(rng),
            rng_state_descriptors={k:dict(type=type(v).__name__,length=len(v) if hasattr(v,'__len__') else None) for k,v in rng.items()},
            AMP_scaler='not applicable: BF16 autocast, no GradScaler in training source',sampler_cursor=cursor,
            sampler_order_length=len(order),next_identity=nxt['identity'] if nxt else None,next_task_id=nxt['task_id'] if nxt else None,
            dual=st['dual'],config_hash=st['config_hash'],source=st['source'],J1_tensor_SHA=st['J1_tensor_SHA'],
            parent_SHA=st['parent_SHA'],immutable_nonactor_system_parent=st['immutable_nonactor_system_parent'],
            publication_mtime_ns=timestamps,scientifically_resumable=False,
            recovery_semantics='full actor state plus immutable unchanged-system parent; exact numerical recovery does not authorize stopped study continuation')
        del st
        print(arm,role,'verified',flush=True)
out['engineering_tests']=read(R/'CHECKPOINT_STOP_RECOVERY_TESTS.json')
out['fault_injection']=read(R/'ACTUAL_STOP_CONTROL_FLOW_FAULT_INJECTION.json')
out['partial_checkpoint_files']=[str(p) for p in (R/'checkpoints').rglob('*.part')]
assert not out['partial_checkpoint_files']
out['no_uncommitted_update_evidence']='250 contiguous logged updates, Adam and scheduler step250, sampler1000, STOP and FINISHED250; mandatory publish precedes STOP and FINISHED; all digests valid'
out['timestamp_unix']=time.time()
write(R/'source/final_checkpoint_work.json',out)
