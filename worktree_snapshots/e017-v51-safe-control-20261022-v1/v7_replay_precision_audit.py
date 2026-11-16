"""Diagnose the stopped replay test without constructing any model."""
import json
from pathlib import Path
import sys
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v7_amortized_lowlevel_20261023_v1')
sys.path.insert(0,str(R/'source/src'))
from protein_joint_sac.geometry.ca_builder import next_ca
from protein_joint_sac.crisp_v3.recovery import atomic_json,sha
ts={t['task_id']:t for t in json.loads((R/'manifests/TRAIN.json').read_text())['tasks']};eps=[json.loads(s) for s in (R/'results/TRAIN_candidates.jsonl').read_text().splitlines()];out=[]
for ep in eps[::max(1,len(eps)//32)][:32]:
 x=torch.tensor(ts[ep['task_id']]['native_prefix'],dtype=torch.float64)
 for action in ep['geometry']:x=torch.cat([x,next_ca(x,torch.tensor(action,dtype=torch.float64))[None]],0)
 a=float((x[-4:]-torch.tensor(ep['tail'],dtype=torch.float64)).abs().max());b=float((x[-4:]-torch.tensor(ep['tail'])).abs().max());out.append(dict(task_id=ep['task_id'],candidate_index=ep['candidate_index'],FP64_error_A=a,default_FP32_reference_error_A=b));assert a<1e-8
atomic_json(R/'REPLAY_PRECISION_DIAGNOSIS.json',dict(PASS=True,original_threshold=1e-8,FP64_max_error_A=max(r['FP64_error_A'] for r in out),default_FP32_max_error_A=max(r['default_FP32_reference_error_A'] for r in out),records=out,TRAIN_candidates_SHA256=sha(R/'results/TRAIN_candidates.jsonl'),scope='CPU coordinate reconstruction from immutable prefix/actions only; no policy/critic updates or GPU evaluation'))
print('FP64_MAX',max(r['FP64_error_A'] for r in out),'FP32_REFERENCE_MAX',max(r['default_FP32_reference_error_A'] for r in out))
