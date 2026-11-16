"""Seal final-only controls while all trained arms are still unstarted."""
from pathlib import Path
import hashlib
import json
import shutil
import time
R=Path('/mnt/d/Simone/proteinGen_data/e017_v7_amortized_lowlevel_20261023_v1')
assert not (R/'results/B_updates.jsonl').exists() and not (R/'CRITIC_CALIBRATION.json').exists()
p=R/'analysis_source';p.mkdir(exist_ok=True)
shutil.copyfile(Path(__file__).with_name('v7_causal_staged.py'),p/'causal.py')
with (R/'AUXILIARY_CONTROL_PROTOCOL.json').open('x') as f:
 json.dump(dict(sealed_UTC=time.time(),script_SHA256=hashlib.sha256((p/'causal.py').read_bytes()).hexdigest(),
 source='analysis_source/causal.py; additive final-only control, primary sealed training protocol unchanged',
 goal_control='all32 confirmation identities, native/alternate same prefix and shared seed ns(identity,i,MATCHED_GOAL_DIAGNOSTIC), i0/1',
 early_control='first16 confirmation identities hash-ranked rank(causal+identity), native goals, same2seeds; compare parent/learned first latent action with recipient-policy continuation',
 boundaries='after all selection/training finishes; no training or checkpoint selection from these outcomes; no threshold changes',
 registered_before_any_trained_arm=True),f,indent=2)
