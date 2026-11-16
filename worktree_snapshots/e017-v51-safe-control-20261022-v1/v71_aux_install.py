from pathlib import Path
import hashlib
import json
import shutil
import time
R=Path('/mnt/d/Simone/proteinGen_data/e017_v71_expected_return_calibration_20261023_v1')
assert not (R/'checkpoints/INDEX.json').exists()
p=R/'analysis_source';p.mkdir(exist_ok=True);src=Path(__file__).with_name('v71_realized_eval.py');shutil.copyfile(src,p/src.name)
with (R/'AUXILIARY_ASSESSMENT_PROTOCOL.json').open('x') as f:json.dump(dict(UTC=time.time(),script_SHA256=hashlib.sha256(src.read_bytes()).hexdigest(),scope='final-only realized-return confirmation predictions for legacy and DEV-selected critics, all2048transitions from64TRAINconfirmation identities; no model updates; no effect on primary selection/gates',binary_analysis='Qscore AUROC/AUPRC discrimination only; shapedQ is not a success probability',registered_before_new_critic_training=True),f,indent=2)
