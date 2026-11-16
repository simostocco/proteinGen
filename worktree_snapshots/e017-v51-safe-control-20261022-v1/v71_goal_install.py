from pathlib import Path
import shutil
import hashlib
import json
import time
R=Path('/mnt/d/Simone/proteinGen_data/e017_v71_expected_return_calibration_20261023_v1')
assert not (R/'RWR_MONITOR_COMPLETED.json').exists()
p=Path(__file__).with_name('v71_rwr_goal_eval.py');shutil.copyfile(p,R/'analysis_source'/p.name)
with (R/'AUXILIARY_RWR_GOAL_PROTOCOL.json').open('x') as f:json.dump(dict(UTC=time.time(),script_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),scope='final-only64matched-noise goal pairs under frozen B250; same V7streams as existing parent/B25 goal diagnostic; no optimization or model selection',checkpoint='V7/B_00250_stop.pt remains scientifically stopped'),f,indent=2)
