from pathlib import Path
import ast,hashlib,json,shutil,subprocess
R=Path('/mnt/d/Simone/proteinGen_data/e017_v9_property_steering_20261023_v1');files=[('v9_analysis_common_staged.py','analysis_common.py'),('v9_final_analysis_staged.py','final_analysis.py'),('v9_confirmation_staged.py','confirmation.py')]
for src,name in files:
 p=Path(src);ast.parse(p.read_text());shutil.copyfile(p,R/'source/scripts/e017_v9'/name)
loaded=subprocess.check_output(['git','-C',str(R/'source'),'show','b71d441:scripts/e017_v9/train.py']);actual=(R/'source/scripts/e017_v9/train.py').read_bytes();assert actual==loaded
(R/'PRODUCTION_IMPLEMENTATION.json').write_text(json.dumps(dict(executed_training_source_commit='b71d441',train_py_SHA256=hashlib.sha256(loaded).hexdigest(),initial_scientific_protocol_source='b838e60',changes_before_production='Added fail-closed resume, graceful pause, resource-stop recovery, and cross-arm common candidate streams. No production updates preceded these fixes.',policy_py_SHA256=hashlib.sha256((R/'source/scripts/e017_v9/policy.py').read_bytes()).hexdigest(),checkpoint_context='Frozen scientific protocol source identifies declaration; this execution record identifies actual loaded implementation. Neither should be conflated.'),indent=2))
subprocess.run(['git','-C',str(R/'source'),'add','scripts/e017_v9'],check=True);subprocess.run(['git','-C',str(R/'source'),'commit','-m','Add matched confirmation and final scientific analysis with uncertainty and source provenance'],check=True)
