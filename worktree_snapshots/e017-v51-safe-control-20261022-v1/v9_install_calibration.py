from pathlib import Path
import ast,hashlib,json,shutil,subprocess
R=Path('/mnt/d/Simone/proteinGen_data/e017_v9_property_steering_20261023_v1');here=Path(__file__).resolve().parent;target=R/'source/scripts/e017_v9';target.mkdir(exist_ok=True)
assert not (R/'SCIENTIFIC_DESIGN_INITIAL.json').exists()
for src,name in [('v9_core_staged.py','core.py'),('v9_calibrate_staged.py','calibrate.py')]:
 p=here/src;ast.parse(p.read_text());shutil.copyfile(p,target/name)
subprocess.run(['git','-C',str(R/'source'),'add','scripts/e017_v9'],check=True);subprocess.run(['git','-C',str(R/'source'),'commit','-m','Declare V9 coordinate-free regional exposure benchmark and staged admission gates'],check=True)
code="import runpy,subprocess;from pathlib import Path;r=Path("+repr(str(R))+");c=runpy.run_path(str(r/'source/scripts/e017_v9/core.py'));c['save']('SCIENTIFIC_DESIGN_INITIAL.json',dict(config=c['DESIGN'],source=subprocess.check_output(['git','-C',str(r/'source'),'rev-parse','HEAD'],text=True).strip(),proxy_constant_rank='undefined within-protein rank contributes0conservatively toadmission; never silentlydiscarded'))"
subprocess.run([str(R/'run.sh'),'-c',code],check=True)
log=R/f'logs/calibration_{__import__("time").time_ns()}.log'
with log.open('x') as f:p=subprocess.Popen([str(R/'run.sh'),str(target/'calibrate.py'),'prepare'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,log=str(log),scope='CPU-only TRAIN/VALIDATION reference preparation; no TEST or policy generation')))
