from pathlib import Path
import ast,shutil,subprocess,sys
R=Path('/mnt/d/Simone/proteinGen_data/e017_v8_constrained_planning_20261023_v1');here=Path(__file__).resolve().parent;target=R/'source/scripts/e017_v8';target.mkdir(exist_ok=True)
assert not (R/'PROTOCOL.json').exists()
for src,name in [('v8_study_staged.py','study.py'),('v8_analyze_staged.py','analyze.py')]:
 p=here/src
 if p.exists():ast.parse(p.read_text());shutil.copyfile(p,target/name)
if sys.argv[1]=='seal':
 assert (target/'analyze.py').exists();subprocess.run(['git','-C',str(R/'source'),'add','scripts/e017_v8'],check=True);subprocess.run(['git','-C',str(R/'source'),'commit','-m','Seal V8 frozen-parent constrained planning and matched computation study'],check=True)
subprocess.run([str(R/'run.sh'),str(target/'study.py'),sys.argv[1]],check=True)
