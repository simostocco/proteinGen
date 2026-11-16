from pathlib import Path
import ast,shutil,subprocess,sys
R=Path('/mnt/d/Simone/proteinGen_data/e017_v72_counterfactual_ranking_20261023_v1');here=Path(__file__).resolve().parent
assert not (R/'PROTOCOL.json').exists()
target=R/'source/scripts/e017_v72';target.mkdir(exist_ok=True)
for src,name in [('v72_study_staged.py','study.py'),('v72_analyze_staged.py','analyze.py')]:
 p=here/src;ast.parse(p.read_text());shutil.copyfile(p,target/name)
if sys.argv[1]=='seal':
 subprocess.run(['git','-C',str(R/'source'),'add','scripts/e017_v72'],check=True);subprocess.run(['git','-C',str(R/'source'),'commit','-m','Seal V7.2 independent-bank counterfactual action-value ranking protocol'],check=True)
subprocess.run([str(R/'run.sh'),str(target/'study.py'),sys.argv[1]],check=True)
