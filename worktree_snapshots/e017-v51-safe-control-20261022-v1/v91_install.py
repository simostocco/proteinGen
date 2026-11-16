from pathlib import Path
import ast,hashlib,json,shutil,subprocess
R=Path('/mnt/d/Simone/proteinGen_data/e017_v91_ordered_goal_recovery_20261023_v1');T=R/'source/scripts/e017_v91'
for src,name in [('v91_goals_staged.py','goals.py'),('v91_policy_staged.py','policy.py')]:
 p=Path(src);ast.parse(p.read_text());shutil.copyfile(p,T/name)
subprocess.run(['git','-C',str(R/'source'),'add','scripts/e017_v91'],check=True);subprocess.run(['git','-C',str(R/'source'),'commit','-m','Add canonical ordered-goal schema and role-aware conditioning without architecture changes'],check=True)
print(R)
