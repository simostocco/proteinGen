"""Install the CPU-only analysis in the isolated V6 worktree and publish."""
from pathlib import Path
import json
import shutil
import subprocess

BASE=Path('/mnt/d/Simone/proteinGen_data/e017_v51_safe_control_20261022_v1/source')
R=BASE.parent.parent/'e017_v6_frozen_parent_candidates_20261023_v1'
assert json.loads((R/'WORKER_COMPLETED.json').read_text())['complete']
p=R/'source/scripts/e017_v6/analyze.py'
if p.exists():
 archive=R/('analysis_versions/v1_1' if (R/'analysis_versions/v1').exists() else 'analysis_versions/v1');archive.mkdir(parents=True,exist_ok=False)
 for name in ['FINAL_E017_V6_SCIENTIFIC_REPORT.md','CANDIDATE_SUCCESS_BY_K.json','VALIDITY_VS_GOAL_ANALYSIS.md','CANDIDATE_SELECTION_COMPARISON.md','INFERENCE_COST_ANALYSIS.md','NEXT_RL_RESEARCH_DECISION.md','PROVENANCE_SHA256.json','ANALYSIS_INTERMEDIATE.json']:
  shutil.copyfile(R/name,archive/name)
shutil.copyfile(BASE/'v6_analysis_staged.py',p)
subprocess.run(['git','-C',str(R/'source'),'add','scripts/e017_v6/analyze.py'],check=True)
subprocess.run(['git','-C',str(R/'source'),'commit','-m','Analyze frozen V6 candidate pools and publish scientific assessment'],check=True)
subprocess.run([str(R/'run.sh'),str(p)],check=True)
