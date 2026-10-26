"""Operational R3 status publication; no training/evaluation or model reads."""
import json,runpy,subprocess
from pathlib import Path
R=Path('/mnt/d/Simone/proteinGen_data/e017_v94r3_substantial_geometry_rl_20261026_v1')
C=runpy.run_path(str(R/'source/scripts/e017_v94r2/common.py'))
def main():
 x=C['live']();cfg=C['read'](R/'TRAINING_CONFIGURATION_AND_SEEDS.json');status=x.pop('V94R2_LIVE_STATUS');x['V94R3_LIVE_STATUS']=status
 C['atomic'](R/'V94R3_LIVE_STATUS.json',x)
 body=f"""# E017 V9.4-R3 — production campaign status

State: {status}. UTC: {x['UTC']}.

Committed updates: {x['COMMITTED_UPDATES']}/5000; generation attempts: {x['GENERATION_ATTEMPTS']}/80000; completed: {x['COMPLETED_TRAJECTORIES']}; aborted: {x['ABORTED_ATTEMPTS']}.

Original pre-V9.4 actor SHA256: {cfg['parent_SHA256']}. Frozen J1 SHA256: {cfg['J1_SHA256']}. Fresh deterministic seed: {cfg['seed']}. Qualified R3 packing and exact one-step shield, eight-proposal budget, full rejected/accepted trace estimator, reward, physical supports, hyperparameters and original safety rules are unchanged. Fixed amino acids; geometry actor only. No native targets, TEST, SAC or high-level learning.

Production source and binding: PRODUCTION_SOURCE_MANIFEST.json. Relocation release: qualification/RELEASE.json. Fresh full-state recovery receipt: ACTUAL_INITIAL_RECOVERY_ROUNDTRIP.json. Immutable counters and traces: production/commits and production/rollouts. Checkpoint: production/RECOVERY_INDEX.json. Fixed-development results: production/diagnostics. Curves: RL_TRAINING_CURVES.csv. Prospective outcomes, only after qualified full-budget completion: FINAL_PROSPECTIVE_COMPARISON.json.

No scientific performance conclusion follows from early TRAIN reward. Raw actor and filtered-controller development comparisons remain separate. Aborts are failures. Any safety stop remains binding and has no automatic retry. All artifacts remain on D:. The external read-only observer reports to monitoring/MONITOR_LOG.txt, with no proactive chat delivery and no survival across WSL termination or Windows shutdown.
"""
 p=R/'FINAL_E017_V94R3_SCIENTIFIC_REPORT.md';tmp=p.with_suffix('.md.part');tmp.write_text(body);tmp.replace(p)
if __name__=='__main__':main()
