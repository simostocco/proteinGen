from pathlib import Path
import collections,json,hashlib,subprocess,shutil,os
r=Path('/mnt/d/Simone/proteinGen_data/e017_v92_prefix_feasibility_20261024_v1');old=r.parent/'e017_v91_ordered_goal_recovery_20261023_v1';read=lambda p:json.loads(p.read_text())
a=read(r/'COMPUTE_AND_STORAGE_ACCOUNTING.json');x=read(r/'FILTERING_NUMERICAL_RESULTS.json');fp=read(r/'FAILURE_POINT_ALTERNATIVES.json')['records'];dev=[z for p in sorted((old/'results').glob('DEV_C_FIXED_[0-9][0-9][0-9].json')) for z in read(p)['records']];assert len(dev)==64 and sum(z['metrics']['hard_valid'] for z in dev)==30
# Correct counter naming: finite-budget failed attempts are not complete proteins.
a['counts']['new_generation_attempts']=a['counts'].pop('new_complete_generation_candidates');a['counts']['new_complete_generated_candidates']=sum(v['marginals']['complete'] for k,v in x['results'].items() if k!='CONFIRM_C_U');a['counts']['finite_budget_aborted_candidates']=a['counts']['new_generation_attempts']-a['counts']['new_complete_generated_candidates'];assert a['counts']['finite_budget_aborted_candidates']==4
a['counts']['future_continuation_residue_steps']=sum(64-z['t']-1 for z in read(r/'FUTURE_VIABILITY.json')['records']);a['counts']['future_first_action_proposals']=sum(z['actions_considered'] for z in read(r/'FUTURE_VIABILITY.json')['prefixes']);a['evidence_source_commit']=a['source_commit'];a['all_research_workers_exited']='pending OS audit';(r/'COMPUTE_AND_STORAGE_ACCOUNTING.json').write_text(json.dumps(a,indent=2)+'\n')
s=r/'source/scripts/e017_v92/analyze.py';text=s.read_text().replace('new_complete_generation_candidates=','new_generation_attempts=');s.write_text(text)
subprocess.run(['git','add','scripts/e017_v92/analyze.py'],cwd=r/'source',check=True);subprocess.run(['git','commit','-m','Name finite-budget generation attempts separately from complete proteins'],cwd=r/'source',check=True);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r/'source',text=True).strip()
C=x['results']['CONFIRM_C_U'];V=x['results']['CONFIRM_C_V'];M=x['results']['CONFIRM_C_M'];VM=x['results']['CONFIRM_C_VM'];dd=x['paired_differences'];d=read(r/'COMPUTE_AND_STORAGE_ACCOUNTING.json')['diversity_gate_measurements']
summary=f'''

## Final decision and verification handoff

**Outcomes A and B are supported. No new RL campaign is launched or admitted by this report.** Immediate action feasibility is a large avoidable loss source; regional property control remains a separate limiting conjunction. Outcome C's predefined large-shift criterion is not met in this small native/generated-prefix probe. Outcome D is not supported as an absolute lack of feasible support: controlled de novo sampling finds full successes without native targets. This does not establish a sufficient population for reliable unrestricted RL or learned instruction following.

For C, ordinary→validity-only hard completion is109/256→232/256; backbone117/256→255/256. The one incomplete candidate exhausted its8-proposal budget and remains a failure. Motif-only full success is6/256, K1=1/32, K8=6/32. Combined full success is17/256, K1=3/32, K8=14/32. K1 exact95% interval is{VM['K1']['CI95']}; paired K1 gain95% interval is{dd['C_VM']['K1_goal']['CI95']}, including zero. K8 paired gain95% interval is{dd['C_VM']['K8_goal']['CI95']}. All pooled candidate estimates use goal/identity-clustered comparisons, not256 independent proteins.

Validity-only rejected301/16,667 proposals; combined rejected311/16,695. Every proposal, finite-budget abort and accepted transition is counted. AA pairwise identity changed by{d['AA_identity_change']:.6f}; CA pairwise Kabsch RMSD retained{d['CA_RMSD_ratio']:.4f} of baseline; unique sequence fraction is1.0. Diversity comparisons condition on complete chains and retain failed candidates in success denominators. Neither accepted filtered likelihood nor KL is claimed equal to the original proposal law.

The original30/64→26/64 stop comparison is **C_FIXED DEV8 vs D100 DEV8**, not the historical32-goal final comparison. The latter used D50. New matched D100 ordinary sampling obtains1/256 full success (K1=1/32, K8=1/32); D100 combined obtains14/256 (K1=2/32, K8=10/32). This does not reverse its validity stop or qualify D100 for promotion.

At the48 retrospectively selected TRAIN failure points,601/768 alternative joint proposals preserve immediate prefix validity. This failure-conditioned rate is deliberately separated from the near-99% valid rates at ordinary surviving fixed-stage prefixes. It shows viable alternatives often exist precisely where recorded failures occurred, without estimating eventual goal reachability or an unbiased population hazard.

The next smallest scientifically justified intervention is a separately authorized, narrow regional-property controllability study with frozen deterministic motif/irreversible-validity controls and a new prospective goal panel. Before any subsequent low-level policy-gradient pilot, derive the actual conditional/finite-rejection trace likelihood and KL, establish reproducible full-goal support and admit complete checkpoint staging above2GiB headroom. Keep J1 frozen and retain originalN64/reward/tolerances. Do not use the original proposal log-density as the accepted filtered-policy density. High-level planning and SAC remain NO-GO.

Numerical-analysis execution commit: `{a['source_commit']}`. Final source publication commit: `{head}` (subsequent change only corrects an accounting-field label). Checkpoint/source/file integrity and final idle-system receipts are in HISTORICAL_AND_FINAL_VERIFICATION.json and PROVENANCE_SHA256.json. No historical evidence or model file was deleted.
'''
for name in ['FINAL_E017_V92_SCIENTIFIC_REPORT.md','NEXT_CRISP_RESEARCH_DECISION.md']:
 p=r/name;p.write_text(p.read_text()+summary)
p=r/'PREFIX_VALIDITY_FAILURE_AUDIT.md';p.write_text(p.read_text()+'\n\nGoal-order-stratified per-residue hazards are saved in HAZARD_BY_GOAL_TYPE.json at sealing. A finite-budget filter abort is recorded separately from an accepted irreversible violation; aborts never disappear from completion/goal denominators.\n')
p=r/'COMPUTE_AND_STORAGE_ACCOUNTING.json';a=read(p);a['final_publication_source_commit']=head;a['final_source_status']=subprocess.check_output(['git','status','--porcelain'],cwd=r/'source',text=True);p.write_text(json.dumps(a,indent=2)+'\n')
(r/'control/MONITOR_DONE').write_text('V92 GPU diagnostics completed; stop resource observer.\n')
print('REPORTS_REVIEWED',head,shutil.disk_usage(r).free)
