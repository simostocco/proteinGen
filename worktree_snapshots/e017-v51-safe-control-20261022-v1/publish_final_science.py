"""Publish seven V5.1 reports from completed CPU analyses. No research execution."""
from __future__ import annotations

import collections
import datetime
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np

from final_scientific_analysis import R, V4, V5, E10, VERSION, SEED, REPS, INPUTS, read, sha, write, paired

OUTPUTS = ['FINAL_E017_V51_SCIENTIFIC_REPORT.md', 'MATCHED_ARM_FINAL_COMPARISON.json',
           'KL_AND_VALIDITY_FAILURE_ANALYSIS.md', 'FINAL_CHECKPOINT_VERIFICATION.json',
           'NEXT_RESEARCH_DECISION.md', 'E010_STORAGE_REVIEW.md', 'PROVENANCE_SHA256.json']


def fmt(x):
    return f'{x:.6f}'


def ci(x, scale=1, field='CI95'):
    a,b=x[field]
    return f"{x['delta']*scale:.4f} [{a*scale:.4f}, {b*scale:.4f}]"


def mdtable(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def main():
    assert all(not (R/n).exists() for n in OUTPUTS), 'Refuse to overwrite an existing final version'
    d=read(R/'source/final_analysis_work.json')
    ck=read(R/'source/final_checkpoint_work.json')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert d['source_commit']==source
    evals=d['evaluations']
    raw={n:read(R/f'results/{n}.json') for n in evals}
    for p in (R/'results').glob('*.json'):
        read(p)
    for p in (R/'manifests').glob('*.json'):
        read(p)
    for p in (R/'results').glob('*_updates.jsonl'):
        INPUTS[str(p)]=sha(p)
    for p in (R/'source/scripts/e017_v51').iterdir():
        if p.is_file():
            INPUTS[str(p)]=sha(p)
    for p in [R/'source/src/protein_joint_sac/crisp_v3/environment.py',R/'source/src/protein_joint_sac/crisp_v3/recovery.py',
              R/'source/scripts/e017_v5/study.py',R/'source/scripts/e017_v4/experiment.py',
              V4/'FINAL_E017_V4_MULTISTEP_REPORT.md',V4/'HELDOUT_COMPARISONS.json',
              V5/'FINAL_E017_V5_SCIENTIFIC_REPORT.md',V5/'MATCHED_ARM_COMPARISONS.json',
              V5/'EXPLORATORY_DIAGNOSTIC_AMP_ERRATUM.md',V5/'C_H4_STOP_FAILURE_DIAGNOSIS.json',
              V5/'CHECKPOINT_AND_RECOVERY_MANIFEST.json', R/'run.sh',
              R/'recovery_storage_20261022/REPORT.md',R/'recovery_storage_20261022/experiment_recovery.json',
              R/'HISTORICAL_H4_COLLISION_AND_KL_ANALYSIS.json',R/'HISTORICAL_CHECKPOINT_AND_MASK_VERIFICATION.json',
              R/'TRAIN_ONLY_GRADIENT_CALIBRATION.json',R/'SEALED_CONFIG.json',R/'INITIAL_AUDIT.json']:
        INPUTS[str(p)]=sha(p)
    for path, expected in read(R/'INPUT_MANIFEST.json').items():
        assert sha(path)==expected, path
        INPUTS[path]=expected
    # Paired factorial contrasts are exploratory. The primary family stays B/C/D versus A.
    for suffix, u in [('AUDIT_H4_native',100),('DEV_H4_native',250)]:
        for a,b in [('C','D'),('B','D')]:
            d['comparisons'][f'{b}-{a}_{u}_{suffix}']=paired(raw[f'{a}_{u:05}_{suffix}']['records'],raw[f'{b}_{u:05}_{suffix}']['records'])
    for name, ev in evals.items():
        rows=raw[name]['records']
        directions=ev['direction']['records']
        byid=collections.defaultdict(list)
        for r in directions:
            byid[r['identity']].append(float(r['correct']))
        if byid:
            a=np.array([np.mean(v) for v in byid.values()])
            rng=np.random.default_rng(SEED)
            sample=a[rng.integers(0,len(a),(REPS,len(a)))].mean(-1)
            ev['direction']['identity_cluster_CI95']=np.quantile(sample,[.025,.975]).tolist()
        # Source already preserves task-level direction records; retain their locator instead of duplicating them.
        del ev['direction']['records']
        ev['direction']['records_source']=ev['source']+'#direction.records'
    # Frozen reference hashes are read on CPU only; no network, forward pass or evaluation.
    refs=read(R/'INITIAL_AUDIT.json')
    protected={
        '/mnt/d/Simone/proteinGen_data/e016_j1_bidirectional_multitask_20261019_v1/runs/J1/checkpoint_07500.pt':'881457ff0396975c1b20dd0a8e2caac8521e65b7e1cec76a7bb531dbc437b412',
        str(V5/'checkpoints/D/checkpoint_00250.pt'):'0b2f0b68783e4234f44c3b1041a5c1f59f04191af79b13232553b511168de265'}
    for p,h in protected.items():
        assert sha(p)==h,p
    ck['protected_reference_hashes_freshly_verified']=protected
    ck['source_commit']=source
    ck['training_resume_command']=None
    ck['reason_no_resume_command']='all four scientific STOP records remain binding; train entrypoint refuses them'
    ck['GPU_ownership_check']='nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader: exit0, no compute owners, 2026-10-23 during finalization'
    ps=subprocess.check_output(['ps','-eo','pid,ppid,comm,args'],text=True)
    workers=[]
    for line in ps.splitlines()[1:]:
        fields=line.split(None,3)
        if len(fields)==4 and fields[2].startswith('python') and any(x in fields[3] for x in ['campaign.py','e017_v51/study.py','ProteinMPNN','colabfold']):
            workers.append(line)
    assert not workers
    ck['active_research_workers']=workers
    ck['scientific_stop_recovery_fix']='all production STOP250 states retained, read back, indexed, marked ineligible before STOP/FINISHED publication; previous V5 C350 missing-state failure addressed'
    # Storage metadata only. Do not open E010 tensor archives, mixed-split arrays or TEST results.
    inv=read(R/'recovery_storage_20261022/storage_inventory.json')
    dirs={x['path']:x for x in inv['directories']}
    e10record=dirs[str(E10/'neighbors')]
    complete=read(E10/'COMPLETED.json')
    freeze=read(E10/'parent_freeze.json')
    protocol=read(E10/'protocol.json')
    extract=read(E10/'EXTRACTION_COMPLETED.json')
    panelmark=read(E10/'PANEL_COMPLETED.json')
    search={n:read(E10/f'neighbors/{n}_SEARCH_COMPLETED.json') for n in ['geometry','latent']}
    dependency=read(E10/'dependency_freeze.json')
    sourcechecks=[]
    for p,h in complete['source_sha256'].items():
        # Source hashes can be rechecked without opening any data population.
        actual=sha(p)
        sourcechecks.append(dict(path=p,expected=h,actual=actual,match=actual==h))
        INPUTS[p]=actual
    canonical={
        '/mnt/d/Simone/proteinGen/outputs/e010_nonprivileged_structural_student_pilot_v1/models/imitation_best.pt':freeze['student_model_sha256'],
        '/mnt/d/Simone/proteinGen/outputs/e010_nonprivileged_structural_student_pilot_v1/splits.json':freeze['splits_sha256'],
        '/mnt/d/Simone/proteinGen/outputs/e010_student_feasibility_margin_v1/COMPLETED.json':freeze['parent_completion_sha256'],
        '/mnt/d/Simone/proteinGen/outputs/e010_structural_teacher_diversity_scaleout_v1/COMPLETED.json':freeze['teacher_completion_sha256']}
    canonicalchecks=[]
    for p,h in canonical.items():
        actual=sha(p) if Path(p).exists() else None
        canonicalchecks.append(dict(path=p,expected=h,actual=actual,match=actual==h))
        if actual:
            INPUTS[p]=actual
    markerchecks=[]
    for p in [E10/'protocol.json',E10/'panel/transforms.json',E10/'PANEL_COMPLETED.json',E10/'EXTRACTION_COMPLETED.json'] + [E10/f'neighbors/{n}_{t}.json' for n in ['geometry','latent'] for t in ['SEARCH_COMPLETED','audit']]:
        actual=sha(p);expected=complete['files'].get(str(p))
        markerchecks.append(dict(path=str(p),expected=expected,actual=actual,match=actual==expected))
        INPUTS[str(p)]=actual
    # One metadata traversal of precisely the reviewed directory; NTFS allocation snapshot is authoritative.
    e10files=list((E10/'neighbors').iterdir())
    e10stats=[(p,p.stat()) for p in e10files]
    assert len(e10stats)==e10record['files'] and sum(st.st_size for _,st in e10stats)==e10record['logical']
    assert all(st.st_nlink==1 for _,st in e10stats)
    assert not any(str(E10/'neighbors') in str(x) for x in inv['hardlinks'])
    large={Path(x['path']).name:x for x in inv['large_files'] if str(E10/'neighbors')+'/' in x['path']}
    dense=[large[n] for n in ['geometry_ids.npy','geometry_distances.npy','latent_ids.npy','latent_distances.npy']]
    indexes=[large[n] for n in ['geometry.faiss','latent.faiss']]
    densebytes=sum(x['allocated'] for x in dense);indexbytes=sum(x['allocated'] for x in indexes)
    e10evidence=dict(directory=e10record,live_metadata_count=len(e10stats),live_nlink_values=sorted({st.st_nlink for _,st in e10stats}),
                     allocation_snapshot_UTC=inv.get('finished_utc'),source_hash_checks=sourcechecks,canonical_checks=canonicalchecks,
                     receipt_checks=markerchecks,panel_completion=panelmark,search_completion=search,
                     raw_dense_allocated_bytes=densebytes,index_allocated_bytes=indexbytes,
                     no_data_deletion=True,scope='metadata, code, package/version and hash receipts; no E010 TEST outputs or mixed-split tensor contents opened')
    d['E010_storage_evidence']=e10evidence
    d['decision']='NO_PROMOTION; all arms stopped scientifically; matched100 confirmation also breaches KL'
    d['TRAIN']['missing_metrics_note']='No fixed TRAIN outcome evaluations exist in V5.1; all geometric errors/hits below are VALIDATION only. TRAIN logs retain component losses and gradients.'
    write(R/'MATCHED_ARM_FINAL_COMPARISON.json',d)
    write(R/'FINAL_CHECKPOINT_VERIFICATION.json',ck)

    report=[f'# E017 V5.1 final scientific report — v1\n\nVersion `{VERSION}`; finalized {datetime.datetime.now(datetime.timezone.utc).isoformat()}.',
    '**Decision: no arm is promoted.** All arms completed250 H4 updates and stopped at their frozen scientific boundary. A/C exceeded the native H2 mean conditional geometry KL limit; B/D lost more than5 percentage points of native H4 backbone validity. Their exact stopped states are preserved. No new training, policy evaluation, SAC, TEST access, architecture/threshold change or cleanup was performed for this analysis.',
    'The independently held-out AUDIT exposes an additional failure: every selected+100 checkpoint exceeds1 nat/component on BOTH native H2 and native H4. Selection was valid under DEV at the time; confirmation rejects generalization of that eligibility. It does not retroactively alter checkpoint selection. All+250 checkpoints remain quarantined recovery evidence.',
    '## Design and provenance',
    f"Sealed source branch `e017-v51-safe-control-20261022-v1`, commit `{source}`. Config hash `{d['sealed']['config_hash']}`. All arms start from V5 D+250 (SHA256`{protected[str(V5/'checkpoints/D/checkpoint_00250.pt')]}`), with fresh AdamW, matched identity-first TRAIN queue/noise, batch4, seed6517. Actor17,438,114 parameters;17,429,134 trainable,8,980 categorical parameters frozen. J1 and all nonactor networks remain immutable; no replay or RL. The full-system parent is bound rather than duplicated in each checkpoint.",
    'A: fixed0.01 KL coefficient. B: adaptive KL multiplier targeting0.8. C: A plus smooth3Å collision margin, weight1.6644497725948837. D: B+C. LR1e-5,25-step warmup, cosine towards1e-6 at500, weight decay0.01, clip5. No training continuation is authorized by retaining a complete checkpoint.',
    '## Panels, denominators and estimands',
    'DEV is reused historical VALIDATION:24 protected identities,46 tasks,92 attempts at each horizon. AUDIT is the last32 V5CONFIRM identities, excluded from V5 scored confirmation: H2=64 tasks/128 attempts; H4=63 tasks/126 attempts. Each task has seeds7101/7102. Targets include native and valid synthetic torsion alternatives; fixed AAs, prefixes20/28,2 or4 generated steps. Success requires position≤8Å AND orientation≤35° AND unchanged Cα fragment validity. These are local geometric outcomes, not full-protein folding/stability results.',
    'Generated-prefix controls replace the last two prefix actions using the original frozen V3 and fixed identity-derived seeds, retain the same target and count every attempt. Feasibility may change. No failed sample is replaced. Exact task IDs, protein identities, protected groups, seeds and source manifest paths are retained in MATCHED_ARM_FINAL_COMPARISON.json panels and original saved records; TRAIN/DEV/AUDIT protected-group separation was rechecked.',
    f'Paired protected-group cluster bootstrap: {REPS:,} replicates, seed{SEED}, resample complete groups with all their task/seed rows. Main descriptive estimand is attempt-weighted (sum/count); equal-group estimates are also retained because one AUDIT H4 identity has only one admitted goal. Nominal95% intervals accompany the predeclared three-arm familywise98.333333% intervals. One training seed and two shared evaluation noise streams limit generalization; CIs do not capture optimization-seed variation. DEV longitudinal analyses and mechanistic contrasts are exploratory.',
    '## Every available geometric outcome',
    'All following outcome rows are VALIDATION. TRAIN absolute position/orientation errors, hits and validity were not logged as fixed panels; scalar goal loss mixes Huber position and rotation and cannot supply them. No evaluation was rerun to fill missing outcomes. Table precision is six decimals; JSON preserves the full recorded floats. INITIAL is shared by A/B/C/D. +250 shifted/AUDIT and +500 evaluations do not exist because stops preceded them. Matched100 inference-only and selected100 AUDIT records were verified identical and counted once.']
    for panel in ['DEV','AUDIT']:
        for H,cond in [(2,'native'),(4,'native'),(4,'shifted')]:
            suffix=f'{panel}_H{H}_{cond}'
            names=[n for n in evals if n.endswith(suffix)]
            names.sort(key=lambda n:(0 if n.startswith('INITIAL') else 1 if n.startswith('FROZEN') else 2,n))
            report.append(f'### {suffix}\n\n'+mdtable(['State','N','Position Å','Rotation °','Translation hits','Rotation hits','Valid','Full success','Collision','KL'],[
                [n.removesuffix('_'+suffix),q['attempts'],fmt(q['position_A']),fmt(q['orientation_deg']),f"{q['translation_hit']}/{q['attempts']}",f"{q['rotation_hit']}/{q['attempts']}",f"{q['valid']}/{q['attempts']}",f"{q['success']}/{q['attempts']}",f"{round(q['collision_fraction']*q['attempts'])}/{q['attempts']}",fmt(q['KL'])]
                for n in names for q in [evals[n]['summary']]]))
    report.extend(['## Primary matched+100 comparison',
    'Endpoint J=valid×exp[−0.5((position/8)²+(orientation/35)²)]. Positive success/J is favorable. Require lower familywise bound>0 plus ≥5pp success OR≥0.03 J, AND all safeguards. None qualifies:',
    mdtable(['Comparison, native H4 AUDIT','Success Δpp [95%]','Success Δpp [98.333%]','J Δ [98.333%]','Position ΔÅ [95%]'],[
        [f'{a}−A',ci(q['success'],100),ci(q['success'],100,'CI98_333333'),ci(q['joint_J'],1,'CI98_333333'),ci(q['position_A'])]
        for a in 'BCD' for q in [d['comparisons'][f'{a}-A_100_AUDIT_H4_native']['endpoints']]]),
    'C’s nominal success interval is positive, but its familywise lower bound is0 and its3.17pp point gain is below5pp. J does not improve. C/D have worse mean position than A on this panel, with nominal paired intervals above0. D improves validity versus A by3.97pp, yet shares the confirmation KL failure. Small differences among failed controllers do not establish a successful controller.',
    'Compared with INITIAL, native H4 AUDIT full success falls from20/126 to A5, B4, C9, D8. Mean position improves while mean rotation worsens from93.065526° to103.54–104.28°. The per-arm full-success deltas with nominal95% intervals are:',
    mdtable(['Arm100−INITIAL','Success Δpp [95%]','H2 success Δpp [95%]'],[[a,ci(d['comparisons'][f'{a}100-INITIAL_AUDIT_H4_native']['endpoints']['success'],100),ci(d['comparisons'][f'{a}100-INITIAL_AUDIT_H2_native']['endpoints']['success'],100)] for a in 'ABCD']),
    'Generated-prefix H4 AUDIT full success declines from14/126 INITIAL to A5/B5/C4/D5; lower positional error is not successful transfer. Full paired position/orientation/hit/validity/KL/J/collision intervals are in the comparison JSON.',
    '## Goal response',
    'Paired alternate-goal directional correctness is dot(predicted endpoint difference, target difference)>0, using identical prefix/noise. The own-minus-cross distance must be interpreted separately: negative favors the intended target. A high direction sign rate can coexist with much worse absolute rotation/full success. No V5.1 saved goal-on/off distribution KL or component-specific policy Jacobians exist.',
    mdtable(['State/panel','Correct/pairs','Rate','Own−cross Å'],[[n,f"{ev['direction']['correct']}/{ev['direction']['pairs']}",fmt(ev['direction']['rate']),fmt(ev['direction']['mean_own_minus_cross_A'])] for n,ev in evals.items()]),
    'Identity-clustered nominal95% direction-rate intervals are retained in JSON. Frozen J1 does not condition on goals, so identical-noise alternate goals produce identical endpoints (zero directional response), despite some chance tolerance hits.',
    '## TRAIN loss and gradient evidence',
    'Each arm has250 contiguous updates,1,000 task presentations/4,000 rollout geometry actions plus1,000 native BC decisions. The queue and per-update task identities match exactly across arms. Distinct identity counts describe training coverage, not independent outcomes.',
    mdtable(['Arm','Distinct identities','Preclip norm mean / max','Clipped','Final KL multiplier'],[[a,t['distinct_identities'],f"{t['gradient_norm']['mean']:.6f} / {t['gradient_norm']['max']:.6f}",f"{t['clipped_updates']}/250",fmt(t['dual_final'])] for a,t in d['TRAIN'].items() if isinstance(t,dict)]),
    'All1,000 optimizer updates across four arms clip at5. Geometry-head gradients dominate shared encoder/residue norms (exact distributions in JSON). This establishes large gradient scaling, not measured negative gradient cosines or a causal explanation. Component-loss windows1–25,76–100,101–125,226–250 are retained for every arm:',
    mdtable(['Arm/window','Goal loss','Local','Native BC','Conditional KL','Smooth collision'],[[f'{a}/{w}',*[fmt(v['terms'][k]) for k in ['goal','local','BC','KL','collision']]] for a,t in d['TRAIN'].items() if isinstance(t,dict) for w,v in t['windows'].items()]),
    'TRAIN losses use arm-dependent coefficients (B/D adaptive multiplier, C/D extra collision); totals are not comparable as a single fixed scoring rule. Negative continuous-density BC NLL is possible. Missing TRAIN outcome panels remain a monitoring limitation.',
    '## Safety mechanism and historical comparison',
    'See KL_AND_VALIDITY_FAILURE_ANALYSIS.md for recorded per-step KL, exact first-invalid steps and paired component contrasts. In A/C the actual+250 trigger is H2 KL (1.296614/1.258109), while H4 means remain0.918748/0.957348. Divergence is horizon/state dependent, not monotonic accumulation in every rollout. No distance/angle/torsion KL decomposition was saved.',
    'V4 (commit d8177be05f900960e89979d1e2867ba9945080a0, H8,32 different VALIDATION identities/64 attempts) observed C500 position16.8549Å versus B17.9553Å, paired95% CI for difference[−2.1938,0.1245], full success1/64 versus3/64. Its longer-horizon inference and differing goals preclude pooling with V5.1. V4 already showed endpoint improvement without reliable full success. Original V4 mean-decoding diagnostics had an AMP cache flaw; use the V5 erratum rather than treating the stale near-identical policy means as evidence.',
    'V5 (commit f1bf4a58ebd3d32741887fb7a57e3745c7236678) established goal-relative H2 conditioning: C−B independent-confirmation success+12.5pp,95%CI[7.8125,17.96875], position−1.1158Å CI[−1.4907,−0.7612]. C54/128,D55/128 versus original frozen-V3 A39/128; D’s H2 weights are V5.1 INITIAL. These V5 confirmation identities are disjoint from V5.1 AUDIT, so cross-study scores are descriptive, not paired.',
    'On the new matched AUDIT, INITIAL retains useful H2 conditioning versus frozen J1:48/128 versus35/128 successes, paired+10.15625pp CI[4.6875,16.40625]; mean position4.571288 versus5.780953Å. After V5.1+100, H2 successes45/128 for A/B and46/128 for C/D, direction response remains high but KL rises above1 for every arm. Thus useful H2 representation behavior survives qualitatively; an admissible new H4 controller does not.',
    'V5 C+350 H4 stopped with position7.088836Å, orientation95.302105°, success12/92, validity66/92, KL2.307407; all26 invalid attempts were nonlocal clashes. Its exact+350 weights were not saved. V5.1 changes the failure balance: dual control lowers DEV KL yet leaves more clashes; smooth collision loss is insufficient. This repeats a constraint-conflict pattern, without proving goal feasibility inside the stochastic KL region or identifying a gradient-conflict mechanism.',
    'Original frozen V3 (full-parent SHA256 ccd11e9788a30a4f3c398b57ee655d24f3e4b636fec875469f080b711ee3e8e9) is a historical control and the generated-prefix producer, not V5.1 INITIAL. No fresh V3 evaluation was run. Frozen J1’s native H4 AUDIT19/126 versus INITIAL20/126 has paired success difference+0.79365pp CI[−2.34375,4.76190]; chance full hits do not imply goal sensitivity.',
    '## Stop/recovery audit',
    'All eight latest/selected checkpoints were freshly SHA256-verified, CPU-deserialized and complete semantic-digest verified. Adam states and scheduler counters agree with100/250; sampler cursors400/1000; Python/NumPy/CPU-Torch/CUDA RNG retained. BF16 uses no GradScaler. Exact next identity/task, config, immutable parents and all checkpoint hashes are in FINAL_CHECKPOINT_VERIFICATION.json. No new replay step was executed. Original engineering evidence includes fresh-process exact-next-update replay and production-control-flow fault injection.',
    mdtable(['Arm','Stop update / reason','Latest full recovery SHA256','Selected100 SHA256'],[[a,f"250 / {ck['arms'][a]['stop']['reason']}",ck['arms'][a]['states']['latest']['record']['sha256'],ck['arms'][a]['states']['best']['record']['sha256']] for a in 'ABCD']),
    'Checkpoint write+fsync, full readback/digest, atomic rename/index precede STOP, and STOP precedes FINISHED. File timestamps agree, all250 logs are contiguous, and there are no partial checkpoint files. The V5 missing-final-state failure is addressed in actual V5.1 stop artifacts. Numerical resumability is preserved, but every stopped arm is scientifically blocked and the entrypoint refuses automatic continuation.',
    '## Resource, storage and next decision',
    'Current GPU ownership query returned no compute process; no research trainer/supervisor remains. LIVE_STATUS is stale and preserved. The original resource-observer JSONL is absent, a telemetry gap; do not fabricate historical utilization. E010 data were inspected by metadata/code/hash receipts only and were not changed. D: free space at publication is recorded exactly in E010_STORAGE_REVIEW.md and PROVENANCE_SHA256.json; no cleanup occurred.',
    'E017 V6 recommendation: a frozen-parent H4 validity-aware finite-candidate selection diagnostic, testing one hypothesis—whether valid goal-reaching continuations are already available in the learned proposal distribution but soft optimization/ranking sacrifices them. See NEXT_RESEARCH_DECISION.md for comparison of the five alternatives and the selected-law KL limitation. No new training/evaluation was launched.',
    '## Reused evidence and limitations',
    '[V4 matched learning curves](../e017_v4_multistep_20261022_v1/MATCHED_ARM_LEARNING_CURVES.png); [V5 learning curves](../e017_v5_goal_observability_20261022_v1/TRAIN_VALIDATION_LEARNING_CURVES.png); [V5 validity/KL](../e017_v5_goal_observability_20261022_v1/BACKBONE_VALIDITY_AND_KL.png). Their populations/horizons are labeled historical; no plot is relabeled as V5.1. Exact V5.1 learning curves are the tabulated saved milestones and original JSON, with no duplicated tensor/plot exports.',
    'Limits: only one training seed; reused DEV;32 confirmation groups with shared seeds; no+250 independent confirmation or shifted output; missing TRAIN outcome panels; no per-component Gaussian statistics or gradient-conflict cosines; generated-prefix target feasibility can change; Cα criteria are not biochemical validity or stability. Missing measurements were not replaced by hypotheses. All historical E016/E017/J1 evidence remains unchanged.'
    ])
    write(R/'FINAL_E017_V51_SCIENTIFIC_REPORT.md','\n\n'.join(report))

    failure=[f'# KL and validity failure analysis — v1\n\n`{VERSION}`; saved-output analysis only.',
    '## What is measured',
    'All41 unique completed evaluation records were checked against their exact frozen task/seed sets. Saved prefix+tail coordinates were reassembled on CPU with NumPy; the unchanged finite/spacing/nonlocal-clash/Rg/frame criteria were applied. Every reconstructed final validity label matched the original saved label. This is a deterministic analysis of stored coordinates, not a new actor evaluation. All measured cause counts, identities, seeds and collision pairs are in MATCHED_ARM_FINAL_COMPARISON.json failure_analysis.',
    'Valid bounds: adjacent2.8504200470168195–4.052024993913842Å; nonlocal index difference≥3, contact<2Å; clashes/100≤2; Rg/N^(1/3) within1.950656875000882–10.184757892570111; all frame cross products>1e−8; finite. The evaluated chains have22/24 or30/32 residues, so a single clash exceeds the rate limit. There is no frozen torsion-range validity test apart from frame nondegeneracy. Torsion can influence clashes, but a distinctive torsion causal mechanism was not recorded.',
    '## A/C: horizon-specific KL violations',
    'The Gaussian conditional KL is averaged over latent distance/angle/torsion, then over rollout steps, then attempts. Shared invertible action transforms cancel in the conditional KL. It is fixed-AA geometry KL, not categorical joint KL or a guarantee for every trajectory. Critically, A/C stop at250 due to H2 meanKL>1, not their H4 average.',
    mdtable(['Arm/update','H2 KL','H4 KL','H2 step means','H4 step means'],[[f'{a}/{u}',fmt(evals[f'{a}_{u:05}_DEV_H2_native']['summary']['KL']),fmt(evals[f'{a}_{u:05}_DEV_H4_native']['summary']['KL']),', '.join(fmt(v) for v in evals[f'{a}_{u:05}_DEV_H2_native']['failure_analysis']['KL_by_step_mean']),', '.join(fmt(v) for v in evals[f'{a}_{u:05}_DEV_H4_native']['failure_analysis']['KL_by_step_mean'])] for a in 'ABCD' for u in [100,250]]),
    'A/C H2 divergence grows at both steps, with step2 largest at250. H4 A/C step1 is largest and can exceed1 although the horizon average passes. B/D at250 suppress earlier-step KL but step4 remains their highest H4 mean. These are state/horizon-specific changes; no universal monotonic-in-step divergence is supported. Every selected100 native AUDIT H2/H4 mean exceeds1, so TRAIN dual feedback and DEV eligibility did not certify held-out KL.',
    '**Action-component attribution is unavailable.** The source records only the mean of the three Gaussian components at each step; it discards individual mean/log-standard-deviation pairs. Saved coordinates identify realized actions, not their distributions. No distance/angle/torsion ranking of KL contributions is scientifically recoverable without a new policy forward pass, which was not run. HISTORICAL_H4_COLLISION_AND_KL_ANALYSIS.json mean_component_sensitivity refers to V5 C350 kinematic sensitivity, not V5.1 per-component KL. It must not be presented as such.',
    'At250, A/C cross-sectional KL–position Pearson correlations are negative: H2−0.453222/−0.479962, H4−0.133876/−0.254207 (larger KL accompanies lower positional error across attempts). However paired change-in-KL versus position gain from INITIAL is A H4−0.087458 and C H4−0.146782, and H2−0.070102/+0.009328. Thus improved mean precision coincides with increased distribution shift across milestones, but individual paired precision gains do not show a consistent positive divergence-gain association. Neither correlation is a causal tradeoff estimate; repeated tasks/seeds are not independent population samples.',
    '## B/D: physical failure subtype and first invalid step',
    mdtable(['Arm/state','Invalid /92','Final causes','First-invalid appended step: count','New colliding pair types'],[[f'{a}/{u}',f"{f['invalid']}/92",str(f['invalid_causes']),str(f['first_invalid_step_counts']),str(f['new_collision_pair_types'])] for a in 'ABCD' for u in [100,250] for f in [evals[f'{a}_{u:05}_DEV_H4_native']['failure_analysis']]]),
    'At B250 all11 invalid attempts fail nonlocal clash limits; first failures: step2 one,step3 four,step4 six. Of14 newly colliding pairs,13 contact the prefix and1 contacts an earlier generated residue. At D250 all10 invalid attempts fail nonlocal clash limits; first failures: step3 three,step4 seven;10 prefix pairs and1 generated pair. No B/D250 H4 invalidity is caused by spacing, Rg, nonfinite coordinates or degenerate frames. B additionally has one H2 clash at250; D H2 remains92/92valid.',
    'All DEV native H4+250 invalidity across A/B/C/D is clash-only (8/11/9/10). INITIAL has5/92. B/D therefore lose6/92=6.521739pp and5/92=5.434783pp validity, crossing the strict>5pp stop. C loses4/92, still within allowance, but fails H2 KL. A loses3/92 but also fails H2 KL. Final causes are assessed on the whole prefix+tail, while the logged adjacent_min/max are generated actions only.',
    'First-invalid analysis tests the same numerical fragment criteria after each appended residue, not a new training gate. Step0 is explicitly retained if a shifted prefix already fails. For native DEV stopped B/D the first invalid steps equal first new clashes. Generated-prefix controls can contain inherited failure; all41 panels preserve those details separately. Causes are not inferred from an aggregate collision fraction.',
    '## Did the mechanisms help?',
    'The adaptive multiplier did change the intended distribution behavior: B250 H4 KL0.614344 versus A0.918748 and D0.594797 versus C0.957348; H2 also falls below1. Final dual multipliers are B0.207303,D0.337449 versus fixed0.01. H4 precision/full-success point estimates improve B6/92,D7/92 versus A1/92,C3/92 at250, but the validity safeguard fails. This is observed redistribution between objectives, not an admissible success.',
    'Collision margin3Å weight1.66445 adds gradients before the original2Å collision. Yet C100 DEV has9/92 clashes versus A7/92, and D1006/92 versus B8/92. At250 C9 versus A8 and D10 versus B11. The effect is neither consistent nor enough to prevent stops. At matched100 independent AUDIT C/D have fewer clashes than A and D−A validity+3.968254pp,95%CI[0.793651,7.377049], while full success remains low, position worsens and KL fails. These mixed effects reject a blanket claim that the mechanism never changes behavior; they also reject robust safety efficacy.',
    'Six TRAIN calibration tasks spanned only two identities with one active precontact probe. That calibrates scale locally, not coverage of the actor’s evolving contact states. Hard validity is discontinuous while squared shallow overlap can be cheap compared with terminal goal loss. This objective mismatch is consistent with the findings; gradient cancellation, inadequate action support or intrinsic unreachability remain hypotheses. Clipping100% of updates provides no component-gradient alignment measurement.',
    '## Feasibility interpretation',
    'Native/synthetic tasks have retained valid kinematic witnesses, so the geometric goals are not generally achievable only by invalid chains. Neither a valid witness nor older IK solves establishes reachability inside the current actor’s stochastic prior-KL region. V5.1 does not show that the actor must leave physical action space to achieve every goal: many valid trajectories exist, while optimized policies trade hard criteria for continuous error gains. Test finite-budget valid proposal availability before adding SAC or a learned critic. No checkpoint violating a frozen safeguard is promoted.'
    ]
    write(R/'KL_AND_VALIDITY_FAILURE_ANALYSIS.md','\n\n'.join(failure))

    decision=f'''# E017 V6 research decision — v1

Version `{VERSION}`. Proposal only; no implementation, optimizer update or evaluation authorized/executed here.

**Recommend one isolated intervention: explicit validity-aware finite-candidate selection with the frozen V5 D+250 parent.** Main hypothesis: its H4 proposal distribution already contains useful valid goal-reaching continuations, but choosing/optimizing terminal geometric error with soft penalties sacrifices these in favor of colliding or poorly oriented trajectories. This is a test of proposal availability and selection, not a claim that sampling cures the learned policy.

## Why this experiment

V5.1 dual control successfully lowers DEV KL yet B/D stop for new nonlocal clashes, predominantly prefix contacts at steps3–4. The smooth collision penalty has inconsistent effects and does not preserve full success. Matched+100 AUDIT success is only4–9/126 versus parent20/126, and every arm violates KL. Native targets retain valid witnesses, so physical impossibility is not established. A frozen proposal diagnostic separates available valid behavior from failures of optimization without introducing a critic, retraining representation or mixing several learning interventions.

## Proposed single-factor design, to freeze before a separately authorized run

- Freeze INITIAL (V5 D+250, SHA256`{protected[str(V5/'checkpoints/D/checkpoint_00250.pt')]}`), J1, action transforms, AAs, goals and all existing physical tolerances. Do not use stopped250 or unsafe100 policies as a promoted parent. Parent AUDIT native H2/H4 KL is0.678509/0.336207.
- Native H4 is primary; retain H2 and common generated-prefix panels as separately labeled controls. Use previously unscored protected VALIDATION identities where possible, excluding every scored V4/V5/V5.1 panel and all TRAIN groups. Verify availability before committing the protocol; if only reused DEV is available, label the study exploratory. No TEST access. Include all admitted targets with original witnesses, never select by neural/IK success.
- Predeclare K=16 complete proposal trajectories per task, fixed identical noise pool for both conditions. Condition1 chooses the minimum original terminal goal cost among all16. Condition2 uses the same cost and pool but restricts eligibility to trajectories passing unchanged whole-prefix+tail validity. The single changed factor is explicit validity eligibility; no projection, extra redraws, new loss, dual update or training. If none is valid, record a failed query. No replacement or unbounded rejection. Report the ordinary first-draw result from the same pool as a descriptive control, not another tuned treatment.
- Record all16 attempts and per-query selection: unconditional full success, orientation/translation, invalidity and first-clash causes, no-valid-candidate frequency, valid-goal-candidate coverage, number/cost of prior calls and query time. Report both per-attempt and per-query denominators; K candidates are not16 independent protein identities. Report best-valid candidate goal errors even when tolerance fails. Pair by identity/group and common pool; cluster intervals and fixed resource budget.
- Primary hypothesis test: paired full-valid H4 query-success improvement of validity-constrained versus unrestricted goal-cost selection, with positive95% group-bootstrap lower bound and predeclared≥5pp point gain. Treat this as a diagnostic endpoint; it cannot waive any original scientific safeguard. Preserve H2 and shifted results without selecting settings on them. If a larger prospective family is tested, adjust its intervals before execution.
- **Selected-law KL is a separate required safeguard.** Proposal conditional KL is measurable but is not the KL of the state-dependent selected policy. Finite-candidate ranking changes that law. Retain raw proposals and selection indices, and predeclare a justified estimator or bound for the actual selected distribution before any promotion. If that cannot be established, label results finite-budget feasibility/selection evidence only, with no controller promotion or claim of compliant policy KL. Do not treat the old actor KL as the new selection-policy KL.
- A positive result supports explicit constraint handling as the next controller-design direction. If valid goal-reaching candidates are rare even with16 samples, selection has failed within the declared budget; it does not prove mathematical infeasibility. Then use a separately proposed feasibility/reachability study to distinguish poor proposal support from geometric impossibility. Do not extend K or train longer after seeing the result.

## Alternatives assessed

| Strategy | Evidence and assessment |
|---|---|
| Better constrained optimization/trust region | Strong future candidate for held-out KL drift; a minibatch dual is not a hard trust region. However B/D already pass DEV KL while clash validity fails, so KL control alone does not test the dominant measured physical failure. Joint hard constrained learning adds solver and optimization confounds before proposal availability is known. |
| Explicit validity-aware selection | Most directly targets the measured nonlocal clash criterion with a frozen actor and one changed selection rule. Cheap bounded diagnostic; selected-law KL and compute-matched denominators must be explicit. Recommended. |
| Separate geometric goal control and collision avoidance | Plausible modular architecture, but no recorded gradient-conflict cosine or causal routing experiment establishes that module interference caused these failures. Changes several mechanisms and requires learning/evaluation. Defer. |
| Differentiable projection/short-horizon constrained planning | Native witnesses and historical IK support geometric feasibility, but IK also produced invalid chains and does not share stochastic KL semantics. Projection changes action laws and contact topology; stronger intervention than needed for the first finite-proposal test. Consider later if proposal coverage is insufficient. |
| Reachability/validity critic before SAC | No SAC now. A critic might predict contact risk but needs calibrated labels, sampling coverage and false-safe evaluation. Exact Cα checks already identify the present failure and provide a clearer first intervention; learning a critic would add an untested approximation. |

Do not simply increase updates:0→100→250 improves position while full success/rotation or validity/KL deteriorate. Stop records are binding. V6 is a proposed new scientific protocol, not resumption of V5.1. Ensure incoming checkpoint/staging allowance plus2GiB headroom and a predeclared compute cap before any later authorization. No E010 deletion is prerequisite authorization for this study.
'''
    write(R/'NEXT_RESEARCH_DECISION.md',decision)

    free=os.statvfs(R).f_bavail*os.statvfs(R).f_frsize
    e10=f'''# E010 neighbor-export storage review — v1

Version `{VERSION}`. **Recommendation: retain the18.22GiB directory for now; do not retire it as disposable cache.** Its per-identity archives are scientific per-decision evidence consumed by report/regeneration code. The pipeline is reproducible in design, but complete current-byte and bitwise-regeneration equivalence has not been established. No E010 data were moved/deleted; no mixed-split tensor archive or TEST outcome was opened.

## Allocation, identity and actual reclaim

Directory: `{E10/'neighbors'}`.

The completed native NTFS inventory measured **{e10record['unique_allocated']:,} unique allocated bytes ({e10record['unique_allocated']/2**30:.6f}GiB)**, logical{e10record['logical']:,}bytes,4,010files. Snapshot finishedUTC`{inv.get('finished_utc')}`. A fresh metadata-only traversal confirms the same file count/logical total and link count1 for every file. The global inventory contains no hardlink group involving this path. Thus removing the whole directory could reclaim approximately{e10record['unique_allocated']:,}allocated bytes, subject to current open handles/filesystem metadata. This is a potential, not observed recovery. Directory overlap is not counted twice. WSL st_blocks is not used to replace native NTFS allocated-byte accounting.

| Component | Native allocated bytes | GiB | Scientific role |
|---|---:|---:|---|
| Four dense candidate ID/distance arrays | {densebytes:,} | {densebytes/2**30:.6f} | Fixed candidate pools consumed by per-identity analysis; label-blind nearest-neighbor query outputs. |
| Two FAISS indices | {indexbytes:,} | {indexbytes/2**30:.6f} | Reusable HNSW search structures, exact versions/insertion parameters recorded. |
|2,000 identity NPZs,2,000 hash JSONs and four search/audit receipts | {e10record['unique_allocated']-densebytes-indexbytes:,} | {(e10record['unique_allocated']-densebytes-indexbytes)/2**30:.6f} | Selected and distance-matched reference identities, paired labels/controls, vectors and per-decision evidence behind published summaries. |

Current D: free space before final provenance publication: **{free:,}bytes ({free/2**30:.6f}GiB)**. Hypothetical entire-directory retirement would yield approximately{(free+e10record['unique_allocated'])/2**30:.6f}GiB free; it is not authorized/executed. A narrower separately reviewed retirement of dense arrays+indices would offer only{(densebytes+indexbytes)/2**30:.6f}GiB while retaining identity evidence. This falls far short of treating all18.22GiB as harmless transients.

## Canonical inputs and reproducibility

Frozen protocol seed104081, student commitfc76211, teacher6d1ed30, parent1de4619; FAISS CPU1.15.1 installed on D:. Complete representations remain present (~13.22GiB), standardized panel (~3.12GiB) and transforms remain present, plus parent model, teacher/source completion receipts and frozen splits. No dataset duplicate or off-drive archive was established. `COMPLETED.json` lists15,012 expected file hashes and13 source hashes; per-identity markers bind representation and neighbor NPZ hashes. Original completed verification is evidence, not a new full rehash of14–19GiB.

Fresh checks in this task: {sum(x['match'] for x in sourcechecks)}/{len(sourcechecks)} source hashes match the frozen completion; {sum(x['match'] for x in canonicalchecks)}/{len(canonicalchecks)} student/split/parent/teacher bindings match; {sum(x['match'] for x in markerchecks)}/{len(markerchecks)} small protocol/transform/completion/search/audit receipts match expected completion hashes. Exact paths and expected/actual SHA256 are retained in MATCHED_ARM_FINAL_COMPARISON.json E010_storage_evidence. Existence/metadata was checked for arrays; their full multi-GiB content and2,000 archive digests were not reverified in this bounded review. No TEST labels were accessed to assess storage.

`analyze_e010_student_direction_observability_v1.py` rebuilds panels from frozen representations, single-thread seeded HNSW insertion(M32,efConstruction128), TRAIN-only reference bank268,800points,768,000queries. Stored recall audits: geometry0.996826171875,latent1.0 at efSearch512 on512exact-audit queries. Search elapsed recorded geometry990.118s,latent217.237s; per-identity analysis2329.844s with the historical eight-worker setup. These are historical timings, not measured current regeneration estimates. Search later uses eight threads; tied distances, library/platform details and parallel analysis can affect byte equality. Fixed seed plus source alone is not a bitwise regeneration certificate.

## Dependencies and last-use evidence

- E010 analyze code reads the four candidate arrays to generate per-identity NPZs; reports read those NPZs directly. `report_e010_student_direction_observability_v1.py` writes identity_summaries/results_AB from them. Action-norm/control reports and verifier/freeze scripts reference this experiment; rerunning verification may require exact original archives/receipts.
- Phase-C student training reads representations and canonical old student data, not the neighbor candidate arrays directly; its selected context was chosen using the neighbor analysis. Deleting direct evidence still weakens reproducibility even when later inference loads only selected model weights.
- Bounded static search covered the canonical ProteinGen scripts and this frozen source scripts/src tree; references are the E010 analyzers/train/report/control/verification family. No E017-specific dependency was found. This is not proof of no dependency across every historical repository, notebook or external user. Last-write dates/COMPLETED/search receipts establish completed historical production, not last-read access or permission to retire.
- Neighbor directories contain mixed historical partitions. This task read storage metadata, generator code and provenance receipts, never their scored TEST outputs. Any later bitwise regeneration experiment must respect the original split-selection governance and receive separate authorization.

## Separately reviewed retirement proposal

Before considering even the narrower{(densebytes+indexbytes)/2**30:.6f}GiB search-structure candidate: inventory consumers and freeze canonical panel/representation/source/environment manifests; verify relevant input content against the recorded hashes; preserve all original per-identity outputs/receipts/reports and their hashes; demonstrate that exact raw candidate lists can be regenerated with the original code/version/tie behavior using a bounded TRAIN-only probe, or explicitly accept loss of byte-exact reproduction. Existing marker short-circuits require a separately reviewed clean regeneration location; never remove markers then silently rerun into the protected experiment. Confirm no active reader and native file IDs/allocation immediately before any later authorized retirement. Store deletion ledger and keep2GiB operational headroom plus any next full checkpoint/staging reserve.

The2,000 per-identity NPZs are **not approved disposable files**. Retiring the whole18.22GiB would remove scientifically meaningful evidence; require a distinct preservation/archival decision. No E016/E017/J1 evidence changed, no compression/copy staging was attempted, and no space was reclaimed in this final-analysis task.
'''
    write(R/'E010_STORAGE_REVIEW.md',e10)
    # Last artifact seals all six outputs and the helpers/inputs, without recursive self-hashing.
    for p in [Path(__file__),R/'source/final_scientific_analysis.py',R/'source/final_checkpoint_verify.py']:
        INPUTS[str(p.resolve())]=sha(p)
    free_after=os.statvfs(R).f_bavail*os.statvfs(R).f_frsize
    provenance=dict(schema=VERSION,created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=source,
                    output_sha256={n:sha(R/n) for n in OUTPUTS if n!='PROVENANCE_SHA256.json'},
                    input_sha256=INPUTS,protected_checkpoint_sha256=protected,
                    verified_checkpoint_sha256={a:{role:s['record']['sha256'] for role,s in v['states'].items()} for a,v in ck['arms'].items()},
                    scientific_actions=dict(new_training_updates=0,new_policy_evaluations=0,new_GPU_evaluations=0,TEST_access=False,deleted_files=0),
                    D_free_bytes_before_provenance=free_after,
                    GPU_compute_processes=[],self_hash='not included to avoid recursion; hash this file externally if needed',
                    source_status='sealed tracked source unchanged; final-analysis and recovery helpers are additional untracked files on D:',
                    validation=dict(all_saved_summaries_recomputed=True,all_task_seed_sets_matched=True,
                                    all_saved_final_validity_labels_reconstructed=True,duplicate_matched100_records_equal=True,
                                    matched_TRAIN_task_order=True,checkpoint_SHA_and_full_semantic_digest_CPU=True,
                                    frozen_input_hashes_match=True),
                    publication='per-file atomic fsync+rename, PROVENANCE last; no historical artifact overwritten')
    write(R/'PROVENANCE_SHA256.json',provenance)
    for n,h in provenance['output_sha256'].items():
        assert sha(R/n)==h
    print(json.dumps(dict(published=OUTPUTS,total_bytes=sum((R/n).stat().st_size for n in OUTPUTS),
                          D_free_bytes=os.statvfs(R).f_bavail*os.statvfs(R).f_frsize,source_checks=sum(x['match'] for x in sourcechecks),canonical_checks=canonicalchecks)))


if __name__=='__main__':
    main()
