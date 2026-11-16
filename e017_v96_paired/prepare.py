"""Freeze a new controlled two-arm study without altering qualified history."""
from paired import *

def main():
    assert not PAIR.exists() and all(not r.exists() for r in ARMS.values()),'NEW_VERSIONED_ROOTS_REQUIRED'
    P,F,C=old.load();old.C.frozen()
    original_provenance=read(ENG/'PROVENANCE_SHA256.json')
    for name,v in original_provenance['files'].items():assert sha(ENG/name)==v['SHA256'],('ENGINEERING_ARTIFACT_CHANGED',name)
    assert read(ENG/'qualification/RELEASE.json')['PASS'] and read(ENG/'qualification/METADATA_COUNT_FIX.json')['PASS']
    assert sha(ENG/'source96/algorithm.py')=='52ddc2560661325add2795af7a57e085909f43f927bc9a977c68626c74077b68'
    assert shutil.disk_usage(PAIR.parent).free>=20*2**30,'INSUFFICIENT_PAIRED_STORAGE'
    PAIR.mkdir();(PAIR/'source').mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copyfile(p,PAIR/'source'/p.name)
    for root in [PAIR,*ARMS.values()]:
        root.mkdir(exist_ok=True)
        for folder in ['runtime/tmp','runtime/cache','control','qualification']:(root/folder).mkdir(parents=True,exist_ok=True)
        script=(ENG/'run.sh').read_text().replace('ROOT='+str(ENG),'ROOT='+str(root))
        (root/'run.sh').write_text(script);(root/'run.sh').chmod(0o755)
    shutil.copyfile(ENG/'VALIDATION_PANEL_MANIFEST.json',PAIR/'VALIDATION_PANEL_MANIFEST.json')
    panel=read(PAIR/'VALIDATION_PANEL_MANIFEST.json');assert panel['identities']==512 and panel['primary_goals']==1024 and panel['diagnostic_groups']==80 and panel['diagnostic_goals']==240
    tasks=read(P['V93']/'manifests/FUTURE_FIXED_TRAIN_POOL.json')['tasks'];ids=sorted({t['source_reference_identity'] for t in tasks});assert len(ids)==434 and len(tasks)==868
    lookup={i:[t for t in tasks if t['source_reference_identity']==i] for i in ids};order=np.random.default_rng(19629).permutation(len(ids)).tolist()
    schedule=[]
    for u in range(1,5001):
        identity=ids[order[(u-1)%len(order)]];task=lookup[identity][((u-1)//len(order))%len(lookup[identity])]
        schedule.append(dict(update=u,identity=identity,goal_id=task['goal']['goal_id'],seeds=[P['S']['ns']('SUBSTANTIAL',19629,u,i) for i in range(16)]))
    atomic(PAIR/'TRAINING_SCHEDULE.json',dict(seed=19629,identities=ids,identity_order=order,updates=schedule),True)
    train_groups={t['goal']['protected_group'] for t in tasks};val_groups={t['goal']['protected_group'] for t in panel['tasks']}
    assert len(val_groups)==512 and not val_groups&train_groups
    pool=read(old.V95/'engineering/QUALIFIED_VALIDATION_POOL.json')['records'];byid={q['reference']['identity']:q for q in pool};witnesses=[]
    for task in panel['tasks']:
        q=byid[task['source_reference_identity']];ref=q['reference'];assert ref['sequence']==task['fixed_sequence'] and ref['group']==task['goal']['protected_group']
        assert P['c']['score'](ref['sequence'],np.asarray(ref['coordinates']),task['goal'],P['S']['SPEC'])['success']
        witnesses.append(dict(identity=ref['identity'],group=ref['group'],goal_id=task['goal']['goal_id'],witness_SHA256=old.C.digest(ref)))
    for d in panel['diagnostics']:
        task=d['base_task'];q=byid[task['source_reference_identity']];seq=task['fixed_sequence'];goals=[task,*d['alternatives'].values()]
        assert all(t['fixed_sequence']==seq and t['goal']['motif']==task['goal']['motif'] and t['goal']['exposure_min']==task['goal']['exposure_min'] and t['goal']['burial_max']==task['goal']['burial_max'] for t in goals)
        assert all(P['c']['score'](seq,np.asarray(q['reference']['coordinates']),t['goal'],P['S']['SPEC'])['success'] for t in goals)
    atomic(PAIR/'ELIGIBILITY_VERIFICATION.json',dict(PASS=True,identities=512,independent_groups=512,primary_goals=1024,diagnostic_groups=80,diagnostic_goals=240,manifest_unchanged=sha(PAIR/'VALIDATION_PANEL_MANIFEST.json')==sha(ENG/'VALIDATION_PANEL_MANIFEST.json'),TRAIN_disjoint=True,native_witnesses=witnesses,native_coordinates_never_policy_inputs=True,TEST=False),True)
    shared=copy.deepcopy(read(ENG/'TRAINING_CONFIGURATION_AND_SEEDS.json'))
    shared.update(experiment='E017_V96_CONTROLLED_PAIRED_CONTINUATION',production_authorized=True,seed=19629,updates=5000,production_updates=5000,generation_attempts=80000,initializer='exact certified R3 milestone05000; fresh optimizer and scheduler for each arm',comparison='matched continuation original encoding versus qualified burial token',main_new_intervention='encoding flag is sole intended scientific difference between arms',production_authorization='Current user explicit paired substantial-training authorization after all admission gates',development_selection='frozen reserved16 TRAIN goals; no VALIDATION selection',checkpoint_selection='final5000 primary; stopped checkpoint explicitly partial')
    for arm,root in ARMS.items():atomic(root/'TRAINING_CONFIGURATION_AND_SEEDS.json',dict(shared,arm=arm,burial_token_enabled=arm=='B'),True)
    a=read(ARMS['A']/'TRAINING_CONFIGURATION_AND_SEEDS.json');b=read(ARMS['B']/'TRAINING_CONFIGURATION_AND_SEEDS.json')
    assert {k:v for k,v in a.items() if k not in ['arm','burial_token_enabled']}=={k:v for k,v in b.items() if k not in ['arm','burial_token_enabled']}
    protocol=dict(schema='V96_CONTROLLED_PAIRED_1',frozen_UTC=now(),arms={a:str(r) for a,r in ARMS.items()},planned_updates_per_arm=5000,planned_fitting_attempts_per_arm=80000,seed=19629,sequence='A then B; no simultaneous heavy CUDA',minimum_free_launch_bytes=20*2**30,operational_reserve_bytes=2*2**30,TRAIN_monitor_updates=[0,250,500,1000,2000,3500,5000],TRAIN_monitor_attempts_per_callback=352,admission_engineering_attempts_per_arm=16,admission_engineering_unique_updates_per_arm=1,validation_attempts=89088,primary=dict(endpoint='raw burial directional response S',contrast='Bfinal-Afinal',practical_between_arm=.01,difference_in_changes_min=.01,within_B_response_min=.01,CI_confidence=.95,bootstrap_groups=80,bootstrap_draws=10000,bootstrap_seed=19329,intersection_gate='positive B-A lower95 and B-A point>=.01; positive difference-in-changes lower95 and point>=.01; positive B-own change lower95; Bfinal mean>=.01 and lower95>0'),matched_secondary=dict(endpoint_families=8,contrasts_per_family=4,Bonferroni_alpha=.05/32,confidence=1-.05/32),legacy=dict(K1_min_gain=.05,K1_confidence=.95,secondary_confidence=1-.05/6,contract_not_replaced=True),all_aborts_failures=True,diagnostics_not_primary=True,single_seed_limitation=True,no_automatic_safety_restart=True,TEST=False)
    atomic(PAIR/'PAIRED_PROTOCOL.json',protocol,True)
    (PAIR/'FROZEN_PAIRED_TRAINING_PROTOCOL.md').write_text('''# Frozen controlled paired V9.6 study

Both arms start from exact certified R3 update5000 parameters/buffers with fresh AdamW, zero moments and fresh scheduler. A uses original encoding; B installs the byte-identical qualified burial token. Same17438114 parameters,434 TRAIN identities/868 goals, deterministic5000-update schedule and seed19629. Only arm identifier and token flag differ in configurations. Original J1, reward, full rejected-proposal likelihood, RLOO, norm clip5, LR schedule, KL beta, finite8 filters, validity tolerances and safety floors stay unchanged.

Each arm targets5000 additional updates/80000 fitting attempts. The common raw random stream is keyed by (candidate seed,residue t,rejection k); earlier acceptance or abort changes which indexed draws are used, not a claim of identical sampled trajectories. CPU/CUDA/NumPy/Python states and journals remain independent. Arms never inherit each other's fitted weights. Original-encoding A0 and augmented B0 measurements quantify token insertion before learning.

Both arms must pass exact initialization, fresh-state/RNG/optimizer/journal replay, actual interrupted-write recovery, hook isolation, likelihood and initial safety before either trains. Admission uses352 TRAIN evaluation attempts per arm (256 fixed development plus96 same-motif sensitivity) and16 fresh engineering attempts per arm; one exact optimizer replay reuses its committed trace. Production initial checkpoint and optimizer are freshly created; no engineering weights are transferred.

Safety is exactly original: initial hard>=.91875, backbone>=.95, diversity>=12.44911875 and proposalKL<=1. Monitoring at0,250,500,1000,2000,3500,5000 retains original <=5pp hard/backbone deterioration with original absolute floors, >=80% diversity versus stronger historical/new baseline, proposalKL<=1 and per-stepKL<=.02. Nonfinite/corrupt/storage failures stop. K1 or reward alone never causes an early safety stop. Scientific stops create cooperative STOP and preserve a checkpoint; no hyperparameter retry. A registered safety stop in A does not preclude B's independently admitted opportunity. Infrastructure/cooperative failures halt the controller. No unqualified automatic restart.

Poll60s, progress reports1800s, immediate scheduled-evaluation/recovery/safety events. Per-update metrics stratify TRAIN identity/difficulty and include reward, properties, validity, packing/clashes, exhaustion, entropy, KL, clipping and compute. Fixed-development metrics include raw/filtered K1/K8, diversity and goal sensitivity. No VALIDATION metrics select optimizer steps or checkpoints.

The unchanged frozen panel has512 independent identities/1024 primary goals and80 matched diagnostic groups/240 diagnostic goals. Diagnostic triples hold sequence/motif/thresholds fixed internally; they may be looser than primary targets and never enter primary denominators. No native future coordinates enter inference. E016 selection exposure is permitted; this is development VALIDATION, not sealed TEST.

Primary scientific estimand is raw burial S(Bfinal)-S(Afinal). Practical difference is .01 on0-1 proxy scale, matching the prior meaningful-response scale. Also require difference in changes [Bfinal-B0]-[Afinal-A0]>=.01, positive B-own change, and meaningful Bfinal S>=.01. Each required directional lower bound uses the paired protected-group bootstrap10000 draws seed19329, individual two-sided95% intervals. These form an intersection claim, not jointly simultaneous95% intervals. Invalid pairs contribute0; valid-only results descriptive. Both B-A and difference-in-changes support are necessary to distinguish insertion from learned benefit. One training seed per arm cannot establish algorithmic robustness.

Matched secondary analysis freezes8 endpoints (raw/filtered K1,K8,burial S,exposure S) x4 contrasts (A change,B change,Bfinal-Afinal,difference in changes): Bonferroni .05/32, confidence99.84375%. Original K1 promotion contracts against A0 retain >=5pp and positive95% lower bound, K8 preservation and original safeguards for each final actor. The legacy six-secondary-family99.1667% intervals are additionally reported as specified in V9.6; they are distinct from the expanded simultaneous paired family. No original promotion criterion is replaced by a favorable between-arm comparison. Exposure preservation reports meaningful positive response and no point decrease versus A0 plus paired intervals, not an invented noninferiority test.

Final states: J1, A0 raw/filtered, B0 raw/filtered, Afinal raw/filtered, Bfinal raw/filtered, eight candidates/goal;89088 attempts. If an arm stops early, assess its preserved checkpoint only as a partial study and report all committed and generated exposure including unsafe uncommitted rollouts. No controlled full-budget advantage claim if either fails to complete5000/80000. Frozen post-stop evaluation is separate from training and never restarts the optimizer. Final5000 is primary, never a favorable selected checkpoint. No historical results are pooled, no TEST/AlphaFold/AA/high-level RL, and no subsequent campaign starts.
''')
    (PAIR/'RUN_STATUS.md').write_text('# Controlled V9.6 paired study\n\nPREPARATION COMPLETE; admission and production not yet launched. Authoritative live records: runtime/CONTROLLER_STATUS.json, runtime/MONITOR_LATEST.json, runtime/EVENTS.jsonl. Scientific findings remain pending.\n')
    files={str(p):sha(p) for p in (PAIR/'source').glob('*.py')}
    for p in [PAIR/'PAIRED_PROTOCOL.json',PAIR/'FROZEN_PAIRED_TRAINING_PROTOCOL.md',PAIR/'TRAINING_SCHEDULE.json',PAIR/'VALIDATION_PANEL_MANIFEST.json',PAIR/'ELIGIBILITY_VERIFICATION.json',PAIR/'run.sh',*[r/'TRAINING_CONFIGURATION_AND_SEEDS.json' for r in ARMS.values()],*[r/'run.sh' for r in ARMS.values()],*[ENG/'source96'/p for p in ['algorithm.py','campaign.py','common96.py','prepare.py','publish.py','qualify.py','study.py']],ENG/'TRAIN_DIAGNOSTIC_MANIFEST.json',ENG/'qualification/RELEASE.json',ENG/'qualification/METADATA_COUNT_FIX.json',P['V93']/'manifests/FUTURE_FIXED_TRAIN_POOL.json',P['V93']/'manifests/TRAIN_TASKS.json',R3/'TRAINING_CONFIGURATION_AND_SEEDS.json',PAIR.parent/'e017_v91_ordered_goal_recovery_20261023_v1/REWARD_SPECIFICATION.json']:
        files[str(p)]=sha(p)
    models={v['path']:v['SHA256'] for v in read(ENG/'HISTORICAL_BINDINGS.json')['models'].values()}
    atomic(PAIR/'FROZEN_SOURCE_AND_INPUTS.json',dict(files=files,historical_models=models,qualified_token_SHA256=sha(ENG/'source96/algorithm.py'),model_start_update=5000),True)
    atomic(PAIR/'PREPARATION_COMPLETED.json',dict(UTC=now(),PASS=True,production_updates=0,D_free_bytes=shutil.disk_usage(PAIR).free),True)
    print('PAIRED_PREPARATION_PASS',str(PAIR),flush=True)
if __name__=='__main__':main()
