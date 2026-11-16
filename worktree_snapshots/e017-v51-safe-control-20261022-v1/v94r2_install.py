"""Create a fresh run using the reviewed training kernel; no worker launch."""
from pathlib import Path
import json,shutil
R=Path('/mnt/d/Simone/proteinGen_data/e017_v94r2_corrected_geometry_rl_20261025_v1');O=R.parent/'e017_v94r_packing_safety_20261025_v1';D=R/'source/scripts/e017_v94r';X=R/'source/scripts/e017_v94r2'
assert not (R/'TRAINING_CONFIGURATION_AND_SEEDS.json').exists(),'refuse adopting an existing experiment'
for n in ['runtime/tmp','runtime/cache','control','qualification']:(R/n).mkdir(parents=True,exist_ok=True)
X.mkdir(exist_ok=True)
for p in D.glob('*.py'):p.write_text(p.read_text().replace(str(O),str(R)))
for src,name in [('v94r2_common_staged.py','common.py'),('v94r2_admission_staged.py','admission.py'),('v94r2_monitor_staged.py','monitor.py'),('v94r2_worker_staged.py','worker.py'),('v94r2_final_staged.py','final_evaluate.py'),('v94r2_publish_staged.py','publish.py')]:shutil.copyfile(src,X/name)
s=(D/'production.py').read_text()
s=s.replace("(ROOT/'source/scripts/e017_v94r').glob('*.py'))}","[q for folder in ['e017_v94r','e017_v94r2'] for q in (ROOT/'source/scripts'/folder).glob('*.py')])}")
needle="    evaluator=runpy.run_path(str(ROOT/'source/scripts/e017_v94r/evaluation.py'))"
roundtrip='''    # Parameterized relocation/seed changes preserve the reviewed Runtime.
    # Qualify the actual fresh schema/optimizer/RNG instance before sampling.
    receipt=ROOT/'ACTUAL_INITIAL_RECOVERY_ROUNDTRIP.json'
    if not receipt.exists():
        assert rt.update==0 and not rt.counters
        initial=state_digest(rt.state())
        def draws():return dict(torch=torch.rand(8).tolist(),cuda=torch.rand(8,device='cuda').cpu().tolist(),numpy=np.random.random(8).tolist(),python=[random.random() for _ in range(8)])
        first=draws();rt.restore();assert state_digest(rt.state())==initial;second=draws();assert first==second;rt.restore()
        assert c['state_digest'](rt.e.actor.state_dict())=='884b1e016135f4bf4b3594b6aefc81c1594a525186e09091ad5bd5c023781f80'
        assert all(float(x['step'])==0 and not x['exp_avg'].count_nonzero() and not x['exp_avg_sq'].count_nonzero() for x in rt.opt.state.values())
        rt.e.prior.verify()
        compact(receipt,dict(PASS=True,optimizer_steps=0,full_actual_state_roundtrip=True,next_CPU_CUDA_numpy_python_draws_equal=True,actor_initializer_fingerprint=c['state_digest'](rt.e.actor.state_dict()),J1_frozen=True,checkpoint=read(rt.root/'RECOVERY_INDEX.json')['latest'],qualification_transfer_SHA256=sha(ROOT/'qualification/RELEASE.json')))
'''
assert s.count(needle)==1;s=s.replace(needle,roundtrip+needle)
needle="        rt.best=dict(update=0,**rt.baseline['filtered']);rt.publish('DEVELOPMENT_BEST')"
s=s.replace(needle,needle+"\n        initial=rt.baseline['filtered']\n        assert initial['hard_valid']>=.91875 and initial['backbone_valid']>=.95 and initial['geometry_KL_per_active']<=1 and initial['structural_pair_RMSD']>=12.449118749897715,'INITIAL_ABSOLUTE_VALIDITY_KL_DIVERSITY_ADMISSION'\n        compact(ROOT/'FROZEN_INITIALIZATION_BASELINE.json',dict(update=0,summary=rt.baseline,original_V94=dict(hard_valid=.96875,backbone_valid=1.,structural_pair_RMSD=15.561398437372143),baseline_will_not_be_reset=True,abort_gate='no new separate abort rule; all aborts count against unchanged hard-validity gate'))")
(D/'production.py').write_text(s)
e=(D/'evaluation.py').read_text()
# Add a TRAIN-only paired-goal diagnostic without changing selection panels.
e=e.replace("    compact(path,dict(summary=result,", "    result['TRAIN_goal_sensitivity']=goal_sensitivity(rt,update)\n    compact(path,dict(summary=result,")
e+='''
def goal_sensitivity(rt,update):
    tasks=read(V93/'manifests/FUTURE_FIXED_TRAIN_POOL.json')['tasks']
    identities=sorted({t['source_reference_identity'] for t in tasks})[:4]
    selected=[t for identity in identities for t in tasks if t['source_reference_identity']==identity]
    out=[];fp=P['c']['state_digest'](rt.e.actor.state_dict())
    for policy,label in [('C','filtered'),('B','underlying_unfiltered')]:
        for i,task in enumerate(selected):
            path=rt.root/f'diagnostics/SENSITIVITY_{update:05}_{label}_{i:02}.json'
            if path.exists():obj=read(path);assert obj['actor_fingerprint']==fp
            else:
                seeds=[S['ns']('R2_TRAIN_SENSITIVITY',task['source_reference_identity'],j) for j in range(4)]
                rows=P['sample'](rt.e,task,policy,4,seeds)
                obj=dict(task=task,records=rows,actor_fingerprint=fp);compact(path,obj)
            stats=summarize([obj]);stats['K4']=stats.pop('K8')
            out.append(dict(identity=task['source_reference_identity'],difficulty=task['goal']['difficulty'],policy=label,summary=stats))
    return dict(fitting_TRAIN_only_not_selection=True,identities=4,candidates_per_goal=4,matched_streams_across_existing_witnessed_relaxed_goals=True,results=out)
'''
(D/'evaluation.py').write_text(e)
cfg=json.loads((O/'TRAINING_CONFIGURATION_AND_SEEDS.json').read_text());cfg.update(experiment='E017_V94R2_CORRECTED_GEOMETRY_RL',authorized=True,engineering_authorized=False,production_authorized=True,
 seed=19327,previous_seed_not_resumed=19317,random_seed_policy='fresh deterministic training shuffle/variant/proposal schedule; development candidate seeds unchanged',
 conservative_peak_full_images=10,peak_distinct_checkpoint_images=10,qualification_checkpoint_images_included=0,production_peak_full_images=10,
 minimum_free_bytes=10*224*2**20+4*2**30+512*2**20,
 qualification_reuse='reviewed exact Runtime/sampler/gradient/packing source equivalence; actual fresh full-state restore before first optimizer; no duplicate qualification model images',
 reviewed_source_commit='e390b87357c31a6cad8958ecd9292c942e01e57d',
 minimum_other_artifact_budget=512*2**20,trace_forecast_margin=1.15,
 scientific_safety='same corrected initialization matched baseline; >5pp hard/backbone decline; original absolute V94 floors also binding; >20percent diversity loss; referenceKL>1;stepKL>.02; nonfinite/corruption/storage stops',
 abort_safety='no added abort-specific threshold; aborted attempts are failures in hard-validity denominator',
 development_selection='same16 reserved TRAIN identities,8candidate matched streams; extra4fit-TRAIN paired-goal sensitivity monitoring never used for selection',
 confirmation='same frozen32goals/16VALIDATIONidentities; no prospective outcome inspected before full-budget completion',
 production_authorization='current explicit E017 V9.4-R2 user authorization; no follow-up authorization required for unchanged admitted run')
(R/'TRAINING_CONFIGURATION_AND_SEEDS.json').write_text(json.dumps(cfg,indent=2)+'\n')
shutil.copyfile(O/'INPUT_VERIFICATION.json',R/'INPUT_VERIFICATION.json')
shutil.copyfile(O/'METRIC_DEFINITIONS.json',R/'METRIC_DEFINITIONS.json')
(R/'run.sh').write_text((O/'run.sh').read_text().replace(str(O),str(R)));(R/'run.sh').chmod(0o755)
for file in list(D.glob('*.py'))+list(X.glob('*.py')):compile(file.read_text(),str(file),'exec')
print('R2_INSTALLED_FRESH',cfg['minimum_free_bytes'],shutil.disk_usage(R).free)
