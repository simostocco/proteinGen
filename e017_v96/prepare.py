"""CPU evidence audit and predeclared TRAIN-only diagnostic design."""
import collections, copy, csv, math, shutil, sys
from pathlib import Path
import numpy as np
from common96 import *


def bootstrap():
    assert not ROOT.exists()
    ROOT.mkdir()
    (ROOT/'source96').mkdir()
    for p in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(p, ROOT/'source96'/p.name)
    script = (R3/'run.sh').read_text().replace(str(R3), str(R3), 1)
    # Use the already qualified environment; relocate only artifact/cache paths.
    v95run = (V95/'run.sh').read_text().replace('ROOT='+str(V95), 'ROOT='+str(ROOT))
    (ROOT/'run.sh').write_text(v95run); (ROOT/'run.sh').chmod(0o755)
    for name in ['runtime/tmp','runtime/cache','qualification','control']:
        (ROOT/name).mkdir(parents=True,exist_ok=True)


def prepare():
    m = C.frozen(); P,F,compat = load()
    contract = read(C.Q/'COMPATIBILITY_CONTRACT.json')
    assert P['bindings']() == F['P']['bindings']() == contract['original_binding']
    published = read(V95/'PROVENANCE_SHA256.json')
    for name in ['FINAL_E017_V95_VALIDATION_REPORT.md','GOAL_CONDITIONING_ANALYSIS.md','EXPANDED_VALIDATION_RESULTS.json']:
        assert sha(V95/name) == published['files'][name]['SHA256']
    for model in m['models'].values():
        assert sha(model['path']) == model['SHA256']
    write('HISTORICAL_BINDINGS.json',dict(PASS=True, models=m['models'],
          original_scientific_binding=contract['original_binding'],
          current_qualified_binding=contract['repaired_infrastructure_binding'],
          historical_results_SHA256=sha(V95/'EXPANDED_VALIDATION_RESULTS.json'),
          historical_manifest_SHA256=sha(V95/'VALIDATION_PANEL_MANIFEST.json'),
          sources={str(p):sha(p) for p in [R3/'TRAINING_CONFIGURATION_AND_SEEDS.json', R3/'source/scripts/e017_v94r/gradients.py',R3/'source/scripts/e017_v94r/sampling.py',P['V93']/'source/scripts/e017_v93/campaign.py']}, TEST_content_read=False))
    curves=list(csv.DictReader((R3/'RL_TRAINING_CURVES.csv').open()))
    def quantile(v):return np.quantile(v,[0,.25,.5,.75,.95,.99,1]).tolist()
    norms=np.array([float(x['gradient_norm']) for x in curves])
    rows=[]; component=[]; within=[]; positions=[]
    indices=np.unique(np.linspace(1,5000,64,dtype=int)).tolist()
    for u in indices:
        data=read(R3/f'production/rollouts/u{u:06d}.json');rr=data['records'];parts=[]
        for row in rr:
            metric=row['metrics'];hard=float(metric['hard_valid']);utilities=metric.get('utilities',[0.]*5)
            part=[.5*float(metric['success'])]+[hard*.1*float(x) for x in utilities]
            assert len(part)==6 and abs(sum(part)-metric['reward'])<1e-10
            parts.append(part);rows.append(dict(reward=metric['reward'],hard=hard,complete=metric['complete'],burial=float(metric.get('buried_mean',1)>-1 and metric.get('buried_mean',1)<=data['task']['goal']['burial_max']),exposure=float(metric.get('exposed_mean',0)>=data['task']['goal']['exposure_min'])))
            positions.append(data['task']['goal']['buried'][0])
        part=np.array(parts);component.extend(parts)
        centered=part-(part.sum(0)-part)/(len(part)-1)
        within.append(dict(update=u,component_variance=part.var(0).tolist(),component_RLOO_variance=centered.var(0).tolist(),total_reward_variance=float(part.sum(1).var())))
    arr=np.array(component);reward=np.array([x['reward'] for x in rows]);burial=np.array([x['burial'] for x in rows])
    write('REWARD_SIGNAL_EVIDENCE.json',dict(selected_updates=indices,selection='64 equally spaced update indices before examining component outcomes',attempts=len(rows),component_names=['full_success','exposure','burial','contrast','motif','compatibility'],component_mean=arr.mean(0).tolist(),component_SD=arr.std(0).tolist(),component_nonzero_rate=(arr>0).mean(0).tolist(),component_reward_correlations=[float(np.corrcoef(arr[:,j],reward)[0,1]) if arr[:,j].std() else None for j in range(6)],reward_burial_satisfaction_correlation=float(np.corrcoef(reward,burial)[0,1]),burial_satisfaction_rate=float(burial.mean()),hard_valid_rate=float(np.mean([x['hard'] for x in rows])),within_group=within,burial_region_start_histogram=dict(collections.Counter(positions))))
    augmented=[__import__('json').loads(line) for line in (R3/'AUGMENTED_TRAINING_METRICS.jsonl').read_text().splitlines()]
    write('OPTIMIZATION_EVIDENCE.json',dict(updates=len(curves),gradient_norm_quantiles=quantile(norms),quantile_probabilities=[0,.25,.5,.75,.95,.99,1],clipped=int((norms>5).sum()),clip_fraction=float((norms>5).mean()),clip_scale_quantiles=quantile(np.minimum(1,5/norms)),step_KL_quantiles=quantile([float(x['step_KL']) for x in curves]),entropy_quantiles=quantile([x['diagnostics']['latent_Gaussian_entropy_per_active_dimension'] for x in augmented]),advantage_variance_quantiles=quantile([x['advantage_variance'] for x in augmented]),trace_KL_quantiles=quantile([x['full_trace_KL_MC'] for x in augmented]),beta=P['SPEC']['trace_KL_beta'],stochastic_residue_nodes=63,active_geometry_dimensions_without_rejections=186,loss_reduction='sum all proposal scores, mean trajectories only',NO_CAUSAL_CLIPPING_CLAIM=True))
    tasks=read(P['V93']/'manifests/FUTURE_FIXED_TRAIN_POOL.json')['tasks']
    refs={r['identity']:r for r in read(ROOT.parent/'e017_v9_property_steering_20261023_v1/manifests/TRAIN_REFERENCE.json')['records']}
    chosen=[]
    for task in sorted(tasks,key=lambda t:(t['goal']['difficulty']!='witness',ns('TRAIN diagnostic selection',t['source_reference_identity']))):
        ref=refs[task['source_reference_identity']];assert ref['split']=='TRAIN'
        alts=alternatives(task,ref,P['c'],P['S']['SPEC'])
        if len(alts)==2:
            chosen.append(dict(base=task,alternatives=alts,native_witness_checked=True))
        if len(chosen)==2:break
    assert len(chosen)==2,'INSUFFICIENT_MATCHED_MOTIF_TRAIN_TASKS'
    write('CPU_DIAGNOSTIC_ADMISSION_NOTE.json',dict(witness_only_pair_admission_failed=True,reason='No two witness-only TRAIN tasks met both same-motif alternatives with identical numerical thresholds',correction='Admit existing witness or relaxed tasks without changing their thresholds; select by feasibility and deterministic metadata only',selected_difficulties=[x['base']['goal']['difficulty'] for x in chosen],model_outcomes_used=False,production_task_distribution_unchanged=True))
    write('TRAIN_DIAGNOSTIC_MANIFEST.json',dict(groups=chosen,TRAIN_only=True,sequence_motif_thresholds_held_fixed=True, native_coordinates_consumed_by_policy=False, seeds=[[ns('goal swap',i,j) for j in range(8)] for i in range(2)],budget=dict(goal_swaps=192,matched_suffix_continuations=64,engineering_training_attempts=32,reserved_TRAIN_baseline=512,total_new_generation_attempts_max=800,engineering_unique_updates_max=2,engineering_physical_optimizer_calls_max=3,one_optimizer_call_is_exact_journal_replay=True),no_VALIDATION_inference=True))
    # Development-panel selection uses eligibility metadata, never model outcomes.
    pool=read(V95/'engineering/QUALIFIED_VALIDATION_POOL.json')['records']
    previous={a['identity'] for a in m['annotations']}
    previous.update(t['source_reference_identity'] for t in read(P['V93']/'manifests/PROSPECTIVE_FIXED.json')['tasks'])
    previous.update(r['identity'] for r in read(ROOT.parent/'e017_v9_property_steering_20261023_v1/manifests/VALIDATION_REFERENCE.json')['records'])
    candidates=[q for q in pool if q['reference']['identity'] not in previous]
    candidates.sort(key=lambda q:ns('VALIDATION panel',q['reference']['identity']))
    selected=candidates[:512];assert len(selected)==512
    assert len({q['reference']['group'] for q in selected})==512
    assert not {q['reference']['group'] for q in selected}&{t['goal']['protected_group'] for t in tasks}
    diag=[]
    for q in selected:
        for base in q['tasks']:
            a=diagnostic_triple(base,q['reference'],P['c'],P['S']['SPEC'])
            if a is not None:
                a['seeds']=[ns('VALIDATION matched diagnostic',base['source_reference_identity'],j) for j in range(8)]
                diag.append(a)
                break
        if len(diag)==128:break
    print('MATCHED_VALIDATION_DIAGNOSTIC_GROUPS',len(diag),flush=True)
    assert len(diag)>=32,'INSUFFICIENT_VALIDATION_MATCHED_DIAGNOSTICS'
    write('VALIDATION_PANEL_MANIFEST.json',dict(identities=512,independent_groups=512,primary_goals=1024,tasks=[t for q in selected for t in q['tasks']],diagnostic_groups=len(diag),diagnostic_goals=3*len(diag),diagnostics=diag,diagnostic_base_separately_generated=True,diagnostic_motif_can_differ_from_primary_but_is_fixed_within_triple=True,seeds={t['goal']['goal_id']:[ns('VALIDATION candidates',t['goal']['goal_id'],j) for j in range(8)] for q in selected for t in q['tasks']},excluded_prior_E017_performance_identity_count=len(previous),prior_E016_selection_exposure_allowed=True,development_not_TEST=True,selection_not_based_on_outcomes=True,no_inference_yet=True))
    config=copy.deepcopy(read(R3/'TRAINING_CONFIGURATION_AND_SEEDS.json'))
    config.update(experiment='E017_V96_EXPLICIT_BURIAL_TOKEN_CONTINUATION',production_authorized=False,engineering_authorized=True,initializer='milestone_05000 plus explicit burial token; fresh AdamW; independent actor',starting_update=5000,production_updates=5000,updates=5000,generation_attempts=80000,seed=19629,main_new_intervention='one explicit burial-region token using existing goal projection; no added parameters',comparison='final V96 versus frozen original-encoding R3 update5000; augmented update0 additionally separates immediate encoding and learning effects',causal_limitation='single continuation arm cannot isolate token change from additional RL without a matched continued-R3 control',development_selection='reserved TRAIN only; no VALIDATION selection during optimization',confirmation='frozen V96 VALIDATION manifest only after final checkpoint',loss_unchanged=True,original_scientific_reward_unchanged=True,neural_architecture_unchanged=True,production_authorization='REQUIRES FUTURE EXPLICIT REVIEW RECEIPT; none issued',next_stage_auto_launch=False,qualification_budget=read(ROOT/'TRAIN_DIAGNOSTIC_MANIFEST.json')['budget'])
    write('TRAINING_CONFIGURATION_AND_SEEDS.json',config)
    write('PRE_DIAGNOSTIC_PLAN.json',dict(algorithm_candidate='explicit burial-region token',selection_basis='source-established motif/exposure anchored tokens versus burial only global query features; confirm with bounded same-motif TRAIN probes',scope='one focused diagnosis/qualification program; no adaptive sweep',training_attempts_production=0,TEST=False,source96_SHA256={p.name:sha(p) for p in (ROOT/'source96').glob('*.py')}))
    print('CPU_PREPARATION_PASS', 'TRAIN diagnostic identities', [x['base']['source_reference_identity'] for x in chosen], 'VALIDATION diagnostic groups',len(diag),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='bootstrap':bootstrap()
    else:prepare()
