"""Post-budget fixed prospective evaluation only; never creates an optimizer."""
import collections,json,runpy,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v94_fixed_sequence_geometry_rl_20261024_v1')
P=runpy.run_path(str(R/'source/scripts/e017_v94/production.py'));E=runpy.run_path(str(R/'source/scripts/e017_v94/evaluation.py'))
S=P['S'];read=P['read'];sha=P['sha'];save=P['compact'];V=P['V93'];O=R/'production/confirmation'

def paired(a,b,K):
    groups=collections.defaultdict(list)
    for x,y in zip(a,b):
        assert x['task']==y['task']
        f=lambda z:int(z['records'][0]['metrics']['success']) if K==1 else int(any(r['metrics']['success'] for r in z['records'][:K]))
        groups[x['task']['goal']['protected_group']].append(f(x)-f(y))
    values=np.array([np.mean(v) for v in groups.values()]);rng=np.random.default_rng(19319)
    draws=values[rng.integers(len(values),size=(10000,len(values)))].mean(1)
    return dict(delta=float(values.mean()),CI95=np.quantile(draws,[.025,.975]).tolist(),independent_groups=len(values),goals=sum(map(len,groups.values())),bootstrap_seed=19319)

def evaluate():
    done=read(R/'production/TRAINING_COMPLETED.json');assert done['update']==5000 and done['counters']['attempts']==80000
    release=read(R/'recovery_qualification_20261025_v1/PRODUCTION_RECOVERY_RELEASE.json');assert release['PASS'] and release['binding']==P['bindings']()
    O.mkdir(exist_ok=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tasks=read(V/'manifests/PROSPECTIVE_FIXED.json')['tasks'];assert len(tasks)==32 and len({t['source_reference_identity'] for t in tasks})==16
    e,_=P['K']['load_engine']('C');initial=P['c']['state_digest'](e.actor.state_dict())
    assert initial==read(V/'COMPLETED_validation.json')['actor_fp'],'historical baseline weights mismatch'
    best=done['best']['update'];cpdir=R/'production/checkpoints';name=f'milestone_{best:05}' if best in P['SPEC']['named_updates'] else f'development_best_{best:05}'
    meta=read(cpdir/(name+'.json'));cp=R/'production'/meta['relative'];assert sha(cp)==meta['SHA256']
    state=torch.load(cp,map_location='cpu',weights_only=False);assert P['state_digest'](state)==meta['state_digest'] and state['binding']==P['bindings']()
    e.actor.load_state_dict(state['actor'],strict=True);e.actor.eval().requires_grad_(False);del state
    trained_fp=P['c']['state_digest'](e.actor.state_dict());groups={}
    for pol,label in [('A','J1'),('B','parent_unfiltered'),('C','parent_filtered')]:
        groups[label]=[]
        for i,task in enumerate(tasks):
            obj=read(V/f'results/PROSPECTIVE_{pol}_{i:03}.json');assert obj['task']==task and len(obj['records'])==8
            assert sha(V/f'results/PROSPECTIVE_{pol}_{i:03}.json')==read(V/f'results/PROSPECTIVE_{pol}_{i:03}.sha.json')['sha256']
            groups[label].append(obj)
    for pol,label in [('C','RL_filtered'),('B','RL_unfiltered')]:
        groups[label]=[]
        for i,task in enumerate(tasks):
            path=O/f'{label}_{i:03}.json'
            if path.exists():
                obj=read(path);assert obj['actor_fp']==trained_fp and obj['task']==task
            else:
                seeds=[S['ns']('CANDIDATE',task['goal']['goal_id'],j) for j in range(8)]
                start=time.perf_counter();rows=P['sample'](e,task,pol,8,seeds)
                obj=dict(task=task,records=rows,actor_fp=trained_fp,seconds=time.perf_counter()-start);save(path,obj)
            groups[label].append(obj)
    # Goal-target changes under the same sequence and stochastic streams.
    sensitivity=[]
    for i,task in enumerate(tasks):
        path=O/f'GOAL_SENSITIVITY_{i:03}.json'
        if path.exists():obj=read(path);assert obj['actor_fp']==trained_fp
        else:
            seeds=[S['ns']('V94_TARGET_SENSITIVITY',task['source_reference_identity'],j) for j in range(8)]
            rows=P['sample'](e,task,'C',8,seeds);obj=dict(task=task,records=rows,actor_fp=trained_fp);save(path,obj)
        sensitivity.append(obj)
    # Frozen non-RL generated-prefix suffix-search control, no native targets.
    initmeta=read(R/'production/checkpoints/development_best_00000.json');initcp=R/'production'/initmeta['relative']
    assert sha(initcp)==initmeta['SHA256'];initialstate=torch.load(initcp,map_location='cpu',weights_only=False)
    e.actor.load_state_dict(initialstate['actor'],strict=True);del initialstate
    assert P['c']['state_digest'](e.actor.state_dict())==initial
    search=[]
    for i,task in enumerate(tasks):
        path=O/f'FROZEN_SUFFIX_SEARCH_{i:03}.json'
        if path.exists():obj=read(path)
        else:
            row=groups['parent_filtered'][i]['records'][0];records=[row]
            if len(row['sequence'])>=32:
                prefix=S['reconstruct'](row,64,32);assert S['prefix_status'](prefix)['valid']
                P['prepare_sampler']();seeds=[S['ns']('V94_SUFFIX_SEARCH',task['goal']['goal_id'],j) for j in range(15)]
                records+=S['generate'](P['K']['LiveActor'](e),task,'C',15,state=prefix,seeds=seeds)
            obj=dict(task=task,records=records,prefix_source=f'{V}/results/PROSPECTIVE_C_{i:03}.json',prefix_candidate=0,
                prefix_length=32,full_initial_generation_cost_included=True,max_terminal_evaluations=16)
            save(path,obj)
        search.append(obj)
    summaries={label:E['summarize'](rows) for label,rows in groups.items()}
    K1=paired(groups['RL_filtered'],groups['parent_filtered'],1);K8=paired(groups['RL_filtered'],groups['parent_filtered'],8)
    a=summaries['RL_filtered'];b=summaries['parent_filtered']
    safety=a['hard_valid']>=b['hard_valid']-.05 and a['backbone_valid']>=b['backbone_valid']-.05 and a['geometry_KL_per_active']<=1 and a['structural_pair_RMSD']>=.8*b['structural_pair_RMSD']
    GO=K1['delta']>=.05 and K1['CI95'][0]>0 and K8['delta']>=0 and safety
    # Independently evaluated final5000 checkpoint may differ from development
    # selection. It is preserved; selected checkpoint is the primary comparison.
    result=dict(outcome='RL_LEARNING_GO' if GO else 'RL_PERFORMANCE_NO_GO',selected_update=best,
        selected_checkpoint=str(cp),selected_checkpoint_SHA256=meta['SHA256'],actor_fingerprint=trained_fp,
        task_manifest_SHA256=sha(V/'manifests/PROSPECTIVE_FIXED.json'),tasks=32,identities=16,
        summaries=summaries,K1_paired=K1,K8_paired=K8,safeguards_pass=safety,
        suffix_search_goals_reached=sum(any(r['metrics']['success'] for r in g['records']) for g in search),
        suffix_search_attempts=sum(len(g['records']) for g in search),
        goal_sensitivity_files=len(sensitivity),native_coordinates_supplied=False,fixed_AA=True,TEST=False,
        frozen_baselines_reused=True,unfiltered_and_filtered_KL_are_not_the_same_density=True)
    e.prior.verify();assert P['c']['state_digest'](e.parent.state_dict())==e.fp
    save(R/'FINAL_PROSPECTIVE_COMPARISON.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':evaluate()
