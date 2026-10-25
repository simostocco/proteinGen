"""Historical packing, reward and likelihood audit; CPU only, no optimizer."""
import collections,hashlib,json,math,os,runpy,statistics,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/mnt/d/Simone/proteinGen_data/e017_v94r3_one_step_feasibility_20261025_v1')
OLD=ROOT.parent/'e017_v94_fixed_sequence_geometry_rl_20261024_v1'
P=runpy.run_path(str(OLD/'source/scripts/e017_v94/production.py'))
S=P['S'];c=P['c'];read=P['read'];sha=P['sha'];save=P['compact']

def contacts(x):
    x=np.asarray(x,np.float64);n=len(x);d=np.linalg.norm(x[:,None]-x[None,:],axis=-1)
    return {(int(i),int(j)) for i,j in zip(*np.where((d<8)&(np.arange(n)[:,None]-np.arange(n)[None,:]>=8)))}

def trajectory(row,goal):
    state=S['reconstruct'](row);x=state.ca.numpy();edges=contacts(x);N=64
    curve=[]
    for n in range(1,len(x)+1):
        C=len(contacts(x[:n]));curve.append(dict(n=n,contacts=C,current_degree=2*C/n,final_degree_lower_bound=2*C/N,
            final_bound_exceeded=2*C/N>P['S']['SPEC']['packing_limits']['64'][1]))
    degree=2*len(edges)/len(x) if len(x) else 0
    if len(x)==64:assert degree==row['metrics']['geometry']['longrange_contact_degree']
    a,b=goal['buried'];e,f=goal['exposed'];counts=collections.Counter(i for edge in edges for i in edge)
    return dict(length=len(x),metrics=row['metrics'],contact_pairs_0based=sorted(edges),contact_count=len(edges),
        top_contact_residues_1based=[[i+1,n] for i,n in counts.most_common(10)],prefix_curve=curve,
        first_proven_overbudget_residue_1based=next((p['n'] for p in curve if p['final_bound_exceeded']),None),
        first_current_density_exceeds_limit_1based=next((p['n'] for p in curve if p['current_degree']>P['S']['SPEC']['packing_limits']['64'][1]),None),
        edges_touching_buried=sum(a<=i<b or a<=j<b for i,j in edges),edges_touching_exposed=sum(e<=i<f or e<=j<f for i,j in edges))

def verify():
    prov=read(OLD/'PROVENANCE_SHA256.json');out={}
    for rel,expected in prov['artifact_SHA256'].items():assert sha(OLD/rel)==expected,rel;out[rel]=expected
    for rel,expected in prov['auxiliary_source_SHA256'].items():assert sha(OLD/rel)==expected,rel;out[rel]=expected
    release=read(OLD/'recovery_qualification_20261025_v1/PRODUCTION_RECOVERY_RELEASE.json')
    for rel,expected in release['binding']['source'].items():assert sha(OLD/'source'/rel)==expected;out['source/'+rel]=expected
    assert sha(OLD/'TRAINING_CONFIGURATION_AND_SEEDS.json')==release['binding']['configuration_SHA256']
    models=read(OLD/'INPUT_VERIFICATION.json')['models']
    for label,meta in models.items():assert sha(Path(meta['path']))==meta['SHA256'];out[label]=meta['SHA256']
    records=[];seen={}
    # Only relevant V94 checkpoint objects, native hardlinks counted once.
    paths=list((OLD/'production/checkpoints').glob('*.pt'))+list((OLD/'recovery_qualification_20261025_v1').glob('*/checkpoints/recovery_*.pt'))
    for path in sorted(paths):
        st=path.stat();key=(st.st_dev,st.st_ino)
        if key in seen:records.append(dict(path=str(path),same_inode_as=seen[key]));continue
        seen[key]=str(path);actual=sha(path);state=torch.load(path,map_location='cpu',weights_only=False)
        digest=P['state_digest'](state);assert all(k in state for k in ['actor','optimizer','scheduler','RNG','identity_order','binding','update','counters','configuration'])
        assert state['binding']==release['binding'];assert state['configuration']['TRAIN_identities']==434
        for value in state['actor'].values():assert torch.isfinite(value).all()
        for value in state['optimizer']['state'].values():
            for tensor in value.values():
                if isinstance(tensor,torch.Tensor):assert torch.isfinite(tensor).all()
        index=read(path.parent.parent/'RECOVERY_INDEX.json')['latest']
        referenced=(str(path.relative_to(path.parent.parent))==index['relative'])
        side=path.with_suffix('.json')
        expected=read(side) if side.exists() else index if referenced else None
        if expected:assert actual==expected['SHA256'] and digest==expected['state_digest']
        records.append(dict(path=str(path),SHA256=actual,state_digest=digest,bytes=st.st_size,update=state['update'],
            compared_to_recorded_hash=expected is not None,all_required_fields=True,finite_weights_and_Adam=True,
            actor_fingerprint=c['state_digest'](state['actor']),Adam_states=len(state['optimizer']['state']),next_goal=state['next_goal_id']))
        del state
    save(ROOT/'HISTORICAL_HASH_AND_CHECKPOINT_AUDIT.json',dict(PASS=True,source_commit=release['source_commit'],verified_artifacts=out,checkpoints=records,
        unindexed_slot_note='SHA measured and payload/binding/finite tensors verified; no invented external hash reference for unindexed slots',historical_files_modified=False))

def analyze():
    torch.set_num_threads(1);verify();failures=[];allpaired=[];baseline={}
    for u in [0,500]:
        for path in sorted((OLD/'production/diagnostics').glob(f'DEV_{u:05}_filtered_*.json')):
            packet=read(path);goal=packet['task']['goal']
            for i,row in enumerate(packet['records']):
                key=(goal['goal_id'],i)
                if u==0:baseline[key]=(packet['task'],row);continue
                task,old=baseline[key];assert task==packet['task']
                if row['metrics']['hard_valid']:continue
                before=trajectory(old,goal);after=trajectory(row,goal);oldedges=set(map(tuple,before['contact_pairs_0based']));newedges=set(map(tuple,after['contact_pairs_0based']))
                doom=after['first_proven_overbudget_residue_1based'];terms=row['proposals'];local_rejects=[q for q in terms if not q[6]]
                affected=dict(goal=goal,candidate_index=i,task=task,seed=row['seed'],new_failure=old['metrics']['hard_valid'],
                    original=before,update500=after,contact_pairs_added=sorted(newedges-oldedges),contact_pairs_removed=sorted(oldedges-newedges),
                    local_rejections=len(local_rejects),rejection_reasons=dict(collections.Counter(reason for q in local_rejects for reason in q[8])),
                    termination=row['termination'],classification='PROPOSAL_EXHAUSTION' if row['termination'] else 'CALIBRATED_GLOBAL_PACKING_LIMIT',
                    first_doom_step_proposals=[q for q in terms if doom and q[0]==doom-1],
                    prefix_source=str(path),baseline_source=str(path).replace('DEV_00500','DEV_00000'))
                failures.append(affected)
    assert len(failures)==11 and sum(f['new_failure'] for f in failures)==7
    save(ROOT/'MATCHED_TRAJECTORY_FAILURES.json',dict(records=failures,N=64,packing_upper=P['S']['SPEC']['packing_limits']['64'][1],
        contact_limit_integer=115,all_pairs_indices_zero_based=True,all_future_coordinate_use='posthoc diagnosis only; controller never receives these contacts'))
    categories=collections.defaultdict(list);groups=[];corr=[]
    for path in sorted((OLD/'production/rollouts').glob('u*.json')):
        packet=read(path);rows=packet['records'];reward=np.array([r['metrics']['reward'] for r in rows]);llr=np.array([r['trace_logprob']-r['reference_trace_logprob'] for r in rows]);aug=reward-P['SPEC']['trace_KL_beta']*llr
        adv=aug-(aug.sum()-aug)/(len(rows)-1);advR=reward-(reward.sum()-reward)/(len(rows)-1)
        decisions=[];dims=[]
        for i,row in enumerate(rows):
            m=row['metrics'];kind='full_success' if m['success'] else 'hard_valid_near_miss' if m['hard_valid'] else 'overpacked_complete' if m['complete'] and not m['packing_valid'] else 'other_invalid'
            categories[kind].append(dict(reward=float(reward[i]),advantage=float(adv[i]),reward_advantage=float(advR[i]),llr=float(llr[i])))
            decisions.append(len(row['proposals']));dims.append(sum(sum(c['mask_for_step'](q[0])) for q in row['proposals']))
            if m['complete'] and 'buried_mean' in m:corr.append((packet['task']['goal']['difficulty'],m['geometry']['longrange_contact_degree'],m['buried_mean'],m['reward'],m['hard_valid']))
        groups.append(dict(update=packet['update'],advantage_variance=float(adv.var()),reward_variance=float(reward.var()),mean_proposals=float(np.mean(decisions)),mean_active_dimensions=float(np.mean(dims)),
            sum_advantage=float(adv.sum()),group_size=len(rows)))
    summary={}
    for name,items in categories.items():
        summary[name]=dict(n=len(items),**{key+'_mean':float(np.mean([x[key] for x in items])) for key in ['reward','advantage','reward_advantage','llr']},
            positive_augmented_advantage=sum(x['advantage']>0 for x in items),reward_min=min(x['reward'] for x in items),reward_max=max(x['reward'] for x in items))
        if 'invalid' in name or name=='overpacked_complete':assert summary[name]['reward_max']==0
    correlations={}
    for difficulty in ['witness','relaxed']:
        values=[x for x in corr if x[0]==difficulty];a=np.array([x[1:3] for x in values]);correlations[difficulty]=dict(n=len(values),packing_buried_Pearson=float(np.corrcoef(a.T)[0,1]))
    save(ROOT/'REWARD_ADVANTAGE_AND_SCALE_AUDIT.json',dict(categories=summary,groups=groups,correlations=correlations,
        no_positive_reward_for_invalid=True,exact_group_size16=all(x['group_size']==16 for x in groups),
        original_reward_SHA256=sha(ROOT.parent/'e017_v91_ordered_goal_recovery_20261023_v1/REWARD_SPECIFICATION.json')))
    print(json.dumps(dict(PASS=True,failed_candidates=len(failures),newly_invalid=sum(f['new_failure'] for f in failures),categories=summary,correlations=correlations)),flush=True)
if __name__=='__main__':analyze()
