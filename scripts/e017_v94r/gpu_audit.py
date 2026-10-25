"""Bounded frozen-policy counterfactuals and historical gradient replay only."""
import copy,json,math,runpy,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v94r2_corrected_geometry_rl_20261025_v1');O=R.parent/'e017_v94_fixed_sequence_geometry_rl_20261024_v1'
P=runpy.run_path(str(O/'source/scripts/e017_v94/production.py'));N=runpy.run_path(str(R/'source/scripts/e017_v94r/production.py'))
S=P['S'];c=P['c'];p=P['p'];T=P['K']['S']['T'];AA=P['AA'];save=P['compact'];read=P['read']

def flat(e):return torch.cat([(x.grad.detach().cpu().reshape(-1) if x.grad is not None else torch.zeros(x.numel())) for x in e.actor.parameters()])

def component(e,task,rows,which):
    rewards=np.array([r['metrics']['reward'] for r in rows]);llr=np.array([r['trace_logprob']-r['reference_trace_logprob'] for r in rows]);rv=rewards-(rewards.sum()-rewards)/(len(rows)-1);kv=llr-(llr.sum()-llr)/(len(rows)-1)
    coef=-rv if which=='reward' else P['SPEC']['trace_KL_beta']*(kv+1)
    e.actor.zero_grad(set_to_none=True);worlds=[S['world'](task['goal']) for _ in rows];maxerr=0.
    for t in range(64):
        ids=[i for i,row in enumerate(rows) if any(q[0]==t for q in row['proposals'])]
        if not ids:break
        ws=[worlds[i] for i in ids];prefix=p['collate']([w.state for w in ws],'cuda')
        with torch.no_grad():ref=e.prior(prefix);batch=p['public_batch'](ws,prefix,ref)
        _,mu,ls=e.actor(batch);terms=[]
        for j,i in enumerate(ids):
            a=AA.index(task['fixed_sequence'][t]);local=[]
            for q in rows[i]['proposals']:
                if q[0]!=t:continue
                lp=T['base_logprob'](mu[j,a],ls[j,a],torch.tensor(q[2],device='cuda',dtype=torch.float64),c['mask_for_step'](t));local.append(lp);maxerr=max(maxerr,abs(float(lp.detach())-q[4]))
            terms.append(coef[i]*torch.stack(local).sum()/16)
        torch.stack(terms).sum().backward()
        for i in ids:
            if t<len(rows[i]['geometry']):worlds[i].state=S['append'](worlds[i].state,task['fixed_sequence'][t],rows[i]['geometry'][t])
    assert maxerr<5e-5;return flat(e),maxerr

def main():
    assert read(R/'FILTER_AND_GRADIENT_CPU_TESTS.json')['PASS'];N['guard']()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    e,_=P['K']['load_engine']('C');e.actor.requires_grad_(True).eval();initial=c['state_digest'](e.actor.state_dict());packet=read(O/'production/rollouts/u000001.json')
    assert initial==packet['behavior']
    if not (R/'HISTORICAL_GRADIENT_REPLAY.json').exists():
        a,errA=component(e,packet['task'],packet['records'],'reward');b,errB=component(e,packet['task'],packet['records'],'KL')
        e.actor.zero_grad(set_to_none=True);P['K']['gradients'](e,packet['task'],packet['records'],P['SPEC']['trace_KL_beta']);actual=flat(e)
        maxdiff=float((a+b-actual).abs().max());assert torch.allclose(a+b,actual,atol=2e-5,rtol=2e-5)
        norm=float(torch.nn.utils.clip_grad_norm_(e.actor.parameters(),float('inf')));historical=read(O/'production/commits/u000001.json')['logical']['gradient_norm'];assert abs(norm-historical)<1e-4
        save(R/'HISTORICAL_GRADIENT_REPLAY.json',dict(PASS=True,update=1,optimizer_updates=0,exact_behavior_fingerprint=True,group_size=16,
            combined_norm=norm,historical_norm=historical,reward_component_norm=float(a.norm()),reference_KL_component_norm=float(b.norm()),
            component_cosine=float(torch.dot(a,b)/(a.norm()*b.norm())),max_component_sum_error=maxdiff,max_logprob_error=max(errA,errB),
            global_clip_factor=5/norm,latent_dimensions_per_trajectory=float(np.mean([sum(sum(c['mask_for_step'](q[0])) for q in row['proposals']) for row in packet['records']])),accepted_full_trajectory_active_dimensions=186,reduction='mean16 trajectories, sum64 residue scores including rejected draws; no hidden token mean',
            original_actor_unchanged=c['state_digest'](e.actor.state_dict())==initial,J1_unchanged=True))
        del a,b,actual;e.actor.zero_grad(set_to_none=True);e.actor.requires_grad_(False)
    e.actor.zero_grad(set_to_none=True);e.actor.requires_grad_(False)
    controls=[];tasks=read(P['V93']/'manifests/FUTURE_FIXED_TRAIN_POOL.json')['tasks']
    for task in tasks[::2][:4]:
        seeds=[S['ns']('V94R_TRAIN_CONTROL',task['goal']['goal_id'],i) for i in range(16)]
        old=P['sample'](e,task,'C',16,seeds);new=N['sample'](e,task,'C',16,seeds)
        assert all(not (r['metrics']['complete'] and r['metrics']['geometry']['longrange_contact_degree']>N['S']['SPEC']['packing_limits']['64'][1]) for r in new)
        controls.append(dict(task=task,old_records=old,new_records=new))
    save(R/'TRAIN_CONTROLLER_CONTROLS.json',dict(records=controls,neural_updates=0,TRAIN_only=True,paired_seeds=True))
    idx=read(O/'production/RECOVERY_INDEX.json')['latest'];cp=O/'production'/idx['relative'];assert P['sha'](cp)==idx['SHA256'];state=torch.load(cp,map_location='cpu',weights_only=False);e.actor.load_state_dict(state['actor'],strict=True);del state
    fp=c['state_digest'](e.actor.state_dict());failures=read(R/'MATCHED_TRAJECTORY_FAILURES.json')['records'];selected=[x for x in failures if x['new_failure']];groups=[];seen=set();counter=[]
    for item in selected:
        task=item['task'];goal=task['goal'];key=goal['goal_id'];ci=item['candidate_index'];oldpacket=read(Path(item['prefix_source']));oldrow=oldpacket['records'][ci]
        if key not in seen:
            seen.add(key);seeds=[S['ns']('CANDIDATE',key,j) for j in range(8)];new=N['sample'](e,task,'C',8,seeds)
            groups.append(dict(task=task,old_records=oldpacket['records'],new_records=new))
        doom=item['update500']['first_proven_overbudget_residue_1based']
        if doom is None:continue
        t=doom-1;prefix=S['reconstruct'](oldrow,64,t);w=S['world'](goal,prefix)
        with torch.no_grad():_,mu,ls,ref,_=e.distribution([w],'D_RL')
        aa=AA.index(task['fixed_sequence'][t]);alternatives=[];accepted=None
        for k in range(8):
            rng=np.random.default_rng(S['ns']('STEP',oldrow['seed'],t,k));eps=torch.tensor(rng.standard_normal(3),device='cuda',dtype=torch.float64)
            geo,diag=S['corrected_action'](mu[0,aa],ls[0,aa],c['mask_for_step'](t),eps)
            after,check=N['S']['F']['extend_checked'](prefix,task['fixed_sequence'][t],geo.cpu().tolist())
            alternatives.append(dict(k=k,geometry=geo.cpu().tolist(),accepted=check['accepted'],reasons=check['reasons'],packing=check.get('packing_after')))
            if check['accepted'] and accepted is None:accepted=after
        rows=[]
        if accepted is not None:
            N['prepare_sampler']();seeds=[S['ns']('V94R_SUFFIX_CONFIRM',key,ci,j) for j in range(8)]
            rows=N['S']['generate'](P['K']['LiveActor'](e),task,'C',8,state=accepted,seeds=seeds)
        counter.append(dict(goal_id=key,candidate=ci,prefix_length=t,prefix_source=item['prefix_source'],
            alternatives=alternatives,continuations=rows,conditional_on_first_valid_alternative=True,
            original_dead_prefix_final_degree_lower_bound=item['update500']['prefix_curve'][t]['final_degree_lower_bound'],
            numerical_note='single-prefix counterfactual forward may differ in FP rounding from original batch8; not a cached behavior-likelihood replay'))
    save(R/'AFFECTED_TRAIN_COUNTERFACTUALS.json',dict(groups=groups,first_crossing_interventions=counter,
        policy_checkpoint_SHA256=idx['SHA256'],policy_fingerprint=fp,no_optimizer_updates=True,all_goals_reserved_TRAIN=True,
        no_prospective_validation=True,one_step_feasibility_not_goal_reachability=True))
    assert c['state_digest'](e.actor.state_dict())==fp;e.prior.verify();assert c['state_digest'](e.parent.state_dict())==e.fp
    save(R/'GPU_ENGINEERING_COMPLETE.json',dict(PASS=True,optimizer_updates=0,TRAIN_control_tasks=len(controls),affected_tasks=len(groups),
        packing_interventions=len(counter),peak_GPU_bytes=torch.cuda.max_memory_allocated(),J1_unchanged=True,actors_unchanged=True))
    print('GPU_ENGINEERING_PASS',len(controls),len(groups),len(counter),flush=True)
if __name__=='__main__':main()
