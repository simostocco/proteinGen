"""One bounded TRAIN diagnosis and qualification program; never production."""
import collections, copy, fcntl, math, runpy, subprocess, time
import numpy as np
import torch
from common96 import *
from algorithm import augment, install


def flat(actor):
    return torch.cat([(p.grad.detach().cpu().flatten() if p.grad is not None else torch.zeros(p.numel())) for p in actor.parameters()])


def diagnostics():
    lock=(ROOT/'control/ENGINEERING.lock').open('a+');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not read(ROOT/'TRAINING_CONFIGURATION_AND_SEEDS.json')['production_authorized']
    assert not subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip(),'CONFLICTING_GPU_WORK'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    P,F,compat=load();manifest=read(ROOT/'TRAIN_DIAGNOSTIC_MANIFEST.json');groups=manifest['groups']
    e,_=P['K']['load_engine']('C');attempts=0;panels={};observation=[];start=time.perf_counter()
    def generate(task,policy,seeds,state=None):
        nonlocal attempts
        assert attempts+len(seeds)<=800
        P['prepare_sampler']()
        rows=P['S']['generate'](P['K']['LiveActor'](e),task,policy,len(seeds),state=state,seeds=seeds)
        attempts+=len(seeds)
        return rows
    for update in [0,5000]:
        F['checkpoint'](e,update);e.actor.eval().requires_grad_(False)
        for gi,group in enumerate(groups):
            task=group['base'];seeds=manifest['seeds'][gi]
            for mode in ['B','C']:
                result={}
                for tag,t in [('base',task)]+list(group['alternatives'].items()):
                    assert t['fixed_sequence']==task['fixed_sequence'] and t['goal']['motif']==task['goal']['motif']
                    rows=generate(t,mode,seeds);result[tag]=rows
                    write(f'diagnostics/swap_{update}_{gi}_{mode}_{tag}.json.gz',dict(task=t,records=rows))
                panels[(update,gi,mode)]=result
                print('SWAP',update,gi,mode,'attempts',attempts,flush=True)
            base=panels[(update,gi,'B')]['base'][0]
            for t in [0,8,32,48]:
                state=P['S']['reconstruct'](base,64,t)
                worlds=[P['S']['world'](z['goal'],state) for z in [task,group['alternatives']['burial'],group['alternatives']['exposure']]]
                with torch.no_grad():
                    _,mu,ls,ref,batch=e.distribution(worlds,'D_RL')
                assert torch.equal(batch['goals'][0],batch['goals'][1]),'BURIAL_SWAP_CHANGED_MOTIF_TOKENS'
                assert not torch.equal(batch['context'][0],batch['context'][1]),'BURIAL_NOT_OBSERVABLE'
                aa=P['AA'].index(task['fixed_sequence'][min(t,63)])
                prefix=P['p']['collate']([worlds[0].state],'cuda')
                with torch.no_grad():reference=e.prior(prefix);b=P['p']['public_batch']([worlds[0]],prefix,reference)
                b['context'].requires_grad_(True);_,m0,l0=e.actor(b)
                jac=torch.autograd.grad(m0[0,aa].sum()+l0[0,aa].sum(),b['context'])[0][0]
                observation.append(dict(update=update,identity=task['source_reference_identity'],prefix=t,burial_context_changed_columns=torch.where(batch['context'][0]!=batch['context'][1])[0].cpu().tolist(),burial_mu_L2=float((mu[1,aa]-mu[0,aa]).norm()),burial_ls_L2=float((ls[1,aa]-ls[0,aa]).norm()),same_motif_exposure_mu_L2=float((mu[2,aa]-mu[0,aa]).norm()),burial_input_gradient_norm=float(jac[[2,6,7]].norm()),exposure_input_gradient_norm=float(jac[[1,4,5]].norm()),burial_has_goal_token=False,exposure_has_motif_anchored_tokens=True))
        e.prior.verify();assert P['c']['state_digest'](e.parent.state_dict())==e.fp
    write('OBSERVABILITY_EVIDENCE.json',dict(probes=observation,all_burial_observations_distinct=True,any_burial_input_gradient=any(x['burial_input_gradient_norm']>0 for x in observation),same_motif_exposure_control=True))
    # Registered S, descriptive only: n=2 TRAIN groups is not scientific power.
    swaps=[]
    for (u,gi,mode),result in panels.items():
        for tag in ['burial','exposure']:
            region='buried' if tag=='burial' else 'exposed';g0=groups[gi]['base']['goal'];g1=groups[gi]['alternatives'][tag]['goal'];a,b=g0[region];c,d=g1[region];values=[];valid=0
            for x,y in zip(result['base'],result[tag]):
                v=0.
                if x['metrics']['hard_valid'] and y['metrics']['hard_valid']:
                    ex=np.array(x['metrics']['exposure']);ey=np.array(y['metrics']['exposure']);v=.5*((ex[a:b].mean()-ey[a:b].mean())+(ey[c:d].mean()-ex[c:d].mean()))
                    if tag=='burial':v=-v
                    valid+=1
                values.append(float(v))
            swaps.append(dict(update=u,group=gi,mode=mode,target=tag,S=float(np.mean(values)),valid_pairs=valid,pairs=8))
    write('GOAL_SWAP_EVIDENCE.json',dict(records=swaps,scientific_improvement_claim=False,all_sequences_motifs_thresholds_identical=True,TRAIN_groups=2))
    # Early-action influence: current R3 controller, same generated prefix and
    # matched suffix seeds. Never native prefixes or future target coordinates.
    F['checkpoint'](e,5000);branches=[]
    for gi,group in enumerate(groups):
        task=group['base'];base=panels[(5000,gi,'C')]['base'][0]
        for t in [8,32]:
            if len(base['sequence'])<t:
                branches.append(dict(group=gi,t=t,status='PREFIX_ABORTED'));continue
            state=P['S']['reconstruct'](base,64,t);world=P['S']['world'](task['goal'],state)
            with torch.no_grad():_,mu,ls,ref,_=e.distribution([world],'D_RL')
            aa=P['AA'].index(task['fixed_sequence'][t]);valid=[]
            for k in range(16):
                eps=torch.tensor(np.random.default_rng(ns('early action',gi,t,k)).standard_normal(3),device='cuda',dtype=torch.float64)
                geo,detail=P['S']['corrected_action'](mu[0,aa],ls[0,aa],P['c']['mask_for_step'](t),eps)
                after,check=P['S']['extend_checked'](state,task['fixed_sequence'][t],geo.cpu().tolist())
                if check['accepted']:valid.append((geo.cpu().tolist(),after))
                if len(valid)==2:break
            for bi,(action,after) in enumerate(valid):
                seeds=[ns('matched suffix',gi,t,j) for j in range(8)]
                rows=generate(task,'C',seeds,after)
                write(f'diagnostics/suffix_{gi}_{t}_{bi}.json.gz',dict(task=task,action=action,records=rows))
                branches.append(dict(group=gi,t=t,branch=bi,action=action,valid=sum(r['metrics']['hard_valid'] for r in rows),burial_means=[r['metrics'].get('buried_mean') for r in rows],burial_satisfied=sum(r['metrics'].get('buried_mean',1)<=task['goal']['burial_max'] for r in rows),full_success=sum(r['metrics']['success'] for r in rows),seeds=seeds))
    write('EARLY_ACTION_EVIDENCE.json',dict(branches=branches,all_prefixes_policy_generated=True,local_validity_not_global_feasibility=True,no_mathematical_infeasibility_claim=True))
    # Component score gradients on exactly the original first on-policy group.
    F['checkpoint'](e,0);e.actor.requires_grad_(True)
    old=read(R3/'production/rollouts/u000001.json');rr=old['records'];grads=[];gn=[]
    for component in range(6):
        modified=copy.deepcopy(rr)
        for row in modified:
            m=row['metrics'];part=[.5*float(m['success'])]+[.1*float(m['hard_valid'])*x for x in m.get('utilities',[0.]*5)]
            row['metrics']['reward']=part[component]
        e.actor.zero_grad(set_to_none=True);stage=[]
        handle=e.actor.geometry[-1].weight.register_hook(lambda g:stage.append(float(g.norm())))
        _,err,_=P['K']['gradients'](e,old['task'],modified,0.)
        handle.remove();v=flat(e.actor);grads.append(v);gn.append(dict(component=component,norm=float(v.norm()),stage_head_gradient_norm=stage,replay_maxerror=err))
    total=sum(grads);cos=[]
    for i in range(6):cos.append([float(torch.dot(grads[i],grads[j])/(grads[i].norm()*grads[j].norm()).clamp_min(1e-30)) for j in range(6)])
    write('COMPONENT_GRADIENT_EVIDENCE.json',dict(components=gn,cosine_matrix=cos,total_reward_gradient_norm=float(total.norm()),scope='one original on-policy TRAIN group at update0; diagnostics, not causal attribution across training',component_names=['success','exposure','burial','contrast','motif','compatibility'],optimizer_steps=0))
    e.actor.zero_grad(set_to_none=True);e.actor.requires_grad_(False)
    write('DIAGNOSTIC_GENERATION_COMPLETED.json',dict(attempts=attempts,seconds=time.perf_counter()-start,TRAIN_only=True,optimizer_updates=0,TEST=False))
    print('DIAGNOSIS_DONE',attempts,flush=True)


if __name__=='__main__':diagnostics()
