"""D-only fixed-AA geometry RL; full-trace replay and atomic dual-slot recovery.

Qualification uses the exact production update, full actor and fresh Adam. No
SAC, critic, AA sampling, TEST, target coordinates, or changed protein objective.
"""
import argparse, collections, copy, csv, hashlib, json, math, os, random, runpy
import shutil, subprocess, sys, time, uuid
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/mnt/d/Simone/proteinGen_data/e017_v94r2_corrected_geometry_rl_20261025_v2')
V93=ROOT.parent/'e017_v93_regional_controllability_20261024_v1'
K=runpy.run_path(str(V93/'source/scripts/e017_v93/campaign.py'))
S=K['S']; c=K['c']; p=K['p']; AA=K['AA']; sha=K['sha']; read=K['read']
from protein_joint_sac.crisp_v3.recovery import (atomic_checkpoint, atomic_json,
    state_digest, serialized_size, StoragePolicy, fsync_dir)
CPU=K['cpu']; SPEC=read(ROOT/'TRAINING_CONFIGURATION_AND_SEEDS.json')
S=runpy.run_path(str(ROOT/'source/scripts/e017_v94r/sampling.py'))
K['gradients']=runpy.run_path(str(ROOT/'source/scripts/e017_v94r/gradients.py'))['gradients']
RESERVE=2*2**30; HINT=224*2**20

def compact(path,obj):
    path=Path(path); tmp=path.with_name(path.name+'.part')
    assert not path.is_symlink() and not tmp.is_symlink()
    with tmp.open('w') as f:
        json.dump(obj,f,separators=(',',':'),allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path); fsync_dir(path.parent)

def guard():
    assert shutil.disk_usage(ROOT).free>RESERVE+2*HINT+128*2**20,'STORAGE_STOP'
    assert not (ROOT/'control/STOP').exists(),'COOPERATIVE_STOP'
    if torch.cuda.is_initialized():
        assert torch.cuda.max_memory_allocated()<6*2**30,'VRAM_STOP'

def source_binding():
    return {str(q.relative_to(ROOT/'source')):sha(q) for q in sorted(
        [q for folder in ['e017_v94r','e017_v94r2'] for q in (ROOT/'source/scripts'/folder).glob('*.py')])}

def bindings():
    return dict(source=source_binding(),V93_source_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=V93/'source',text=True).strip(),
        V94_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'source',text=True).strip(),
        configuration_SHA256=sha(ROOT/'TRAINING_CONFIGURATION_AND_SEEDS.json'),
        TRAIN_SHA256=sha(V93/'manifests/FUTURE_FIXED_TRAIN_POOL.json'),
        development_SHA256=sha(V93/'manifests/TRAIN_TASKS.json'),
        confirmation_SHA256=sha(V93/'manifests/PROSPECTIVE_FIXED.json'),
        reward_SHA256=sha(ROOT.parent/'e017_v91_ordered_goal_recovery_20261023_v1/REWARD_SPECIFICATION.json'),
        J1_SHA256=c['J1SHA'],parent_SHA256=c['PSHA'])

def verify_inputs():
    old=read(ROOT/'INPUT_VERIFICATION.json'); assert old['PASS']
    for label,meta in old['models'].items(): assert sha(Path(meta['path']))==meta['SHA256'],label
    for name,expected in old['verified_SHA256'].items(): assert sha(Path(name))==expected,name
    tasks=read(V93/'manifests/FUTURE_FIXED_TRAIN_POOL.json')['tasks']
    refpath=ROOT.parent/'e017_v9_property_steering_20261023_v1/manifests/TRAIN_REFERENCE.json'
    refs={r['identity']:r for r in read(refpath)['records']}
    ids=set(); groups=set()
    for t in tasks:
        identity=t['source_reference_identity']; ref=refs[identity]; g=t['goal']
        assert ref['split']=='TRAIN' and len(t['fixed_sequence'])==64
        assert t['fixed_sequence']==ref['sequence'] and set(ref['sequence'])<=set(AA)
        assert g['protected_group']==ref['group'] and t['native_witness_success']
        assert t['source_reference']==str(refpath)
        ids.add(identity); groups.add(g['protected_group'])
    assert len(ids)==len(groups)==434 and len(tasks)==868
    dev=read(V93/'manifests/TRAIN_TASKS.json')['tasks']
    val=read(V93/'manifests/PROSPECTIVE_FIXED.json')['tasks']
    assert not groups & {t['goal']['protected_group'] for t in dev+val}
    assert sha(ROOT.parent/'e017_v91_ordered_goal_recovery_20261023_v1/REWARD_SPECIFICATION.json')=='989f5d83310b77a891b47e903ee5a36b96b956e7cd57cb2d8ef98e07df608f92'
    return dict(PASS=True,TRAIN_identities=434,TRAIN_groups=434,TRAIN_tasks=868,bindings=bindings())

def prepare_sampler():
    # Override only operational root/finite-job guards; preserve sampling law.
    fn=S['generate']; fn.__wrapped__.__globals__['guard']=guard
    fn.__wrapped__.__globals__['START']=time.perf_counter()
    S['COUNTS'].clear()

def sample(e,task,policy,n,seeds):
    prepare_sampler(); return S['generate'](K['LiveActor'](e),task,policy,n,seeds=seeds)

def stats(rows):
    m=[z['metrics'] for z in rows]
    return dict(attempts=len(rows),complete=sum(x['complete'] for x in m),
        full=sum(x['success'] for x in m),hard=sum(x['hard_valid'] for x in m),
        backbone=sum(x['backbone_valid'] for x in m),
        accepted_actions=sum(len(z['geometry']) for z in rows),
        residue_actions=sum(max(0,len(z['geometry'])-1) for z in rows),
        proposals=sum(len(z['proposals']) for z in rows),
        rejected=sum(not q[6] for z in rows for q in z['proposals']),
        aborts=sum(z['termination'] is not None for z in rows),
        reward_mean=float(np.mean([x['reward'] for x in m])),
        reward_SD=float(np.std([x['reward'] for x in m])),
        geometry_KL=float(np.mean([v for z in rows for v in z['geometry_KL']])),
        trace_log_ratio=float(np.mean([z['trace_logprob']-z['reference_trace_logprob'] for z in rows])),global_contact_violations=sum(z['metrics']['complete'] and z['metrics']['geometry']['longrange_contact_degree']>S['SPEC']['packing_limits']['64'][1] for z in rows),packing_proposals_rejected=sum('provably_exceeds_final_contact_budget' in q[8] for z in rows for q in z['proposals']))

class Runtime:
    def __init__(self,root):
        self.root=Path(root); self.root.mkdir(exist_ok=True)
        assert self.root.resolve().is_relative_to(ROOT.resolve())
        for n in ['rollouts','commits','checkpoints','diagnostics']: (self.root/n).mkdir(exist_ok=True)
        self.binding=bindings(); self.policy=StoragePolicy(checkpoint_hint_bytes=HINT)
        self.e,_=K['load_engine']('C'); self.e.actor.requires_grad_(True).eval()
        assert self.e.fp=='8e21d75c2fe8f78ee4fd355183f56913f975f5f3cfc38240cc5271cd60c74373'
        assert self.e.jfp=='57e3cb301e240fdce05fcc012b186faba1b9c410c930ca37dc6bafc2a2511f4f'
        self.opt=torch.optim.AdamW(self.e.actor.parameters(),lr=SPEC['peak_lr'],weight_decay=0.,betas=(.9,.999),eps=1e-8)
        # Full-size fresh zero moments make measured staging conservative from0.
        for param in self.e.actor.parameters():
            self.opt.state[param]=dict(step=torch.tensor(0.),exp_avg=torch.zeros_like(param),exp_avg_sq=torch.zeros_like(param))
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.opt,lambda k: min(1.,(k+1)/100)*(.1+.45*(1+math.cos(math.pi*max(0,k-4000)/1000))) if k>=4000 else min(1.,(k+1)/100))
        tasks=read(V93/'manifests/FUTURE_FIXED_TRAIN_POOL.json')['tasks']
        self.ids=sorted({t['source_reference_identity'] for t in tasks})
        self.lookup={i:[t for t in tasks if t['source_reference_identity']==i] for i in self.ids}
        self.order=np.random.default_rng(SPEC['seed']).permutation(len(self.ids)).tolist()
        self.update=0; self.counters=collections.Counter(); self.best=None; self.baseline=None
        torch.manual_seed(SPEC['seed']); torch.cuda.manual_seed_all(SPEC['seed'])
        np.random.seed(SPEC['seed']); random.seed(SPEC['seed'])
        self.epoch_start=time.perf_counter()
        if (self.root/'RECOVERY_INDEX.json').exists(): self.restore()
        else: self.publish('INITIAL')

    def task(self,u):
        i=self.ids[self.order[(u-1)%len(self.order)]]
        return self.lookup[i][((u-1)//len(self.order))%len(self.lookup[i])]

    def state(self):
        u=self.update
        return dict(schema='V94R_PACKING_FULL_TRACE_RECOVERY_1',actor=CPU(self.e.actor.state_dict()),
            optimizer=CPU(self.opt.state_dict()),scheduler=copy.deepcopy(self.scheduler.state_dict()),
            AMP=None,RNG=K['rng'](),update=u,counters=dict(self.counters),identity_order=self.order,
            next_update=u+1,next_identity=self.task(u+1)['source_reference_identity'],
            next_goal_id=self.task(u+1)['goal']['goal_id'],next_seeds=[S['ns']('SUBSTANTIAL',SPEC['seed'],u+1,i) for i in range(16)],
            binding=self.binding,beta=SPEC['trace_KL_beta'],configuration=read(ROOT/'TRAINING_CONFIGURATION_AND_SEEDS.json'),
            baseline=self.baseline,best=self.best,reward_normalization='frozen reward; independent RLOO; no running moments',
            filter_state='stateless STEP/candidate/rejection seeds; pending rollouts/ and commits/',
            journal_head_SHA256=sha(self.root/f'commits/u{u:06}.json') if u and (self.root/f'commits/u{u:06}.json').exists() else None)

    def restore(self):
        idx=read(self.root/'RECOVERY_INDEX.json'); r=idx['latest']; path=self.root/r['relative']
        assert path.parent==self.root/'checkpoints' and not path.is_symlink()
        assert sha(path)==r['SHA256'],'checkpoint hash mismatch'
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state_digest(state)==r['state_digest'] and state['binding']==self.binding,'checkpoint binding mismatch'
        self.e.actor.load_state_dict(state['actor'],strict=True); self.opt.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler']); K['setrng'](state['RNG'])
        self.order=state['identity_order']; self.update=state['update']; self.counters=collections.Counter(state['counters'])
        self.baseline=state['baseline']; self.best=state['best']
        assert self.task(self.update+1)['goal']['goal_id']==state['next_goal_id']
        if self.update: assert sha(self.root/f'commits/u{self.update:06}.json')==state['journal_head_SHA256']
        assert state_digest(self.state())==r['state_digest'],'complete restore mismatch'

    def semantic(self):
        s=self.state(); s.pop('journal_head_SHA256'); return state_digest(s)

    def publish(self,reason,fault=None):
        state=self.state(); n=serialized_size(state); self.policy.require(self.root,n+HINT)
        old=read(self.root/'RECOVERY_INDEX.json') if (self.root/'RECOVERY_INDEX.json').exists() else None
        slot=1-int(old['latest']['slot']) if old else 0
        incoming=self.root/'checkpoints'/('incoming_'+uuid.uuid4().hex+'.pt')
        if fault=='partial':
            # Interrupt the actual full-state Torch writer, not a mock header.
            import builtins
            original=builtins.open;target=str(incoming)+'.part'
            compact(self.root/'INTERRUPTED_WRITE.json',dict(partial=target,previous_index_SHA256=sha(self.root/'RECOVERY_INDEX.json')))
            class Interrupt:
                def __init__(self,f):self.f=f;self.n=0
                def __getattr__(self,n):return getattr(self.f,n)
                def __enter__(self):return self
                def __exit__(self,*a):return self.f.__exit__(*a)
                def write(self,b):
                    n=self.f.write(b);self.n+=n
                    if self.n>=8*2**20:self.f.flush();os.fsync(self.f.fileno());os._exit(75)
                    return n
            def opened(file,*a,**kw):
                f=original(file,*a,**kw)
                return Interrupt(f) if str(file)==target and a and a[0]=='xb' else f
            builtins.open=opened
        record=atomic_checkpoint(incoming,state,self.policy)
        path=self.root/'checkpoints'/f'recovery_{slot}.pt'
        # Replace only this run's inactive rolling slot; linked named evidence
        # is a different directory entry and its immutable inode is preserved.
        assert not path.is_symlink()
        os.replace(incoming,path); fsync_dir(path.parent)
        latest=dict(relative=str(path.relative_to(self.root)),slot=slot,SHA256=record['sha256'],
            bytes=record['bytes'],state_digest=record['state_digest'],update=self.update,reason=reason)
        compact(self.root/'RECOVERY_INDEX.json',dict(schema='V94R_DUAL_SLOT_1',latest=latest))
        compact(self.root/'LIVE_STATUS.json',dict(state='STOPPED' if 'STOP' in reason or 'PAUSE' in reason else 'COMPLETED' if 'COMPLETED' in reason else 'RUNNING',PID=os.getpid(),update=self.update,counters=dict(self.counters),latest_checkpoint=latest,reason=reason))
        if self.update in SPEC['named_updates'] or reason=='DEVELOPMENT_BEST':
            label='milestone' if self.update in SPEC['named_updates'] else 'development_best'
            target=path.parent/f'{label}_{self.update:05}.pt'
            if not target.exists():
                try: os.link(path,target)
                except OSError: self.policy.require(self.root,n); shutil.copyfile(path,target)
                assert sha(target)==record['sha256']
                compact(target.with_suffix('.json'),dict(latest,relative=str(target.relative_to(self.root))))
        return latest

    def step(self,u,fault=None):
        assert u==self.update+1; guard(); start=time.perf_counter(); task=self.task(u)
        rp=self.root/f'rollouts/u{u:06}.json'; behavior=c['state_digest'](self.e.actor.state_dict())
        seeds=[S['ns']('SUBSTANTIAL',SPEC['seed'],u,i) for i in range(16)]
        if rp.exists():
            packet=read(rp)
            assert packet['behavior']==behavior and packet['seeds']==seeds and packet['task']==task and packet['binding']==self.binding,'pending trace mismatch'
            rows=packet['records']
        else:
            rows=sample(self.e,task,'C',16,seeds)
            compact(rp,dict(update=u,behavior=behavior,seeds=seeds,task=task,binding=self.binding,records=rows))
        if fault=='rollout': os._exit(73)
        s=stats(rows); assert all(math.isfinite(v) for v in s.values())
        self.opt.zero_grad(set_to_none=True); self.e.actor.eval()
        rewards,error,probes=K['gradients'](self.e,task,rows,SPEC['trace_KL_beta'])
        assert not any(p_.grad is not None and bool(p_.grad.abs().max()) for name,p_ in self.e.actor.named_parameters() if name.startswith('aa.'))
        norm=float(torch.nn.utils.clip_grad_norm_(self.e.actor.parameters(),5.)); assert math.isfinite(norm),'nonfinite gradient'
        self.opt.step(); self.scheduler.step(); self.e.actor.eval(); maxstep=0.
        with torch.no_grad():
            for ws,oldmu,oldls,a in probes:
                _,mu,ls,_,_=self.e.distribution(ws,'D_RL'); mask=torch.tensor(c['mask_for_step'](ws[0].state.t),device='cuda')
                new=ls[:,a].double().clamp(-5,2); old=oldls[:,a].double().clamp(-5,2)
                kl=new-old+.5*((old.exp()/new.exp()).square()+((oldmu[:,a].double()-mu[:,a].double())/new.exp()).square())-.5
                maxstep=max(maxstep,float((kl*mask).sum(-1).max()))
        # Uncommitted unsafe step is never published. Resume returns to last
        # verified recovery and journals replay prior accepted work only.
        assert math.isfinite(maxstep) and maxstep<=.02,('STEP_KL_STABILITY_STOP',maxstep)
        assert s['geometry_KL']<=1,('GEOMETRY_KL_STABILITY_STOP',s['geometry_KL'])
        self.update=u
        for k in ['attempts','complete','accepted_actions','residue_actions','proposals','rejected','aborts']: self.counters[k]+=s[k]
        self.counters['optimizer_updates']=u
        semantic=self.semantic(); cp=self.root/f'commits/u{u:06}.json'
        logical=dict(update=u,identity=task['source_reference_identity'],goal_id=task['goal']['goal_id'],
            rollout_SHA256=sha(rp),training_state_digest=semantic,stats=s,gradient_norm=norm,
            step_KL=maxstep,replay_logprob_maxerror=error,diagnostics=self.e.last_gradient_diagnostics,lr=self.scheduler.get_last_lr()[0])
        if cp.exists():
            old=read(cp); assert old['logical']==logical,('committed replay mismatch',u)
        else: compact(cp,dict(logical=logical,seconds=time.perf_counter()-start,D_free_bytes=shutil.disk_usage(ROOT).free,
            peak_GPU_bytes=torch.cuda.max_memory_allocated(),UTC=time.time()))
        compact(self.root/'LIVE_STATUS.json',dict(state='RUNNING',PID=os.getpid(),update=u,counters=dict(self.counters),
            latest_checkpoint=read(self.root/'RECOVERY_INDEX.json')['latest'],last_commit_SHA256=sha(cp),last_update_seconds=time.perf_counter()-start))
        self.e.prior.verify(); assert c['state_digest'](self.e.parent.state_dict())==self.e.fp
        print(json.dumps(dict(update=u,**s,grad_norm=norm,step_KL=maxstep,seconds=time.perf_counter()-start)),flush=True)
        if fault=='commit': os._exit(74)
        if fault=='partial': self.publish('FAULT_PARTIAL',fault='partial')
        return logical

    def replay(self):
        paths=sorted((self.root/'commits').glob('u*.json'))
        end=int(paths[-1].stem[1:]) if paths else 0
        for u in range(self.update+1,end+1): self.step(u)
        assert len(paths)==end,'journal gap'
        return end

def qualify_child(args):
    rt=Runtime(Path(args.run)); rt.replay()
    if args.fault=='partial' and rt.update==1:rt.publish('FAULT_PARTIAL',fault='partial')
    for u in range(rt.update+1,3): rt.step(u,args.fault if u==1 else None)
    latest=rt.publish('QUALIFICATION_BOUNDARY')
    # Repeated publication is tested deliberately; old immutable-latest bug.
    latest2=rt.publish('QUALIFICATION_REPEAT_PUBLICATION')
    semantic=rt.semantic()
    draws=dict(torch=torch.rand(8).tolist(),cuda=torch.rand(8,device='cuda').cpu().tolist(),numpy=np.random.random(8).tolist(),python=[random.random() for _ in range(8)])
    compact(rt.root/'QUALIFICATION_FINAL.json',dict(PASS=True,update=rt.update,counters=dict(rt.counters),
        state_digest=latest2['state_digest'],semantic_state=semantic,next_RNG_draws=draws,
        checkpoint=latest2,J1_unchanged=True,parent_unchanged=True))

def production(args):
    authorization=read(Path(args.authorization)) if args.authorization else {}
    assert authorization.get('explicit_review_authorized') is True and authorization.get('configuration_SHA256')==sha(ROOT/'TRAINING_CONFIGURATION_AND_SEEDS.json'),'FULL PRODUCTION REQUIRES SEPARATE USER REVIEW; NOT AUTHORIZED BY V94R ENGINEERING'
    release=read(ROOT/'qualification/RELEASE.json')
    assert release['PASS'] and release['binding']==bindings(),'unqualified production'
    rt=Runtime(ROOT/'production'); rt.replay()
    # Parameterized relocation/seed changes preserve the reviewed Runtime.
    # Qualify the actual fresh schema/optimizer/RNG instance before sampling.
    receipt=ROOT/'ACTUAL_INITIAL_RECOVERY_ROUNDTRIP.json'
    if not receipt.exists():
        assert rt.update==0 and not rt.counters
        initial=state_digest(rt.state())
        def draws():return dict(torch=torch.rand(8).tolist(),cuda=torch.rand(8,device='cuda').cpu().tolist(),numpy=np.random.random(8).tolist(),python=[random.random() for _ in range(8)])
        first=draws();rt.restore();assert state_digest(rt.state())==initial;second=draws();assert first==second;rt.restore()
        assert c['state_digest'](CPU(rt.e.actor.state_dict()))=='884b1e016135f4bf4b3594b6aefc81c1594a525186e09091ad5bd5c023781f80'
        assert all(float(x['step'])==0 and not x['exp_avg'].count_nonzero() and not x['exp_avg_sq'].count_nonzero() for x in rt.opt.state.values())
        rt.e.prior.verify()
        compact(receipt,dict(PASS=True,optimizer_steps=0,full_actual_state_roundtrip=True,next_CPU_CUDA_numpy_python_draws_equal=True,actor_initializer_fingerprint=c['state_digest'](CPU(rt.e.actor.state_dict())),behavior_OrderedDict_fingerprint=c['state_digest'](rt.e.actor.state_dict()),fingerprint_container='plain dict matching certified serialized state',J1_frozen=True,checkpoint=read(rt.root/'RECOVERY_INDEX.json')['latest'],qualification_transfer_SHA256=sha(ROOT/'qualification/RELEASE.json')))
    evaluator=runpy.run_path(str(ROOT/'source/scripts/e017_v94r/evaluation.py'))
    if rt.baseline is None:
        rt.baseline=evaluator['development'](rt,0)
        rt.best=dict(update=0,**rt.baseline['filtered']);rt.publish('DEVELOPMENT_BEST')
        initial=rt.baseline['filtered']
        assert initial['hard_valid']>=.91875 and initial['backbone_valid']>=.95 and initial['geometry_KL_per_active']<=1 and initial['structural_pair_RMSD']>=12.449118749897715,'INITIAL_ABSOLUTE_VALIDITY_KL_DIVERSITY_ADMISSION'
        compact(ROOT/'FROZEN_INITIALIZATION_BASELINE.json',dict(update=0,summary=rt.baseline,original_V94=dict(hard_valid=.96875,backbone_valid=1.,structural_pair_RMSD=15.561398437372143),baseline_will_not_be_reset=True,abort_gate='no new separate abort rule; all aborts count against unchanged hard-validity gate'))
    try:
        # Reconcile a crash after commit before the checkpoint/evaluation callback.
        milestones=set(SPEC['development_updates'])
        if rt.update in milestones and not (rt.root/f'diagnostics/DEVELOPMENT_{rt.update:05}.json').exists():
            raise RuntimeError('COMMITTED_MILESTONE_REQUIRES_CALLBACK_RECONCILIATION')
        for u in range(rt.update+1,5001):
            if (ROOT/'control/STOP').exists():
                rt.publish('COOPERATIVE_PAUSE'); return
            rt.step(u)
            if u%50==0: rt.publish('ROLLING')
            if u in milestones:
                report=evaluator['development'](rt,u); a=report['filtered']; b=rt.baseline['filtered']
                safe=(a['hard_valid']>=max(b['hard_valid'],.96875)-.05 and a['backbone_valid']>=max(b['backbone_valid'],1.)-.05
                    and a['geometry_KL_per_active']<=1 and a['structural_pair_RMSD']>=.8*max(b['structural_pair_RMSD'],15.561398437372143))
                if not safe:
                    evaluator['curves'](rt.root)
                    rt.publish('TRAINING_STABILITY_MONITOR_STOP')
                    compact(rt.root/'STOP_RECORD.json',dict(outcome='RL_STABILITY_NO_GO',update=u,report=report,baseline=rt.baseline))
                    return
                if (a['K1'],a['K8'],a['hard_valid'])>(rt.best['K1'],rt.best['K8'],rt.best['hard_valid']):
                    rt.best=dict(update=u,**a); rt.publish('DEVELOPMENT_BEST')
            if u%50==0: evaluator['curves'](rt.root)
        rt.publish('COMPLETED_DECLARED_BUDGET_PAUSED')
        compact(rt.root/'TRAINING_COMPLETED.json',dict(update=rt.update,counters=dict(rt.counters),best=rt.best,checkpoint=read(rt.root/'RECOVERY_INDEX.json')))
    except BaseException as error:
        # No recovery checkpoint may contain an uncommitted unsafe step.
        compact(rt.root/'EXECUTION_FAILURE.json',dict(error=repr(error),committed_update=rt.update,PID=os.getpid(),resume_automatic=False))
        raise

def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['verify','qualify-child','production'])
    ap.add_argument('--authorization'); ap.add_argument('--run'); ap.add_argument('--fault',choices=['rollout','commit','partial']); args=ap.parse_args()
    if args.mode=='verify': print(json.dumps(verify_inputs()),flush=True)
    elif args.mode=='qualify-child': qualify_child(args)
    else: production(args)
if __name__=='__main__': main()
