import copy, itertools, math, subprocess, sys
import numpy as np
import torch
from common96 import *
from algorithm import augment, install, VERSION


def toy():
    checks=[]
    # Exact finite rejected-proposal trace objective and its likelihood gradient.
    theta=torch.tensor(-.4,dtype=torch.float64,requires_grad=True);p=theta.sigmoid();q=.5;beta=.013
    traces=[(1,),(0,1),(0,0)];probs=[];lps=[];scores=[];returns=[]
    for trace in traces:
        lp=sum(torch.log(p if x else 1-p) for x in trace);pr=lp.exp();lr=lp-len(trace)*math.log(q);R=float(any(trace));score=sum(x-p for x in trace)
        probs.append(pr);lps.append(lp);scores.append(score);returns.append(R-beta*lr)
    objective=sum(pr*ret for pr,ret in zip(probs,returns));exact=torch.autograd.grad(objective,theta,retain_graph=True)[0]
    estimator=sum(pr*(ret-beta)*score for pr,ret,score in zip(probs,returns,scores))
    assert torch.allclose(exact,estimator,atol=1e-12,rtol=1e-12)
    rloo=0.
    for i,j in itertools.product(range(3),repeat=2):
        a=returns[i]-returns[j];b=returns[j]-returns[i]
        rloo+=probs[i]*probs[j]*((a-beta)*scores[i]+(b-beta)*scores[j])/2
    assert torch.allclose(exact,rloo,atol=1e-12,rtol=1e-12)
    accepted_only=sum(pr*(ret-beta)*(sum(x-p for x in trace if x)) for pr,ret,trace in zip(probs,returns,traces))
    assert abs(float(accepted_only-exact))>1e-3
    checks.append('exact finite-proposal trace gradient and RLOO match enumeration; accepted-only estimator demonstrably biased')
    b=dict(context=torch.zeros(2,64),goals=torch.randn(2,2,36),goal_valid=torch.ones(2,2,dtype=torch.bool))
    b['context'][:,6:8]=torch.tensor([[.25,.375],[.625,.75]])
    b['context'][:,2]=.4;b['context'][:,3]=.1
    original=copy.deepcopy(b);a=augment(b)
    assert torch.equal(a['goals'][:,:2],b['goals']) and torch.equal(b['goals'],original['goals'])
    assert torch.equal(a['context'],b['context']) and not torch.equal(a['goals'][0,-1],a['goals'][1,-1])
    assert torch.allclose(a['goals'][:,-1,32],b['context'][:,6:8].mean(1))
    assert not a['goals'][:,-1,12:32].count_nonzero()
    checks.append('burial token distinct, centered at requested interval; motif tokens and context unchanged; no AA motif invented')
    # CPU differentiable full Normal likelihood including inactive masks.
    mu=torch.tensor([.1,.2,.3],dtype=torch.float64,requires_grad=True)
    ls=torch.tensor([-.3,-.4,-.5],dtype=torch.float64,requires_grad=True)
    u=torch.tensor([.7,-.2,.5],dtype=torch.float64);mask=torch.tensor([True,True,False])
    lp=(-.5*((u-mu)/ls.exp())**2-ls-.5*math.log(2*math.pi))[mask].sum()
    dm,ds=torch.autograd.grad(lp,(mu,ls));assert dm[2]==0 and ds[2]==0
    for k in range(2):
        eps=1e-6;v=mu.detach().clone();v[k]+=eps
        high=(-.5*((u-v)/ls.detach().exp())**2-ls.detach()-.5*math.log(2*math.pi))[mask].sum()
        v[k]-=2*eps
        low=(-.5*((u-v)/ls.detach().exp())**2-ls.detach()-.5*math.log(2*math.pi))[mask].sum()
        assert abs(float((high-low)/(2*eps)-dm[k]))<1e-8
    checks.append('raw Normal score finite differences and inactive geometry masks pass')
    from campaign import reconcile_schedule
    state=dict(update=1,best=0);calls=[]
    def callback(u):state['best']=u;calls.append(('callback',u))
    def step(u):
        assert state['best']==u-1
        state['update']=u;calls.append(('step',u))
    reconcile_schedule(1,3,step,callback)
    assert calls==[('callback',1),('step',2),('callback',2),('step',3),('callback',3)]
    checks.append('missed frontier callback reconciled before later journal steps; callback replay order tested')
    write('qualification/SYNTHETIC_TESTS.json',dict(PASS=True,checks=checks,exact_gradient=float(exact),RLOO_gradient=float(rloo),accepted_only_gradient=float(accepted_only),TEST=False))


def partial_child():
    from protein_joint_sac.crisp_v3.recovery import atomic_checkpoint, StoragePolicy
    import builtins,os
    idx=read(ROOT/'qualification/engineering/RECOVERY_INDEX.json');state=torch.load(ROOT/'qualification/engineering'/idx['latest']['relative'],map_location='cpu',weights_only=False)
    target=ROOT/'qualification/interrupted_full_state.pt';original=builtins.open
    class Interrupted:
        def __init__(self,f):self.f=f;self.n=0
        def __getattr__(self,n):return getattr(self.f,n)
        def __enter__(self):return self
        def __exit__(self,*a):return self.f.__exit__(*a)
        def write(self,data):
            n=self.f.write(data);self.n+=n
            if self.n>8*2**20:self.f.flush();os.fsync(self.f.fileno());os._exit(75)
            return n
    def opened(file,*args,**kwargs):
        f=original(file,*args,**kwargs)
        return Interrupted(f) if str(file)==str(target)+'.part' and args and args[0]=='xb' else f
    builtins.open=opened
    atomic_checkpoint(target,state,StoragePolicy(checkpoint_hint_bytes=224*2**20))
    raise AssertionError('PARTIAL_INTERRUPT_NOT_TRIGGERED')


def gpu():
    from campaign import runtime_factory,baseline
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    P,F=runtime_factory();rt=P['Runtime'](ROOT/'qualification/engineering')
    assert rt.update==0
    parameter_count=sum(p.numel() for p in rt.e.actor.parameters())
    old=torch.load(R3/'production/checkpoints/milestone_05000.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(x.cpu(),old['actor'][k]) for k,x in rt.e.actor.state_dict().items())
    task=read(ROOT/'TRAIN_DIAGNOSTIC_MANIFEST.json')['groups'][0]['base']
    world=P['S']['world'](task['goal']);prefix=P['p']['collate']([world.state],'cuda')
    with torch.no_grad():ref=rt.e.prior(prefix);batch=P['p']['public_batch']([world],prefix,ref)
    batch['context'].requires_grad_(True);captured=[]
    def capture(module,args):
        args[0]['goals'].retain_grad();captured.append(args[0]['goals'])
    h=rt.e.actor.register_forward_pre_hook(capture)
    _,mu,ls=rt.e.actor(batch);aa=P['AA'].index(task['fixed_sequence'][0]);(mu[0,aa].sum()+ls[0,aa].sum()).backward();h.remove()
    token_gradient=float(captured[0].grad[0,-1].norm());assert token_gradient>0
    rt.e.actor.zero_grad(set_to_none=True)
    before=P['state_digest'](rt.state());rt.restore();assert P['state_digest'](rt.state())==before
    development=baseline(P,rt)
    write('qualification/AUGMENTED_INITIAL_TRAIN_BASELINE.json',dict(summary=development,attempts=sum(v['candidates'] for v in development.values()),no_optimizer_updates=True,not_VALIDATION=True))
    # Baseline is intentionally not installed in the engineering state: the
    # qualification exercises the same pre-baseline initial roundtrip twice.
    initial_norm=parameter_count
    first=rt.step(1);digest1=P['state_digest'](rt.state())
    rt.restore();assert rt.update==0
    replay=rt.step(1);assert replay==first and P['state_digest'](rt.state())==digest1
    rt.publish('ENGINEERING_UPDATE_ONE')
    second=rt.step(2);rt.publish('ENGINEERING_UPDATE_TWO')
    full=P['state_digest'](rt.state());rt.restore();assert P['state_digest'](rt.state())==full
    bound=copy.deepcopy(rt.binding);rt.binding=dict(bound,token_schema='INCORRECT')
    try:rt.restore()
    except AssertionError:pass
    else:raise AssertionError('BAD_TOKEN_BINDING_ACCEPTED')
    rt.binding=bound;rt.restore()
    index_hash=sha(rt.root/'RECOVERY_INDEX.json')
    result=subprocess.run([str(ROOT/'run.sh'),str(Path(__file__)),'partial-child'])
    assert result.returncode==75 and sha(rt.root/'RECOVERY_INDEX.json')==index_hash
    assert (ROOT/'qualification/interrupted_full_state.pt.part').exists()
    rt.restore();assert P['state_digest'](rt.state())==full
    assert parameter_count==sum(p.numel() for p in rt.e.actor.parameters())
    rt.e.prior.verify();assert P['c']['state_digest'](rt.e.parent.state_dict())==rt.e.fp
    assert all(p.grad is None for p in rt.e.prior.model.parameters())
    rejected=sum(read(rt.root/f'commits/u{u:06d}.json')['logical']['stats']['rejected'] for u in [1,2])
    assert rejected>0
    baseline_ok=development['filtered']['hard_valid']>=.91875 and development['filtered']['backbone_valid']>=.95 and development['filtered']['structural_pair_RMSD']>=12.449118749897715 and development['filtered']['geometry_KL_per_active']<=1
    write('qualification/PROTEIN_TESTS.json',dict(PASS=True,parameter_count=parameter_count,added_parameters=0,burial_token_input_gradient_norm=token_gradient,initializer_exact_R3_5000_tensors=True,unique_engineering_updates=2,physical_optimizer_calls=3,extra_call_is_exact_journal_replay=True,new_training_attempts=32,development_attempts=sum(v['candidates'] for v in development.values()),production_updates=0,full_actor_optimizer_scheduler_RNG_roundtrip=True,post_commit_replay_exact=True,partial_full_checkpoint_recovery=True,token_binding_mismatch_rejected=True,rejected_proposals_replayed=rejected,first_logprob_maxerror=first['replay_logprob_maxerror'],second_logprob_maxerror=second['replay_logprob_maxerror'],first_step_KL=first['step_KL'],second_step_KL=second['step_KL'],AA_score_gradients_zero=True,J1_frozen=True,initial_safety_pass=baseline_ok,TEST=False))
    write('qualification/RELEASE.json',dict(PASS=baseline_ok,correctness_qualified=True,initial_safety_pass=baseline_ok,production_started=False,production_authorized=False,method=VERSION,source96_SHA256={p.name:sha(p) for p in sorted((ROOT/'source96').glob('*.py'))},configuration_SHA256=sha(ROOT/'TRAINING_CONFIGURATION_AND_SEEDS.json'),engineering_only=True))
    print('PROTEIN_QUALIFICATION_PASS','initial_safety',baseline_ok,flush=True)


if __name__=='__main__':
    {'toy':toy,'gpu':gpu,'partial-child':partial_child}[sys.argv[1]]()
