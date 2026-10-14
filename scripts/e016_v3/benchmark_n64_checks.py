"""Bounded N64 validation; never a protein-training controller."""
import copy,json,torch
from protein_joint_sac.geometry.local_frames import features
from protein_joint_sac.geometry.ca_builder import replay
from protein_joint_sac.contracts import BuildAction
from protein_joint_sac.rl.checkpointing import save_atomic,load_checked

def record(root,name,values):
    (root/'benchmarks'/f'{name}.json').write_text(json.dumps(values,indent=2))

def prechecks(state,actor,q,root):
    assert state.N == 64
    cpu=state.to('cpu'); gpu=features(state); ref=features(cpu)
    for k,v in gpu.items():
        if isinstance(v,torch.Tensor):
            assert v.device == state.ca.device
            torch.testing.assert_close(v.cpu(),ref[k],atol=1e-8,rtol=1e-8)
    ca=copy.deepcopy(actor).cpu()
    with torch.no_grad():
        a=actor.logits(state).cpu();b=ca.logits(cpu)
        torch.testing.assert_close(a,b,atol=5e-5,rtol=5e-5)
    assert not any(p.requires_grad for p in q.target1.parameters())
    assert next(q.q1.parameters()).data_ptr()!=next(q.q2.parameters()).data_ptr()
    save_atomic(root/'scratch/cuda_actor.pt',{'actor':actor.state_dict()})
    ca.load_state_dict(load_checked(root/'scratch/cuda_actor.pt')['actor'],strict=True)
    torch.testing.assert_close(ca.logits(cpu),b,atol=0,rtol=0)
    record(root,'device_checks',{'features_checked':len(gpu),'max_cpu_cuda_logit_error':float((a-b).abs().max()),'checkpoint_strict_load':True,'parameter_counts':{'actor':sum(p.numel() for p in actor.parameters()),'critics_including_targets':sum(p.numel() for p in q.parameters())},'torch':torch.__version__,'CUDA_build':torch.version.cuda,'GPU':torch.cuda.get_device_name()})

def gradient_check(state,actor,q,alg,root):
    flags=[p.requires_grad for p in q.parameters()]
    for p in q.parameters():p.requires_grad_(False)
    actor.zero_grad(set_to_none=True);torch.manual_seed(1664)
    loss=alg.actor_loss(state);loss.backward()
    grads={n:p.grad.detach().clone() for n,p in actor.named_parameters() if p.grad is not None}
    value=float(loss.detach());del loss
    actor.zero_grad(set_to_none=True);torch.manual_seed(1664)
    streamed=alg.backward_actor_exact_streamed(state)
    assert abs(streamed-value)<5e-5
    err=0.
    for n,p in actor.named_parameters():
        if n in grads:
            torch.testing.assert_close(p.grad,grads[n],atol=1e-5,rtol=1e-4)
            err=max(err,float((p.grad-grads[n]).abs().max()))
    for p,f in zip(q.parameters(),flags):p.requires_grad_(f)
    # Smooth candidate geometry gradient and central finite difference, FP64 state / FP32 network.
    g=state.ca.new_tensor([3.8,1.9,.4],requires_grad=True)
    out=q.q1(state,BuildAction('A',g,(True,True,True)))
    grad=torch.autograd.grad(out,g)[0];eps=.002
    numerical=[]
    with torch.no_grad():
        for i in range(3):
            delta=torch.zeros_like(g);delta[i]=eps
            numerical.append((q.q1(state,BuildAction('A',g+delta,(True,True,True)))-q.q1(state,BuildAction('A',g-delta,(True,True,True))))/(2*eps))
    fd=torch.stack(numerical).double()
    torch.testing.assert_close(grad,fd,atol=2e-4,rtol=.03)
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
    record(root,'gradient_checks',{'eager_loss':value,'streamed_loss':streamed,'max_actor_gradient_error':err,'geometry_autograd':grad.detach().cpu().tolist(),'finite_difference':fd.cpu().tolist(),'fd_epsilon':eps})

def postchecks(state,actor,q,alg,actions,root):
    cpu=replay(64,actions);cuda=replay(64,actions,device='cuda')
    torch.testing.assert_close(cpu.ca,cuda.ca.cpu(),atol=1e-8,rtol=1e-8)
    old=next(q.target1.parameters()).detach().clone()
    alg.actor_optimizer.step();alg.critic_optimizer.step();q.polyak(.005)
    expected=old.lerp(next(q.q1.parameters()).detach(),.005)
    torch.testing.assert_close(next(q.target1.parameters()),expected,atol=0,rtol=0)
    for p in list(actor.parameters())+list(q.parameters()):
        assert torch.isfinite(p).all()
        if p.grad is not None:assert torch.isfinite(p.grad).all()
    save_atomic(root/'scratch/synthetic_updated_checkpoint.pt',{'actor':actor.state_dict(),'critics':q.state_dict(),'actor_optimizer':alg.actor_optimizer.state_dict(),'critic_optimizer':alg.critic_optimizer.state_dict()})
    loaded=load_checked(root/'scratch/synthetic_updated_checkpoint.pt')
    assert set(loaded)=={'actor','critics','actor_optimizer','critic_optimizer'}
    record(root,'replay_and_update_checks',{'CPU_CUDA_replay_max_error_A':float((cpu.ca-cuda.ca.cpu()).abs().max()),'actor_optimizer_steps':1,'critic_optimizer_steps':1,'target_polyak':.005,'finite_parameters':True,'CPU_checkpoint_load':True,'scope':'SYNTHETIC_BENCHMARK_ONLY'})
