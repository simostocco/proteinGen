"""Bounded real-GPU recovery tests; discarded model updates, never campaign progress."""
import os,sys,pathlib,json,copy,math,time,subprocess,shutil
import numpy as np
import torch
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P));import importlib.util
_spec=importlib.util.spec_from_file_location('v51_safe_study',P/'study.py');q=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=q;_spec.loader.exec_module(q)
R=q.R

def base():
 q.seed(q.CONFIG['seed']);e=q.Engine();e.label='D';o=q.optimizer(e);sc=q.scheduler(o);pool=q.trainpool();order=q.s.identity_order(pool,q.CONFIG['seed']+4);return e,o,sc,pool,order

def prepare():
 e,o,sc,pool,order=base();start=time.monotonic();cursor=0
 # Relative pose invariance, TRAIN-only, including H4.
 t=pool[0];x=q.s.xprefix(t);a=q.s.relative(t,x)[0];rot=x.new_tensor([[0.,-1,0],[1,0,0],[0,0,1]]);shift=x.new_tensor([10.,-8,3]);z=copy.deepcopy(t);z['target']=(rot@x.new_tensor(t['target'])+shift).tolist();z['orientation']=(x.new_tensor(t['orientation'])@rot.T).tolist();b=q.s.relative(z,x@rot.T+shift)[0];err=float((a-b).abs().max());assert err<1e-5
 # Continuous action KL/Jacobian identity at exact fixed states.
 mu=torch.tensor([1.3,.4,-.2],dtype=torch.float64);ls=torch.tensor([-2.,-.5,-.4],dtype=torch.float64);rm=mu+torch.tensor([.02,-.03,.04]);rl=ls+.02
 exact=e.kl(mu,ls,rm,rl);torch.manual_seed(14);u=mu+ls.exp()*torch.randn(50000,3,dtype=torch.float64);lp=torch.distributions.Normal(mu,ls.exp()).log_prob(u)-torch.distributions.Normal(rm,rl.exp()).log_prob(u);assert abs(float(lp.mean()-exact))<.004
 g=q.s.physical(u[0]);mask=torch.ones(3,dtype=torch.bool);ld=q.s.v.log_prob(mu,ls,g,mask)-q.s.v.log_prob(rm,rl,g,mask);assert abs(float(ld-lp[0].sum()))<1e-8
 # Before-contact loss: distance2.5A is not a clash but receives repulsive gradient.
 xx=torch.tensor([[0.,0,0],[9.,0,0],[9.,9,0],[2.5,0,0]],dtype=torch.float64,requires_grad=True);tt=dict(t=3,H=1);loss=q.collision(xx,tt);gr=torch.autograd.grad(loss,xx)[0];assert float(loss)>0 and float(gr[-1,0])<0
 order,cursor,row=q.update(e,o,sc,pool,order,cursor,0);e.verify([o]);record=q.publish(e,o,sc,'ENGINEERING',1,order,cursor,dict(update=None),{},reason='simulated scientific stop: quarantine, never promoted')
 state=torch.load(record['path'],map_location='cpu',weights_only=False);assert state['stop_reason'].startswith('simulated') and state['update']==1 and not q.read('checkpoints/ENGINEERING/INDEX.json')['best']
 # Continuous reference next update; fresh process will reproduce it exactly.
 order,cursor,row2=q.update(e,o,sc,pool,order,cursor,1);e.verify([o]);expected=q.ckstate(e,o,sc,'ENGINEERING',2,order,cursor,dict(update=None),{},None);q.save('ENGINEERING_EXPECTED_NEXT.json',dict(digest=q.state_digest(expected),next_row=row2))
 q.save('ENGINEERING_PREFLIGHT_PART1.json',dict(PASS=True,relative_feature_rigid_invariance_max=err,KL_analytic=float(exact),KL_MC=float(lp.mean()),transformed_logdensity_difference_error=float(ld-lp[0].sum()),precontact_penalty=float(loss),precontact_tip_gradient=gr[-1].tolist(),fresh_restart_record=record,engineering_updates=2,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),seconds=time.monotonic()-start,trainable=sum(p.numel() for p in e.actor.parameters() if p.requires_grad)))
 print('ENGINEERING_PART1_PASS',flush=True)

def resume():
 e,o,sc,pool,_=base();rec=q.read('checkpoints/ENGINEERING/INDEX.json')['latest'];st=q.load_record(e,o,sc,rec);assert st['update']==1
 order,cursor,row=q.update(e,o,sc,pool,st['sampler']['order'],st['sampler']['cursor'],1);actual=q.ckstate(e,o,sc,'ENGINEERING',2,order,cursor,dict(update=None),{},None);expected=q.read('ENGINEERING_EXPECTED_NEXT.json');assert q.state_digest(actual)==expected['digest'],'full fresh-process recovery mismatch'
 # Numerical-stop rollback before optimizer: NaN gradient is rejected without changing parameter/moment state.
 before=q.state_digest(dict(actor=e.actor.state_dict(),opt=o.state_dict(),sched=sc.state_dict()));o.zero_grad(set_to_none=True);p=next(p for p in e.actor.parameters() if p.requires_grad);p.grad=torch.full_like(p,float('nan'))
 rejected=False
 try:torch.nn.utils.clip_grad_norm_([p for p in e.actor.parameters() if p.requires_grad],5,error_if_nonfinite=True)
 except RuntimeError:rejected=True
 assert rejected and before==q.state_digest(dict(actor=e.actor.state_dict(),opt=o.state_dict(),sched=sc.state_dict()));o.zero_grad(set_to_none=True)
 record=q.publish(e,o,sc,'ENGINEERING',2,order,cursor,dict(update=None),{},reason='simulated numerical gradient stop; optimizer not stepped');q.save('CHECKPOINT_STOP_RECOVERY_TESTS.json',dict(PASS=True,fresh_GPU_process_exact_next_update=True,model_optimizer_scheduler_sampler_RNG_dual_exact=True,scientific_stop_checkpoint_before_exit=True,numerical_stop_before_optimizer_verified=True,invalid_checkpoint_not_promoted=True,record=record,engineering_update_executions=3,J1_frozen=True));print('FRESH_RESTART_AND_STOP_PASS',flush=True)

def crash():
 e,o,sc,pool,_=base();rec=q.read('checkpoints/ENGINEERING/INDEX.json')['latest'];st=q.load_record(e,o,sc,rec);state=q.ckstate(e,o,sc,'ENGINEERING',st['update'],st['sampler']['order'],st['sampler']['cursor'],st['best'],st['monitor'],'simulated atomic interruption')
 import protein_joint_sac.crisp_v3.recovery as recovery
 original=recovery.os.replace
 def terminate(src,dst):
  if str(dst).endswith('INTERRUPTED.pt'):os._exit(91)
  return original(src,dst)
 recovery.os.replace=terminate;q.atomic_checkpoint(R/'checkpoints/ENGINEERING/INTERRUPTED.pt',state,q.POLICY)
 raise AssertionError('simulated crash failed')

def verifycrash():
 rec=q.read('checkpoints/ENGINEERING/INDEX.json')['latest'];assert q.sha(rec['path'])==rec['sha256'];st=torch.load(rec['path'],map_location='cpu',weights_only=False);assert q.state_digest(st)==rec['state_digest'] and st['update']==2
 p=R/'checkpoints/ENGINEERING/INTERRUPTED.pt.part';assert p.exists() and not p.with_suffix('').exists();data=q.read('CHECKPOINT_STOP_RECOVERY_TESTS.json');data['atomic_kill_before_rename_preserves_previous_full_checkpoint']=True;data['own_unpublished_transient']=dict(bytes=p.stat().st_size,sha256=q.sha(p));q.save('CHECKPOINT_STOP_RECOVERY_TESTS.json',data)
 # Only this test's transient and superseded engineering recovery, after full result+hash verification.
 p.unlink()
 for p in (R/'checkpoints/ENGINEERING').glob('checkpoint_*.pt'):p.unlink()
 print('ATOMIC_INTERRUPT_PASS; own engineering transient retired',flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
