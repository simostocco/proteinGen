"""Bounded TRAIN-only GPU admission, scaling and exact recovery, no validation predictions."""
import pathlib,sys,json,time,math,copy,hashlib
import numpy as np
import torch
R=pathlib.Path('/mnt/d/Simone/proteinGen_data/e017_v5_goal_observability_20261022_v1');sys.path.insert(0,str(R/'source/scripts/e017_v5'));import study as s
from protein_joint_sac.crisp_v3.recovery import state_digest,atomic_checkpoint
from protein_joint_sac.crisp_v3.storage import rng_state,set_rng

def main():
 s.seed(s.CONFIG['seed']);e=s.Engine();e.arm='C';e.actor.eval();pool=s.tasks('TRAIN');native_bonds=[a[0] for t in pool if t['variant']=='native' for a in t['native_actions']];native_scale=float(np.median(native_bonds));s.CONFIG['position_scale_A']=round(native_scale,1);start=time.monotonic();scales=[];checks={};torch.cuda.reset_peak_memory_stats()
 for H in [2,4,8]:
  ts=[t for t in pool if t['H']==H and t['variant']=='native'][:2]
  for task in ts:
   with torch.autocast('cuda',dtype=torch.bfloat16):loss,terms,x,z=e.rollout(task,8811,retain=True);guide,parts=e.guide(x,task)
   def norm(term):
    gs=torch.autograd.grad(term,[p for p in e.actor.parameters() if p.requires_grad],retain_graph=True,allow_unused=True);return math.sqrt(sum(float(g.square().sum()) for g in gs if g is not None))
   pos=norm(parts['position']);ori=norm(parts['orientation']);scales.append(dict(H=H,position_gradient=pos,orientation_gradient=ori,ratio=pos/max(ori,1e-12),goal_norm=norm(terms['goal']),BC_norm=norm(terms['BC']) if terms['BC'].requires_grad else 0,KL_norm=norm(terms['KL'])))
 # All six TRAIN probes, no prospective VALIDATION predictions.
 coefficient=float(np.clip(np.median([x['ratio'] for x in scales]),.1,5.));s.CONFIG['orientation_weight']=round(coefficient,3)
 task=[t for t in pool if t['H']==8 and t['variant']=='native'][0];x0=s.xprefix(task);c,_,_=s.relative(task,x0);q,_=torch.linalg.qr(torch.tensor([[.2,.7,-.3],[.6,.2,.9],[.8,-.1,.4]],dtype=torch.float64,device='cuda'));q[:,0]*=torch.linalg.det(q);shift=x0.new_tensor([12.,-8.,4.]);changed=copy.deepcopy(task);changed['target']=(x0.new_tensor(task['target'])@q.T+shift).tolist();changed['orientation']=(x0.new_tensor(task['orientation'])@q.T).tolist();cc,*_=s.relative(changed,x0@q.T+shift);assert torch.allclose(c,cc,atol=2e-5,rtol=1e-5);checks['rigid_transform_context_maxdiff']=float((c-cc).abs().max())
 # Matrix relative orientation stays finite and continuous near pi, avoiding ambiguous axis-log charts.
 for angle in [math.pi-1e-8,math.pi,math.pi+1e-8]:
  fake=copy.deepcopy(task);rr=torch.tensor([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]],device='cuda',dtype=torch.float64);fake['orientation']=(rr@s.frame(x0)).tolist();assert torch.isfinite(s.relative(fake,x0)[0]).all()
 checks['near_pi_rotation_matrix_finite']=True
 # Same model and identical noise: ONLY feedback gradient edges change, never forward semantics.
 with torch.autocast('cuda',dtype=torch.bfloat16):
  full,ft,fx,fz=e.rollout(task,8812,retain=True,route='full');det,dt,dx,dz=e.rollout(task,8812,retain=True,route='feedback_detached')
 assert torch.equal(fx,dx) and torch.equal(full,det);fg=torch.autograd.grad(ft['goal'],fz['mus'],retain_graph=True);dg=torch.autograd.grad(dt['goal'],dz['mus'],retain_graph=True);checks['forward_route_bitwise_equal']=True;checks['full_mu_gradient_norms']=[float(g.norm()) for g in fg];checks['full_mu_gradient_norm_by_d_angle_torsion']=torch.stack(fg).norm(dim=0).tolist();lg=torch.autograd.grad(ft['goal'],fz['lss'],retain_graph=True,allow_unused=True);checks['full_logstd_gradient_norm_by_d_angle_torsion']=torch.stack(lg).norm(dim=0).tolist();wg=torch.autograd.grad(ft['goal'],e.actor.geometry[-1].weight,retain_graph=True)[0];checks['geometry_output_row_gradient_norms_d_angle_torsion_ls']=wg.norm(dim=1).tolist();checks['detached_mu_gradient_norms']=[float(g.norm()) for g in dg];checks['route_gradient_maxdiff']=max(float((a-b).abs().max()) for a,b in zip(fg,dg));assert min(checks['full_mu_gradient_norms'])>0 and min(checks['detached_mu_gradient_norms'])>0 and checks['route_gradient_maxdiff']>0
 # Compare identical prefix predictions after perturbing forbidden future-only arrays/suffix identities.
 bad=copy.deepcopy(task);bad['hidden_native_future']=(np.ones_like(bad['hidden_native_future'])*999).tolist();bad['native_actions']=(np.ones_like(bad['native_actions'])*999).tolist();bad['sequence']=bad['sequence'][:task['t']+1]+'A'*(len(bad['sequence'])-task['t']-1)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):a=e.predict(task,x0);b=e.predict(bad,x0)
 assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]);checks['no_hidden_intermediate_or_future_suffix_leak']=True
 # Terminal FK latent-action finite differences independent of neural BF16 finite-difference jitter.
 u=torch.stack(fz['u']).detach().double().requires_grad_(True)
 def terminal(z):return e.guide(s.fk(task,s.physical(z)),task)[0]
 gg=torch.autograd.grad(terminal(u),u)[0];fd=[]
 for k in [0,3,7]:
  j=int(gg[k].abs().argmax());d=torch.zeros_like(u);d[k,j]=1e-5;estimate=float((terminal(u+d)-terminal(u-d))/(2e-5));analytic=float(gg[k,j]);er=abs(estimate-analytic)/max(abs(analytic),1e-5);assert er<1e-4;fd.append(dict(step=k,analytic=analytic,finite_difference=estimate,relative_error=er))
 checks['terminal_FK_FD']=fd
 # FP32 neural independent FD through generated-state feedback, including the new goal-relative features.
 test4=next(t for t in pool if t['H']==4 and t['variant']=='native');delta=torch.zeros(4,3,dtype=torch.float64,device='cuda',requires_grad=True);_,tt,_,_=e.rollout(test4,8840,delta=delta);dgfull=torch.autograd.grad(tt['goal'],delta)[0];fullfd=[]
 for k in [0,1,3]:
  j=int(dgfull[k].abs().argmax());h=.001;diff=torch.zeros_like(delta);diff[k,j]=h
  with torch.no_grad():_,a,_,_=e.rollout(test4,8840,delta=diff);_,b,_,_=e.rollout(test4,8840,delta=-diff)
  estimate=float((a['goal']-b['goal'])/(2*h));analytic=float(dgfull[k,j]);rel=abs(estimate-analytic)/max(abs(estimate),abs(analytic),.001);assert rel<.02,(k,estimate,analytic,rel);fullfd.append(dict(step=k,finite_difference=estimate,analytic=analytic,relative_error=rel))
 checks['full_feedback_FD_FP32']=fullfd
 # Nonzero input derivative through J1, no J1 parameter gradients.
 xx=x0.clone().requires_grad_(True);h=e.prior_forward(e.p(task,xx))['h'];ig=torch.autograd.grad(h.square().sum(),xx)[0];assert ig.abs().max()>0;checks['J1_input_gradient_norm']=float(ig.norm());e.verify()
 del full,det,ft,dt,fx,dx,fz,dz,fg,dg,h,xx,u,gg
 # Clipping comparison on the SAME TRAIN task, then restore exact initial actor.
 snapshots={};initial=copy.deepcopy(e.actor.state_dict());rn=rng_state();raw=[]
 short=[t for t in pool if t['H']==2 and t['variant']=='native'][0]
 for clip in [1.,5.]:
  e.actor.load_state_dict(initial);set_rng(rn);opt=s.optimizer(e);opt.zero_grad(set_to_none=True);e.actor.train()
  with torch.autocast('cuda',dtype=torch.bfloat16):loss,terms,_,_=e.rollout(short,8813)
  loss.backward();norm=float(torch.nn.utils.clip_grad_norm_([p for p in e.actor.parameters() if p.requires_grad],clip,error_if_nonfinite=True));opt.step();e.verify([opt]);snapshots[str(clip)]=dict(loss=float(loss),preclip_norm=norm,actor=state_digest(e.actor.state_dict()),finite=True);raw.append(norm)
 s.CONFIG['clip']=5. if min(raw)>5 else 1.;e.actor.load_state_dict(initial);set_rng(rn)
 teacher_probes=[];teacherfile=R/'manifests/IK_TEACHERS.json'
 if teacherfile.exists():
  db=json.load(open(teacherfile));tm={t['task_id']:t for t in pool}
  for rec in sum([[r for r in db['records'] if r['accepted'] and tm[r['task_id']]['H']==H][:2] for H in [2,4,8]],[]):
   e.actor.eval();e.actor.zero_grad(set_to_none=True);task_ik=tm[rec['task_id']]
   with torch.autocast('cuda',dtype=torch.bfloat16):ik=e.teacher_loss(task_ik,rec['best']);gl,tt,_,_=e.rollout(task_ik,8830)
   pp=[p for p in e.actor.parameters() if p.requires_grad];ig=torch.autograd.grad(ik,pp,allow_unused=True);gg=torch.autograd.grad(tt['goal'],pp,allow_unused=True);inorm=math.sqrt(sum(float(g.float().square().sum()) for g in ig if g is not None));gnorm=math.sqrt(sum(float(g.float().square().sum()) for g in gg if g is not None));teacher_probes.append(dict(H=task_ik['H'],IK_norm=inorm,goal_norm=gnorm,ratio=gnorm/max(inorm,1e-12)))
  if teacher_probes:s.CONFIG['weights']['IK']=float(min(.1,.5*np.median([p['ratio'] for p in teacher_probes])))
 s.save('TRAIN_LOSS_CALIBRATION.json',dict(TRAIN_native_bond_median_A=native_scale,chosen_position_scale_A=s.CONFIG['position_scale_A'],probes=scales,teacher_probes=teacher_probes,chosen_IK_weight=s.CONFIG['weights']['IK'],chosen_orientation_coefficient=s.CONFIG['orientation_weight'],clip_shadow_updates=snapshots,chosen_clip=s.CONFIG['clip'],no_validation_predictions=True,clip_change_interpretation='same clipping direction tested at1 and5 on same TRAIN state; weights differ only numerically at first AdamW step; main arm comparisons all share sealed choice'))
 # Exact interruption/restart next-step replay, including actor/AdamW/scheduler/RNG.
 opt=s.optimizer(e);sched=s.scheduler(opt);sample_pool=s.tasks('TRAIN',2);sample_order=s.identity_order(sample_pool,s.CONFIG['seed']+2);sample_cursor=[0];last_draw={}
 def one():
  e.actor.train();opt.zero_grad(set_to_none=True);chosen=sample_pool[sample_order[sample_cursor[0]]];noise=8820+sample_cursor[0];sample_cursor[0]+=1;last_draw.update(task_id=chosen['task_id'],noise_seed=noise)
  with torch.autocast('cuda',dtype=torch.bfloat16):loss,_,_,_=e.rollout(chosen,noise)
  loss.backward();torch.nn.utils.clip_grad_norm_(e.actor.parameters(),s.CONFIG['clip']);opt.step();sched.step();e.verify([opt]);return dict(actor=state_digest(e.actor.state_dict()),optimizer=state_digest(opt.state_dict()),scheduler=state_digest(sched.state_dict()),rng=state_digest(rng_state()),loss=float(loss))
 one();st=dict(actor=e.actor.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),rng=rng_state(),sampler_order=sample_order,sampler_cursor=sample_cursor[0],configuration=s.CONFIG);record=atomic_checkpoint(R/'checkpoints/engineering_recovery.pt',st,s.v.POLICY);expected=one();st=torch.load(record['path'],map_location='cpu',weights_only=False);assert s.sha(record['path'])==record['sha256'] and state_digest(st)==record['state_digest'];e.actor.load_state_dict(st['actor']);opt.load_state_dict(st['optimizer']);sched.load_state_dict(st['scheduler']);set_rng(st['rng']);sample_order=st['sampler_order'];sample_cursor[0]=st['sampler_cursor'];actual=one();assert actual==expected
 s.save('results/RESTART_EXPECTED.json',dict(expected=expected,config=s.CONFIG,task_id=last_draw['task_id'],noise_seed=last_draw['noise_seed']));checks['same_process_atomic_resume_exact']=True;checks['resume_expected']=expected;checks['J1_frozen']=True;checks['peak_allocated']=torch.cuda.max_memory_allocated();checks['peak_reserved']=torch.cuda.max_memory_reserved();checks['seconds']=time.monotonic()-start;checks['engineering_optimizer_executions']=5;checks['actor_parameters']=sum(p.numel() for p in e.actor.parameters());checks['trainable_parameters']=sum(p.numel() for p in e.actor.parameters() if p.requires_grad);s.save('results/ENGINEERING_CHECKS.json',checks)
 s.save('PROPOSED_FINAL_CONFIG.json',s.CONFIG);print('CHECKS_PASS',checks,flush=True)

if __name__=='__main__':main()
