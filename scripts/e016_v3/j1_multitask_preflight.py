"""TRAIN-only correctness, task-gradient calibration, GPU profiling and discardable restart proof."""
import copy,gc,math,random,time,resource,subprocess
import numpy as np
import torch
from torch.nn import functional as F
import j1_bidirectional_multitask as q
from protein_joint_sac.models.bidirectional_multitask import span_batch,span_loss,masked_states,SpanSampler
R=q.R;m=q.m;s=q.s;c=q.c;write=q.write;read=q.read;sha=q.sha

def cpu_grad(model):return {n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
def gnorm(g,prefix='encoder.'):return math.sqrt(sum(float(v.double().square().sum()) for n,v in g.items() if n.startswith(prefix)))
def cosine(a,b,prefix='encoder.'):
 dot=sum(float((a[n].double()*b[n].double()).sum()) for n in set(a)&set(b) if n.startswith(prefix));return dot/max(gnorm(a,prefix)*gnorm(b,prefix),1e-30)

def structural_panel(e):
 ex=[]
 for kind in ['complete','partial_only','unused']:
  for band in range(5):
   h=next(h for h in e.data.populations['TRAIN'][kind] if s.j.band(len(e.data.seqs[h]))==band);o=None if kind=='unused' else (e.data.owners['TRAIN'][h][0] if kind=='partial_only' else e.data.base.owners[h][0]);N=len(e.data.seqs[h])
   for f in [.25,.75]:ex.append((h,o,min(N-1,max(3,int(N*f)))))
 return sorted(ex,key=lambda v:v[2])

def calibrate(e):
 rng=s.rng_state();sam=copy.deepcopy(e.sampler.state_dict());ms=copy.deepcopy(e.mask_sampler.state_dict());e.model.eval();e.aux.eval();e.opt.zero_grad(set_to_none=True);panel=structural_panel(e)
 for off in range(0,len(panel),8):
  b,y,g=e.data.batch(panel[off:off+8])
  with torch.autocast('cuda',dtype=torch.bfloat16):loss=F.cross_entropy(e.model.aa_head(m.encode_batch(e.model,b)).float(),y,reduction='sum')/len(panel)
  loss.backward()
 aa=cpu_grad(e.model);e.opt.zero_grad(set_to_none=True);plans=[]
 for band in range(5):
  h=next(h for h in e.data.base.ids['TRAIN'] if s.j.band(len(e.data.seqs[h]))==band);plans.append(e.mask_sampler.plan(h,record=False))
 for p in plans:
  with torch.autocast('cuda',dtype=torch.bfloat16):loss,_,_=span_loss(e.model,e.aux,span_batch(e.data.seqs,[p]));loss.mean().div(len(plans)).backward()
 masked=cpu_grad(e.model);e.opt.zero_grad(set_to_none=True)
 pairs=[p for p in panel if p[1] in e.data.base.ca][:8];b,y,g=e.data.batch(pairs)
 with torch.autocast('cuda',dtype=torch.bfloat16):pa=e.model.aa_head(m.encode_batch(e.model,b)).float().detach().softmax(-1)
 bm={k:v.clone() for k,v in b.items()};bm['valid'].zero_();bm['bond_valid'].zero_()
 with torch.autocast('cuda',dtype=torch.bfloat16):lb=e.model.aa_head(m.encode_batch(e.model,bm)).float();kl=F.kl_div(lb.log_softmax(-1),pa,reduction='batchmean')
 kl.backward();consistency=cpu_grad(e.model);e.opt.zero_grad(set_to_none=True)
 geo=e.sampler.draw()[1]
 for off in range(0,32,8):
  b,y,g=e.data.batch(geo[off:off+8]);a,l,p=s.loss_call(e.model,b,y,g);(l.sum()/32).backward()
 geometry=cpu_grad(e.model);e.opt.zero_grad(set_to_none=True);an=gnorm(aa);mn=gnorm(masked);kn=gnorm(consistency)
 e.c['lambda_masked']=min(.20,.30*an/max(mn,1e-30));e.c['lambda_consistency']=min(.025,.05*an/max(kn,1e-30))
 gradients={'primary_AA':aa,'masked_raw':masked,'consistency_raw':consistency,'local_geometry_raw':geometry};weights={'primary_AA':1.,'masked_raw':e.c['lambda_masked'],'consistency_raw':e.c['lambda_consistency'],'local_geometry_raw':e.c['lambda_local']}
 result={'TRAIN_only':True,'panel':panel,'norms':{k:{'shared_L2':gnorm(v),'weighted_shared_L2':weights[k]*gnorm(v),'AA_head_L2':gnorm(v,'aa_head.'),'geometry_L2':gnorm(v,'geom.'),'weight':weights[k],'per_layer_shared_L2':{str(i):gnorm(v,f'encoder.blocks.{i}.') for i in range(8)}} for k,v in gradients.items()},'pairwise_shared_cosines':{a+'__'+b:cosine(gradients[a],gradients[b]) for a in gradients for b in gradients if a<b},'frozen_weights':{k:e.c[k] for k in ['lambda_masked','lambda_consistency','lambda_local','lambda_nonlocal']},'shared_caps':{'sequence_aux_vs_primary_AA':.35,'geometry_vs_primary_AA':.5},'interpretation':'Norm/cosine diagnostics are TRAIN-only engineering evidence; independently transformed/clipped updates are not the unmodified scalar-loss gradient.'}
 write(R/'MULTITASK_GRADIENT_BALANCE.json',result);e.sampler.load_state_dict(sam);e.mask_sampler.load_state_dict(ms);s.restore_rng(rng)

def correctness(e):
 rng=s.rng_state();sam=copy.deepcopy(e.sampler.state_dict());ms=copy.deepcopy(e.mask_sampler.state_dict());m.mask_tests(e);e.opt.zero_grad(set_to_none=True);e.model.eval();e.aux.eval();h=next(h for h in e.data.base.ids['TRAIN'] if len(e.data.seqs[h])==64);plan=(h,list(range(24,30)));batch=span_batch(e.data.seqs,[plan]);ids,N,targets,mask=batch
 captured=[];handle=e.model.encoder.token.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()));paircalls=[];ph=e.model.encoder.pair.register_forward_hook(lambda *args:paircalls.append(1))
 with torch.autocast('cuda',dtype=torch.bfloat16):loss,logits,_=span_loss(e.model,e.aux,batch)
 handle.remove();ph.remove();assert not paircalls;assert (captured[0][...,:34]==0).all() and (captured[0][...,37]==0).all()
 altered=targets.clone();altered[mask]=(altered[mask]+7)%20
 with torch.autocast('cuda',dtype=torch.bfloat16):loss2,logits2,_=span_loss(e.model,e.aux,(ids,N,altered,mask))
 assert torch.equal(logits,logits2);unmasked=targets.clone();unmasked[~mask]=(unmasked[~mask]+9)%20
 with torch.autocast('cuda',dtype=torch.bfloat16):loss3,logits3,_=span_loss(e.model,e.aux,(ids,N,unmasked,mask))
 assert torch.equal(loss,loss3);grad_logits=torch.autograd.grad(loss.sum(),logits,retain_graph=True)[0];assert not grad_logits[~mask].any()
 changed=ids.clone();j=next(j for j in range(int(N[0])-1,max(plan[1]),-1) if ids[0,j]>=0);changed[0,j]=(changed[0,j]+1)%20
 with torch.autocast('cuda',dtype=torch.bfloat16):_,other,_=span_loss(e.model,e.aux,(changed,N,targets,mask))
 future_difference=float((other[mask]-logits[mask]).abs().max());assert future_difference>0,'auxiliary bidirectional context unavailable'
 loss.mean().backward();mg=gnorm(cpu_grad(e.model));ag=sum(float(p.grad.double().square().sum()) for p in e.aux.parameters() if p.grad is not None)**.5;assert mg>0 and ag>0 and all(p.grad is None for n,p in e.model.named_parameters() if n.startswith(('aa_head.','geom.','selected_aa.')));e.opt.zero_grad(set_to_none=True)
 # Coupled dropout on exact same prefix dimensions, geometry available vs hidden.
 ex=[x for x in structural_panel(e) if x[1] in e.data.base.ca][:4];b,y,g=e.data.batch(ex);e.model.train();before=s.rng_state()
 with torch.autocast('cuda',dtype=torch.bfloat16):pa=e.model.aa_head(m.encode_batch(e.model,b)).float().detach().softmax(-1)
 after=s.rng_state();s.restore_rng(before);bm={k:v.clone() for k,v in b.items()};bm['valid'].zero_();bm['bond_valid'].zero_();assert torch.equal(bm['ids'],b['ids']) and torch.equal(bm['t'],b['t']) and torch.equal(bm['N'],b['N'])
 with torch.autocast('cuda',dtype=torch.bfloat16):pb=e.model.aa_head(m.encode_batch(e.model,bm)).float();kl=F.kl_div(pb.log_softmax(-1),pa,reduction='batchmean')
 dropout_equal=c.same(after,s.rng_state());assert dropout_equal;kl.backward();kg=gnorm(cpu_grad(e.model));assert kg>0 and not pa.requires_grad and all(p.grad is None for n,p in e.model.named_parameters() if n.startswith(('geom.','selected_aa.')));e.opt.zero_grad(set_to_none=True)
 e.model.eval();bm['x'].fill_(float('nan'))
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):z1=e.model.aa_head(m.encode_batch(e.model,bm))
 bm['x'].fill_(1e12)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):z2=e.model.aa_head(m.encode_batch(e.model,bm))
 assert torch.equal(z1,z2)
 a,l,p=s.loss_call(e.model,b,y,g);l.mean().backward();assert all(torch.isfinite(p.grad).all() for p in e.model.parameters() if p.grad is not None);assert e.model.geom[2].weight.grad.abs().sum()>0
 density_semantics='unchanged losses()/Jacobian implementation; complete reference density parity rechecked by frozen native evaluation'
 write(R/'MASKED_SPAN_LEAKAGE_AND_GRADIENT_TESTS.json',{'status':'PASS','actual_GPU':torch.cuda.get_device_name(),'production_parameters':12351746,'training_only_auxiliary_parameters':7412,'masked_target_identity_cannot_affect_logits':True,'unmasked_targets_loss_zero':True,'unmasked_readout_gradient_zero':True,'unmasked_future_context_changes_mask_logits':future_difference,'masked_geometry_channels_all_zero':True,'masked_geometric_pair_calls':0,'masked_shared_gradient_L2':mg,'masked_auxiliary_gradient_L2':ag,'masked_original_AA_and_geometry_heads_no_gradient':True,'consistency_shared_gradient_L2':kg,'consistency_teacher_detached':True,'paired_AA_prefix_and_positions_identical':True,'paired_dropout_RNG_after_views_identical':dropout_equal,'masked_coordinate_placeholders_invariant':True,'geometry_head_gradient_nonzero':True,'geometry_density':density_semantics,'original_prefix_and_partial_mask_tests':str(R/'AA_ONLY_INPUT_MASK_TESTS.json'),'optimizer_updates_in_these_checks':0})
 e.opt.zero_grad(set_to_none=True);e.sampler.load_state_dict(sam);e.mask_sampler.load_state_dict(ms);s.restore_rng(rng)

def benchmark(e):
 rng=s.rng_state();sam=copy.deepcopy(e.mask_sampler.state_dict());rows=[];e.model.train();e.aux.train();maxbatch=16;best_at500=1
 for length in [64,128,256,384,500]:
  hs=[h for h in e.data.base.ids['TRAIN'] if len(e.data.seqs[h])==length];assert hs
  for B in [1,2,4,8,16,32]:
   e.opt.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.monotonic();plans=[e.mask_sampler.plan(hs[k%len(hs)],record=False) for k in range(B)];batch=span_batch(e.data.seqs,plans);prep=time.monotonic()-start
   try:
    timings=[]
    for trial in range(2):
     e.opt.zero_grad(set_to_none=True);torch.cuda.synchronize();t=time.monotonic()
     with torch.autocast('cuda',dtype=torch.bfloat16):ls,logits,mask=span_loss(e.model,e.aux,batch);v=ls.mean()
     torch.cuda.synchronize();forward=time.monotonic()-t;t=time.monotonic();v.backward();torch.cuda.synchronize();backward=time.monotonic()-t;timings.append((forward,backward));del v,ls,logits,mask
    alloc=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved();dt=sum(timings[-1])+prep;nt=sum(len(p) for h,p in plans);row={'length':length,'live_batch':B,'data_prepare_seconds':prep,'forward_seconds':timings[-1][0],'backward_seconds':timings[-1][1],'masked_targets':nt,'masked_targets_per_second':nt/dt,'sequence_tokens_per_second':B*length/dt,'Torch_allocated_peak_bytes':alloc,'Torch_reserved_peak_bytes':reserved,'finite_shared_gradient':all(torch.isfinite(p.grad).all() for p in e.model.parameters() if p.grad is not None),'status':'PASS'};assert row['finite_shared_gradient'];rows.append(row)
   except torch.cuda.OutOfMemoryError as err:
    rows.append({'length':length,'live_batch':B,'status':'OOM','error':str(err)[:220]});e.opt.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();break
   finally:
    e.opt.zero_grad(set_to_none=True);del batch
  q.m.status('MASKED_PROFILE_LENGTH_DONE',length=length)
 candidates=[v for v in rows if v['length']==500 and v['status']=='PASS' and v['live_batch']<=16 and v['Torch_reserved_peak_bytes']<6*2**30];assert candidates;best=max(candidates,key=lambda v:v['masked_targets_per_second']);e.mask_micro=best['live_batch'];e.c['masked_microbatch']=e.mask_micro;write(R/'MASKED_BATCH_BENCHMARKS.json',{'rows':rows,'effective_masked_examples':16,'selected_live_microbatch':e.mask_micro,'selection':'highest measured full-length500 masked-target throughput among safe live batches<=effective16; shorter batches use same upper bound;32 profiled as capacity evidence','GPU':torch.cuda.get_device_name(),'no_optimizer_steps':True});e.mask_sampler.load_state_dict(sam);s.restore_rng(rng);torch.cuda.empty_cache()

def replay_proof(e):
 # All diagnostic optimizer work is discarded; sealed main lineage starts exactly at accepted parent.
 e.history=[];e.update=0;e.lineage['config']=copy.deepcopy(e.c);start=e.checkpoint(R/'preflight/checkpoint_00000.pt');steps=[e.step() for _ in range(2)];two=e.checkpoint(R/'preflight/checkpoint_00002.pt');reference=[e.step() for _ in range(2)];state={'model':s.cpu_copy(e.model.state_dict()),'auxiliary':s.cpu_copy(e.aux.state_dict()),'optimizer':s.cpu_copy(e.opt.state_dict()),'scheduler':copy.deepcopy(e.scheduler.state_dict()),'sampler':e.sampler.state_dict(),'masked_sampler':e.mask_sampler.state_dict(),'rng':s.rng_state()};e.resume(two);repeated=[e.step() for _ in range(2)];actual={'model':e.model.state_dict(),'auxiliary':e.aux.state_dict(),'optimizer':e.opt.state_dict(),'scheduler':e.scheduler.state_dict(),'sampler':e.sampler.state_dict(),'masked_sampler':e.mask_sampler.state_dict(),'rng':s.rng_state()};checks={k:c.same(v,actual[k]) for k,v in state.items()};checks['sample_order']=[v['sample_order_sha256'] for v in reference]==[v['sample_order_sha256'] for v in repeated];assert all(checks.values()),checks
 write(R/'MULTITASK_GPU_RESTART_TEST.json',{'status':'PASS','checks':checks,'diagnostic_unique_updates':4,'diagnostic_reexecuted_updates':2,'diagnostic_physical_optimizer_calls':6,'all_discarded_from_main_progress':True,'main_start_restored_checkpoint':str(start),'main_start_sha256':sha(start)})
 e.resume(start);parent=s.load_checked(m.SOURCE);assert c.same(e.model.state_dict(),parent['model']);del parent;assert e.update==0 and e.sampler.cursor==0 and e.mask_sampler.cursor==0;write(R/'TRAIN_ONLY_SMOKE_UPDATES.json',steps+reference);e.elapsed=0.;e.start=time.monotonic();mean=float(np.mean([v['seconds'] for v in steps+reference]));project=10000*mean+6*300+6*100+300
 write(R/'MAIN_RESOURCE_GATE.json',{'status':'PASS' if project<6*3600 else 'BLOCKED','measured_update_seconds_mean':mean,'expected_10k_optimizer_hours':10000*mean/3600,'expected_10k_with_evaluation_hours':project/3600,'allowed_projection_hours':6,'Torch_allocated_peak_bytes':torch.cuda.max_memory_allocated(),'Torch_reserved_peak_bytes':torch.cuda.max_memory_reserved(),'D_free_bytes':q.m.guard()['D_free_bytes'],'CPU_RSS_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'MLM_live_batch':e.mask_micro,'MLM_effective_sequences':16,'main_start_exact_parent':True,'no_new_main_optimizer_updates_yet':True})
 if project>=6*3600:raise RuntimeError('measured10k runtime exceeds6h gate; propose5000 under separate review')
 return steps+reference

def preflight(e):
 m.status('MULTITASK_CORRECTNESS_TESTS');correctness(e);m.status('MULTITASK_GRADIENT_CALIBRATION');calibrate(e);m.status('MULTITASK_MASKED_BATCH_PROFILING');benchmark(e);m.status('MULTITASK_DISCARDABLE_RESTART_PROOF');replay_proof(e);m.status('MULTITASK_PREFLIGHT_PASS',**read(R/'MAIN_RESOURCE_GATE.json'))
