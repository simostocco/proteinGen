"""Frozen fixed-sequence geometry, finite filtering, delayed interventions."""
import collections,copy,json,math,runpy,sys,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v93_regional_controllability_20261024_v1');d=runpy.run_path(str(R/'source/scripts/e017_v93/common.py'));globals().update({k:d[k] for k in ['R','OLD','V91','V9','p','c','g','AA','SPEC','read','save','sha','ns','guard','world','append','prefix_status','extend_checked','load_engine','verify_engine','reconstruct']});T=runpy.run_path(str(R/'source/scripts/e017_v93/trace.py'));corrected_action=p['legacy']['corrected_action'];START=time.perf_counter();COUNTS=collections.Counter()
# compact proposal columns: t,k,raw_u3,physical_g3,baseLP,refBaseLP,valid,repairs,reason-list
@torch.no_grad()
def generate(e,task,policy='C',count=8,state=None,seeds=None):
 guard();assert time.perf_counter()-START<2400;assert COUNTS['simulated_transitions']<130000;goal=task['goal'];seq=task['fixed_sequence'];ws=[world(goal,state) for i in range(count)];seeds=seeds or [ns('CANDIDATE',goal['goal_id'],i) for i in range(count)];records=[dict(sequence='',geometry=[],proposals=[],seed=seeds[i],candidate_index=i,goal_id=goal['goal_id'],initial_prefix_length=ws[i].state.t,termination=None,AA_decisions=0,geometry_KL=[],repairs=0) for i in range(count)];live=list(range(count));start=time.perf_counter();model='A_J1' if policy=='A' else 'C_FIXED'
 for t in range(ws[0].state.t,64):
  if not live:break
  logits,mu,ls,ref,_=e.distribution([ws[i] for i in live],model);nextlive=[];a=AA.index(seq[t]);mask=c['mask_for_step'](t)
  for j,i in enumerate(live):
   w=ws[i];rec=records[i];accepted=False
   for k in range(8 if policy=='C' else 1):
    rng=np.random.default_rng(ns('STEP',seeds[i],t,k));eps=torch.tensor(rng.standard_normal(3),device='cuda',dtype=torch.float64);u=mu[j,a].double()+ls[j,a].double().clamp(-5,2).exp()*eps;geo,diag=corrected_action(mu[j,a],ls[j,a],mask,eps);assert diag['latent']==u.cpu().tolist();lp=float(T['base_logprob'](mu[j,a],ls[j,a],u,mask));rp=float(T['base_logprob'](ref['mu'][j,a],ref['ls'][j,a],u,mask));geo=geo.cpu().tolist();after,check=extend_checked(w.state,seq[t],geo);COUNTS['simulated_transitions']+=1;repairs=sum(diag['adjusted_components']);rec['repairs']+=repairs;rec['proposals'].append([t,k,u.cpu().tolist(),geo,lp,rp,check['accepted'],repairs,check['reasons']])
    if policy!='C' or check['accepted']:
     w.state=after if after is not None else append(w.state,seq[t],geo);rec['geometry'].append(geo);accepted=True;break
   if not accepted:rec['termination']='ABORT_8_INVALID_PROPOSALS'
   else:
    ml=ls[j,a].double().clamp(-5,2);rl=ref['ls'][j,a].double().clamp(-5,2);kl=rl-ml+.5*((ml.exp()/rl.exp()).square()+((mu[j,a].double()-ref['mu'][j,a].double())/rl.exp()).square())-.5;rec['geometry_KL'].append(float((kl*torch.tensor(mask,device='cuda')).sum()/max(1,sum(mask))));nextlive.append(i)
  live=nextlive
 for w,rec in zip(ws,records):
  rec['sequence']=''.join(w.state.residues);rec['metrics']=c['score'](rec['sequence'],w.state.ca.numpy(),goal,SPEC);rec['seconds']=time.perf_counter()-start;rec['trace_logprob']=sum(q[4] for q in rec['proposals']);rec['reference_trace_logprob']=sum(q[5] for q in rec['proposals']);COUNTS['terminal_evaluations']+=1
 return records

def group(e,panel,index,task,policy,count=8):
 path=R/f'results/{panel}_{policy}_{index:03}.json'
 if path.exists():assert sha(path)==read(path.with_suffix('.sha.json'))['sha256'];return read(path)['records']
 before=dict(e.calls);rows=generate(e,task,policy,count);save(str(path.relative_to(R)),dict(task=task,policy=policy,records=rows,calls={k:e.calls[k]-before.get(k,0) for k in e.calls}));save(str(path.with_suffix('.sha.json').relative_to(R)),dict(sha256=sha(path)));print('GROUP',panel,policy,index,len(rows),'hard',sum(r['metrics']['hard_valid'] for r in rows),'full',sum(r['metrics']['success'] for r in rows),flush=True);return rows

def delayed(e,tasks):
 if (R/'DELAYED_ACTION_EFFECTS.json').exists():return
 out=[];selected=[]
 for gi,task in enumerate(tasks[:4]):
  pool=read(R/f'results/TRAIN_C_{gi:03}.json')['records'];goal=task['goal'];seq=task['fixed_sequence']
  for t in [8,32,48]:
   row=next((r for r in pool if len(r['sequence'])>=t),None)
   if row is None:continue
   state=reconstruct(row,64,t);assert prefix_status(state)['valid'];logits,mu,ls,ref,_=e.distribution([world(goal,state)],'C_FIXED');a=AA.index(seq[t]);actions=[]
   for k in range(16):
    rng=np.random.default_rng(ns('INTERVENTION',goal['goal_id'],t,k));geo,diag=corrected_action(mu[0,a],ls[0,a],c['mask_for_step'](t),torch.tensor(rng.standard_normal(3),device='cuda',dtype=torch.float64));after,check=extend_checked(state,seq[t],geo.cpu().tolist());COUNTS['simulated_transitions']+=1
    if check['accepted']:actions.append((k,geo.cpu().tolist(),diag['latent'],after))
    if len(actions)==3:break
   selected.append(dict(goal_id=goal['goal_id'],identity=goal['witness_identity'],t=t,attempts=k+1,retained=len(actions),prefix_source_candidate=row['candidate_index']))
   for ai,geo,u,after in actions:
    for split in ['FIT','CONFIRM']:
     seeds=[ns('CONTINUATION',goal['goal_id'],t,split,j) for j in range(16)];rows=generate(e,task,'C',16,after,seeds)
     for j,r in enumerate(rows):out.append(dict(goal_id=goal['goal_id'],identity=goal['witness_identity'],t=t,action=ai,action_geometry=geo,action_latent=u,split=split,replicate=j,seed=seeds[j],metrics=r['metrics'],proposal_count=len(r['proposals']),seconds=r['seconds']))
   print('DELAYED',gi,t,len(out),flush=True)
 save('DELAYED_ACTION_EFFECTS.json',dict(records=out,selection=selected,policy='frozen C',AA_fixed=True,matched_seeds_across_actions=True,independent_confirmation=True))

def search(e,tasks):
 if (R/'SUFFIX_SEARCH.json').exists():return
 out=[]
 for i,task in enumerate(tasks[:4]):
  pool=read(R/f'results/TRAIN_C_{i:03}.json')['records'];row=pool[0]
  if len(row['sequence'])<32:out.append(dict(goal_id=task['goal']['goal_id'],failure='initial prefix failed before32'));continue
  state=reconstruct(row,64,32);seeds=[ns('SUFFIX_SEARCH',task['goal']['goal_id'],j) for j in range(15)];before=dict(e.calls);rows=generate(e,task,'C',15,state,seeds);out.append(dict(goal_id=task['goal']['goal_id'],identity=task['goal']['witness_identity'],original_candidate_metrics=row['metrics'],records=rows,calls={k:e.calls[k]-before.get(k,0) for k in e.calls},initial_candidate_cost_included=True))
 save('SUFFIX_SEARCH.json',dict(records=out,algorithm='first full C candidate, rewind32 then15 frozen filtered suffixes; select exact terminal maximum',terminal_evaluations=16,not_neural_training=True))

def alternate(e,tasks):
 if (R/'SUPPORTED_ALTERNATIVE_GOALS.json').exists():return
 out=[]
 for i,task in enumerate(tasks):
  original=task['goal'];pool=read(R/f'results/TRAIN_C_{i:03}.json')['records'];seq=task['fixed_sequence']
  for ci,row in enumerate(pool):
   v=row['metrics']
   if not v['hard_valid'] or v.get('contrast',0)>-.15:continue
   b,z=original['buried'];a,y=original['exposed'];motifs=[seq[k:k+2] for k in range(b,z-1) if seq[k]!=seq[k+1]]
   if not motifs:continue
   goal=dict(original,goal_id=original['goal_id']+'_empirically_reversed',exposed=[b,z],buried=[a,y],exposure_min=math.floor(v['buried_mean']*20)/20,burial_max=math.ceil(v['exposed_mean']*20)/20,motif=motifs[0]);X=reconstruct(row).ca.numpy();assert c['score'](seq,X,goal,SPEC)['success'];newtask=dict(task,goal=goal);seeds=[ns('CANDIDATE',original['goal_id'],j) for j in range(8)];new=generate(e,newtask,'C',8,seeds=seeds);out.append(dict(task=newtask,witness_source=f'results/TRAIN_C_{i:03}.json',witness_candidate=ci,original_task=task,records=new,matched_original_first8=True));break
  if len(out)==4:break
 save('SUPPORTED_ALTERNATIVE_GOALS.json',dict(records=out,original_windows_reversed_only_if_observed_hard_valid_same_sequence_witness=True,unsupported_reversal_not_assumed=True))

def backward_check(e,task,rows):
 """Gradient audit only: no protein optimizer exists or takes a step."""
 if (R/'PROTEIN_TRACE_BACKWARD_TEST.json').exists():return
 before=e.audit_actor_fingerprint;e.actor.requires_grad_(True);e.actor.eval();ws=[world(task['goal']) for _ in rows];logref=np.asarray([r['reference_trace_logprob'] for r in rows]);logp=np.asarray([r['trace_logprob'] for r in rows]);reward=np.asarray([r['metrics']['reward'] for r in rows]);aug=reward-.01*(logp-logref);adv=aug-(aug.sum()-aug)/(len(aug)-1);maxerr=0.;start=time.perf_counter();replayed=0
 for t in range(64):
  active=[i for i,r in enumerate(rows) if any(q[0]==t for q in r['proposals'])]
  if not active:break
  worlds=[ws[i] for i in active];prefix=p['collate']([w.state for w in worlds],'cuda')
  with torch.no_grad():reference=e.prior(prefix);batch=p['public_batch'](worlds,prefix,reference)
  logits,mu,ls=e.actor(batch);terms=[]
  for j,i in enumerate(active):
   a=AA.index(task['fixed_sequence'][t]);local=[]
   for q in rows[i]['proposals']:
    if q[0]!=t:continue
    lp=T['base_logprob'](mu[j,a],ls[j,a],torch.tensor(q[2],device='cuda',dtype=torch.float64),c['mask_for_step'](t));maxerr=max(maxerr,abs(float(lp.detach())-q[4]));local.append(lp);replayed+=1
   terms.append((-.0-adv[i]+.01)*torch.stack(local).sum()/len(rows))
  loss=torch.stack(terms).sum();loss.backward()
  for i in active:
   # An aborted state contributes all rejected scores, with no accepted action.
   if t<len(rows[i]['geometry']):ws[i].state=append(ws[i].state,task['fixed_sequence'][t],rows[i]['geometry'][t])
 save('PROTEIN_REPLAY_NUMERIC_AUDIT.json',dict(group_size=len(rows),replayed_proposals=replayed,max_logprob_error=maxerr,behavior_batch_must_match_replay_batch=True));grads={name:float(v.grad.norm()) for name,v in e.actor.named_parameters() if v.grad is not None};assert grads and all(math.isfinite(v) for v in grads.values());assert any(v>0 for k,v in grads.items() if 'geometry' in k);assert not any(v>0 for k,v in grads.items() if k.startswith('aa.'));assert maxerr<5e-5;norm=math.sqrt(sum(z*z for z in grads.values()));e.actor.zero_grad(set_to_none=True);e.actor.requires_grad_(False);assert c['state_digest'](e.actor.state_dict())==before;verify_engine(e)
 # Measure a complete future Adam recovery without writing duplicated weights.
 from protein_joint_sac.crisp_v3.recovery import serialized_size
 actor_cpu={k:v.detach().cpu().clone() for k,v in e.actor.state_dict().items()};params=list(e.actor.named_parameters());ostate={i:dict(step=torch.tensor(1.),exp_avg=torch.zeros_like(v,device='cpu'),exp_avg_sq=torch.zeros_like(v,device='cpu')) for i,(name,v) in enumerate(params)};state=dict(actor=actor_cpu,optimizer=dict(state=ostate,param_groups=[dict(params=list(range(len(params))),lr=1e-6,betas=(.9,.999),eps=1e-8,weight_decay=0.,amsgrad=False,maximize=False,foreach=None,capturable=False,differentiable=False,fused=None)]),scheduler=dict(update=1,total=5000),RNG=dict(torch=torch.get_rng_state(),numpy=np.random.default_rng(9317).bit_generator.state),next_identity=task['goal']['goal_id'],update=1,group_size=16,manifest_SHA256=sha(R/'manifests/TRAIN_TASKS.json'),reference_SHA256=c['PSHA'],sampler='stateless candidate/step/rejection-index seeds',replay='full proposal trace file references and committed journal hashes',configuration=read(R/'FROZEN_PROTOCOL.json'));size=serialized_size(state)
 save('PROTEIN_TRACE_BACKWARD_TEST.json',dict(PASS=True,no_protein_optimizer_created=True,protein_updates=0,actor_unchanged=True,J1_unchanged=True,replayed_proposals=replayed,logprob_max_error=maxerr,total_gradient_norm=norm,geometry_head_gradient=True,forced_AA_head_score_gradient_zero=True,backward_seconds=time.perf_counter()-start,complete_recovery_measured_bytes=size,measurement='torch serialization counting sink, no duplicate model artifact',peak_GPU_bytes=torch.cuda.max_memory_allocated(),grads=grads));print('BACKWARD_PASS',size,norm,flush=True)

def main():
 assert read(R/'POLICY_GRADIENT_TESTS.json')['PASS'];guard();e,_=load_engine('C');stage=sys.argv[1];train=read(R/'manifests/TRAIN_TASKS.json')['tasks']
 if stage=='train':
  for i,task in enumerate(train):
   for pol in ['A','B','C']:group(e,'TRAIN',i,task,pol,16 if pol=='C' and i<4 else 8)
  backward_check(e,train[0],read(R/'results/TRAIN_C_000.json')['records']);search(e,train);alternate(e,train);delayed(e,train)
 elif stage=='validation':
  assert read(R/'TRAIN_READINESS.json')['baseline_evaluation_admitted'];tasks=read(R/'manifests/PROSPECTIVE_FIXED.json')['tasks']
  for i,task in enumerate(tasks):
   for pol in ['A','B','C']:group(e,'PROSPECTIVE',i,task,pol)
 verify_engine(e);save(f'COMPLETED_{stage}.json',dict(PASS=True,counts=dict(COUNTS),calls=dict(e.calls),seconds=time.perf_counter()-START,peak_GPU_bytes=torch.cuda.max_memory_allocated(),actor_fp=e.audit_actor_fingerprint,J1_fp=e.jfp,actor_parameters=sum(v.numel() for v in e.actor.parameters()),J1_parameters=sum(v.numel() for v in e.prior.model.parameters()),protein_updates=0,all_frozen_at_exit=True,source_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip()));print('COMPLETE',stage,dict(COUNTS),flush=True)
if __name__=='__main__':main()
