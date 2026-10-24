"""First substantial fixed-sequence campaign implementation. NOT authorized in V93.
--preflight is read-only apart from its report. --execute requires an explicit
future authorization receipt and admission/recovery release, never made here.
"""
import argparse,collections,copy,json,math,os,random,runpy,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
R=Path('/mnt/d/Simone/proteinGen_data/e017_v93_regional_controllability_20261024_v1');S=runpy.run_path(str(R/'source/scripts/e017_v93/study.py'));globals().update({k:S[k] for k in ['c','p','T','AA','SPEC','read','sha','ns','world','append','load_engine']})
from protein_joint_sac.crisp_v3.recovery import atomic_checkpoint,StoragePolicy,atomic_json
class LiveActor:
 def __init__(self,e):self.e=e;self.calls=e.calls
 def distribution(self,ws,tag):return self.e.distribution(ws,'D_RL' if tag=='C_FIXED' else tag)

def gradients(e,task,rows,beta):
 rewards=np.array([r['metrics']['reward'] for r in rows]);lp=np.array([r['trace_logprob'] for r in rows]);rp=np.array([r['reference_trace_logprob'] for r in rows]);aug=rewards-beta*(lp-rp);adv=aug-(aug.sum()-aug)/(len(rows)-1);worlds=[world(task['goal']) for r in rows];maxerr=0.;probes=[]
 for t in range(64):
  active=[i for i,r in enumerate(rows) if any(q[0]==t for q in r['proposals'])]
  for start in range(0,len(active),16):
   ids=active[start:start+16];ws=[worlds[i] for i in ids];prefix=p['collate']([w.state for w in ws],'cuda')
   with torch.no_grad():ref=e.prior(prefix);batch=p['public_batch'](ws,prefix,ref)
   logits,mu,ls=e.actor(batch);loss=[]
   for j,i in enumerate(ids):
    a=AA.index(task['fixed_sequence'][t]);terms=[]
    for q in rows[i]['proposals']:
     if q[0]==t:
      value=T['base_logprob'](mu[j,a],ls[j,a],torch.tensor(q[2],dtype=torch.float64,device='cuda'),c['mask_for_step'](t));maxerr=max(maxerr,abs(float(value.detach())-q[4]));terms.append(value)
    loss.append((-adv[i]+beta)*torch.stack(terms).sum()/len(rows))
   torch.stack(loss).sum().backward()
   if t in [1,16,32,48,63]:probes.append((copy.deepcopy(ws),mu.detach().clone(),ls.detach().clone(),AA.index(task['fixed_sequence'][t])))
  for i in active:
   if t<len(rows[i]['geometry']):worlds[i].state=append(worlds[i].state,task['fixed_sequence'][t],rows[i]['geometry'][t])
 assert maxerr<5e-5,('on-policy replay mismatch',maxerr)
 return rewards,maxerr,probes

def cpu(x):
 if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
 if isinstance(x,list):return [cpu(v) for v in x]
 if isinstance(x,tuple):return tuple(cpu(v) for v in x)
 return copy.deepcopy(x)
def rng():return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate())
def setrng(s):torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda']);np.random.set_state(s['numpy']);random.setstate(s['python'])
def run(spec,authorization,root,resume=None):
 # Current V93 evidence never contains either release receipt. These gates
 # prevent a preflight invocation from starting neural optimization.
 assert authorization['explicit_user_authorization'] is True and authorization['study']=='FIRST_SUBSTANTIAL_FIXED_SEQUENCE_RL';assert sha(R/'SUBSTANTIAL_CAMPAIGN_SPEC.json')==authorization['plan_SHA256'];release=read(Path(authorization['protein_recovery_release_receipt']));assert release['bit_exact_protein_optimizer_sampler_RNG_replay'] is True and release['current_execution_source_SHA256']==sha(Path(__file__));assert read(R/'TRAIN_READINESS.json')['environment_GO']
 assert shutil.disk_usage(root.parent).free>=spec['minimum_free_bytes'];root.mkdir(exist_ok=False) if not resume else None
 policy=StoragePolicy(min_headroom_bytes=2*2**30,near_term_growth_bytes=128*2**20,checkpoint_hint_bytes=spec['measured_checkpoint_bytes']);e,_=load_engine('C');e.actor.requires_grad_(True).train();live=LiveActor(e);opt=torch.optim.AdamW(e.actor.parameters(),lr=spec['peak_lr'],weight_decay=0.);schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda k:min(1.,(k+1)/100)*(.1+.9*.5*(1+math.cos(math.pi*max(0,k-4000)/1000))) if k>=4000 else min(1.,(k+1)/100));tasks=read(R/'manifests/FUTURE_FIXED_TRAIN_POOL.json')['tasks'];ids=sorted({t['source_reference_identity'] for t in tasks});lookup={identity:[t for t in tasks if t['source_reference_identity']==identity] for identity in ids};order=np.random.default_rng(spec['seed']).permutation(len(ids)).tolist();update=0;best=None;beta=spec['trace_KL_beta'];config_hash=sha(R/'SUBSTANTIAL_CAMPAIGN_SPEC.json');source_hash=sha(Path(__file__))
 def checkpoint(reason):
  state=dict(schema='V93_FULL_TRACE_FIXED_RECOVERY_1',actor=cpu(e.actor.state_dict()),optimizer=cpu(opt.state_dict()),scheduler=schedule.state_dict(),AMP=None,RNG=rng(),next_update=update+1,update=update,next_identity=ids[order[update%len(order)]],identity_order=order,task_manifest_SHA256=sha(R/'manifests/FUTURE_FIXED_TRAIN_POOL.json'),config_SHA256=config_hash,source_SHA256=source_hash,J1_SHA256=c['J1SHA'],parent_SHA256=c['PSHA'],beta=beta,best=best,reason=reason,accepted_attempts=16*update,journal='UPDATES.jsonl',pending_rollout=None)
  record=atomic_checkpoint(root/'latest.pt',state,policy);atomic_json(root/'LATEST.json',record)
  if update in spec['named_updates']:
   # Link only new run-owned immutable files; never mutate a linked inode.
   # If unsupported, full copy is admitted by the conservative storage budget.
   name=root/f'milestone_{update:05}.pt'
   if not name.exists():
    try:os.link(root/'latest.pt',name)
    except OSError:shutil.copyfile(root/'latest.pt',name)
  return record
 if resume:
  state=torch.load(resume,map_location='cpu',weights_only=False);assert state['config_SHA256']==config_hash and state['source_SHA256']==source_hash and state['task_manifest_SHA256']==sha(R/'manifests/FUTURE_FIXED_TRAIN_POOL.json');e.actor.load_state_dict(state['actor'],strict=True);opt.load_state_dict(state['optimizer']);schedule.load_state_dict(state['scheduler']);setrng(state['RNG']);update=state['update'];order=state['identity_order'];best=state['best'];beta=state['beta']
 else:torch.manual_seed(spec['seed']);np.random.seed(spec['seed']);random.seed(spec['seed']);checkpoint('INITIAL')
 # A full microbatch/replay admission run and monitored development evaluator
 # are mandatory release callbacks, supplied and hashed before authorization.
 evaluator=runpy.run_path(authorization['development_evaluator_source'])['evaluate'];assert sha(Path(authorization['development_evaluator_source']))==authorization['development_evaluator_SHA256'];baseline=evaluator(e,root,0)
 for u in range(update+1,spec['updates']+1):
  if (root/'STOP').exists():checkpoint('COOPERATIVE_PAUSE');return
  policy.require(root,spec['measured_checkpoint_bytes']);identity=ids[order[(u-1)%len(order)]];task=lookup[identity][((u-1)//len(order))%len(lookup[identity])];e.actor.eval();S['generate'].__globals__['START']=time.perf_counter();S['COUNTS'].clear();seeds=[ns('SUBSTANTIAL',spec['seed'],u,i) for i in range(16)];rows=S['generate'](live,task,'C',16,seeds=seeds);tracefile=root/'PENDING_ROLLOUT.json';atomic_json(tracefile,dict(update=u,behavior_fingerprint=c['state_digest'](e.actor.state_dict()),task=task,records=rows));opt.zero_grad(set_to_none=True);e.actor.train();reward,error,probes=gradients(e,task,rows,beta);norm=float(torch.nn.utils.clip_grad_norm_(e.actor.parameters(),5.));assert math.isfinite(norm);before=cpu(e.actor.state_dict());obefore=cpu(opt.state_dict());sbefore=copy.deepcopy(schedule.state_dict());opt.step();schedule.step();e.actor.eval();maxstep=0.
  with torch.no_grad():
   for ws,oldmu,oldls,a in probes:
    _,mu,ls,_,_=e.distribution(ws,'D_RL');mask=torch.tensor(c['mask_for_step'](ws[0].state.t),device='cuda');new=ls[:,a].double();old=oldls[:,a].double();delta=new-old+.5*((old.exp()/new.exp()).square()+((oldmu[:,a].double()-mu[:,a].double())/new.exp()).square())-.5;maxstep=max(maxstep,float((delta*mask).sum(-1).max()))
  if not math.isfinite(maxstep) or maxstep>.02:e.actor.load_state_dict(before);opt.load_state_dict(obefore);schedule.load_state_dict(sbefore);checkpoint('TRAINING_STABILITY_STEP_KL_ROLLBACK');return
  update=u
  with (root/'UPDATES.jsonl').open('a') as f:f.write(json.dumps(dict(update=u,identity=identity,goal_id=task['goal']['goal_id'],attempts=16,complete=sum(z['metrics']['complete'] for z in rows),full=sum(z['metrics']['success'] for z in rows),hard=sum(z['metrics']['hard_valid'] for z in rows),reward=float(reward.mean()),grad_norm=norm,step_KL=maxstep,geometry_KL=float(np.mean([v for z in rows for v in z['geometry_KL']])),trace_SHA256=sha(tracefile),seeds=seeds))+'\n');f.flush();os.fsync(f.fileno())
  if u%50==0:checkpoint('ROLLING')
  if u in spec['development_updates']:
   report=evaluator(e,root,u)
   if report['hard_valid']<baseline['hard_valid']-.05 or report['backbone_valid']<baseline['backbone_valid']-.05 or report['geometry_KL_per_active']>1 or report['structural_pair_RMSD']<.8*baseline['structural_pair_RMSD']:checkpoint('TRAINING_STABILITY_MONITOR_STOP');return
   if best is None or report['K1']>best['K1']:
    best=dict(update=u,**report);record=checkpoint('DEVELOPMENT_BEST');target=root/'best.pt';tmp=root/'best.pt.part'
    try:os.link(root/'latest.pt',tmp)
    except OSError:shutil.copyfile(root/'latest.pt',tmp)
    os.replace(tmp,target)
  e.prior.verify();assert c['state_digest'](e.parent.state_dict())==e.fp
 checkpoint('COMPLETED_DECLARED_BUDGET_PAUSED')
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--execute',action='store_true');parser.add_argument('--authorization');parser.add_argument('--output');parser.add_argument('--resume');args=parser.parse_args();spec=read(R/'SUBSTANTIAL_CAMPAIGN_SPEC.json');free=shutil.disk_usage(R).free
 if args.preflight:
  report=dict(storage_GO=free>=spec['minimum_free_bytes'],free_bytes=free,required_bytes=spec['minimum_free_bytes'],launch_authorized=False,protein_updates=0,missing_release=['future explicit user authorization','full protein optimizer/RNG replay under exact campaign configuration','hashed development evaluator release']);(R/'CAMPAIGN_PREFLIGHT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return
 if not args.execute or not args.authorization or not args.output:raise SystemExit('No training: explicit future authorization, output and execution flag are required')
 run(spec,read(Path(args.authorization)),Path(args.output),args.resume)
if __name__=='__main__':main()
