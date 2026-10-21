"""Sealed supervised-initialization admission. No optimizer work on import/preflight."""
import copy,hashlib,json,math,random,time
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import torch
from torch.nn import functional as F
from protein_joint_sac.contracts import AA,BuildState,BuildAction
from protein_joint_sac.models.batched_bc import collate
from protein_joint_sac.data.scaled_supervised import canonical,internal
from protein_joint_sac.data.partial_geometry import derive
from protein_joint_sac.geometry.ca_builder import next_ca,append
from protein_joint_sac.models.geometry_inference import corrected_action
from .environment import frame,goal_tokens,assess
from .networks import prepare
from .density import log_prob,kl
from .prior import file_sha,tensor_sha
from .storage import digest

BANDS=[(20,63),(64,96),(97,128),(129,256),(257,384),(385,500)]
CATEGORIES=['complete','partial','aa_only']
def band(N):
 for lo,hi in BANDS:
  if lo<=N<=hi:return f'{lo}-{hi}'
 raise ValueError('length outside certified range')
def rank(seed,*v):return hashlib.sha256(('|'.join(map(str,(seed,)+v))).encode()).hexdigest()
def owner(pop,h,category,split):
 if h not in pop.groups[split][category]:raise ValueError('authoritative split/population mismatch')
 if category=='aa_only':return None
 owners=pop.base.owners[h] if category=='complete' else pop.partial_owners[h]
 rows=pop.base.rows if category=='complete' else pop.partial
 owners=sorted(o for o in owners if rows[o]['split']==split)
 return min(owners,key=lambda o:rank(1729,h,o))
def native_task(pop,h,o,kind,split,seed=1729):
 r=pop.base.rows[o]
 if r['sequence_hash']!=h or r['split']!=split or pop.base.proposal.sequences[h]['split']!=split:raise ValueError('goal ownership')
 seq=pop.base.seqs[h];x=canonical(pop.coordinates(r));N=len(seq);rng=random.Random(rank(seed,h,kind))
 targets=[]
 for i in ([] if kind=='P0' else [N-1] if kind=='P1' else [N//2,N-1]):
  targets.append(dict(index=i,position=x[i].tolist(),tolerance_A=8.,orientation=frame(torch.tensor(x[:i+1])).tolist() if kind=='P2' else None,orientation_tolerance_deg=35.))
 motifs={str(i):seq[i] for i in sorted(rng.sample(range(3,N),1 if kind=='P0' else 2))}
 return dict(schema='crisp_goal_v3',kind=kind,N=N,targets=targets,motifs=motifs,source_identity=h,source_observation=o,source_sha=r['source_sha'],split=split,initial_sequence=seq[:3],initial_CA=x[:3].tolist(),experimental=True)

def panel_record(pop,h,category,split):
 seq=pop.base.seqs[h];N=len(seq);o=owner(pop,h,category,split);s=pop.base.proposal.sequences[h]
 r=dict(identity=h,category=category,split=split,N=N,length_band=band(N),observation=o,positions=sorted(set(min(N-1,max(3,int((N-1)*f))) for f in [.125,.375,.625,.875])),protected_group_id=s.get('protected_group_id'))
 if o is not None:
  source=(pop.base.rows if category=='complete' else pop.partial)[o];r.update(source_sha256=source['source_sha'],coordinate_path=source['coordinate_path'],coordinate_row=source['coordinate_row'])
 return r

def build_panels(pop,limits,seed=1729,per_cell=8):
 """Select by seeded identity hash BEFORE predictions; never select on scores/validity."""
 manifests={}
 for split in ['TRAIN','VALIDATION']:
  rows=[];tasks=[];generation=[];available={}
  for lo,hi in BANDS:
   for category in CATEGORIES:
    ids=[h for h in pop.groups[split][category] if lo<=len(pop.base.seqs[h])<=hi];available[f'{lo}-{hi}/{category}']=len(ids)
    selected=sorted(ids,key=lambda h:rank(seed,split,category,lo,h))[:per_cell]
    for j,h in enumerate(selected):
     r=panel_record(pop,h,category,split);rows.append(r)
     if category=='complete' and lo>=64:
      x=torch.tensor(canonical(pop.coordinates(pop.base.rows[r['observation']])));seq=pop.base.seqs[h]
      for kind in ['P0','P1','P2']:
       task=native_task(pop,h,r['observation'],kind,split,seed);w=assess(BuildState(tuple(seq),x,len(seq)),task,limits)
       # Native witness failures remain in panel as flagged out-of-calibration stress cases.
       positions=sorted(set([len(seq)//3]+[p['index'] for p in task['targets']]+[int(i) for i in task['motifs']]))
       tasks.append(dict(identity=h,observation=r['observation'],category='complete',split=split,N=len(seq),length_band=r['length_band'],task=task,positions=positions,native_witness_valid=w['valid'],native_witness_success=w['success']))
       if j<2 and split=='VALIDATION':
        for repeat in range(2):generation.append(dict(task_index=len(tasks)-1,seed=180000+len(generation),tier='core' if j==0 and hi<=128 else 'standard' if j==0 else 'extended',repeat=repeat))
  manifests[split]=dict(schema='crisp_frozen_panel_v1',split=split,seed=seed,selection='seeded SHA256 identity ranking; no prediction or witness filtering',available_cells=available,records=rows,tasks=tasks,generation=generation,input_hashes=pop.input_hashes(),identity_weighting='equal length band, equal population, equal identity; equal target positions within identity',task_weighting='equal length band, equal P0/P1/P2, equal identity, equal positions',no_TEST_outcomes=True)
 return manifests

def verify_panel(pop,panel,expected_split):
 if panel['schema']!='crisp_frozen_panel_v1' or panel['split']!=expected_split or expected_split not in ['TRAIN','VALIDATION']:raise ValueError('panel split/schema')
 if panel['input_hashes']!=pop.input_hashes():raise ValueError('frozen data hashes')
 seen=set()
 for r in panel['records']:
  h=r['identity'];category=r['category'];key=(h,category)
  if key in seen:raise ValueError('duplicate identity weighting')
  seen.add(key)
  if r['split']!=expected_split or pop.base.proposal.sequences[h]['split']!=expected_split or h not in pop.groups[expected_split][category]:raise ValueError('identity split leakage')
  if r['N']!=len(pop.base.seqs[h]) or r['length_band']!=band(r['N']) or any(not 0<=t<r['N'] for t in r['positions']):raise ValueError('position/length')
  if r['observation']!=owner(pop,h,category,expected_split):raise ValueError('observation ownership')
 for r in panel['tasks']:
  if r['identity'] not in {v['identity'] for v in panel['records'] if v['category']=='complete'} or r['split']!=expected_split:raise ValueError('task not in same split complete population')
  t=r['task'];h=r['identity'];seq=pop.base.seqs[h]
  if t['source_identity']!=h or t['source_observation']!=r['observation'] or t['split']!=expected_split:raise ValueError('goal provenance')
  if any(seq[int(i)]!=aa for i,aa in t['motifs'].items()):raise ValueError('motif mapping')
 return dict(identities=len(seen),AA_decisions=sum(len(r['positions']) for r in panel['records']),tasks=len(panel['tasks']),goal_decisions=sum(len(r['positions']) for r in panel['tasks']),free_generation_attempts=len(panel['generation']),cells=Counter((r['length_band'],r['category']) for r in panel['records']))

def load_decision(pop,r,t,device,geometry_free=False):
 """Read-only loader checks TRUE source split, not a caller-provided tag."""
 h=r['identity'];category=r['category'];split=r['split'];N=len(pop.base.seqs[h]);seq=pop.base.seqs[h]
 if split not in ['TRAIN','VALIDATION'] or pop.base.proposal.sequences[h]['split']!=split or h not in pop.groups[split][category]:raise ValueError('authoritative identity ownership')
 if not 0<=t<N:raise ValueError('target position')
 g=np.zeros((N,3));mask=np.zeros((N,3),bool);valid=np.zeros(N,bool);links=np.zeros(N-1,bool);x=np.zeros((N,3))
 if category!='aa_only':
  o=r['observation'];source=(pop.base.rows if category=='complete' else pop.partial)[o]
  if source['sequence_hash']!=h or source['split']!=split:raise ValueError('observation ownership')
  x=pop.coordinates(source)
  if category=='complete':x=canonical(x);g=internal(x);valid[:]=True;links[:]=True;mask=np.arange(N)[:,None]>=np.arange(1,4)[None]
  else:
   valid=np.array(source['ca_observed'],bool);links=np.array(source['continuity'],bool);x=np.where(valid[:,None],x,0.);g,mask,_=derive(x,valid,links)
 if geometry_free:x=np.zeros_like(x);valid=np.zeros_like(valid);links=np.zeros_like(links)
 s=BuildState(tuple(seq[:t]),torch.tensor(x[:t].copy(),dtype=torch.float64),N,torch.tensor(valid[:t]));p=collate([s],device);p['bond_valid']=torch.zeros_like(p['valid'])
 if t>=2:p['bond_valid'][0,1:t]=torch.tensor(links[:t-1],device=device)&p['valid'][0,:t-1]&p['valid'][0,1:t]
 return p,torch.tensor([AA.index(seq[t])],device=device),torch.tensor(g[t:t+1].copy(),device=device,dtype=torch.float64),torch.tensor(mask[t:t+1].copy(),device=device)

def configure_initialization(system,learner,schedule):
 """Preserve full system capacity; exactly ONE optimizer is permitted to update."""
 for key,m in system.models.items():
  m.requires_grad_(key=='actor')
  if key!='actor':m.eval()
 system.targets.requires_grad_(False).eval()
 actor_ids={id(p) for p in system.models['actor'].parameters()}
 # Retain serializable inactive optimizer/scheduler entries for exact whole-system recovery.
 # No inactive moment allocation or step is allowed by the initialization orchestrator.
 if {id(p) for g in learner.optimizers['actor'].param_groups for p in g['params']}!=actor_ids:raise ValueError('actor optimizer membership')
 warm=schedule['warmup'];total=schedule['total_updates'];floor=schedule['floor_ratio']
 def multiplier(k):
  if k<warm:return (k+1)/warm
  return floor+(1-floor)*.5*(1+math.cos(math.pi*min(1,(k-warm)/max(1,total-warm))))
 learner.optimizers={'actor':learner.optimizers['actor']}
 learner.steps={'actor':learner.steps['actor']}
 learner.schedulers={'actor':torch.optim.lr_scheduler.LambdaLR(learner.optimizers['actor'],multiplier)}
 return dict(active_parameters=sum(p.numel() for p in system.parameters() if p.requires_grad),inactive_parameters=sum(p.numel() for p in system.models.parameters() if not p.requires_grad),actor_optimizer_only=True)

class InitGoals:
 """TRAIN-only identity sweeps; categories balanced across a three-decision cycle."""
 def __init__(self,pop,seed,limits):
  self.population=pop;self.rng=random.Random(seed);self.limits=limits;self.stage=0;self.prefix_cursor=0;self.draws=0;self.rejected=0;self.category_cursor=0;self.orders={};self.cursors={};self.seen=set();self.kind_counts=Counter()
 def draw(self):
  upper=[96,128,256][min(self.stage,2)];key=str(upper)
  if key not in self.orders:
   self.orders[key]=sorted(h for h in self.population.groups['TRAIN']['complete'] if 64<=len(self.population.base.seqs[h])<=upper);self.rng.shuffle(self.orders[key]);self.cursors[key]=0
  order=self.orders[key]
  if not order:raise ValueError('empty goal curriculum')
  for _ in range(64):
   i=self.cursors[key]
   if i==len(order):self.rng.shuffle(order);i=0
   h=order[i];self.cursors[key]=i+1;o=self.rng.choice(self.population.base.owners[h]);kind=['P0','P1','P2'][self.category_cursor%3];task=native_task(self.population,h,o,kind,'TRAIN',self.rng.randrange(2**32));x=canonical(self.population.coordinates(self.population.base.rows[o]));seq=self.population.base.seqs[h]
   if not assess(BuildState(tuple(seq),torch.tensor(x),len(seq)),task,self.limits)['success']:self.rejected+=1;continue
   self.category_cursor+=1;self.draws+=1;self.seen.add(h);self.kind_counts[kind]+=1;return task
  raise ValueError('no native-valid TRAIN goal in bounded 64 identity draws')
 def state_dict(self):return dict(rng=self.rng.getstate(),stage=self.stage,prefix_cursor=self.prefix_cursor,draws=self.draws,rejected=self.rejected,category_cursor=self.category_cursor,orders=self.orders,cursors=self.cursors,seen=self.seen,kind_counts=dict(self.kind_counts),limits=self.limits,membership=self.population.membership()['TRAIN']['membership_sha256'])
 def load_state_dict(self,s):
  if s['limits']!=self.limits or s['membership']!=self.population.membership()['TRAIN']['membership_sha256']:raise ValueError('goal sampler provenance')
  self.rng.setstate(s['rng'])
  for k in ['stage','prefix_cursor','draws','rejected','category_cursor','orders','cursors','seen']:setattr(self,k,copy.deepcopy(s[k]))
  self.kind_counts=Counter(s['kind_counts'])

def deterministic_goal_loss(b,mu,ls,mask):
 """Local one-step, deterministic latent-mean representative; NOT physical expectation.
 Positions and right-handed frames only at explicit next-residue targets. No suffix derivative.
 """
 terms=[]
 for i in range(len(mu)):
  t=int(b['prefix']['t'][i]);N=int(b['prefix']['N'][i]);positions=(b['goals'][i,:,32]*N).round().long();eligible=b['goal_valid'][i]&b['goals'][i,:,33].bool()&(positions==t)
  if not eligible.any() or t<3 or not mask[i].all() or not b['prefix']['valid'][i,:t].all():continue
  u=mu[i].double();g=torch.stack([u[0].exp(),math.pi*u[1].sigmoid(),math.pi*u[2].tanh()]);x=b['prefix']['x'][i,:t].double();tip=next_ca(x,g);target=b['goals'][i,eligible,:3].double()*50;d2=(tip-target).square().sum(-1);value=(d2/(64+d2)).mean();oriented=eligible&b['goals'][i,:,34].bool()
  if oriented.any():
   f=frame(torch.cat([x,tip[None]],0));wanted=b['goals'][i,oriented,3:12].reshape(-1,3,3).double();orientation=(3-(f[None]*wanted).sum((1,2)))/4;value=value+.5*orientation.clamp(0,1).mean()
  terms.append(value)
 return torch.stack(terms).mean() if terms else (mu.sum()+ls.sum())*0

def actor_loss(system,b,aa,g,valid,weights,with_goal=False):
 l,mu,ls=system.models['actor'](b);idx=torch.arange(len(aa),device=aa.device);selected_mu=mu[idx,aa];selected_ls=ls[idx,aa];ce=F.cross_entropy(l,aa,reduction='none');ka,kg,_=kl(l,mu,ls,b['reference'],valid);count=valid.sum(-1);has=count>0
 nll=-log_prob(selected_mu,selected_ls,g,valid)/count.clamp_min(1)
 # Rows without valid experimental targets contribute ZERO geometry likelihood / geometry KL.
 geometry=nll[has].mean() if has.any() else (mu.sum()+ls.sum())*0
 geometry_kl=(kg/count.clamp_min(1))[has].mean() if has.any() else (mu.sum()+ls.sum())*0
 goal=deterministic_goal_loss(b,selected_mu,selected_ls,valid) if with_goal else mu.sum()*0
 values=dict(aa=ce.mean(),geometry=geometry,kl_aa=ka.mean(),kl_geometry=geometry_kl,goal=goal)
 total=sum(weights[k]*values[k] for k in values)
 return total,values

def balanced_mean(rows,key,goal=False):
 # Fixed equal band/population/identity weights; no observation-count weighting.
 per=defaultdict(list)
 for r in rows:
  if key in r and r[key] is not None:per[(r['length_band'],r.get('kind') if goal else r['category'],r['identity'])].append(r[key])
 cells=defaultdict(list)
 for (length,c,h),v in per.items():cells[(length,c)].append(float(np.mean(v)))
 return float(np.mean([np.mean(v) for v in cells.values()])) if cells else None

@torch.no_grad()
def evaluate(system,prior,pop,panel,device,config,full=True):
 verify_panel(pop,panel,panel['split']);m=system.models['actor'];was=m.training;m.eval();records=[];tasks=[]
 try:
  for r in panel['records']:
   for t in r['positions']:
    for condition in ['native','geometry_free']:
     p,aa,g,mask=load_decision(pop,r,t,device,condition=='geometry_free');b=prepare(p,prior(p),torch.zeros(1,1,36,device=device),torch.zeros(1,1,device=device,dtype=torch.bool));l,mu,ls=m(b);a=int(aa[0]);pred=l.argsort(-1,descending=True)[0];x=dict(identity=r['identity'],category=r['category'],length_band=r['length_band'],t=t,relative_position=t/r['N'],condition=condition,aa=float(F.cross_entropy(l,aa)),top1=float(pred[0]==a),top3=float((pred[:3]==a).any()))
     if r['category']!='aa_only' and condition=='native':
      for c,name in enumerate(['distance','angle','torsion']):
       cm=torch.zeros_like(mask);cm[:,c]=mask[:,c];x[name]=float(-log_prob(mu[:,a],ls[:,a],g,cm)) if cm.any() else None
     records.append(x)
  for r in panel['tasks']:
   base=next(v for v in panel['records'] if v['identity']==r['identity']);task=r['task']
   for t in r['positions']:
    p,aa,g,mask=load_decision(pop,base,t,device);ref=prior(p);gt,gv=goal_tokens(task,device);b=prepare(p,ref,gt,gv);loss,values=actor_loss(system,b,aa,g,mask,config['stream_weights']['goal'],True);l,mu,ls=m(b);zero=prepare(p,ref,torch.zeros_like(gt),torch.zeros_like(gv));l0,mu0,ls0=m(zero);prob=l.softmax(-1);sensitivity=(prob*(l.log_softmax(-1)-l0.log_softmax(-1))).sum(-1)
    tasks.append(dict(identity=r['identity'],category='complete',length_band=r['length_band'],kind=task['kind'],t=t,loss=float(loss),aa=float(values['aa']),geometry=float(values['geometry']),goal=float(values['goal']),goal_AA_KL=float(sensitivity),goal_geometry_mu_change=float((mu-mu0).abs().mean()),native_witness_valid=r['native_witness_valid']))
 finally:m.train(was)
 summary={}
 for cond in ['native','geometry_free']:
  rr=[r for r in records if r['condition']==cond];summary[cond]={k:balanced_mean(rr,k) for k in ['aa','top1','top3','distance','angle','torsion']};summary[cond]['ppl']=math.exp(summary[cond]['aa'])
 summary['goal']={k:balanced_mean(tasks,k,True) for k in ['loss','aa','geometry','goal','goal_AA_KL','goal_geometry_mu_change']}
 summary['by_length']={lb:{cond:{k:balanced_mean([r for r in records if r['condition']==cond and r['length_band']==lb],k) for k in ['aa','top1','top3','distance','angle','torsion']} for cond in ['native','geometry_free']} for lb in sorted({r['length_band'] for r in records})}
 summary['by_population']={cat:{cond:{k:balanced_mean([r for r in records if r['condition']==cond and r['category']==cat],k) for k in ['aa','top1','top3']} for cond in ['native','geometry_free']} for cat in CATEGORIES}
 summary['goal_by_kind']={kind:{k:balanced_mean([r for r in tasks if r['kind']==kind],k,True) for k in ['aa','geometry','goal','goal_AA_KL']} for kind in ['P0','P1','P2']}
 summary['goal_by_witness_validity']={str(flag):{k:balanced_mean([r for r in tasks if r['native_witness_valid']==flag],k,True) for k in ['aa','geometry','goal']} for flag in [False,True]}
 summary['by_relative_position']={str(j):{cond:balanced_mean([r for r in records if r['condition']==cond and min(3,int(r['relative_position']*4))==j],'aa') for cond in ['native','geometry_free']} for j in range(4)}
 return dict(split=panel['split'],summary=summary,records=records,tasks=tasks)

@torch.no_grad()
def free_generation(system,prior,pop,panel,device,limits,tier='core',use_reference=False):
 """Independent prefix recomputation, matched RNG draws, no planner or hidden suffix."""
 rows=[];m=system.models['actor'];was=m.training;m.eval();allowed={'core':{'core'},'standard':{'core','standard'},'extended':{'core','standard','extended'}}[tier]
 try:
  for attempt in panel['generation']:
   if attempt['tier'] not in allowed:continue
   task=panel['tasks'][attempt['task_index']]['task'];rng=torch.Generator().manual_seed(attempt['seed']);s=BuildState(tuple(task['initial_sequence']),torch.tensor(task['initial_CA'],dtype=torch.float64),task['N']);out=dict(identity=task['source_identity'],kind=task['kind'],length_band=band(task['N']),seed=attempt['seed'],reference=use_reference,completed=False)
   try:
    while s.t<s.N:
     p=collate([s],device);ref=prior(p);gt,gv=goal_tokens(task,device);b=prepare(p,ref,gt,gv)
     l,mu,ls=(ref['logits'],ref['mu'],ref['ls']) if use_reference else m(b)
     u=torch.rand((),generator=rng);a=int(torch.searchsorted(l.softmax(-1)[0].cpu().cumsum(-1),u).clamp_max(19));eps=torch.randn(3,generator=rng,dtype=torch.float64);g,_=corrected_action(mu[0,a].cpu(),ls[0,a].cpu(),(True,True,True),eps);s=append(s,BuildAction(AA[a],g,(True,True,True)))
    out.update(assess(s,task,limits));out['completed']=True
    x=s.ca;d=torch.linalg.vector_norm(x[1:]-x[:-1],dim=-1);gi=internal(x.numpy());D=torch.cdist(x,x);ix=torch.arange(s.N);long=(ix[:,None]-ix[None,:])>=12
    out.update(finite=bool(torch.isfinite(x).all()),adjacent_min_A=float(d.min()),adjacent_max_A=float(d.max()),long_range_contact_fraction=float(((D<8)&long).sum()/long.sum().clamp_min(1)),angle_mean_rad=float(np.mean(gi[2:,1])),signed_torsion_mean_rad=float(np.mean(gi[3:,2])),sequence_composition={aa:s.residues.count(aa)/s.N for aa in AA})
    out['sequence_sha256']=hashlib.sha256(''.join(s.residues).encode()).hexdigest()
   except (ValueError,RuntimeError,FloatingPointError) as e:out['failure']=str(e)
   rows.append(out)
 finally:m.train(was)
 return rows

class Monitor:
 """Prospective patience rules; state lives in full recovery counters."""
 def __init__(self,policy,state=None):self.policy=policy;self.state=copy.deepcopy(state) if state else dict(baseline=None,history=[],strikes={})
 def observe(self,update,train,val,generation):
  s=self.state
  row=dict(update=update,train=train['summary'],validation=val['summary'],generation=generation)
  if s['baseline'] is None:s['baseline']=row
  base=s['baseline'];v=val['summary'];t=train['summary'];b=base['validation'];gap=v['geometry_free']['aa']-t['geometry_free']['aa'];bgap=b['geometry_free']['aa']-base['train']['geometry_free']['aa']
  conflict=val.get('objective_conflict',{}).get('shared_encoder_AA_vs_geometry_goal_cosine')
  conditions=dict(objective_conflict=conflict is not None and conflict<-.5 and v['geometry_free']['aa']>b['geometry_free']['aa']+.025,AA_regression=v['geometry_free']['aa']>b['geometry_free']['aa']+self.policy['AA_regression_nats'],native_AA_regression=v['native']['aa']>b['native']['aa']+self.policy['AA_regression_nats'],gap_growth=gap>bgap+self.policy['gap_growth_nats'] and v['geometry_free']['aa']>=b['geometry_free']['aa'],goal_regression=v['goal']['loss']>b['goal']['loss']+self.policy['goal_loss_regression'])
  for k in ['distance','angle','torsion']:
   if v['native'][k] is not None and b['native'][k] is not None:conditions[k+'_regression']=v['native'][k]>b['native'][k]+self.policy['component_regression_nats']
  reasons=[]
  for k,flag in conditions.items():
   s['strikes'][k]=s['strikes'].get(k,0)+1 if flag else 0
   if s['strikes'][k]>=self.policy['patience']:reasons.append(k+' at two successive scheduled evaluations')
  actor=[r for r in generation if not r.get('reference')]
  if actor and sum(r['completed'] for r in actor)/len(actor)<self.policy['completion_floor']:reasons.append('free-generation completion below 90%')
  if actor and sum(bool(r.get('valid')) for r in actor)/len(actor)==0 and update>=2500:reasons.append('no valid generated backbone at/after +2500')
  previous=[r for r in s['history'] if r['update']<=2500]
  if update>=self.policy['futility_after'] and previous:
   anchor=previous[-1]['validation'];AA_gain=anchor['geometry_free']['aa']-v['geometry_free']['aa'];goal_gain=anchor['goal']['loss']-v['goal']['loss']
   if AA_gain<.005 and goal_gain<.01:reasons.append('no useful held-out AA or goal progress since +2500')
  if not all(math.isfinite(v[c]['aa']) for c in ['native','geometry_free']):reasons.append('nonfinite evaluation')
  s['history'].append(row)
  return dict(stop=bool(reasons),reasons=reasons,gap=gap,baseline_gap=bgap,strikes=copy.deepcopy(s['strikes']))

def initialization_update(system,learner,prior,pop,sampler,geometry_sampler,goals,config):
 from .pipeline import native_goal_batch
 from .optimization import finish
 m=system.models['actor'];opt=learner.optimizers['actor'];opt.zero_grad(set_to_none=True);device=next(m.parameters()).device;totals=defaultdict(float);parts=[];B=config['effective_AA_batch'];counts=Counter();masks=Counter()
 def backward(stream,b,aa,g,mask,scale,with_goal=False):
  loss,terms=actor_loss(system,b,aa,g,mask,config['stream_weights'][stream],with_goal);weighted=loss*scale;weighted.backward();parts.append(float(weighted.detach()))
  for k,v in terms.items():totals[stream+'/'+k]+=float(v.detach())*scale
  for j,k in enumerate(['distance','angle','torsion']):masks[stream+'/'+k]+=int(mask[:,j].sum())
 records=sampler.draw(B)
 for start in range(0,B,config['microbatch']):
  subset=records[start:start+config['microbatch']];p,aa,g,mask,_=pop.batch(subset,device);b=prepare(p,prior(p),torch.zeros(len(aa),1,36,device=device),torch.zeros(len(aa),1,device=device,dtype=torch.bool));backward('AA',b,aa,g,mask,len(aa)/B);counts.update(r[0] for r in subset)
 for _ in range(config['complete_geometry_decisions']):
  p,aa,g,mask,_=pop.batch(geometry_sampler.geometry(1),device);b=prepare(p,prior(p),torch.zeros(1,1,36,device=device),torch.zeros(1,1,device=device,dtype=torch.bool));backward('complete_geometry',b,aa,g,mask,1/config['complete_geometry_decisions'])
 for _ in range(config['goal_demonstrations']):
  b,aa,g,mask=native_goal_batch(pop,goals,prior,device);backward('goal',b,aa,g,mask,config['goal_stream_scale']/config['goal_demonstrations'],True)
 grad={k:math.sqrt(sum(float(p.grad.float().square().sum()) for p in group.parameters() if p.grad is not None)) for k,group in [('encoder',m.encoder),('AA',m.aa),('geometry',m.geometry)]}
 if any(p.grad is not None for k,model in system.models.items() if k!='actor' for p in model.parameters()):raise ValueError('inactive-network gradient')
 norm=finish(opt,m.parameters(),config['gradient_clip']);learner.steps['actor']+=1;learner.schedulers['actor'].step()
 return dict(loss=sum(parts),terms=dict(totals),gradient_groups=grad,gradient_norm=norm,AA_decisions=B,AA_categories=dict(counts),valid_geometry_components=dict(masks),goal_kind_counts=dict(goals.kind_counts),goal_identities=len(goals.seen),goal_rejections=goals.rejected)

def validation_run(system,prior,pop,config,update,run,monitor):
 """Runs only from the separately authorized future training entrypoint."""
 vp=json.load(open(config['validation_panel']));tp=json.load(open(config['matched_TRAIN_panel']));device=next(system.parameters()).device
 with torch.autocast('cuda',dtype=torch.bfloat16):
  train=evaluate(system,prior,pop,tp,device,config);val=evaluate(system,prior,pop,vp,device,config)
  tier='core' if update<=250 else 'standard' if update<10000 else 'extended'
  gen=free_generation(system,prior,pop,vp,device,json.load(open(config['native_limits'])),tier)
  # Frozen J1 draws: calculate once per tier and reuse by exact panel/reference/protocol identity.
  cache=Path(run)/('reference_generation_'+tier+'.json')
  if cache.exists():ref=json.load(open(cache))
  else:
   ref=free_generation(system,prior,pop,vp,device,json.load(open(config['native_limits'])),tier,True)
   from .recovery import atomic_json
   atomic_json(cache,ref)
 gen+=ref
 with torch.autocast('cuda',dtype=torch.bfloat16):val['objective_conflict']=objective_conflict(system,prior,pop,tp,device,config)
 baseline_file=Path(run)/'validation_00000.json'
 if baseline_file.exists():val['paired_uncertainty']=paired_intervals(val,json.load(open(baseline_file))['VALIDATION'])
 stop=monitor.observe(update,train,val,gen)
 # Compact per-identity/cell metrics retain paired inference without large prefix tensors.
 result=dict(update=update,TRAIN=train,VALIDATION=val,generation=gen,stop=stop)
 from .recovery import atomic_json
 atomic_json(Path(run)/f'validation_{update:05d}.json',result)
 value=val['summary']['geometry_free']['aa']
 best=dict(metric='balanced_geometry_free_VALIDATION_AA_NLL_v1',value=value,mode='min',panel_sha256=file_sha(config['validation_panel']),identities=len(vp['records']),note='retention ranking only; task success and safety remain separate')
 return best,stop

def paired_intervals(current,baseline,seed=1731,repeats=2000):
 """Paired bootstrap identities within fixed length/population cells; not residue iid."""
 result={};rng=np.random.default_rng(seed)
 for condition in ['native','geometry_free']:
  def groups(rows):
   d=defaultdict(list)
   for r in rows:
    if r['condition']==condition:d[(r['length_band'],r['category'],r['identity'])].append(r['aa'])
   return {k:float(np.mean(v)) for k,v in d.items()}
  now=groups(current['records']);old=groups(baseline['records'])
  if now.keys()!=old.keys():raise ValueError('paired evaluation identities changed')
  cells=defaultdict(list)
  for (length,cat,h),v in now.items():cells[(length,cat)].append(v-old[(length,cat,h)])
  draws=np.zeros(repeats)
  for vals in cells.values():
   a=np.array(vals);draws+=a[rng.integers(0,len(a),(repeats,len(a)))].mean(-1)/len(cells)
  result[condition]=dict(delta_NLL=balanced_mean([r for r in current['records'] if r['condition']==condition],'aa')-balanced_mean([r for r in baseline['records'] if r['condition']==condition],'aa'),identity_bootstrap_95pct=np.quantile(draws,[.025,.975]).tolist(),replicates=repeats,scope='fixed selected identities, not evolutionary independence')
 return result

def objective_conflict(system,prior,pop,panel,device,config):
 """Fixed TRAIN diagnostic, no optimizer call or persistent gradient accumulation."""
 if panel['split']!='TRAIN':raise ValueError('gradient diagnostics TRAIN only')
 r=next(r for r in panel['tasks'] if r['task']['kind']=='P2' and r['length_band']=='64-96');record=next(v for v in panel['records'] if v['identity']==r['identity']);t=r['task']['targets'][0]['index'];p,aa,g,mask=load_decision(pop,record,t,device);ref=prior(p);gt,gv=goal_tokens(r['task'],device);parameters=list(system.models['actor'].encoder.parameters());all_grads=[]
 with torch.enable_grad():
  for goal in [False,True]:
   b=prepare(p,ref,gt if goal else torch.zeros_like(gt),gv if goal else torch.zeros_like(gv));w=dict(aa=0. if goal else 1.,geometry=.1 if goal else 0.,kl_aa=0.,kl_geometry=0.,goal=.05 if goal else 0.);loss,_=actor_loss(system,b,aa,g,mask,w,goal);grads=torch.autograd.grad(loss,parameters,allow_unused=True);all_grads.append([torch.zeros_like(v) if z is None else z for v,z in zip(parameters,grads)])
 dot=sum(float((a.float()*b.float()).sum()) for a,b in zip(*all_grads));norms=[math.sqrt(sum(float(z.float().square().sum()) for z in gr)) for gr in all_grads];cosine=dot/(norms[0]*norms[1]) if min(norms)>1e-12 else None
 return dict(shared_encoder_AA_vs_geometry_goal_cosine=cosine,gradient_norms=norms,TRAIN_identity=r['identity'],t=t,optimizer_updates=0)
