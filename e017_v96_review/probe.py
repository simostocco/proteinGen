"""Bounded TRAIN-only forward/backward probes. No optimizer instantiated or called."""
from common_review import *
import copy,math,runpy,subprocess,time
import numpy as np,torch
sys.path.insert(0,str(ENG/'source96'));import common96,algorithm

def flatgrad(actor):return torch.cat([p.grad.detach().flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in actor.parameters()])
def cosine(a,b):return float(torch.dot(a.double(),b.double())/(a.double().norm()*b.double().norm()).clamp_min(1e-30))
def norm(a):return float(a.double().norm())
def main():
 assert not subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 P,F,_=common96.load();e,_=P['K']['load_engine']('C');F['checkpoint'](e,5000);initial=copy.deepcopy(e.actor.state_dict());fp=P['c']['state_digest'](initial);e.actor.requires_grad_(True).eval();params=list(e.actor.parameters())
 contexts=[];sources=[]
 for i in [0,1]:
  obj=read(ARMS['A']/f'production/diagnostics/DEV_00000_filtered_{i:02d}.json');sources.append(str(ARMS['A']/f'production/diagnostics/DEV_00000_filtered_{i:02d}.json'))
  for ci in [0,1]:
   row=obj['records'][ci]
   for t in [8,24,40,56]:
    if len(row['geometry'])<t:continue
    state=P['S']['reconstruct'](row,64,t);w=P['S']['world'](obj['task']['goal'],state);prefix=P['p']['collate']([state],'cuda')
    with torch.no_grad():ref=e.prior(prefix);batch=P['p']['public_batch']([w],prefix,ref)
    contexts.append((batch,obj['task'],t,ci))
 def forward(batch,layers=False):
  traces=[];hooks=[]
  if layers:
   for block in e.actor.encoder.blocks:hooks.append(block.register_forward_hook(lambda m,a,y:traces.append(y.detach().clone())))
  with torch.no_grad():logits,mu,ls=e.actor(batch)
  for h in hooks:h.remove()
  return (logits,mu,ls),traces
 def comparison(a,b,task,t):
  aa=P['AA'].index(task['fixed_sequence'][t]);mask=torch.tensor(P['c']['mask_for_step'](t),device='cuda');am,al=a[1][0,aa].double(),a[2][0,aa].double();bm,bl=b[1][0,aa].double(),b[2][0,aa].double();kl=bl-al+.5*((al.exp()/bl.exp()).square()+((am-bm)/bl.exp()).square())-.5
  return dict(selected_active_KL=float((kl*mask).sum()),selected_mu_delta=(bm-am).tolist(),selected_logstd_delta=(bl-al).tolist(),allAA_mu_RMS=float((a[1]-b[1]).square().mean().sqrt()),AA_logits_max_delta=float((a[0]-b[0]).abs().max()))
 records=[]
 for b,task,t,ci in contexts:
  if (ROOT/'TOKEN_RECORDED_PREFIX_PROBES.json').exists():break
  base,layers=forward(b,True);aug=algorithm.augment(b);ins,altlayers=forward(aug,True);masked=dict(aug,goal_valid=aug['goal_valid'].clone());masked['goal_valid'][:,-1]=False;outmask,_=forward(masked)
  zero=dict(aug,goals=aug['goals'].clone());zero['goals'][:,-1]=0;outzero,_=forward(zero)
  role=dict(aug,goals=aug['goals'].clone());role['goals'][:,-1]=0;role['goals'][:,-1,0]=-1;role['goals'][:,-1,33]=1;role['goals'][:,-1,35]=1;outrole,_=forward(role)
  with torch.no_grad():
   enc=e.actor.encoder;ids=b['prefix']['ids'];ft,pair,present=b['features'];ref=b['reference'];progress=torch.stack([b['prefix']['N']/500,b['prefix']['t']/b['prefix']['N'],b['prefix']['valid'].sum(-1)/b['prefix']['t'].clamp_min(1),b['subgoal']/4],-1).float();q=torch.cat([ref['h'],ref['logits'],ref['mu'].flatten(1),ref['ls'].flatten(1),b['context'],progress],-1)
   x=torch.cat([enc.aa(ids)+enc.token(ft),enc.goal(aug['goals']),enc.query(q)[:,None]],1);block=enc.blocks[0];B,L,W=x.shape;H=block.heads;qkv=block.qkv(block.n1(x)).reshape(B,L,3,H,W//H);qq,kk=[qkv[:,:,i].transpose(1,2) for i in [0,1]];pos=torch.cat([torch.arange(ids.shape[1],device='cuda')[None],aug['goals'][:,:,32]*b['prefix']['N'][:,None],b['prefix']['t'][:,None]],1);rope=block.forward.__globals__['rope'];qq=rope(qq,pos);kk=rope(kk,pos);scores=qq@kk.transpose(-1,-2)/math.sqrt(W//H);valid=torch.cat([present,aug['goal_valid'],torch.ones(1,1,device='cuda',dtype=torch.bool)],1);att=scores.masked_fill(~valid[:,None,None,:],float('-inf')).softmax(-1)
  records.append(dict(goal_id=task['goal']['goal_id'],prefix=t,candidate=ci,token_effect=comparison(base,ins,task,t),masked_token_effect=comparison(base,outmask,task,t),visible_zero_token_effect=comparison(base,outzero,task,t),visible_role_only_effect=comparison(base,outrole,task,t),new_token_query_attention_first_layer=att[0,:,-1,-2].cpu().tolist(),projected_new_token_norm=float(enc.goal(aug['goals'])[:,-1].norm()),zero_token_projection_bias_norm=float(enc.goal.bias.norm()),layers=[dict(layer=j+1,prefix_RMS=float((x[:,:t]-y[:,:t]).square().mean().sqrt()),motif_RMS=float((x[:,t:-1]-y[:,t:-2]).square().mean().sqrt()),query_RMS=float((x[:,-1]-y[:,-1]).square().mean().sqrt())) for j,(x,y) in enumerate(zip(layers,altlayers))]))
  assert float((base[1]-outmask[1]).abs().max())<1e-5
 if records:write('TOKEN_RECORDED_PREFIX_PROBES.json',dict(TRAIN_only=True,contexts=len(contexts),sources=sources,records=records,new_trajectories=0,optimizer_updates=0,interpretation='new forward inference on recorded TRAIN states; masked-token and visible-zero ablations are not production implementations'))
 print('TOKEN_PROBES_COMPLETE',len(contexts),flush=True)
 # Existing checkpoint changes on a common TRAIN-state distribution.
 checkpoint_effects={};initial_vector=torch.cat([v.flatten() for v in initial.values()]);original_outputs=[forward(b)[0] for b,_,_,_ in contexts]
 for arm in ['A','B']:
  root=ARMS[arm];idx=read(root/'production/RECOVERY_INDEX.json')['latest'];state=torch.load(root/'production'/idx['relative'],map_location='cpu',weights_only=False);e.actor.load_state_dict(state['actor']);delta=torch.cat([(v-initial[k].cpu()).flatten() for k,v in state['actor'].items()]);theta=torch.cat([v.flatten() for v in state['actor'].values()]);updates=[];opt=state['optimizer'];group=opt['param_groups'][0];lr=group['lr']
  for i in group['params']:
   s=opt['state'][i];step=float(s['step']);
   if step==0:updates.append(torch.zeros_like(s['exp_avg']).flatten());continue
   m=s['exp_avg']/(1-.9**step);v=s['exp_avg_sq']/(1-.999**step);updates.append((-lr*m/(v.sqrt()+1e-8)).flatten())
  dlast=torch.cat(updates);outputs=[]
  for (b,task,t,ci),oldout in zip(contexts,original_outputs):
   used=algorithm.augment(b) if arm=='B' else b;nowout,_=forward(used);e.actor.load_state_dict(initial);ownout,_=forward(used);e.actor.load_state_dict(state['actor']);outputs.append(comparison(ownout,nowout,task,t))
  checkpoint_effects[arm]=dict(update=state['update'],cumulative_parameter_L2=norm(delta),relative_parameter_L2=norm(delta)/norm(initial_vector),last_applied_Adam_update_L2=norm(dlast),last_update_relative_to_parameters=norm(dlast)/norm(theta),last_lr=lr,common_TRAIN_prefix_changes_from_own_initialization=outputs)
  e.actor.load_state_dict(initial);del state
 write('CHECKPOINT_POLICY_CHANGE.json',checkpoint_effects)
 # Exact first on-policy trace backward: decompose unchanged reward, not new objectives or updates.
 gradients_report={}
 for arm in ['A','B']:
  e.actor.load_state_dict(initial);handle=algorithm.install(e.actor) if arm=='B' else None
  packet=read(ARMS[arm]/'production/rollouts/u000001.json');rows=packet['records'];task=packet['task'];n=len(rows);beta0=read(ARMS[arm]/'TRAINING_CONFIGURATION_AND_SEEDS.json')['trace_KL_beta']
  assert P['c']['state_digest'](e.actor.state_dict())==packet['behavior']
  rewards=np.array([r['metrics']['reward'] for r in rows]);components=np.array([[.5*int(r['metrics']['success'])]+[.1*int(r['metrics']['hard_valid'])*u for u in r['metrics'].get('utilities',[0]*5)] for r in rows]);assert np.allclose(components.sum(1),rewards)
  llr=np.array([r['trace_logprob']-r['reference_trace_logprob'] for r in rows]);adv=(n*components-components.sum(0))/(n-1);coef=np.concatenate([-adv,beta0*((n*llr-llr.sum())/(n-1)+1)[:,None]],1);names=['success','exposure','burial','contrast','motif','compatibility','KL']
  dim=sum(p.numel() for p in params);vecs=[torch.zeros(dim,device='cuda') for _ in names];stages=[torch.zeros(dim,device='cuda') for _ in range(4)];worlds=[P['S']['world'](task['goal']) for _ in rows];maxerr=0.
  for t in range(64):
   ids=[i for i,r in enumerate(rows) if any(q[0]==t for q in r['proposals'])]
   if not ids:continue
   ws=[worlds[i] for i in ids];prefix=P['p']['collate']([w.state for w in ws],'cuda')
   with torch.no_grad():ref=e.prior(prefix);batch=P['p']['public_batch'](ws,prefix,ref)
   _,mu,ls=e.actor(batch);terms=[]
   for j,i in enumerate(ids):
    aa=P['AA'].index(task['fixed_sequence'][t]);terms.append(torch.stack([P['K']['S']['T']['base_logprob'](mu[j,aa],ls[j,aa],torch.tensor(q[2],device='cuda',dtype=torch.float64),P['c']['mask_for_step'](t)) for q in rows[i]['proposals'] if q[0]==t]).sum())
    recorded=sum(q[4] for q in rows[i]['proposals'] if q[0]==t);maxerr=max(maxerr,abs(float(terms[-1].detach())-recorded))
   terms=torch.stack(terms)
   for k in range(7):
    loss=(terms*torch.tensor(coef[ids,k],device='cuda')).sum()/n
    gs=torch.autograd.grad(loss,params,retain_graph=k<6,allow_unused=True);v=torch.cat([(g.flatten() if g is not None else torch.zeros_like(p).flatten()) for g,p in zip(gs,params)]);vecs[k]+=v;stages[t//16]+=v
   for i in ids:
    if t<len(rows[i]['geometry']):worlds[i].state=P['S']['append'](worlds[i].state,task['fixed_sequence'][t],rows[i]['geometry'][t])
  total=sum(vecs);record=read(ARMS[arm]/'production/commits/u000001.json')['logical'];assert abs(norm(total)-record['gradient_norm'])<2e-4 and maxerr<5e-5
  clipped=total*min(1,5/(norm(total)+1e-6));lr=read(ARMS[arm]/'TRAINING_CONFIGURATION_AND_SEEDS.json')['peak_lr']/100
  analytic_step=-lr*clipped/(clipped.abs()+1e-8)
  # Directional first-order action response along the actual first fresh-Adam formula; do not apply it.
  b,task0,t0,ci=contexts[0];used=algorithm.augment(b) if arm=='B' else b;_,mu,ls=e.actor(used);aa=P['AA'].index(task0['fixed_sequence'][t0]);derivatives=[]
  for j,z in enumerate(torch.cat([mu[0,aa],ls[0,aa]])):
   gs=torch.autograd.grad(z,params,retain_graph=j<5,allow_unused=True);v=torch.cat([(g.flatten() if g is not None else torch.zeros_like(p).flatten()) for g,p in zip(gs,params)]);derivatives.append(float(torch.dot(v.double(),analytic_step.double())))
  gradients_report[arm]=dict(first_group_only=True,exact_recorded_norm=record['gradient_norm'],reconstructed_norm=norm(total),max_trace_logprob_error=maxerr,component_norms={k:norm(v) for k,v in zip(names,vecs)},component_cosines={a+'__'+b:cosine(vecs[i],vecs[j]) for i,a in enumerate(names) for j,b in enumerate(names) if j>i},stage_norms=[norm(v) for v in stages],stage_cosine_with_total=[cosine(v,total) for v in stages],clip_factor=norm(clipped)/norm(total),clipping_direction_cosine=cosine(total,clipped),analytic_first_Adam_update_L2=norm(analytic_step),negative_gradient_vs_Adam_cosine=cosine(-total,analytic_step),first_order_mu_logstd_change=derivatives,AA_parameter_score_gradient_norm=norm(total[sum(p.numel() for n,p in e.actor.named_parameters() if not n.startswith('aa.')):]) if False else 'AA heads unused by geometry loss; inherited exact score tests and zero AA optimization terms',optimizer_calls=0)
  if handle:handle.remove();delattr(e.actor,'_v96_burial_token_installed')
  del vecs,total,clipped,analytic_step,stages;torch.cuda.empty_cache();print('GRADIENT_COMPONENTS_COMPLETE',arm,flush=True)
 e.actor.load_state_dict(initial);e.actor.zero_grad(set_to_none=True);assert P['c']['state_digest'](e.actor.state_dict())==fp;e.prior.verify();assert P['c']['state_digest'](e.parent.state_dict())==e.fp
 write('TRAIN_GRADIENT_COMPONENT_PROBES.json',gradients_report)
 write('ENGINEERING_PROBE_COMPLETED.json',dict(PASS=True,UTC=utc(),TRAIN_contexts=len(contexts),on_policy_groups=2,optimizer_calls=0,new_generation_attempts=0,J1_unchanged=True,actor_restored_in_memory=True,no_checkpoint_written=True,peak_GPU_bytes=torch.cuda.max_memory_allocated()))
 print('READ_ONLY_MODEL_PROBES_PASS',flush=True)
if __name__=='__main__':main()
