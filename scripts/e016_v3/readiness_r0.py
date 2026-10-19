"""R0 bounded GPU engineering tests, immutable J1 initialization and R1 panel."""
import copy,hashlib,json,math,os,random,shutil,subprocess,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from protein_joint_sac.contracts import AA,BuildState,BuildAction,GeneratedProtein,digest,OracleProtocol,OracleResult,OracleStatus
from protein_joint_sac.geometry.ca_builder import append,replay,mask_for_step
from protein_joint_sac.models.dense_geometry_attention import ModelConfig
from protein_joint_sac.models.joint_actor import JointActor
from protein_joint_sac.models.build_critic import TwinCritics
from protein_joint_sac.models.batched_bc import collate,encode_batch,features_batch,losses
from protein_joint_sac.models.hybrid_distributions import GeometryDistribution
from protein_joint_sac.rl.sequence_rl import rollout,regularized_value,reference_kl,rsample_physical,double_distribution,bellman_target,discounted_nstep
from protein_joint_sac.rl.sequence_sac import backward_sequence_first
from protein_joint_sac.rl.sequence_replay import SequenceReplay,OracleWallBudget
from protein_joint_sac.oracle.queue import EvaluationQueue
from protein_joint_sac.rl.checkpointing import save_atomic,load_checked
from protein_joint_sac.rl.bc_training import rng_state,restore_rng
from protein_joint_sac.data.aa_only_supervised import ExpandedData,ExpandedSampler,augment_available
from protein_joint_sac.data.certified_unified import file_sha
from protein_joint_sac.rl.sequence_first import Coverage
from protein_joint_sac.models.geometry_inference import corrected_action
R=Path(__file__).resolve().parents[3];D=R.parent;PREV=D/'e016_j1_bidirectional_multitask_20261019_v1';SOURCE=PREV/'runs/J1/checkpoint_07500.pt';SHA='881457ff0396975c1b20dd0a8e2caac8521e65b7e1cec76a7bb531dbc437b412';FALLBACK=D/'e016_j1_torsion_stabilization_20261018_v1/replay/checkpoint_00500_full.pt'
def write(name,value):
 p=R/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.part');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(tmp,p)
def guard():
 free=shutil.disk_usage(R).free
 if free<10*2**30+600*2**20:raise RuntimeError('10GiB reserve plus checkpoint staging gate')
 return free
def actor():return JointActor(ModelConfig(width=352,heads=8,layers=8,ffn_mult=4,chemistry=False,backend='sdpa')).cuda().eval()
def state_fixture(t=8,N=64):
 actions=[]
 for i in range(t):
  mask=mask_for_step(i);g=torch.tensor([3.8 if mask[0] else 0,1.9 if mask[1] else 0,.7*math.sin(i)+.2 if mask[2] else 0],dtype=torch.float64);actions.append(BuildAction(AA[i%20],g,mask))
 return replay(N,actions)
def tensor_same(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(tensor_same(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(tensor_same(x,y) for x,y in zip(a,b))
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 return a==b

def hybrid_bandit():
 """Exact-representable twin critics; genuine discrete + continuous SAC learning."""
 device='cuda';dtype=torch.float64;u_star=torch.tensor([math.log(3.8),math.log(1.9/(math.pi-1.9)),math.atanh(.7/math.pi)],device=device,dtype=dtype)
 class Bandit(nn.Module):
  def __init__(self):
   super().__init__();self.logits=nn.Parameter(torch.zeros(20,device=device,dtype=dtype));self.mu=nn.Parameter(u_star[None].repeat(20,1)+torch.tensor([.5,-.5,.5],device=device));self.ls=nn.Parameter(torch.full((20,3),-1.5,device=device,dtype=dtype))
  def draw(self,K=8):
   u=self.mu[:,None]+self.ls.clamp(-5,0).exp()[:,None]*torch.randn(20,K,3,device=device,dtype=dtype);g=torch.stack([u[...,0].exp(),math.pi*u[...,1].sigmoid(),math.pi*u[...,2].tanh()],-1);v=torch.stack([g[...,0].log(),torch.logit(g[...,1]/math.pi),torch.atanh(g[...,2]/math.pi)],-1);return u,v
 a=Bandit();q1=nn.Linear(21,1,dtype=dtype,device=device);q2=nn.Linear(21,1,dtype=dtype,device=device);t1=copy.deepcopy(q1);t2=copy.deepcopy(q2);optQ=torch.optim.Adam(list(q1.parameters())+list(q2.parameters()),lr=.04);optA=torch.optim.Adam(a.parameters(),lr=.03)
 util=torch.linspace(-.2,.8,20,device=device,dtype=dtype);util[-1]=1.4
 def feature(ids,u):return torch.cat([F.one_hot(ids,20).to(dtype),((u-u_star)**2).mean(-1,keepdim=True)],-1)
 x_id=torch.arange(20,device=device).repeat(128);u=u_star+torch.randn(len(x_id),3,device=device,dtype=dtype);x=feature(x_id,u);y=util[x_id]-.5*((u-u_star)**2).mean(-1)
 for k in range(500):
  idx=torch.randint(len(x),(128,),device=device);loss=(q1(x[idx]).squeeze()-y[idx]).square().mean()+(q2(x[idx]).squeeze()-y[idx]).square().mean();optQ.zero_grad();loss.backward();optQ.step()
  with torch.no_grad():
   for online,target in [(q1,t1),(q2,t2)]:
    for p,z in zip(online.parameters(),target.parameters()):z.lerp_(p,.01)
 with torch.no_grad():rmse=[float((q(x).squeeze()-y).square().mean().sqrt()) for q in [q1,q2]]
 for q in [q1,q2]:q.requires_grad_(False)
 def reward_estimate():return float((a.logits.softmax(-1)*(util-.5*((a.mu-u_star)**2+a.ls.exp()**2).mean(-1))).sum().detach())
 before=reward_estimate()
 def step():
  u,v=a.draw();ids=torch.arange(20,device=device)[:,None].expand(u.shape[:2]);f=feature(ids,v);q=torch.minimum(q1(f),q2(f)).squeeze(-1);logbase=-.5*((u-a.mu[:,None])/a.ls.clamp(-5,0).exp()[:,None])**2-a.ls.clamp(-5,0)[:,None]-.5*math.log(2*math.pi);jac=torch.stack([u[...,0],math.log(math.pi)-F.softplus(-u[...,1])-F.softplus(u[...,1]),math.log(math.pi)+2*(math.log(2)-u[...,2]-F.softplus(-2*u[...,2]))],-1);lg=(logbase-jac).sum(-1);lp=a.logits.log_softmax(-1);loss=(lp.exp()*(.02*lp+(.001*lg-q).mean(1))).sum();optA.zero_grad();loss.backward();optA.step();return float(loss.detach())
 for _ in range(150):step()
 snap={'actor':copy.deepcopy(a.state_dict()),'optimizer':copy.deepcopy(optA.state_dict()),'q1':q1.state_dict(),'q2':q2.state_dict(),'target1':t1.state_dict(),'target2':t2.state_dict(),'RNG':rng_state()};save_atomic(R/'tests/toy_restart.pt',snap)
 v1=[step() for _ in range(3)];end=copy.deepcopy(a.state_dict());opt_end=copy.deepcopy(optA.state_dict());rng_end=rng_state();s=load_checked(R/'tests/toy_restart.pt');a.load_state_dict(s['actor']);optA.load_state_dict(s['optimizer']);restore_rng(s['RNG']);v2=[step() for _ in range(3)];exact=tensor_same(end,a.state_dict()) and tensor_same(opt_end,optA.state_dict()) and tensor_same(rng_end,rng_state()) and v1==v2
 after=reward_estimate();best=float(a.logits.softmax(-1)[-1]);err=float(((a.mu-u_star)**2).mean().sqrt().detach());assert max(rmse)<.02 and after-before>.5 and best>.8 and exact
 # Expected categorical gradient equals the score-function estimator mean.
 logits=torch.randn(20,device=device,dtype=dtype,requires_grad=True);values=torch.linspace(-1,1,20,device=device,dtype=dtype);lp=logits.log_softmax(-1);p=lp.exp();exactg=torch.autograd.grad((p*values).sum(),logits)[0];draw=torch.multinomial(p.detach(),200000,replacement=True);one=F.one_hot(draw,20).to(dtype);mc=((one-p.detach())*values[draw,None]).mean(0);maxdiff=float((mc-exactg).abs().max());assert maxdiff<.003
 assert bellman_target(1.,True,999.)==1 and bellman_target(0.,False,1.)==1
 returns=[discounted_nstep([0.]*(n-1)+[1.]) for n in [20,64,128,500]];assert returns==[1.]*4
 write('HYBRID_SAC_TOY_RESULTS.json',{'status':'PASS','scope':'synthetic reference reward only; no protein-quality improvement','bandit_twin_Q_RMSE':rmse,'policy_expected_return_start':before,'policy_expected_return_end':after,'best_AA_probability':best,'latent_mean_error_RMS':err,'discrete_exact_vs_score_function_max_gradient_error':maxdiff,'score_function_samples':200000,'no_integer_backprop':True,'gamma':1.,'terminal_no_bootstrap':True,'length20_64_128_500_MC_returns':returns,'nstep_terminal_equals_MC':discounted_nstep([0.,0.,1.])==1.,'GPU_restart_bitwise_exact':exact,'toy_policy_updates':153,'toy_critic_updates':500,'J1_optimizer_updates':0,'raw_reward_vs_entropy_units_separate':True})

def fit_transformer_critics(state):
 critics=TwinCritics(ModelConfig(width=32,heads=4,layers=1,ffn_mult=2,chemistry=False,backend='sdpa')).cuda();base=state.to('cpu');acts=[];ys=[];rng=np.random.default_rng(19019)
 for i in range(128):
  u=np.array([math.log(3.8),math.log(1.9/(math.pi-1.9)),math.atanh(.7/math.pi)])+rng.normal(0,.15,3);g=np.array([np.exp(u[0]),math.pi/(1+np.exp(-u[1])),math.pi*np.tanh(u[2])]);acts.append(BuildAction(AA[i%20],torch.tensor(g),mask_for_step(base.t)));ys.append(.5*(i%20)/19+.5*np.exp(-np.square(u-np.array([math.log(3.8),math.log(1.9/(math.pi-1.9)),math.atanh(.7/math.pi)])).sum()/.08))
 candidate=collate([append(base,a) for a in acts],'cuda');prepared=features_batch(candidate,False);y=torch.tensor(ys,device='cuda',dtype=torch.float32);train=torch.arange(96,device='cuda');held=torch.arange(96,128,device='cuda');opt=torch.optim.AdamW(list(critics.q1.parameters())+list(critics.q2.parameters()),lr=.002)
 def pred(q,idx):
  b={k:v[idx] for k,v in candidate.items()};feat=tuple(None if v is None else (tuple(w[idx] for w in v) if isinstance(v,tuple) else v[idx]) for v in prepared);h=q.current(state);z=encode_batch(SimpleNamespace(encoder=q.candidate),b,prepared=feat);return q.head(torch.cat([h.expand(len(idx),-1),z],-1)).squeeze(-1)
 initial=[]
 with torch.no_grad():initial=[float((pred(q,held)-y[held]).square().mean().sqrt()) for q in [critics.q1,critics.q2]]
 started=time.monotonic();losses=[]
 for k in range(400):
  idx=train[torch.randperm(96,device='cuda')[:32]];v=sum((pred(q,idx)-y[idx]).square().mean() for q in [critics.q1,critics.q2]);opt.zero_grad();v.backward();torch.nn.utils.clip_grad_norm_(list(critics.q1.parameters())+list(critics.q2.parameters()),5.,error_if_nonfinite=True);opt.step();critics.polyak(.01)
  if k%100==0:print(json.dumps({'phase':'TRANSFORMER_CRITIC_TOY','update':k,'loss':float(v.detach())}),flush=True)
 with torch.no_grad():rmse=[float((pred(q,held)-y[held]).square().mean().sqrt()) for q in [critics.q1,critics.q2]]
 result={'status':'PASS' if max(rmse)<.10 else 'NO_GO','independent_candidate_transformers':True,'parameters_per_online_Q':sum(p.numel() for p in critics.q1.parameters()),'training_actions':96,'heldout_actions':32,'heldout_RMSE_start':initial,'heldout_RMSE_final':rmse,'updates':400,'seconds':time.monotonic()-started,'reward':'known synthetic AA-plus-continuous-action response on a generated-prefix fixture','real_protein_return_calibration':'NOT_ESTABLISHED'};write('TRANSFORMER_CRITIC_DIAGNOSTIC.json',result);return critics

def real_gpu_checks(model,reference,source):
 state,trace,states=rollout(model,64,2026101901,SHA);protein=GeneratedProtein(state);store=SequenceReplay(R/'replay/generated.sqlite');store.add_trace('J1_initial64',64,trace,GeneratedProtein(state.to('cpu')),{'policy_sha256':SHA,'scope':'R0_UNOPTIMIZED_REFERENCE','reward':None});reconstructed=store.reconstruct('J1_initial64');cpu_replay_error=float((reconstructed.ca-state.ca.cpu()).abs().max());assert cpu_replay_error<1e-8
 gpu_reconstructed=replay(64,store.episode('J1_initial64').actions,device='cuda');assert torch.equal(gpu_reconstructed.ca,state.ca)
 # Fixed Gaussian noise + categorical uniforms reproduce every prefix/action exactly.
 state2,trace2,_=rollout(model,64,2026101901,SHA);assert torch.equal(state.ca,state2.ca) and trace==trace2
 diffs=[]
 for t in [0,1,2,3,16,63]:
  s=states[t];v=trace[t]
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):h=model.encode(s);lp=model.aa_head(h).float().log_softmax(-1);pd=double_distribution(model,s,v['residue'],h)
  d=abs(float(pd.log_prob(torch.tensor(v['geometry'],device='cuda',dtype=torch.float64)))-v['geometry_represented_log_density']);diffs.append(d);assert d<1e-7 and abs(float(lp[AA.index(v['residue'])])-v['AA_logp'])<1e-7
 s=states[8];critics=fit_transformer_critics(s);bench={}
 for p in critics.parameters():p.requires_grad_(False)
 for est in ['exact','score_function']:
  model.zero_grad(set_to_none=True);torch.cuda.synchronize();st=time.monotonic();value,_=regularized_value(model,critics,s,reference,estimator=est);(-value).backward();torch.cuda.synchronize();grad=math.sqrt(sum(float(p.grad.double().square().sum()) for p in model.parameters() if p.grad is not None));assert math.isfinite(grad) and grad>0;bench[est]={'seconds':time.monotonic()-st,'gradient_L2':grad,'semantics':'exact categorical expectation, one reparameterized G draw/category' if est=='exact' else 'unbiased categorical score term plus direct conditional gradient; one sampled AA'}
 p=double_distribution(model,s,'A');g,lp,u=rsample_physical(p);g.retain_grad();value=critics.q1(s,BuildAction('A',g,mask_for_step(s.t)));model.zero_grad();value.backward();assert g.grad is not None and torch.isfinite(g.grad).all() and g.grad.abs().sum()>0
 geompath=float(g.grad.norm());kA,kG=reference_kl(model,reference,s);assert abs(float(kA))<1e-9 and abs(float(kG))<1e-9
 # All-missing geometry is a legitimate sequence-only state; no phantom atoms.
 absent=BuildState(s.residues,torch.zeros_like(s.ca),s.N,torch.zeros(s.t,device='cuda',dtype=torch.bool));changed=BuildState(s.residues,torch.full_like(s.ca,1e9),s.N,absent.valid)
 with torch.no_grad():a=model.logits(absent);b=model.logits(changed)
 assert torch.equal(a,b) and torch.isfinite(a).all()
 extreme=[]
 for u in [-1e6,-100.,-10.,0.,10.,100.,1e6]:
  mu=torch.tensor([math.log(3.8),0.,u],device='cuda');g,info=corrected_action(mu,torch.zeros(3,device='cuda'),(True,True,True),torch.zeros(3,device='cuda'));assert -math.pi<float(g[2])<math.pi;extreme.append({'latent_tau':u,'support_repaired':any(info['adjusted_components']),'continuous_SAC_sample_eligible':not any(info['adjusted_components'])})
 # Actor/reference remain equal: all J1 calculations above are forward/backward only.
 assert all(torch.equal(v,source['model'][n].to(v)) for n,v in model.state_dict().items())
 write('RL_ACTION_LOGPROB_TESTS.json',{'status':'PASS','production_parameter_count':sum(p.numel() for p in model.parameters()),'AA_logp_recomputation_exact':True,'geometry_physical_logdensity_max_error':max(diffs),'valid_components_bootstrap':'t>=1,2,3','same_GPU_trajectory_reconstruction_bitwise':True,'cross_CPU_reconstruction_max_error_A':cpu_replay_error,'rollout_common_draws_bitwise':True,'prefix_states_literal_no_future_fields':True,'same_policy_AA_KL':float(kA.detach()),'same_policy_geometry_KL':float(kG.detach()),'continuous_gradient_through_builder_and_Q_norm':geompath,'categorical_estimators':bench,'extreme_torsions':extreme,'finite_grid_repairs_not_claimed_continuous_density_exact':True,'all_missing_placeholders_invariant':True,'J1_optimizer_updates':0,'J1_weights_unchanged':True})
 write('generated/J1_initial64.json',{'sequence':protein.sequence,'ca':state.ca.cpu().tolist(),'trace':trace,'policy_sha256':SHA,'seed':2026101901});store.close();return state

def pdb_integration(model,data,source):
 sampler=ExpandedSampler(data,{'coverage':Coverage(20261019).state_dict(),'cursor':0});aa=[];ge=[]
 for _ in range(5):a,g=sampler.draw();aa+=a;ge+=g
 assert [sampler.counts[k] for k in ['complete','partial_only','unused']]==[128,96,96]
 snap=sampler.state_dict();expected=sampler.draw();sampler.load_state_dict(snap);assert expected==sampler.draw();sampler.load_state_dict(snap)
 model.train()
 for b in model.encoder.blocks:b.dropout_p=.15
 model.zero_grad(set_to_none=True);lossvalue=0.
 # One exact effective 64-AA mixed batch; rest of five-update cycle is sampler-only.
 examples=aa[:64]
 for off in range(0,64,16):
  b,y,g=data.batch(examples[off:off+16]);b=augment_available(b,.075)
  with torch.autocast('cuda',dtype=torch.bfloat16):v=F.cross_entropy(model.aa_head(encode_batch(model,b)).float(),y,reduction='sum')/64
  v.backward();lossvalue+=float(v.detach())
 aaNorm=math.sqrt(sum(float(p.grad.double().square().sum()) for n,p in model.named_parameters() if p.grad is not None and n.startswith('encoder.')));model.zero_grad(set_to_none=True);components=[]
 for off in range(0,32,16):
  b,y,g=data.batch(ge[off:off+16]);b=augment_available(b,.075)
  with torch.autocast('cuda',dtype=torch.bfloat16):_,geom,parts=losses(model,b,y,g,geometry_fp32=True)
  (geom.sum()/32).backward();components+=parts.detach().cpu().tolist()
 gn=math.sqrt(sum(float(p.grad.double().square().sum()) for n,p in model.named_parameters() if p.grad is not None and n.startswith('encoder.')));assert math.isfinite(aaNorm+gn) and aaNorm>0 and gn>0
 assert all(torch.equal(v,source['model'][n].to(v)) for n,v in model.state_dict().items());model.zero_grad(set_to_none=True);model.eval()
 write('PDB_BC_GPU_INTEGRATION.json',{'status':'PASS','TRAIN_populations':{k:len(v) for k,v in data.populations['TRAIN'].items()},'five_update_AA_quota_counts':{k:sampler.counts[k] for k in ['complete','partial_only','unused']},'sampler_resume_exact':True,'real_AA_forward_backward_targets':64,'qualified_complete_geometry_forward_backward_targets':32,'AA_shared_gradient_L2':aaNorm,'geometry_shared_gradient_L2_unweighted':gn,'AA_cross_entropy':lossvalue,'geometry_components_mean_NLL':np.mean(components,0).tolist(),'separate_losses':True,'unused_coordinates_loaded_for_AA_only':False,'actual_J1_optimizer_steps':0,'weights_unchanged':True,'normalization_changed':False});save_atomic(R/'actor/PDB_sampler_initial.pt',snap)

def make_panel(data):
 from protein_joint_sac.data.accepted_supervised import CoordinateReader
 reader=CoordinateReader(data.proposal.external);rows=[];used=set();history=[];old=json.load(open(D/'e016_v3_blocker_resolution_20261014_v1/NATIVE_CONTROL_PANEL.json'))
 # Historical cases are selectively re-mapped from their two preserved sources.
 import gemmi
 for hrow in old['controls']:
  if hrow['id'].split('_')[0] not in ['3fm1','2qmi']:continue
  seq=hrow['full_deposited_canonical_sequence'];h=hashlib.sha256(seq.encode()).hexdigest();owner=data.proposal.sequences.get(h)
  if not owner or owner['canonical_full_sequence']!=seq or owner['split']!='TRAIN':history.append({'id':hrow['id'],'included':False,'reason':'exact sequence not certified TRAIN','split':owner['split'] if owner else None});continue
  assert file_sha(hrow['source'])==hrow['source_sha256'];assert file_sha(hrow['authoritative_audit'])==hrow['audit_sha256'];aud=json.load(open(hrow['authoritative_audit']))['controls'][hrow['id']];assert aud['deposited_sequence']==seq and aud['source_sha256']==hrow['source_sha256'];auth=hrow['id'].rsplit('_',1)[1];assert auth in ','.join(aud['mmcif_metadata']['_entity_poly.pdbx_strand_id']).split(',');structure=gemmi.read_structure(hrow['source']);x=[None]*len(seq);observed=[]
  for chain in structure[0]:
   if chain.name!=auth:continue
   for residue in chain:
    if residue.label_seq is None:continue
    atoms=[a for a in residue if a.name=='CA' and a.altloc in ['\x00',' ','A']]
    if len(atoms)>1:raise ValueError('ambiguous coherent blank/A conformer')
    if atoms:
     i=residue.label_seq-1;assert x[i] is None;x[i]=list(atoms[0].pos);observed.append(i+1)
  assert observed==aud['residue_mapping'];rows.append({'id':'train_historical_'+hrow['id'],'kind':'native','split':'TRAIN','sequence_id':h,'sequence':seq,'N':len(seq),'native_ca':x,'observed_positions':observed,'source_path':hrow['source'],'source_sha256':hrow['source_sha256'],'source_model':'first deposited model','conformer_policy':'blank plus coherent A only','historical_case':True,'seed':151502});used.add(h);history.append({'id':hrow['id'],'included':True,'sequence_id':h})
 train_targets=[64,128,256,384,96,448]
 for role,targets in [('TRAIN',train_targets),('VALIDATION',[96,256])]:
  needed=(6-len(rows)) if role=='TRAIN' else 2
  for L in targets[:needed]:
   h=min((h for h in data.base.jointids[role] if h not in used),key=lambda h:(abs(len(data.seqs[h])-L),h));o=data.base.owners[h][0];seq=data.seqs[h]
   if o not in data.base.ca:
    from protein_joint_sac.data.accepted_supervised import CoordinateReader
    from protein_joint_sac.data.scaled_supervised import canonical
    data.base.ca[o]=canonical(reader.read(data.base.rows[o]))
   x=data.base.ca[o];r=data.base.rows[o];rows.append({'id':role.lower()+'_'+str(L),'kind':'native','split':role,'sequence_id':h,'sequence':seq,'N':len(seq),'native_ca':x.tolist(),'observed_positions':list(range(1,len(seq)+1)),'observation_id':o,'source_sha256':r['source_sha'],'entry_id':r['entry_id'],'historical_case':False,'seed':151502});used.add(h)
 base=min((v for v in rows if v['split']=='TRAIN' and v['kind']=='native'),key=lambda v:abs(v['N']-128));seq=base['sequence'];rs=random.Random(20261019);shuffled=list(seq);rs.shuffle(shuffled)
 for label,sq in [('composition_shuffle',''.join(shuffled)),('repetitive_helix_probe',('AL'*((len(seq)+1)//2))[:len(seq)])]:rows.append({'id':label,'kind':'variant','split':'TRAIN_DIAGNOSTIC_DERIVATIVE','sequence':sq,'N':len(sq),'native_ca':None,'parent_sequence_id':base['sequence_id'],'seed':151502,'negative_control_status':'not experimentally demonstrated nonfolding'})
 repeat=copy.deepcopy(min((v for v in rows if v['kind']=='native' and v['split']=='TRAIN'),key=lambda v:v['N']));repeat['id']+='_repeat_seed';repeat['seed']=151503;repeat['kind']='repeat';rows.append(repeat)
 write('configs/NATIVE_CONTROL_PANEL.json',{'controls':rows,'historical_ownership':history,'selected_before_AF_calls':True,'TRAIN_native_controls':6,'VALIDATION_native_audit':2,'sequence_variants':2,'repeat_attempts':1,'additional_generated_reference_and_fallback':2,'planned_total_attempts':13,'hard_cap_attempts':16,'hard_aggregate_wall_charged_seconds':2700,'per_call_timeout_seconds':240,'per_call_reservation_seconds':360,'GO_rules':{'TRAIN_median_robust_fixed_map_TM':.6,'TRAIN_fraction_TM_atleast_.5':2/3,'TRAIN_pTM_accuracy_rank_correlation':.4,'VALIDATION_audit_median_TM':.5,'required_global_metrics':['pTM','PAE'],'variant_native_confidence_separation':True},'no_TEST':True,'no_structure_eligibility_modified':True})

def main():
 guard();start=time.monotonic();assert file_sha(SOURCE)==SHA;torch.set_num_threads(6);torch.manual_seed(20261019);np.random.seed(20261019);random.seed(20261019);torch.cuda.manual_seed_all(20261019);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 source=load_checked(SOURCE);assert source['new_update']==7500 and source.get('total_optimizer_updates',35500+source['new_update'])==43000
 model=actor();model.load_state_dict(source['model'],strict=True);reference=actor();reference.load_state_dict(source['model'],strict=True);reference.requires_grad_(False)
 for path in [R/'reference/frozen_J1.pt',R/'actor/initial_J1.pt']:
  if path.exists():assert tensor_same(load_checked(path)['model'],source['model'])
  else:save_atomic(path,{'model':source['model'],'source_checkpoint':str(SOURCE),'source_sha256':SHA,'historical_optimizer_updates':43000,'production_parameters':12351746,'auxiliary_masked_head_required_for_inference':False})
 os.chmod(R/'reference/frozen_J1.pt',0o444)
 write('INITIALIZER_VERIFICATION.json',{'status':'PASS','user_selected_new_RL_initializer':True,'source_path':str(SOURCE),'source_sha256':SHA,'historical_updates':43000,'new_RL_updates':0,'production_parameters':12351746,'immutable_reference_copy_sha256':file_sha(R/'reference/frozen_J1.pt'),'separate_actor_initial_sha256':file_sha(R/'actor/initial_J1.pt'),'historical_supervised_qualification_not_rewritten':True})
 hybrid_bandit();real_gpu_checks(model,reference,source)
 data=ExpandedData();pdb_integration(model,data,source);make_panel(data)
 # A frozen fallback episode shares the same base-noise stream, not model weights.
 fallback=load_checked(FALLBACK);assert file_sha(FALLBACK)=='9208263300fb9e88519ff0d40e6d9d29a9744b709d9a6181819cd0894b593000';model.load_state_dict(fallback['model']);state,trace,_=rollout(model,64,2026101901,file_sha(FALLBACK));write('generated/fallback_initial64.json',{'sequence':''.join(state.residues),'ca':state.ca.cpu().tolist(),'trace':trace,'policy_sha256':file_sha(FALLBACK),'seed':2026101901});model.load_state_dict(source['model'])
 write('R0_GPU_RESOURCE.json',{'seconds':time.monotonic()-start,'Torch_allocated_peak':torch.cuda.max_memory_allocated(),'Torch_reserved_peak':torch.cuda.max_memory_reserved(),'D_free_bytes':guard(),'J1_optimizer_updates':0,'main_actor_unchanged':tensor_same({k:v.cpu() for k,v in model.state_dict().items()},source['model']),'GPU_name':torch.cuda.get_device_name(),'PDB_split_manifest_sha256':file_sha(data.base.proposal.root/'PROPOSAL_MANIFEST.json') if hasattr(data.base.proposal,'root') else None})
 print(json.dumps({'phase':'R0_COMPLETE','J1_optimizer_updates':0,'D_free_GiB':guard()/2**30}),flush=True)
if __name__=='__main__':main()
