"""Finish checks on the finalized protocol without any optimizer step or study launch."""
from pathlib import Path
import ast,json,subprocess,torch,copy,os,time
from protein_joint_sac.crisp_v3 import admission as a
from protein_joint_sac.crisp_v3.data import Population
from protein_joint_sac.crisp_v3.networks import System
from protein_joint_sac.crisp_v3.prior import FrozenPrior,tensor_sha,file_sha,MAIN,SHA
from protein_joint_sac.crisp_v3.training import seal_config,authorization
R=Path(__file__).resolve().parents[3];config=json.load(open(R/'configs/SEALED_RESEARCH_INITIALIZATION.json'));seal=seal_config(config);pop=Population();vp=json.load(open(config['validation_panel']));tp=json.load(open(config['matched_TRAIN_panel']));assert file_sha(MAIN)==SHA
for f in [R/'source/src/protein_joint_sac/crisp_v3/admission.py',R/'source/src/protein_joint_sac/crisp_v3/training.py']:ast.parse(f.read_text())
prior=FrozenPrior('cuda');s=System('research').cuda();state=tensor_sha(s);out={}
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
 small=copy.deepcopy(vp);small['records']=[next(r for r in vp['records'] if r['category']==cat and r['length_band']=='64-96') for cat in a.CATEGORIES];ids={r['identity'] for r in small['records']};small['tasks']=[r for r in vp['tasks'] if r['identity'] in ids];small['generation']=[]
 for r in small['records']:r['positions']=r['positions'][:1]
 for r in small['tasks']:r['positions']=r['positions'][:1]
 metrics=a.evaluate(s,prior,pop,small,'cuda',config);assert metrics['summary']['goal_by_kind']['P2']['aa'] is not None
 # Empty/short prefixes preserve original next-AA semantics and component masks.
 for target in [0,1,2,3]:
  r=small['records'][0];pp,y,gg,mm=a.load_decision(pop,r,target,'cuda');bb=a.prepare(pp,prior(pp),torch.zeros(1,1,36,device='cuda'),torch.zeros(1,1,device='cuda',dtype=torch.bool));loss,tt=a.actor_loss(s,bb,y,gg,mm,config['stream_weights']['AA']);assert torch.isfinite(loss);assert mm[0].tolist()==[target>=1,target>=2,target>=3]
 # Forward-only next-C-alpha fixture exercises corrected inference and assessment path.
 record=small['records'][0];task=next(r['task'] for r in small['tasks'] if r['task']['kind']=='P2');p,y,g,mask=a.load_decision(pop,record,10,'cuda');gt,gv=a.goal_tokens(task,'cuda');b=a.prepare(p,prior(p),gt,gv);l,mu,ls=s.models['actor'](b);aa=int(y[0]);action,info=a.corrected_action(mu[0,aa].cpu(),ls[0,aa].cpu(),(True,True,True),torch.zeros(3,dtype=torch.float64));assert torch.isfinite(action).all();assert info['represented_action_log_density']==info['represented_action_log_density']
assert tensor_sha(s)==state;prior.verify([])
# No actual complete generated design is produced during admission.
assert all(p.grad is None for p in s.parameters())
for update in [0,250,1000,2500,5000,10000]:assert update<=config['updates']
try:authorization(None,config);raise AssertionError('unapproved launch')
except PermissionError:pass
out=dict(final_configuration_sha256=seal,reference_sha256=file_sha(MAIN),sealed_panel_hashes=config['frozen_artifact_hashes'],new_loader_eval_summary_paths=True,one_step_corrected_geometry_fixture=True,empty_and_short_prefix_components=True,frozen_all_system_tensors=True,frozen_J1=True,gradients_created=0,optimizer_updates=0,study_updates=0,no_TEST_scoring=True,full_generation_runs=0)
with open(R/'results/FINAL_PROTOCOL_CHECKS.json','x') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
