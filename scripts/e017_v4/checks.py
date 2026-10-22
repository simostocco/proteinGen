"""TRAIN-only recovery/leakage/performance gates. Diagnostic updates never enter pilot."""
import importlib.util,sys,pathlib,json,time,copy,math
import numpy as np
import torch
ROOT=pathlib.Path('/mnt/d/Simone/proteinGen_data/e017_v4_multistep_20261022_v1');sys.path.insert(0,str(ROOT/'source/scripts/e017_v4'));import experiment as v
from protein_joint_sac.crisp_v3.storage import rng_state,set_rng
from protein_joint_sac.crisp_v3.recovery import state_digest,atomic_checkpoint

def optimizer(e):
 o=torch.optim.AdamW([p for p in e.actor.parameters() if p.requires_grad],lr=1e-5,weight_decay=.01,foreach=False);s=torch.optim.lr_scheduler.LambdaLR(o,lambda u:min(1.,(u+1)/25)*(.1+.9*(1+math.cos(math.pi*min(u,500)/500))/2));return o,s

def step(e,o,s,task,ns):
 e.actor.train();o.zero_grad(set_to_none=True);start=time.monotonic()
 with torch.autocast('cuda',dtype=torch.bfloat16):loss,terms,_,_=e.rollout(task,ns,'C')
 loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in e.actor.parameters() if p.requires_grad],1.,error_if_nonfinite=True);o.step();s.step();torch.cuda.synchronize();return dict(loss=float(loss.detach()),norm=float(norm),seconds=time.monotonic()-start,actor=v.tensor_sha(e.actor),optimizer=state_digest(o.state_dict()),scheduler=state_digest(s.state_dict()),rng=state_digest(rng_state()))

def main(stage):
 v.seed(v.CONFIG['seed']);e=v.Engine();tasks=v.dataset('TRAIN');o,s=optimizer(e);path=ROOT/'checkpoints'/('engineering_restart_'+v.config_hash()[:8]+'.pt')
 if stage=='parent':
  e.actor.eval();t=tasks[0];changed=copy.deepcopy(t);changed['hidden_native_future']=(np.array(changed['hidden_native_future'])+1000).tolist();changed['sequence']=changed['sequence'][:t['t']+8]+changed['sequence'][t['t']+8:][::-1]
  with torch.no_grad():
   _,_,x,_=e.rollout(t,8104,'C');_,_,y,_=e.rollout(changed,8104,'C');assert torch.equal(x,y)
   p=e.p(t,v.prefix(t));a=e.prior(p);b=e.prior_forward(p);assert all(torch.equal(a[k],b[k]) for k in a)
  # B guide has no gradients into the first seven distributions; C finite-difference test already passed all steps.
  _,terms,_,z=e.rollout(t,8104,'B');gg=torch.autograd.grad(terms['goal'],z['mus'],allow_unused=True);assert all(g is None or not g.any() for g in gg[:7]) and gg[-1] is not None
  result=dict(hidden_intermediate_geometry_not_input=True,future_AA_suffix_not_input=True,J1_forward_matches_original_exactly=True,B_one_step_gradient_norms=[float(g.norm()) if g is not None else 0. for g in gg]);del gg,z,terms
  calibration=[]
  for t in tasks[:4]:
   _,tt,_,_=e.rollout(t,8104,'C');nn={}
   for k,val in tt.items():
    gs=torch.autograd.grad(v.CONFIG['weights'][k]*val,[p for p in e.actor.parameters() if p.requires_grad],retain_graph=True,allow_unused=True);nn[k]=math.sqrt(sum(float(g.float().square().sum()) for g in gs if g is not None))
   calibration.append(nn)
  del gs,tt;result['four_TRAIN_weighted_gradient_norms']=calibration
  e.actor.zero_grad(set_to_none=True);step(e,o,s,tasks[0],8111)
  payload=dict(actor=e.actor.state_dict(),optimizer=o.state_dict(),scheduler=s.state_dict(),rng=rng_state(),sampler=dict(order=list(range(64)),cursor=1),config=v.CONFIG,TRAIN_SHA=v.sha(ROOT/'manifests/TRAIN.json'),parent_SHA=v.SHA)
  receipt=atomic_checkpoint(path,payload,v.POLICY);v.save('results/RESTART_RECEIPT.json',receipt);expected=step(e,o,s,tasks[1],8112);v.save('results/RESTART_EXPECTED.json',expected);e.prior.verify([o]);result.update(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),diagnostic_updates=2,single_task_update_seconds=expected['seconds'],projected_batch2_1000_updates_seconds=expected['seconds']*2*1000)
  v.save('results/ENGINEERING_PARENT.json',result);print(result,flush=True)
 else:
  st=torch.load(path,map_location='cpu',weights_only=False);receipt=json.load(open(ROOT/'results/RESTART_RECEIPT.json'));assert v.sha(path)==receipt['sha256'];e.actor.load_state_dict(st['actor']);o.load_state_dict(st['optimizer']);s.load_state_dict(st['scheduler']);set_rng(st['rng']);actual=step(e,o,s,tasks[1],8112);expected=json.load(open(ROOT/'results/RESTART_EXPECTED.json'));assert all(actual[k]==expected[k] for k in ['loss','norm','actor','optimizer','scheduler','rng']),(actual,expected);e.prior.verify([o]);v.save('results/FRESH_GPU_RESTART.json',dict(PASS=True,expected=expected,actual=actual,J1_frozen=True,diagnostic_replay_updates=1,main_lineage_updates=0));print('FRESH_GPU_RESTART_PASS',flush=True)

if __name__=='__main__':main(sys.argv[1])
