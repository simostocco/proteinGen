"""Read-only checkpoint assessment and bounded BC-prefix cost measurements; no optimizer step."""
import json,time,math,collections
from dataclasses import asdict
import numpy as np
import torch
import real_pdb_pilot as m
from protein_joint_sac.rl.bc_training import BCConfig,BCTrainer
from protein_joint_sac.data.accepted_supervised import AcceptedPopulation
from protein_joint_sac.data.pdb_split_proposal import PDBSplitProposal

def main():
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 proposal=PDBSplitProposal(m.P);costs={};components={}
 for name in ['S1','S2','J1','J2']:
  if not (m.R/'runs'/name/'RESULT.json').exists():raise RuntimeError('all pilots must pass first')
  c=BCConfig(**json.load(open(m.R/'configs'/f'{name}.json')));p=AcceptedPopulation(m.R/'ADMISSION.json',c.mode,name!='S2');t=BCTrainer(c,p);t.resume(m.R/'runs'/name/'checkpoint_012.pt');t.model.eval()
  if name=='S2':
   original=p.groups['VALIDATION'];p.groups['VALIDATION']={h:ii for h,ii in original.items() if proposal.sequences[h]['joint_eligible']}
   m.write(m.R/'runs/S2/VALIDATION_MATCHED_FINAL.json',m.validation(t));p.groups['VALIDATION']=original
  entries=[]
  for b,Lgoal in enumerate([64,128,256,384,500]):
   candidates=[(abs(len(p.rows[ii[0]]['sequence'])-Lgoal),h,ii[0]) for h,ii in p.groups['TRAIN'].items() if m.band(len(p.rows[ii[0]]['sequence']))==b];_,h,i=min(candidates);L=len(p.rows[i]['sequence']);prefix=L-1
   # Warm source and kernel paths; report finite gradient evaluation, no Adam step.
   t.model.zero_grad(set_to_none=True);loss,_=t.loss(i,prefix);loss.backward();torch.cuda.synchronize();times=[];peak=0;reserved=0
   for repeat in range(2):
    torch.cuda.reset_peak_memory_stats();t.model.zero_grad(set_to_none=True);torch.cuda.synchronize();a=time.monotonic();loss,metrics=t.loss(i,prefix);torch.cuda.synchronize();forward=time.monotonic()-a;a=time.monotonic();loss.backward();torch.cuda.synchronize();back=time.monotonic()-a
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in t.model.parameters());times.append({'forward_and_prefix_seconds':forward,'backward_seconds':back});peak=max(peak,torch.cuda.max_memory_allocated());reserved=max(reserved,torch.cuda.max_memory_reserved())
   entries.append({'band':b,'requested_band_upper_length':Lgoal,'actual_length':L,'prefix_length':prefix,'sequence_hash':h,'timings':times,'mean_total_prefix_seconds':float(np.mean([x['forward_and_prefix_seconds']+x['backward_seconds'] for x in times])),'allocated_peak_bytes':peak,'reserved_peak_bytes':reserved,'optimizer_steps':0})
  costs[name]=entries
  if c.mode=='joint':
   details=json.load(open(m.R/'runs'/name/'VALIDATION_FINAL.json'))['details'];byid={r['observation_id']:i for i,r in enumerate(p.rows)};sums=np.zeros(3);counts=np.zeros(3,dtype=int);bands=collections.defaultdict(lambda:[np.zeros(3),np.zeros(3,dtype=int)])
   with torch.no_grad():
    for r in details:
     state,label,g=p.target(byid[r['observation_id']],r['t'],'joint','cuda');h=t.model.encode(state);dist=t.model.geometry_distribution(state,m.AA[label],h);mask=dist.mask
     safe=torch.stack([g[0] if mask[0] else g.new_tensor(1.),g[1] if mask[1] else g.new_tensor(math.pi/2),g[2] if mask[2] else g.new_tensor(0.)]);u=torch.stack([safe[0].log(),torch.logit(safe[1]/math.pi),torch.atanh(safe[2]/math.pi)]);z=torch.where(mask,-dist.base.log_prob(u)+dist.log_jacobian(u),torch.zeros_like(u)).cpu().numpy();valid=mask.cpu().numpy().astype(int)
     if not np.isfinite(z).all():raise RuntimeError('component NLL not finite')
     sums+=z;counts+=valid;bands[str(r['length_band'])][0]+=z;bands[str(r['length_band'])][1]+=valid
   components[name]={'components':['distance_A','angle_rad','signed_torsion_rad'],'NLL_per_defined_component':(sums/np.maximum(counts,1)).tolist(),'valid_component_counts':counts.tolist(),'bands':{k:{'NLL':(s/np.maximum(c,1)).tolist(),'counts':c.tolist()} for k,(s,c) in bands.items()},'density_units_note':'differential NLL depends on coordinate units; torsion tanh seam retained','full_validation_population':False}
  del t,p
  torch.cuda.empty_cache()
 m.write(m.R/'REAL_PREFIX_COST_BY_LENGTH.json',{'status':'PASS','scope':'REAL_TRAIN_PREFIX_FORWARD_BACKWARD_NO_OPTIMIZER_STEPS','measurements':costs,'resources':m.guard()});m.write(m.R/'VALIDATION_GEOMETRY_COMPONENTS.json',components)
 print('POSTPILOT_PASS',flush=True)
if __name__=='__main__':main()
