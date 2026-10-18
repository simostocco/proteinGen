"""Read-only gradient scale diagnostics on fixed real TRAIN prefixes, no optimizer."""
import json,math
import torch
import j1_global as j
from protein_joint_sac.data.accepted_supervised import CoordinateReader
from protein_joint_sac.data.scaled_supervised import canonical,internal
from protein_joint_sac.rl.checkpointing import load_checked
def run():
 j.core.setup();data=j.Data(False);panels=json.load(open(j.R/'configs/EVALUATION_PANELS.json'));ids=sorted({h for h,o,t in panels['TRAIN']})
 selected=[]
 for k in range(5):selected+= [h for h in ids if j.band(len(data.seqs[h]))==k][:2]
 reader=CoordinateReader(data.proposal.external);examples=[]
 for h in selected:
  o=data.owners[h][0];x=reader.read(data.rows[o]);data.ca[o]=canonical(x);data.g[o]=internal(x);examples.append((h,o,len(x)//2))
 final=json.load(open(j.R/'FINAL_SELECTION.json'));done=json.load(open(j.R/'runs/J1/COMPLETED.json'));model,parent=j.parent_model();del parent;model.eval();results={};cache={}
 for role,path,sha in [('PARENT',j.PARENT,j.PARENT_SHA),('SELECTED',final['selected_checkpoint'],final['selected_sha256']),('LATEST_TRAINED',done['latest_checkpoint'],done['latest_sha256'])]:
  if sha in cache:results[role]=cache[sha];continue
  state=load_checked(path);model.load_state_dict(state['model'],strict=True);del state
  b,y,g=j.core.make(data,examples,True);loss,aa,geo,parts=j.core.loss_call(model,b,y,g,j.CFG)
  values={'AA':aa.mean(),'geometry':geo.mean(),'distance_effective':parts[:,0].mean()/3,'angle_effective':parts[:,1].mean()/3,'torsion_effective':parts[:,2].mean()/3,'base':loss.mean()}
  rows={}
  for name,obj in values.items():
   grads=torch.autograd.grad(obj,list(model.parameters()),allow_unused=True,retain_graph=True)
   total=math.sqrt(sum(float(v.double().square().sum()) for v in grads if v is not None))
   if any(not torch.isfinite(v).all() for v in grads if v is not None):raise ValueError('nonfinite gradient diagnostic')
   rows[name]={'loss':float(obj.detach()),'gradient_L2_preclip':total};del grads
  cache[sha]={'checkpoint_sha256':sha,'prefix_decisions':len(examples),'valid_components_per_decision':3,'normalization':'mean over10midchain decisions, all3geometry components; geometry component effective1/3 weighting; not the64decision calibration batch','loss_gradient_scales':rows};results[role]=cache[sha];del values,loss,aa,geo,parts,b,y,g
 j.write(j.R/'BASE_LOSS_GRADIENT_COMPONENTS.json',{'results':results,'examples':examples,'TEST':'NOT_READ_OR_EVALUATED','optimizer_updates':0})
 print('BASE_GRADIENT_COMPONENTS_COMPLETE',flush=True)
if __name__=='__main__':run()
