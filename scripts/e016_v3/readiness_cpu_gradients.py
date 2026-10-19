"""CPU-only nonzero-drift KL isolation and streaming-gradient restart tests."""
import copy,json
from pathlib import Path
import torch
from torch.nn import functional as F
from protein_joint_sac.contracts import AA
from protein_joint_sac.models.joint_actor import JointActor
from protein_joint_sac.models.dense_geometry_attention import ModelConfig
from protein_joint_sac.models.batched_bc import collate
from protein_joint_sac.rl.sequence_rl import reference_kl
from protein_joint_sac.rl.sequence_sac import backward_sequence_first
from protein_joint_sac.rl.sequence_pdb_regularizer import PDBRegularizer
from readiness_r0 import state_fixture,R,tensor_same
class Sampler:
 def __init__(self):self.cursor=0
 def draw(self):self.cursor+=1;return list(range(64)),list(range(32))
 def state_dict(self):return {'cursor':self.cursor}
 def load_state_dict(self,s):self.cursor=s['cursor']
class ControlledData:
 def batch(self,examples,device='cpu'):
  b=collate([state_fixture() for _ in examples],device);return b,torch.tensor([i%20 for i in examples],device=device),torch.tensor([[3.8,1.9,.3] for i in examples],dtype=torch.float64,device=device)
def main():
 torch.set_num_threads(2);torch.manual_seed(919);model=JointActor(ModelConfig(width=32,heads=4,layers=1,ffn_mult=2,chemistry=False,backend='sdpa')).eval();ref=copy.deepcopy(model).requires_grad_(False)
 for n,p in model.named_parameters():p.requires_grad_(not n.startswith(('geom.','selected_aa.')))
 with torch.no_grad():ref.geom[2].bias[2]+=.3
 s=state_fixture();ka,kg=reference_kl(model,ref,s);kg.backward();assert kg>0 and model.aa_head.weight.grad is None and any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.startswith('encoder.'))
 pdb=PDBRegularizer(model,ControlledData(),Sampler());pdb.begin();before={'PDB':copy.deepcopy(pdb.state_dict()),'RNG':torch.get_rng_state()};weights=copy.deepcopy(model.state_dict())
 def sequence():return F.cross_entropy(model.logits(s)[None],torch.tensor([3]))+pdb.backward_AA(.03)
 def geometry():return .02*reference_kl(model,ref,s)[1]+pdb.backward_geometry(.03)
 a,b,stats=backward_sequence_first(model,sequence,geometry,{'S':2.58,'A':2.57});grads={n:None if p.grad is None else p.grad.clone() for n,p in model.named_parameters()};pdb.load_state_dict(before['PDB']);torch.set_rng_state(before['RNG']);x,y,again=backward_sequence_first(model,sequence,geometry,{'S':2.58,'A':2.57});assert tensor_same(grads,{n:p.grad for n,p in model.named_parameters()}) and stats==again and torch.equal(a,x) and torch.equal(b,y);assert stats['geometry_to_sequence_shared_ratio']<=.5+1e-12 and not model.training and tensor_same(weights,model.state_dict()) and not torch.cuda.is_initialized()
 out={'status':'PASS','scope':'controlled software fixture; actual PDB GPU integration reported separately','nonzero_geometry_KL':float(kg.detach()),'geometry_KL_expectation_AA_weights_detached_for_gradient_budget':True,'geometry_KL_no_AA_head_gradient_under_drift':True,'geometry_parameters_frozen':True,'streaming_BC_AA_targets':64,'streaming_BC_geometry_targets':32,'gradient_restart_bitwise':True,'model_weights_unchanged':True,'actor_mode_restored':True,'CUDA_initialized':False,'clipping':stats};(R/'STREAMED_BC_AND_KL_ISOLATION_TESTS.json').write_text(json.dumps(out,indent=2)+'\n');print('Streaming BC, nonzero-KL gradient isolation and exact CPU gradient restart PASS.',flush=True)
if __name__=='__main__':main()
