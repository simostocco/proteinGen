"""Final initialized RL-state GPU recovery and nonzero-KL routing; no J1 steps."""
import copy,json,hashlib,torch
from pathlib import Path
from protein_joint_sac.models.build_critic import TwinCritics
from protein_joint_sac.models.dense_geometry_attention import ModelConfig
from protein_joint_sac.rl.sequence_sac import SequenceFirstSAC
from protein_joint_sac.rl.sequence_rl import reference_kl
from protein_joint_sac.rl.checkpointing import save_atomic,load_checked
from protein_joint_sac.rl.bc_training import rng_state,restore_rng
from readiness_r0 import R,SOURCE,SHA,actor,state_fixture,tensor_same,write,guard

def main():
 guard();torch.set_num_threads(6);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(7119);torch.cuda.manual_seed_all(7119)
 source=load_checked(SOURCE);a=actor();a.load_state_dict(source['model']);ref=actor();ref.load_state_dict(source['model']);ref.requires_grad_(False);q=TwinCritics(ModelConfig(width=32,heads=4,layers=1,ffn_mult=2,chemistry=False,backend='sdpa')).cuda();c={'actor_lr':1e-6,'critic_lr':1e-4,'alpha_A':.01,'alpha_G':.001,'target_AA_entropy':2.,'target_geometry_entropy':0.,'temperature_lr':1e-4,'production_authorized':False,'numeric_scope':'engineering initialization only, uncalibrated'};learner=SequenceFirstSAC(a,ref,q,c);s=state_fixture(8,64).to('cuda')
 # Deliberately perturb a TEST-ONLY reference, never actor or immutable file.
 perturbed=copy.deepcopy(ref)
 with torch.no_grad():next(p for n,p in perturbed.named_parameters() if n.startswith('geom.') and n.endswith('bias')).add_(.05)
 a.zero_grad(set_to_none=True);gkl=reference_kl(a,perturbed,s)[1];gkl.backward();assert float(gkl)>0 and torch.isfinite(gkl);assert all(p.grad is None or not torch.any(p.grad!=0) for n,p in a.named_parameters() if n.startswith('aa_head.'));assert any(p.grad is not None and torch.any(p.grad!=0) for n,p in a.named_parameters() if not n.startswith(('geom.','selected_aa.','aa_head.')))
 a.zero_grad(set_to_none=True);sampler=torch.load(R/'actor/PDB_sampler_initial.pt',map_location='cpu',weights_only=True);payload={'model':a.state_dict(),'reference_source_sha256':SHA,'critics_and_targets':q.state_dict(),'actor_optimizer':learner.actor_opt.state_dict(),'critic_optimizer':learner.critic_opt.state_dict(),'temperature_optimizer':learner.alpha_opt.state_dict(),'temperature_state':learner.temperatures.state_dict(),'sampler':sampler,'rng':rng_state(),'config':c,'new_J1_updates':0,'historical_supervised_updates':43000,'scheduler':{'status':'UNSEALED_NO_PRODUCTION_SCHEDULE'},'oracle_budget_db':'replay/oracle_wall.sqlite','scope':'R0 initialized recovery state, production disabled'};path=R/'actor/readiness_initial_full.pt';h=save_atomic(path,payload)
 with torch.no_grad():logits=a.logits(s).clone();draw=torch.randn(16,device='cuda')
 recovered=load_checked(path);a2=actor();a2.load_state_dict(recovered['model']);q2=TwinCritics(ModelConfig(width=32,heads=4,layers=1,ffn_mult=2,chemistry=False,backend='sdpa')).cuda();q2.load_state_dict(recovered['critics_and_targets']);other=SequenceFirstSAC(a2,ref,q2,c);other.actor_opt.load_state_dict(recovered['actor_optimizer']);other.critic_opt.load_state_dict(recovered['critic_optimizer']);other.alpha_opt.load_state_dict(recovered['temperature_optimizer']);other.temperatures.load_state_dict(recovered['temperature_state']);restore_rng(recovered['rng'])
 with torch.no_grad():assert torch.equal(a2.logits(s),logits) and torch.equal(torch.randn(16,device='cuda'),draw)
 assert tensor_same(a.state_dict(),a2.state_dict()) and tensor_same(q.state_dict(),q2.state_dict());assert not learner.actor_opt.state and not learner.critic_opt.state;assert tensor_same({n:p.cpu() for n,p in a.state_dict().items()},source['model'])
 write('J1_INITIAL_STATE_GPU_RECOVERY.json',{'status':'PASS','checkpoint_path':str(path),'sha256':h,'actor_reference_binding_and_twin_targets_saved':True,'optimizer_temperature_sampler_rng_saved':True,'initial_optimizer_states_empty_no_J1_updates':True,'same_GPU_restored_logits_and_rng_bitwise':True,'nonzero_test_reference_geometry_KL':float(gkl.detach()),'geometry_KL_has_no_AA_head_gradient':True,'shared_geometry_KL_gradient_present':True,'immutable_reference_not_perturbed':True,'J1_parameters_unchanged':True,'J1_optimizer_updates':0,'scope':'initialized full-state restart; nonempty optimizer moment restart separately tested on synthetic GPU bandit','scheduler_production_sealed':False})
 print('Initialized J1 RL state GPU recovery PASS; zero J1 optimizer updates.',flush=True)
if __name__=='__main__':main()
