"""Resume control preparation only after completed R0 tests; no optimizer."""
import json,math,time
from types import SimpleNamespace
import torch
from protein_joint_sac.data.scaled_supervised import Data
from protein_joint_sac.rl.checkpointing import load_checked
from protein_joint_sac.data.certified_unified import file_sha
from protein_joint_sac.rl.sequence_rl import rollout
from readiness_r0 import R,SOURCE,SHA,FALLBACK,actor,make_panel,write
def main():
 for n in ['RL_ACTION_LOGPROB_TESTS.json','PDB_BC_GPU_INTEGRATION.json','HYBRID_SAC_TOY_RESULTS.json']:assert json.load(open(R/n))['status']=='PASS'
 write('R0_PANEL_SCHEMA_RECOVERY.json',{'original_error':'KeyError auth_asym_id in preserved historical NATIVE_CONTROL_PANEL record','fixed_by':'resolve author chain and residue positions from exact SHA-verified TARGET_AUDIT.json; no coordinate/identity guess','GPU_correctness_and_BC_tests_not_repeated':True,'J1_optimizer_updates':0})
 start=time.monotonic();base=Data(False);data=SimpleNamespace(base=base,proposal=base.proposal,seqs=base.seqs);make_panel(data)
 torch.set_num_threads(6);torch.manual_seed(20261019);torch.cuda.manual_seed_all(20261019);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;model=actor();old=load_checked(FALLBACK);assert file_sha(FALLBACK)=='9208263300fb9e88519ff0d40e6d9d29a9744b709d9a6181819cd0894b593000';model.load_state_dict(old['model']);state,trace,_=rollout(model,64,2026101901,file_sha(FALLBACK));write('generated/fallback_initial64.json',{'sequence':''.join(state.residues),'ca':state.ca.cpu().tolist(),'trace':trace,'policy_sha256':file_sha(FALLBACK),'seed':2026101901})
 write('R0_GPU_RESOURCE.json',{'resume_preparation_seconds':time.monotonic()-start,'Torch_allocated_peak_this_resume':torch.cuda.max_memory_allocated(),'J1_optimizer_updates':0,'main_actor_unchanged':True,'GPU_name':torch.cuda.get_device_name(),'PDB_split_manifest_sha256':file_sha(base.proposal.root/'PROPOSAL_MANIFEST.json'),'schema_recovery':'R0_PANEL_SCHEMA_RECOVERY.json'})
 print(json.dumps({'phase':'CONTROL_PANEL_READY','controls':len(json.load(open(R/'configs/NATIVE_CONTROL_PANEL.json'))['controls']),'J1_optimizer_updates':0}),flush=True)
if __name__=='__main__':main()
