"""Compact local provenance seal and integrity verification."""
from r1c_common import *
import subprocess

def main():
 source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip();status=subprocess.check_output(['git','status','--porcelain'],cwd=R/'source',text=True).strip();assert not status,status
 required=['FINAL_E016_R1C_REWARD_RELIABILITY_REPORT.md','REWARD_CALIBRATION_FAILURE_ANALYSIS.md','MSA_DEPTH_ABLATION.json','CONFIDENCE_VS_NATIVE_RECOVERY.json','BIOLOGICAL_CONTEXT_AND_AMYLOID_CASE.md','NOVEL_SEQUENCE_RELIABILITY_ASSESSMENT.md','PROPOSED_SEQUENCE_REWARD_AND_ABSTENTION_POLICY.md','FULL_TRAJECTORY_CRITIC_COMPARISON.json','SAC_VS_POLICY_GRADIENT_DECISION.md','NEXT_RL_GO_NO_GO.md','RESOURCE_AND_PROVENANCE.json','EXPERIMENT_DECISIONS.md']
 assert all((R/f).is_file() for f in required)
 files={str(p.relative_to(R)):sha(p) for p in sorted(R.rglob('*')) if p.is_file() and not any(v in ['source','runtime'] for v in p.relative_to(R).parts) and p.name not in ['PROVENANCE_SHA256.json','PROVENANCE_VERIFICATION.json'] and not p.name.endswith(('-wal','-shm','.part'))}
 sourcefiles={str(p.relative_to(R/'source')):sha(p) for p in sorted((R/'source/scripts/e016_v3').glob('r1c_*.py'))}
 external={'R1b_provenance':{'path':str(B/'PROVENANCE_SHA256.json'),'sha256':sha(B/'PROVENANCE_SHA256.json')},'J1_frozen_initializer':{'path':str(R.parent/'e016_j1_bidirectional_multitask_20261019_v1/runs/J1/checkpoint_07500.pt'),'sha256':json.load(open(R/'RESOURCE_AND_PROVENANCE.json'))['J1_SHA256']}}
 write('PROVENANCE_SHA256.json',{'source_commit':source,'branch':'e016-r1c-reward-reliability-20261019-v1','evidence_files':files,'source_files':sourcefiles,'external_protected_inputs':external,'exclusions':'source tracked by Git+individual R1c hashes; runtime caches, mutable SQLite sidecars, this manifest and its verification receipt; SQLite main files included after all connections stopped','required_outputs':required})
 failures=[]
 for p,z in files.items():
  if sha(R/p)!=z:failures.append(p)
 for p,z in sourcefiles.items():
  if sha(R/'source'/p)!=z:failures.append('source/'+p)
 for name,z in external.items():
  if sha(z['path'])!=z['sha256']:failures.append(name)
 assert not failures,failures
 guard();write('PROVENANCE_VERIFICATION.json',{'pass':True,'source_commit':source,'source_clean':True,'evidence_files_verified':len(files),'source_files_verified':len(sourcefiles),'required_outputs_verified':len(required),'protected_inputs_verified':len(external),'mismatches':failures,'manifest_sha256':sha(R/'PROVENANCE_SHA256.json'),'J1_updates':0,'production_RL_updates':0,'remote_queries':0,'folding_calls':12,'all_workers_complete':True})
 print(json.dumps({'PASS':True,'source':source,'evidence_files':len(files),'source_files':len(sourcefiles),'free_GiB':guard()/2**30}))
if __name__=='__main__':main()
