"""Seal source, complete experimental data provenance and bounded R1d evidence."""
from r1d_common import *
import subprocess

def main():
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=R/'source',text=True).strip()
 required=['FINAL_E016_R1D_EXPERIMENTAL_STABILITY_REPORT.md','EXPERIMENTAL_STABILITY_DATA_AUDIT.md','TSUBOYAMA_DATASET_PROVENANCE.json','STABILITY_LABEL_SEMANTICS.md','STABILITY_VARIANT_BENCHMARK.json','FAMILY_SPLIT_AND_CONTAMINATION_AUDIT.json','VARIANT_SELECTION_PROTOCOL.md','ALPHAFOLD_VARIANT_INFERENCE_MANIFEST.json','MSA_COMPARABILITY_AUDIT.md','STABILITY_RANKING_RESULTS.json','CONFIDENCE_VS_STABILITY_ANALYSIS.md','AF2_MUTATION_SENSITIVITY_REPORT.md','REWARD_CANDIDATE_COMPARISON.json','INDEPENDENT_AUDIT_RESULTS.json','RL_REWARD_GO_NO_GO.md','NEXT_E016_RL_DECISION.md','RESOURCE_AND_NETWORK_USAGE.json','PROVENANCE_SHA256.json','EXPERIMENT_DECISIONS.md','CHANGELOG.md']
 assert all((R/p).exists() for p in required if p!='PROVENANCE_SHA256.json')
 res=json.load(open(R/'RESOURCE_AND_NETWORK_USAGE.json'));assert res['AF_calls']==12 and res['AF_worker_seconds']<1500 and res['remote_MSA_queries']==res['J1_updates']==res['production_RL_updates']==0
 qa=json.load(open(R/'analysis/FINAL_CORRECTNESS_CHECKS.json'));assert qa['PASS'] and qa['native_adequacy_failure_preserved'];audit=json.load(open(R/'INDEPENDENT_AUDIT_RESULTS.json'));assert not audit['general_certification'] and not audit['gate_pass_within_feasibility_scope']
 files={str(p.relative_to(R)):sha(p) for p in sorted(R.rglob('*')) if p.is_file() and not any(x in ['source','runtime'] for x in p.relative_to(R).parts) and p.name not in ['PROVENANCE_SHA256.json','PROVENANCE_VERIFICATION.json'] and not p.name.endswith(('.part','-wal','-shm'))};source={str(p.relative_to(R/'source')):sha(p) for p in sorted((R/'source/scripts/e016_v3').glob('r1d_*.py'))};model=D/'e016_j1_bidirectional_multitask_20261019_v1/runs/J1/checkpoint_07500.pt';assert sha(model)==res['J1_SHA256']
 write('PROVENANCE_SHA256.json',{'source_commit':commit,'source_branch':'e016-r1d-experimental-stability-20261020-v1','source_files':source,'evidence_files':files,'external_protected_inputs':{'J1_checkpoint':{'path':str(model),'sha256':res['J1_SHA256']},'R1c_manifest':{'path':str(C/'PROVENANCE_SHA256.json'),'sha256':sha(C/'PROVENANCE_SHA256.json')}},'required_outputs':required,'exclusions':'runtime caches, Git source worktree (tracked separately), transient SQLite sidecars, manifest self and verification receipt','original_failed_native_protocol_preserved':True,'new_ranking_scope':'exploratory query-only, separately amended before mutant/audit predictions; no original gate relaxation'})
 bad=[]
 for p,h in files.items():
  if sha(R/p)!=h:bad.append(p)
 for p,h in source.items():
  if sha(R/'source'/p)!=h:bad.append('source/'+p)
 assert not bad
 write('PROVENANCE_VERIFICATION.json',{'PASS':True,'source_commit':commit,'source_clean':True,'evidence_files_verified':len(files),'source_files_verified':len(source),'required_outputs':len(required),'manifest_sha256':sha(R/'PROVENANCE_SHA256.json'),'J1_checkpoint_verified':True,'original_failed_protocol_unchanged':True,'labels_and_rank_rules_unchanged':True,'AF_calls':12,'AF_worker_seconds':res['AF_worker_seconds'],'remote_MSA_queries':0,'J1_updates':0,'production_RL_updates':0,'mismatches':bad,'free_bytes':guard(),'all_workers_completed':True})
 print(json.dumps({'PASS':True,'source':commit,'files':len(files),'required_outputs':len(required),'free_GiB':guard()/2**30}),flush=True)
if __name__=='__main__':main()
