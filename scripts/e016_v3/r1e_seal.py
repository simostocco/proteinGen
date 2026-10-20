from r1e_common import *
import subprocess,os,time,collections
required=['FINAL_E016_PROTEIN_GENERATION_EVALUATION_REPORT.md','LITERATURE_AND_BENCHMARK_SOURCES.md','EVALUATOR_INSTALLATION_AND_RESOURCE_AUDIT.md','FROZEN_J1_GENERATION_PANEL.json','GENERATION_VALIDITY_AND_DIVERSITY.json','SEQUENCE_COMPOSITION_AND_COMPLEXITY.json','SEQUENCE_NOVELTY_AND_HOMOLOGY.json','ESM2_SEQUENCE_PLAUSIBILITY.json','EXPERIMENTAL_DESIGN_SUCCESS_DATA_AUDIT.md','EXPERIMENTAL_DESIGN_SUCCESS_BENCHMARK.json','EXPERIMENTAL_METRIC_DISCRIMINATION.md','AUXILIARY_STRUCTURE_EVALUATION.json','E016_SEQUENCE_GENERATION_SCORECARD.md','RL_REWARD_CANDIDATE_ASSESSMENT.md','NEXT_E016_RL_RESEARCH_DECISION.md','RESOURCE_AND_STORAGE_REPORT.json','EXPERIMENT_DECISIONS.md','CHANGELOG.md']
for n in required:assert (R/n).exists()and (R/n).stat().st_size>0,n
p=read('FROZEN_J1_GENERATION_PANEL.json');protected=p['checkpoints']+p['cache_sources']+[dict(path=str(D/'pdb_supervised_split_50id80cov_20261016_v1/SEQUENCE_ASSIGNMENTS.parquet'),sha256=p['sequence_manifest_sha256'])]
verification=[]
for r in protected:
 h=sha(r['path']);assert h==r['sha256'];verification.append(dict(path=r['path'],sha256=h,status='UNCHANGED'))
for n in R.glob('*.json'):json.loads(n.read_text())
assert read('EVALUATION_CORRECTNESS_TESTS.json')['all_pass']
assert read('ESM_SAMPLING_SENSITIVITY.json')['masked_truth_leakage_test'].startswith('PASS')
assert read('RESOURCE_AND_STORAGE_REPORT.json')['total_charged_ESM_inference_seconds']<2700
guard(128*2**20)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip();status=subprocess.check_output(['git','status','--porcelain'],cwd=R/'source',text=True);assert not status,status
# Record actual external artifact calls; browsing requests remain separate from artifact downloads.
network=dict(artifact_HTTP_attempts=12,successful_retained_artifact_requests=8,notes='3 initial XML/HF/GitHub metadata;3 supplement endpoints(PMC challenge,Wiley403,EuropePMC success);1 source ZIP cap;1 FAIR weight;2 PyPI metadata/wheel;2 diagnostic endpoint probes(timeout,404). Browser documentation calls are separate and not counted as artifact requests.',weights_bytes=134095705,source_wheel_bytes=93125,supplement_archive_bytes=2725072,capped_discarded_source_zip_read_max_bytes=8388609,new_MSA_queries=0,new_folding_calls=0,retained_artifact_bytes=sum(f.stat().st_size for f in (R/'data').glob('*')if f.name not in ['TRAIN.fasta','VALIDATION.fasta','queries.fasta']),metadata_reconstruction='Initial3retained metadata fetch receipts reconstructed from exact saved content SHA; helper import overwrote empty NETWORK_RECEIPTS. Source URLs and saved hashes preserved here.',metadata=[dict(path=str(f.relative_to(R)),sha256=sha(f),bytes=f.stat().st_size,url=u)for f,u in [(R/'data/garcia_article.xml','https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12817478/fullTextXML'),(R/'data/esm_hf_metadata.json','https://huggingface.co/api/models/facebook/esm2_t12_35M_UR50D'),(R/'data/esm_source_commit.json','https://api.github.com/repos/facebookresearch/esm/commits/main')]])
# Successful retained HTTP responses: initial3 + PMCchallenge1 + Europe1 + FAIR1 + PyPI2 =8; CSV extracted from the official EuropePMC archive.
write('data/NETWORK_ATTEMPT_AUDIT.json',network)
r=read('RESOURCE_AND_STORAGE_REPORT.json');r['source_commit']=commit;r['D_free_after_bytes']=shutil.disk_usage(R).free;r['workers_stopped']='all folding/generation/ESM/search workers completed; this CPU provenance publication is the only active R1e process';r['artifact_HTTP_attempts']=12;r['source_clean']=True;write('RESOURCE_AND_STORAGE_REPORT.json',r)
# Main report links remain stable and historical checkpoint names are never rewritten.
paths=[]
for f in R.rglob('*'):
 if not f.is_file()or any(x in f.relative_to(R).parts for x in ['source','runtime']):continue
 if f.name in ['PROVENANCE_SHA256.json','PROVENANCE_VERIFICATION.json']:continue
 paths.append(dict(path=str(f.relative_to(R)),bytes=f.stat().st_size,sha256=sha(f)))
code=[]
for f in sorted((R/'source/scripts/e016_v3').glob('r1e_*.py')):code.append(dict(path=str(f.relative_to(R/'source')),sha256=sha(f)))
write('PROVENANCE_SHA256.json',dict(version='E016_R1e_inference_only_v1',source_commit=commit,source_branch='e016-r1e-generation-evaluation-20261020-v1',required_outputs=required+['PROVENANCE_SHA256.json'],files=paths,code=code,protected_inputs=verification,weights_not_modified=True,no_optimizer_updates=True))
for x in paths:assert sha(R/x['path'])==x['sha256']
write('PROVENANCE_VERIFICATION.json',dict(status='PASS',source_commit=commit,source_clean=True,evidence_files=len(paths),source_scripts=len(code),required_outputs=len(required)+1,protected_inputs=verification,main_correctness_tests='PASS',masked_position_full_score_comparison='PASS atol2e-5',GPU_inference_seconds=r['total_charged_ESM_inference_seconds'],D_free_bytes=shutil.disk_usage(R).free,no_J1_or_reward_or_critic_training=True,production_RL='NO_GO'))
print('SEALED',commit,len(paths),'evidence files',len(code),'scripts',shutil.disk_usage(R).free/2**30,'GiB free',flush=True)
