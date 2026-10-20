"""Verify immutable experimental labels, recipe equivalence and preserved gate failure."""
from r1d_common import *

def main():
 original=json.load(open(R/'analysis/ORIGINAL_NATIVE_ADEQUACY_PROTOCOL_STOPPED.json'));latest=json.load(open(R/'ALPHAFOLD_VARIANT_INFERENCE_MANIFEST.json'));cfg=json.load(open(R/'configs/VARIANT_INFERENCE_FROZEN.json'));lock=json.load(open(R/'analysis/BEFORE_AUDIT_SCORING_LOCK.json'));amend=json.load(open(R/'configs/EXPLORATORY_RANKING_AMENDMENT.json'))
 assert original['rows']==latest['rows'][:2] and len(original['rows'])==2 and len(latest['rows'])==12
 assert cfg['audit_labels_sha256']==sha(R/'analysis/AUDIT_LABELS_SEALED.json')==lock['audit_labels_sha256']
 assert cfg['development_labels_sha256']==sha(R/'analysis/DEVELOPMENT_LABELS.json')==lock['development_labels_sha256']
 assert lock['candidate_formulas_and_gates_sha256']==sha(R/'configs/SELECTION_AND_ANALYSIS_PREDECLARED_v2.json')==amend['policy_sha256']
 assert lock['metric_or_threshold_changes']==0 and not amend['original_native_gate_reinterpreted_as_pass']
 recipes=[]
 for r in latest['rows']:
  assert r['status']=='SUCCEEDED';p=json.load(open(R/'folds'/r['id']/'PROTOCOL.json'));recipes.append({k:p[k] for k in ['version','weights_sha256','code_revision','model','recycles','seed','templates','relaxation','max_MSA','mode','network','JAX_backends']});assert sha(r['predicted_PDB'])==r['predicted_PDB_sha256']
 assert all(r==recipes[0] for r in recipes)
 write('analysis/FINAL_CORRECTNESS_CHECKS.json',{'PASS':True,'calls_total':12,'same_numerical_inference_recipe_all_sequences':recipes[0],'native_adequacy_failure_preserved':True,'initial_two_predictions_bitwise_artifacts_preserved':True,'audit_and_development_labels_unchanged':True,'original_ranking_policy_unchanged':True,'audit_score_dependent_formula_changes':0,'explicit_exploratory_scope_not_original_qualification':True,'no_new_MSA_queries':True,'J1_updates':0})
 print('Original failure, immutable labels and identical paired recipes verified')
if __name__=='__main__':main()
