"""Bounded CPU correctness tests for R1b mapping, scoring and retrieval predicates."""
import json,copy,hashlib,tempfile,subprocess,re
import numpy as np
from pathlib import Path
from r1b_common import R,PREV,write,sha
from r1b_afdb import candidate
from r1b_prepare import aligned_deviations
from protein_joint_sac.rl.sequence_rewards import mapped_structure_scores

def main():
 seq='ACDEFGHIKLMNPQRSTVWY';c={'sequence':seq};m=[{'start':{'residue_number':1},'end':{'residue_number':20},'unp_start':1,'unp_end':20}];r={'sequence':seq,'sequenceStart':1,'sequenceEnd':20,'modelEntityId':'AF_SYNTHETIC'};assert candidate(c,'SYNTHETIC',r,m)['match_status']=='EXACT_FULL_SEQUENCE_MATCH';rec=dict(r,sequence='M'+seq+'A',sequenceEnd=22);mp=[dict(m[0],unp_start=2,unp_end=21)];assert candidate(c,'SYNTHETIC',rec,mp)['match_status']=='VALID_MAPPED_FRAGMENT_OR_CONSTRUCT';mut=dict(r,sequence='V'+seq[1:]);assert candidate(c,'SYNTHETIC',mut,m)['match_status']=='HOMOLOGOUS_REFERENCE_ONLY';old={'uniprotSequence':seq,'uniprotStart':1,'uniprotEnd':20,'entryId':'OLD'};assert candidate(c,'SYNTHETIC',old,m) is None
 x=np.random.default_rng(33).normal(size=(30,3));rot=np.array([[0.,-1.,0],[1,0,0],[0,0,1]]);y=x@rot+3;assert aligned_deviations(x,y).max()<1e-10;assert mapped_structure_scores(x,y)['robust_fixed_map_TM_similarity_fullN']>1-1e-10;x[:3]=np.nan;v=mapped_structure_scores(x,y);assert abs(v['robust_fixed_map_TM_similarity_fullN']-.9)<1e-10 and abs(v['observed_only_TM_similarity']-1)<1e-10
 panel=json.load(open(R/'configs/NATIVE_PANEL_32.json'));assert len(panel['controls'])==32 and len({c['sequence_id'] for c in panel['controls']})==32 and sum(c['split']=='TRAIN' for c in panel['controls'])==24 and all(c['split'] in ['TRAIN','VALIDATION'] for c in panel['controls']);assert all(hashlib.sha256(c['sequence'].encode()).hexdigest()==c['sequence_id'] for c in panel['controls'])
 alignment=json.load(open(R/'STRUCTURE_ALIGNMENT_METHOD_COMPARISON.json'));assert max(abs(r['diagnostics']['original_mapped_score_delta']) for r in alignment['rows'])<1e-10;assert all(r['diagnostics']['USalign']['returncode']==0 for r in alignment['rows'])
 initial=json.load(open(R/'afdb/INITIAL_ALIGNMENT_RESULTS.json'))['rows'];current=json.load(open(R/'ALPHAFOLD_DB_NATIVE_RECOVERY.json'))['rows']
 for row in current:
  if row['status']!='SCORED':continue
  before=next(v for v in initial if v['id']==row['id']);assert abs(row['native_recovery']['robust_fixed_map_TM_similarity_fullN']-before['native_recovery']['robust_fixed_map_TM_similarity_fullN'])<1e-10
  pred=R/'alignment'/('AFDB_'+row['id']+'_predicted.pdb');assert sum(v.startswith('ATOM') for v in pred.read_text().splitlines())==row['native_recovery']['USalign']['predicted_observed_length']
 before=json.load(open(R/'folds/TRAIN_CALIBRATION_BEFORE_VALIDATION.json'));after=json.load(open(R/'folds/REWARD_REASSESSMENT.json'));assert before['beta_TRAIN_only']==after['beta_TRAIN_only'] and before['TRAIN_fit_ids']==after['TRAIN_fit_ids'] and before['TRAIN_holdout_ids']==after['TRAIN_holdout_ids']
 for p in (R/'source/scripts/e016_v3').glob('r1b_*.py'):compile(p.read_text(),str(p),'exec')
 tool=json.load(open(R/'STRUCTURE_ALIGNMENT_METHOD_COMPARISON.json'))['USalign_binary_path'];native=R/'alignment/single_train_128_native.pdb';run=subprocess.run([tool,str(native),str(native),'-mol','prot'],capture_output=True,text=True,check=True,timeout=30);scores=[float(v) for v in re.findall(r'TM-score=\s*([0-9.]+)',run.stdout)];assert len(scores)>=2 and min(scores)>1-1e-8
 write('R1B_CORRECTNESS_TESTS.json',{'status':'PASS','synthetic_exact_construct_homolog_predicates_checked':True,'new_API_fields_required_not_deprecated_names':True,'rigid_invariance_and_full_vs_observed_denominators':True,'32_unique_frozen_TRAIN_VALIDATION_sequences_verified':True,'prior_mapped_scores_reproduced_to_1e-10':True,'USalign_old_eight_successful':True,'USalign_identical_trace_similarity_one':True,'AFDB_USalign_export_omits_unmapped_coordinates':True,'AFDB_fixed_mapping_scores_unchanged_after_export_fix':True,'TRAIN_calibration_unchanged_by_VALIDATION_audit':True,'J1_optimizer_updates':0,'production_RL_updates':0});print('R1b CPU correctness tests PASS',flush=True)
if __name__=='__main__':main()
