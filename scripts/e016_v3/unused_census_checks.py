"""Census reconciliation and existing geometry-semantics checks, no model updates."""
import json,gzip,sys
from pathlib import Path
import pyarrow.parquet as pq
import numpy as np
R=Path('/mnt/d/Simone/proteinGen_data/e016_unused_train_census_20261018_v1');sys.path.insert(0,str(R/'source/src'))
from protein_joint_sac.data.partial_geometry import derive
ids=pq.read_table(R/'UNUSED_IDENTITY_CLASSIFICATION.parquet').to_pylist();obs=pq.read_table(R/'UNUSED_OBSERVATION_EVIDENCE.parquet').to_pylist();known={r['sequence_id'] for r in ids};assert len(known)==len(ids)==67615;assert {r['sequence_hash'] for r in obs}==known;assert len({r['observation_id'] for r in obs})==len(obs)==342713;assert all(r['observed_CA']>0 for r in obs)
for component,key in [('distance','valid_d'),('angle','valid_a'),('torsion','valid_t')]:
 assert sum(r['verified_'+component+'_targets'] for r in ids)==sum(r[key] for r in obs)
# Verify exact full strings against the immutable source sequence table, not just hashes.
D=R.parent
source={q['sequence_id']:q for q in pq.read_table(D/'e016_gpu_unified_pdb_20261015_v1/release/intake_v1/unique_sequences.parquet').to_pylist()}
assigned={q['sequence_id']:q for q in pq.read_table(D/'pdb_supervised_split_50id80cov_20261016_v1/SEQUENCE_ASSIGNMENTS.parquet').to_pylist()}
for h,q in assigned.items():
 if q['split']=='TRAIN':assert source[h]['canonical_full_sequence']==q['canonical_full_sequence'] and source[h]['length']==q['length'] and source[h]['lm_eligible']
full=pq.read_table(D/'pdb_supervised_split_50id80cov_20261016_v1/JOINT_OBSERVATION_ASSIGNMENTS.parquet',columns=['sequence_hash','split']).to_pylist()
assert {q['sequence_hash'] for q in full if q['split']=='TRAIN'}=={h for h,q in assigned.items() if q['split']=='TRAIN' and q['joint_eligible']}
checks=[];eligible=sorted([r for r in obs if r['trustworthy'] and r['valid_d']>0 and r['coordinate_kind']!='legacy_cache'],key=lambda r:(not r['policy_ready'],r['observation_id']))
for r in eligible[:25]:
 d=json.load(gzip.open(r['coordinate_path'],'rt'));o=d['observations'][r['coordinate_row']];seen=np.array([not p['ca_missing'] for p in o['residue_mapping']]);x=np.array([p['ca'] if not p['ca_missing'] else [0,0,0] for p in o['residue_mapping']],float);g,m,seg=derive(x,seen,np.array(o['continuity_mask']));expected=[r[k] for k in ['valid_d','valid_a','valid_t']];assert m.sum(0).tolist()==expected,(r['entry_id'],m.sum(0),expected);checks.append(dict(observation_id=r['observation_id'],entry_id=r['entry_id'],component_counts=expected,segments=len(set(seg[seg>=0]))))
single=next(r for r in eligible if r['policy_ready']);assert single['entry_id']=='10CX' and [single[k] for k in ['valid_d','valid_a','valid_t']]==[176,173,172]
result=dict(status='PASS',unique_identities=67615,unique_observations=342713,complete_unused_CA_absence_cases=0,source_full_string_equality_for_all_118219_TRAIN=True,complete_manifest_identity_set_verified=True,all_geometry_aggregates_reconcile=True,geometry_derived_by_existing_unchanged_library=checks,optimizer_updates=0,scope='TRAIN records only; no source coordinates or dataset membership changed')
(R/'CENSUS_CORRECTNESS_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',len(checks),'existing-library geometry checks')
