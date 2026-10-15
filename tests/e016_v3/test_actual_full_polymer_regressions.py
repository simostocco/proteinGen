"""Bounded real-source regressions; fixtures are immutable D:-resident parsed CIFs."""
import os,gzip,json
from pathlib import Path
import pytest
R=Path(os.environ['E016_ARTIFACT_ROOT'])/'mapping/regression_sources'
@pytest.mark.parametrize('entry,auth,length,nmiss,cmiss',[('3FM1','A',331,0,12),('2QMI','A',447,0,0),('7Z53','W',59,0,3),('9QEX','E',60,0,4),('6ZK9','b',124,28,1),('6ZKC','b',124,28,1)])
def test_real_full_polymer_labels_and_terminal_masks(entry,auth,length,nmiss,cmiss):
 d=json.load(gzip.open(R/(entry+'.json.gz'),'rt'));o=next(o for o in d['observations'] if auth in o['auth_asym_ids']);mapping=o['residue_mapping'];seq=o['canonical_full_sequence']
 assert o['status']=='VERIFIED' and len(seq)==length and len(mapping)==length
 assert [r['label_seq_id'] for r in mapping]==list(range(1,length+1))
 assert o['n_terminal_missing']==nmiss and o['c_terminal_missing']==cmiss
 assert o['missing_ca_positions']==[r['label_seq_id'] for r in mapping if r['ca_missing']]
 assert o['observed_ca_sequence']==''.join(seq[r['label_seq_id']-1] for r in mapping if not r['ca_missing'])
 for r in mapping:
  assert (r['ca'] is None)==bool(r['ca_missing'])
 if entry=='9QEX':assert o['label_asym_id']=='C' and o['auth_asym_ids']==['E']
 if entry in ['6ZK9','6ZKC']:assert o['observed_label_seq_ids'][0]==29 and len(o['observed_ca_sequence'])==95

def test_duplicate_full_polymer_preserves_distinct_experimental_sources():
 ds=[json.load(gzip.open(R/(entry+'.json.gz'),'rt')) for entry in ['6ZK9','6ZKC']]
 obs=[next(o for o in d['observations'] if 'b' in o['auth_asym_ids']) for d in ds]
 assert obs[0]['full_sha256']==obs[1]['full_sha256']
 assert ds[0]['source_sha256']!=ds[1]['source_sha256']
 assert obs[0]['observation_id']!=obs[1]['observation_id']
