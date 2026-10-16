"""Descriptive TRAIN coordinate-extreme audit. No admission/filter changes."""
import json,collections,time
import numpy as np
import real_pdb_pilot as m
from protein_joint_sac.data.accepted_supervised import CoordinateReader
from protein_joint_sac.data.pdb_split_proposal import PDBSplitProposal
start=time.monotonic();p=PDBSplitProposal(m.P);rows=list(p.iter_joint('TRAIN'));reader=CoordinateReader(p.external);flags=[];pairs=0;low=0;high=0;minimum=None;maximum=None
for r,x in reader.stream(rows):
 d=np.linalg.norm(np.diff(x,axis=0),axis=1);pairs+=len(d)
 for i in np.where((d<2.)|(d>5.))[0]:
  seq=p.sequences[r['sequence_hash']]['canonical_full_sequence'];v={'observation_id':r['observation_id'],'sequence_hash':r['sequence_hash'],'entry_id':r['entry_id'],'label_chain':r['label_chain'],'auth_chain':r['auth_chain'],'model':r['model'],'positions_1based':[int(i+1),int(i+2)],'residues':seq[i:i+2],'distance_A':float(d[i]),'coordinates':x[i:i+2].tolist(),'coordinate_path':r['coordinate_path'],'coordinate_row':r['coordinate_row'],'coordinate_kind':r['coordinate_kind'],'source_sha':r['source_sha'],'metadata':json.loads(r['metadata'])};flags.append(v);low+=int(d[i]<2);high+=int(d[i]>5)
  if minimum is None or v['distance_A']<minimum['distance_A']:minimum=v
  if maximum is None or v['distance_A']>maximum['distance_A']:maximum=v
m.write(m.R/'priors/COORDINATE_EXTREMES.json',{'status':'DESCRIPTIVE_TRAIN_ONLY_NOT_NEW_FILTER','thresholds':'d<2 A or d>5 A are conspicuous audit triggers, not universal support or admission cutoffs','observations_checked':len(rows),'adjacent_pairs_checked':pairs,'low_pairs':low,'high_pairs':high,'affected_observations':len(set(x['observation_id'] for x in flags)),'affected_identities':len(set(x['sequence_hash'] for x in flags)),'minimum_flag':minimum,'maximum_flag':maximum,'examples':flags,'elapsed_seconds':time.monotonic()-start,'source_files_verified':len(reader.checked)})
print('EXTREME_AUDIT',len(flags),'pairs',len(set(x['sequence_hash'] for x in flags)),'identities',flush=True)
