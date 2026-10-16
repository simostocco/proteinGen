"""Six read-only residue-pair checks against five existing original mmCIFs."""
import json
from pathlib import Path
import numpy as np
import gemmi
import real_pdb_pilot as m
from protein_joint_sac.data.certified_unified import file_sha
flags=json.load(open(m.R/'priors/COORDINATE_EXTREMES.json'))['examples'];out=[];sources={}
for row in flags:
 path=Path('/mnt/d/Simone/proteinGen/data/full/raw/mmcif')/(row['entry_id'].lower()+'.cif.gz')
 if not path.exists():out.append({'entry_id':row['entry_id'],'status':'RAW_SOURCE_UNAVAILABLE'});continue
 if row['entry_id'] not in sources:
  sources[row['entry_id']]={'path':str(path),'sha256':file_sha(path),'block':gemmi.cif.read(str(path)).sole_block()}
 src=sources[row['entry_id']];b=src['block'];tags=['label_asym_id','auth_asym_id','pdbx_PDB_model_num','label_seq_id','auth_seq_id','label_atom_id','label_alt_id','Cartn_x','Cartn_y','Cartn_z'];table=b.find('_atom_site.',tags);ca=[[],[]];atoms=[{},{}]
 for atom in table:
  if atom[0]!=row['label_chain'] or atom[2]!=str(row['model']):continue
  for j,pos in enumerate(row['positions_1based']):
   if atom[3]==str(pos):
    xyz=np.asarray([float(atom[k]) for k in [7,8,9]])
    if atom[5]=='CA':ca[j].append({'xyz':xyz.tolist(),'auth_seq_id':atom[4],'altloc':atom[6],'label_seq_id':atom[3]})
    if atom[5] in ['C','N']:atoms[j].setdefault(atom[5],[]).append(xyz)
 errors=[min([float(np.linalg.norm(np.asarray(a['xyz'])-row['coordinates'][j])) for a in ca[j]],default=None) for j in [0,1]]
 peptide=[float(np.linalg.norm(c-n)) for c in atoms[0].get('C',[]) for n in atoms[1].get('N',[])]
 status='RAW_COORDINATES_CONFIRMED' if all(e is not None and e<1e-6 for e in errors) else 'MAPPING_OR_REVISION_REQUIRES_REVIEW'
 out.append({'entry_id':row['entry_id'],'positions_1based':row['positions_1based'],'cached_CA_distance_A':row['distance_A'],'raw_path':src['path'],'raw_sha256':src['sha256'],'raw_sha_matches_observation_source_sha':src['sha256']==row['source_sha'],'source_sha_declared':row['source_sha'],'CA_coordinate_error_A':errors,'native_candidates':ca,'adjacent_C_N_distances_A':peptide,'status':status})
m.write(m.R/'priors/RAW_EXTREME_SOURCE_CHECK.json',{'status':'PASS_RAW_EXTREMES_CONFIRMED' if all(r['status']=='RAW_COORDINATES_CONFIRMED' for r in out) else 'REVIEW_REQUIRED','raw_files_read':len(sources),'pairs':out,'new_exclusions':0,'interpretation':'C-N continuity and complete mapping do not guarantee physically plausible CA geometry; no cutoff or correction applied'})
print([(r['entry_id'],r['status']) for r in out])
