from pathlib import Path
import json
import pyarrow.parquet as pq
R=Path('/mnt/d/Simone/proteinGen_data/e017_v9_property_steering_20261023_v1');rows=json.loads((R/'manifests/TRAIN_REFERENCE.json').read_text())['records'];r=next(x for x in rows if x['source']['coordinate_kind']=='legacy_cache');s=r['source'];pf=pq.ParquetFile(s['coordinate_path']);i=int(s['coordinate_row']);offset=0
for k in range(pf.num_row_groups):
 n=pf.metadata.row_group(k).num_rows
 if i<offset+n:
  names=['label_asym_id','auth_asym_id','model_id','residue_mappings','selected_calpha_coordinates','residue_names','atom_label_sequence_ids','chain_mapping_status','ambiguous_altloc_position_count'];table=pf.read_row_group(k,columns=names,use_threads=False);row=table.slice(i-offset,1).to_pylist()[0];print('LEGACY_METADATA',{z:row[z] for z in ['label_asym_id','auth_asym_id','model_id','chain_mapping_status','ambiguous_altloc_position_count']});print('RESIDUEMAPPING',str(json.loads(row['residue_mappings']))[:2400]);print('LENGTHS',{k:len(json.loads(row[k])) for k in ['residue_names','atom_label_sequence_ids','selected_calpha_coordinates']});print('LABELS',json.loads(row['atom_label_sequence_ids'])[:8]);print('REC',len(r['sequence']),r['offset']);print('NAMES',json.loads(row['residue_names'])[:3]);print('CA_FIRST',json.loads(row['selected_calpha_coordinates'])[:2]);break
 offset+=n
for r in rows[:8]:
 s=r['source'];p=Path('/mnt/d/Simone/proteinGen/data/full/raw/mmcif')/(s['entry_id'].lower()+'.cif.gz');print('RAW',s['entry_id'],p.exists(),__import__('hashlib').sha256(p.read_bytes()).hexdigest()==s['source_sha'] if p.exists() else None)
