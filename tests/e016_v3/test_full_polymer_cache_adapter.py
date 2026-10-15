"""Exercise actual adapter proof against actual cached source evidence, without opening its DB."""
import ast,json,copy,sys
from pathlib import Path
import pytest
import pyarrow.parquet as pq
R=Path('/mnt/d/Simone/proteinGen_data/e016_gpu_unified_pdb_20261015_v1')
sys.path.insert(0,'/mnt/d/Simone/proteinGen_data/thesis_pdb_sequence_audit_v2/reproduce')
from core import array,AA3,PARENTS,cached_canonical
path=R/'source/scripts/e016_v3/build_unified_catalogue.py';tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_full_polymer');scope=dict(array=array,AA3=AA3,PARENTS=PARENTS)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope);proof=scope['validate_full_polymer']
C=Path('/mnt/d/Users/Simone Stocco/proteinGen_audits/sequence_data_readiness_raw_full_compact_v1/tables/protein_chain_models/part-000000.parquet')
def actual():return next(pq.ParquetFile(C).iter_batches(batch_size=1)).to_pylist()[0]
def test_actual_numeric_string_labels_are_full_polymer_evidence():
 r=actual();seq=cached_canonical(r['entity_poly_sequence']);a=proof(r,seq)
 assert len(seq)==154 and a[0]==list(range(1,155))
 r['label_sequence_ids']=json.dumps(a[0]);assert proof(r,seq)==a
@pytest.mark.parametrize('corruption',['order','monomer','chirality'])
def test_unambiguous_source_contradictions_fail_closed(corruption):
 r=actual();seq=cached_canonical(r['entity_poly_sequence'])
 if corruption=='order':
  x=array(r['label_sequence_ids']);x[:2]=x[1::-1];r['label_sequence_ids']=json.dumps(x)
 elif corruption=='monomer':
  x=array(r['residue_names']);x[0]='SEP';r['residue_names']=json.dumps(x)
 else:r['polymer_type']="'polypeptide(D)'"
 with pytest.raises(ValueError):proof(r,seq)
def test_missing_coordinates_do_not_change_full_entity_proof():
 r=actual();seq=cached_canonical(r['entity_poly_sequence']);expected=proof(r,seq)
 r['missing_calpha_mask']=json.dumps([True]*154);r['selected_calpha_coordinates']=json.dumps([None]*154)
 assert proof(r,seq)==expected
