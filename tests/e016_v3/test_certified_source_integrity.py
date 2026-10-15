import json
import pytest
from protein_joint_sac.contracts import DatasetRef
from protein_joint_sac.data.certified_unified import verify_release,file_sha

def setup_mock_release(tmp_path):
 root=tmp_path/'MOCK_RELEASE_NOT_SCIENTIFIC';root.mkdir();source=tmp_path/'mock_source';source.write_text('immutable test source')
 (root/'split_assignments.parquet').write_text('mock split bytes')
 (root/'INDEPENDENT_LEAKAGE_AUDIT.json').write_text(json.dumps({'status':'PASS','independent_screen_complete':True}))
 manifest={'status':'CERTIFIED','artifact_sha256':{n:file_sha(root/n) for n in ['split_assignments.parquet','INDEPENDENT_LEAKAGE_AUDIT.json']},'external_artifact_sha256':{str(source):file_sha(source)}}
 (root/'RELEASE_MANIFEST.json').write_text(json.dumps(manifest))
 ref=DatasetRef(str(root),file_sha(root/'RELEASE_MANIFEST.json'),file_sha(root/'split_assignments.parquet'),'CERTIFIED');return ref,source

def test_external_source_changes_fail_closed(tmp_path):
 ref,source=setup_mock_release(tmp_path);verify_release(ref);source.write_text('changed')
 with pytest.raises(ValueError,match='external source changed'):verify_release(ref)

def test_internal_manifest_or_split_changes_fail_closed(tmp_path):
 ref,source=setup_mock_release(tmp_path);(tmp_path/'MOCK_RELEASE_NOT_SCIENTIFIC/split_assignments.parquet').write_text('changed split')
 with pytest.raises(ValueError,match='provenance mismatch'):verify_release(ref)

def test_public_adapter_uses_complete_certification_gate(tmp_path):
 from protein_joint_sac.data.pdb_v3_adapter import require_certified
 ref,source=setup_mock_release(tmp_path);assert require_certified(ref)["status"]=="CERTIFIED"
 source.write_text("changed")
 with pytest.raises(ValueError,match="external source changed"):require_certified(ref)
