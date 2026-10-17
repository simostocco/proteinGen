import hashlib,json
import pytest
from partial_execute import published_receipt

def test_completed_publication_is_verified_without_reexecution(tmp_path):
 p=tmp_path/'artifact';p.write_text('preserved');(tmp_path/'ACTIVE_CONTROLLER.json').write_text(json.dumps({'phase':'COMPLETED'}));m={'artifact_sha256':{'artifact':hashlib.sha256(p.read_bytes()).hexdigest()},'selected_checkpoints':{}};(tmp_path/'PROVENANCE_SHA256.json').write_text(json.dumps(m));assert published_receipt(tmp_path)
 p.write_text('corrupt')
 with pytest.raises(ValueError,match='artifact differs'):published_receipt(tmp_path)

def test_incomplete_finalizer_receipt_is_not_called_complete(tmp_path):
 (tmp_path/'ACTIVE_CONTROLLER.json').write_text(json.dumps({'phase':'COMPLETED'}));(tmp_path/'PROVENANCE_SHA256.json').write_text(json.dumps({'artifacts':{}}));assert not published_receipt(tmp_path)
