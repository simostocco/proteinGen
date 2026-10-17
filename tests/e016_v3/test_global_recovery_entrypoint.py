import hashlib,json
import pytest
from j1_global_resume import inspect
def fixture(tmp_path):
 s=tmp_path/'source';s.mkdir();p=s/'core.py';p.write_text('trusted')
 (tmp_path/'ENVIRONMENT_AND_SOURCE_RECEIPT.json').write_text(json.dumps({'critical_source_sha256':{'core.py':hashlib.sha256(b'trusted').hexdigest()}}))
 return p
def test_active_training_blocks_duplicate(tmp_path):
 fixture(tmp_path);x=inspect(tmp_path,[123]);assert not x['resume_allowed'] and x['active_training_pids']==[123]
def test_source_change_blocks_recovery(tmp_path):
 p=fixture(tmp_path);p.write_text('changed')
 with pytest.raises(ValueError,match='source differs'):inspect(tmp_path,[])
def test_completed_run_is_not_automatically_reopened(tmp_path):
 fixture(tmp_path);d=tmp_path/'runs/J1';d.mkdir(parents=True);cp=d/'saved.pt';cp.write_bytes(b'fixture');sha=hashlib.sha256(b'fixture').hexdigest()
 (d/'COMPLETED.json').write_text(json.dumps({'selected_checkpoint':str(cp),'selected_sha256':sha,'latest_checkpoint':str(cp),'latest_sha256':sha}))
 x=inspect(tmp_path,[]);assert x['already_completed'] and not x['resume_allowed']
