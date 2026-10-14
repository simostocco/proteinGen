"""Fail-closed full-polymer admission regressions; no source or training side effects."""
import importlib.util,hashlib,copy
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[2]/'scripts/e016_v3/build_mapped_identity_tables.py'
spec=importlib.util.spec_from_file_location('identity_admission',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
SEQ='ACDEFGHIKLMNPQRSTVWYA'
def entity():return dict(canonical_full_sequence=SEQ,full_length=len(SEQ),full_sha256=hashlib.sha256(SEQ.encode()).hexdigest(),native_monomers=['ALA']*len(SEQ),status='VERIFIED',lm_eligible=True,method_allowed=True)
def test_valid_complete_deposited_sequence():assert m.check_entity(entity())==SEQ
@pytest.mark.parametrize('field,value',[('full_length',19),('full_sha256','0'*64),('canonical_full_sequence','X'+SEQ[1:]),('native_monomers',[]),('status','QUARANTINE'),('method_allowed',False)])
def test_contradiction_not_silently_admitted(field,value):
 e=entity();e[field]=value
 with pytest.raises(AssertionError):m.check_entity(e)
