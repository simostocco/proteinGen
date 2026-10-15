import importlib.util
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[2]/'scripts/e016_v3/mmseqs_stream.py';s=importlib.util.spec_from_file_location('native_stream_reference',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
@pytest.mark.parametrize('body',[b'',b'3\t22\t-5\n',b'3\t22\t-5\n'*1000])
def test_native_record_preserves_exact_bytes(tmp_path,body):
 base=tmp_path/'NATIVE_FORMAT_TEST_ONLY';w=m.Writer(base);w.add(4,body);w.close();r=m.Reader(base);assert list(r.entries())==[(4,body)];r.close()
def test_invalid_compression_marker_fails_closed(tmp_path):
 base=tmp_path/'BAD_NATIVE_FORMAT_TEST_ONLY';w=m.Writer(base);w.add(4,b'3\t22\t-5\n');w.close()
 with base.open('r+b') as f:f.seek(-1,2);f.write(b'\x01')
 r=m.Reader(base)
 try:
  with pytest.raises(AssertionError):list(r.entries())
 finally:r.close()
def test_duplicate_native_query_keys_fail_closed(tmp_path):
 base=tmp_path/'DUPLICATE_QUERY_TEST_ONLY';w=m.Writer(base);w.add(4,b'3\t22\t-5\n');w.close();idx=Path(str(base)+'.index');idx.write_text(idx.read_text()*2)
 with pytest.raises(AssertionError):m.Reader(base)
