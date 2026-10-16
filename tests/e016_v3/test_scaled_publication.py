import pytest
from scaled_finalize import write_report

def test_markdown_written_as_text(tmp_path):
 p=tmp_path/'report.md';text='# Report\n\nMeasured values.\n';write_report(p,text)
 assert p.read_text()==text
 assert not p.with_name('report.md.part').exists()

def test_markdown_rejects_nontext(tmp_path):
 with pytest.raises(TypeError):write_report(tmp_path/'report.md',{'wrong':'type'})
