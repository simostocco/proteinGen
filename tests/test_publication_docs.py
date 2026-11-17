"""Check links and provenance for the new public entry points."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_entrypoint_relative_links_exist():
    for relative in ("README.md", "docs/research_progression.md", "docs/reproducibility.md"):
        page = ROOT / relative
        for link in re.findall(r"\]\(([^)]+)\)", page.read_text()):
            if "://" in link or link.startswith("#"):
                continue
            assert (page.parent / link.split("#")[0]).exists(), (relative, link)


def test_original_result_reports_match_provenance_hashes():
    records = json.loads((ROOT / "docs/publication/integration_sources.json").read_text())
    for record in records:
        path = ROOT / record["path"]
        assert path.is_file()
        if "sha256" in record:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_selected_files_and_original_evidence_hashes():
    records = json.loads((ROOT / 'docs/publication/integration_sources.json').read_text())
    for record in records:
        path = ROOT / record['path']
        assert path.is_file()
        if 'sha256' in record:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
