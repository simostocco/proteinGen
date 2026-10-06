import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("preflight", ROOT / "scripts/prepare_e012_large_sequence_corpus.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
REPORT = ROOT / "reports/experiments/E012_causal_rope_sequence/data_v2_uniref50_2026_03"

class PreflightTests(unittest.TestCase):
    def test_official_pinned_metadata(self):
        self.assertTrue(module.verify_metadata(REPORT)["release_note_official_md5_verified"])

    def test_tampered_note_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["RELEASE.metalink", "uniref50.release_note"]:
                (root / name).write_bytes((REPORT / name).read_bytes())
            with (root / "uniref50.release_note").open("ab") as f:
                f.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                module.verify_metadata(root)

    def test_wrong_release_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "RELEASE.metalink").write_text((REPORT / "RELEASE.metalink").read_text().replace("2026_03", "2026_02"))
            with self.assertRaisesRegex(ValueError, "wrong release"):
                module.verify_metadata(root)

    def test_disk_fail_closed_and_margin_is_separate(self):
        peak = module.disk_gate(0)["estimated_peak_bytes"]
        self.assertFalse(module.disk_gate(peak - 1)["download_authorized_by_gate"])
        self.assertTrue(module.disk_gate(peak)["download_authorized_by_gate"])
        self.assertFalse(module.disk_gate(peak)["preferred_margin_pass"])
        self.assertTrue(module.disk_gate((peak * 5 + 1) // 2)["preferred_margin_pass"])

    def test_budget_sum(self):
        gate = module.disk_gate(0)
        self.assertEqual(sum(gate["components_bytes"].values()), gate["estimated_peak_bytes"])
