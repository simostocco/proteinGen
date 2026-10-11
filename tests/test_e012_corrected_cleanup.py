"""Corrected E012 cleanup retains the borrowed, frozen historical reservoir."""
import importlib.util
from pathlib import Path
import types
import unittest

SPEC=importlib.util.spec_from_file_location('native_cleanup',Path(__file__).resolve().parents[1]/'scripts/run_e012_v2b_native.py')
native=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


class CorrectedCleanupTests(unittest.TestCase):
    def fixture(self, root_name='scientific_corrected_protein_v1'):
        class Build:
            def cleanup(self, paths, certificate):
                if certificate!='certified.complete.json':
                    raise RuntimeError('invalid certificate')
                for p in paths:
                    if not p.is_relative_to(self.root):
                        raise RuntimeError('cleanup outside derivable temporary roots')
                return list(paths)
        native.install_corrected_cleanup(types.SimpleNamespace(Build=Build))
        b=Build()
        b.root=Path('/mnt/d/e012/v2c_optimization')/root_name
        b.reservoir=b.root.parent/'scientific/filtered/reservoir.sqlite'
        return b

    def test_corrected_diversity_retains_historical_reservoir(self):
        b=self.fixture()
        temporary=b.root/'tmp/shard_assignments.sqlite'
        self.assertEqual(b.cleanup([b.reservoir,temporary],'certified.complete.json'),[temporary])

    def test_unrelated_outside_path_still_rejected(self):
        b=self.fixture()
        with self.assertRaisesRegex(RuntimeError,'outside'):
            b.cleanup([b.reservoir,b.root.parent/'scientific/manifests/exclusions.sqlite'],'certified.complete.json')

    def test_certificate_guard_still_required(self):
        b=self.fixture()
        with self.assertRaisesRegex(RuntimeError,'invalid certificate'):
            b.cleanup([b.reservoir],'missing.complete.json')

    def test_other_namespaces_and_reservoirs_are_not_exempt(self):
        b=self.fixture('other')
        with self.assertRaisesRegex(RuntimeError,'outside'):
            b.cleanup([b.reservoir],'certified.complete.json')
        b=self.fixture()
        b.reservoir=b.root.parent/'other/filtered/reservoir.sqlite'
        with self.assertRaisesRegex(RuntimeError,'outside'):
            b.cleanup([b.reservoir],'certified.complete.json')
