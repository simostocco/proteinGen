"""Corrected E012 cleanup retains the borrowed, frozen historical reservoir."""
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import patch

SPEC=importlib.util.spec_from_file_location('native_cleanup',Path(__file__).resolve().parents[1]/'scripts/run_e012_v2b_native.py')
native=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


class CorrectedCleanupTests(unittest.TestCase):
    def test_windows_inaccessible_reparse_alias_is_not_a_regular_file(self):
        module=types.SimpleNamespace(Path=Path)
        native.install_diagnostic_lf(module)
        p=module.Path('/mnt/d/clusters/pilot_tmp/latest')
        error=OSError(22,'inaccessible WSL reparse link')
        error.winerror=1920
        with patch.object(type(Path()),'is_file',side_effect=error),patch.object(type(Path()),'lstat',return_value=types.SimpleNamespace(st_file_attributes=0x400)):
            self.assertFalse(p.is_file())
        with patch.object(type(Path()),'is_file',side_effect=error),patch.object(type(Path()),'lstat',return_value=types.SimpleNamespace(st_file_attributes=0)):
            with self.assertRaises(OSError):p.is_file()

    def test_only_corrected_diagnostic_writers_force_lf(self):
        module=types.SimpleNamespace(Path=Path)
        native.install_diagnostic_lf(module)
        root=Path('/mnt/d/e012/scientific_corrected_protein_v1/clusters')
        for identity in ['identity_50','identity_30']:
            for name in ['pilot_10000.fasta','diagnostic_100000.fasta']:
                with patch.object(type(Path()),'open') as opened:
                    module.Path(root/identity/name).open('w')
                    self.assertEqual(opened.call_args.args[-1],'\n')
        for p,mode in [(root/'identity_50/pilot_10000.fasta','r'),
                       (root/'identity_50/statistics.json','w'),
                       (Path('/mnt/d/other/clusters/identity_50/pilot_10000.fasta'),'w')]:
            with patch.object(type(Path()),'open') as opened:
                module.Path(p).open(mode)
                self.assertIsNone(opened.call_args.args[-1])

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
