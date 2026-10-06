import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

SPEC=importlib.util.spec_from_file_location('v2b',Path(__file__).resolve().parents[1]/'scripts/prepare_e012_sequence_corpus_v2b.py')
m=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)

class V2BTests(unittest.TestCase):
    def test_parser_whitespace_uppercase_malformed_and_rejections(self):
        records=list(m.fasta_records(io.BytesIO(b'orphan\n>UniRef50_ok description\nacde fghik\tlmnpqrstvwy\n>\nAAAA\n>UniRef50_bad\nACDEFGHIKLMNPQRSTVWYX\n')))
        self.assertEqual(len(records),4)
        self.assertTrue(records[0][3])
        self.assertTrue(m.valid(records[1][2]))
        self.assertTrue(records[2][3])
        self.assertFalse(m.valid(records[3][2]))

    def test_length_and_alphabet_contract(self):
        self.assertFalse(m.valid('A'*19))
        self.assertTrue(m.valid('A'*20))
        self.assertTrue(m.valid('A'*500))
        self.assertFalse(m.valid('A'*501))
        for symbol in 'BJOUXZ*-012':
            self.assertFalse(m.valid('A'*20+symbol))

    def test_priority_is_frozen(self):
        seq='ACDEFGHIKLMNPQRSTVWY'
        self.assertEqual(m.priority(seq),hashlib.sha256(b'E012-V2B:12014\0'+seq.encode()).digest())

    def test_raw_checksum_rejects_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            b,_=self.fixture(root,5)
            b.marker('raw').unlink()
            expected=m.digest(b.raw,'md5')
            size=b.raw.stat().st_size
            with b.raw.open('r+b') as f:
                f.seek(25)
                byte=f.read(1)
                f.seek(25)
                f.write(bytes([byte[0]^1]))
            with patch.object(m,'RAW_BYTES',size),patch.object(m,'RAW_MD5',expected):
                with self.assertRaisesRegex(RuntimeError,'MD5 mismatch'):
                    b.verify_raw()
            self.assertFalse(b.marker('raw').exists())

    def test_policy_bidirectional_coverage(self):
        b=object.__new__(m.Build)
        b.protected=Path('/tmp/protected.fasta')
        cmd=b.search_command('query','out','tmp')
        self.assertEqual(cmd[cmd.index('--min-seq-id')+1],'0.30')
        self.assertEqual(cmd[cmd.index('-c')+1],'0.80')
        self.assertEqual(cmd[cmd.index('--cov-mode')+1],'0')
        self.assertEqual(cmd[cmd.index('-s')+1],'7.5')
        self.assertIn('--remove-tmp-files',cmd)
        self.assertNotIn('easy-cluster',cmd)

    def fixture(self, root, cap):
        import pyarrow as pa
        import pyarrow.parquet as pq
        for d in ['stats','raw','tmp','filtered','manifests','protected',m.REPORT_REL,'outputs/e012_causal_rope_sequence/pilot_v1']:
            (root/d).mkdir(parents=True,exist_ok=True)
        m.save(root/'stats/resource_plan.json',{'starting_free_bytes':1000000})
        seqs=['A'*19+c for c in 'CDEFGHIK']
        records=[('UniRef50_z',seqs[0]),('UniRef50_a',seqs[0])]+[(f'UniRef50_{i}',s) for i,s in enumerate(seqs[1:])]+[('UniRef50_invalid','X'*20),('UniRef50_long','A'*501)]
        with gzip.open(root/'raw/uniref50.fasta.gz','wt') as f:
            for sid,s in records:
                f.write(f'>{sid}\n{s}\n')
        pq.write_table(pa.Table.from_pylist([{'sample_id':'hist0','sequence':seqs[0]},
            {'sample_id':'hist1','sequence':'T'*20},{'sample_id':'hist2','sequence':'V'*20}]),root/'outputs/e012_causal_rope_sequence/pilot_v1/train.parquet')
        b=m.Build(root,root,cap=cap,target=4)
        m.save(b.marker('raw'),{'outputs':{str(b.raw):m.digest(b.raw)}})
        return b,records

    def test_bounded_reservoir_collision_verification_and_historical_union(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            b,records=self.fixture(root,cap=5)
            with patch.object(m,'RAW_COUNT',len(records)),patch.object(m,'HISTORICAL_UNIQUE',3),patch.object(m,'CHECKPOINT_EVERY',3),patch.object(m,'sequence_digest',lambda s:b'x'*32):
                b.filter_reservoir()
            stats=json.loads((root/'stats/filtering_stats.json').read_text())
            self.assertEqual(stats['external_exact_unique_encountered'],8)
            self.assertEqual(stats['exact_duplicates_removed'],1)
            self.assertEqual(stats['external_reservoir_count'],5)
            db=sqlite3.connect(b.reservoir)
            rows=db.execute('SELECT seq,historical,external FROM candidates').fetchall()
            self.assertEqual(len(rows),len({r[0] for r in rows}))
            self.assertEqual(sum(r[1] for r in rows),3)
            expected=sorted({r[1] for r in records if m.valid(r[1])},key=m.priority)[:5]
            for seq in expected:
                self.assertIn(seq,[r[0] for r in rows])
            db.close()

    def test_checkpoint_resume_does_not_recount_completed_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            b,records=self.fixture(root,cap=20)
            real_save=m.save
            crashed=False
            def injected(path,obj):
                nonlocal crashed
                real_save(path,obj)
                if Path(path).name=='filter_checkpoint.json' and not crashed:
                    crashed=True
                    raise RuntimeError('simulated interruption')
            with patch.object(m,'RAW_COUNT',len(records)),patch.object(m,'HISTORICAL_UNIQUE',3),patch.object(m,'CHECKPOINT_EVERY',3):
                with patch.object(m,'save',injected):
                    with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                        b.filter_reservoir()
                b.filter_reservoir()
            stats=json.loads((root/'stats/filtering_stats.json').read_text())
            self.assertEqual(stats['raw_count'],len(records))
            self.assertEqual(stats['external_exact_unique_encountered'],8)
            self.assertEqual(stats['exact_duplicates_removed'],1)

    def test_historical_union_commit_resume_preserves_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            b,records=self.fixture(root,cap=5)
            real_save=m.save
            crashed=False
            def injected(path,obj):
                nonlocal crashed
                if Path(path).name=='filtering_stats.json' and not crashed:
                    crashed=True
                    raise RuntimeError('simulated post-union interruption')
                real_save(path,obj)
            with patch.object(m,'RAW_COUNT',len(records)),patch.object(m,'HISTORICAL_UNIQUE',3):
                with patch.object(m,'save',injected):
                    with self.assertRaisesRegex(RuntimeError,'post-union'):
                        b.filter_reservoir()
                b.filter_reservoir()
            stats=json.loads((root/'stats/filtering_stats.json').read_text())
            self.assertEqual(stats['external_reservoir_count'],5)
            self.assertEqual(stats['historical_unique'],3)
            self.assertEqual(stats['union_count'],5+stats['historical_added_to_reservoir'])

    def test_reusable_target_index_preserves_search_results(self):
        import subprocess
        import random
        rng=random.Random(12014)
        seq=''.join(rng.choice(m.AA) for _ in range(100))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            b,_=self.fixture(root,5)
            (root/'mmseqs').mkdir()
            b.protected.write_text('>exact\n'+seq+'\n')
            query=root/'tmp/query.fasta'
            query.write_text('>query\n'+seq+'\n')
            out0=root/'tmp/hits0.tsv'
            subprocess.run(b.search_command(query,out0,root/'tmp/search0'),check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            b.target_index()
            out1=root/'tmp/hits1.tsv'
            subprocess.run(b.search_command(query,out1,root/'tmp/search1'),check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            self.assertEqual(out0.read_text(),out1.read_text())

    def test_marker_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            b,_=self.fixture(root,5)
            path=root/'tmp/output'
            path.write_text('ok')
            b.complete('fixture',{'input':'fixed'},[path])
            self.assertIsNotNone(b.completed('fixture',{'input':'fixed'}))
            path.write_text('corrupt')
            with self.assertRaisesRegex(RuntimeError,'corrupt'):
                b.completed('fixture',{'input':'fixed'})

    def test_cleanup_refuses_raw_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            b,_=self.fixture(Path(tmp),5)
            with self.assertRaisesRegex(RuntimeError,'outside'):
                b.cleanup([b.raw],b.marker('raw'))

    def test_real_batched_screen_exact_overlap_and_independent_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            b,records=self.fixture(root,cap=20)
            (root/'mmseqs').mkdir()
            b.protected.write_text('>protected_external\n'+records[0][1]+'\n>protected_historical\n'+'T'*20+'\n>unrelated\n'+m.AA*3+'\n')
            with patch.object(m,'RAW_COUNT',len(records)),patch.object(m,'HISTORICAL_UNIQUE',3):
                b.filter_reservoir()
            b.screen()
            b.screen(independent=True)
            s=json.loads((root/'stats/screen.json').read_text())
            v=json.loads((root/'stats/verify_protected.json').read_text())
            self.assertEqual(s['removed'],2)
            self.assertEqual(s['historical_removals'],2)
            self.assertEqual(s['external_removals'],1)
            self.assertTrue(v['zero_detected_violations'])
            self.assertEqual(v['candidates_screened'],4)
            # Independent verification must not mutate the immutable exclusions DB.
            self.assertIsNotNone(b.completed('screen',json.loads(b.marker('screen').read_text())['inputs']))

    def test_real_mmseqs_both_sides_coverage_boundary(self):
        import random
        import subprocess
        rng=random.Random(12014)
        seq=''.join(rng.choice(m.AA) for _ in range(100))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            query=root/'query.fasta'
            protected=root/'protected.fasta'
            query.write_text('>query\n'+seq+'\n')
            protected.write_text('>coverage80\n'+seq+'W'*25+'\n>coverage79\n'+seq+'W'*26+'\n')
            b=object.__new__(m.Build)
            b.protected=protected
            out=root/'hits.tsv'
            subprocess.run(b.search_command(query,out,root/'tmp'),check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            targets=[line.split('\t')[1] for line in out.read_text().splitlines()]
            self.assertIn('coverage80',targets)
            self.assertNotIn('coverage79',targets)

    def test_final_shards_fasta_manifest_consistency_and_checkpoint_recovery(self):
        import subprocess
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            b,records=self.fixture(root,cap=20)
            (root/'mmseqs').mkdir()
            (root/'clusters').mkdir()
            (root/'corpus').mkdir()
            b.protected.write_text('>unrelated\n'+m.AA*3+'\n')
            with patch.object(m,'RAW_COUNT',len(records)),patch.object(m,'HISTORICAL_UNIQUE',3):
                b.filter_reservoir()
            b.screen()
            b.screen(independent=True)
            real_save=m.save
            crashed=False
            def injected(path,obj):
                nonlocal crashed
                real_save(path,obj)
                if Path(path).name=='final_checkpoint.json' and not crashed:
                    crashed=True
                    raise RuntimeError('simulated shard interruption')
            original_check_output=subprocess.check_output
            def git_stub(cmd,*args,**kwargs):
                if cmd[0]=='git':
                    return '' if kwargs.get('text') else b''
                return original_check_output(cmd,*args,**kwargs)
            with patch.object(m,'MIN_FINAL',1),patch.object(m,'SHARD_COUNT',4),patch.object(m,'DIAGNOSTIC_SAMPLE_N',4),patch.object(m,'AUDIT_SAMPLE_N',2):
                with patch.object(m,'save',injected):
                    with self.assertRaisesRegex(RuntimeError,'simulated shard interruption'):
                        b.final()
                b.final()
                with patch.object(m.subprocess,'check_output',git_stub):
                    b.certify()
            cert=json.loads((root/'corpus/certification.json').read_text())
            self.assertTrue(cert['shard_exclusivity'])
            self.assertTrue(cert['manifest_index_consistency'])
            self.assertTrue(cert['representation_counts_agree'])
            self.assertTrue(cert['bos_target_shift'])
            self.assertEqual(cert['sequences'],4)

if __name__=='__main__':
    unittest.main()
