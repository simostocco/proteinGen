import gzip
from contextlib import closing
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
    def test_external_checkpoint_resume_does_not_recount_candidate_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            b,records=self.fixture(root,cap=5)
            real_save=m.save
            def crash(path,value):
                if Path(path).name=='filter_checkpoint.json' and value['raw']==6:
                    raise RuntimeError('interrupt after atomic checkpoint')
                return real_save(path,value)
            with patch.object(m,'RAW_COUNT',len(records)),patch.object(m,'HISTORICAL_UNIQUE',3),patch.object(m,'CHECKPOINT_EVERY',3):
                with patch.object(m,'save',crash):
                    with self.assertRaisesRegex(RuntimeError,'atomic checkpoint'):
                        b.filter_reservoir()
                # Previous versions lacked an explicit count; SQL progress is authoritative.
                with closing(m.connection(b.reservoir,wal=True)) as db:
                    state=json.loads(db.execute("SELECT v FROM progress WHERE k='checkpoint'").fetchone()[0])
                    state.pop('reservoir_count',None)
                    db.execute("UPDATE progress SET v=? WHERE k='checkpoint'",(json.dumps(state),))
                    db.commit()
                traces=[]
                real_connection=m.connection
                def tracked(*args,**kwargs):
                    db=real_connection(*args,**kwargs)
                    db.set_trace_callback(traces.append)
                    return db
                with patch.object(m,'connection',tracked):
                    b.filter_reservoir()
            queries=[q.lower().strip() for q in traces]
            first_insert=next(i for i,q in enumerate(queries) if q.startswith('insert into exact_hashes'))
            self.assertTrue(all(i>first_insert for i,q in enumerate(queries) if q=='select count(*) from candidates'))
            stats=json.loads((root/'stats/filtering_stats.json').read_text())
            self.assertEqual(stats['external_reservoir_count'],5)
            self.assertEqual(stats['external_exact_unique_encountered'],8)

    def test_wal_resume_preserves_committed_frames_with_an_existing_reader(self):
        from contextlib import ExitStack, closing
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            path=Path(tmp)/'resume.sqlite'
            writer=stack.enter_context(closing(m.connection(path,wal=True)))
            writer.execute('PRAGMA wal_autocheckpoint=0')
            writer.execute('CREATE TABLE population(value INTEGER)')
            writer.execute('INSERT INTO population VALUES(0)')
            writer.commit()
            reader=stack.enter_context(closing(sqlite3.connect(path)))
            reader.execute('BEGIN')
            self.assertEqual(reader.execute('SELECT count(*) FROM population').fetchone()[0],1)
            writer.execute('INSERT INTO population VALUES(1)')
            writer.commit()
            wal=Path(str(path)+'-wal')
            before=wal.read_bytes()
            resumed=stack.enter_context(closing(m.connection(path,wal=True)))
            self.assertEqual(resumed.execute('PRAGMA journal_mode').fetchone()[0],'wal')
            self.assertEqual(resumed.execute('SELECT count(*) FROM population').fetchone()[0],2)
            self.assertEqual(wal.read_bytes(),before)
            self.assertEqual(reader.execute('SELECT count(*) FROM population').fetchone()[0],1)

    def test_trim_keeps_exact_priority_prefix_and_deletes_in_row_order(self):
        with sqlite3.connect(':memory:') as db:
            db.execute('CREATE TABLE candidates(h BLOB,off INTEGER,p BLOB,seq TEXT,PRIMARY KEY(h,off))')
            db.execute('CREATE INDEX priority_index ON candidates(p,h,off,length(seq))')
            db.execute('CREATE TABLE trail(position INTEGER PRIMARY KEY,deleted_rowid INTEGER)')
            db.execute('CREATE TRIGGER deletion_order AFTER DELETE ON candidates BEGIN INSERT INTO trail(deleted_rowid) VALUES(old.rowid); END')
            for i,value in enumerate([8,3,6,1,7,2,5,4]):
                db.execute('INSERT INTO candidates VALUES(?,?,?,?)',(b"same_hash",10-i,bytes([value%3]),m.AA+'A'*i))
            expected=list(db.execute('SELECT rowid,seq FROM candidates ORDER BY p,h,off LIMIT 4'))
            all_ids={r[0] for r in db.execute('SELECT rowid FROM candidates')}
            expected_removed=sorted(all_ids-{r[0] for r in expected})
            self.assertEqual(m.trim_candidates(db,8,4),4)
            self.assertEqual(list(db.execute('SELECT rowid,seq FROM candidates ORDER BY p,h,off')),expected)
            self.assertEqual([r[0] for r in db.execute('SELECT deleted_rowid FROM trail ORDER BY position')],expected_removed)
            self.assertEqual(m.trim_candidates(db,4,4),4)
            self.assertEqual(db.execute('SELECT count(*) FROM trail').fetchone()[0],4)

    def test_sorted_cursor_preserves_empty_shard_boundaries(self):
        db=sqlite3.connect(':memory:')
        rows=m.ShardRows(db.execute("SELECT 'a',0 UNION ALL SELECT 'b',0 UNION ALL SELECT 'c',2"))
        self.assertEqual(list(rows.shard(0)),[('a',),('b',)])
        self.assertIsNone(rows.shard(1).fetchone())
        self.assertEqual(list(rows.shard(2)),[('c',)])
        self.assertIsNone(rows.shard(3).fetchone())
        db.close()

    def test_independent_screen_uses_priority_prefix_despite_append_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            b,records=self.fixture(root,cap=20)
            with patch.object(m,'RAW_COUNT',len(records)),patch.object(m,'HISTORICAL_UNIQUE',3):
                b.filter_reservoir()
            db=sqlite3.connect(b.reservoir)
            physical=db.execute('SELECT seq FROM candidates ORDER BY rowid').fetchall()
            ranked=db.execute('SELECT seq FROM candidates ORDER BY p,h,off').fetchall()
            for k in range(1,len(ranked)):
                outside=set(physical[:k])-set(ranked[:k])
                if outside:
                    break
            self.assertTrue(outside)
            b.target=k
            b.protected.write_text('>heldout_outside_selected_prefix\n'+next(iter(outside))[0]+'\n')
            ex=sqlite3.connect(root/'manifests/exclusions.sqlite')
            ex.execute('CREATE TABLE removed(h BLOB,off INTEGER,protected TEXT,identity REAL,PRIMARY KEY(h,off)) WITHOUT ROWID')
            ex.commit()
            ex.close()
            db.close()
            def empty_search(command,log):
                Path(command[4]).write_text('')
            with patch.object(b,'run',empty_search):
                b.screen(True)
            stats=json.loads((root/'stats/verify_protected.json').read_text())
            self.assertEqual(stats['candidates_screened'],k)
            self.assertTrue(stats['zero_detected_violations'])

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

    def test_attached_hash_index_checkpoint_migrates_without_recounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            b,records=self.fixture(root,cap=20)
            real_save=m.save
            def injected(path,obj):
                real_save(path,obj)
                if Path(path).name=='filter_checkpoint.json':
                    raise RuntimeError('stop at checkpoint')
            with patch.object(m,'RAW_COUNT',len(records)),patch.object(m,'HISTORICAL_UNIQUE',3),patch.object(m,'CHECKPOINT_EVERY',3):
                with patch.object(m,'save',injected):
                    with self.assertRaisesRegex(RuntimeError,'stop at checkpoint'):
                        b.filter_reservoir()
                db=sqlite3.connect(b.reservoir)
                db.execute('ATTACH DATABASE ? AS old_seen',(str(root/'tmp/exact_hash_offsets.sqlite'),))
                db.execute('CREATE TABLE old_seen.hashes(h BLOB,off INTEGER,PRIMARY KEY(h,off)) WITHOUT ROWID')
                db.execute('INSERT INTO old_seen.hashes SELECT h,off FROM exact_hashes')
                db.execute('DROP TABLE exact_hashes')
                db.commit()
                db.close()
                b.filter_reservoir()
            s=json.loads((root/'stats/filtering_stats.json').read_text())
            self.assertEqual(s['raw_count'],len(records))
            self.assertEqual(s['external_exact_unique_encountered'],8)
            self.assertEqual(s['exact_duplicates_removed'],1)

    def test_sequence_table_layout_migration_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            b,records=self.fixture(root,cap=5)
            real_save=m.save
            def injected(path,obj):
                real_save(path,obj)
                if Path(path).name=='filter_checkpoint.json':
                    raise RuntimeError('stop at checkpoint')
            with patch.object(m,'RAW_COUNT',len(records)),patch.object(m,'HISTORICAL_UNIQUE',3),patch.object(m,'CHECKPOINT_EVERY',3):
                with patch.object(m,'save',injected):
                    with self.assertRaisesRegex(RuntimeError,'stop at checkpoint'):
                        b.filter_reservoir()
                db=sqlite3.connect(b.reservoir)
                before=db.execute('SELECT * FROM candidates ORDER BY p,h,off').fetchall()
                db.execute('CREATE TABLE legacy_candidates(h BLOB, off INTEGER, p BLOB, seq TEXT, source_id TEXT, multiplicity INTEGER, historical INTEGER, external INTEGER, PRIMARY KEY(h,off)) WITHOUT ROWID')
                db.execute('INSERT INTO legacy_candidates SELECT * FROM candidates')
                db.execute('DROP TABLE candidates')
                db.execute('ALTER TABLE legacy_candidates RENAME TO candidates')
                db.execute('CREATE INDEX priority_index ON candidates(p,h,off)')
                db.commit()
                db.close()
                with patch.object(m,'save',injected):
                    with self.assertRaisesRegex(RuntimeError,'stop at checkpoint'):
                        b.filter_reservoir()
                db=sqlite3.connect(b.reservoir)
                self.assertNotIn('WITHOUT ROWID',db.execute("SELECT sql FROM sqlite_master WHERE name='candidates'").fetchone()[0])
                after=db.execute('SELECT * FROM candidates ORDER BY p,h,off').fetchall()
                self.assertTrue(all(row in after for row in before))
                db.close()
                b.filter_reservoir()
            stats=json.loads((root/'stats/filtering_stats.json').read_text())
            self.assertEqual(stats['raw_count'],len(records))
            self.assertEqual(stats['external_exact_unique_encountered'],8)
            self.assertEqual(stats['exact_duplicates_removed'],1)
            self.assertEqual(stats['external_reservoir_count'],5)

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
                with patch.object(m.subprocess,'check_output',git_stub),patch.object(m.time,'monotonic',return_value=1000.0):
                    b.certify()
            cert=json.loads((root/'corpus/certification.json').read_text())
            self.assertTrue(cert['shard_exclusivity'])
            self.assertTrue(cert['manifest_index_consistency'])
            self.assertTrue(cert['representation_counts_agree'])
            self.assertTrue(cert['bos_target_shift'])
            self.assertEqual(cert['sequences'],4)

if __name__=='__main__':
    unittest.main()
