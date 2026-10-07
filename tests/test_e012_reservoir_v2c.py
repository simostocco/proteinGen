"""All test data/scratch is created under the D: V2C root."""
from pathlib import Path
from contextlib import closing
import hashlib, importlib.util, sqlite3, tempfile, unittest
import numpy as np

spec=importlib.util.spec_from_file_location('v2c',Path(__file__).resolve().parents[1]/'scripts/e012_reservoir_v2c.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ROOT=m.configure_storage(Path('D:/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization') if __import__('os').name=='nt' else Path('/mnt/d/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization'))

class V2CTests(unittest.TestCase):
    def test_export_payload_stream_decodes_exact_sequences_and_metadata(self):
        script=Path(__file__).resolve().parents[1]/'scripts/union_e012_reservoir_v2c.py'
        spec=importlib.util.spec_from_file_location('union',script)
        union=importlib.util.module_from_spec(spec);spec.loader.exec_module(union)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            path=Path(tmp)/'payload.bin';seq=b'ACDEFGHIKLMNPQRSTVWY';sid=b'UniRef50_test'
            h=hashlib.sha256(seq).digest();p=m.priority(seq.decode())
            import struct
            body=union.HEADER.pack(31,h,p,3,0,1,len(sid),len(seq))+sid+seq
            path.write_bytes(struct.pack('<I',len(body))+body)
            rows=list(union.base_rows(path));self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['seq'],seq.decode());self.assertEqual(rows[0]['multiplicity'],3)
            self.assertEqual(rows[0]['source_id'],sid.decode());self.assertEqual(rows[0]['h'],h.hex())
            path.write_bytes(struct.pack('<I',len(body))+body[:-1])
            with self.assertRaises(AssertionError):list(union.base_rows(path))

    def test_generation_outputs_survive_commit_before_mirror_crash(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            root=Path(tmp);first=root/'generation_1.bin';first.write_bytes(b'first durable generation')
            with closing(m.open_delta(root/'delta.sqlite')) as db:
                one=m.commit_continuation(db,root,{'raw':100},{'seed':12014},[first])
                mirror=(root/'continuation_checkpoint.json').read_bytes()
                second=root/'generation_2.bin';second.write_bytes(b'second durable generation')
                def crash():raise RuntimeError('mirror interrupted')
                with self.assertRaises(RuntimeError):
                    m.commit_continuation(db,root,{'raw':200},{'seed':12014},[second],crash,one['outputs'])
                self.assertEqual((root/'continuation_checkpoint.json').read_bytes(),mirror)
            with closing(m.open_delta(root/'delta.sqlite')) as db:
                recovered=m.resume_continuation(db,{'seed':12014})
                self.assertEqual(recovered['state']['raw'],200)
                self.assertEqual(len(recovered['outputs']),2)
                self.assertEqual(first.read_bytes(),b'first durable generation')
                with self.assertRaisesRegex(AssertionError,'overwrite'):
                    m.commit_continuation(db,root,{'raw':300},{'seed':12014},[first],previous_outputs=recovered['outputs'])

    def test_commit_before_json_resume_and_delta_bloom_reseed(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            root=Path(tmp);out=root/'data.bin';out.write_bytes(b'sealed output')
            h=b'Z'*32
            with closing(m.open_delta(root/'delta.sqlite')) as db:
                db.execute('INSERT INTO exact_hashes VALUES(?,?)',(h,11))
                def crash():raise RuntimeError('after committed SQL, before JSON')
                with self.assertRaisesRegex(RuntimeError,'before JSON'):
                    m.commit_continuation(db,root,{'raw':100,'offset':11}, {'seed':12014},[out],crash)
            self.assertFalse((root/'continuation_checkpoint.json').exists())
            with closing(m.open_delta(root/'delta.sqlite')) as db:
                recovered=m.resume_continuation(db,{'seed':12014})
                self.assertEqual(recovered['state']['raw'],100)
                bloom=m.Bloom(bits=1024);self.assertFalse(bloom.possible(h))
                m.seed_committed_delta(bloom,db);self.assertTrue(bloom.possible(h))
                out.write_bytes(b'corrupt output')
                with self.assertRaises(AssertionError):m.resume_continuation(db,{'seed':12014})

    def test_vector_seed_matches_runtime_hashing_without_false_negatives(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            p=Path(tmp)/'seen.bin'
            hashes=[hashlib.sha256(str(i).encode()).digest() for i in range(10000)]
            values=np.array([(h,i) for i,h in enumerate(hashes)],dtype=m.SEEN_DTYPE)
            values.tofile(p)
            bloom=m.Bloom(bits=1<<18)
            bloom.seed_file(p)
            self.assertTrue(all(bloom.possible(h) for h in hashes))
            extra=b'\0'*32;bloom.add(extra);self.assertTrue(bloom.possible(extra))

    def test_false_positive_collision_and_current_interval_exactness(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            root=Path(tmp)
            with closing(m.open_delta(root/'base.sqlite')) as base,closing(m.open_delta(root/'delta.sqlite')) as delta:
                shared=b'\0'*32
                base.execute('INSERT INTO exact_hashes VALUES(?,?)',(shared,1));base.commit()
                sequences={1:'A'*20,2:'C'*20,3:'D'*20}
                bloom=m.Bloom(bits=1024);bloom.data[:]=255
                lookup=m.ExactMembership(base,delta,bloom,lambda h,off:sequences[off])
                self.assertEqual(lookup.find(shared,'A'*20),1)
                self.assertIsNone(lookup.find(shared,'C'*20))
                lookup.add(shared,2,'C'*20)
                q=lookup.queries
                self.assertEqual(lookup.find(shared,'C'*20),2)
                self.assertEqual(lookup.queries,q)
                self.assertIsNone(lookup.find(shared,'D'*20))
                lookup.add(shared,3,'D'*20);lookup.commit_hashes();delta.commit()
                resumed=m.ExactMembership(base,delta,bloom,lambda h,off:sequences[off])
                self.assertEqual(resumed.find(shared,'C'*20),2)
                self.assertEqual(resumed.find(shared,'D'*20),3)

    def test_definite_negative_performs_zero_database_lookups(self):
        class Forbidden:
            def execute(self,*args):raise AssertionError('Bloom negative queried SQLite')
        b=m.Bloom(bits=1024)
        e=m.ExactMembership(Forbidden(),Forbidden(),b,lambda *a:None)
        self.assertIsNone(e.find(b'X'*32,'A'*20));self.assertEqual(e.queries,0)

    def test_abandoned_hash_cache_cannot_create_false_duplicates(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            root=Path(tmp)
            with closing(m.open_delta(root/'base.sqlite')) as base,closing(m.open_delta(root/'delta.sqlite')) as delta:
                h=b'A'*32;b=m.Bloom(bits=1024);b.add(h)
                delta.execute('INSERT INTO exact_hashes VALUES(?,?)',(h,99));delta.rollback()
                e=m.ExactMembership(base,delta,b,lambda *a:None)
                self.assertIsNone(e.find(h,'A'*20))

    def test_tiered_pool_matches_full_priority_sort_across_intervals(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            root=Path(tmp)
            def row(n,off):return (n.to_bytes(32,'big'),b'\0'*32,off,off,20)
            base=np.array([row(1,1),row(4,4),row(5,5)],dtype=m.KEY_DTYPE)
            path=root/'base.bin';base.tofile(path)
            pool=m.TieredPool(path,root,cap=5)
            all_rows=list(base)
            for batch in [[row(6,6)],[row(9,9)],[row(2,2),row(4,40),row(8,8)]]:
                for value in batch:
                    key=(value[0],value[1],value[2])
                    if pool.accepts(*key):pool.add(*value)
                    all_rows.append(np.array([value],dtype=m.KEY_DTYPE)[0])
                pool.select_interval()
                expected=np.array(all_rows,dtype=m.KEY_DTYPE);expected.sort(order=['p','h','off']);expected=expected[:5]
                actual=pool.selected()
                self.assertEqual([m.key_tuple(x) for x in actual],[m.key_tuple(x) for x in expected])
                if len(actual)<5:self.assertIsNone(pool.cutoff)
            pool.close()

    def test_rejects_non_d_artifact_paths(self):
        with self.assertRaises(AssertionError):m.require_d(Path('C:/unsafe') if __import__('os').name=='nt' else Path('/tmp/unsafe'))

if __name__=='__main__':
    import time,sys
    start=time.perf_counter()
    with (ROOT/'logs/v2c_tests.log').open('a',encoding='utf-8') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(V2CTests))
    summary={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'passed':result.wasSuccessful(),'wall_seconds':time.perf_counter()-start,
        'test_source_sha256':m.sha_file(__file__),'core_source_sha256':m.sha_file(Path(__file__).resolve().parents[1]/'scripts/e012_reservoir_v2c.py'),
        'fixtures_and_logs_root':str(ROOT),'training_launched':False}
    m.atomic_json(ROOT/'reports/TESTS.json',summary)
    print(__import__('json').dumps(summary));sys.exit(0 if result.wasSuccessful() else 1)
