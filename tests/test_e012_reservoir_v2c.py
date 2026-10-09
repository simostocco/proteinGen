"""All test data/scratch is created under the D: V2C root."""
from pathlib import Path
from contextlib import closing
import hashlib, importlib.util, sqlite3, tempfile, unittest
import numpy as np
import json, os, subprocess, types
from contextlib import nullcontext
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('v2c',Path(__file__).resolve().parents[1]/'scripts/e012_reservoir_v2c.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ROOT=m.configure_storage(Path('D:/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization') if __import__('os').name=='nt' else Path('/mnt/d/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization'))

def production_module(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).resolve().parents[1]/'scripts'/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class ProductionRecoveryTests(unittest.TestCase):
    def test_control_c_exit_cannot_publish_a_search_completion(self):
        p=production_module('run_e012_v2c_screen_production')
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            folder=Path(tmp);output=folder/'result'
            def interrupted(args,log):
                log.write_text('MMseqs parameters emitted; WSL bridge terminated\n')
                log.with_suffix('.rss.txt').write_bytes(b'')
                raise AssertionError('Command failed (3221225786) [STATUS_CONTROL_C_EXIT]')
            with patch.multiple(p,PROD=folder,STATE={}):
                with self.assertRaisesRegex(AssertionError,'3221225786'):
                    p.step(types.SimpleNamespace(run=interrupted),'search',['mmseqs','search'],[output])
            self.assertTrue((folder/'search.request.json').exists())
            self.assertFalse((folder/'search.complete.json').exists())
            self.assertFalse(output.exists())

    def test_windows_control_c_isolation(self):
        pipeline=production_module('complete_e012_v2c_pipeline')
        options=pipeline.command_process_options(windows=True)
        self.assertEqual(options['stdin'],subprocess.DEVNULL)
        self.assertEqual(options['creationflags'],0x200|0x08000000)
        self.assertTrue(pipeline.command_process_options(windows=False)['start_new_session'])

    def test_partial_search_reuses_verified_db_and_commit_restart_skips_prefix(self):
        p=production_module('run_e012_v2c_screen_production')
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            base=Path(tmp);sci=base/'scientific';prod=base/'production'
            for path in (sci/'manifests',sci/'filtered',sci/'protected',prod):path.mkdir(parents=True)
            sequences=['ACDEFGHIKLMNPQRSTVWY','Y'*20,'C'*20,'G'*20]
            rows=[(hashlib.sha256(seq.encode()).digest(),i+1,seq,0,1) for i,seq in enumerate(sequences)]
            protected=sci/'protected/protected.fasta';protected.write_text('>p_known\n'+sequences[0]+'\n')
            reservoir=sci/'filtered/reservoir.sqlite'
            with sqlite3.connect(reservoir) as db:
                db.execute('CREATE TABLE candidates(h BLOB,off INTEGER,seq TEXT,historical INTEGER,external INTEGER)')
                db.executemany('INSERT INTO candidates VALUES(?,?,?,?,?)',rows);db.commit()
            inputs={'reservoir_sha256':p.sha(reservoir),'protected_sha256':p.sha(protected),'policy':{'identity':.30,'coverage':.80,'cov_mode':0,'sensitivity':7.5},'verification':False,'batch_order':'append-table scan; independent hash-priority cutoff','query_dbtype':1}
            p.save(sci/'manifests/reservoir.complete.json',{'outputs':{str(reservoir):inputs['reservoir_sha256']}})
            cp={'inputs':inputs,'batch':1,'screened':2,'last_rowid':2,'exclusions':1,'slow_batches':0}
            ex=sci/'manifests/exclusions.sqlite'
            with sqlite3.connect(ex) as db:
                db.execute('CREATE TABLE removed(h BLOB,off INTEGER,protected TEXT,identity REAL,PRIMARY KEY(h,off)) WITHOUT ROWID')
                db.execute('CREATE TABLE batches(name TEXT,batch INTEGER,n INTEGER,marker TEXT,PRIMARY KEY(name,batch))')
                db.execute('CREATE TABLE production_progress(k TEXT PRIMARY KEY,v TEXT)')
                db.execute('INSERT INTO removed VALUES(?,?,?,?)',(rows[0][0],1,'p_known',1.0))
                db.execute('INSERT INTO batches VALUES(?,?,?,?)',('screen',1,2,'committed-original'))
                db.execute('INSERT INTO production_progress VALUES(?,?)',('checkpoint',json.dumps(cp)));db.commit()
            original=sci/'manifests/screen-batch-0001.json';original.write_bytes(b'original committed certificate')
            folder=sci/'mmseqs/screen_production/batch-0002';gen=folder/'interrupted';gen.mkdir(parents=True)
            fasta=gen/'query.fasta';fasta.write_text(''.join(f'>s_{h.hex()}_{off}\n{seq}\n' for h,off,seq,*_ in rows[2:]))
            query=gen/'query_db';query.write_bytes(b'verified persistent candidate DB')
            Path(str(query)+'.dbtype').write_bytes(b'\0\0\0\0')
            p.save(gen/'createdb.complete.json',{'outputs':{p.canonical(query):p.sha(query)}})
            (gen/'search.request.json').write_text('{}');(gen/'search.rss.txt').write_bytes(b'')
            row_hash=hashlib.sha256(b''.join(h+str(off).encode()+b'\0' for h,off,*_ in rows[2:])).hexdigest()
            p.save(folder/'active_job.json',{'generation':p.canonical(gen),'query_db':p.canonical(query),'input_rows_sha256':row_hash,'query_sha256':p.sha(fasta)})
            self.assertEqual(p.reusable_job(folder,row_hash),(None,query))
            original_hash=p.sha(original);db_hash=p.sha(query);calls=[]
            def connect(path):
                db=sqlite3.connect(path);db.execute('PRAGMA synchronous=FULL');db.execute('PRAGMA temp_store=FILE');return db
            legacy=types.SimpleNamespace(json=json,digest=p.sha,connection=connect,save=p.save,now=lambda:'bounded-fixture')
            def run(command,log):
                calls.append(command[1]);log.write_text('bounded simulated transport\n')
                if command[1]=='search':
                    self.assertEqual(Path(command[2]),query);Path(command[4]).write_bytes(b'complete result DB')
                elif command[1]=='convertalis':
                    h,off,*_=rows[2];Path(command[5]).write_text(f's_{h.hex()}_{off}\tp_known\t0.500\t1.000\t1.000\t1E-20\n')
                else:raise AssertionError('A committed work unit or reusable DB was recomputed')
                log.with_suffix('.rss.txt').write_text('Exit status: 0\n')
            def crash_after_commit(*args):raise RuntimeError('crash after durable batch commit')
            b=types.SimpleNamespace(reservoir=reservoir,protected=protected,report=sci/'reports',marker=lambda name:sci/'manifests'/(name+'.complete.json'),completed=lambda *args:False,
                stage=lambda *args:nullcontext(),cleanup=crash_after_commit,complete=lambda *args,**kwargs:None,run=run,
                search_command=lambda fa,out,tmp:['mmseqs','easy-search',fa,'protected_db',out,tmp,'--threads','16','--format-output','query,target,fident,qcov,tcov,evalue'])
            with patch.multiple(p,ROOT=base,SCI=sci,PROD=prod,TOTAL=4,SIZE=2,STATE={}):
                with self.assertRaisesRegex(RuntimeError,'durable batch'):p.screen(legacy,b)
                self.assertEqual(p.durable(),{'batch':2,'screened':4,'exclusions':2})
                before_calls=list(calls);cert=sci/'manifests/screen-batch-0002.json';cert_hash=p.sha(cert)
                p.screen(legacy,b)
                self.assertEqual(calls,before_calls);self.assertEqual(p.sha(cert),cert_hash)
                self.assertEqual(p.sha(original),original_hash);self.assertEqual(p.sha(query),db_hash)
                self.assertEqual(calls,['search','convertalis'])
                self.assertTrue(gen.exists(),'Interrupted evidence remains preserved before production cleanup')

    def test_bounded_real_frozen_relation_replay(self):
        p=production_module('run_e012_v2c_screen_production')
        first=ROOT/'screen_optimization'
        expected=json.loads((first/'equivalence.expected-original-order.json').read_text())
        sequences=dict(p.rows_from_fasta(first/'equivalence.fasta'))
        hit_ids=list(dict.fromkeys(line.split('\t')[0] for line in expected))[:6]
        nonhits=[sid for sid in sequences if sid not in {line.split('\t')[0] for line in expected}][:6]
        selected=set(hit_ids+nonhits)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            folder=Path(tmp);fasta=folder/'query.fasta';output=folder/'hits.tsv'
            fasta.write_text(''.join(f'>{sid}\n{sequences[sid]}\n' for sid in hit_ids+nonhits))
            command=['mmseqs','easy-search',str(fasta),str(ROOT.parent/'mmseqs/protected_index/db'),str(output),str(folder/'mmseqs_tmp'),
                '--min-seq-id','0.30','-c','0.80','--cov-mode','0','-s','7.5','--seq-id-mode','0','--alignment-mode','3','--threads','16',
                '--split-memory-limit','4G','--max-seqs','1000','--remove-tmp-files','0','--format-output','query,target,fident,qcov,tcov,evalue']
            with (folder/'mmseqs.log').open('w') as log:subprocess.run(command,cwd=folder,stdout=log,stderr=log,check=True)
            actual=output.read_text().splitlines();reference=[line for line in expected if line.split('\t')[0] in selected]
            self.assertEqual(set(actual),set(reference))
            rows=[(bytes.fromhex(sid.split('_')[1]),int(sid.split('_')[2]),sequences[sid]) for sid in hit_ids+nonhits]
            protected={seq:sid for sid,seq in p.rows_from_fasta(p.SCI/'protected/protected.fasta')}
            self.assertEqual(p.decisions(rows,actual,protected),p.decisions(rows,reference,protected))

class V2CTests(unittest.TestCase):
    def test_array_baseline_retains_collisions_and_zero_bytes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            root=Path(tmp);h=b'\0'*32;other=b'A'+b'\0'*31
            path=root/'sorted.bin';np.array([(h,1),(h,2),(other,3)],dtype=m.SEEN_DTYPE).tofile(path)
            base=m.ArrayBaseline(path)
            with closing(m.open_delta(root/'delta.sqlite')) as delta:
                bloom=m.Bloom(bits=1024);bloom.data[:]=255
                seqs={1:'A'*20,2:'C'*20,3:'D'*20}
                lookup=m.ExactMembership(base,delta,bloom,lambda _,off:seqs[off])
                self.assertEqual(lookup.find(h,'A'*20),1);self.assertEqual(lookup.find(h,'C'*20),2)
                self.assertIsNone(lookup.find(h,'E'*20));self.assertEqual(lookup.find(other,'D'*20),3)
                self.assertEqual(lookup.array_queries,4)
            base.close()
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
