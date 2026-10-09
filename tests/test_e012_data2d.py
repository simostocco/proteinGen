"""Bounded regressions for protein query autodetection; all fixtures on D:."""
import gzip, hashlib, importlib.util, json, os, sqlite3, subprocess, tempfile, types, unittest
from pathlib import Path
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('D:/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization') if os.name=='nt' else Path('/mnt/d/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization')
def module(name):
    spec=importlib.util.spec_from_file_location(name,REPO/'scripts'/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class Data2DTests(unittest.TestCase):
    def test_query_db_type_guard_rejects_nucleotide_missing_and_corrupt(self):
        p=module('run_e012_v2c_screen_production')
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            prefix=Path(tmp)/'db';dtype=Path(str(prefix)+'.dbtype')
            for content in (None,b'\1\0\0\0',b''):
                if content is not None:dtype.write_bytes(content)
                with self.assertRaisesRegex(RuntimeError,'DATA2-D'):p.assert_aminoacid_db(prefix)
            dtype.write_bytes(b'\0\0\0\0');p.assert_aminoacid_db(prefix)

    def test_historical_data2d_cannot_be_resumed(self):
        p=module('run_e012_v2c_screen_production')
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            root=Path(tmp);f=root/'screen_production/reports/FAILURE.json';f.parent.mkdir(parents=True);f.write_text('{"classification":"DATA2-D"}')
            before=f.read_bytes()
            with patch.multiple(p,ROOT=root,CORRECTION=False):
                with self.assertRaisesRegex(RuntimeError,'DATA2-D'):p.reject_historical_data2d()
            self.assertEqual(f.read_bytes(),before)

    def test_actual_offender_lost_by_auto_detect_and_restored_by_protein_type(self):
        diag=ROOT/'data2d_batch200_diagnostic'
        offender=json.loads((diag/'OFFENDERS.json').read_text())[0]
        self.assertEqual(hashlib.sha256(offender['sequence'].encode()).hexdigest(),offender['hash'])
        expected={x['target'] for x in json.loads((diag/'ALIGNMENTS.json').read_text()) if x['query']==offender['query']}
        self.assertTrue(expected)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            folder=Path(tmp);fa=folder/'query.fasta'
            fa.write_text(''.join(f'>poly_a_{n}\n'+('A'*(20+n))+'\n' for n in range(10))+'>'+offender['query']+'\n'+offender['sequence']+'\n')
            sets=[]
            for dtype in ('0','1'):
                out=folder/('hits-'+dtype+'.tsv')
                args=['mmseqs','easy-search',str(fa),str(ROOT.parent/'mmseqs/protected_index/db'),str(out),str(folder/('mmseqs-'+dtype)), '--dbtype',dtype,'--min-seq-id','0.30','-c','0.80','--cov-mode','0','-s','7.5','--seq-id-mode','0','--alignment-mode','3','--threads','16','--split-memory-limit','4G','--max-seqs','1000','--remove-tmp-files','0','--format-output','query,target,fident,qcov,tcov,evalue']
                if dtype=='1':args.extend(['--search-type','1'])
                with (folder/('mmseqs-'+dtype+'.log')).open('w') as log:subprocess.run(args,cwd=folder,stdout=log,stderr=log,check=True)
                sets.append({line.split('\t')[1] for line in out.read_text().splitlines() if line.split('\t')[0]==offender['query']})
            self.assertFalse(sets[0]);self.assertEqual(sets[1],expected)

    def test_parser_complement_nearest_and_frozen_boundary_rounding(self):
        p=module('run_e012_v2c_screen_production');seq='ACDEFGHIKLMNPQRSTVWY';h=hashlib.sha256(seq.encode()).digest();sid=f's_{h.hex()}_-2'
        rows=[(h,-2,seq)];lines=[f'{sid}\tp_boundary\t0.300\t0.800\t0.800\t1E-12\n',f'{sid}\tp_better\t0.500\t0.850\t0.950\t1E-20\n',f'{sid}\tp_tie\t0.500\t0.900\t0.900\t1E-20\n']
        mapping,nearest,survivors=p.decisions(rows,lines,{})
        self.assertEqual(nearest,{sid:('p_better',.5)});self.assertFalse(survivors)
        self.assertEqual(p.decisions(rows,lines,{seq:'p_exact'})[1],{sid:('p_exact',1.)})
        self.assertEqual(p.decisions(rows,[],{})[2],{sid})
        with self.assertRaisesRegex(RuntimeError,'DATA2-D'):p.decisions(rows,[f'{sid}\tp\t0.298\t0.800\t0.800\t1E-12\n'],{})

    def test_all_new_paths_resolve_on_d(self):
        p=module('run_e012_v2c_screen_production')
        self.assertTrue(p.canonical(ROOT/'scientific_corrected_protein_v1').startswith('/mnt/d/'))
        self.assertTrue(p.canonical(ROOT/'data2d_batch200_diagnostic').startswith('/mnt/d/'))

    def test_finalization_gate_requires_complete_zero_audit(self):
        p=module('run_e012_v2c_screen_production')
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            root=Path(tmp)
            for name in ('manifests','stats','production/reports'):(root/name).mkdir(parents=True)
            (root/'stats/screen.json').write_text('{"clean_pool":4}')
            b=types.SimpleNamespace(target=4,marker=lambda name:root/'manifests'/(name+'.complete.json'),completed=lambda *args:True)
            with patch.multiple(p,SCI=root,PROD=root/'production',CORRECTION=True):
                with self.assertRaisesRegex(RuntimeError,'DATA2-D'):p.assert_full_corrected_audit(b)

                b.marker('verify_protected').write_text('{"inputs":{}}')
                (root/'stats/verify_protected.json').write_text('{"zero_detected_violations":true,"candidates_screened":3}')
                with self.assertRaisesRegex(RuntimeError,'DATA2-D'):p.assert_full_corrected_audit(b)
                (root/'stats/verify_protected.json').write_text('{"zero_detected_violations":true,"candidates_screened":4}')
                progress=root/'manifests/verification_progress.sqlite'
                with sqlite3.connect(progress) as db:
                    db.execute('CREATE TABLE batches(name,batch,n,marker)');db.execute('CREATE TABLE removed(h,off)')
                    db.execute("INSERT INTO batches VALUES('verify_protected',1,4,'certificate')")
                certpath=root/'manifests/verify_protected-batch-0001.json';retained=certpath.with_suffix('.tsv.gz')
                with gzip.open(retained,'wb') as f:f.write(b'')
                flags=['--dbtype','1','--search-type','1','--threads','16','--min-seq-id','0.30','-c','0.80','--cov-mode','0','-s','7.5','--seq-id-mode','0','--alignment-mode','3','--max-seqs','53615']
                cert={'n':4,'hits':0,'command':flags,'retained_result_sha256':p.sha(retained)};certpath.write_text(json.dumps(cert))
                p.assert_full_corrected_audit(b)
                self.assertTrue((root/'production/reports/FULL_CORRECTED_AUDIT_GATE.json').exists())
                cert['hits']=1;certpath.write_text(json.dumps(cert))
                with self.assertRaisesRegex(RuntimeError,'DATA2-D'):p.assert_full_corrected_audit(b)

    def test_actual_prefilter_truncation_false_negative(self):
        diag=ROOT/'data2d_batch200_diagnostic';case=json.loads((diag/'CAP_OFFENDER.json').read_text())
        self.assertIsNone(case['original_exclusion']);self.assertEqual(case['original_main_alignments'],[])
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            folder=Path(tmp);fa=folder/'query.fasta';fa.write_text(f">{case['query']}\n{case['sequence']}\n")
            outputs=[]
            for cap in (1000,53615):
                out=folder/f'hits-{cap}.tsv'
                args=['mmseqs','easy-search',str(fa),str(ROOT.parent/'mmseqs/protected_index/db'),str(out),str(folder/f'tmp-{cap}'),'--dbtype','1','--search-type','1','--min-seq-id','0.30','-c','0.80','--cov-mode','0','-s','7.5','--seq-id-mode','0','--alignment-mode','3','--threads','16','--split-memory-limit','4G','--max-seqs',str(cap),'--remove-tmp-files','0','--format-output','query,target,fident,qcov,tcov,evalue']
                with (folder/f'command-{cap}.log').open('w') as log:subprocess.run(args,cwd=folder,stdout=log,stderr=log,check=True)
                outputs.append(out.read_text().splitlines())
            self.assertEqual(outputs[0],[]);self.assertEqual(len(outputs[1]),1)
            q,t,i,qc,tc,e=outputs[1][0].split('\t')
            self.assertEqual(q,case['query']);self.assertEqual(t,'p_ecf4906ab6fab8852ef18c7277271f30ec6c642865df96ecddd91c1996f47c4a')
            self.assertEqual((float(i),float(qc),float(tc)),(.301,.867,.875))

    def test_new_correction_starts_from_zero_without_adopting_history(self):
        p=module('run_e012_v2c_screen_production')
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            root=Path(tmp);(root/'manifests').mkdir()
            with sqlite3.connect(root/'manifests/exclusions.sqlite') as db:
                db.execute('CREATE TABLE batches(name,batch,n)');db.execute('CREATE TABLE removed(h,off)')
            with patch.object(p,'SCI',root):self.assertEqual(p.durable(),{'batch':0,'screened':0,'exclusions':0})
