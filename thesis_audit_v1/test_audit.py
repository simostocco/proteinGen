"""Bounded source tests and proposed mapping-contract tests; fixtures on D:."""
import sys, tempfile, unittest
from pathlib import Path
import numpy as np
REPO=Path('/mnt/d/Simone/proteinGen');HIST=Path('/home/simostocco/proteinGen-causal-rope')
sys.path.insert(0,str(REPO/'src'));sys.path.insert(0,str(REPO/'tests'));sys.path.insert(0,str(HIST/'src'))
from test_mmcif_parser import _write_custom_mmcif,_atom_row
from protein_distance_diffusion.data.mmcif_parser import parse_mmcif_file_with_rejections
from contracts import validate_mapping,assert_disjoint
from audit import metrics
class Contracts(unittest.TestCase):
    def test_mapping_orientation(self):
        with self.assertRaises(ValueError):validate_mapping('ACD',[3,2,1],'DCA',[True]*3)
    def test_terminal_trim_and_internal_gap(self):
        m=validate_mapping('ACDEFG',[2,3,5],'CDF',[True]*3)
        self.assertEqual((m['n_terminal_trim'],m['c_terminal_trim']),(1,1));self.assertEqual(m['missing_residue_mask'],[True,False,False,True,False,True]);self.assertEqual(m['peptide_adjacency'],[True,False])
    def test_missing_ca_blocks_adjacency(self):
        m=validate_mapping('ACD',[1,2,3],'ACD',[True,False,True]);self.assertFalse(m['complete_ca']);self.assertEqual(m['peptide_adjacency'],[False,False])
    def test_sequence_correspondence(self):
        with self.assertRaises(ValueError):validate_mapping('ACD',[1,2,3],'AAA',[True]*3)
    def test_leakage(self):
        base={'sample_id':'x','canonical_sequence_sha256':'s','pdb_id':'1ABC','homology_cluster_id':'c','split':'train'}
        for key in ['sample_id','canonical_sequence_sha256','pdb_id','homology_cluster_id']:
            second={k:'new' for k in base};second['split']='test';second[key]=base[key]
            with self.subTest(key=key),self.assertRaises(ValueError):assert_disjoint([base,second])
    def test_rigid_duplicate_structure(self):
        x=np.random.default_rng(13).normal(size=(30,3));rot=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        m=metrics(x,x@rot+8);self.assertLess(m['rmsd_angstrom'],1e-10);self.assertAlmostEqual(m['contact_jaccard'],1)
    def test_deformed_duplicate_structure(self):
        x=np.random.default_rng(13).normal(size=(30,3));y=x.copy();y[15:]+=30
        m=metrics(x,y);self.assertGreater(m['rmsd_angstrom'],4);self.assertLess(m['contact_jaccard'],0.7)
class Parser(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=str(REPO/'thesis_audit_v1'));self.path=Path(self.tmp.name)/'example.cif'
    def tearDown(self): self.tmp.cleanup()
    def parse(self,atoms,scheme=None,**kwargs):
        _write_custom_mmcif(self.path,atoms,scheme_rows=scheme,entity_poly_rows=["1 'polypeptide(L)'"])
        return parse_mmcif_file_with_rejections(self.path,min_length=1,**kwargs)
    def test_numeric_label_order_not_lexicographic_author(self):
        samples,_=self.parse([_atom_row(3,seq=3,auth_seq=100,resname='ASP'),_atom_row(1,seq=1,auth_seq=2,resname='ALA'),_atom_row(2,seq=2,auth_seq=10,resname='CYS')]);self.assertEqual(samples[0].sequence,'ACD');self.assertEqual(samples[0].residue_ids,['2','10','100'])
    def test_insertion_codes_survive(self):
        samples,_=self.parse([_atom_row(1,seq=1,auth_seq=10,resname='ALA'),_atom_row(2,seq=2,auth_seq=10,ins='A',resname='CYS')]);self.assertEqual(samples[0].residue_ids,['10','10A'])
    def test_trim_both_termini(self):
        scheme=['A 1 1 ALA A 1 .','A 1 2 CYS A 2 .','A 1 3 ASP A 3 .','A 1 4 GLU A 4 .']
        samples,_=self.parse([_atom_row(1,seq=2,resname='CYS'),_atom_row(2,seq=3,resname='ASP')],scheme,missing_calpha_policy='trim_terminal',max_terminal_trim_fraction=.5)
        self.assertEqual(samples[0].sequence,'CD');self.assertEqual(samples[0].metadata['trimmed_n_terminal_residues'],1);self.assertEqual(samples[0].metadata['trimmed_c_terminal_residues'],1)
    def test_internal_missing_ca_rejected(self):
        scheme=['A 1 1 ALA A 1 .','A 1 2 CYS A 2 .','A 1 3 ASP A 3 .']
        samples,rejects=self.parse([_atom_row(1,seq=1,resname='ALA'),_atom_row(2,seq=3,resname='ASP')],scheme,missing_calpha_policy='trim_terminal');self.assertFalse(samples);self.assertEqual(rejects[0].reason,'internal_missing_calpha')
    def test_modified_residue_mapping(self):
        samples,_=self.parse([_atom_row(1,seq=1,resname='MSE',group='HETATM')]);self.assertEqual(samples[0].sequence,'M')
    def test_altloc_occupancy(self):
        samples,_=self.parse([_atom_row(1,seq=1,altloc='A',occupancy=.3,x=1),_atom_row(2,seq=1,altloc='B',occupancy=.7,x=2)]);self.assertEqual(samples[0].metadata['selected_altlocs'],['B']);self.assertAlmostEqual(samples[0].ca_coordinates[0,0],2)
    def test_model_one_only(self):
        atom=_atom_row(1,seq=1,x=1);atom2=_atom_row(2,seq=1,x=100).rsplit(' ',1)[0]+' 2'
        samples,_=self.parse([atom,atom2]);self.assertAlmostEqual(samples[0].ca_coordinates[0,0],1)
    def test_unsupported_terminal_scheme_residue_can_disappear(self):
        # Demonstrates a source-level loss: no atom exists to trigger unsupported_by_chain.
        scheme=['A 1 1 UNK A 1 .','A 1 2 ALA A 2 .']
        samples,_=self.parse([_atom_row(1,seq=2,resname='ALA')],scheme)
        self.assertEqual(samples[0].sequence,'A');self.assertEqual(samples[0].metadata['original_chain_length'],1)
class LegacyParser(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=str(REPO/'thesis_audit_v1'));self.path=Path(self.tmp.name)/'legacy.pdb'
    def tearDown(self):self.tmp.cleanup()
    def write_atoms(self,indices,resnames,insertions=None,seqres=''):
        lines=[]
        for i,(n,name,ins) in enumerate(zip(indices,resnames,insertions or ['']*len(indices)),1):
            lines.append(f'ATOM  {i:5d}  CA  {name:3s} A{n:4d}{ins:1s}   {float(i):8.3f}{0.:8.3f}{0.:8.3f}  1.00 20.00           C')
        self.path.write_text(seqres+'\n'.join(lines)+'\nTER\nEND\n')
    def test_legacy_seqres_missing_positions_not_checked(self):
        from protein_distance_diffusion.data.mdanalysis_parser import parse_pdb_file
        self.write_atoms([1,3],['ALA','ASP'],seqres='SEQRES   1 A    3  ALA CYS ASP\n')
        samples=parse_pdb_file(self.path,min_length=1,strict_contiguous_ca=True,missing_calpha_policy='reject',allowed_methods=['IMPOSSIBLE'])
        self.assertEqual(samples[0].sequence,'AD');self.assertEqual(samples[0].metadata['original_chain_length'],2)
    def test_legacy_insertion_code_lost_from_returned_ids(self):
        from protein_distance_diffusion.data.mdanalysis_parser import parse_pdb_file
        self.write_atoms([10,10],['ALA','CYS'],['','A'])
        samples=parse_pdb_file(self.path,min_length=1)
        self.assertEqual(samples[0].sequence,'AC');self.assertEqual(samples[0].residue_ids,['10','10'])
class Sequence(unittest.TestCase):
    def test_historical_bos_shift_eos_padding_loss(self):
        import torch
        torch.set_num_threads(1)
        from protein_sequence_generation.e012 import example,VOCAB
        from protein_sequence_generation.collate import collate_sequences
        from protein_sequence_generation.metrics import sequence_cross_entropy
        b=collate_sequences([example({'sample_id':'a','sequence':'ACD'}),example({'sample_id':'b','sequence':'WY'})])
        self.assertEqual(b['input_ids'][0].tolist(),[VOCAB.bos_id]+VOCAB.encode('AC'))
        self.assertEqual(b['target_ids'][0].tolist(),VOCAB.encode('ACD'));self.assertNotIn(VOCAB.eos_id,b['target_ids'][0])
        self.assertEqual(b['attention_mask'].tolist(),[[True,True,True],[True,True,False]])
        logits=torch.zeros(2,3,24);loss=sequence_cross_entropy(logits,b['target_ids'],b['attention_mask']);logits[1,2]=100*torch.arange(24);loss2=sequence_cross_entropy(logits,b['target_ids'],b['attention_mask']);self.assertAlmostEqual(loss.item(),loss2.item())
if __name__=='__main__':unittest.main(verbosity=2)
