"""Focused contracts and actual-v1-example regressions; no model inference."""
import argparse,json,sys,unittest
from pathlib import Path
from core import AA3,cached_canonical,reconstruct,choose_evidence,hash_sequence,future_bos_annotation
ROOT=None
def chain(seq='ACDE',missing=None):
    inverse={a:n for n,a in AA3.items()};mask=missing or [False]*len(seq)
    return {'entity_poly_sequence':';'+seq+';','label_sequence_ids':list(map(str,range(1,len(seq)+1))),'residue_names':[inverse[a] for a in seq],'residue_tokens':list(seq),'auth_sequence_ids':['2','10','100','-1'][:len(seq)],'insertion_codes':['','','A',''][:len(seq)],'missing_residue_mask':mask,'missing_calpha_mask':mask,'atom_sequence':''.join(a for a,m in zip(seq,mask) if not m),'chain_mapping_status':'unambiguous','entity_id':'1'}
class ContractTests(unittest.TestCase):
    def test_cif_lexical_delimiters_not_residues(self):self.assertEqual(cached_canonical(';ACDE;'),'ACDE')
    def test_unknown_residue_rejected(self):
        with self.assertRaises(ValueError):cached_canonical(';ACXE;')
    def test_full_label_coverage_required(self):
        c=chain();c['label_sequence_ids']=['1','2','4','5']
        with self.assertRaises(ValueError):reconstruct(c,'ACDE',[0,1,2,3])
    def test_monomer_contradiction_rejected(self):
        c=chain();c['residue_names'][0]='CYS'
        with self.assertRaises(ValueError):reconstruct(c,'ACDE',[0,1,2,3])
    def test_insertion_codes_and_author_numbering(self):
        m=reconstruct(chain(),'ACDE',[0,1,2,3]);self.assertEqual([r['auth_seq_id'] for r in m['residue_mapping']],['2','10','100','-1']);self.assertEqual(m['residue_mapping'][2]['insertion_code'],'A')
    def test_internal_missing_positions_preserved(self):
        m=reconstruct(chain(missing=[False,True,False,False]),'ADE',[0,2,3]);self.assertEqual(m['internal_missing_positions'],[2]);self.assertEqual(m['retained_label_seq_ids'],[1,3,4])
    def test_selected_missing_ca_rejected(self):
        with self.assertRaises(ValueError):reconstruct(chain(missing=[False,True,False,False]),'ACDE',[0,1,2,3])
    def test_reversed_mapping_rejected(self):
        with self.assertRaises(ValueError):reconstruct(chain(),'EDCA',[3,2,1,0])
    def test_conflicting_strong_evidence_rejected(self):
        a={'physical_candidate_id':'a','mapped_indices':[0,1],'strong_evidential_match':True,'identity_mapping_available':True,'identity_mapping_ambiguous':False,'author_chain_match':True}
        with self.assertRaises(ValueError):choose_evidence([a,{**a,'mapped_indices':[1,2]}])
    def test_weak_evidence_not_promoted(self):self.assertIsNone(choose_evidence([{'strong_evidential_match':False}]))
    def test_c_terminal_crop_is_still_fragment(self):
        a=future_bos_annotation(hash_sequence('ACDE'),hash_sequence('ACD'),'polymer_N_terminus');self.assertEqual(a['future_view_bos'],'fragment_start');self.assertTrue(a['starts_at_deposited_N_terminus'])
    def test_full_and_unknown_bos(self):
        h=hash_sequence('ACDE');self.assertEqual(future_bos_annotation(h,h,'polymer_N_terminus')['future_view_bos'],'complete_deposited_polymer_start');self.assertEqual(future_bos_annotation(None,h,'UNKNOWN')['future_view_bos'],'UNKNOWN')
class ActualExamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=json.loads((ROOT/'tests/actual_examples_evidence_v2.json').read_text())
    def test_four_actual_examples_verified(self):self.assertEqual(set(self.data),{'7z53_W','9qex_E','6zk9_b','6zkc_b'});self.assertTrue(all(r['retained_correspondence_status']=='VERIFIED_V1_AUTHORITATIVE_ANCHOR' for r in self.data.values()))
    def test_7z53_9qex_duplicate_crop_distinct_full(self):
        a=self.data['7z53_W'];b=self.data['9qex_E'];self.assertEqual(a['historical_retained_sequence'],b['historical_retained_sequence']);self.assertEqual(len(a['historical_retained_sequence']),56);self.assertEqual((len(a['canonical_complete_sequence']),len(b['canonical_complete_sequence'])),(59,60));self.assertNotEqual(a['canonical_complete_sha256'],b['canonical_complete_sha256']);self.assertEqual((a['positions_after_retained_interval'],b['positions_after_retained_interval']),(3,4))
    def test_6zk9_6zkc_terminal_mapping(self):
        for sid in ['6zk9_b','6zkc_b']:
            r=self.data[sid];self.assertEqual((len(r['canonical_complete_sequence']),len(r['historical_retained_sequence'])),(124,95));self.assertEqual((r['positions_before_retained_interval'],r['positions_after_retained_interval']),(28,1));self.assertEqual(r['retained_label_seq_ids'],list(range(29,124)));self.assertEqual(r['future_view_bos'],'fragment_start');self.assertFalse(r['starts_at_deposited_N_terminus'])
    def test_actual_full_mapping_and_unknown_disorder(self):
        for r in self.data.values():
            full=r['canonical_complete_sequence'];self.assertEqual(''.join(full[p-1] for p in r['retained_label_seq_ids']),r['historical_retained_sequence']);self.assertEqual(r['canonical_complete_sha256'],hash_sequence(full));self.assertEqual(r['disorder_evidence']['status'],'unknown');self.assertTrue(all(a<b for a,b in zip(r['retained_label_seq_ids'],r['retained_label_seq_ids'][1:])));self.assertEqual(len(r['retained_auth_seq_ids']),len(r['historical_retained_sequence']))
class Publication(unittest.TestCase):
    def test_unknown_scope_never_certified(self):
        m=json.loads((ROOT/'E012_VALIDATION_MEMBERSHIP_V2.json').read_text())
        if m['unknown_full_polymer_train_records'] or m['unknown_full_polymer_validation_records']:self.assertFalse(m['entire_e012_homology_scope_fully_established']);self.assertEqual(m['views']['strict_entire_e012_full_polymer_scope_v2'],[])
    def test_original_panels_unchanged_and_views_subsets(self):
        from run import sha
        d=json.loads((ROOT/'FROZEN_PANEL_VIEWS_V2.json').read_text());self.assertEqual(sha(d['original_panels_path']),d['original_panels_sha256']);orig=json.loads(Path(d['original_panels_path']).read_text())
        for name,views in d['views'].items():
            for ids in views.values():self.assertTrue(set(ids)<=set(orig[name]))
    def test_protocol_frozen_before_reaggregation(self):
        from run import sha
        d=json.loads((ROOT/'EXISTING_PREDICTION_REAGGREGATION_V2.json').read_text());self.assertEqual(d['frozen_protocol_sha256'],sha(ROOT/'PROPOSED_EVALUATION_PROTOCOL_V2.md'));self.assertFalse(d['model_inference'])
    def test_known_mixed_never_enters_conditional_view(self):
        d=json.loads((ROOT/'E012_VALIDATION_MEMBERSHIP_V2.json').read_text());lookup={r['sample_id']:r for r in d['records']}
        for sid in d['views']['retained_complement_assessed_no_detected_relationship_v2']:self.assertFalse(lookup[sid]['known_retained_mixed'])
        for sid in d['views']['retained_and_reconstructed_full_assessed_no_detected_relationship_v2']:self.assertFalse(lookup[sid]['known_any_representation_mixed']);self.assertEqual(lookup[sid]['full_mapping_status'],'VERIFIED')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a,other=p.parse_known_args();ROOT=a.root;assert str(ROOT.resolve()).startswith('/mnt/d/');sys.argv=[sys.argv[0],*other];unittest.main(verbosity=2)
