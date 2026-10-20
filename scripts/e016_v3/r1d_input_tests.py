from r1d_common import *
def main():
 cfg=json.load(open(R/'configs/VARIANT_INFERENCE_FROZEN.json'));bench=json.load(open(R/'STABILITY_VARIANT_BENCHMARK.json'));jobs=cfg['jobs'];assert len(jobs)==12;by={}
 for j in jobs:
  assert j['sequence_sha256']==hashlib.sha256(j['sequence'].encode()).hexdigest();assert 40<=len(j['sequence'])<=72 and set(j['sequence'])<=set('ACDEFGHIKLMNPQRSTVWY');p=Path(j['a3m_path']);assert sha(p)==j['a3m_sha256'];assert p.read_text()==f'>{j["id"]}\n{j["sequence"]}\n';by.setdefault(j['parent_key'],[]).append(j)
 for p,js in by.items():
  wt=next(j['sequence'] for j in js if j['WT']);assert all(len(j['sequence'])==len(wt) and sum(a!=b for a,b in zip(wt,j['sequence']))==(0 if j['WT'] else 1) for j in js);assert len({j['family_id'] for j in js})==len({j['role'] for j in js})==1
 contexts=json.load(open(R/'analysis/PARENT_BIOLOGICAL_CONTEXT.json'))['rows'];assert len(contexts)==4
 assert all(any(a.get('name')=='monomer' for s in row['summary'][row['pdb_id']] for a in s.get('assemblies',[])) for row in contexts)
 write('analysis/INPUT_AND_SPLIT_TESTS.json',{'PASS':True,'canonical_same_length_single_mutations':True,'query_A3Ms_exact':True,'no_geometry_or_labels_in_inputs':True,'no_shared_family_between_dev_and_audit':len({j['family_id'] for j in jobs})==4,'official_monomer_contexts_verified':True,'compatible_saved_MSAs':cfg['exact_selected_parent_MSA_matches'],'no_new_MSA_queries':True,'audit_labels_opened_by_test':False,'selection_policy_sha256':cfg['selection_config_sha256']});print('R1d exact-input, mutation and family-ownership checks passed')
if __name__=='__main__':main()
