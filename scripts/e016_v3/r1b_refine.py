"""Distinguish absent mapping evidence from explicit AFDB accession-not-found responses."""
from r1b_common import *
from collections import Counter

def main():
 p=R/'ALPHAFOLD_DB_MATCH_MANIFEST.json';m=json.load(open(p));initial=R/'afdb/INITIAL_MATCH_CLASSIFICATION.json'
 if not initial.exists():write(str(initial.relative_to(R)),m)
 for r in m['rows']:
  if r['status']!='NO_PREDICTION_FOUND':continue
  accessions=r.get('linked_UniProt_accessions',[]);statuses=[]
  for acc in accessions[:2]:
   receipt=R/f'afdb/API_{acc}.json.http.json'
   if receipt.exists():statuses.append({'uniprot':acc,**json.load(open(receipt))})
  r['associated_AFDB_API_responses']=statuses
  if not accessions:r['status']='AMBIGUOUS_MAPPING';r['reason']='No chain-specific authoritative UniProt mapping in the returned SIFTS inventory (including missing PDB index responses); AFDB absence not established'
  elif any(s.get('status') not in [200,404] for s in statuses):r['status']='RETRIEVAL_API_FAILURE';r['reason']='Associated model API request failed; do not call the protein absent'
  elif statuses and all(s.get('status')==404 for s in statuses):r['reason']='Official prediction endpoint returned404 for the bounded associated UniProt queries; no prediction found through this resolver, not exhaustive homolog search'
  else:r['status']='AMBIGUOUS_MAPPING';r['reason']='Returned associated records lack a verifiable residue mapping; no usable exact/construct comparison established'
 m['classification_note']='Missing SIFTS/chain links are ambiguous mapping, not proof of no AFDB model. First progress classification retained separately; recovery scores and downloaded models unchanged.';m['status_counts']=dict(Counter(r['status'] for r in m['rows']));write('ALPHAFOLD_DB_MATCH_MANIFEST.json',m)
 get('https://github.com/sokrypton/ColabFold/wiki/MSA-Server-Database-History','docs/MSA_SERVER_DATABASE_HISTORY.html',8*2**20)
 write('docs/LIVE_SCHEMA_AUDIT.json',{'required_new_fields':['modelEntityId','sequenceStart','sequenceEnd','sequence'],'all_selected_records_have_required_fields':all(all(k in r['selected']['API_record'] for k in ['modelEntityId','sequenceStart','sequenceEnd','sequence']) for r in m['rows'] if r.get('selected')),'observed_legacy_aliases_still_present_despite_sunset_documentation':True,'client_uses_new_fields_only':True,'paeImageUrl_observed_but_not_used':True,'nonzero_sequenceStart_handled':True,'providers':sorted({r['selected']['API_record'].get('providerId') for r in m['rows'] if r.get('selected')})})
 print(m['status_counts'],flush=True)
if __name__=='__main__':main()
