"""Targeted SIFTS/AFDB retrieval for the frozen32 panel; no proteome/MSA DB downloads."""
from r1b_common import *
def candidate(c,acc,rec,maps):
 seq=c['sequence'];dbseq=rec.get('sequence');start=rec.get('sequenceStart');end=rec.get('sequenceEnd');model=rec.get('modelEntityId')
 if not isinstance(dbseq,str) or not isinstance(start,int) or not isinstance(end,int) or not model:return None
 extent='MODEL_INTERVAL' if len(dbseq)==end-start+1 else 'FULL_CANONICAL' if len(dbseq)>=end else 'INCONSISTENT'
 if extent=='INCONSISTENT':return None
 pairs={};conflict=False
 for m in maps:
  a=m.get('start',{}).get('residue_number');b=m.get('end',{}).get('residue_number');u=m.get('unp_start');v=m.get('unp_end')
  if not all(isinstance(z,int) for z in [a,b,u,v]) or b-a!=v-u:continue
  for i,k in zip(range(a,b+1),range(u,v+1)):
   if not 1<=i<=len(seq) or not start<=k<=end:continue
   if i in pairs and pairs[i]!=k:conflict=True
   pairs[i]=k
 def aa(k):return dbseq[k-start] if extent=='MODEL_INTERVAL' else dbseq[k-1]
 valid={i:k for i,k in pairs.items() if aa(k)==seq[i-1]};coverage=len(valid)/len(seq);identity=len(valid)/max(1,len(pairs));exact=(dbseq==seq and start==1 and end==len(seq))
 if exact:
  # Exact string independently fixes correspondence; SIFTS must not contradict it.
  if any(i!=k for i,k in pairs.items()):return None
  valid={i:i for i in range(1,len(seq)+1)};kind='EXACT_FULL_SEQUENCE_MATCH'
 elif not conflict and coverage>=.8 and identity==1. and len(valid)>=20:kind='VALID_MAPPED_FRAGMENT_OR_CONSTRUCT'
 elif len(pairs)>=20:kind='HOMOLOGOUS_REFERENCE_ONLY'
 else:kind='AMBIGUOUS_MAPPING'
 return {'match_status':kind,'uniprot':acc,'modelEntityId':model,'sequenceStart':start,'sequenceEnd':end,'sequence_field_extent':extent,'sequence':dbseq,'native_to_uniprot_positions':{str(i):k for i,k in valid.items()},'SIFTS_position_count':len(pairs),'identical_mapped_position_count':len(valid),'canonical_sequence_coverage':coverage,'mapped_identity':identity,'conflicting_mapping':conflict,'API_record':rec,'SIFTS_segments':maps}
def main():
 panel=json.load(open(R/'configs/NATIVE_PANEL_32.json'));old=R/'ALPHAFOLD_DB_MATCH_MANIFEST.json';rows=json.load(open(old))['rows'] if old.exists() else [];done={c['id'] for c in rows};downloaded={r['selected']['modelEntityId'] for r in rows if r.get('selected') and r['selected'].get('coordinate_path')}
 for c in panel['controls']:
  if c['id'] in done:continue
  guard();pdb=c['entry_id'].lower();row={'id':c['id'],'sequence_id':c['sequence_id'],'split':c['split'],'entry_id':pdb,'N':c['N'],'status':'RETRIEVAL_API_FAILURE','selected':None};auth=c.get('observation_metadata',{}).get('auth_chain','A');label=c.get('observation_metadata',{}).get('label_chain',auth);row['auth_chain']=auth;row['label_chain']=label
  try:
   s,sm=getjson(f'https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb}',f'network/SIFTS_{pdb}.json');row['SIFTS_http']=sm
   mol,mm=getjson(f'https://www.ebi.ac.uk/pdbe/api/pdb/entry/molecules/{pdb}',f'network/PDB_MOLECULES_{pdb}.json');row['molecules_path']=str(R/f'network/PDB_MOLECULES_{pdb}.json');summ,summm=getjson(f'https://www.ebi.ac.uk/pdbe/api/pdb/entry/summary/{pdb}',f'network/PDB_SUMMARY_{pdb}.json');row['summary_path']=str(R/f'network/PDB_SUMMARY_{pdb}.json')
   uni=(s or {}).get(pdb,{}).get('UniProt',{});choices=[];linked=[]
   for acc,item in sorted(uni.items()):
    maps=[m for m in item.get('mappings',[]) if m.get('chain_id')==auth and (m.get('struct_asym_id') in [None,label])]
    if maps:linked.append((acc,maps))
   row['linked_UniProt_accessions']=[a for a,_ in linked]
   for acc,maps in linked[:2]:
    records,meta=getjson(f'https://alphafold.ebi.ac.uk/api/prediction/{acc}',f'afdb/API_{acc}.json');
    if isinstance(records,list):
     for rec in records:
      z=candidate(c,acc,rec,maps)
      if z:choices.append(z)
   if choices:
    ranks={'EXACT_FULL_SEQUENCE_MATCH':0,'VALID_MAPPED_FRAGMENT_OR_CONSTRUCT':1,'HOMOLOGOUS_REFERENCE_ONLY':2,'AMBIGUOUS_MAPPING':3};choices.sort(key=lambda x:(ranks[x['match_status']],-x['canonical_sequence_coverage'],x['modelEntityId']));sel=choices[0];row['status']=sel['match_status'];row['candidate_summary']=[{k:x[k] for k in ['match_status','uniprot','modelEntityId','canonical_sequence_coverage','mapped_identity']} for x in choices];row['selected']=sel
    if row['status']!='AMBIGUOUS_MAPPING':
     model=sel['modelEntityId'];rec=sel['API_record'];safe=model.replace('/','_');version=rec.get('latestVersion');url=rec.get('pdbUrl')
     if not url:row['status']='RETRIEVAL_API_FAILURE';row['reason']='No downloadable PDB coordinate URL advertised'
     else:
      if model not in downloaded and len(downloaded)>=40:raise RuntimeError('40 unique AFDB models hard cap')
      path,m=get(url,f'afdb/models/{safe}_v{version}.pdb',16*2**20)
      if m['status']!=200:row['status']='RETRIEVAL_API_FAILURE';row['coordinate_HTTP']=m
      else:
       sel.update(coordinate_path=str(path),coordinate_sha256=sha(path),coordinate_url=url,release_version=version,download_license='CC-BY-4.0; AFDB official data license with provider attribution');downloaded.add(model)
       # PAE can be large for parent proteins. Bound retrieval; do not invent pTM.
       pae=rec.get('paeDocUrl')
       if pae and len(sel['sequence'])<=1000:
        path,m=get(pae,f'afdb/models/{safe}_v{version}_pae.json',48*2**20)
        if m['status']==200:sel.update(PAE_path=str(path),PAE_sha256=sha(path))
       elif pae:sel['PAE_not_downloaded_reason']='parent length>1000, bounded targeted download policy'
   else:
    row['status']='AMBIGUOUS_MAPPING' if not linked else 'NO_PREDICTION_FOUND';row['reason']='No chain-specific SIFTS link; AFDB absence unestablished' if not linked else 'No qualifying record through targeted associated-UniProt resolver; inspect HTTP receipts, not an exhaustive homolog search'
  except Exception as e:row['error']=type(e).__name__+': '+str(e)
  rows.append(row);write('ALPHAFOLD_DB_MATCH_MANIFEST.json',{'panel_sha256':sha(R/'configs/NATIVE_PANEL_32.json'),'rows':rows,'unique_models_downloaded':len(downloaded),'max_unique_models':40,'API_fields_used':['modelEntityId','sequenceStart','sequenceEnd','sequence'],'legacy_aliases_not_required':True,'all_retrievals_targeted_to_frozen_panel':True,'no_sequence_reward_fit':True});print(json.dumps({'id':c['id'],'status':row['status'],'model':row.get('selected',{}).get('modelEntityId') if row.get('selected') else None,'error':row.get('error')}),flush=True)
if __name__=='__main__':main()
