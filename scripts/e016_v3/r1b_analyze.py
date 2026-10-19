"""Read-only native mapping and cached AFDB structure analysis; no network/model updates."""
import json,math,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
import gemmi
from r1b_common import R,PREV,write,sha,guard
from r1b_prepare import comparisons,aligned_deviations
from protein_joint_sac.rl.sequence_rewards import mapped_structure_scores

def parse(value,default=None):
 if isinstance(value,str):
  try:return json.loads(value)
  except ValueError:return value
 return value if value is not None else default

def main():
 guard();panel=json.load(open(R/'configs/NATIVE_PANEL_32.json'));meta=json.load(open(R/'native/SOURCE_METADATA.json'));matches=json.load(open(R/'ALPHAFOLD_DB_MATCH_MANIFEST.json'));audit=[];data=[]
 for c in panel['controls']:
  mid=next(r for r in matches['rows'] if r['id']==c['id']);mol=json.load(open(mid['molecules_path']));summary=json.load(open(mid['summary_path']));auth=mid['auth_chain'];entities=[v for v in mol.get(c['entry_id'],[]) if auth in v.get('in_chains',[]) and v.get('molecule_type')=='polypeptide(L)'];item=meta.get(c['id'],{});row={'id':c['id'],'split':c['split'],'sequence_id':c['sequence_id'],'entry_id':c['entry_id'],'canonical_sequence':c['sequence'],'full_deposited_polymer_length':c['N'],'label_chain':mid['label_chain'],'auth_chain':auth,'observed_CA_count':len(c['observed_positions']),'observed_CA_fraction':len(c['observed_positions'])/c['N'],'sequence_sha_correct':hashlib.sha256(c['sequence'].encode()).hexdigest()==c['sequence_id'],'PDBe_chain_entities_matching_sequence':[v.get('entity_id') for v in entities if v.get('sequence')==c['sequence']],'PDBe_sequence_agrees':any(v.get('sequence')==c['sequence'] for v in entities),'molecule_names':[v.get('molecule_name') for v in entities],'summary':summary.get(c['entry_id']),'stored_vs_authoritative_CA_Kabsch_max_A':c.get('stored_vs_authoritative_CA_Kabsch_max_A'),'mapping_status':c.get('observation_metadata',{}).get('mapping_status'),'model':c.get('observation_metadata',{}).get('model',c.get('source_model')),'source_coordinate_provenance':c.get('observation_metadata',c.get('source_path')),'experimental_metadata':parse(c.get('observation_metadata',{}).get('metadata'),{}),'known_context_is_not_proven_cause':True}
  if item:
   row.update(label_sequence_ids=parse(item.get('label_sequence_ids')),auth_sequence_ids=parse(item.get('auth_sequence_ids')),insertion_codes=parse(item.get('insertion_codes')),selected_altlocs=parse(item.get('selected_altlocs')),altloc_outcome_counts=parse(item.get('altloc_outcome_counts')),chain_count=item.get('chain_count'),nonpolymer_component_counts=parse(item.get('nonpolymer_component_counts')),source_entity_id=item.get('entity_id'),cached_scheme_sequence_agrees=item.get('scheme_sequence')==c['sequence'])
   if row['label_sequence_ids'] is not None:assert [int(i) for i in row['label_sequence_ids']]==list(range(1,c['N']+1))
  elif c.get('source_path'):
   assert sha(c['source_path'])==c['source_sha256'];st=gemmi.read_structure(c['source_path']);block=gemmi.cif.read_file(c['source_path']).sole_block();residues=[res for ch in st[0] if ch.name==auth for res in ch if res.label_seq is not None and any(a.name=='CA' and a.altloc in ['\x00',' ','A'] for a in res)];row.update(label_sequence_ids=[res.label_seq for res in residues],auth_sequence_ids=[str(res.seqid) for res in residues],insertion_codes=[res.seqid.icode.strip() for res in residues],experimental_method=block.find_value('_exptl.method'),resolution=block.find_value('_refine.ls_d_res_high'),nonpolymer_components=sorted({res.name for ch in st[0] for res in ch if res.label_seq is None}),chain_count=len(st[0]),source_SHA_reverified=True)
   for res in residues:assert gemmi.find_tabulated_residue(res.name).one_letter_code==c['sequence'][res.label_seq-1]
  audit.append(row)
  s=mid.get('selected')
  if not s or not s.get('coordinate_path'):continue
  out={'id':c['id'],'split':c['split'],'N':c['N'],'match_status':mid['status'],'modelEntityId':s['modelEntityId'],'provider':s['API_record'].get('providerId'),'toolUsed':s['API_record'].get('toolUsed'),'release_version':s['release_version'],'coordinate_sha256':s['coordinate_sha256'],'confidence_metadata_global_mean_pLDDT':s['API_record'].get('globalMetricValue'),'ptm':None,'ptm_reason':'not advertised by selected AFDB API record; do not substitute average pLDDT'}
  try:
   assert sha(s['coordinate_path'])==s['coordinate_sha256'];st=gemmi.read_structure(s['coordinate_path']);wanted=s['sequence'];begin=s['sequenceStart'];end=s['sequenceEnd'];want_model=wanted if s['sequence_field_extent']=='MODEL_INTERVAL' else wanted[begin-1:end];chains=[]
   for ch in st[0]:
    residues=[r for r in ch if any(a.name=='CA' for a in r)];seq=''.join(gemmi.find_tabulated_residue(res.name).one_letter_code for res in residues)
    if seq==want_model:chains.append((ch.name,residues))
   if len(chains)!=1:raise ValueError('model sequence/chain not unique or disagrees with actual API sequence interval')
   chain,residues=chains[0];coordinates={};confidence={};numbers=[r.seqid.num for r in residues]
   if numbers!=list(range(begin,end+1)) and numbers!=list(range(1,len(residues)+1)):raise ValueError('coordinate/model interval numbering mismatch')
   for k,res in enumerate(residues,begin):
    atoms=[a for a in res if a.name=='CA'];assert len(atoms)==1;coordinates[k]=list(atoms[0].pos);confidence[k]=float(atoms[0].b_iso)
   y=np.full((c['N'],3),np.nan);pred_conf=np.full(c['N'],np.nan);indices={}
   for i,k in s['native_to_uniprot_positions'].items():
    i=int(i)
    if k in coordinates:y[i-1]=coordinates[k];pred_conf[i-1]=confidence[k];indices[i-1]=k-begin
   x=np.array([v if v is not None else [np.nan]*3 for v in c['native_ca']],float);valid=np.isfinite(x).all(1)&np.isfinite(y).all(1);assert valid.sum()>=4
   # Missing mapped predictions stay missing; scoring uses their explicit mask.
   nativec=dict(c,native_ca=[v.tolist() if mask else None for v,mask in zip(x,valid)]);filled=np.where(np.isfinite(y),y,0.);d=comparisons(nativec,y,'AFDB_'+c['id']);out.update(status='SCORED',full_query_mapping_coverage=float(np.isfinite(y).all(1).mean()),observed_comparison_coverage=float(valid.mean()),AFDB_chain_sequence_verified=True,actual_AFDB_chain=chain,mean_pLDDT_mapped_region=float(np.nanmean(pred_conf)),native_recovery=d,qualified_exact_or_construct=(mid['status'] in ['EXACT_FULL_SEQUENCE_MATCH','VALID_MAPPED_FRAGMENT_OR_CONSTRUCT']),full_predicted_model_length=len(residues))
   if s.get('PAE_path'):
    a=json.load(open(s['PAE_path']));a=a[0] if isinstance(a,list) else a;arr=np.asarray(a.get('predicted_aligned_error'),float)
    if arr.shape!=(len(residues),len(residues)):raise ValueError('PAE size does not match model interval')
    ii=np.array(sorted(indices));jj=np.array([indices[i] for i in ii]);sub=arr[np.ix_(jj,jj)];sep=np.abs(ii[:,None]-ii[None,:])>=min(24,max(4,c['N']//4));out['mapped_long_range_PAE_mean_A']=float(sub[sep].mean()) if sep.any() else None;out['mapped_long_range_PAE_fraction_below10A']=float((sub[sep]<10).mean()) if sep.any() else None
   out['inexact_sequence_is_not_prediction_of_queried_sequence']=mid['status']!='EXACT_FULL_SEQUENCE_MATCH'
  except Exception as e:out.update(status='COMPARISON_UNRESOLVED',error=type(e).__name__+': '+str(e))
  data.append(out);print(json.dumps({'id':c['id'],'status':out['status'],'score':out.get('native_recovery',{}).get('robust_fixed_map_TM_similarity_fullN'),'error':out.get('error')}),flush=True)
 write('native/MAPPING_REAUDIT.json',{'rows':audit,'original_eight_retained':True,'no_sequence_or_coordinate_relabeling':True,'new_REMOTE_metadata_does_not_replace_frozen_source':True});qualified=[r for r in data if r['status']=='SCORED' and r['qualified_exact_or_construct']];train=[r for r in qualified if r['split']=='TRAIN'];val=[r for r in qualified if r['split']=='VALIDATION'];from scipy.stats import spearmanr
 correlation=lambda rows:float(spearmanr([r['mean_pLDDT_mapped_region'] for r in rows],[r['native_recovery']['robust_fixed_map_TM_similarity_fullN'] for r in rows]).statistic) if len(rows)>=3 else None
 write('ALPHAFOLD_DB_NATIVE_RECOVERY.json',{'rows':data,'match_status_counts':dict(Counter(r['status'] for r in matches['rows'])),'qualified_exact_or_construct_scored':len(qualified),'qualified_TRAIN':len(train),'qualified_VALIDATION':len(val),'TRAIN_median_recovery':float(np.median([r['native_recovery']['robust_fixed_map_TM_similarity_fullN'] for r in train])) if train else None,'VALIDATION_median_recovery':float(np.median([r['native_recovery']['robust_fixed_map_TM_similarity_fullN'] for r in val])) if val else None,'TRAIN_mapped_region_pLDDT_recovery_Spearman':correlation(train),'VALIDATION_mapped_region_pLDDT_recovery_Spearman':correlation(val),'calibration_fit':False,'protocols_not_pooled_with_local_AF2':True,'homologous_references_not_counted_as_exact_query_predictions':True,'no_invented_pTM':True});print('AFDB and mapping cached analysis complete.',flush=True)
if __name__=='__main__':main()
