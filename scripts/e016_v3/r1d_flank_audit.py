"""Post hoc construct sensitivity; retain all original labels/predictions, no refitting."""
from r1d_common import *
import copy,collections,io,zipfile,pandas as pd,numpy as np
from r1d_analyze import family_stats,aggregate

def flanks(row,seq):
 full=row['aa_seq_full'];i=full.find(seq);assert i>=0 and full.find(seq,i+1)<0
 return full[:i],full[i+len(seq):]
def main():
 labels=json.load(open(R/'analysis/DEVELOPMENT_LABELS.json'))+json.load(open(R/'analysis/AUDIT_LABELS_SEALED.json'));wanted={int(r['source_row']) for l in labels for r in l['source_rows']};z=zipfile.ZipFile(R/'data/Processed_K50_dG_datasets.zip');encoded={};start=0
 code=dict(zip([a+b+c for a in 'TCAG' for b in 'TCAG' for c in 'TCAG'],'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'))
 with z.open('Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv') as f:
  for chunk in pd.read_csv(f,usecols=['name','dna_seq','aa_seq_full','aa_seq'],dtype=str,keep_default_na=False,chunksize=25000):
   for i in sorted(v for v in wanted if start<=v<start+len(chunk)):
    row=chunk.iloc[i-start];dna=row['dna_seq'];assert len(dna)%3==0;translation=''.join(code.get(dna[j:j+3],'?') for j in range(0,len(dna),3));encoded[i]={'name':row['name'],'DNA_sha256':hashlib.sha256(dna.encode()).hexdigest(),'DNA_length':len(dna),'translated_payload_length':len(translation),'translated_equals_aa_seq_full':translation==row['aa_seq_full'],'aa_seq_full':row['aa_seq_full']};assert translation==row['aa_seq_full']
   start+=len(chunk)
 assert set(encoded)==wanted
 detail=[];strict=[];notmatched=[]
 for parent in sorted(set(l['parent'] for l in labels)):
  ls=[l for l in labels if l['parent']==parent];wt=next(l for l in ls if l['class']=='WT');variants=[l for l in ls if l['class']!='WT'];vflanks=[flanks(v['source_rows'][0],v['sequence']) for v in variants];assert len(set(vflanks))==1;target=vflanks[0];matches=[]
  for row in wt['source_rows']:
   try:dg=float(row['dG_ML']);lo=float(row['deltaG_95CI_low']);hi=float(row['deltaG_95CI_high']);width=float(row['deltaG_95CI'])
   except ValueError:continue
   if flanks(row,wt['sequence'])==target and np.isfinite([dg,lo,hi,width]).all() and width<=.5:matches.append(row)
  item={'parent':parent,'role':wt['role'],'mutant_flanks':list(target),'mutant_full_payload_lengths':[len(v['source_rows'][0]['aa_seq_full']) for v in variants],'WT_flank_variants':[{'source_row':r['source_row'],'flanks':list(flanks(r,wt['sequence'])),'dG_ML':r['dG_ML']} for r in wt['source_rows']],'matched_WT_source_rows':[r['source_row'] for r in matches],'no_original_row_removed':True}
  if matches:
   w=copy.deepcopy(wt);w['experimental_dG']=float(np.median([float(r['dG_ML']) for r in matches]));w['experimental_low']=min(float(r['deltaG_95CI_low']) for r in matches);w['experimental_high']=max(float(r['deltaG_95CI_high']) for r in matches);w['source_rows']=matches;w['uncertainty']='envelope of matched-flank WT replicate95 intervals, not formal median CI';item.update(matched_flank_WT_reference=w['experimental_dG'],difference_from_author_WT=w['experimental_dG']-wt['experimental_dG']);strict.append(w)
  else:item['WT_pairs_status']='NO_MATCHED_FLANK_WT_MEASUREMENT; WT confidence contrasts retained only as author-reference/core-model analysis';notmatched.append(parent)
  strict.extend(copy.deepcopy(variants));detail.append(item)
 cfg=json.load(open(R/'configs/SELECTION_AND_ANALYSIS_PREDECLARED_v2.json'));by={r['id']:r for r in json.load(open(R/'ALPHAFOLD_VARIANT_INFERENCE_MANIFEST.json'))['rows']};stats=family_stats(strict,by,cfg);ag={role:aggregate([f for f in stats if f['role']==role]) for role in ['DEVELOPMENT','AUDIT']}
 write('analysis/MATCHED_FLANK_SENSITIVITY.json',{'status':'POST_HOC_CORRECTNESS_SENSITIVITY_NOT_NEW_PRIMARY_SELECTION','all_original_rows_and_predictions_retained':True,'families':stats,'aggregate':ag,'construct_details':detail,'parents_without_matched_WT':notmatched,'matched_label_rows':strict,'DNA_payload_checks':encoded,'assumption_corrected':'aa_seq is intended core; aa_seq_full/dna_seq show heterogeneous assay flanks, especially WT replicate controls. No claim all full experimental constructs match.','no_neural_or_reward_refitting':True,'formula_tolerances_unchanged':True,'conclusion':'Primary pTM conclusion remains negative on development and unresolved/limited on audit; report original author-reference results separately from this strict construct sensitivity'})
 write('analysis/CONSTRUCT_AND_FLANK_CHECK.json',{'all_selected_core_substrings_valid':True,'all_selected_assay_flanks_identical':False,'DNA_translation_equals_recorded_full_payload_all_checked':True,'source_rows_checked':len(encoded),'core_sequence_exact':True,'three_parents_have_matched_flank_WT':len(detail)-len(notmatched),'unmatched_WT_parents':notmatched,'report':'analysis/MATCHED_FLANK_SENSITIVITY.json'})
 print(json.dumps({'parents_no_matched_WT':notmatched,'DNA_rows':len(encoded),'pTM':{r:ag[r]['pTM']['family_mean_pair_accuracy'] for r in ag},'pLDDT':{r:ag[r]['mean_pLDDT']['family_mean_pair_accuracy'] for r in ag}}),flush=True)
if __name__=='__main__':main()
