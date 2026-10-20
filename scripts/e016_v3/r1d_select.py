"""Family-separated stability inventory and prospective three-sequence contrasts."""
from r1d_common import *
import pandas as pd,numpy as np,re,collections,subprocess,bisect
import pyarrow.parquet as pq
AA=set('ACDEFGHIKLMNPQRSTVWY');CF=Path('/mnt/d/Simone/proteinGen/e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1/external_designability_loop_v1/tools/colabfold_env/bin/python')
def number(x):
 try:v=float(x);return v if np.isfinite(v) else None
 except (TypeError,ValueError):return None
def raw_record(r):return {k:int(v) if k=='source_row' else str(v) for k,v in r.items()}
def main():
 guard();cfg=json.load(open(R/'configs/SELECTION_AND_ANALYSIS_PREDECLARED_v2.json'));df=pd.read_parquet(R/'data/STABILITY_SOURCE_ROWS.parquet');dms=pd.read_csv(R/'data/Single_DMS_list.csv',dtype=str,keep_default_na=False).set_index('name');meta={};eligible_parts=[];fail=collections.Counter();counts={};st=time.monotonic()
 nums={c:pd.to_numeric(df[c],errors='coerce') for c in ['dG_ML','ddG_ML','deltaG_95CI','deltaG_95CI_low','deltaG_95CI_high']};good=np.isfinite(nums['dG_ML'])&np.isfinite(nums['ddG_ML'])&nums['deltaG_95CI'].between(0,.5)&(nums['dG_ML']>-.95)&(nums['dG_ML']<4.95)&np.isfinite(nums['deltaG_95CI_low'])&np.isfinite(nums['deltaG_95CI_high'])
 ddg_res=[];stable_flag=[]
 for name,g in df.groupby('WT_name',sort=False):
  wt=g[g['mut_type']=='wt'];source=dms.loc[name] if name in dms.index else None
  if source is not None and isinstance(source,pd.DataFrame):fail['ambiguous_DMS_parent_metadata']+=1;continue
  seq=str(source['aa_seq']) if source is not None else (str(wt['aa_seq'].iloc[0]) if len(wt) else '')
  if not seq or set(seq)-AA:fail['noncanonical_or_unknown_parent']+=1;continue
  cluster=str(g['WT_cluster'].iloc[0]);origin='natural' if re.fullmatch(r'[0-9]+',cluster) else 'designed'
  ref=number(source['wt_dg_med']) if source is not None else None
  if ref is None:fail['missing_author_WT_reference']+=1;continue
  canonical_length=40<=len(seq)<=72
  subset=g[good.loc[g.index]];exact=[]
  for i,row in subset.iterrows():
   mt=re.fullmatch(r'([A-Z])(\d+)([A-Z])',row['mut_type']);s=row['aa_seq']
   if mt is None or len(s)!=len(seq) or set(s)-AA:continue
   a,pos,b=mt.groups();pos=int(pos)-1
   if not 0<=pos<len(seq) or seq[pos]!=a or s!=seq[:pos]+b+seq[pos+1:] or a==b:continue
   err=float(nums['dG_ML'].loc[i]-nums['ddG_ML'].loc[i]-ref);ddg_res.append(err)
   if abs(err)>1e-6:fail['uncensored_WT_reference_disagreement']+=1;continue
   if b=='C':fail['introduced_Cys_excluded_from_primary_pairs']+=1;continue
   exact.append(i)
  wc=wt[good.loc[wt.index]];wc=wc[wc['aa_seq']==seq]
  low=min([float(nums['deltaG_95CI_low'].loc[i]) for i in wc.index],default=None);high=max([float(nums['deltaG_95CI_high'].loc[i]) for i in wc.index],default=None)
  base={'name':name,'sequence':seq,'N':len(seq),'WT_cluster':cluster,'origin':origin,'PDB_accession':re.search(r'([0-9][A-Za-z0-9]{3})\.pdb',name).group(1).lower() if origin=='natural' and re.search(r'([0-9][A-Za-z0-9]{3})\.pdb',name) else None,'DMS_group':str(source['DMS_group']) if source is not None else None,'parent_source':source.to_dict() if source is not None else None,'author_WT_stability':ref,'WT_source_rows':[raw_record(r) for _,r in wt.iterrows()],'qualified_WT_rows':len(wc),'WT_posterior95_envelope_low':low,'WT_posterior95_envelope_high':high,'WT_envelope_semantics':'envelope of individual qualified WT replicate posterior95 intervals, not a formal median CI','raw_variant_rows':len(g),'qualified_exact_single_substitution_rows':len(exact),'length_40_72':canonical_length,'suitable_WT_reference':bool(len(wc) and low is not None and high-low<=.5 and low<=ref<=high),'source_tier':'finite author ddG_ML and dG_ML; no generated ddG for Dataset2-only parents'}
  meta[name]=base
  if canonical_length:
   sub=df.loc[exact].copy();sub['parent_reference']=ref;eligible_parts.append(sub)
 allgood=pd.concat(eligible_parts,ignore_index=True) if eligible_parts else df.iloc[:0];allgood.to_parquet(R/'data/QUALIFIED_SINGLE_VARIANTS.parquet',index=False,compression='zstd')
 # Identity ownership/construct audit only: no TEST measurements or model scoring.
 path=D/'pdb_supervised_split_50id80cov_20261016_v1/SEQUENCE_ASSIGNMENTS.parquet';assign=pq.read_table(path,columns=['sequence_id','canonical_full_sequence','split','protected_group_id']).to_pylist();byseq={v['canonical_full_sequence']:v for v in assign};joined='|'.join(v['canonical_full_sequence'] for v in assign);starts=[];offset=0
 for v in assign:starts.append(offset);offset+=len(v['canonical_full_sequence'])+1
 for name,m in meta.items():
  s=m['sequence'];matches=[];total=0;splits=collections.Counter();groups=set();at=0
  while True:
   p=joined.find(s,at)
   if p<0:break
   at=p+1;k=bisect.bisect_right(starts,p)-1;v=assign[k];total+=1;splits[v['split']]+=1;groups.add(v['protected_group_id'])
   if len(matches)<12:matches.append({'sequence_id':v['sequence_id'],'split':v['split'],'protected_group_id':v['protected_group_id'],'start0':p-starts[k],'exact_full_sequence':s==v['canonical_full_sequence']})
  m.update(E016_exact_full=byseq.get(s),E016_exact_construct_match_count=total,E016_construct_matches_examples=matches,E016_matching_splits=dict(splits),E016_protected_groups=sorted(groups))
 write('analysis/PARENT_PRELIMINARY.json',{'parents':meta,'assay_sign_verification':{'num_uncensored_single_rows_checked':len(ddg_res),'max_abs_dG_minus_ddG_minus_authorWT':float(max(map(abs,ddg_res),default=0)),'median_abs_error':float(np.median(np.abs(ddg_res))) if ddg_res else None,'definition_verified':'ddG_ML = dG_ML - author WT median; positive means more stable in this release'},'extra_filter_counts':dict(fail),'all_source_rows':len(df),'qualified_single_rows':len(allgood),'qualified_unique_variant_sequences':allgood['aa_seq'].nunique(),'parents_with_qualified_variants':allgood['WT_name'].nunique(),'sequence_assignments_path':str(path),'sequence_assignments_sha256':sha(path),'no_TEST_labels_or_predictions_used':True})
 subprocess.run([str(CF),'-B',str(Path(__file__).with_name('r1d_families.py'))],check=True,timeout=180)
 graph=json.load(open(R/'analysis/FAMILY_GRAPH.json'));family_by=graph['family_by_parent'];components={f['id']:f for f in graph['families']};blocked=set()
 for n,m in meta.items():
  m['family_id']=family_by[n]
  if m['E016_matching_splits'].get('TEST',0):blocked.add(m['family_id'])
 candidates=[];diagnostics=[]
 for name,g in allgood.groupby('WT_name',sort=False):
  m=meta[name];why=[]
  if m['origin']!='natural':why.append('not primary natural domain')
  if m['family_id'] in blocked:why.append('family has accepted TEST construct overlap')
  if not m['E016_matching_splits'].get('TRAIN'):why.append('no exact TRAIN construct link for qualified native-reference search')
  if not m['suitable_WT_reference']:why.append('WT uncertainty envelope or reference unsuitable')
  if not 2.5<=m['author_WT_stability']<=4.5:why.append('WT stability outside prespecified2.5..4.5')
  if len(g)<50:why.append('fewer than50 qualified substitutions')
  if why:diagnostics.append({'parent':name,'reasons':why});continue
  ref=m['author_WT_stability'];dg=pd.to_numeric(g['dG_ML']);hi=pd.to_numeric(g['deltaG_95CI_high']);lo=pd.to_numeric(g['deltaG_95CI_low']);moderate=g[(ref-dg>=1)&(ref-dg<=1.5)&(hi<m['WT_posterior95_envelope_low']-.5)];strong=g[(ref-dg>=2.5)&(hi<m['WT_posterior95_envelope_low']-.5)]
  pair=None
  for _,r in moderate.assign(key=moderate['name'].map(lambda n:hashlib.sha256(('R1d20261020'+n).encode()).hexdigest())).sort_values('key').iterrows():
   available=strong[(pd.to_numeric(strong['dG_ML'])<=float(r['dG_ML'])-1)&(pd.to_numeric(strong['deltaG_95CI_high'])<float(r['deltaG_95CI_low']))]
   if len(available):
    t=available.assign(key=available['name'].map(lambda n:hashlib.sha256(('R1d20261020'+n).encode()).hexdigest())).sort_values('key').iloc[0];pair=(r,t);break
  if pair is None:diagnostics.append({'parent':name,'reasons':['no conservative separated moderate/strong single-mutant pair']});continue
  wtrows=[r for r in m['WT_source_rows'] if r['aa_seq']==m['sequence']];variants=[{'class':'WT','sequence':m['sequence'],'name':name,'experimental_dG':ref,'experimental_low':m['WT_posterior95_envelope_low'],'experimental_high':m['WT_posterior95_envelope_high'],'ddG_vs_authorWT':0.,'source_rows':wtrows,'uncertainty':'conservative envelope, not formal median CI'}]
  for cl,r in zip(['moderate','strong'],pair):variants.append({'class':cl,'sequence':r['aa_seq'],'name':r['name'],'mut_type':r['mut_type'],'experimental_dG':float(r['dG_ML']),'experimental_low':float(r['deltaG_95CI_low']),'experimental_high':float(r['deltaG_95CI_high']),'ddG_vs_authorWT':float(r['ddG_ML']),'source_rows':[raw_record(r.drop(labels=['key','parent_reference'],errors='ignore').to_dict())]})
  candidates.append({'parent':name,'family_id':m['family_id'],'N':m['N'],'sequence':m['sequence'],'PDB_accession':m['PDB_accession'],'variants':variants,'selection_key':hashlib.sha256(('R1d20261020'+name).encode()).hexdigest()})
 write('analysis/ELIGIBLE_PARENT_INVENTORY.json',{'parents':meta,'candidate_contrast_parents':candidates,'exclusions':diagnostics,'family_graph_path':'analysis/FAMILY_GRAPH.json','blocked_TEST_overlap_families':sorted(blocked),'source_QC_counts':dict(fail),'qualified_single_rows':len(allgood),'qualified_unique_sequences':allgood['aa_seq'].nunique(),'qualified_parents':allgood['WT_name'].nunique(),'family_components_total':len(components),'seconds':time.monotonic()-st})
 print(json.dumps({'qualified_rows':len(allgood),'qualified_parents':allgood['WT_name'].nunique(),'families':len(components),'candidate_parents':[(v['parent'],v['N'],v['family_id']) for v in candidates],'failure_counts':collections.Counter(r for x in diagnostics for r in x['reasons'])}),flush=True)
if __name__=='__main__':main()
