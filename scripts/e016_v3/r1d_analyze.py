"""Locked experimental ranking, first development then untouched audit families."""
from r1d_common import *
import math,itertools,numpy as np
from scipy.stats import spearmanr,kendalltau
from protein_joint_sac.rl.sequence_rewards import mapped_structure_scores
from protein_joint_sac.data.scaled_supervised import internal
METRICS=['pTM','mean_pLDDT','fraction_pLDDT_above70','long_range_PAE_good_fraction','fixed_composite']
def signals(r):
 f=r['features'];return {'pTM':f['ptm'],'mean_pLDDT':f['mean_pLDDT'],'fraction_pLDDT_above70':f['fraction_pLDDT_above70'],'long_range_PAE_good_fraction':f['long_range_PAE_good_fraction'],'fixed_composite':.5*(f['ptm']+f['mean_pLDDT']/100)}
def finite(x):return float(x) if np.isfinite(x) else None
def family_stats(labels,by,cfg):
 out=[]
 for parent,ls in itertools.groupby(sorted(labels,key=lambda l:l['parent_key']),key=lambda l:l['parent_key']):
  ls=list(ls);present=[l for l in ls if l['id'] in by and by[l['id']]['status']=='SUCCEEDED'];metrics={};pairs=[]
  for a,b in itertools.combinations(present,2):
   hi,lo=sorted([a,b],key=lambda l:l['experimental_dG'],reverse=True);gap=hi['experimental_dG']-lo['experimental_dG'];valid=gap>=1-1e-9 and hi['experimental_low']>lo['experimental_high']
   if not valid:continue
   delta={m:signals(by[hi['id']])[m]-signals(by[lo['id']])[m] for m in METRICS};correct={m:.5 if abs(v)<=cfg['score_ties'][m]+1e-9 else float(v>0) for m,v in delta.items()};pairs.append({'more_stable':hi['id'],'less_stable':lo['id'],'more_stable_class':hi['class'],'less_stable_class':lo['class'],'experimental_difference_kcal_mol':gap,'intervals_nonoverlap':True,'score_difference':delta,'correct':correct,'large_effect_atleast2':gap>=2})
  for metric in METRICS:
   y=[l['experimental_dG'] for l in present];x=[signals(by[l['id']])[metric] for l in present];rho=finite(spearmanr(x,y).statistic) if len(set(x))>1 and len(x)>=3 else None;tau=finite(kendalltau(x,y).statistic) if len(set(x))>1 and len(x)>=3 else None
   metrics[metric]={'Spearman':rho,'Kendall':tau,'pairwise_accuracy':float(np.mean([p['correct'][metric] for p in pairs])) if pairs else None,'decisive_pair_count':sum(p['correct'][metric]!=.5 for p in pairs),'eligible_pair_count':len(pairs),'WT_vs_strong_correct':next((p['correct'][metric] for p in pairs if p['more_stable_class']=='WT' and p['less_stable_class']=='strong'),None),'large_effect_accuracy':float(np.mean([p['correct'][metric] for p in pairs if p['large_effect_atleast2']])) if any(p['large_effect_atleast2'] for p in pairs) else None}
  out.append({'parent_key':parent,'parent':ls[0]['parent'],'family_id':ls[0]['family_id'],'role':ls[0]['role'],'predicted_sequences':len(present),'expected_sequences':len(ls),'metrics':metrics,'pairs':pairs})
 return out

def aggregate(families):
 out={};rng=np.random.default_rng(2026102017)
 for m in METRICS:
  vals=[f['metrics'][m]['pairwise_accuracy'] for f in families if f['metrics'][m]['pairwise_accuracy'] is not None];rho=[f['metrics'][m]['Spearman'] for f in families if f['metrics'][m]['Spearman'] is not None]
  out[m]={'family_mean_pair_accuracy':float(np.mean(vals)) if vals else None,'n_families':len(vals),'family_mean_Spearman':float(np.mean(rho)) if rho else None,'n_defined_Spearman_families':len(rho),'descriptive_family_bootstrap95_pair_accuracy':np.quantile([np.mean(rng.choice(vals,len(vals),replace=True)) for _ in range(2000)],[.025,.975]).tolist() if vals else None,'bootstrap_limitation':'only two independent operational family units per role; interval is descriptive and cannot certify a general reward'}
 return out

def main():
 cfg=json.load(open(R/'configs/SELECTION_AND_ANALYSIS_PREDECLARED_v2.json'));fold=json.load(open(R/'ALPHAFOLD_VARIANT_INFERENCE_MANIFEST.json'));by={r['id']:r for r in fold['rows']};dev=json.load(open(R/'analysis/DEVELOPMENT_LABELS.json'));ds=family_stats(dev,by,cfg);dag=aggregate(ds)
 # All candidate signals/gates fixed before first fold; lock proof comes before audit inference.
 write('analysis/DEVELOPMENT_RANKING_ONLY.json',{'families':ds,'aggregate':dag,'primary':'pTM','new_fitted_coefficients':0,'no_candidate_selection_on_audit':True})
 lock=R/'analysis/BEFORE_AUDIT_SCORING_LOCK.json';audit=json.load(open(R/'analysis/AUDIT_LABELS_SEALED.json'));icfg=json.load(open(R/'configs/VARIANT_INFERENCE_FROZEN.json'));assert sha(R/'analysis/AUDIT_LABELS_SEALED.json')==icfg['audit_labels_sha256']
 if any(r['role']=='AUDIT' for r in fold['rows']):assert lock.exists() and json.load(open(lock))['metric_or_threshold_changes']==0
 ass=family_stats(audit,by,cfg);aag=aggregate(ass);labels=dev+audit;comparison=[];sensitivity=[]
 for l in labels:
  if l['id'] not in by or by[l['id']]['status']!='SUCCEEDED':comparison.append(dict(l,status='NOT_SCORED'));continue
  row=by[l['id']];res=json.load(open(R/'folds'/l['id']/'RESULT.json'));p=np.asarray(res['plddt']);x=np.asarray(res['predicted_ca']);geo=internal(x);summary={'adjacent_CA_mean_A':float(geo[1:,0].mean()),'adjacent_CA_q05_q95_A':np.quantile(geo[1:,0],[.05,.95]).tolist(),'virtual_angle_mean_rad':float(geo[2:,1].mean()),'torsion_circular_resultant':float(abs(np.exp(1j*geo[3:,2]).mean())),'fraction_pLDDT_above90':float((p>=90).mean()),'fraction_pLDDT_below50':float((p<50).mean())};comparison.append(dict(l,status='SUCCEEDED',signals=signals(row),features=row['features'],local_structure=summary,predicted_PDB=row['predicted_PDB'],predicted_PDB_sha256=row['predicted_PDB_sha256'],WT_native_fold_compatibility=row['WT_native_fold_compatibility']))
  wt=next(v for v in labels if v['parent_key']==l['parent_key'] and v['class']=='WT');wr=by.get(wt['id'])
  if l['class']!='WT' and wr and wr['status']=='SUCCEEDED':
   wres=json.load(open(R/'folds'/wt['id']/'RESULT.json'));wx=np.asarray(wres['predicted_ca']);wg=internal(wx);delta_tau=np.angle(np.exp(1j*(geo[3:,2]-wg[3:,2])));site=next(i for i,(a,b) in enumerate(zip(wt['sequence'],l['sequence'])) if a!=b);lo=max(0,site-4);hi=min(len(p),site+5);match=mapped_structure_scores(wx,x);sensitivity.append({'id':l['id'],'parent':l['parent'],'role':l['role'],'class':l['class'],'mutation':l.get('mut_type'),'experimental_WT_dG':wt['experimental_dG'],'experimental_mutant_dG':l['experimental_dG'],'experimental_stability_loss':wt['experimental_dG']-l['experimental_dG'],'mutant_mean_pLDDT':row['features']['mean_pLDDT'],'mutant_pTM':row['features']['ptm'],'delta_mean_pLDDT':row['features']['mean_pLDDT']-wr['features']['mean_pLDDT'],'delta_pTM':row['features']['ptm']-wr['features']['ptm'],'delta_PAEgood':row['features']['long_range_PAE_good_fraction']-wr['features']['long_range_PAE_good_fraction'],'AF_mutant_vs_AF_WT':match,'native_WT_reference_compatibility_is_not_mutant_ground_truth':True,'local_mutation_window':[lo+1,hi],'delta_mutation_window_pLDDT':float((p[lo:hi]-np.asarray(wres['plddt'])[lo:hi]).mean()),'CA_distance_RMS_change_A':float(np.sqrt(np.mean((geo[1:,0]-wg[1:,0])**2))),'virtual_angle_RMS_change_rad':float(np.sqrt(np.mean((geo[2:,1]-wg[2:,1])**2))),'torsion_circular_RMS_change_rad':float(np.sqrt(np.mean(delta_tau**2))),'confident_same_fold_despite_large_destabilization':bool(wt['experimental_dG']-l['experimental_dG']>=2 and row['features']['mean_pLDDT']>=80 and row['features']['ptm']>=.6 and match['robust_fixed_map_TM_similarity_fullN']>=.8)})
 primary_pass=bool(len(ds)==2 and len(ass)==2 and all(f['predicted_sequences']==3 for f in ds+ass) and dag['pTM']['family_mean_pair_accuracy']>=.75 and aag['pTM']['family_mean_pair_accuracy']>=.75 and all(f['metrics']['pTM']['pairwise_accuracy']>=.5 and (f['metrics']['pTM']['Spearman'] or 0)>0 and f['metrics']['pTM']['WT_vs_strong_correct']==1 for f in ass))
 complete=all(r['status']=='SUCCEEDED' for r in comparison);same=[r['id'] for r in sensitivity if r['confident_same_fold_despite_large_destabilization']]
 outcome='A_PROMISING_FEASIBILITY_ONLY' if primary_pass else ('B_STRUCTURAL_CONFIDENCE_NOT_STABILITY_RANKING' if complete and same else ('C_PROTOCOL_OR_FAMILY_DEPENDENT' if complete else 'D_INSUFFICIENT_PREDICTION_EVIDENCE'))
 write('STABILITY_RANKING_RESULTS.json',{'families':ds+ass,'DEVELOPMENT':dag,'AUDIT':aag,'primary':'pTM','primary_predeclared_gate_pass':primary_pass if complete else None,'ranking_status':'EVALUATED' if complete else 'NOT_EVALUATED_INCOMPLETE_PREDICTIONS','outcome':outcome,'sequence_rows':comparison,'equal_family_weighting':True,'effect_size_and_posterior_interval_pair_rules':cfg['pair_eligibility'],'no_independent_variant_bootstrap':True,'no_reward_fit':True,'inference_scope':'query-only; four natural parent families, 40..72-supporting domain study, one fixed seed','stability_label_sign':'larger dG_ML more stable; ddG_ML=mutant dG-authorWT median; no sign flip'})
 write('REWARD_CANDIDATE_COMPARISON.json',{'primary':'pTM','signals':METRICS,'fixed_composite':'.5*pTM+.5*mean_pLDDT/100','development':dag,'audit':aag,'no_formula_selection_on_audit':True,'fitted_parameters':0,'historical_R1b_reward_not_retuned':True,'production_reward_GO':False})
 write('INDEPENDENT_AUDIT_RESULTS.json',{'gate_pass_within_feasibility_scope':primary_pass if complete else None,'status':'EVALUATED' if complete else 'NOT_EVALUATED','audit_families_with_predictions':sum(f['predicted_sequences']>0 for f in ass),'outcome':outcome,'families':ass,'aggregate':aag,'audit_labels_sha256':sha(R/'analysis/AUDIT_LABELS_SEALED.json'),'before_audit_lock_sha256':sha(lock) if lock.exists() else None,'analysis_policy_sha256':sha(R/'configs/SELECTION_AND_ANALYSIS_PREDECLARED_v2.json'),'audit_families':len(ass),'audit_labels_used_for_fitting':False,'general_certification':False,'reason':'two audit families and one seed; operational family separation and no matched MSA evidence do not certify broad or novel-sequence transfer'})
 write('analysis/MUTATION_SENSITIVITY.json',{'rows':sensitivity,'highconfidence_samefold_large_destabilizations':same,'seed_consistency':'not measured; one seed under fixed matched protocol; no extra calls authorized after12','experimental_mutant_structures_available':False,'J1_geometry_used_in_score':False})
 print(json.dumps({'outcome':outcome,'primary_development':dag['pTM'],'primary_audit':aag['pTM'],'confident_large_destabilization_cases':same,'complete':complete}),flush=True)
if __name__=='__main__':main()
