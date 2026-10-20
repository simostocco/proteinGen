from r1e_common import *
import pandas as pd
from scipy.stats import rankdata,mannwhitneyu
p=R/'data/Garcia_DataS1.csv';df=pd.read_csv(p);assert len(df)==614 and df['article'].nunique()==11 and int(df['pdb'].sum())==44
assert df['Experimental Success'].notna().all();assert df['Name'].nunique()==614
if (R/'ESM2_SEQUENCE_PLAUSIBILITY.json').exists():
 es=read('ESM2_SEQUENCE_PLAUSIBILITY.json')['scores'];em={r['index']:r['mean_logp'] for r in es if r.get('group')=='Garcia_designs'};df['ESM2_sampled_masked_mean_logp']=[em.get(i,np.nan) for i in range(len(df))]
metrics={'AlphaFold2 pLDDT 3 recycles':1,'AlphaFold2 pLDDT 25 recycles':1,'ESMFold pLDDT':1,'MPNN Score':-1,'AlphaFold2 PAE':-1,'AlphaFold3 pLDDT':1}
if 'ESM2_sampled_masked_mean_logp' in df:metrics['ESM2_sampled_masked_mean_logp']=1

def auc(y,s):
 n=int(y.sum());m=len(y)-n
 return float((rankdata(s)[y].sum()-n*(n+1)/2)/(n*m)) if min(n,m)>0 else None

def ap(y,s):
 order=np.argsort(-s,kind='stable');y=y[order];s=s[order];ends=np.r_[np.where(np.diff(s)!=0)[0],len(s)-1];tp=np.cumsum(y)[ends];precision=tp/(ends+1);recall=tp/max(1,y.sum());value=float(np.sum(np.diff(np.r_[0,recall])*precision));return value,[dict(recall=float(r),precision=float(p),oriented_threshold=float(s[e])) for r,p,e in zip(recall,precision,ends)]

def evaluate(frame,c,direction):
 f=frame[frame[c].notna()].copy();y=f['Experimental Success'].to_numpy(bool);s=f[c].to_numpy(float)*direction;overall=auc(y,s);apv,pr=ap(y,s);study={};topology={}
 for group,col in [(study,'article'),(topology,'Fold')]:
  for name,g in f.groupby(col):
   yy=g['Experimental Success'].to_numpy(bool);ss=g[c].to_numpy(float)*direction;group[str(name)]=dict(n=len(g),successes=int(yy.sum()),AUROC=auc(yy,ss),prevalence=float(yy.mean()))
 valid=[v['AUROC'] for v in study.values() if v['AUROC'] is not None];rng=np.random.default_rng(871);boot=[];macro=[];names=sorted(f.article.unique()); blocks={n:(f.loc[f.article==n,'Experimental Success'].to_numpy(bool),f.loc[f.article==n,c].to_numpy(float)*direction) for n in names}
 for _ in range(1000):
  picked=rng.choice(names,len(names),replace=True);a=auc(np.concatenate([blocks[n][0] for n in picked]),np.concatenate([blocks[n][1] for n in picked]))
  if a is not None:boot.append(a)
  vv=[study[n]['AUROC'] for n in picked if study[n]['AUROC']is not None]
  if vv:macro.append(float(np.mean(vv)))
 # Precision-recall is descriptive of actual class prevalence; no threshold selected for reward.
 threshold=float(np.quantile(s,.75));top=y[s>=threshold];fail=f[(s>=threshold)&~y]
 return dict(n=len(f),successes=int(y.sum()),prevalence=float(y.mean()),AUROC=overall,AUROC_study_cluster_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),study_macro_AUROC=float(np.mean(valid)) if valid else None,study_macro_bootstrap95=np.quantile(macro,[.025,.975]).tolist(),average_precision=apv,PR_curve=pr,per_study=study,per_topology=topology,success_score=desc(f.loc[y,c]),failure_score=desc(f.loc[~y,c]),oriented_rank_biserial=2*overall-1 if overall is not None else None,top_quartile_success_fraction=float(top.mean()),top_quartile_failed_count=len(fail),high_scoring_failures=fail[['Name','article','Fold','Sequence Length','Experimental Success',c]].head(12).to_dict('records'))
results={}
for label,f in [('all614',df),('paper_main570_excluding_solved_NMR_pdb',df[~df['pdb']])]:
 results[label]={c:evaluate(f,c,d) for c,d in metrics.items()};print(label,{c:round(v['AUROC'],3) for c,v in results[label].items()},flush=True)
audit=dict(rows=len(df),unique_names=df.Name.nunique(),unique_sequences=df.sequence.nunique(),studies=df.article.nunique(),topologies=df.Fold.nunique(),successes=int(df['Experimental Success'].sum()),columns=df.columns.tolist(),missing_counts=df.isna().sum().to_dict(),canonical_sequences=int(df.sequence.map(lambda s:set(s)<=set(AA)).sum()),length_agreement=int((df.sequence.str.len()==df['Sequence Length']).sum()),solved_NMR_exclusion='pdb True44, not NMR True87',source_sha256=sha(p),original_labels_preserved=True,metric_directions=metrics,article_counts=df.article.value_counts().to_dict(),statistical_limitations='study clusters11; topology/study overlap; variants/designs are not independent families; study bootstrap exploratory not causal generalization')
write('EXPERIMENTAL_DESIGN_SUCCESS_BENCHMARK.json',dict(audit=audit,results=results,no_new_predictor_fitted=True,published_Predicted_Probability_excluded=True))
(R/'EXPERIMENTAL_DESIGN_SUCCESS_DATA_AUDIT.md').write_text('# Garcia experimental design-success data\n\nOfficial PMC supplementary file PRO-35-e70453-s002.csv (publisher DataS1 alias), retrieved through Europe PMC supplementaryFiles ZIP.\n\n'+json.dumps(audit,indent=2)+'\n\nSuccess is the authors’ conjunction of expression, solubility, monomericity and intended secondary structure, not a folding free-energy measurement. Missing component observations remain missing; they are not recoded as experimental failures. All614 original overall labels are present. The44 pdb=True records are the authors’ solved-NMR exclusion; NMR=True alone includes87 and is not the exclusion predicate. Published fitted Predicted Probability and Predicted Success are retained in the raw CSV but excluded from evaluator/reward comparisons.\n\nProteinMPNN score is mean negative log probability using the best AF2-predicted full backbone, not a CA-only J1 score. AF2 model2.1.1 predictions choose highest mean pLDDT among5models,3/25recycles; PAE from pTM protocol; AF3 and ESMFold are different protocols. Code/data availability and license are in the preserved article XML.\n')
