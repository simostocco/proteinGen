from r1e_common import *
from scipy.stats import rankdata
import pandas as pd
p=read('FROZEN_J1_GENERATION_PANEL.json');e=read('ESM2_SEQUENCE_PLAUSIBILITY.json');nov=read('SEQUENCE_NOVELTY_AND_HOMOLOGY.json');df=pd.read_csv(R/'data/Garcia_DataS1.csv');checks={}
checks['320_unique_attempt_keys']=len({r['id']for r in p['rows']if r['group'].startswith('J1')})==320
checks['exact_canonical_lengths']=all(len(r['sequence'])==r['N']and set(r['sequence'])<=set(AA)for r in p['rows'])
checks['TRAIN_shuffles_composition_equal']=all(sorted(r['sequence'])==sorted(next(z['sequence']for z in p['rows']if z['group']=='natural_TRAIN'and z['N']==r['N']and z['index']==r['index']))for r in p['rows']if r['group']=='shuffled')
checks['one_mask_per_forward_protocol']=read('configs/SCORING_PROTOCOL.json')['masked_positions']==16
checks['scores_all_sequences']=len(e['scores'])==1371 and all(np.isfinite(r['masked_logp']).all()for r in e['scores'])
checks['J1_main_and_parent_hashes_preserved']=all(sha(r['path'])==r['sha256']for r in p['checkpoints'])
checks['cache_hashes_preserved']=all(sha(r['path'])==r['sha256']for r in p['cache_sources'])
checks['experimental614_unique_designs']=len(df)==614 and df.Name.nunique()==614
checks['experimental_labels269successes']=int(df['Experimental Success'].sum())==269
checks['author_main_subset570']=int((~df['pdb']).sum())==570
checks['PLL_normalized16']=all(abs(r['mean_logp']-np.mean(r['masked_logp']))<1e-12 for r in e['scores'])
checks['search_TRAIN_positive_control_all160_exact']=all(r['exact_full_sequence_match']for r in nov['rows']['TRAIN']if r['group']=='natural_TRAIN')
checks['search_VALIDATION_positive_control_all117_exact']=all(r['exact_full_sequence_match']for r in nov['rows']['VALIDATION']if r['group']=='natural_VALIDATION')
# Independently verify AUROC from pairwise success/failure comparisons including ties.
b=read('EXPERIMENTAL_DESIGN_SUCCESS_BENCHMARK.json');f=df[~df['pdb']];a=f.loc[f['Experimental Success'],'AlphaFold2 pLDDT 3 recycles'].to_numpy();z=f.loc[~f['Experimental Success'],'AlphaFold2 pLDDT 3 recycles'].to_numpy();direct=float(((a[:,None]>z).sum()+.5*(a[:,None]==z).sum())/(len(a)*len(z)));checks['AUROC_matches_independent_pairwise']=abs(direct-b['results']['paper_main570_excluding_solved_NMR_pdb']['AlphaFold2 pLDDT 3 recycles']['AUROC'])<1e-12
# Equal-length stratified bootstrap; each length contributes one fifth, seed-paired per length.
rng=np.random.default_rng(916);diff={}
for N in [64,128,256,384,500]:
 a=sorted([r for r in e['scores']if r['group']=='J1_7500'and r['N']==N],key=lambda r:r['index']);z=sorted([r for r in e['scores']if r['group']=='J1_500'and r['N']==N],key=lambda r:r['index']);diff[N]=np.array([x['mean_logp']-y['mean_logp']for x,y in zip(a,z)])
boot=np.mean([rng.choice(v,(10000,len(v))).mean(1)for v in diff.values()],axis=0);overall=dict(mean=float(np.mean([v.mean()for v in diff.values()])),bootstrap95=np.quantile(boot,[.025,.975]).tolist(),definition='equal5length weighting; seed-paired bootstrap within length; conditional on fixed16position scoring; no evolutionary family-independence assertion')
write('EVALUATION_CORRECTNESS_TESTS.json',dict(checks=checks,all_pass=all(checks.values()),AUROC_independent_value=direct,overall_paired_ESM_delta=overall));assert all(checks.values()),checks
print('checks',checks,'ESM delta',overall)
