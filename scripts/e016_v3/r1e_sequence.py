from r1e_common import *
import collections,math,itertools,subprocess,time
from scipy.spatial.distance import jensenshannon
panel=read('FROZEN_J1_GENERATION_PANEL.json');rows=panel['rows'];results=[]
for r in rows:
 s=r['sequence'] or '';n=len(s);c=collections.Counter(s);p=np.array([c[a]/max(1,n) for a in AA]);h=float(-sum(v*math.log2(v) for v in p if v));runs=[len(list(g)) for _,g in itertools.groupby(s)];w=min(12,n)
 low=sum(-sum((v/w)*math.log2(v/w) for v in collections.Counter(s[i:i+w]).values())<2.0 for i in range(n-w+1))/max(1,n-w+1)
 v={k:r[k] for k in ['id','group','N','index']};v.update(valid=n==r['N'] and set(s)<=set(AA),composition=p.tolist(),entropy_bits=h,max_run=max(runs or [0]),low_complexity_window_fraction=low,hydrophobic_fraction=sum(c[a] for a in 'AVILMFWY')/max(1,n),polar_fraction=sum(c[a] for a in 'STNQ')/max(1,n),charged_fraction=sum(c[a] for a in 'DEKR')/max(1,n),approx_net_charge=c['K']+c['R']-c['D']-c['E'],repeat_triplet_fraction=1-len(set(s[i:i+3] for i in range(n-2)))/max(1,n-2));results.append(v)
groups=sorted(set(x['group'] for x in rows));summary={};div={};geometry={}
for g in groups:
 summary[g]={};div[g]={}
 for N in [64,128,256,384,500]:
  rr=[x for x in results if x['group']==g and x['N']==N];ss=[r['sequence'] for r in rows if r['group']==g and r['N']==N and r['sequence']]
  if not rr:continue
  summary[g][str(N)]={k:desc([x[k] for x in rr]) for k in ['entropy_bits','max_run','low_complexity_window_fraction','hydrophobic_fraction','polar_fraction','charged_fraction','approx_net_charge','repeat_triplet_fraction']}
  ref=[r['sequence'] for r in rows if r['group']=='natural_TRAIN' and r['N']==N];dev={}
  for k in [1,2,3]:
   def hist(ss):return collections.Counter(s[i:i+k] for s in ss for i in range(len(s)-k+1))
   a,b=hist(ss),hist(ref);keys=sorted(a.keys()|b.keys());av=np.array([a[k] for k in keys],float);bv=np.array([b[k] for k in keys],float);dev[str(k)]=float(jensenshannon(av/av.sum(),bv/bv.sum(),base=2)**2) if ref else None
  summary[g][str(N)]['kmer_JS_bits_vs_TRAIN']=dev
  arr=np.array([list(s) for s in ss]);pi=[float(np.mean(a==b)) for a,b in itertools.combinations(arr,2)];nn=[max([float(np.mean(a==b)) for j,b in enumerate(arr) if i!=j] or [0]) for i,a in enumerate(arr)];counts=np.array(list(collections.Counter(ss).values()))/len(ss)
  div[g][str(N)]=dict(n=len(ss),valid=sum(x['valid'] for x in rr),unique=len(set(ss)),duplicates=len(ss)-len(set(ss)),effective_unique=float(1/(counts**2).sum()),pairwise_ungapped_identity=desc(pi),nearest_ungapped_identity=desc(nn),note='aligned global equal-position identity, not optimized sequence alignment; local alignment reported separately')
for g in ['J1_7500','J1_500']:
 geometry[g]={}
 for N in [64,128,256,384,500]:
  rr=[x for x in rows if x['group']==g and x['N']==N];mm=[r['geometry_metrics'] for r in rr if r['geometry_metrics']]
  geometry[g][str(N)]={k:desc([v[k] for v in mm]) for k in ['Rg_A','Rg_over_N_cuberoot','clashes_per100_residues','long_contact_fraction','torsion_positive_fraction','torsion_resultant','degenerate_frames']};geometry[g][str(N)]['completion']=sum(r['status']=='COMPLETE' for r in rr);geometry[g][str(N)]['adjacent_A_mean']=desc([v['adjacent_A']['mean'] for v in mm]);geometry[g][str(N)]['angle_rad_mean']=desc([v['angle_rad']['mean'] for v in mm])
write('SEQUENCE_COMPOSITION_AND_COMPLEXITY.json',dict(definitions=dict(low_complexity='12-residue sliding Shannon entropy<2bits; descriptive',charge='K+R-D-E; excludes pH/titration'),rows=results,summary=summary))
write('GENERATION_VALIDITY_AND_DIVERSITY.json',dict(summary=div,attempts=320,failed_attempts=[r for r in rows if r['group'].startswith('J1') and r['status']!='COMPLETE']))
write('AUXILIARY_STRUCTURE_EVALUATION.json',dict(metrics=geometry,source='immutable corrected sampler cached CA metrics; seqsep>=3 clash<2A, seqsep>=24 contact<8A',foldability_claim=False))
with (R/'data/queries.fasta').open('w') as f:
 for r in rows:
  if r['sequence']:f.write(f">{r['id']}\n{r['sequence']}\n")
print('sequence metrics done',len(results),flush=True)
