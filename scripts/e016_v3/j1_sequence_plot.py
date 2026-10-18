"""Standalone exported figures, D:-only Matplotlib cache."""
import os,json
from pathlib import Path
R=Path('/mnt/d/Simone/proteinGen_data/e016_j1_sequence_first_20261018_v1')
os.environ['MPLCONFIGDIR']=str(R/'runtime/cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
def main():
 (R/'figures').mkdir(exist_ok=True);h=json.load(open(R/'TRAIN_VALIDATION_PERPLEXITY_HISTORY.json'))
 fig,ax=plt.subplots(1,2,figsize=(10,4))
 u=np.array([v['update'] for v in h])/1000
 for role in ['TRAIN','VALIDATION']:ax[0].plot(u,[v[role]['perplexity'] for v in h],'-o',label=role)
 ax[0].axhline(15.7330813943,color='gray',ls=':',label='Historical unselected 60k VAL')
 ax[0].set(xlabel='Additional updates (thousands)',ylabel='AA perplexity');ax[0].legend(fontsize=8)
 ax[1].plot(u,[v['VALIDATION']['AA_NLL']-v['TRAIN']['AA_NLL'] for v in h],'-o')
 ax[1].set(xlabel='Additional updates (thousands)',ylabel='VALIDATION minus TRAIN NLL')
 for a in ax:
  for x in [2.5,5]:
   if x<=max(u):a.axvline(x,color='gray',ls='--',lw=.8)
  a.grid(alpha=.25)
 fig.suptitle('Same frozen native-prefix panels; stage changes are not causal ablations');fig.tight_layout();fig.savefig(R/'figures/TRAIN_VALIDATION.png',dpi=180);plt.close(fig)
 g=json.load(open(R/'NATIVE_VS_GENERATED_GEOMETRY.json'));native=json.load(open(g['native_reference_path']))['native']['TRAIN'];candidates=g['candidates'];last=max(candidates,key=int);new=candidates[last]['summary'];old=g['baseline'];ns=[64,128,256,384,500]
 fig,axes=plt.subplots(1,3,figsize=(13,4))
 for a,key,title in zip(axes,['Rg_A','clashes_per100_residues','long_contact_fraction'],['Radius of gyration (A)','Clashes / 100 residues','Long-range contact fraction']):
  for name,source in [('Original quality-selected',old),('Latest continuation',new)]:
   a.plot(ns,[source[str(N)][key]['mean'] for N in ns],'-o',label=name)
  a.plot(ns,[np.median([r['metrics'][key] for r in native[str(N)]['rows']]) for N in ns],':',color='black',label='Native TRAIN median')
  a.set(xlabel='Requested length',title=title);a.grid(alpha=.25)
 axes[0].legend(fontsize=7)
 fig.suptitle('32 free-running samples/length; means shown, uncertainty in JSON; CA proxies only')
 fig.tight_layout();fig.savefig(R/'figures/GLOBAL_GEOMETRY.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
