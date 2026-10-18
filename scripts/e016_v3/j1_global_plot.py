"""Standalone scientific plots from sealed result tables; no model execution."""
import os,json
from pathlib import Path
R=Path('/mnt/d/Simone/proteinGen_data/e016_j1_global_pretraining_20261017_v1')
os.environ['MPLCONFIGDIR']=str(R/'runtime/cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
def main():
 h=json.load(open(R/'TRAIN_VALIDATION_PERPLEXITY_HISTORY.json'));g=json.load(open(R/'NATIVE_VS_GENERATED_GLOBAL_GEOMETRY.json'));out=R/'figures';out.mkdir(exist_ok=True)
 x=np.array([v['additional_update'] for v in h])/1000
 fig,ax=plt.subplots(1,2,figsize=(11,4))
 for role,color in [('TRAIN','#355c9a'),('VALIDATION','#bf5b32')]:ax[0].plot(x,[v[role]['perplexity'] for v in h],'-o',label=role,color=color)
 ax[0].axhline(20,color='gray',ls=':',label='Uniform')
 ax[0].axhline(17.1359974043,color='gray',ls='--',label='TRAIN Markov on VALIDATION')
 ax[0].set(xlabel='Additional updates (thousands)',ylabel='Native-prefix AA perplexity');ax[0].legend(fontsize=8)
 ax[1].plot(x,[v['TRAIN_VALIDATION_NLL_gap'] for v in h],'-o',color='#76538e');ax[1].set(xlabel='Additional updates (thousands)',ylabel='VALIDATION minus TRAIN AA NLL (nats)')
 fig.suptitle('J1: fixed 2,947-identity / 23,576-decision panels');fig.tight_layout();fig.savefig(out/'TRAIN_VALIDATION_PERPLEXITY.png',dpi=180);plt.close(fig)
 lengths=[64,128,256,384,500];rawbase=json.load(open('/mnt/d/Simone/proteinGen_data/e016_partial_supervision_20261016_v1/runs/J1-FULL/SELECTED_GENERATION.json'));rawnew=json.load(open(R/'LATEST_FREE_GENERATION.json'))
 fig,ax=plt.subplots(1,2,figsize=(11,4))
 for role,key,color in [('Original','baseline','#355c9a'),('Trained +30k','latest_actual_policy','#bf5b32')]:
  ax[0].plot(lengths,[g[key][str(n)]['Rg_A']['mean'] for n in lengths],'-o',label=role,color=color)
  ax[1].plot(lengths,[g[key][str(n)]['clashes_per100_residues']['mean'] for n in lengths],'-o',label=role,color=color)
 native=[np.median([v['metrics']['Rg_A'] for v in g['native']['TRAIN'][str(n)]['rows']]) for n in lengths]
 ax[0].plot(lengths,native,'--s',label='Native TRAIN window median',color='gray')
 ax[0].set(xlabel='Sequence length',ylabel='Mean generated Rg (Angstrom)');ax[1].set(xlabel='Sequence length',ylabel='Nonlocal CA clash pairs / 100 residues')
 for a in ax:a.legend(fontsize=8);a.grid(alpha=.15)
 fig.suptitle('32 independent samples per length; point estimates, not folding evidence')
 fig.tight_layout();fig.savefig(out/'GLOBAL_GEOMETRY_COMPARISON.png',dpi=180);plt.close(fig)
 print('PLOTS_WRITTEN',flush=True)
if __name__=='__main__':main()
