"""Standalone scientific plots from already-published metrics; no model or optimization."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[3]
def read(name):return json.loads((R/name).read_text())
def main():
 rows=read('SEQUENCE_TRAIN_VALIDATION_METRICS.json');x=np.array([v['new_update'] for v in rows]);fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
 for ax,cond,label in [(axes[0,0],'FREE','Geometry-unavailable next-AA'),(axes[0,1],'NATIVE','Native-prefix next-AA')]:
  for role,style in [('TRAIN','--'),('VALIDATION','-')]:ax.plot(x,[v[cond+'_'+role]['perplexity'] for v in rows],style,marker='o',label=role)
  if cond=='FREE':ax.axhline(17.303353,color='gray',ls=':',label='Previous research');ax.axhline(17.303353*np.exp(-.02),color='green',ls=':',label='New target')
  ax.set(title=label,ylabel='Perplexity',xlabel='Additional optimizer updates');ax.legend(fontsize=8);ax.grid(alpha=.2)
 ax=axes[1,0]
 for role,style in [('TRAIN','--'),('VALIDATION','-')]:ax.plot(x,[v['MASKED_'+role]['identity_mean_NLL'] for v in rows],style,marker='o',label=role)
 ax.set(title='Bidirectional masked reconstruction',ylabel='Identity-weighted masked NLL (nats)',xlabel='Additional optimizer updates');ax.legend(fontsize=8);ax.grid(alpha=.2)
 ax=axes[1,1];ax.plot(x,[v['NATIVE_VALIDATION']['geometry_NLL'] for v in rows],'-o',label='VALIDATION local NLL');ax.axhline(-.10796287496467151,color='green',ls=':',label='New allowance');ax.axhline(-.113708968,color='gray',ls='--',label='Original threshold');ax.set(title='Native local geometry',ylabel='Historical per-decision NLL',xlabel='Additional optimizer updates');ax.legend(fontsize=8);ax.grid(alpha=.2)
 fig.suptitle('J1 bidirectional multitask continuation: distinct evaluation conditions',fontsize=12)
 for ext in ['png','pdf']:fig.savefig(R/('LEARNING_CURVES.'+ext),dpi=170)
 plt.close(fig)
 if not (R/'FREE_GENERATION_COMPARISON.json').exists():return
 gen=read('FREE_GENERATION_COMPARISON.json');pairs=gen['paired_vs_accepted_parent'];fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True);lengths=[64,128,256,384,500]
 for key,ax,label,mult in [('clashes_per100_residues',axes[0],'Clashes per100 residues',1.),('Rg_over_N_cuberoot',axes[1],'Rg/N^(1/3) (Å)',1.),('long_contact_fraction',axes[2],'Long-range contacts (percentage points)',100.)]:
  for tag,values in pairs.items():
   y=np.array([values[str(N)][key]['mean'] for N in lengths])*mult;lo=np.array([values[str(N)][key]['bootstrap_95pct'][0] for N in lengths])*mult;hi=np.array([values[str(N)][key]['bootstrap_95pct'][1] for N in lengths])*mult;ax.errorbar(lengths,y,yerr=[y-lo,hi-y],marker='o',capsize=3,label='+'+tag)
  ax.axhline(0,color='black',lw=.8);ax.set(xlabel='Generated length',ylabel='Change versus parent',title=label);ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.suptitle('Common-noise generation comparison; paired95% bootstrap intervals; Cα proxies',fontsize=11)
 for ext in ['png','pdf']:fig.savefig(R/('GENERATION_COMPARISON.'+ext),dpi=170)
 plt.close(fig);print('Scientific plots published; no model loaded or trained.')
if __name__=='__main__':main()
