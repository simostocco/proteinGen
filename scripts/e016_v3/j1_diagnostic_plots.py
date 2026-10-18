"""Small standalone measured figures, with all caches confined to evidence root."""
import os,json
from pathlib import Path
R=Path(__file__).resolve().parents[3];os.environ['MPLCONFIGDIR']=str(R/'runtime/cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
z=json.load(open(R/'LOCAL_COMPONENT_TRAIN_VALIDATION.json'));tags=list(z);fig,axs=plt.subplots(1,3,figsize=(12,3.5))
for ax,c in zip(axs,['distance','angle','torsion']):
 for role in ['TRAIN','VALIDATION']:ax.plot(range(6),[z[t][role]['components'][c]['target_weighted']['mean'] for t in tags],marker='o',label=role)
 ax.set_title(c+' NLL');ax.set_xticks(range(6),tags,rotation=40,ha='right');ax.set_ylabel('nats / valid target');ax.grid(alpha=.25)
axs[0].legend();fig.suptitle('Frozen native-prefix panels; historical/configuration stages, not a controlled ablation');fig.tight_layout();fig.savefig(R/'COMPONENT_NLL_COMPARISON.png',dpi=150);plt.close(fig)
cal=json.load(open(R/'TORSION_CALIBRATION.json'));fig,ax=plt.subplots(figsize=(7,3))
for t in ['original','pilot500','pilot2500']:
 v=cal[t]['VALIDATION'];ax.plot(np.arange(10)+.5,np.array(v['PIT_histogram10'])/20629,label=t)
ax.axhline(.1,color='black',linestyle='--',label='calibrated uniform PIT');ax.set_xlabel('PIT bin');ax.set_ylabel('fraction');ax.set_title('Torsion VALIDATION probability integral transform');ax.legend();fig.tight_layout();fig.savefig(R/'TORSION_PIT_COMPARISON.png',dpi=150)
print('PLOTS_COMPLETE')
