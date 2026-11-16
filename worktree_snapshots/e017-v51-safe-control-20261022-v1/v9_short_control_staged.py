"""Prespecified shorter-length feasibility control; never replaces N64."""
import runpy,time,sys
from pathlib import Path
import numpy as np
R=Path('/mnt/d/Simone/proteinGen_data/e017_v9_property_steering_20261023_v1');c=runpy.run_path(str(R/'source/scripts/e017_v9/core.py'));p=runpy.run_path(str(R/'source/scripts/e017_v9/policy.py'));read=c['read'];save=c['save']
def run():
 assert not read(R/'LOW_LEVEL_ADMISSION.json')['PASS'];assert read(R/'CALIBRATION_ADMISSION.json')['PASS'] and read(R/'REWARD_TESTS.json')['PASS'];spec=read(R/'REWARD_SPECIFICATION.json');population=read(R/'manifests/TRAIN_REFERENCE.json')['records'][:256];goals=[];exclusions=[]
 for rec in sorted(population,key=lambda x:c['ns']('N32control',x['identity'])):
  if len(goals)==16:break
  x=np.array(rec['coordinates'])[:32];seq=rec['sequence'][:32];ex=c['exposure'](x,spec['proxy']);a,b=(4,20) if ex[4:12].mean()>=ex[20:28].mean() else (20,4);gap=ex[a:a+8].mean()-ex[b:b+8].mean();motifs=[seq[i:i+2] for i in range(a,a+7) if seq[i]!=seq[i+1]]
  if gap<.15 or not motifs:continue
  goal=dict(goal_id='V9_N32_'+rec['identity'][:16],witness_identity=rec['identity'],protected_group=rec['group'],N=32,exposed=[a,a+8],buried=[b,b+8],exposure_min=float(np.floor(ex[a:a+8].mean()*20)/20),burial_max=float(np.ceil(ex[b:b+8].mean()*20)/20),contrast_min=.1,motif=motifs[c['ns']('motif',rec['identity'])%len(motifs)],motif_scope='anywhere within exposedregion',native_coordinates_provided=False,condition='TRAIN-only N32 feasibility; not primary N64')
  if not c['score'](seq,x,goal,spec)['success']:exclusions.append(rec['identity']);continue
  goals.append(goal)
 save('manifests/N32_CONTROL_GOALS.json',dict(goals=goals,requested=16,exclusions=exclusions,rule='same TRAIN-calibrated reward; regions4:12 and20:28; independent hash selection; no threshold relaxation',primary=False))
 assert len(goals)==16
 engine=p['Engine']();start=time.perf_counter()
 for policy in ['A_J1','B_V3']:
  for i,g in enumerate(goals):
   c['POLICY'].require(R,2*2**20);name=f'results/N32_{policy}_{i:03}.json'
   if (R/name).exists():continue
   rows,_=p['sample'](engine,g,spec,policy,1,'N32_CONTROL');save(name,dict(goal=g,records=rows));print('N32',policy,i+1,16,flush=True)
 engine.verify();save('N32_CONTROLS_COMPLETED.json',dict(trajectories=32,goals=16,K=1,TRAIN_only=True,not_primary_N64=True,actor_updates=0,calls=dict(engine.calls),elapsed_seconds=time.perf_counter()-start,peak_GPU_bytes=__import__('torch').cuda.max_memory_allocated()))
if __name__=='__main__':run()
