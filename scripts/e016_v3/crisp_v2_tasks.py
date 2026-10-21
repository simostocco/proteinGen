from crisp_v2_common import *
import subprocess,time
from protein_joint_sac.data.scaled_supervised import Data,canonical,internal
from protein_joint_sac.data.accepted_supervised import CoordinateReader
from partial_finalize import metrics
start=time.monotonic();guard();audit=[]
for p,h in [(MAIN,MAIN_SHA),(D/'e016_j1_torsion_stabilization_20261018_v1/replay/checkpoint_00500_full.pt','9208263300fb9e88519ff0d40e6d9d29a9744b709d9a6181819cd0894b593000'),(D/'e016_partial_supervision_20261016_v1/runs/J1-FULL/checkpoint_20000.pt','9e9215d3720ffc73edb0c3e382db25332f91cb26b0c8d38d01668e383b1e3810')]:
 actual=sha(p);assert actual==h;audit.append(dict(path=str(p),sha256=actual))
data=Data(False);reader=CoordinateReader(data.proposal.external);tasks=[];native=[];provenance=[];used=set()
for split,count in [('TRAIN',16),('VALIDATION',8)]:
 ids=sorted([h for h in data.jointids[split]if 64<=len(data.seqs[h])<=96],key=lambda h:hashlib.sha256((h+'E017v2').encode()).hexdigest());picked=[]
 for h in ids:
  group=data.proposal.sequences[h]['protected_group_id']
  if group in used:continue
  o=data.owners[h][0];row=data.rows[o];x=canonical(reader.read(row));g=internal(x);assert np.isfinite(x).all()and len(x)==len(data.seqs[h]);used.add(group);picked.append(h);s=data.seqs[h];N=len(s);met=metrics(x)
  native.append(dict(identity=h,split=split,sequence=s,CA=x.tolist(),geometry=g.tolist(),metrics=met,observation=o,group=group))
  for name in ['P1','P2']:
   tasks.append(dict(id=name+'_'+h[:16],identity=h,group=group,split=split,kind=name,N=N,seed=2026102100+len(tasks),initial_sequence=s[:3],initial_CA=x[:3].tolist(),motifs={str(int(N/3)):s[int(N/3)],str(int(2*N/3)):s[int(2*N/3)]},endpoint=x[-1].tolist(),orientation=frame(x).tolist() if name=='P2'else None,waypoint=dict(index=N//2,position=x[N//2].tolist())if name=='P2'else None,position_tolerance_A=8.,orientation_tolerance_deg=45.,waypoint_tolerance_A=8.))
  provenance.append(dict(identity=h,observation=o,entry_id=row['entry_id'],coordinate_path=row['coordinate_path'],coordinate_row=row['coordinate_row'],coordinate_kind=row['coordinate_kind'],source_sha=row['source_sha'],group=group,split=split,coordinates_sha256=reader.external[str(Path(row['coordinate_path']).resolve())]))
  if len(picked)==count:break
 assert len(picked)==count
# Native calibration: entire TRAIN task parents only, no filtering of observed anomalies.
tr=[r for r in native if r['split']=='TRAIN'];all_d=np.concatenate([np.linalg.norm(np.diff(np.array(r['CA']),axis=0),axis=1)for r in tr]);limits=dict(adjacent_low_A=float(np.quantile(all_d,.001)-.15),adjacent_high_A=float(np.quantile(all_d,.999)+.15),clashes_per100_max=float(max(r['metrics']['clashes_per100_residues']for r in tr)+2),Rg_normalized_low=float(min(r['metrics']['Rg_over_N_cuberoot']for r in tr)*.75),Rg_normalized_high=float(max(r['metrics']['Rg_over_N_cuberoot']for r in tr)*1.5),no_degenerate_frames=True,calibration='16TRAIN complete parents; native extrema retained; bounds are feasibility constraints, not physical stability certification')
write('tasks/PUBLIC_TASKS.json',tasks);write('tasks/HIDDEN_NATIVE_SUPERVISION.json',native);write('tasks/SOURCE_PROVENANCE.json',provenance);write('configs/NATIVE_CALIBRATION.json',limits)
write('configs/FROZEN_PROTOCOL.json',dict(version='E017_CRISP_V2_feasibility1',task_seed=2026102100,TRAIN_parents=16,VALIDATION_parents=8,tasks=48,adapter_updates=80,batch=16,lr=.0003,AA_loss=1.,geometry_loss='TRAIN-gradient calibrated max0.1; target25% AA gradient norm',KL_A=.01,KL_G=.01,gamma=.995,shaping_coefficient=.1,discount_clock='one tick per low-level residue action and per high-level rewind/stop; CONTINUE macro expands to residue ticks',administrative_max_ticks_factor=3,administrative_max_seconds_per_episode=30,scientific_correction_budget=None,supervised_sampling='50%random prefix,50%task motif prefix; TRAIN only',comparison_arms=['frozen_J1','supervised_goal_joint','AA_only','geometry_only','joint_subgoal_rewind','frozen_rewind'],evaluation='8VAL parents x2tasks x2matched seeds x6arms=192attempts',max_GPU_worker_seconds=1200,no_full_RL=True,no_J1_updates=True))
write('AUDIT.json',dict(checkpoints=audit,dataset_manifest_sha256=sha(P/'PROPOSAL_MANIFEST.json'),sequence_manifest_sha256=sha(P/'SEQUENCE_ASSIGNMENTS.parquet'),coordinate_manifest=data.proposal.manifest['coordinate_provenance_manifest'],prior_admission='verified by existing Data(False) independent accepted E016 receipts; legacy proposal pending label does not alter user-authorized bounded E017 scope',retired_prefilters_not_required=True,original_branch=subprocess.check_output(['git','-C','/mnt/d/Simone/proteinGen','branch','--show-current'],text=True).strip(),seconds=time.monotonic()-start,D_free_bytes=shutil.disk_usage(R).free))
print('TASKS',len(tasks),'calibration',limits,flush=True)
