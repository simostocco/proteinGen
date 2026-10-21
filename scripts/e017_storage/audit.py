"""Actual serialization and compact replay/log census; no optimizer/J1 updates."""
from pathlib import Path
import copy,json,math,os,resource,sqlite3,time
import torch
import pyarrow.parquet as pq
from protein_joint_sac.crisp_v3.networks import System
from protein_joint_sac.crisp_v3.optimization import Learner
from protein_joint_sac.crisp_v3.recovery import serialized_size
E=Path(__file__).resolve().parents[3];R=E.parent;start=time.monotonic()
old=json.load(open(R/'results/ENTRYPOINT_ENGINEERING_SMOKE.json'));donor_path=old['runs']['reachability']['checkpoint']['path'];donor=torch.load(donor_path,map_location='cpu',weights_only=False)
system=System('research');learner=Learner(system)
# Allocate the EXACT full AdamW state schema, without optimizer.step().
for optimizer in learner.optimizers.values():
 for group in optimizer.param_groups:
  for p in group['params']:optimizer.state[p]={'step':torch.tensor(0.),'exp_avg':torch.zeros_like(p),'exp_avg_sq':torch.zeros_like(p)}
state=dict(donor);state.update(profile='research',system=system.state_dict(),optimizers={k:o.state_dict() for k,o in learner.optimizers.items()},schedulers={k:v.state_dict() for k,v in learner.schedulers.items()},steps={k:0 for k in learner.optimizers})
actual=serialized_size(state);sampler_initial=serialized_size(state['samplers'])
# Upper coverage scenario uses actual TRAIN polymer lengths, not replicated sequences.
P=R.parent/'pdb_supervised_split_50id80cov_20261016_v1'
rows=pq.read_table(P/'SEQUENCE_ASSIGNMENTS.parquet',columns=['sequence_id','length'],filters=[('split','=','TRAIN')]).to_pylist();lengths={r['sequence_id']:r['length'] for r in rows};full=copy.deepcopy(state['samplers'])
for name in ['AA','geometry']:
 s=full[name];ids=[h for cat,v in s['orders'].items() if name=='AA' or cat=='complete' for h in v]
 s['position']={h:list(range(lengths[h])) for h in ids};s['cycles']={h:0 for h in ids};s['seen']={k:set(v) if name=='AA' or k=='complete' else set() for k,v in s['orders'].items()}
full_sampler=serialized_size(full);state['samplers']=full;full_checkpoint=serialized_size(state)
logs=[];dbs=[]
for p in sorted((R/'engineering_runs').glob('run_*/updates.jsonl')):
 lines=p.read_bytes().splitlines();logs.extend(len(x)+1 for x in lines)
for p in sorted((R/'engineering_runs').glob('run_*/replay.sqlite')):
 db=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True)
 tables={r[0] for r in db.execute('SELECT name FROM sqlite_master WHERE type="table"')}
 if 'events' not in tables:db.close();continue
 counts={t:db.execute('SELECT COUNT(*) FROM '+t).fetchone()[0] for t in ['episodes','events','transitions','policies']};payload=sum(db.execute('SELECT COALESCE(SUM(length(data)),0) FROM '+t).fetchone()[0] for t in ['episodes','events','transitions']);db.close();dbs.append(dict(path=str(p),bytes=p.stat().st_size,allocated_bytes=p.stat().st_blocks*512,logical_json_bytes=payload,counts=counts))
result=dict(model_counts=system.counts(),actual_serialized_research_checkpoint_bytes=actual,actual_full_identity_coverage_checkpoint_bytes=full_checkpoint,current_sampler_bytes=sampler_initial,full_identity_coverage_sampler_bytes=full_sampler,checkpoint_tensor_bytes=905613992,minimum_atomic_checkpoint_stage_bytes=full_checkpoint,serialization_method='torch.save to counting stream; full real architecture and complete allocated AdamW schema; no optimizer updates',donor_full_recovery_checkpoint=donor_path,log_row_bytes=logs,projected_10000_update_log_bytes=max(logs)*10000 if logs else None,replay_databases=dbs,seconds=time.monotonic()-start,host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,optimizer_steps=0,J1_updates=0)
with open(E/'results/FULL_CHECKPOINT_AND_GROWTH_AUDIT.json','x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k not in ['replay_databases','model_counts']},indent=2),flush=True)
