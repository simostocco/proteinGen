"""Rolling storage engineering tests; only new run-owned fixture artifacts retire."""
from pathlib import Path
import copy,json,shutil,time
from unittest.mock import patch
import torch
from protein_joint_sac.crisp_v3 import recovery as rec
from protein_joint_sac.crisp_v3.storage import Replay
from protein_joint_sac.crisp_v3.prior import file_sha
E=Path(__file__).resolve().parents[3];R=E.parent;start=time.monotonic();old=json.load(open(R/'results/ENTRYPOINT_ENGINEERING_SMOKE.json'));parent=old['runs']['reachability']['checkpoint'];original_sha=file_sha(parent['path']);state=torch.load(parent['path'],map_location='cpu',weights_only=False)
run=E/'fixture_run';manager=rec.RollingRecovery(run,rec.StoragePolicy(checkpoint_hint_bytes=32*rec.MIB),create=True);replay=Replay(run/'replay.sqlite');fingerprints={};counts=[]
def factory(update):
 def make(snap):
  s=dict(state);s['counters']=dict(state['counters'],updates=update);s['replay_snapshot']=snap;fingerprints[update]=rec.state_digest(s);return s
 return make
def metric(value):return dict(metric='ENGINEERING_SYNTHETIC_NOT_MODEL_QUALITY',value=value,mode='min',panel_sha256='a'*64)
def count():return (len(list((run/'checkpoints').glob('*.pt'))),len(list((run/'replay_snapshots').glob('*.sqlite'))))
a=manager.publish(replay,factory(1),1,metric(.5));assert count()==(1,1);b=manager.publish(replay,factory(2),2,metric(.7));assert count()==(2,1) and b['snapshot_reused'];old_latest=b['bundle']['checkpoint'];best=a['bundle']['checkpoint']
# New replay content requires a new snapshot; old best snapshot remains.
replay.episode('e',dict(prefix='engineering'),dict(N=64),'test','SYNTHETIC_ENGINEERING');c=manager.publish(replay,factory(3),3,metric(.6));assert not Path(old_latest).exists() and Path(best).exists() and count()==(2,2)
# Crash before the single latest/best index commit cannot retire old recovery.
real=rec.atomic_json
def fail(path,value):
 if Path(path).name=='RECOVERY_INDEX.json' and value.get('latest',{}).get('update')==4:raise OSError('ENGINEERING injected index-publication failure')
 return real(path,value)
with patch.object(rec,'atomic_json',fail):
 try:manager.publish(replay,factory(4),4,metric(.4));raise AssertionError('fault not injected')
 except OSError as error:assert 'injected' in str(error)
assert manager.current()['latest']['update']==3 and Path(best).exists();manager._recover_pending();assert count()==(2,2)
# Best promotion retires BOTH superseded old bundles only after verified commit.
d=manager.publish(replay,factory(5),5,metric(.4));assert manager.current()['latest']==manager.current()['best'] and count()==(1,1);counts.append(count())
e=manager.publish(replay,factory(6),6,metric(.9));assert count()==(2,1);loaded=torch.load(e['bundle']['checkpoint'],map_location='cpu',weights_only=False);assert rec.state_digest(loaded)==fingerprints[6]
for key in ['system','optimizers','schedulers','samplers','rng','steps','counters','config','replay_snapshot']:assert key in loaded
# Refuse arbitrary paths, adoption of old runs, changed metric definitions.
for rel in ['../../historical.pt',parent['path'],'checkpoints/../outside.pt']:
 try:manager.safe(rel);raise AssertionError('unsafe path accepted')
 except ValueError:pass
try:rec.RollingRecovery(Path(parent['path']).parent,create=True);raise AssertionError('historical run adopted')
except ValueError:pass
try:manager.publish(replay,factory(7),7,{**metric(.1),'panel_sha256':'b'*64});raise AssertionError('selection protocol changed')
except ValueError:pass
p=rec.StoragePolicy();ad=p.admission(run,500*rec.MIB,free=3*rec.GIB);assert ad['admitted'] and ad['warning'];assert not p.admission(run,500*rec.MIB,free=2*rec.GIB+100*rec.MIB)['admitted']
# The normal forecast pause window can still publish coherent exact current state.
free=p.min_headroom_bytes+32*rec.MIB+2*rec.MIB;disk=shutil.disk_usage(run)
with patch.object(rec.shutil,'disk_usage',return_value=disk._replace(free=free)):
 try:manager.runway(replay);raise AssertionError('future write admitted without runway')
 except rec.StoragePause:pass
 paused=manager.publish(replay,factory(8),8,near_growth=0)
assert rec.state_digest(torch.load(paused['bundle']['checkpoint'],map_location='cpu',weights_only=False))==fingerprints[8]
assert manager.current()['best']['update']==5 and count()==(2,1)
replay.close();assert file_sha(parent['path'])==original_sha
out=dict(status='PASS',latest_best_atomic_index=True,unchanged_replay_snapshot_reused=True,changed_replay_bound=True,retire_only_verified_superseded_owned=True,failed_publication_keeps_old_recovery=True,pending_crash_recovery=True,promoted_best_no_duplicate_copy=True,complete_state_readback=True,forecast_warning_and_pause=True,emergency_boundary_save_exact=True,historical_checkpoint_unchanged=True,final_artifact_counts=dict(checkpoints=count()[0],replay_snapshots=count()[1]),seconds=time.monotonic()-start,optimizer_steps=0,J1_updates=0)
with open(E/'results/ROLLING_STORAGE_TESTS.json','x') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2),flush=True)
