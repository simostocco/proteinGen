from pathlib import Path
import sys,torch,time,json
from protein_joint_sac.crisp_v3.recovery import RollingRecovery,StoragePolicy
from protein_joint_sac.crisp_v3.storage import Replay
r=Path(sys.argv[1]);m=RollingRecovery(r,StoragePolicy(checkpoint_hint_bytes=32*2**20,near_term_growth_bytes=0),create=True);replay=Replay(r/'replay.sqlite');x=torch.zeros(4*1024**2)
try:
 m.publish(replay,lambda snap:dict(x=x,update=0,replay_snapshot=snap,config_sha256='fixture',config={'source_commit':'fixture'}),0)
 time.sleep(.5)
 m.publish(replay,lambda snap:dict(x=x,update=1,replay_snapshot=snap,config_sha256='fixture',config={'source_commit':'fixture'}),1)
 (r/'NEXT_OPTIMIZER_UPDATE_STARTED').write_text('Fixture marker only; no real optimizer exists.')
 time.sleep(5)
finally:
 replay.close();(r/'FINALLY_CLEANUP.json').write_text(json.dumps(dict(cleanup=True,optimizer_updates=0)))
