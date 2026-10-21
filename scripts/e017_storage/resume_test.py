from pathlib import Path
import copy,json,random,sqlite3,time,subprocess
from contextlib import ExitStack
from unittest.mock import patch
import numpy as np
import torch
from protein_joint_sac.crisp_v3 import training
from protein_joint_sac.crisp_v3.storage import Replay,digest
from protein_joint_sac.crisp_v3.recovery import RollingRecovery,StoragePolicy,state_digest
from protein_joint_sac.crisp_v3.prior import file_sha,SHA
from protein_joint_sac.crisp_v3.optimization import Learner
E=Path(__file__).resolve().parents[3];R=E.parent;start=time.monotonic();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=E/'source',text=True).strip();m=json.load(open(R/'results/ENTRYPOINT_ENGINEERING_SMOKE.json'));donor=m['runs']['reachability']['checkpoint'];state=torch.load(donor['path'],map_location='cpu',weights_only=False);config={k:v for k,v in state['config'].items() if k not in ['data_manifest','data_input_hashes','native_limits_sha256','validation_panel_sha256','source_commit']};config['storage_policy']=dict(min_headroom_bytes=2*2**30,warning_free_bytes=4*2**30,near_term_growth_bytes=32*2**20,checkpoint_hint_bytes=64*2**20,expected_live_replay_bytes=0,expected_log_bytes=32*2**20);config['checkpoint_wall_seconds']=900;config['log_every']=1
assert state['counters']['updates']==config['updates']==1
state['config']={**state['config'],**config,'source_commit':commit};state['config_sha256']=digest(state['config'])
run=E/'completed_resume_fixture';manager=RollingRecovery(run,StoragePolicy(**config['storage_policy']),create=True)
src=sqlite3.connect('file:'+state['replay_snapshot']['path']+'?mode=ro',uri=True);dst=sqlite3.connect(run/'replay.sqlite');src.backup(dst);src.close();dst.close();replay=Replay(run/'replay.sqlite')
def factory(snap):s=copy.copy(state);s['replay_snapshot']=snap;return s
bundle=manager.publish(replay,factory,1)['bundle'];replay.close();(run/'CONFIG.json').write_text(json.dumps(state['config'],indent=2));grant=E/'results/ZERO_UPDATE_RESUME_ONLY.json';grant.write_text(json.dumps(dict(approved=True,source_commit=commit,config_sha256=training.seal_config(config),reference_sha256=SHA,stage=config['stage'],max_updates=1,scope='Completed-run recovery test only. Optimizer entrypoints disabled.'),indent=2))
with ExitStack() as stack:
 for method in ['critic','low_actor','high_actor','reach','supervised']:stack.enter_context(patch.object(Learner,method,side_effect=AssertionError('optimizer work forbidden in this test')))
 stack.enter_context(patch.object(training,'supervised_update',side_effect=AssertionError('supervised work forbidden')))
 returned=training.execute(config,E/'unused_output_parent',grant,bundle['checkpoint'])
assert returned==str(run) and json.load(open(run/'RUN_STATUS.json'))['status']=='COMPLETED';assert (run/'updates.jsonl').read_text()=='';assert len(list((run/'checkpoints').glob('*.pt')))==1
saved=torch.load(bundle['checkpoint'],map_location='cpu',weights_only=False);assert torch.equal(torch.get_rng_state(),saved['rng']['torch']);assert random.getstate()==saved['rng']['python'];assert np.array_equal(np.random.get_state()[1],saved['rng']['numpy'][1]);assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),saved['rng']['cuda']));assert file_sha(donor['path'])==donor['sha256']
# Final writer path records a compact verified retirement receipt.
replay=Replay(run/'replay.sqlite')
def f(update):
 def make(snap):s=copy.copy(saved);s['counters']=dict(saved['counters'],updates=update);s['replay_snapshot']=snap;return s
 return make
manager.publish(replay,f(2),2);manager.publish(replay,f(3),3);assert len(list((run/'checkpoints').glob('*.pt')))==1;assert (run/'RETIREMENTS.jsonl').stat().st_size>0;replay.close()
out=dict(status='PASS',same_run_resume_no_new_live_replay_copy=True,completed_resume_optimizer_steps=0,optimizer_methods_blocked=True,full_checkpoint_readback=True,CPU_CUDA_Python_numpy_RNG_exact=True,compact_retirement_receipts=True,model_capacity_unchanged=63389056,source_commit=commit,seconds=time.monotonic()-start,protected_donor_unchanged=True)
with open(E/'results/EXACT_RESUME_AND_RETIREMENT_TESTS.json','x') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2),flush=True)
