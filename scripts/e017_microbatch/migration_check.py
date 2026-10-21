from pathlib import Path
import copy,json,sqlite3,torch,time
from protein_joint_sac.crisp_v3 import admission as A,optimization as O,networks as N
from protein_joint_sac.crisp_v3.execution import migrate_execution,scientific_config,enable_loader
from protein_joint_sac.crisp_v3.data import Population,CoverageSampler
from protein_joint_sac.crisp_v3.storage import Replay,rng_state
from protein_joint_sac.crisp_v3.recovery import state_digest
from protein_joint_sac.crisp_v3.prior import file_sha
R=Path(__file__).resolve().parents[3];c=json.load(open(R/'configs/OPTIMIZED_EXECUTION.json'));p=torch.load(c['execution_parent']['path'],map_location='cpu',weights_only=False);enable_loader(c);system=N.System('research');learner=O.Learner(system,**c['optimizer']);A.configure_initialization(system,learner,c['schedule']);pop=Population();samplers=dict(AA=CoverageSampler(pop),geometry=CoverageSampler(pop),goals=A.InitGoals(pop,1705,json.load(open(c['native_limits']))));run=R/'benchmarks/migration_fixture';run.mkdir(exist_ok=False);replay=Replay(run/'replay.sqlite');effective={**c,'data_input_hashes':pop.input_hashes(),'source_commit':'ENGINEERING_MIGRATION_CHECK_ONLY'};counters=migrate_execution(c,system,learner,samplers,effective,replay,run)
replay.close();replay=Replay(run/'replay.sqlite');assert replay.db.execute('PRAGMA integrity_check').fetchone()[0]=='ok';replay.close()
assert state_digest(system.state_dict())==state_digest(p['system']);assert state_digest(rng_state())==state_digest(p['rng']);assert counters['updates']==250;assert learner.steps=={'actor':250};assert state_digest({k:s.state_dict() for k,s in samplers.items()})==state_digest(p['samplers'])
# Only explicitly execution-scoped fields differ. No research override is accepted.
bad=copy.deepcopy(c);bad['optimizer']['lr']*=2;assert scientific_config(bad)!=scientific_config(p['config'])
result=dict(status='PASS',next_global_update=251,model_and_targets_exact=True,optimizer_scheduler_exact=True,samplers_exact=True,RNG_exact=True,replay_integrity=True,actor_optimizer_only=True,all_scientific_fields_unchanged=True,LR_change_rejected=True,LR_restored=learner.optimizers['actor'].param_groups[0]['lr'],scheduler_last_epoch=learner.schedulers['actor'].last_epoch,config_sha256=__import__('protein_joint_sac.crisp_v3.storage',fromlist=['digest']).digest(c),optimizer_calls=0)
with open(R/'results/EXECUTION_MIGRATION_CHECK.json','x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
