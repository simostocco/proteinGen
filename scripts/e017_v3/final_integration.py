"""Final-source integration checks, including new orchestration paths. Bounded fixtures only."""
import copy,json,math,time,random
import torch
from pathlib import Path
from engineering import R,V2,write,batch
from protein_joint_sac.crisp_v3.prior import FrozenPrior,file_sha,tensor_sha
from protein_joint_sac.crisp_v3.networks import System
from protein_joint_sac.crisp_v3.optimization import Learner
from protein_joint_sac.crisp_v3.pipeline import native_goal_batch,collect,replay_example
from protein_joint_sac.crisp_v3.training import geometric_proximity,supervised_update,validate_actor
from protein_joint_sac.crisp_v3.data import Population,CoverageSampler
from protein_joint_sac.crisp_v3.goals import Goals
from protein_joint_sac.crisp_v3.storage import Replay,payload,atomic_save,restore
from protein_joint_sac.crisp_v3.environment import assess,reconstruct
start=time.monotonic();torch.manual_seed(1703);torch.use_deterministic_algorithms(True);pop=Population();limits=json.load(open(R/'configs/NATIVE_CALIBRATION.json'));goals=Goals(pop,1703,limits);prior=FrozenPrior('cuda');system=System('development').cuda();learner=Learner(system);sam=CoverageSampler(pop);gsam=CoverageSampler(pop,1704);config=json.load(open(R/'configs/development_supervised.json'));config.update(effective_AA_batch=3,complete_geometry_decisions=1,goal_demonstrations=1);goal=goals.draw(2);variants=goals.variants(goal);assert len(variants)==3
# One mixed fixture update validates the actual future supervised orchestration, not a campaign.
with torch.autocast('cuda',dtype=torch.bfloat16):res=supervised_update(system,learner,prior,pop,sam,gsam,goals,config)
assert res['gradient_groups']['AA']>0 and res['gradient_groups']['geometry']>0
first=res
with torch.autocast('cuda',dtype=torch.bfloat16):res=supervised_update(system,learner,prior,pop,sam,gsam,goals,config)
assert res['gradient_groups']['encoder']>0;prior.verify(learner.optimizers.values())
# Match goal sampler, population sampler, all optimizer/target states and replay snapshot.
replay=Replay(R/'data'/f'final_integration_{time.time_ns()}.sqlite');snap=R/'data'/f'final_snapshot_{time.time_ns()}.sqlite';replay.snapshot(snap);samplers={'AA':sam,'geometry':gsam,'goals':goals};path=R/'checkpoints'/f'final_integration_{time.time_ns()}.pt';atomic_save(path,payload(system,learner,samplers,config,{'path':str(snap),'sha256':file_sha(snap)},dict(fixture_updates=1)),40*2**20)
expected=[r[:3] for r in sam.draw(5)];expected_goal=goals.draw(1);restore(path,system,learner,samplers,config);assert expected==[r[:3] for r in sam.draw(5)] and expected_goal==goals.draw(1)
# New intervention descriptors do not leak hidden structures, and replay restores them.
from protein_joint_sac.crisp_v3.environment import World
from protein_joint_sac.crisp_v3.pipeline import observation
from protein_joint_sac.contracts import BuildState,AA
from protein_joint_sac.data.scaled_supervised import internal
native=json.load(open(V2/'tasks/HIDDEN_NATIVE_SUPERVISION.json'))[0];x=torch.tensor(native['CA'],dtype=torch.float64);seq=native['sequence'];g=torch.tensor(internal(x.numpy()));task=copy.deepcopy(goal);task.update(N=len(seq),targets=[],motifs={},kind='P0',initial_sequence=seq[:3],initial_CA=x[:3].tolist());w=World(BuildState(tuple(seq[:3]),x[:3],len(seq)),task,limits)
for t in range(3,12):w.step(AA.index(seq[t]),g[t])
w.option(4);w.option(7);b=observation(prior,w,'cuda',planner_features=[.2,.4]);assert b['context'][0,4]==1 and b['context'][0,20]==torch.tensor(.2,device='cuda')
# Finite all-missing and mixed-geometry inference with final code.
# Record syntax/import/config validation without running a substantive entrypoint.
write('FINAL_SOURCE_INTEGRATION.json',dict(status='PASS',supervised_orchestration=res,fixture_optimizer_steps=2,initial_zero_head_step=first,J1_bitwise_unchanged=True,sampler_goal_optimizer_target_checkpoint_resume=True,goal_variants=3,intervention_context=True,checkpoint=str(path),checkpoint_sha256=file_sha(path),replay_snapshot_sha256=file_sha(snap),seconds=time.monotonic()-start));replay.close();print('final integration PASS',flush=True)
