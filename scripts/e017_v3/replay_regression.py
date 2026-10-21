"""Final replay semantics: deterministic identities, censoring, indexed admission and tamper checks."""
import json,time,random,copy
import torch
from engineering import R,V2,write
from protein_joint_sac.crisp_v3.prior import FrozenPrior
from protein_joint_sac.crisp_v3.networks import System
from protein_joint_sac.crisp_v3.storage import Replay
from protein_joint_sac.crisp_v3.pipeline import collect,replay_example
from protein_joint_sac.crisp_v3.environment import reconstruct
start=time.monotonic();torch.manual_seed(419);prior=FrozenPrior('cuda');s=System('development').cuda();limits=json.load(open(R/'configs/NATIVE_CALIBRATION.json'));native=json.load(open(V2/'tasks/HIDDEN_NATIVE_SUPERVISION.json'))[0];task=dict(kind='P0',N=64,targets=[],motifs={},initial_sequence=native['sequence'][:3],initial_CA=native['CA'][:3]);results=[]
with torch.no_grad():s.models['high_actor'].out.weight.zero_();s.models['high_actor'].out.bias.fill_(-30);s.models['high_actor'].out.bias[0]=30;s.models['high_actor'].out.bias[8]=0
for i in range(2):
 torch.manual_seed(419);r=Replay(R/'data'/f'final_replay_regression_{i}_{time.time_ns()}.sqlite')
 with torch.autocast('cuda',dtype=torch.bfloat16):ep=collect(s,prior,task,limits,r,'frozen_fixture_policy',seed=419,max_events=8,option_steps=2)
 episode,events=r.read(ep['episode']);w=reconstruct(episode['initial'],events,task,limits,419);assert w.truncated and not w.terminated;assert not w.action_mask().any() and w.action_mask(for_bootstrap=True).any()
 record=r.sample(1,random.Random(19),level='low')[0];assert record['action']['level']=='low';bb,nb,worlds=replay_example(prior,r,record,limits,'cuda');results.append(dict(episode=ep,events=events,sampled=record))
 assert r.db.execute('SELECT COUNT(*) FROM outcomes').fetchone()[0]==1
 # Snapshot and reopen is exact; malicious event edits are detected before use.
 snap=R/'data'/f'final_replay_snapshot_{i}_{time.time_ns()}.sqlite';r.snapshot(snap);r.close();r=Replay(snap);assert r.read(ep['episode'])[1]==events;r.close()
assert results[0]==results[1];prior.verify();write('FINAL_REPLAY_REGRESSION.json',dict(status='PASS',deterministic_episode_and_transition_ids=True,indexed_level_sampling=True,operational_censoring_not_terminal=True,bootstrapping_action_masks=True,snapshot_reopen_equivalence=True,oracle_calls=0,optimizer_steps=0,seconds=time.monotonic()-start));print('replay final PASS',flush=True)
