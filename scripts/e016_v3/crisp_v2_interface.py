"""Domain-independent high-level CRISP interface. No protein/robotics dependencies."""
from dataclasses import dataclass,field
from enum import Enum
from typing import Protocol,Mapping,Any
import copy
class Kind(str,Enum):
 CONTINUE='CONTINUE';REWIND='REWIND';REGENERATE='REGENERATE';STOP='STOP'
@dataclass(frozen=True)
class HighAction:
 kind:Kind
 subgoal:str='endpoint'
 arguments:Mapping[str,Any]=field(default_factory=dict)
class ControlledWorld(Protocol):
 def observation(self)->Mapping[str,Any]:...
 def transition(self,action:HighAction)->Mapping[str,Any]:...
class ProteinWorld:
 """Protein-specific boundary; only already generated prefixes and explicit goals exposed."""
 def __init__(self,environment):self.environment=environment
 def observation(self):
  e=self.environment
  return dict(goal=copy.deepcopy(e.task),state_ref=f'prefix:{e.state.t}:branch:{e.branch}',prefix_sequence=''.join(e.state.residues),prefix_coordinates=e.state.ca.detach().cpu().tolist(),valid_masks=e.state.valid.cpu().tolist(),intervention_results=copy.deepcopy(e.interventions),primitive_ticks=e.ticks,complete=e.state.t==e.state.N)
 def transition(self,action):
  if not isinstance(action,HighAction):raise TypeError('typed high-level action required')
  kw=dict(action.arguments)
  if action.kind==Kind.CONTINUE:kw['subgoal']=dict(endpoint=0,waypoint=1)[action.subgoal]
  e=self.environment;before=len(e.reward_events);ticks=e.ticks;e.act(action.kind.value,**kw);events=e.reward_events[before:];raw=[x['reward']for x in events];out=self.observation();out['transition']=dict(primitive_rewards=raw,option_reward=sum(.995**i*r for i,r in enumerate(raw)),duration=e.ticks-ticks,discount=.995**(e.ticks-ticks),terminated=e.ended,administratively_truncated=False,accepted=bool(e.ended and events and events[-1]['terminal_success']==1.));return out
