"""Domain-independent option interface, separate from protein observations."""
from dataclasses import dataclass,field
from enum import Enum
from typing import Any,Mapping,Protocol
class Kind(str,Enum):
 CONTINUE='CONTINUE';REWIND='REWIND';REGENERATE='REGENERATE';STOP='STOP'
@dataclass(frozen=True)
class Option:
 kind:Kind
 subgoal:int=0
 arguments:Mapping[str,Any]=field(default_factory=dict)
class ControlledWorld(Protocol):
 def observation(self)->Mapping[str,Any]:...
 def transition(self,option:Option)->Mapping[str,Any]:...
class ProteinInterface:
 def __init__(self,world,continuation):self.world=world;self.continuation=continuation
 def observation(self):return self.world.observe()
 def transition(self,option):
  if option.kind==Kind.CONTINUE:i=option.subgoal
  elif option.kind==Kind.REWIND:i=4+[4,8,16].index(int(option.arguments['residues']))
  elif option.kind==Kind.REGENERATE:i=7
  elif option.kind==Kind.STOP:i=8
  else:raise ValueError('option')
  r=self.world.option(i,self.continuation,steps=int(option.arguments.get('steps',1)));return dict(observation=self.observation(),transition=r)
