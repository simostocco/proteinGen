import random
from types import SimpleNamespace
import numpy as np
import pytest
from protein_joint_sac.data.scaled_supervised import Sampler
from j1_global_train import MixedSampler,Engine

def fixture():
 seqs={"a":"A"*20,"b":"G"*30,"p":"C"*25}
 base=SimpleNamespace(seqs=seqs,jointids={"TRAIN":["a","b"]},ids={"TRAIN":["a","b"]},owners={"a":["ao"],"b":["bo"]})
 data=SimpleNamespace(base=base,seqs=seqs,ids={"TRAIN":["p"]},owners={"TRAIN":{"p":["po"]}},rows={"po":{}},cache={"po":{"seen":np.array([False]*5+[True]*20)}})
 return data,{"complete":Sampler(base).state_dict()}

def test_mixture_is_identity_first_and_exactly56_complete8_partial():
 data,parent=fixture();s=MixedSampler(data,parent);out=s.draw()
 assert len(out)==64 and sum(o=="po" for h,o,t in out)==8
 assert s.counts["complete"]==56 and s.counts["partial"]==8
 assert all(0<=t<len(data.seqs[h]) for h,o,t in out)
 assert s.cursor==64 and len(s.seen)==3

def test_sampler_recovery_reproduces_actual_following_decisions():
 data,parent=fixture();s=MixedSampler(data,parent);s.draw();saved=s.state_dict();expected=s.draw()
 r=MixedSampler(data,parent);r.load_state_dict(saved);actual=r.draw()
 assert actual==expected and r.state_dict()==s.state_dict()

def test_configuration_does_not_silently_change_effective_batch():
 with pytest.raises(ValueError,match="frozen effective batch"):
  Engine(None,{"effective_batch":32,"partial_decisions_per_batch":8,"live_batch":16})
 with pytest.raises(ValueError,match="frozen effective batch"):
  Engine(None,{"effective_batch":64,"partial_decisions_per_batch":16,"live_batch":16})


def test_sampler_snapshot_has_no_mutable_aliases():
 import copy
 data,parent=fixture();s=MixedSampler(data,parent);s.draw();snapshot=s.state_dict();saved=copy.deepcopy(snapshot);s.draw()
 assert snapshot==saved
