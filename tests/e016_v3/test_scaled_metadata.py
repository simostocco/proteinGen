import time
from types import SimpleNamespace
import numpy as np
import torch
from scaled_train import Engine
from protein_joint_sac.rl.checkpointing import load_checked

def test_checkpoint_normalizes_numpy_validation_scalars(d_tmp):
 e=Engine.__new__(Engine);e.lineage={'scope':'SERIALIZATION_FIXTURE_ONLY'};e.model=torch.nn.Linear(2,1);e.opt=torch.optim.AdamW(e.model.parameters());e.scheduler=torch.optim.lr_scheduler.StepLR(e.opt,1);e.sampler=SimpleNamespace(state_dict=lambda:{'fixture':True});e.update=0;e.validation=[{'AA_NLL':np.float64(3.),'component':np.float64(-.1)}];e.elapsed_prior=0.;e.start=time.monotonic();e.band_counts=np.zeros(5,dtype=int);e.flagged=0
 p=d_tmp/'numpy_metadata.pt';e.checkpoint(p);s=load_checked(p);assert type(s['validation'][0]['AA_NLL']) is float;assert s['validation'][0]['component']==-.1
