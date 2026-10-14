import os, sys
from pathlib import Path
import torch
import pytest

R = Path(os.environ["E016_ARTIFACT_ROOT"])
torch.set_num_threads(1)


@pytest.fixture
def d_tmp(tmp_path):
    assert str(tmp_path).startswith("/mnt/d/")
    return tmp_path
