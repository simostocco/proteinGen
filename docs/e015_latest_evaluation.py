# ruff: noqa: E402
"""Stage entry point; no framework import until the reviewed implementation is ready."""

import os
import sys
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["E015_LATEST_OUT"] = sys.argv[2]
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
while not Path(__file__).with_name(".e015_latest_ready").exists():
    time.sleep(10)
import e015_latest_core

e015_latest_core.main(sys.argv[1])
