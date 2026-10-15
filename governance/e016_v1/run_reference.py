"""D-only, bounded stdlib synthetic test runner. Never imports protein/data tools."""
import os
import argparse
import re
from pathlib import Path
import hashlib
import json
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "source"
RESERVE = 10 * 2**30
PEAK_BUDGET = 256 * 2**20

def main():
    if not str(ROOT).startswith("/mnt/d/"):
        raise RuntimeError("D-only evidence root")
    before = shutil.disk_usage(ROOT).free
    if before < RESERVE + PEAK_BUDGET:
        raise RuntimeError("reserve plus stage budget unavailable")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"run[0-9]{3}", args.run_id):
        raise RuntimeError("versioned run ID required")
    out = ROOT / "outputs" / args.run_id
    out.mkdir(exist_ok=False)
    env = os.environ.copy()
    for key in ["TMPDIR", "TEMP", "TMP", "SQLITE_TMPDIR", "JOBLIB_TEMP_FOLDER"]:
        env[key] = str(ROOT / "tmp")
    for key in ["XDG_CACHE_HOME", "HF_HOME", "TORCH_HOME", "PYTHONPYCACHEPREFIX", "JAX_COMPILATION_CACHE_DIR", "TRITON_CACHE_DIR"]:
        env[key] = str(ROOT / "cache" / key.lower())
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-B", "-m", "unittest", "discover", "-s", "governance/e016_v1", "-p", "test_framework.py", "-v"]
    start = time.monotonic()
    with (out / "unit_tests.log").open("x") as log:
        result = subprocess.run(command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=60)
    log = (out / "unit_tests.log").read_text()
    status = {"scope": "SYNTHETIC_ONLY", "exit_code": result.returncode, "status": "PASS" if result.returncode == 0 else "FAIL", "command": command, "seconds": time.monotonic()-start, "child_max_RSS_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024, "disk_free_before": before, "disk_free_after": shutil.disk_usage(ROOT).free, "disk_reserve_bytes": RESERVE, "projected_peak_budget_bytes": PEAK_BUDGET, "environment": {k:v for k,v in env.items() if k in ["TMPDIR","TEMP","TMP","SQLITE_TMPDIR","JOBLIB_TEMP_FOLDER","XDG_CACHE_HOME","HF_HOME","TORCH_HOME","PYTHONPYCACHEPREFIX","JAX_COMPILATION_CACHE_DIR","TRITON_CACHE_DIR","CUDA_VISIBLE_DEVICES","PYTHONDONTWRITEBYTECODE"]}, "test_log_sha256": hashlib.sha256(log.encode()).hexdigest(), "real_data_processed": False, "training_or_inference": False}
    (out / "TEST_RESULTS.json").write_text(json.dumps(status, indent=2)+"\n")
    print(log)
    print(json.dumps(status, indent=2))
    return result.returncode

if __name__ == "__main__":
    raise SystemExit(main())
