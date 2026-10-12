"""Exact pure E015 metric functions using the frozen E007 library implementation."""

import os, sys, ast, hashlib, json, math, time, pathlib, types

W = pathlib.Path("/mnt/d/Simone/proteinGen/.e014_worktree")
OUT = pathlib.Path("/mnt/d/Simone/proteinGen/e007_e015_epoch1_comparison_v1")
WORK = pathlib.Path("/mnt/d/Simone/proteinGen/.e015_worktree")
for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[key] = "1"
sys.path.insert(0, str(W / "src"))
import numpy as np, torch, pyarrow as pa, pyarrow.parquet as pq, yaml
from protein_distance_diffusion.models.e013_unconditional_equivariant_denoiser import local_geometry
from protein_distance_diffusion.evaluation.e014_evaluation import coordinate_batch, extended_metrics
from protein_distance_diffusion.evaluation.reverse_field_diagnostic import paired_state, vector_metrics
from protein_distance_diffusion.evaluation.marginal_distribution_diagnostic import (
    descriptor,
    NAMES,
    ReferenceStandardizer,
    Comparison,
    bandwidth,
    projections,
    distances,
)
from protein_distance_diffusion.training.e014_runtime import noise_seed, sha
from protein_distance_diffusion.data.e014_population import CoordinateCache
import protein_distance_diffusion

assert pathlib.Path(protein_distance_diffusion.__file__).resolve().is_relative_to(W / "src"), (
    "Wrong scientific source tree"
)
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
cfg = yaml.safe_load((WORK / "configs/e015_integrated_multiscale_parity_selfconditioned_v1.yaml").read_text())
SCALE = 12.22820347644835
GRID = [5, 50, 250, 400, 490]
STRATA = ["20-64", "65-128", "129-256", "257-384", "385-500"]
LENGTHS = [48, 96, 192, 320, 448]
source = (WORK / "docs/e015_serial_helpers.py").read_text()
wanted = {"clean", "write", "read", "table", "key", "intrinsic", "aligned_rmsd"}
nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in wanted]
assert {n.name for n in nodes} == wanted
# Exact function ASTs, with output root redirected only by globals.
from pathlib import Path

exec(compile(ast.Module(body=nodes, type_ignores=[]), "frozen_E015_pure_metric_functions", "exec"), globals())
