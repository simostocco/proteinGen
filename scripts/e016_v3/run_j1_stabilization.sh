#!/bin/bash
set -eu
ROOT=/mnt/d/Simone/proteinGen_data/e016_j1_torsion_stabilization_20261018_v1
mkdir -p "$ROOT/generation" "$ROOT/runtime/tmp" "$ROOT/runtime/cache"
export TMPDIR="$ROOT/runtime/tmp" TEMP="$ROOT/runtime/tmp" TMP="$ROOT/runtime/tmp" SQLITE_TMPDIR="$ROOT/runtime/tmp" JOBLIB_TEMP_FOLDER="$ROOT/runtime/tmp"
export XDG_CACHE_HOME="$ROOT/runtime/cache" HF_HOME="$ROOT/runtime/cache/hf" TORCH_HOME="$ROOT/runtime/cache/torch" PYTHONPYCACHEPREFIX="$ROOT/runtime/cache/pyc" JAX_COMPILATION_CACHE_DIR="$ROOT/runtime/cache/jax" TRITON_CACHE_DIR="$ROOT/runtime/cache/triton" CUDA_CACHE_PATH="$ROOT/runtime/cache/cuda" TORCHINDUCTOR_CACHE_DIR="$ROOT/runtime/cache/inductor"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=6 CUBLAS_WORKSPACE_CONFIG=:4096:8
export E016_ARTIFACT_ROOT="$ROOT" PYTHONPATH="$ROOT/source/src:$ROOT/source/scripts/e016_v3"
exec /home/simostocco/miniforge3/envs/proteingen/bin/python -B "$@"
