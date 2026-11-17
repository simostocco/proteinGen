# Validation — 2026-11-17

Checks use Python 3.12 and PyTorch 2.13.0+cu130 in the existing `proteingen` environment, with CUDA hidden and two CPU threads. No GPU training or expensive experiments are run. The environment is preinstalled; a fresh dependency installation is not claimed.

Final CPU pytest result: **151 passed, 3 skipped in 7.44 seconds**. All three skips are CUDA-only tests with CUDA deliberately hidden. The selection includes `tests/sequence`, all newly selected context/denoiser/objective tests, the two new public tests, and baseline U-Net/variable-length/symmetric diffusion/DDPM/training/distance/attention/triangle/global-equivariant tests. Full source/scripts/tests Ruff checks pass after import-order and exact-type style fixes. The fatal-error Ruff subset also passes. Research results were read from retained reports, not recomputed.

Historical data-dependent lifecycle, loader, pretrained-model, recovery and population-evaluation tests are not part of the portable selection. They require external manifests, checkpoints, executables or machine-specific artifact roots and remain unverified here. No unavailable scientific test is relabeled as passed. A full historical suite run is not claimed.

All 380 Python files under src/scripts/tests were parsed successfully. New entry-point links, selected file existence and imported original-evidence SHA256 values are tested. Two linked original aggregate figures were copied without modification to resolve the E016 report image links. No source installation from scratch, GPU numerical validation, raw-result reproduction or full historical suite execution is claimed.

The 17 selected source modules import successfully. `git diff --check` reports only two original blank lines at EOF in imported E017 reports; they are retained to preserve the original evidence bytes and hashes. Full Ruff remains clean. The new documentation test was rerun after importing the figures: 3 passed.
