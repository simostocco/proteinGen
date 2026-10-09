"""Prepared unchanged E007 matched control; only after primary completion."""

# ruff: noqa: I001
import e014_bootstrap  # noqa: F401
import sys
from pathlib import Path
import yaml
from protein_distance_diffusion.training.e014_runtime import sha
from run_e014_scaleup import main

if __name__ == "__main__":
    control = yaml.safe_load(Path("configs/e014_matched_e007_control_v1.yaml").read_text())
    for name in ("shared_training_config", "model_config"):
        if sha(control[name]) != control[name + "_sha256"]:
            raise ValueError("matched comparator configuration changed; refusing execution")
    sys.argv.extend(["--model", "e007_control"])
    main()
