"""Prepared unchanged E007 matched control; only after primary completion."""

# ruff: noqa: I001
import e014_bootstrap  # noqa: F401
import sys
from run_e014_scaleup import main

if __name__ == "__main__":
    sys.argv.extend(["--model", "e007_control"])
    main()
