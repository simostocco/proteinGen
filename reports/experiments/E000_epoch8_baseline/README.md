# E000 Epoch-8 Baseline

This directory stores the reproducible generative evaluation for the selected
epoch-8 checkpoint:

`outputs/recovered_full_b2_v/checkpoints/final_validation_selected.pt`

The model is conditioned only on requested length `N`. Generated samples are
not reconstructions of matched controls; real structures are distributional
references.

## Smoke Test

Use a temporary output directory for a small trial:

```bash
python scripts/evaluate_generated_ensemble.py \
  --checkpoint outputs/recovered_full_b2_v/checkpoints/final_validation_selected.pt \
  --weights ema \
  --config configs/train_recovered_full_v.yaml \
  --normalization-file data/full/processed_recovery/normalization_train.json \
  --train-manifest data/full/splits_recovered_all_structures/train.parquet \
  --reference-manifest data/full/splits_recovered_all_structures/validation.parquet \
  --output-dir /tmp/proteingen_E000_smoke \
  --length-samples 64:1,128:1 \
  --master-seed 8000 \
  --contact-threshold 8.0 \
  --num-triangles 256 \
  --novelty-candidate-count 4 \
  --control-count 4 \
  --real-length-tolerance 8 \
  --diversity-pair-limit 32 \
  --bootstrap-iterations 20 \
  --no-plots \
  --resume
```

## Full Evaluation

```bash
python scripts/evaluate_generated_ensemble.py \
  --checkpoint outputs/recovered_full_b2_v/checkpoints/final_validation_selected.pt \
  --weights ema \
  --config configs/train_recovered_full_v.yaml \
  --normalization-file data/full/processed_recovery/normalization_train.json \
  --train-manifest data/full/splits_recovered_all_structures/train.parquet \
  --reference-manifest data/full/splits_recovered_all_structures/validation.parquet \
  --output-dir reports/experiments/E000_epoch8_baseline \
  --length-samples 64:100,128:100,256:100,384:50,500:25 \
  --master-seed 8000 \
  --contact-threshold 8.0 \
  --num-triangles 2048 \
  --novelty-candidate-count 32 \
  --workers 1 \
  --control-count 64 \
  --real-length-tolerance 8 \
  --diversity-pair-limit 1000 \
  --bootstrap-iterations 200 \
  --plots \
  --resume
```

## Interrupt And Resume

Ctrl-C preserves completed generated NPZ files and the SQLite state database in
`state/evaluation_state.sqlite`. Resume with the same command and `--resume`.
Changing protocol inputs such as checkpoint hash, manifests, lengths, seed
schedule, thresholds, or weights fails clearly instead of mixing outputs.

## Inspect Outputs

- `protocol.json`: hashes, runtime, seed schedule, thresholds, and status.
- `generated/`: atomically written generated samples grouped by length.
- `metrics/validity_per_sample.parquet`: generated-sample diagnostics.
- `metrics/real_control_metrics.parquet`: matched validation controls.
- `metrics/distribution_matching.csv`: generated-vs-real descriptor summaries.
- `metrics/diversity_pairs.parquet`: deterministic within-length pair samples.
- `metrics/diversity_summary.csv`: per-length diversity summaries.
- `metrics/novelty_per_sample.parquet`: approximate generated novelty.
- `metrics/novelty_calibration.parquet`: real-control calibration.
- `metrics/summary.json`: compact non-composite overview.
- `figures/`: optional figures with sample counts.

## Verify Completion

```bash
python - <<'PY'
import json
from pathlib import Path

protocol = json.loads(Path("reports/experiments/E000_epoch8_baseline/protocol.json").read_text())
print(protocol["status"])
print(protocol["sample_counts_by_length"])
PY
```

E000 does not call samples “physically stable”. It reports numerical,
EDM-compatible, chain-like, and protein-like diagnostics with explicit
empirical criteria.


## Calibrated Final Analysis

Run the analysis-only finalizer without regenerating samples:

```bash
python scripts/finalize_generated_ensemble_analysis.py \
  --evaluation-dir reports/experiments/E000_epoch8_baseline \
  --output-dir reports/experiments/E000_epoch8_baseline \
  --real-quantile 0.99 \
  --pair-limit 1000 \
  --bootstrap-iterations 200 \
  --contact-threshold 8.0 \
  --seed 8000 \
  --plots \
  --resume
```

Calibrated outputs include empirical real-like geometry thresholds, corrected
novelty by requested length, real-vs-real diversity calibration, deterministic
sample rankings, figures, `metrics/calibrated_summary.json`, and
`E000_FINAL_REPORT.md`. The original `protocol.json` and raw metric files are
not modified by the finalizer.
