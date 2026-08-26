# Protein Distance Diffusion

Stage 1 implements unconditional generation of continuous protein residue-distance matrices conditioned on residue count:

`D ~ p_theta(D | N)`, where `D[i, j]` is the Euclidean distance in angstrom between the C-alpha atoms of residues `i` and `j`.

The model learns a distribution, not a deterministic function `D = f(N)`. Different noise seeds at the same length `N` should produce different plausible distance maps. This stage does not generate amino-acid sequences, reconstruct coordinates, predict coordinates, or train with triangle, EDM, bond-length, or steric losses.

## Repository Structure

- `src/protein_distance_diffusion/`: package source.
- `scripts/`: command-line entry points for download, preprocessing, splitting, statistics, training, sampling, summaries, and visualization.
- `configs/`: documented YAML defaults.
- `tests/`: unit and smoke tests with tiny synthetic fixtures.
- `requirements.txt` and `requirements-dev.txt`: convenience installation files; `pyproject.toml` remains the canonical dependency source.

## Distance Maps And Contact Maps

The training target is always the continuous matrix `D in R^(N x N)` in angstrom. Binary contact maps are derived only for diagnostics and visualization:

`C_ij^(tau) = 1[D_ij < tau]`

The documented visualization default is `tau = 8.0 A`. The diagonal is excluded from contact statistics, and near-diagonal sequence contacts can be excluded explicitly with `--exclude-near-diagonal`.

## Structure-Unconditioned Sequence Generation

The repository also contains an independent baseline for protein-chain sequence
generation:

`p(S | N) = product_i p(a_i | a_<i, N)`.

Here `S = (a_1, ..., a_N)` is a sequence of the 20 canonical amino acids and `N`
is the requested length. This model is deliberately separate from the distance
diffusion model. It does not consume distance matrices, coordinates, contact
maps, secondary structure, PDB metadata, or generated outputs from the U-Net
diffusion component. Future work may condition a sequence model on a valid
structure or distance matrix, but that is outside this baseline.

Tokenization uses one token per residue with the canonical alphabet:

```text
A C D E F G H I K L M N P Q R S T V W Y
```

The vocabulary also contains `<PAD>`, `<BOS>`, `<EOS>`, and `<UNK>`. By default,
training data must contain only canonical residues; unresolved or noncanonical
symbols are rejected and counted in the dataset audit. `<UNK>` can be enabled
explicitly for recovery experiments, but generated biological sequences never
include special tokens.

Training examples use teacher forcing. For a length-`N` sequence:

```text
input_ids  = [BOS, a_1, ..., a_(N-1)]
target_ids = [a_1, a_2, ..., a_N]
```

The baseline predicts next-amino-acid logits over the full 24-token vocabulary,
with padding masked out of the loss. Exact-length sampling masks `<PAD>`,
`<BOS>`, `<EOS>`, and `<UNK>` before sampling and stops only after exactly `N`
canonical residues have been generated.

Architecture:

```text
amino-acid tokens
  -> learned token embeddings
  -> RoPE causal self-attention + continuous length conditioning
  -> pre-normalization Transformer blocks
  -> final normalization
  -> next-amino-acid logits
```

The requested length is embedded continuously as `log(N) / log(max_length)` and
passed through `Linear(1, d_model) -> SiLU -> Linear(d_model, d_model)`. Each
Transformer block injects this length vector through a learned projection, so
the conditioning is not a discrete length bin or length-specific model.

Sequence data are expected to come from leakage-safe split manifests:

```text
data/full/splits/train.parquet
data/full/splits/validation.parquet
data/full/splits/test.parquet
```

Only `sequence` and `length` are required by the model, but metadata such as
`sample_id`, `pdb_id`, `chain_id`, `cluster_id`, `sequence_hash`, and
`experimental_method` is retained for auditing when present. The sequence
baseline also accepts FASTA files for small smoke tests and recovery situations.
It never loads `.npz` distance matrices. If only an unsplit manifest exists, the
preparation script refuses to random split it; provide homology-safe split files
or add an explicit MMseqs2 grouping step first.

Prepare or audit sequence splits:

```bash
python scripts/prepare_sequence_dataset.py \
  --config configs/sequence/data.yaml

python scripts/inspect_sequence_dataset.py \
  --manifest data/sequence/splits/train.parquet \
  --output-dir reports/sequence_dataset
```

Smoke train without starting a full run:

```bash
python scripts/train_sequence_transformer.py \
  --config configs/sequence/train.yaml \
  --max-optimizer-steps 10
```

Full sequence-baseline training command:

```bash
python scripts/train_sequence_transformer.py \
  --config configs/sequence/train.yaml
```

Resume from an optimizer-step checkpoint:

```bash
python scripts/train_sequence_transformer.py \
  --config configs/sequence/train.yaml \
  --resume-from outputs/sequence_baseline/checkpoints/last.pt
```

Sample exact-length canonical sequences:

```bash
python scripts/sample_sequences.py \
  --checkpoint outputs/sequence_baseline/checkpoints/best_validation.pt \
  --length 128 \
  --num-sequences 10 \
  --temperature 1.0 \
  --top-p 0.95 \
  --seed 42 \
  --output-dir outputs/sequence_baseline/samples/N128
```

Evaluate held-out language-model metrics:

```bash
python scripts/evaluate_sequence_model.py \
  --checkpoint outputs/sequence_baseline/checkpoints/best_validation.pt \
  --manifest data/sequence/splits/validation.parquet \
  --output-dir reports/sequence_baseline/validation
```

Validation reports token-weighted NLL, perplexity, sequence-weighted loss,
token accuracy, accuracy by residue type, NLL by position, and NLL as a
diagnostic function of `N`. Generated-sequence evaluation reports exact length
compliance, canonical validity, uniqueness, exact novelty against training
sequences, amino-acid composition, Jensen-Shannon divergence, Shannon entropy,
low-complexity fraction, homopolymer runs, and repeated-substring statistics.

Sequence plausibility does not prove foldability. PDB-derived sequence data are
structurally selected and biased, and homology-safe splitting is mandatory.
Proper biological validation remains future work and should include tools such
as structure prediction, predicted confidence, globularity/disorder checks,
structural diversity analysis, and aggregation or solubility screening.

## Environment Setup

Linux with Python 3.11 is the primary target:

```bash
cd /home/simostocco/proteinGen

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
```

Install the appropriate PyTorch build first. CPU example:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For CUDA, use the exact command from the official PyTorch installation selector for your driver/CUDA combination. Do not blindly install a CUDA-specific wheel on CPU-only machines.

Then install the project dependencies and editable package:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Verify:

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import MDAnalysis as mda; print('MDAnalysis:', mda.__version__)"
python -c "import gemmi; print('Gemmi:', gemmi.__version__)"
python -c "import protein_distance_diffusion; print('Package import successful')"

pytest -q
ruff check .
```

Use `source .venv/bin/activate` to reactivate the environment and `deactivate` to leave it.

Troubleshooting:

- If `python3.11` is missing, install it with your system package manager or use another Python `>=3.11` supported by the dependencies.
- If PyTorch reports NumPy ABI errors, reinstall compatible PyTorch/NumPy wheels inside the clean virtual environment.
- If `gemmi` is missing, install runtime requirements again; Gemmi is required for mmCIF.
- If CUDA is not detected, reinstall the PyTorch build matching your driver and CUDA runtime.
- MMseqs2 is an external executable, not a Python package. Install it separately and ensure `mmseqs` is on `PATH`.

## Parser Backends

Both Gemmi and MDAnalysis are runtime dependencies by design:

- `.cif`, `.mmcif`, `.cif.gz`, `.mmcif.gz`: Gemmi backend.
- `.pdb`, `.ent`, `.pdb.gz`, `.ent.gz`: MDAnalysis backend.
- future topology-plus-trajectory inputs: MDAnalysis backend.

All backends are converted into the same internal `ProteinSample` representation: PDB ID, model/frame ID, chain ID, sequence, residue identifiers, C-alpha coordinates `[N, 3]`, metadata, and later the stored distance matrix `[N, N]`.

## Pipeline

```mermaid
flowchart LR
  A[PDB mmCIF or PDB] --> B[backend selection]
  B --> C[chain filtering]
  C --> D[C-alpha extraction]
  D --> E[D calculation]
  E --> F[exact deduplication]
  F --> G[MMseqs2 clustering]
  G --> H[cluster-level splits]
  H --> I[training-only normalization]
  I --> J[variable-size DataLoader]
  J --> K[conditional U-Net diffusion]
  K --> L[generated distance matrices]
  L --> M[evaluation and visualization]
```

Distance matrices are never resized or interpolated. Batches are padded to a side length divisible by the U-Net downsampling factor; padded values are masked out. The sequence-separation channel is created during collation as `abs(i-j) / max(N-1, 1)`.

## First Real-Data Experiment

Query a reproducible pilot ID file with the RCSB Search API v2. This queries
experimental protein polymer entities from X-ray diffraction or electron
microscopy entries with protein entity length `<= 500`, retrieves more
candidates than requested, then uses a seeded deterministic sample to avoid
bias toward the first PDB IDs:

```bash
python scripts/query_rcsb.py \
  --output data/pilot/pdb_ids.txt \
  --num-entries 100 \
  --candidate-limit 1000 \
  --seed 42
```

The query script writes one PDB ID per line and saves the exact query JSON plus
retrieval metadata at `data/pilot/pdb_ids.txt.metadata.json`. It does not
download structures.

Download the bounded pilot dataset in mmCIF format:

```bash
python scripts/download_pdb.py \
  --ids-file data/pilot/pdb_ids.txt \
  --output-dir data/pilot/raw/mmcif \
  --manifest data/pilot/raw/download_manifest.csv \
  --max-entries 100 \
  --delay-seconds 0.2
```

Preprocess:

```bash
python scripts/preprocess_pdb.py --config configs/preprocess_pilot.yaml
```

## Full Filtered PDB Dataset

The full workflow uses the same reproducible RCSB filters as the pilot:
experimental structures, protein polymer entities, X-ray diffraction or electron
microscopy, and `entity_poly.rcsb_sample_sequence_length <= 500`. Use an
inclusive release-date cutoff to make later reruns independent of newly released
PDB entries.

Directory layout:

```text
data/full/
  pdb_ids.txt
  raw/
    mmcif/
    download_manifest.csv
  processed/
    samples/
    manifest.parquet
    preprocess_summary.json
```

Count matching polymer entities without retrieving IDs:

```bash
python scripts/query_rcsb.py \
  --output data/full/rcsb_count.txt \
  --count-only \
  --release-date-cutoff 2026-07-24
```

Retrieve every matching polymer entity with resumable pagination, then write
unique PDB entry IDs:

```bash
python scripts/query_rcsb.py \
  --output data/full/pdb_ids.txt \
  --all-matches \
  --page-size 10000 \
  --release-date-cutoff 2026-07-24
```

The all-match query shows a tqdm progress bar and saves resumable state after
each successful page at `data/full/pdb_ids.txt.query_state.json`. Resume is
enabled by default; use `--restart` to discard the saved state and begin again.
After Ctrl-C, rerun the same command to continue, or be explicit:

```bash
python scripts/query_rcsb.py \
  --output data/full/pdb_ids.txt \
  --all-matches \
  --page-size 10000 \
  --resume \
  --release-date-cutoff 2026-07-24
```

Download all selected entries with resumable, checksummed mmCIF downloads:

```bash
python scripts/download_pdb.py \
  --ids-file data/full/pdb_ids.txt \
  --output-dir data/full/raw/mmcif \
  --manifest data/full/raw/download_manifest.csv \
  --workers 4 \
  --retries 3 \
  --backoff-seconds 1.0
```

Preprocess the full filtered dataset:

```bash
python scripts/preprocess_pdb.py \
  --config configs/preprocess_full.yaml \
  --workers 8 \
  --resume \
  --checkpoint-every 1000
```

Preprocessing is resumable through a SQLite state database at
`data/full/processed/preprocess_state.sqlite`. The main process is the only
SQLite writer; worker processes parse structures and atomically write sample
`.npz` files. Partial snapshots are written periodically as
`data/full/processed/manifest.partial.parquet` and
`data/full/processed/preprocess_summary.partial.json`. Use `--restart` to
discard prior processing state, and `--retry-failures` to retry technical
failures while still skipping completed files and deterministic biological
rejections. Ctrl-C writes an incremental summary and prints the exact resume
command.

The default preprocessing config uses `min_length: null`, meaning no lower length
constraint, and keeps `max_length: 500`. Set `min_length` to a positive integer to
enable a lower bound.
`allowed_methods` is enforced as an exact list when non-null; set
`allowed_methods: null` to disable method filtering. X-ray and cryo-EM resolution
thresholds are applied only to `X-RAY DIFFRACTION` and `ELECTRON MICROSCOPY`,
respectively. Non-finite, zero, or negative resolutions are treated as missing.

Visualize processed samples:

```bash
python scripts/visualize_samples.py \
  --manifest data/pilot/processed/manifest.parquet \
  --output-dir outputs/pilot_dataset_samples \
  --num-samples 8 \
  --seed 42 \
  --contact-threshold 8.0
```

Summarize:

```bash
python scripts/summarize_dataset.py \
  --manifest data/pilot/processed/manifest.parquet \
  --output-dir reports/pilot_dataset
```

For the full dataset, summarization separates manifest-only statistics from
distance/contact statistics that require reading `.npz` sample files. The
distance phase is resumable and stores compact per-sample aggregates in
`reports/full/dataset_summary_state.sqlite`; it does not store full distance
matrices or all pairwise distances. Partial summaries are written atomically as
`reports/full/dataset_summary.partial.json`.

```bash
python scripts/summarize_dataset.py \
  --manifest data/full/processed/manifest.parquet \
  --output-dir reports/full \
  --workers 2 \
  --resume \
  --checkpoint-every 10000
```

Use `--restart` to discard prior summary state. Ctrl-C preserves completed
sample aggregates, writes a partial summary, and prints the exact resume command.

Build exact-sequence deduplicated manifests, cluster retained sequences with
MMseqs2, and write leakage-safe train/validation/test manifests:

```bash
python scripts/build_splits.py --config configs/split.yaml
python scripts/compute_train_statistics.py \
  --train-manifest data/full/splits/train.parquet \
  --output data/full/processed/normalization.json \
  --workers 1 \
  --checkpoint-every 5000 \
  --resume
```

Train-only normalization reads only the training manifest. It streams one sample
`.npz` at a time, selects finite non-negative valid upper-triangular
off-diagonal distances, and updates a fixed histogram instead of concatenating
all pairwise distances. Statistics computation is currently serial; pass
`--workers 1`. This is necessary for the full dataset because hundreds
of thousands of distance matrices contain several billion valid pairs, which is
too large to hold as one NumPy array on an 8 GB machine. The default histogram
uses 0.05 Angstrom bins over 0-2000 Angstrom, which is about 40,000 int64
counters. The percentile scale is therefore approximate, with error bounded
approximately by one histogram bin width.

Normalization checkpoints are written next to the output as
`normalization.state.npz` and `normalization.state.json`. They contain the
histogram, completed deterministic chunks, rejection diagnostics, manifest
SHA-256, configuration hash and algorithm version. Resume is enabled by default;
use `--restart` to discard partial state. If Ctrl-C occurs, the command writes a
consistent checkpoint, prints the resume command, and does not write the final
`normalization.json`.

MMseqs2 cache reuse requires a matching completed metadata sidecar containing
the retained FASTA SHA-256, input manifest SHA-256, command, version and split
configuration. Use the split workflow's force option to recompute stale cached
clusters. The split workflow itself is not resumable; `state_db_path` and
`checkpoint_every` are accepted only to warn about ignored historical config
fields.

The split assignment keeps connected groups indivisible by MMseqs2 cluster,
exact sequence hash and PDB entry. External group files are not currently
implemented and raise `NotImplementedError`. Groups are assigned by minimizing
deviation from the target total sample counts
80/10/10. Length and experimental-method distributions are reported after the
split for auditability, but they are not used as stratification variables.
`configs/split.yaml` applies `minimum_sequence_length: 20` before exact
deduplication and FASTA generation. Existing MMseqs2 cluster output is reused
when valid; `--force` rebuilds it, and new runs write a cache metadata sidecar
that records the filtered FASTA checksum and exact MMseqs2 command.

### Relaxed All-Structures Mode

The baseline preprocessing mode rejects any chain with a missing C-alpha. For
large experimental datasets, a scientifically safe relaxed mode can instead trim
only missing terminal residues while still rejecting internal missing C-alpha
positions, duplicated canonical `label_seq_id` positions, non-finite
coordinates, unsupported mixed polymer chains, and chains outside the final
20-500 residue length range. This avoids imputing coordinates or compressing
internal gaps while recovering structures with unresolved termini.
`max_terminal_trim_fraction` can cap the accepted terminal loss fraction;
`null` preserves uncapped trimming, while `configs/preprocess_full_relaxed.yaml`
uses `0.25`.

```bash
python scripts/preprocess_pdb.py \
  --config configs/preprocess_full_relaxed.yaml \
  --workers 8 \
  --resume \
  --checkpoint-every 1000
```

For training sets that should retain multiple experimental structures for the
same exact sequence, use the all-structures split mode. It clusters one
representative per unique exact sequence with MMseqs2, propagates cluster IDs
back to every retained structure, keeps exact sequence, MMseqs cluster and PDB
entry constraints in one connected component, and writes inverse-frequency
`sample_weight` values. Validation and test manifests are deterministic
one-representative-per-exact-sequence evaluation sets; corresponding
`*_all_structures.parquet` files are written for distributional analysis.

```bash
python scripts/build_splits.py --config configs/split_all_structures.yaml
python scripts/compute_train_statistics.py \
  --train-manifest data/full/splits_all_structures/train.parquet \
  --output data/full/processed_relaxed/normalization.json \
  --workers 1 \
  --checkpoint-every 5000 \
  --resume
```

Train and sample:

```bash
python scripts/train_diffusion.py --config configs/train.yaml
tensorboard --logdir outputs/pilot_baseline/tensorboard
python scripts/sample_distance_maps.py \
  --checkpoint outputs/pilot_baseline/checkpoints/best_validation.pt \
  --length 96 \
  --num-samples 16 \
  --output-dir outputs/samples/N96
```

Training writes JSONL logs plus TensorBoard scalars for loss, validation loss,
optimizer state, throughput, GPU memory and batch length ranges. Progress bars
are controlled by `logging.progress_bar`; TensorBoard is controlled by
`logging.tensorboard` and `logging.tensorboard_dir`. If training is interrupted
with Ctrl-C, an atomic checkpoint is written to
`outputs/pilot_baseline/checkpoints/interrupted.pt` and can be resumed by setting:

```yaml
resume_from: outputs/pilot_baseline/checkpoints/interrupted.pt
```

The trainer maintains `last.pt`, `best_validation.pt`, compatibility
`latest.pt`, and numbered `step_*.pt` checkpoints. Launch and resume commands:

```bash
python scripts/train_diffusion.py --config configs/train.yaml
python scripts/train_diffusion.py --config configs/train.yaml
```

## Preventing Data Leakage

Random PDB-ID splitting is insufficient because homologous proteins, duplicate chains, and related structures can appear under different entries. This code hashes exact sequences and keeps one deterministic representative by default. Retained sequences are clustered with MMseqs2, then connected components from cluster ID, exact sequence hash, and PDB entry are assigned whole to train, validation, or test. Explicit validation rejects cluster, exact-sequence, sample-ID, and PDB-entry overlap across splits.

Normalization is computed after splitting from valid upper-triangular training distances only. Validation and test files are not read by the statistics command.

Sequence clustering reduces homolog leakage, but low-sequence-identity proteins can still share folds. The split code is designed so external CATH or SCOP grouping labels can be merged into the group ID in future experiments.

One variable-size diffusion model is trained jointly on all accepted lengths up
to `model.max_length: 500`. Training uses a standard shuffled PyTorch
`DataLoader` with the variable-size collate function, dynamic padding, and
validity masks; it does not use length buckets, length bins, matrix-budget
batches, or separate length-specific models. The exact length `N` remains a
continuous conditioning input. The masked epsilon-prediction loss is normalized
per protein by its own valid upper-triangular pair count before averaging across
proteins, so longer chains do not dominate solely because their distance
matrices contain O(N^2) pairs.

For full-dataset GPU training, use `configs/train_full.yaml`. A short preflight
can stop after a fixed number of optimizer steps while still using the real
shuffled full manifest:

```bash
python scripts/train_diffusion.py --config configs/train_full.yaml --max-optimizer-steps 100
tensorboard --logdir outputs/full_baseline/tensorboard
python scripts/train_diffusion.py --config configs/train_full.yaml
```

Use `--resume-from outputs/full_baseline/checkpoints/last.pt` or set
`resume_from` in the YAML to resume. Checkpoints are written only after
optimizer steps, so exact mid-accumulation resume is not supported.

## Recommended Progression

Experiment A: parser and visualization check.

Download 10-20 structures, preprocess them, generate all visualizations, and manually inspect residue ordering, backbone traces, and distance-map patterns.

Experiment B: pilot dataset.

Download 100-500 entries, use configurable length filters such as `min_length: 40` and
`max_length: 500`, record rejection causes, deduplicate exact sequences, run MMseqs2
clustering, produce leakage-safe splits, and generate dataset statistics.

Experiment C: model overfitting test.

Select 16-32 similar-length samples, train until the loss drops strongly, sample at matching lengths, and check symmetry, zero diagonal, distance distributions, and visual structure. Only then move to mixed-length pilot training.

## Architecture

For `N=128` with channel multipliers `[1, 2, 4, 8]`, collation produces `[B, 1, 128, 128]` distance matrices plus `[B, 1, 128, 128]` sequence-separation and pair-mask channels. The model input is `[B, 3, 128, 128]`. Downsampling gives spatial sizes `128 -> 64 -> 32 -> 16`; masked multi-head attention operates on `16*16` bottleneck tokens; upsampling restores `128`. The output is `[B, 1, 128, 128]`, then it is symmetrized, has the diagonal set to zero, and is multiplied by the pair mask.

The `model.max_length` value in training configs is used to scale the logarithmic
length-conditioning embedding. It is not the preprocessing dataset filter. The
dataset filter is `max_length` in preprocessing configs and is currently `500`.
Practical GPU-memory limits still exist because each target distance matrix grows
as `O(N^2)`, and padded batches are sized by the largest sample in the batch.

## Expected Outputs

- `data/raw/download_manifest.csv`: download status by PDB ID.
- `data/processed/samples/*.npz`: processed samples.
- `data/processed/manifest.parquet`: processed manifest.
- `data/processed/preprocess_summary.json`: accepted/rejected preprocessing counts.
- `data/splits/*.parquet` and `data/splits/split_audit.json`: leakage-safe splits.
- `data/processed/normalization.json`: train-only normalization.
- `reports/dataset/`: summary JSON and plots.
- `outputs/dataset_samples/`: sample visualization PNGs.
- `outputs/pilot_baseline/`: logs, TensorBoard events and checkpoints.
- `outputs/samples/`: generated distance matrices and heatmaps.

## Validation

```bash
python -m py_compile $(rg --files -g '*.py')
pytest -q
ruff check .
ruff format --check .
```

## Limitations

The baseline does not model sequences or coordinates. The 3D trace in visualization is a plot of input C-alpha coordinates only. Topology-plus-trajectory support is intentionally only prepared at the backend interface level; a molecular-dynamics dataset pipeline is not implemented in this stage.
