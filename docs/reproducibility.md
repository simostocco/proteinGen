# Reproducibility and execution boundaries

## From a clean source clone

Install the package and dev dependencies using the README. Synthetic CPU tests cover tokenization, causal masks, context controls, denoiser shape/padding, rotational/reflection behavior, parity types, self-conditioning, PCGrad, signed internal-coordinate round trips, action densities and terminal SAC targets. They do not train a research model or call external folding services.

The selected new source modules are listed with exact original commits in [integration_sources.json](publication/integration_sources.json). A small E011 path adaptation removes a personal machine path. Original source-seal checks are retained. E011/E012 configurations preserve historical settings, not a new recommended training recipe.

## To reproduce the research numbers

You need the exact external data release, accepted identity/observation manifests, split definitions, missing-residue masks, normalization statistics, checkpoint and optimizer lineage, evaluator revision and random seeds. Follow the experiment's original branch rather than mixing later model, loader and reward versions. The repository does not include these payloads or promise their availability. The baseline [usage guide](baseline_usage.md) explains historical preprocessing/training commands; those commands are not part of the verified clean-clone test procedure.

PDB observations, UniRef corpora, AlphaFold outputs and pretrained models have their own provenance and terms. A code license does not license those assets. No new dataset acquisition or redistribution is performed here. Later E016 practical split limitations and E017 validation reuse must remain part of interpretation; TEST independence has not been established.

## Retained on experiment branches

E014/E015 full trainers include local population/storage/recovery contracts. E016/J1 and E017/CRISP production pipelines couple chemistry/data/prior bindings, authorization receipts and durable journals. Importing them wholesale would suggest a portable integrated training system that has not been verified. Their full implementations, configs and original tests remain linked in the [branch inventory](publication/branches.md). The 2026-11-16 snapshot preserves loose source at its original root paths or under `worktree_snapshots/`; those archived paths are not a combined application.

No training campaign, GPU experiment, checkpoint deserialization or external biological evaluator was launched for this presentation work. The validation record states which checks ran and why broad historical tests may be unavailable.
