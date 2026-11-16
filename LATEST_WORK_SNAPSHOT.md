# Latest local work snapshot — 2026-11-16

This branch preserves previously uncommitted source and compact summaries without changing the original worktrees. Existing experiment commits are published on their original branches; this snapshot does not merge those experiments into main.

Root files retain their original relative paths. Files from other worktrees are archived under `worktree_snapshots/<original-branch>/`, including the detached V97 engineering worktree. These archived paths preserve source; they are not a combined runnable application. Some scripts depend on local absolute paths and external datasets or checkpoints.

`LATEST_WORK_MANIFEST.json` records each original location, base commit, size, and SHA256. Raw datasets, checkpoints, model outputs, environments, caches, and large runtime artifact folders remain local. Storage audit source and its summary are included, while census/process dumps are excluded.

Validation: copied bytes and hashes were checked, Python sources were parsed, and common credential formats were scanned. Training, GPU jobs, recovery scripts, and deletion scripts were not executed. See the manifest for any pre-existing syntax errors.

Two archived source lines have pre-existing trailing whitespace; file contents are preserved exactly. The manifest also lists all published experiment branch tips.
