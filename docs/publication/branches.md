# Branch and worktree inventory

Audit: 2026-11-17. All original branches and 56 registered worktrees retained. Local origin tracking refs in the original workspace were stale; the refreshed GitHub refs below are authoritative for publication. All 60 original local heads match the refreshed remote tips. Useful pending source still matches the pushed 282-file snapshot; no additional source commit was necessary. Raw artifacts were not recursively copied or published.

The integration selects files, not entire histories. Exact file origins and evidence hashes: [manifest](integration_sources.json). Full worktree counts/status: [inventory](branch_inventory.json). Local worktree absolute paths are omitted from the public inventory.

| Branch | Published tip | Disposition |
|---|---|---|
| `audit-baseline-20261002` | [058ac3e747](https://github.com/simostocco/proteinGen/tree/058ac3e747d9dc28d1a3e9d0f449627de24cf2be) | retained without integration |
| `e007_matched_fullscale_v1` | [f717a98dfb](https://github.com/simostocco/proteinGen/tree/f717a98dfba48aa9fe574bdc9952c6442a52dd1e) | retained without integration |
| `e007_vs_e014_readonly_comparison_v1` | [7e0ab0804d](https://github.com/simostocco/proteinGen/tree/7e0ab0804deb06b8344d51659a7acb079dee3123) | retained without integration |
| `e010-condition-weight-counterfactual` | [058ac3e747](https://github.com/simostocco/proteinGen/tree/058ac3e747d9dc28d1a3e9d0f449627de24cf2be) | retained without integration |
| `e010-phase4c-local-auxiliary` | [bc5ae4fe83](https://github.com/simostocco/proteinGen/tree/bc5ae4fe83a710f8b89401fa1d2b2d586f193d6e) | selected files incorporated; full branch retained |
| `e010-phase4d-hybrid-local-global` | [bfc840a9bd](https://github.com/simostocco/proteinGen/tree/bfc840a9bd7e421c7e508cf41b3d66fdb31a6aa2) | retained without integration |
| `e011-sequence-context-only` | [30b657be81](https://github.com/simostocco/proteinGen/tree/30b657be81cd7bd943918980812d68ecaa0778de) | retained without integration |
| `e012-causal-rope-sequence` | [52fa7dcf8b](https://github.com/simostocco/proteinGen/tree/52fa7dcf8bc0546c3ef6a91e60bccc0c76e41180) | selected files incorporated; full branch retained |
| `e013_unconditional_se3_coordinate_generator_v1` | [dd421bde2a](https://github.com/simostocco/proteinGen/tree/dd421bde2a6b3a9b5eb549f72b874ae2ca875bac) | retained without integration |
| `e014_unconditional_coordinate_v_scaleup_v1` | [053ec32dd2](https://github.com/simostocco/proteinGen/tree/053ec32dd256349a6ff8504a0f796167f5c1c727) | retained without integration |
| `e015_integrated_multiscale_parity_selfconditioned_v1` | [0c49a5e936](https://github.com/simostocco/proteinGen/tree/0c49a5e93641c8e447e6b1d492ffbbeaaf7fecbf) | selected files incorporated; full branch retained |
| `e016-dataset-governance-20261015-v1` | [9005f9de61](https://github.com/simostocco/proteinGen/tree/9005f9de61399814765acffe88d06a73fd0100fb) | retained without integration |
| `e016-integrated-engineering-20261015-v1` | [9ff2fc5737](https://github.com/simostocco/proteinGen/tree/9ff2fc573715bebf8598ff9b8e11ccbf194d61bc) | retained without integration |
| `e016-j1-aa-head-only-20261019-v1` | [de4bb261f4](https://github.com/simostocco/proteinGen/tree/de4bb261f4908ff682072d2f6db63d394cdf44b7) | retained without integration |
| `e016-j1-bidirectional-multitask-20261019-v1` | [7eb0e9d25e](https://github.com/simostocco/proteinGen/tree/7eb0e9d25e64a122ed4a75404cf31ad198adb7c6) | retained without integration |
| `e016-j1-clip-precision-pilot-20261018-v1` | [3e6cbe09d0](https://github.com/simostocco/proteinGen/tree/3e6cbe09d026b17e8e7695c8eed157726391a57d) | retained without integration |
| `e016-j1-full-sequence-20261018-v1` | [28d818d501](https://github.com/simostocco/proteinGen/tree/28d818d50184a7735b6822210d7021847e1372bd) | retained without integration |
| `e016-j1-global-pretraining-20261017-v1` | [f0a0e2d2f5](https://github.com/simostocco/proteinGen/tree/f0a0e2d2f5a43a45d91c36612244b84a561be509) | retained without integration |
| `e016-j1-sequence-first-20261018-v1` | [51a1831166](https://github.com/simostocco/proteinGen/tree/51a18311660bb0c6a4e565a3d4e3c64bdfcb61d1) | retained without integration |
| `e016-j1-torsion-diagnostics-storage-20261018-v1` | [bbbc941be9](https://github.com/simostocco/proteinGen/tree/bbbc941be9635ac3acb647f8ec46a43e0ca56c6a) | retained without integration |
| `e016-j1-torsion-stabilization-20261018-v1` | [4cd6bd9f20](https://github.com/simostocco/proteinGen/tree/4cd6bd9f20803e44c1cb4afc4e3b3d02d182f5e1) | retained without integration |
| `e016-partial-supervision-20261016-v1` | [51b628ee2b](https://github.com/simostocco/proteinGen/tree/51b628ee2bc544125b41441285356db1fde02526) | retained without integration |
| `e016-r1b-afdb-msa-critic-20261019-v1` | [89e56c075d](https://github.com/simostocco/proteinGen/tree/89e56c075d18d50370e058220acc84b908a1a48c) | retained without integration |
| `e016-r1c-reward-reliability-20261019-v1` | [5ba906ba67](https://github.com/simostocco/proteinGen/tree/5ba906ba67b6b59f90ed83c5dfaff1cff4742da9) | retained without integration |
| `e016-r1d-experimental-stability-20261020-v1` | [5bf80ec163](https://github.com/simostocco/proteinGen/tree/5bf80ec1636afb7f3639ca561a018d154c73fa8e) | retained without integration |
| `e016-r1e-generation-evaluation-20261020-v1` | [b3af952083](https://github.com/simostocco/proteinGen/tree/b3af952083099520e3fbebdda97130ad05f1b098) | retained without integration |
| `e016-real-pdb-pilots-20261016-v1` | [90ca5ad5fc](https://github.com/simostocco/proteinGen/tree/90ca5ad5fcb6f8501d6ce6ae19b0db51d742e219) | retained without integration |
| `e016-scaled10m-training-20261016-v1` | [b43807b760](https://github.com/simostocco/proteinGen/tree/b43807b76069893fda87fc630fd7f1f230ac18b1) | retained without integration |
| `e016-sequence-first-rl-readiness-20261019-v1` | [def9a0c764](https://github.com/simostocco/proteinGen/tree/def9a0c7641a11b6da9675b1ebc39b3277a18610) | retained without integration |
| `e016-unused-train-census-20261018-v1` | [832a028340](https://github.com/simostocco/proteinGen/tree/832a028340c90406cd6526a6324de6d7e8315737) | retained without integration |
| `e016-v3-cuda-n64-pdb-20261014-v1` | [5b26b8859c](https://github.com/simostocco/proteinGen/tree/5b26b8859c7cf9ec5999bcb1a06b2de29cc34100) | retained without integration |
| `e016-v3-foundations-20261014-v1` | [7acd3646c7](https://github.com/simostocco/proteinGen/tree/7acd3646c73c142b1ec4ea6b5fd40941e34872d2) | retained without integration |
| `e016-v3-gpu-unified-pdb-20261015-v1` | [5c09c4baf3](https://github.com/simostocco/proteinGen/tree/5c09c4baf3ea506a7f202c24a9f9568dfe174e3e) | retained without integration |
| `e016-v3-readiness-20261014-v1` | [2081a56920](https://github.com/simostocco/proteinGen/tree/2081a569201dfd9bd2f45d60d8b4ac1d21c4a11c) | retained without integration |
| `e017-crisp-v2-20261021-v1` | [5ad0afccd9](https://github.com/simostocco/proteinGen/tree/5ad0afccd93db379dcf469f910cc61f54ad4a8a1) | retained without integration |
| `e017-crisp-v3-20261021-v1` | [cc0915ef24](https://github.com/simostocco/proteinGen/tree/cc0915ef24b57e8c368e1b8bd43681ab8bbb98c4) | retained without integration |
| `e017-crisp-v3-dynamic-storage-20261021-v1` | [4374c4a766](https://github.com/simostocco/proteinGen/tree/4374c4a766a088da4daf0b475a26357408876a8c) | retained without integration |
| `e017-crisp-v3-microbatch-20261021-v1` | [9c244462f3](https://github.com/simostocco/proteinGen/tree/9c244462f3980ae8008e27fd957200e1354c62c3) | retained without integration |
| `e017-crisp-v3-throughput-20261021-v1` | [774371b3ea](https://github.com/simostocco/proteinGen/tree/774371b3ea8724425c9fcc43702ad2233e70641e) | retained without integration |
| `e017-crisp-v3-training-admission-20261021-v1` | [b62978364d](https://github.com/simostocco/proteinGen/tree/b62978364dc054850f6b37e888baaa9d1638d4f6) | retained without integration |
| `e017-v4-multistep-20261022-v1` | [d8177be05f](https://github.com/simostocco/proteinGen/tree/d8177be05f900960e89979d1e2867ba9945080a0) | retained without integration |
| `e017-v5-goal-observability-20261022-v1` | [f1bf4a58eb](https://github.com/simostocco/proteinGen/tree/f1bf4a58ebd3d32741887fb7a57e3745c7236678) | retained without integration |
| `e017-v51-safe-control-20261022-v1` | [732134d9aa](https://github.com/simostocco/proteinGen/tree/732134d9aa2b3b30102d3a9e009b4665127f9506) | retained without integration |
| `e017-v6-frozen-candidates-20261023-v1` | [04ea639438](https://github.com/simostocco/proteinGen/tree/04ea6394381203f88dfc180972ccf512ee18f3e5) | retained without integration |
| `e017-v7-amortized-lowlevel-20261023-v1` | [e6ad343c10](https://github.com/simostocco/proteinGen/tree/e6ad343c10b5836415df18c4bf098ab5c38dfead) | retained without integration |
| `e017-v71-expected-return-20261023-v1` | [1f9a91b226](https://github.com/simostocco/proteinGen/tree/1f9a91b226460f163e13c501b9d2e1c524d84a42) | retained without integration |
| `e017-v72-counterfactual-ranking-20261023-v1` | [5e9c0f54e5](https://github.com/simostocco/proteinGen/tree/5e9c0f54e539c992220a8e2cdc292bf7aa5bce32) | retained without integration |
| `e017-v8-constrained-planning-20261023-v1` | [d3eb8b7d0e](https://github.com/simostocco/proteinGen/tree/d3eb8b7d0ef361f7224c495abc1b71860b1f1bac) | retained without integration |
| `e017-v9-property-steering-20261023-v1` | [2ba4ccfc31](https://github.com/simostocco/proteinGen/tree/2ba4ccfc31b194323f7d1227da5a724bc260c72a) | retained without integration |
| `e017-v91-ordered-goal-recovery-20261023-v1` | [82d23e47d1](https://github.com/simostocco/proteinGen/tree/82d23e47d1d33a0a6b2d2ff542aef3e8d86d0bfe) | retained without integration |
| `e017-v92-prefix-feasibility-20261024-v1` | [95b27b0834](https://github.com/simostocco/proteinGen/tree/95b27b0834fb36b0373bc89e7fdaec6773a85fb3) | retained without integration |
| `e017-v93-regional-controllability-20261024-v1` | [ea75b1d55f](https://github.com/simostocco/proteinGen/tree/ea75b1d55f1883bf1635edd1f403f6334b15b895) | retained without integration |
| `e017-v94-production-recovery-20261025-v1` | [cec94486cb](https://github.com/simostocco/proteinGen/tree/cec94486cb7ab53cd739a218b84e416e7ac9a7df) | retained without integration |
| `e017-v94r-packing-safety-20261025-v1` | [e390b87357](https://github.com/simostocco/proteinGen/tree/e390b87357c31a6cad8958ecd9292c942e01e57d) | retained without integration |
| `e017-v94r2-corrected-production-20261025-v1` | [afb72d824a](https://github.com/simostocco/proteinGen/tree/afb72d824a8ab2bd9e438a47a4f1ae23cc615add) | retained without integration |
| `e017-v94r2-corrected-production-20261025-v2` | [44f1b69be6](https://github.com/simostocco/proteinGen/tree/44f1b69be638d492ee53e541f359d617a4320586) | retained without integration |
| `e017-v94r3-one-step-feasibility` | [5f13a45a20](https://github.com/simostocco/proteinGen/tree/5f13a45a205c6a328a37b5b2af6ec2df705f26b2) | retained without integration |
| `e017-v94r3-production-20261026` | [91576ff071](https://github.com/simostocco/proteinGen/tree/91576ff071bd42a07f4306bc315b496d22d5a3bf) | selected files incorporated; full branch retained |
| `main` | [3b200ffb03](https://github.com/simostocco/proteinGen/tree/3b200ffb0389be87cc928c391117ce96bd8b5eb2) | PR base; unchanged until merge |
| `pdb-supervised-split-20261016-v1` | [dbc43ebde0](https://github.com/simostocco/proteinGen/tree/dbc43ebde00e2fb0a31ed1100d6ecb88ef6532c0) | retained without integration |
| `latest-work-snapshot-20261116` | [2030c692e6](https://github.com/simostocco/proteinGen/tree/2030c692e6ebb90ddff05a91746f29978217ca65) | preservation snapshot; no wholesale integration |
