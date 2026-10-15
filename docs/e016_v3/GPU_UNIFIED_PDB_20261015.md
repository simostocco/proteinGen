# GPU and unified PDB evidence

Evidence root: `/mnt/d/Simone/proteinGen_data/e016_gpu_unified_pdb_20261015_v1`.

Complete synthetic CUDA updates passed at N=64,128,256,500 for the unchanged 128-wide four-block actor and independent 96-wide three-block critics/targets. N=500 allocated peak540.188MiB, reserved650MiB; effective batch2/live1 also passes. CPU suite110passed,8CUDA-gated skips; separate native-tool equivalence/capacity and source integrity checks pass.

Unified intake:226127PDB entries,147769eligible unique full sequences,71884primary full-chain observations/27371identities. Priority2418/2440DONE;22explicit size/resource exclusions, no outside-priority acquisition. Historical raw/weights/reservoirs/evidence are preserved.

Independent all-node screening reports no detected cross-component/cross-split policy violations. The frozen control/context policy assigns130929LMidentities to calibration, producing candidate16839TRAIN/130929VALIDATION/1TEST. Ordinary joint BC validation/test have zero observations. Dataset status BLOCKED_EMPTY_REQUIRED_POPULATION; no TRAIN priors fit, no production BC/SAC/AF/E015 continuation. A scientific split/control decision is required.

Completed acquisition/component CLIs have checked read-only resume guards; provisional dataset admission is explicitly rejected. Full reports, source snapshots, journals, counts and checksum manifests reside in the evidence root. Mathematical G0–G2 contracts and historical state-dict schemas remain compatible.
