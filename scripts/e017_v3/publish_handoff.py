from pathlib import Path
import json,hashlib,subprocess,shutil,platform,time
import torch,numpy,pyarrow
R=Path('/mnt/d/Simone/proteinGen_data/e017_crisp_v3_20261021_v1');S=R/'source';V2=R.parent/'e017_crisp_v2_20261021_v1'
def read(n):return json.load(open(R/'results'/n))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def put(n,x):
 p=R/n
 if p.exists():raise FileExistsError(p)
 with open(p,'x')as f:f.write(json.dumps(x,indent=2,allow_nan=False) if not isinstance(x,str) else x)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip();counts=read('SOURCE_AND_CHECKPOINT_RECOVERY_AUDIT.json')['profiles'];data=read('DATA_AND_SAMPLER_TESTS.json');profile=read('RESEARCH_RESOURCE_PROFILE.json');sac=read('RESEARCH_STREAMED_SAC_PROFILE.json');gpu=read('GPU_DEVELOPMENT_TESTS.json');final=read('FINAL_SOURCE_INTEGRATION.json');e2e=read('ENTRYPOINT_ENGINEERING_SMOKE.json');free=shutil.disk_usage(R).free
put('MODEL_PARAMETER_COUNTS.json',counts)
put('TRAINING_DATA_MANIFEST.json',dict(populations=data['counts'],source_sequence_split=str(R.parent/'pdb_supervised_split_50id80cov_20261016_v1'),partial_view=str(R.parent/'e016_partial_supervision_20261016_v1/view/VIEW_FROZEN.json'),sampling='identity-first; shuffled sweeps and shuffled per-identity target positions; independently selected qualified observation',AA_quota='40 complete /30 partial-only /30 AA-only decisions per100; 128/96/96 per320',geometry_stream='32 independent complete decisions; separate sampler',TEST_evaluation=False))
put('DEPENDENCIES.json',dict(python=platform.python_version(),torch=torch.__version__,torch_cuda=torch.version.cuda,numpy=numpy.__version__,pyarrow=pyarrow.__version__,python_executable='/home/simostocco/miniforge3/envs/proteingen/bin/python',run_wrapper=str(R/'run.sh'),required=['torch','numpy','pyarrow','inherited protein_joint_sac package'],external_network_calls=0,new_pretrained_models=0,geometry='FP64 CA builder and open-support transformed actions',precision='BF16 neural autocast, FP32 master weights/AdamW, FP64 geometry/density',determinism='torch deterministic algorithms; CUBLAS_WORKSPACE_CONFIG=:4096:8; no dropout in independent encoders; J1 eval'))
put('MODEL_ARCHITECTURE_AND_COUNTS.md',f'''# Independent frozen-J1 CRISP V3 architecture

Source `{commit}` (functional training code tested at d0d5031; the final publication commit adds only this report generator), branch `e017-crisp-v3-20261021-v1`. J1 is external to System, eval-only and no-grad, with 12,351,746 immutable parameters. It retains its eight-layer 352-wide bidirectional prefix Transformer, original features and geometry distributions. No pretrained weights are copied into the new optimizers.

| Research component | Parameters | Width × layers |
|---|---:|---|
| Low joint actor |17,438,114|448 × 7|
| Low Q1 |12,839,497|384 × 7|
| Low Q2 |12,839,497|384 × 7|
| High actor |6,378,769|320 × 5|
| High Q1 |5,278,377|288 × 5|
| High Q2 |5,278,377|288 × 5|
| Reachability |3,336,425|256 × 4|
| **Trainable total** |**63,389,056**| |

Separate target critics contain 36,235,748 frozen parameters. Development has1,235,428 trainable parameters; optional large has97,756,704. Full counts per component/profile are in MODEL_PARAMETER_COUNTS.json. Width/layers/heads/checkpointing are configurable through Shape and optional `shapes` configuration, with RoPE dimension validation. No profile is silently resized.

Each component has an independent encoder. Tokens contain canonical AA embedding, 38 safe geometry/chemical/validity descriptors and a learned 25-feature geometric pair bias. The query combines frozen J1 hidden state352, AA logits20, conditional Gaussian parameters120, public context64 and progress4. Explicit goal tokens36 encode position3, orientation9, constrained AA20, relative index and three modality masks. Prefix-only bidirectional SDPA uses RoPE. Goals never supply the hidden complete native structure. Input coordinate gauge is the canonical first-three-anchor frame; arbitrary global coordinates must first be converted to that gauge.

The low actor has separate AA and selected-AA-conditioned geometry readouts. Residual AA logits are bounded by3*tanh. Gaussian mean changes are bounded by[.15,1,1]*tanh in latent distance/angle/torsion units; log-standard-deviation changes by.2*tanh with existing[-5,2] limits. Zero final readouts make the initial distribution exactly J1. Encoder gradients are intentionally zero on the first update, then become active. There is no extra production teacher or modified J1 architecture.

Critics encode the prefix independently, then combine candidate AA embedding, physical geometry/inverse-transform features and validity masks. Geometry placeholders are replaced under false masks before inverse transforms. High actors/critics use nine masked options. Reachability uses a registered policy-version embedding with256 explicit slots; exhaustion raises an error, requiring a planned migration rather than silently aliasing policies. Optional subgoal probability/influence descriptors enter the public context interface; they are disabled in default collection until independently calibrated.
''')
put('DATA_AND_GOAL_PROTOCOL.md','''# Frozen data, native goals and curriculum

The accepted E016 PDB v3 split and full deposited sequence identities are unchanged. TRAIN disjoint categories are21,475 complete /29,129 partial-only /67,615 AA-only, total118,219. VALIDATION categories are2,947/3,178/8,650, total14,775. The inherited split loader reads ownership metadata, including TEST membership, for integrity; no TEST coordinates, features, losses or evaluation targets enter this work.

Complete TRAIN observations:63,315; partial TRAIN observations:118,461 across32,144 partial-view identities including overlap with complete identities. Partial observation/component counts therefore describe the full partial structural view, not only the disjoint partial-only AA sampling category. Complete distance/angle/torsion targets:9,958,717 /9,895,402 /9,832,087. Partial targets:21,266,079 /21,144,554 /21,023,121. Exact VALIDATION counts and membership hashes are in TRAINING_DATA_MANIFEST.json. TRAIN AA positions in disjoint groups:3,568,792 /6,336,204 /16,625,418.

Population loads sequence/observation metadata once and coordinates lazily; raw coordinate files are hash-verified when first used. Complete source membership/admission seals and partial frozen artifacts are verified. No retired prefilter is recreated or required. Legacy proposal flags are preserved; the accepted downstream E016 view plus this task's bounded engineering authorization define the present scope. No new scientific dataset is published.

AA sampling visits a shuffled identity sweep before repeating identities, with independently shuffled target positions per identity and uniform observation choice within identity. Ten-decision quota cycle realizes4/3/3; five64-target batches realize128/96/96 exactly. Geometry uses a separate complete-identity sampler, so AA mixture changes never remove its exposure. Sampler orders, position lists, cursors, coverage and RNG are recoverable. No claim of complete TRAIN exposure is made from the engineering fixtures.

AA-only prefixes retain true known tokens and length; all native-coordinate availability masks are false, and geometry target masks are zero. Partial geometry requires every mapped consecutive anchor and continuity link; missing atoms are not joined or interpolated. A later generated-prefix AA-continuation interface accepts only correctly corresponding known native AA prefixes; generated coordinates are context, never experimental geometry labels. Divergent AA prefixes cannot silently inherit a native next-AA label.

Goals are built only from complete TRAIN observations in the canonical first-three-anchor gauge. P0 imposes sequence/validity constraints; P1 adds an endpoint position; P2 adds midpoint/endpoint positions and frames. Motifs use true native residues only as explicit task constraints. All native targets are supplied sparsely; source IDs and hidden native suffixes are not neural input features. Task specifications may have multiple simultaneous targets. Tolerance variants reuse the same prefix/native witness; these are not independent experimental proteins.

Curriculum configuration begins64–96 residues, then64–128 and64–256, with configurable target-distance ranges, orientation tolerances45/35/25 degrees and motif counts1/2/3. Endpoint tolerances default8 A. Native witness validity is checked against the frozen16-parent TRAIN calibration; bounded rejection fails explicitly after64 draws. Its generalizability to longer lengths remains a scientific limitation. Goal demonstration prefixes alternate target/motif positions and broad positions; they do not rely on the unfulfilled V2 motif-quota claim. Goal sampling and this position cursor are checkpointed.

No fresh VALIDATION goals are generated. Existing V2 held-out native tasks/panels may be supplied as frozen audit inputs. The validation function accepts only explicitly tagged TRAIN/VALIDATION panels; future scientific panels must also be bound to authoritative identities and source hashes before an experiment is approved. Synthetic kinematic fixtures are labeled engineering data, not experimentally verified designs.
''')
put('HIERARCHICAL_MDP_AND_REWARD.md','''# Hierarchical MDP and reward v3

State contains N,t, generated AA/CA prefix, availability/continuity masks, explicit global goals, selected subgoal, public intervention summaries, pending-regeneration state and policy-dependent planner descriptors where enabled. Prefixes contain no native suffix. J1 has bidirectional attention only within this prefix. First three genuine native anchors are explicit task seeds. Neither arbitrary internal editing nor causal KV caching is assumed.

Low action factorization: pi(a,g|s,z)=pi_AA(a|s,z)*pi_G(g|s,z,a). Geometry uses d=exp(u_d), theta=pi*sigmoid(u_theta), tau=pi*tanh(u_tau), diagonal Gaussian latent u. Undefined components have zero action/zero log-density contribution. Construction preserves the historical right-handed signed-torsion builder. Inference uses ca_fp64_interior_v1, with explicit nearest-interior finite-grid repairs and separate latent/representative densities. Differentiable training rejects boundary saturation instead of clamping actions. Physical differential density is not a discrete probability mass.

High options0..3 CONTINUE with a selected subgoal;4..6 REWIND4/8/16 residues to an existing valid snapshot;7 REGENERATE;8 STOP. CONTINUE requires positive duration, an incomplete prefix and no pending regeneration. REGENERATE is required after rewind and starts a reproducible new branch stream. STOP on an incomplete chain is terminal failure. Rewind restores sequence, coordinates, masks, prefix history, RNG and prefix-context branch; irreversible intervention history/event clock remain. No correction-count reward or computational budget exists.

Protocol crisp_v3_terminal_geometric_v1 uses binary completed-task success: every explicit sequence constraint, position tolerance, orientation tolerance and backbone validity condition must pass. Coordinates must be finite, adjacent distances within[2.850420047,4.052024994] A, clashes<=2/100 residues, no degenerate frames and Rg/N^(1/3) within[1.950656875,10.184757893]. These are V2 native-calibrated geometric feasibility bounds, not a biological stability test. Orientation compares right-handed row frames through acos((trace(F_current F_target^T)-1)/2).

A bounded potential phi in[-1,0] measures distance to future tolerated target balls, normalized by remaining contour length. Optional shaping is c*(gamma*phi(s_next)-phi(s)), c=.1 by default, with absorbing terminal potential zero. Primary terminal reward is1 for success and0 for failure. There is no runtime/retry penalty. Gamma=.995 on the primitive event clock favors earlier successful completion, including intervention duration. A necessary contour reachability bound includes the target tolerance; it is not a sufficient success guarantee.

High option reward is sum_i gamma^i r_i; Bellman target is R_option+gamma^k*(1-terminated)*V(next). Operational event limits(default4096) censor rather than fail trajectories. True terminations have zero bootstrap; operational truncations retain a valid bootstrap action mask. Reachability labels exclude censored or unresolved outcomes. The low continuation critic's finite-horizon task ends at completion and includes terminal construction success; manager STOP/rewind values belong to the high critic. Thus the low critic estimates continuation value, not an unconditional manager marginal return. This hierarchical decomposition must be evaluated scientifically before a joint learning campaign.

Exact local next-CA Jacobians are available for valid anchors, checked against finite differences. Counterfactual interventions restore a common prefix/RNG, perturb one physical action and reconstruct a bounded suffix under a specified continuation. Influence features include endpoint/max/RMS displacement and AA changes. No derivative through an ungenerated future is claimed. Learned reachability is Pr(success | state, goal, identified low policy and manager); predictions are neither calibrated certificates nor interchangeable across policies.

The typed Option/ControlledWorld boundary has no robotics dependencies. A future robotics adapter needs its own state/action/kinematics mapping, not a claim of demonstrated robotic transfer. No AlphaFold or external biological predictor participates in this reward.
''')
put('SUPERVISED_AND_RL_LOSS_SPEC.md','''# Independent-model losses and staged optimization

J1 is always eval/no_grad, outside System and every optimizer. Its tensor/buffer SHA is verified around updates. Independent models use separate AdamW optimizers/schedulers. Default proposals: lr1e-4, weight decay.01, finite-gradient group norm limit1. These are engineering starting settings, not calibrated scientific training recommendations. FP32 master weights and Adam moments, BF16 neural operations and FP64 geometry/densities are used.

Supervised initialization: AA CE plus lambda_G times physical geometry NLL averaged over valid components; separate AA and geometry KL to J1; optional one-step goal proximity. Defaults1/.1/.02/.01/.05. AA batch64 uses independent-prefix microbatch accumulation. Thirty-two additional complete-geometry decisions and eight native goal demonstrations are independently normalized before one actor optimizer step. Goal proximity differentiates only a supplied target at the next residue, using next-CA construction and a bounded squared-distance score. It does not pretend to differentiate an entire unknown suffix. Native/generated-prefix interfaces and frozen-panel validation are separate from policy-generated RL rewards.

Physical log density is log Normal(u;mu,sigma)-sum log|dg/du| over valid components. Jacobians are u_d; log(pi)-softplus(-u_angle)-softplus(u_angle); log(pi)+2*(log(2)-u_tau-softplus(-2*u_tau)). KL_G equals Gaussian latent KL because both policies share the same invertible transform; component masks are preserved. AA KL is exact categorical KL. Geometry KL is conditional per AA, then expectation under current AA probabilities, normalized by valid-component count in actor regularization.

Low SAC uses twin independent candidate-action critics and min-Q targets. Target value sums exactly over20 AA actions, sampling reparameterized continuous geometry conditionally for each. Actor loss is expectation[alpha_A*log pi_A + alpha_G*log density_G - min(Q1,Q2) + beta_A*log(pi_A/pi_A_ref) + beta_G*conditional_KL_G/valid_count], plus optional PDB BC regularization. Defaults alpha_A=.01, alpha_G=.001, beta_A=.02,beta_G=.01. Separate entropy coefficients are fixed/configurable, not automatically tuned or assumed commensurate. Differential entropy depends on physical units. Exact20-way AA expectation avoids differentiation through integer sampling. One actor graph is recomputed/backpropagated per AA; critic prefix features are cached detached while candidate-action gradients remain valid. Streamed and summed gradients agree within1.49e-8 on the fixture.

High SAC uses masked exact categorical expectation over nine options, independent twin high Qs and semi-Markov targets. Polyak tau=.005 by default. Critic, actor, high actor and reachability schedules are independently configurable; stages run separately by default. Reachability BCE uses identified-policy, genuinely terminated outcomes only. Its256-version registry is explicit and persisted; long campaigns must plan registry capacity rather than alias IDs. Synthetic engineering success demonstrates mechanics, not real-return critic accuracy.

PDB AA BC retains the disjoint40/30/30 policy. Qualified geometry imitation has a smaller independent weight; AA-only examples never acquire geometry labels. Configured BC hooks add gradients before the low actor optimizer step. Later-stage initialization requires a full named/hash-verified independent-system checkpoint, carrying compatible moments, samplers and policy-indexed replay; J1 optimizer state is never imported. A stage change is distinct from exact same-config interruption recovery.
''')
put('REPLAY_AND_CHECKPOINT_SCHEMA.md','''# Replay and complete recovery

SQLite WAL/FULL synchronization stores episode headers, policy-slot registry, hash-chained primitive events, typed transitions and resolved/censored episode outcomes. Headers bind initial prefix, seed, operational safeguard, goal, policy identity, source class and immutable J1 hash. Header and event hashes detect changed input/action records. Stable episode and transition IDs derive from content/event positions, so restart does not change replay ordering through random UUIDs.

Transitions reference an episode and before/after event indices; -1 means initial state. They carry hierarchy/action, component mask, actual behavior density, reward, duration, terminal/truncated flags and a goal hash. Goal/native arrays are not duplicated per transition. Actual action information includes selected-AA log probability, represented physical geometry density, latent density, numerical support repairs and random-stream semantics. Low decisions consume one saved-stream uniform and one Normal(3) draw; manager sampling has its own checkpointed global RNG. Rewind/regeneration reconstruct branch state exactly.

Policy-generated, native demonstration, counterfactual and synthetic engineering records are separately typed. Native demonstrations are BC data and cannot enter policy replay with invented behavior likelihoods. Unresolved transitions are rejected; timeouts are not failure labels. Indexed source/hierarchy sampling avoids JSON-decoding the entire replay at each update. Off-policy SAC uses current actor/critics on reconstructed states; stored behavior likelihoods support diagnostics, not an unimplemented importance-ratio estimator.

A full checkpoint contains all seven independent models, four targets, optimizers, schedulers, independent samplers, curriculum/goal-prefix cursors, RNG, counters, source commit, exact input hashes, native-limit hash, configuration and replay snapshot identity/hash. Frozen J1 is referenced by immutable SHA rather than duplicated. Atomic .part write+fsync+rename publishes new immutable checkpoints; LATEST_RECOVERY binds checksum and snapshot. Previous unselected checkpoints created by the same future run may be pruned after new publication; create a .keep marker beside a selected checkpoint to protect its paired replay snapshot. No historical artifact is a retention candidate.

Same-config recovery restores optimizer/scheduler/RNG/sampler/target state and logical replay contents. Bytewise SQLite file identity across backups is not required. Stage-to-stage initialization explicitly preserves policy registry and replay mapping, preventing trained reachability embeddings from being assigned to different policy IDs. A separately approved stage/config/source/reference-specific receipt is required for substantive execution. Engineering-only one-update receipts cannot authorize research-profile campaigns.

GPU tests cover same-process and two fresh-process recovery. The fixtures do not simulate physical power loss at every filesystem write boundary or certify Windows/NTFS durability beyond fsync/rename behavior. Full research checkpoints were sized in memory; persistent complete-checkpoint recovery was tested on development capacity to protect the storage reserve. Older intermediate engineering replay schemas are preserved as draft evidence, not silently reinterpreted by final readers.
''')
put('SCIENTIFIC_RISKS_AND_LIMITATIONS.md','''# Limits and unresolved scientific questions

V3 is training infrastructure, not evidence of improved task success or protein quality. V2's zero observed complete-task successes and censored manager outcomes remain unchanged. No prolonged learning, J1 update, TEST evaluation, external prediction or new dataset was executed.

The large encoders may be sample-inefficient. Zero residual readouts initially block encoder gradients; finite gradients after activation do not prove useful goal representations. Residual bounds, KL scales, entropy temperatures, curriculum and optimizers need prospective TRAIN calibration and held-out task evaluation before a large campaign. Local geometric likelihood/validity is not experimental folding stability.

Native calibration uses16 short TRAIN parents and may not adequately characterize longer chains. Exact/kinematic goal feasibility does not establish reachability under the frozen prior plus bounded residual family. Initial native anchors make these conditional construction tasks, not unconditional de novo generation. Higher-level nine-option control is intentionally narrower than arbitrary segment editing.

Real long-horizon reward learning, critic calibration, intervention usefulness and reachability calibration remain untested. Replay can carry counterfactual records but default actor sampling does not imitate forced interventions. Learned reachability descriptors are interface-ready and disabled in default collection until calibrated. The policy-version embedding currently supports256 explicitly registered versions; extension needs a deliberate schema/model migration. No ungenerated-future Jacobian is available.

Current validation accepts caller-supplied frozen TRAIN/VALIDATION records; full scientific panel/goal provenance and no-TEST membership must be prospectively approved before training. No broad held-out quality assessment was performed. E016 biological-reward NO-GO conclusions are unchanged and independent of E017 geometric-task engineering.

Research capacity fits the bounded GPU profile, but atomic two-checkpoint disk headroom fails at current free capacity. Larger real batches, CPU coordinate I/O, longer replay and validation costs have not been certified by these microbatch profiles. The97.76M profile has exact meta counts only, no GPU feasibility claim. No capacity was silently reduced.
''')
put('FUTURE_STAGED_TRAINING_PROTOCOL.md','''# Future staged execution; no training approval implied

1. Resolve D: capacity before research allocation. Preserve10 GiB plus two full staged checkpoints, sampler/replay snapshots and logs. Do not delete protected evidence automatically. Freeze qualified TRAIN goals and a provenance-bound held-out VALIDATION task panel.
2. Begin with the independent actor's supervised goal-conditioned initialization; J1 remains frozen permanently. Use broad40/30/30 identity coverage, separate qualified geometry and explicit goal demonstrations. Calibrate gradient/entropy/KL scales on TRAIN only. Seal losses/schedule/evaluation gates before a campaign.
3. Validate direct AA-only, geometry-only and joint control, with frozen J1 and simple rewind/continuation baselines. Distinguish learning under goals from correction effects. Measure task success, geometry, plausibility, diversity and resource cost separately.
4. Initialize low SAC from a verified full independent-system checkpoint. Train critics/actor on separately scheduled updates with PDB BC and frozen-reference KL. Evaluate held-out Monte Carlo return calibration and action-specific error before trusting Q-based planning.
5. Initialize high SAC from the qualified low-policy checkpoint. Validate option durations, rewind behavior and termination/censoring on real trajectories. A high controller should beat simple correction baselines before causal/reachability features receive credit.
6. Train reachability on identified-policy resolved success/failure outcomes; keep censored data out of binary targets. Only after calibration enable its planner-context interface. Controlled same-noise interventions provide policy-dependent supervision, not hidden future derivatives.
7. Predeclare substantive update counts/milestones and source/config-specific authorization separately. Default configs describe10000 potential slots, not an authorization or proven runtime. Optional97.76M profile requires its own GPU/storage profile. Stop for numerical/resource failures and prospectively declared scientific futility; no automatic long campaign follows this handoff.

Future architectural ablations must isolate goal conditioning, low-level steering, high-level corrections, intervention features and reachability. Keep J1 bitwise checks, biological-quality evaluation and geometric task success separate throughout.
''')
put('LAUNCH_COMMANDS.md',f'''# Runnable entrypoints

Source branch `e017-crisp-v3-20261021-v1`, commit `{commit}`. Python/dependencies are in DEPENDENCIES.json. All persistent scratch/caches/checkpoints are routed to D: by run.sh.

Dry-run only (safe now):
```bash
bash {R}/run.sh {S}/scripts/e017_v3/train.py --config {R}/configs/research_supervised.json
```

Later, after separate approval and storage remediation:
```bash
bash {R}/run.sh {S}/scripts/e017_v3/train.py --config {R}/configs/research_supervised.json --execute --authorization-file /mnt/d/Simone/proteinGen_data/APPROVED_STAGE_RECEIPT.json --output {R}/future_runs
```

A receipt must name `approved`, exact `source_commit`, `config_sha256` printed by dry-run, `reference_sha256`, `stage` and `max_updates`. No production receipt is supplied. Later low_sac/high_sac/reachability configs also require `initial_checkpoint: {{path,sha256}}` for a full compatible independent-system checkpoint. Edit/seal a new versioned config before requesting its authorization. Add `--resume /absolute/checkpoint.pt` for exact same-config recovery. Read LATEST_RECOVERY.json and verify its hashes first. Training may fail the disk reserve gate before allocating models; do not bypass it by lowering capacity silently.

All six development/research/large/stage configs and native limits are also tracked under source/configs/e017_v3. The run wrapper assumes this evidence root; a fresh reproduction root needs its own wrapper/config path bindings. Engineering scripts deliberately refuse to replace successful result names. Use a fresh versioned evidence directory for reproduction; do not rerun the completed V2 experiment or overwrite current evidence. The recovered unused verify draft is archived under docs/e017_v3/recovery and is not an entrypoint.
''')
# Resource and consolidated tests are filled from actual artifacts only.
fixtures={p.name:json.load(open(p)) for p in (R/'results').glob('*.json') if 'FAILURE' not in p.name and 'SOURCE_AND_CHECKPOINT' not in p.name and 'CHILD_' not in p.name}
put('ENGINEERING_TEST_RESULTS.json',dict(status='PASS_COMPLETED_BOUNDED_ENGINEERING',source_commit=commit,results=fixtures,initial_failures_preserved=[p.name for p in (R/'results').glob('*FAILURE*.json')],substantive_training_updates=0,J1_updates=0,engineering_optimizer_calls_before_entrypoint_fixtures=42,engineering_optimizer_calls_total=50,entrypoint_fixture_results=e2e,scope='Engineering correctness, not learning efficacy'))
put('ENGINEERING_TEST_REPORT.md',f'''# Engineering verification

Frozen J1 tensors/buffers retain SHA `{gpu['J1_tensor_sha256']}`. No J1 optimizer membership or gradient accumulation. Development supervised/low SAC/high SAC/reachability steps, target updates and mixed complete/partial/AA-only inputs pass. Independent AA and geometry gradients and goal sensitivity pass. Exact categorical streamed-gradient error1.49e-8; physical density matches the historical law; common-transform KL agrees with bounded Monte Carlo. FP64 extreme signed torsions stay inside support. Local Jacobian finite-difference error is reported in results/ENVIRONMENT_DENSITY_REPLAY_TESTS.json; rigid transforms preserve construction.

Native P0/P1/P2 witnesses, rewind/pending-regeneration restoration, immutable event reconstruction and operational timeout masks pass. Replay IDs/goal hashes/source hierarchy, snapshot reopen and policy mapping are tested. Same-process GPU continuation and two independent fresh processes reproduce model/target/AdamW/RNG results. Final mixed orchestration fixture produces encoder gradient norm0.050638, AA30.8163 and geometry3.00743 on its second engineering update; J1 stays fixed.

Initial failed checks are retained: AdamW scalar CPU/CUDA comparison/load placement; AA-only bond-mask off-by-one; long-prefix fixture declaring N64; initial assertion expecting encoder gradients before zero readouts activate. The latter is a test expectation correction, not failed mathematical freezing. Final repairs pass their affected tests. Intermediate tests ran during development; final binding/replay/entrypoint regressions cover the changed interfaces. No historical scientific result was relabeled.

The staged-entrypoint smoke uses at most one development update slot per stage and a completed-run resume that must perform zero additional slots. The complete task used50 bounded engineering optimizer calls across independent fixture models, including discarded restart/repaired-test calls. Eight of these exercise the four one-slot staged entrypoints. Exact stage calls are in results/ENTRYPOINT_ENGINEERING_SMOKE.json. Engineering receipts authorize only those tiny fixtures. No research campaign, real-return critic-quality study or model promotion occurred. Development recovery checkpoint `{final['checkpoint']}` has SHA `{final['checkpoint_sha256']}`; it is an engineering fixture, not a research-policy candidate.
''')
put('RESOURCE_AND_STORAGE_REPORT.json',dict(source_commit=commit,research_profile=profile,streamed_SAC=sac,D_free_bytes=free,reserve_bytes=10*2**30,minimum_two_checkpoint_tensor_plus_128MiB_free_bytes=10*2**30+2*profile['full_tensor_checkpoint_bytes']+128*2**20,storage_ready=False,downloads=0,deletions_of_historical_artifacts=0,J1_updates=0,production_training_updates=0,optional_large_GPU_profile='NOT_RUN',staged_entrypoint_fixture=e2e))
put('RESOURCE_AND_FEASIBILITY_REPORT.md',f'''# Measured resources

Research system63.389M trainable,36.236M target and12.352M frozen J1 parameters. All seven research optimizers received real engineering updates, allocating their AdamW states. Peak PyTorch allocation at a500-residue prefix with microbatch1 was{profile['lengths']['500']['peak_allocated_bytes']/2**30:.3f} GiB; reserved{profile['lengths']['500']['peak_reserved_bytes']/2**30:.3f} GiB. Actual observed Windows/display baseline was about1299 MiB, outside PyTorch accounting. These are bounded profiles, not a guarantee for larger batches.

Actual20-way streamed research actor step at t96: {sac['actor_seconds']:.3f}s; twin low critic step:{sac['twin_critic_seconds']:.3f}s. Its measured allocation was{sac['peak_allocated_bytes']/2**30:.3f} GiB; adding the other optimizers' unallocated moment tensors conservatively gives{sac['conservative_peak_plus_other_moments_bytes']/2**30:.3f} GiB. The earlier all-optimizer profile is independently measured. Host RSS and all length timings are in RESOURCE_AND_STORAGE_REPORT.json. Metadata/fixture loader audit took{data['seconds']:.2f}s without a full coordinate cache.

Tensor-only full recovery size:{profile['full_tensor_checkpoint_bytes']:,} bytes. Two generations:{profile['atomic_two_generation_bytes']:,} bytes, before sampler/replay/serialization overhead. Current measured free:{free/2**30:.3f} GiB. The10-GiB reserve plus those two tensors and128-MiB margin requires{(10*2**30+2*profile['full_tensor_checkpoint_bytes']+128*2**20)/2**30:.3f} GiB. Research launch is therefore storage-blocked. Seek at least12.5–13 GiB free for modest replay/log headroom, more for long replay. No historical deletion is proposed automatically.

At the measured microbatch actor forward/backward scale,104 independently processed supervised decisions may take roughly8–12s/update before additional loader/validation overhead: approximately22–34 GPU-hours for10000 slots is a rough extrapolation, not a campaign quote. A20-action actor SAC step is about1.1s for one fixture state, plus critics/collection/BC; batch64 exact enumeration can dominate. Stage schedules and accumulation are explicit; benchmark the approved real workload before reserving a long run. The97.76M optional profile is meta-counted only. No substantial training was launched.
''')
put('EXPERIMENT_DECISIONS.md',f'''# Decisions

1. Resume V2 from clean5ad0afc and verify preserved checkpoint hashes; do not repeat V2's80-update/192-attempt study. V3 drafts survived shutdown; no V3 completed experiments existed at resumption.
2. J1 remains external/permanently frozen; independent63.389M system supplies sequence/geometry residuals and planning. Zero readout initialization preserves the exact prior. V2 adapters remain historical evidence, not silently transplanted into incompatible larger models.
3. Preserve physical density/builder/FP64 inference semantics. Use exact categorical expectations with streamed graphs and separate entropy/KL controls. No external biological predictor enters geometric task reward.
4. Preserve frozen data ownership, disjoint AA groups and independent geometry supervision. TRAIN-only native goal construction checks feasibility; no new split, no retired-prefilter rebuild.
5. Separate true terminal success, operational censoring and policy-dependent reachability. Corrected timeout bootstrap masks and deterministic replay IDs are engineering changes, not revised scientific outcomes.
6. Store full independent model/target/optimizer/sampler/RNG/replay recovery. Development and bounded research engineering updates pass; target research disk staging is inadequate, so future research launch must remain gated.
7. Source `{commit}`. Engineering-only one-slot receipts are not production approvals. No J1 optimization, external oracle call, substantial supervised run or RL campaign. Stop after this handoff.
''')
put('CHANGELOG.md',f'''# V3 changelog

- Recovered clean V2 source5ad0afc and its preserved full-state evidence after shutdown.
- Added independent configurable geometry-aware actors, twin low/high critics, targets and reachability; exact63,389,056 research trainable parameters.
- Added lazy complete/partial/AA-only data paths, identity/position coverage and TRAIN native-goal curriculum.
- Added supervised/staged hybrid and high SAC entrypoints, PDB regularization, KL/densities, deterministic reward, event-sourced replay and complete atomic recovery.
- Repaired and preserved failed engineering fixtures; verified bitwise J1 freezing and separate-process GPU continuation.
- Archived the recovered unused verification draft rather than expose it as an active entrypoint.
- Published source `{commit}`, measured GPU feasibility and explicit research checkpoint-storage blocker. No substantive training executed.
''')
put('FINAL_E017_CRISP_V3_ENGINEERING_HANDOFF.md',f'''# E017 CRISP V3 engineering handoff

**Infrastructure implemented and bounded engineering checks pass. No evidence of improved design quality is claimed. Substantive training remains unstarted and requires separate approval plus storage remediation.**

Source branch `e017-crisp-v3-20261021-v1`, commit `{commit}`; parent V2 `5ad0afc`. J1 reference `{R.parent}/e016_j1_bidirectional_multitask_20261019_v1/runs/J1/checkpoint_07500.pt`, SHA `881457ff0396975c1b20dd0a8e2caac8521e65b7e1cec76a7bb531dbc437b412`, remains bitwise frozen. Historical V2 and E016 qualification/results are unchanged.

Seven independent trainable networks total **63,389,056 parameters**, plus36,235,748 separate target parameters. Development1,235,428; optional larger97,756,704. Inputs combine safe prefix geometry/chemistry, frozen J1 representations/distributions, goals, subgoals and public intervention context. The low Transformer steers AA logits and selected-AA-conditioned distance/angle/signed-torsion distributions; J1 is never an optimizer member.

All118,219 TRAIN identities are available through exact40/30/30 decision quotas, with separate qualified geometry. TRAIN-only native goal curriculum, masked partial/AA-only paths, hierarchical continuation/rewind/regeneration/stop, deterministic geometric rewards, streamed hybrid SAC, high SAC, counterfactual/reachability interfaces, indexed replay and complete recovery are implemented. No new scientific split or biological reward model was created.

Measured tests establish finite gradients, independent head steering, correct physical densities/KL, target updates, local Jacobians, rigid transforms, native witnesses, replay/rewind/RNG restoration and GPU interrupted recovery. Two fresh processes produce identical model/target/AdamW/RNG continuations. Actual staged entrypoint fixtures and completed-run recovery are in the test report. These are engineering tests, not a training campaign.

**GPU feasibility:** full research model/targets/AdamW states fit the bounded RTX5060 profile; length500 microbatch peak{profile['lengths']['500']['peak_allocated_bytes']/2**30:.2f} GiB PyTorch allocated. A20-action research actor update took{sac['actor_seconds']:.2f}s on one t96 fixture. **Storage NO-GO:** full tensor recovery~{profile['full_tensor_checkpoint_bytes']/2**20:.0f} MiB each; two-checkpoint staging plus reserve exceeds current{free/2**30:.2f} GiB free. Preserve10 GiB; do not silently reduce capacity or delete artifacts.

Read:

- [Source/checkpoint recovery audit](results/SOURCE_AND_CHECKPOINT_RECOVERY_AUDIT.json)
- [Architecture/counts](MODEL_ARCHITECTURE_AND_COUNTS.md), [exact profile counts](MODEL_PARAMETER_COUNTS.json)
- [Data/goals](DATA_AND_GOAL_PROTOCOL.md), [population manifest](TRAINING_DATA_MANIFEST.json)
- [Hierarchical MDP/reward](HIERARCHICAL_MDP_AND_REWARD.md), [losses/optimization](SUPERVISED_AND_RL_LOSS_SPEC.md)
- [Replay/checkpoint schema](REPLAY_AND_CHECKPOINT_SCHEMA.md)
- [Engineering tests](ENGINEERING_TEST_REPORT.md), [detailed results](ENGINEERING_TEST_RESULTS.json)
- [Resources](RESOURCE_AND_FEASIBILITY_REPORT.md), [measurements](RESOURCE_AND_STORAGE_REPORT.json)
- [Staged training protocol](FUTURE_STAGED_TRAINING_PROTOCOL.md), [runnable commands](LAUNCH_COMMANDS.md), [dependencies](DEPENDENCIES.json)
- [Risks and untested assumptions](SCIENTIFIC_RISKS_AND_LIMITATIONS.md)
- [Decisions](EXPERIMENT_DECISIONS.md), [changelog](CHANGELOG.md), [SHA provenance](PROVENANCE_SHA256.json)

**Stop for review.** No prolonged SAC, J1 fine-tuning, broad quality evaluation, TEST evaluation, external model inference or robotics run occurred. Development checkpoints are engineering fixtures, not promoted policies. Direct goal learning, real-return critic calibration and hierarchical/causal benefits remain future scientific experiments.
''')
# Verify protected inputs again. Avoid hashing the same large file twice.
protected=read('SOURCE_AND_CHECKPOINT_RECOVERY_AUDIT.json')['verified']
for p,h in protected.items():
 if sha(p)!=h:raise AssertionError('protected checkpoint changed: '+p)
if subprocess.check_output(['git','status','--porcelain'],cwd=V2/'source',text=True).strip():raise AssertionError('V2 source dirty')
# Exclude transient runtime caches and Git internals, include compact engineering recovery.
artifacts={}
for directory in ['results','configs','checkpoints','data','engineering_runs']:
 for p in sorted((R/directory).rglob('*')):
  if p.is_file() and not p.name.endswith(('-wal','-shm','.part')):artifacts[str(p.relative_to(R))]=sha(p)
for p in sorted(R.glob('*.md')):artifacts[p.name]=sha(p)
for n in ['MODEL_PARAMETER_COUNTS.json','TRAINING_DATA_MANIFEST.json','DEPENDENCIES.json','ENGINEERING_TEST_RESULTS.json','RESOURCE_AND_STORAGE_REPORT.json']:artifacts[n]=sha(R/n)
source={str(p.relative_to(S)):sha(p) for base in ['src/protein_joint_sac/crisp_v3','scripts/e017_v3','configs/e017_v3','docs/e017_v3'] for p in sorted((S/base).rglob('*')) if p.is_file()}
put('PROVENANCE_SHA256.json',dict(source_commit=commit,source_branch='e017-crisp-v3-20261021-v1',source_files=source,artifacts=artifacts,protected_checkpoint_sha256=protected,run_wrapper_sha256=sha(R/'run.sh'),historical_source_preserved=True,scope='No runtime cache/SQLite journal/Git internal duplication; authoritative replay snapshots and checkpoints included'))
print(json.dumps(dict(source_commit=commit,artifacts=len(artifacts),source_files=len(source),D_free_bytes=shutil.disk_usage(R).free),indent=2))
