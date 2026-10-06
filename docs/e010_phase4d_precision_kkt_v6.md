# E010 Phase 4D v6 radial-bound precision and KKT audit

Historical v5 commit f846606e6784eda9517aba4d3c25d0c63266cb4c is immutable.
The audit uses all sixteen stalled examples plus six fixed strongly-converged
controls, two per condition, selected in historical cache order with residual
<=0.0005. All identities are frozen in the versioned config before execution.

K=4, radius=.04 Å, beta=16.8, gamma=2, masks, current geometric eligibility,
targets, frozen Pg, objective terms and reductions remain unchanged. The
historical Cartesian term includes the source-residual coefficient 1e-5.
Pure float64 evaluation removes its prediction.float() cast only: it does not
change the mathematical objective. Local/chiral tensors and frozen loss
quartets are taken from the historical implementation without replacement.

V5 retained scalar records and coordinate SHA256 digests but not coordinates
or variables. Historical deterministic state recovery requires user approval
because the task forbids optimization before loading the final states. If
approved, bind an observing initialize function to an isolated clone of the
same solver code object, capturing the final variables without changing the
solver. Accept recovered states only after exact coordinate digests, metrics,
histories and stopping results match the historical record. Save recovered
tensors outside Git; version scalar recovery hashes. No new result replaces v5.

For delta=a*v and a=(1+||v||²/s²)^(-1/2), J=a*I-a³*v*v^T/s².
J has eigenvalues a³ in the radial direction and a in the two tangential
directions; condition number is a^-2=1+||v||²/s². At zero J=I.
Audit physical v=.04*z; historical solver gradients were with respect to z.
Report g_v=g_z/.04, g_delta and J^T*g_delta separately. Current eligibility
is Boolean, fixed on differentiable branches, and recomputed for perturbations.
Mask changes invalidate smooth directional interpretations rather than being
hidden. Ineligible correction variables are fixed, with zero feasible residual.

Extract correction-space gradients from independent copies of actual bounded
Cartesian delta tensors. At near-active balls (||delta||/s>=1-1e-6), minimize
||g+2*lambda*delta|| for lambda>=0. Lambda=max(0,-g·delta/(2||delta||²)).
The residual consists of tangential g and positive radial g (feasible inward
descent); negative radial g admits an outward-pointing constraint normal.
Interior residual is g. Record absolute residuals and normalize each example
by its maximum eligible ||g_delta||. Near-stationary tolerance is .001.
This correction-space residual is distinct from historical parameter gradients
and historical projected finite-step mappings.

Predeclared precision materiality: objective difference >max(1e-8,1e-6*|L|),
gradient relative difference >.001 or direction cosine<.999. Scientific metric
relative changes below 1e-4 are immaterial; inversion counts and assessability
are reported exactly. Three deterministic normalized directions and three
fixed float64 epsilon scales per space are in the config. Gradient validation
requires one passing epsilon per direction/space with absolute+relative
tolerance 1e-10+1e-5*max(|FD|,|AD|); all scales and errors remain visible.

Single shadow steps of 1e-6/1e-5/1e-4 Å are direct correction-space projected
negative KKT residual or active tangential gradient, normalized so the largest
per-residue direction norm is one. Reproject to the same closed .04 Å ball,
recompute current eligibility, and never iterate. Material descent exceeds
max(1e-8,1e-6*|L|). This is numerical telemetry, not a new optimizer.

Classify PREC-D if float64 directional validation/chain arithmetic fails;
PREC-B if suppressed v-gradients coexist with substantial correction KKT
residuals and material feasible descent; PREC-A if all stalled states are
near-KKT and no material feasible descent remains; PREC-C requires material
precision effects and improved convergence in the optional unchanged-settings
zero-init float64 replay; otherwise PREC-E. Preserve historical CART-O5.

Even a converged weighted-sum optimum does not answer maximum achievable local
repair under Cartesian/chirality safety constraints. A constrained local
feasibility objective asks a different question. Saturation or <5% weighted
oracle improvement cannot alone establish correction-budget insufficiency.

CPU only. No neural models, CUDA, environment changes, historical edits,
budget/coefficients changes or automatic downstream experiments.
