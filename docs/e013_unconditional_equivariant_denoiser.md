# E013 unconditional scalar/vector coordinate generator v1

Base: `bc5ae4fe83a710f8b89401fa1d2b2d586f193d6e`.
Branch: `e013_unconditional_se3_coordinate_generator_v1`.
Worktree: `/mnt/d/Simone/proteinGen/.e013_worktree`.
Artifacts: `/mnt/d/Simone/proteinGen/e013_artifacts`.

## Scientific question and status

E007 separated denoiser quality from reverse-trajectory quality and enforced O(3)
symmetry, preventing explicit chirality sensitivity. E010 found local geometry
could conflict with Cartesian descent; its auxiliary diagnostic did not establish
sustained training benefit. E013 asks whether directional residue states and a
parity-sensitive local branch improve the same E007 coordinate-v target.

Architecture implemented; eight CPU tests passed including N=500 backward.
Preflight, training, evaluation and conditional sampling status are recorded below
as execution proceeds. One-step denoising cannot establish de novo generation,
foldability, stability or function.

## Inference information and tensors

The required distribution is **p(X | N)**. No amino-acid sequence is required.
Forward arguments are only noisy coordinates `[B,N,3]`, integer timestep `[B]`,
length `[B]`, and contiguous-prefix mask `[B,N]`. Connectivity follows chain
order. Current coordinates are centered internally; padding is sliced away before
computation, including nonfinite padded coordinates. N>=2 is supported.
There are no teacher corrections, targets, RMSD, shield or SLSQP inputs.

```mermaid
flowchart TD
  X[Current X_t, t, N, chain order] --> H[Scalar H: B x N x 128; vector V: B x N x 32 x 3, initially zero]
  H --> S[4 sparse scalar/vector blocks]
  S --> C[Segment mean pooling, stride 8]
  C --> G[2 dense coarse scalar/vector blocks with RoPE]
  G --> B[Channel-linear broadcast back to residues]
  B --> O[Scalar-weighted vector-channel output]
  X --> L[32 local features and assessable frame]
  L --> P[128 scalar channels, 2 polymer residual blocks, 3 frame coefficients]
  P --> F[Cartesian local output and reliability/time gate]
  O --> V[Sum, subtract valid centroid: B x N x 3]
  F --> V
```

Time: 32 sine/cosine frequencies, 64->64->64 SiLU MLP. Length features
`N/500, log(N)/log(500), 1/sqrt(N)` project 3->16->16. Position features
`i/(N-1), exp(-i/8), exp(-(N-1-i)/8), exp(-i/32), exp(-(N-1-i)/32)`.
Concatenated 85 scalars project to 128. No E012 length stack is used.

## Edges, radial basis and messages

Directed edges are the unique union of offsets +/-1/2/3 and 24 current-coordinate
nearest neighbors. Stable distance sort resolves exact ties by chain index;
row-major extraction fixes edge order. Spatial provenance remains set on an edge
also supplied by polymer order. All selected edges use `r=x_j-x_i`, `d=||r||`,
`u=r/(d+1e-6)`. kNN at maximal noise has no biological interpretation.

22 edge scalars: 16 Gaussian RBFs; signed `(j-i)/(N-1)`;
`log(1+|j-i|)/log(max(N,2))`; indicators for separations 1/2/3; spatial flag.
RBF centers are uniformly spaced 0..5; gamma is inverse squared spacing.
These are overlapping Gaussian functions, not hard bins. The frozen scale
12.22820347644835 Å means the range extends to 61.14101738224175 Å.
The verified train-only normalization hash is
`1be9ef05c82110861482eb46e61ef433045fa9678b9cdfffd921f0defebb03cd`.
Source: original `coordinate_normalization_v1/normalization.json` under E007
reports. Its fitted identity hash is
`d3f2ee345d0e4e9a27d0136aad1eca9dfe87821a3a82d90aa4c3194bd928a3f9`.

Each block pre-normalizes H with LayerNorm. V is divided by
`sqrt(1+mean_c ||V_c||²)`. Invariants `dot(V_i,c,u_ij)` join h_i/h_j,
22 edge scalars and 64 time scalars in the message MLP. It emits 128 scalar
messages, 32 directional coefficients, 32 vector gates, and one attention logit.
32-dimensional q/k apply RoPE to interleaved pairs using frequencies
`10000^(-m/16)`, m=0..15 and signed chain positions. Edge logits add q_i·k_j/sqrt(32).
Stable per-receiver softmax weights vector messages `a*u+tanh(b)*V_j`.
Scalar-channel linear vector mixing and scalar MLP updates use residual factor
0.1. Scalar updates include vector norms; no raw xyz enters scalar MLPs.

Coarse nodes are contiguous segments: `segment(i)=floor(i/8)`, M=ceil(N/8).
H_c,V_c,X_c are arithmetic means over actual segment membership. Counts exclude
padding and include the partial final segment. Coarse directed all-pairs edges
exclude self; RoPE uses segment order. After two blocks, scalar and vector channel
linear maps broadcast segment states by membership and add them to residues.

## Local features and output

Use a=x_i-x_(i-1), b=x_(i+1)-x_i, e1=b/||b||,
e2=normalize(a-dot(a,e1)e1), e3=cross(e1,e2). Frame columns are e1/e2/e3.
Missing neighbors or norms <=1e-6 mark invalid frames and set the frame to zero;
denominator protection never supplies a global fallback axis.

32 features: six distances, eighteen frame coordinates at offsets -3/-2/-1/+1/+2/+3,
angle cosine, torsion sine/cosine, normalized triple product, frame validity,
torsion validity, sin(angle) reliability, and fraction of available neighbors.
For a,b,c consecutive bonds, torsion uses n1=normalize(a×b), n2=normalize(b×c):
cos=dot(n1,n2), sin=dot(cross(n1,n2),normalize(b)); q=dot(a×b,c)/(||a||||b||||c||).
These signed terms change under reflections. Invalid torsions are zero with
explicit validity. Local features project to 128; two mean-neighbor polymer
residual blocks feed a 3-coefficient head. Frame multiplication returns `[N,3]`.

Global output is `sum_c MLP(H)_c V_c / sqrt(32)`.
Local multiplier is `sigmoid(MLP(time,reliability))*reliability*(1-t/499)^2`.
This declared smooth envelope suppresses high-noise frames; it is a design choice
requiring a later ablation. Sum global and gated local outputs, then subtract the
valid centroid. All operations guarantee translation invariance and proper
rotation equivariance mathematically; reflections are deliberately unconstrained.

## Diffusion, loss, comparison and gates

Reuse E007 `CoordinateVPDiffusion(500)` read-only, including cosine schedule:
X_t=alpha_t X0+sigma_t epsilon; v*=alpha_t epsilon-sigma_t X0;
X0_hat=alpha_t X_t-sigma_t v_hat. X0 and Gaussian epsilon are centered.
Normalize clean coordinates by the frozen scale. Loss is mean over proteins of
masked squared component error divided by 3N_valid. The local offset-distance
auxiliary helper is available but all auxiliary coefficients are zero.
Historical E007 config has auxiliary losses: the prospective comparison must
train the E007 architecture with the same v-only objective and matched corruptions,
optimizer/exposure. A historical checkpoint alone has this objective confound.

Preflight thresholds fixed before real-data evaluation: max and relative rotation
error <=5e-5; finite output and gradients; reflection discrepancy >1e-6;
N=500 peak reserved memory <=5 GiB. Synthetic nonzero reflected-output difference
is an implementation check, not a chirality performance metric.
CPU smoke uses one thread while the 16-thread UniRef50 MMseqs job is active.

Stage C requires verified source/shard and conservative 30%-identity/80%-coverage
clean-validation provenance, deterministic bounded training and prospective
validation identities in five length strata, and frozen training-only numeric
thresholds for aggregate non-inferiority, length regression, collapse and one
failure-family improvement. Noise steps proposed: 5/50/250/400/490. No test tuning.
Diagnostics must include aligned reference error, v loss, offsets 1/2/3,
adjacent distribution, torsions/triple sign and denominators, clashes, gyration,
contacts, finite/collapse and length/noise strata. Stage D requires C evidence;
it must distinguish target-associated rollout diagnostics from unconditional
generation. Stage E requires C and D pass and uses target-independent metrics.
Numeric scientific gates are not yet frozen; C is not authorized by preflight alone.

## Complexity and limitations

Polymer messages O(N), spatial messages O(kN), coarse communication O((N/8)²),
local features/messages O(N). **Current kNN construction uses dense cdist and
stable sorting: O(N² log N) time and O(N²) storage.** Edge discovery therefore is
not fully linear; only residue message blocks are sparse. E007 dense pair-grid
states scale O(N²). This pilot processes variable-length proteins separately.
Message and attention state at coarse nodes are dense only in M.

Self-conditioning, time-aware radial basis, flow matching, richer backbone and
sequence conditioning are proposed future experiments, not implemented/activated.
Optional geometry helper is prepared only. This branch preserves all historical
E007/E010 files and artifacts. No biochemical claim follows from any current check.

## Execution record

Plan and CUDA preflight executed at architecture commit
`5dd02cd5475133ab51ca36a37b69dad454926e43`, configuration hash
`301fddb0e47ba8caa09082cfde5baa85a570581364c365cdb136cbfb7c81f07f`.
Parameter count **980,794**. Environment: PyTorch 2.13.0+cu130, CUDA 13.0;
GPU RTX 5060 8151 MiB, driver 610.57.01. TF32 disabled, float32 throughout.
Synthetic timings are five-call means under concurrent sequence-production CPU
load; nonmonotonic values should not be interpreted as throughput scaling laws.

| N | Forward ms | Forward/backward ms | Peak allocated MiB | Reserved MiB | Max symmetry error |
|---|---:|---:|---:|---:|---:|
|64|67.36|482.84|108.93|116|6.33e-8|
|128|52.84|234.66|152.55|176|1.04e-7|
|256|100.78|376.22|242.56|278|1.19e-7|
|384|71.13|179.65|346.31|376|7.45e-8|
|500|47.18|230.57|435.50|486|9.69e-8|

Relative equivariance errors ranged 4.10e-7..5.91e-7; all finite and memory
gates passed. Reflection discrepancies were 0.95..2.63 L2, confirming the
active branch is not constrained to O(3). Immutable reports: `plan_cpu.json`,
`preflight_cuda.json` under the artifact directory. The CPU test suite passed.

Stage C protocol now implemented: preserve the historical bounded panel, intersect
validation identities with the hash-verified identity-30 clean manifest, and verify
each selected source shard against the immutable dataset protocol. This yields
20 training and 17 validation identities (four per stratum except one at 385–500).
Projected data reads never request sequence. Full dataset rehash is not required
for unselected shards; all accessed shards are hashed.
Proposed bounded exposure: 400 AdamW updates/model, physical batch one, lr 2e-4,
weight decay 0.01, clip 1. Both models receive identical clean proteins, t and epsilon.
No checkpoint selection uses validation. E007 parameter count is recorded at launch.

The prospective gate algorithm is fixed before calibration: use initial control
training-only duplicate noise-replicate v losses at t=250; let delta be their
paired-difference standard deviation divided by sqrt(2). Aggregate tolerance is
`min(0.10,max(0.02,2*delta/(sqrt(n_train)*mean_train_loss)))`; length tolerance is
twice this. Require >=5% local-distance or torsion MSE improvement, finite outputs,
and no reconstruction with radius less than 25% of its reference. Freeze these
values in protocol.json before any prospective validation inference. This is a
feasibility gate, not a claim of statistical superiority or training convergence.
Training and validation are not yet executed at this documentation boundary.
