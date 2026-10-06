# Bounded correction oracle v4

The pre-registered protocol is in
[the v4 document](../../../../../docs/e010_phase4d_bounded_oracle_v4.md) and
[config](../../../../../configs/e010_phase4d_bounded_oracle_v4.yaml).
The source recurrent-capacity result is immutable at
`43864b38001c2ee0118bc8b7530b547cba4ca5f1`.

Only independent free coordinate variables are optimized. No neural parameters,
global forward, GPU use, panel redraw or changed scientific coefficients.
K=4; s_max=.04 Å; beta=16.8; gamma=2; exact v3 panel objective reductions.
CPU L-BFGS settings and convergence checks are fixed before panel execution.
`execution_contract.json` pins the implementation/config and all tracked v3
records; the immutable cache and historical source pins are also checked.
`synthetic_preflight.json` records numerical preflight without panel outcomes.

Focused CPU tests: **80 passed**, including 10 v4 cases and inherited hybrid,
objective-v2, diagnostic and recurrent-capacity checks. Ruff checks passed.
The previous full-suite baseline has 134 inherited failures; the complete suite
is not repeated for this isolated diagnostic. No environment modification.

Every example result and final aggregate are added separately after execution.
No cache, dataset or coordinate checkpoint is committed. Classification remains
inconclusive if numerical convergence fails, or if a low fixed-objective result
cannot distinguish geometric insufficiency from objective limitation.
