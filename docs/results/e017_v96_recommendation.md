# Final controlled V9.6 scientific recommendation

{
  "BURIAL_SPECIFICITY_IMPROVEMENT": "PARTIAL_BUDGET_ONLY",
  "TOKEN_VS_ADDITIONAL_TRAINING": "NOT_ESTABLISHED",
  "EXPOSURE_PRESERVATION": "NOT_ESTABLISHED",
  "K1_PRACTICAL_PROMOTION": {
    "Af": false,
    "Bf": false
  },
  "K8_IMPROVEMENT": {
    "Af": false,
    "Bf": false
  },
  "RAW_ACTOR_LEARNING": {
    "Af": false,
    "Bf": false
  },
  "VALIDITY_TRADEOFF": {
    "Af": true,
    "Bf": true
  },
  "TRAINING_STABILITY": {
    "A": "REGISTERED_SAFETY_STOP",
    "B": "REGISTERED_SAFETY_STOP"
  },
  "GOAL_CONDITIONING": "INCOMPLETE_OR_NOT_ESTABLISHED",
  "TEST_SEAL": "UNKNOWN_HISTORICAL_ACCESS",
  "equal_full_budget": false,
  "one_seed_per_arm": true
}

| Policy | K1 /1024 | K8 /1024 | Success /8192 | Hard /8192 | Backbone /8192 | Aborts | RMSD | Proposal KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| J1 | 25 | 247 | 300 | 4008 | 4058 | 0 | 15.364829 | 0.00000000 |
| A0_raw | 42 | 308 | 371 | 3903 | 3944 | 0 | 15.243884 | 0.00650586 |
| A0_filtered | 94 | 540 | 861 | 7899 | 7899 | 292 | 15.176301 | 0.00643920 |
| B0_raw | 37 | 239 | 288 | 3938 | 3993 | 0 | 15.397828 | 0.00777329 |
| B0_filtered | 79 | 458 | 705 | 7916 | 7916 | 274 | 15.306855 | 0.00771575 |
| Af_raw | 41 | 317 | 392 | 3920 | 3963 | 0 | 15.302489 | 0.00667643 |
| Af_filtered | 93 | 538 | 872 | 7933 | 7933 | 258 | 15.186921 | 0.00663775 |
| Bf_raw | 32 | 241 | 286 | 3939 | 3980 | 0 | 15.419943 | 0.00693357 |
| Bf_filtered | 80 | 459 | 708 | 7899 | 7899 | 291 | 15.292202 | 0.00687942 |

Training exposure: {"A": {"updates": 500, "attempts": 8000, "status": "REGISTERED_SAFETY_STOP"}, "B": {"updates": 250, "attempts": 4000, "status": "REGISTERED_SAFETY_STOP"}}

Equal-budget causal interpretation is permitted only if both arms completed5000 updates/80000 fitting attempts. Otherwise this is a partial/asymmetric study and cannot establish a controlled full-budget token advantage. One training seed per arm does not establish algorithmic robustness; confidence intervals resample validation identities conditional on these fitted models.

Continue low-level conditioning/credit research unless meaningful K1 promotion, raw burial specificity, exposure preservation and validity/diversity safeguards all support escalation. Do not automatically launch AA training, hierarchical RL, another experiment, or TEST. Historical V9.5 and V9.6 preparation artifacts remain unchanged.
