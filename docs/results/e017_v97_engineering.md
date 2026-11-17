# Final V9.7 engineering qualification

{
  "FUNCTION_PRESERVATION": "PASS",
  "GATE_GRADIENT": "PASS",
  "POLICY_LIKELIHOOD": "PASS",
  "SAFETY_STATISTICAL_DESIGN": "QUALIFIED",
  "SAFETY_MONITOR_RELIABILITY": "QUALIFIED_UNDER_FROZEN_CONDITIONAL_IID_DESIGN; INCONCLUSIVE_PAUSES",
  "RECOVERY_QUALIFICATION": "PASS",
  "INITIAL_POLICY_IDENTICAL": "YES",
  "ADDITIONAL_PARAMETERS": 1,
  "ESTIMATED_TRAINING_RUNTIME": {
    "A_training_seconds": 89326.87986001838,
    "B_training_seconds": 100446.50670999545,
    "method": "one full engineering group per arm; coarse linear extrapolation, not a runtime guarantee",
    "safety_seconds_per_B_look": 13290.451474428177,
    "B_safety_seven_looks_seconds": 93033.16032099724,
    "full_pair_safety_attempts": 57344,
    "full_pair_fit_attempts": 160000,
    "final_validation_attempts": 89088,
    "checkpoint_bytes": 209399051,
    "available_D_bytes": 81435602944,
    "forecast_required_bytes": 15627478812.0,
    "storage_admission": true
  },
  "STORAGE_ADMISSION": "PASS",
  "ENGINEERING_READY": true,
  "READY_FOR_SUBSTANTIAL_RL": "NO",
  "BLOCKER": "Full4096-draw initialization admission and explicit production authorization remain pending; no campaign launched",
  "PRODUCTION_UPDATES": 0,
  "TEST_SEAL": "UNKNOWN_HISTORICAL_ACCESS; NO_TEST_CONTENT_ACCESSED"
}

The gate correction, policy/gradient/recovery pathways and fixed-look monitor design qualify under documented assumptions. All historical scientific conclusions remain unchanged. Full4096-candidate initialization safety admission is intentionally outstanding; the16 real monitor candidates exercise collection only. ENGINEERING_READY distinguishes implementation qualification from permission to start production. Both future arms still need actual safety PASS and reviewed authorization. No production process/optimizer schedule is launched.

New generation:128 equivalence attempts +16 randomized single-candidate monitor attempts +32 fresh fitting engineering attempts =176. Actual engineering optimizer calls4, comprising1 unique update per arm plus exact replay; production updates0. Synthetic different-length fixtures are robustness checks, not a changed task. All workers exit after their bounded checks. No TEST/VALIDATION candidate generation or J1 update occurred. Eight reports, training/validation protocols, resource forecast, machine-readable evidence and SHA256 provenance are included.

