# V9.7 full initialization admission

{
  "UTC": "2026-10-31T16:04:10.169290+00:00",
  "V97_INITIALIZATION_ADMISSION": "PASS",
  "arms": {
    "A": "PASS",
    "B": "PASS"
  },
  "STATISTICAL_DESIGN_ADHERENCE": "VERIFIED",
  "RECOVERY_QUALIFICATION": "PASS",
  "STORAGE_ADMISSION": "PASS",
  "D_free_bytes": 81222221824,
  "required_staging_reserve_bytes": 18193760562.0,
  "READY_FOR_PAIRED_5000_UPDATE_TRAINING": true,
  "production_authorized": false,
  "PRODUCTION_UPDATES": 0,
  "TEST_SEAL": "UNKNOWN_HISTORICAL_ACCESS; NO_TEST_CONTENT_ACCESSED",
  "next_action": "Request separate production authorization; no automatic training",
  "comparison": {
    "same_frozen_seed_arrays": true,
    "records_reused_between_arms": false,
    "matching_primary_validity": true,
    "cross_arm_independence_not_required": true
  }
}

Arm A: PASS; backbone 3947/4096; hard 3947/4096; aborts 149/4096; adjusted backbone interval [0.9535635773878443, 0.9721074486318335].
Arm B: PASS; backbone 3947/4096; hard 3947/4096; aborts 149/4096; adjusted backbone interval [0.9535635773878443, 0.9721074486318335].

Both arms used the frozen matched task/seed arrays and separately generated results. B own qualified16-prefix was recovered; no ArmA records were used as B data. Counts are conditional on the frozen finite TRAIN monitoring population; no claim of arbitrary-cluster binomial validity is made. No VALIDATION/TEST inference or optimizer operation occurred. All GPU workers exited.
