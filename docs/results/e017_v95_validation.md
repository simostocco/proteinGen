# Final E017 V9.5 expanded VALIDATION evaluation

**Completed frozen-model development-stage validation; not an unexposed replication or TEST confirmation.**

768 fixed-sequence identities in 768 protected groups; 1536 primary goals plus 256 added goal-response specifications. 69632 attempts across frozen J1, initial/trained raw, initial/trained R3 filtered actors. All batches committed exactly once with deterministic seeds. Neural optimizer updates:0. TEST seal intact.

| Policy | K1 / goals | K8 / goals | Success / attempts | Hard / attempts | Backbone / attempts | Exposure / attempts | Burial / attempts | Aborts | Overpacking | Reward | Error complete-only | Pair RMSD | Proposal KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| J1 | 63/1536 | 417/1536 | 502/12288 | 5982/12288 | 6049/12288 | 4636/12288 | 4268/12288 | 0 | 806 | 0.161774 | 0.382138 | 15.260845 | 0.00000000 |
| initial_raw | 51/1536 | 363/1536 | 439/12288 | 5882/12288 | 5940/12288 | 5223/12288 | 3703/12288 | 0 | 664 | 0.155079 | 0.387332 | 15.815057 | 0.01167205 |
| initial_filtered | 127/1536 | 721/1536 | 1057/12288 | 11964/12288 | 11964/12288 | 5349/12288 | 3203/12288 | 323 | 0 | 0.324465 | 0.380865 | 15.723050 | 0.01165946 |
| trained_raw | 62/1536 | 443/1536 | 539/12288 | 5763/12288 | 5815/12288 | 4979/12288 | 4516/12288 | 0 | 845 | 0.162390 | 0.350849 | 15.194990 | 0.00637501 |
| trained_filtered | 154/1536 | 813/1536 | 1310/12288 | 11843/12288 | 11843/12288 | 5104/12288 | 3885/12288 | 445 | 0 | 0.344260 | 0.343181 | 15.103778 | 0.00631540 |

## Paired effects and interpretation

Filtered K1 change: 1.758 pp [0.260, 3.190] (95.000% CI). Filtered K8 change: 5.990 pp [2.604, 9.310] (99.167% CI). Raw K1 change: 0.716 pp [-0.326, 1.758] (95.000% CI); raw K8: 5.208 pp [2.018, 8.464] (99.167% CI). Primary discordance: {'0->0': 1327, '0->1': 82, '1->0': 55, '1->1': 72}. Original promotion expression on this revised validation panel passes: False; this cannot rewrite the original32-goal primary NO-GO. Safeguards pass: True.

Goal-specificity: NOT_ESTABLISHED_MEANINGFUL. Positive yield without meaningful directional raw goal response is not fully demonstrated property-conditioned steering. See explicit matched target interventions and adjusted intervals. Target error, exposure/burial, structural diversity and validity may move differently; all are reported. Previous E016 model-selection exposure is allowed and creates possible selection bias beyond these confidence intervals.

## Recommendation

improving low-level credit assignment and goal conditioning. No new campaign is launched. Complete JSON, identity-paired results, exposure-history/difficulty sensitivity, frozen protocol, model hashes, source bindings and cost records accompany this report. Source scientific revision 91576ff071bd42a07f4306bc315b496d22d5a3bf; manifest SHA256 318c3d11bed70248bd57b234cbdad16414ddae662d79cd84a9fb7a5a3628fa2c. Exact model tensors and buffers were checked before evaluation. Scientific samplers, rewards, eight-proposal support, filters and checkpoints remain unchanged.

C-alpha proxy success does not demonstrate protein folding stability, function or learned amino-acid design. These fixed-sequence outputs address geometric generation only. TEST remains reserved for later final assessment.

