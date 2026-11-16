# Computational comparison verdict

- E007_reconstruction: promising
- E007_unconditional_generation: bounded but weak
- E007_designability: INCONCLUSIVE
- E015_designability: INCONCLUSIVE
- matched_epoch1_comparison: inconclusive
- practical_de_novo_design: inconclusive
- qualification: Exposure-matched reconstruction/generation and, where the positive-control gate permits, designability are separately reported from the314,256-exposure E015 context. Equal exposure does not isolate architecture. Generation maturity labels are descriptive, not a paired earlier-E007 improvement claim.

Positive controls: E007 TOOLCHAIN_NOT_VALIDATED; E015 TOOLCHAIN_NOT_VALIDATED. Failed controls make model designability inconclusive, never evidence of non-designability.

ProteinMPNN scores and AlphaFold confidence are supporting evidence; neither replaces target TM-score/coverage/RMSD agreement. A low PDB similarity without successful refolding is not novelty. These are computational results, not experimental stability, solubility or functionality.

Reconstruction: E007_RECONSTRUCTION.json. Generation: E007_GENERATION.json and generation_distribution_metrics.json. Designability: E007_EXTERNAL_DESIGNABILITY.json, e007_inverse_folding/designability_summary.parquet, and the referenced E015 external root. Nearest confirmed PDB hits and novelty tables remain separate. Galleries:e007_galleries/, comparison_galleries/, and e007_inverse_folding/candidates/.
