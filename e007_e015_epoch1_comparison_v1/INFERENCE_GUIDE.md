# Reading the inference figures

Traces show C-alpha positions only, colored from N to C. Fixed camera and common spatial limits reveal shrinkage or expansion. Intermediate frames are pre-update states; frame0 is the final clean estimate. DirectX0 is the model-implied clean estimate. At high noise many structures fit the same noisy input, so directX0 shrinkage is not automatically sampler collapse: inspect the actual trajectory, Rg and covariance.

Generation receives only noisy coordinates, timestep and length. E015 also receives its preceding model-generated clean estimate. Neither receives a target or sequence. E007 has no EMA/parity/chiral-head equivalents. Distance maps use Å. Rg/covariance measure size and shape, not stability. Galleries follow the frozen numeric score.

Refold cards show independently sequence-conditioned predictions. Confidence does not replace the TM/coverage/RMSD/finite-complete gates. Nearest confirmed PDB neighbors establish measured similarity, not exhaustive novelty. No experimental stability, solubility or function is inferred.
