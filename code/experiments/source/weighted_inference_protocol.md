# Shared weighted affine calibration

For reference center q_j, half-width h_j, and report error scale epsilon_j,
the shared WLS front end uses weight w_j=3/(epsilon_j^2+h_j^2). Weights are
normalized to sum to the number of references. For each source i, fit a gain
from weighted centered cross-products and clip it to [0.7,1.3]; fit the bias
from the weighted residual mean. Transform reports as (Y_ji-b_i)/a_i.

Zero or one reference and equal-weight cases use the declared common adapter
branches in weighted_inference.py. Reference-only MLNI selection is performed
with fold-local WLS parameters and reference centers. Proper-public-prior MLNI
uses the same WLS input and the separate fixed prior specification.
The weighting is a nominal statistical interface, not a deterministic coverage
claim. All compared methods within a profile receive the same purchased input.
