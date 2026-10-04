# Robust D-optimal procurement comparator

Reference: S. Joshi and S. Boyd, *Sensor Selection via Convex Optimization*,
IEEE Transactions on Signal Processing 57(2), 451–462, 2009,
DOI 10.1109/TSP.2008.2007095. Section V-F, Eq. (21) maximizes the worst-case
log determinant. The implementation specializes this criterion to complete
affine source packets and enumerates the six-reference pools exactly. It does
not reproduce the paper's relaxation runtime.

For reference i with common value q_i in [l_i,u_i], declared positive report
scale epsilon_i, and weight w_i=1/epsilon_i^2, each source has information block

    M_S(q) = sum_{i in S} w_i [1,q_i]^T [1,q_i].

For m sources the full matrix is I_m tensor M_S(q), so m does not change the
selection order. The determinant admits the exact reduction

    det M_S(q) = sum_{i<j} w_i w_j (q_i-q_j)^2
                = (sum_i w_i) min_t sum_i w_i (q_i-t)^2,
    D_rob(S) = (sum_i w_i) min_t sum_i w_i dist(t,[l_i,u_i])^2.

The last objective is convex and piecewise quadratic. Its derivative condition
is sum_i w_i(t-clip(t,[l_i,u_i]))=0. An endpoint scan finds a rational root and
the attaining common values q_i=clip(t,[l_i,u_i]); both are preserved as witnesses.
Empty and common-intersection sets have zero robust determinant.

The fixed ranking is maximum robust determinant, maximum midpoint determinant,
minimum cost, minimum cardinality, then lexicographic reference IDs. No target
truth, attack label, empirical error, or SourceBridge width enters that objective.
The shared width certificate is used only by the identical precision-admission
test. Nonnegative reference prices are charged once per purchased column;
common coarse-measurement and target-report costs are added consistently.

replay_procurement.py recomputes all 7,680 subset determinants in 120 pools,
their attaining values and stationarity equations, 1,932 integer-budget actions,
240 selected budget cases, and 480 precision requests. The exact comparator
supports the primary cost comparison without attributing interval-certification
guarantees to statistical D-optimal design itself.
