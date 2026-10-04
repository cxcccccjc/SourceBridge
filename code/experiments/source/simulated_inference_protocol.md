# Span and target-position mechanism protocol

This synthetic diagnostic uses six references at
250+span*[-0.5,-0.3,-0.1,0.1,0.3,0.5], spans 20,150,450, targets 40,250,460,
and seeds 419131,419132. The same source gain/bias/noise and interval offsets
are shared across spans and target positions within a seed. Acquisition sees
public intervals and T=[0,500], without target truth or future reports.

Exact and heterogeneous half-widths [2,5,10,20,40,80], source-report prices
[1,2,1,2,1,2], and budgets 4,7 are fixed. There are nine sources, f=2,
gain bounds [0.7,1.3], epsilon=5, and two cases: affine honest responses and
a +80 target-report shift on the last two source identities. All 576 inference
conditions are retained.

The diagnostic contains CRH, EPTD, FETD-AK, FETD-D, and MLNI with native and
shared WLS interfaces, a WLS median control, and the interval-depth midpoint.
The reference-only MLNI prior and its reference-only selection procedure remain
separate from the primary proper-public-prior MLNI interface. Precision
requirements, cost components, pre-query W, post-observation hulls, and point
errors are evaluated as distinct quantities. This diagnostic is not an
independent real-world performance trial.
