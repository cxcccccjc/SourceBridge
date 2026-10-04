# Later-block public inference protocol

The primary semi-synthetic evaluation uses the official MSRA Data-1.zip archive,
fixed dates 2014-05-29 through 2014-06-25, eight observed target stations at
hour 12, and three reference stations at hours 11 and 13. There are 224 planned
station/date targets and 168 planned reference observations; 213 targets are
observed in each reference profile. Missing, conflicting, negative, and
out-of-domain values retain their declared states. The experiment is offline.

Public inputs are T=[0,500], gain bounds [0.7,1.3], report bound 5, nine sources,
and at most two controlled identities. Exact and heterogeneous reference
profiles use the fixed costs and half-widths in public_inference_protocol.json.
The 16 conditions contain a fixed 14-condition main library and two whole-packet
translations reported separately. Seeds and all scientific output labels remain
identical to the frozen protocol; names inside hash namespaces are data identities.

All 15 inference interfaces are retained, including the reference-only and
proper-public-prior MLNI variants and both certificate compositions. Published
cores use the same purchased reports within a profile. Different profiles may
purchase different subsets. Proper MLNI and SourceBridge share their nominal
estimate; SourceBridge additionally supplies acquisition, admission, and the
conditional certificate. See mlni_prior_specification.md for the prior formulas.

The data directory preserves all 102,240 prediction rows. The replay fixture
contains 64 conditions, both reference profiles, all 16 conditions and all 15
interfaces at the first and last observed events. The model replay reads no
evaluation truths. Complete-row reanalysis separately uses truth for errors,
paired day-block bootstrap intervals, and procurement comparisons.

The frozen JSON protocol records archive hashes, source identities, dates,
station IDs, condition definitions, iteration limits, and information boundaries.
Rerunning or summarizing these records is reproduction, not independent data.
