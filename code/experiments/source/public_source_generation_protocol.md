# Public source-generation and calibration protocol

The shared generator implements an offline bounded-affine observation model.
Public station/time measurements provide target and reference values; source
gain, source bias, report noise, and interval offsets are generated from the
recorded identity-based random seeds. The generation, purchase, fulfillment,
prediction, and evaluation stages have separate functions and input records.

The base public protocol uses eight target stations and dates 2014-05-22 through
2014-05-28. The primary later-block protocol invokes the same generator with
the dates and prior rule recorded in public_inference_protocol.json. These
protocols are separate; the seven-day base block is not pooled into the primary
28-day results.

Reference stations are 001003, 001004, and 001005 at hours 11 and 13. Target hour
is 12. The model has nine source identities, f=2, T=[0,500], gain [0.7,1.3], and
epsilon=5. Honest simulated noise is uniform on [-4.5,4.5]. Exact and
heterogeneous interval profiles use report prices [1,2,1,2,1,2] and budget 4.
The original source-seed namespace is retained byte-for-byte to reproduce the
same random responses. Its literal spelling has no version-selection behavior.

All native, common-WLS, midpoint, and nominal-plus-certificate interfaces and
the complete fixed attack library remain available in the source modules.
Predictions are saved before evaluation truths are loaded. Repeated attacks
and profiles are correlated conditions, not independent observational targets.
