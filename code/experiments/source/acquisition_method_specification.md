# Acquisition and inference implementation contract

The scientific workflow separates public candidate information, anchor purchase,
complete report fulfillment, point prediction, and evaluation. Candidate geometry
uses T=[0,500], gain bounds [0.7,1.3], epsilon=5, nine source identities, and f=2.
The development protocol has its own dates, stations, seeds, budgets, and cases
in acquisition_evaluation_protocol.json; it is distinct from the later-block
primary evaluation in public_inference_protocol.json.

The source modules preserve native and common-WLS interfaces for CRH, EPTD,
FETD-AK, FETD-D and MLNI. The reference-only MLNI prior and the proper-public
prior have separate labels and implementations. Reference-only cross-validation
uses fold-local adapters and calibration centers, without evaluator truth. The
source-level packet geometry and certificate projection are separate modules.

All costs include only purchased data; repeat timing, solver state, convergence,
unavailable inputs, exact selection checks, and conditional validity are explicit
output fields. The complete record is authoritative for point-error comparisons;
theoretical certificate claims follow the manuscript assumptions.
