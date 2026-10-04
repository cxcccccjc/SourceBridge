# Selection-scaling protocol

The fixed candidate sizes are 8,16,32,64; seeds are 731991 and 731992; budgets
are 4 and 8. Public centers are uniform on [0,500], half-widths are drawn from
[0,2,5,10,20,50], and prices from integers 1 through 4. The input order is retained.

Both exhaustive support pairing and additive representative selection use the
same precomputed floating-point LP catalog and selected-union verification.
Catalog construction is measured separately. Each method receives one warmup
and three retained repetitions, with alternating execution order. All result
tuples, widths, timings, and catalog hashes are saved before validation.
The exact generation and measurement fields are in selection_scaling_protocol.json.

The separate run_timing.py workflow measures complete construction-plus-query
batches at n=8,32,64 and K=1,10. These two timing designs remain distinct.
