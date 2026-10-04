SourceBridge error metrics

Purpose
  Recompute the main point-estimator table and supplementary error summaries
  from frozen predictions. All 11 displayed interfaces, both reference
  profiles and all 213 targets per profile are retained. No experiment or
  bootstrap is rerun. All reported error measures use micrograms per cubic
  meter; lower values are better.

Run from the package root
  python code/analysis/compute_error_metrics.py

  Python 3.10+ is sufficient; only the standard library is used. Inputs are
  resolved relative to the script, so the command also works when invoked by
  absolute path from another working directory. The directory structure
  inside code/analysis/ must be preserved.

  Default reports:
    code/analysis/results/error_metrics.json
    code/analysis/results/error_metrics.csv
  Use --output-dir PATH to write reports elsewhere. The script does not edit
  TeX, figures, PDFs or the frozen inputs.

Bundled sufficient statistics
  data/target_error_summaries.csv contains 4,686 method/profile/target records:
  the original targetwise maximum squared error across the 14-condition
  library and the clean absolute error. Maximum squared errors were verified
  against the original metrics.json event records and the full prediction CSV.

  data/condition_error_summaries.csv contains 308 method/profile/condition
  records: target count, sum of squared errors, sum of absolute errors and
  observed maximum absolute error. Each record contains all 213 targets.
  These sums permit direct recomputation of pooled RMSE and clean errors.

  data/error_metric_provenance.json gives the exact library, interface mapping,
  original-source hashes, bundled-input hashes and extraction checks. The
  script verifies the bundled hashes before calculating the metrics.
  The complete predictions and metrics.json are provided in code/experiments;
  they are not duplicated within this analysis directory.

Definitions
  e_i,a = prediction_i,a - truth_i; d_i = max_{a in A}|e_i,a|.
  N=213, |A|=14; all target-condition cells have equal weight in pooled RMSE.

  Main table, per profile:
    Clean RMSE = sqrt(mean_i e_i,clean^2).
    Pooled RMSE = sqrt(sum_i sum_{a in A} e_i,a^2 / (213*14)).
    Targetwise worst R_A = sqrt(mean_i d_i^2).
    Clean MAE = mean_i |e_i,clean|.

  Supplementary table, per profile:
    M_A = mean_i d_i.
    Q95(d) = Type-7 linear sample quantile; with 213 sorted values it is
      0.6*d_(202) + 0.4*d_(203), using one-based ascending ranks.
    Observed MaxAE = max_i d_i.

  The CSV and JSON additionally retain the mean of the 22 largest d_i values
  (ceil(10%*213), covering 10.3286% of targets) and clean MaxAE. This tail mean
  is not a fractionally weighted exact-10% empirical CVaR.
  The two whole-packet translation controls are excluded from A, exactly as
  for the original R_A. Their exclusion is not chosen separately per metric.
  Rankings use all 11 displayed interfaces and unrounded values; equality
  tolerance is 1e-12. TeX tables display three decimals.

Optional independent source-record verification
  python code/analysis/compute_error_metrics.py --predictions-csv code/experiments/data/public_inference_cells.csv

  The complete original prediction table is included under the formal name:
    code/experiments/data/public_inference_cells.csv
  This option checks its SHA-256, all 102,240 predictions, error arithmetic,
  6,390 complete event/interface/profile bundles, 16 conditions per bundle,
  the exact 14-condition library, all bundled target and condition summaries,
  and the 6,816 equal SourceBridge/proper-MLNI prediction pairs.

Interpretation
  Pooled RMSE describes equal weighting of the declared finite library; this
  is not an estimated real-world distribution of attacks. R_A instead gives
  each target its most damaging recorded condition before aggregation.
  Observed maxima are sample summaries, not uniform conditional guarantees.
  No confidence interval or significance claim is attached to the newly
  aggregated summaries. The existing date-block intervals pertain to their
  specified clean-RMSE and targetwise worst-RMSE (R_A) comparisons.

  SourceBridge has the smallest pooled RMSE among the 11 displayed interfaces
  in both profiles. WLS median has smaller M_A in both profiles and smaller
  heterogeneous R_A. SourceBridge also does not minimize observed MaxAE.
  All these results are retained in the numerical reports and relevant tables.

  SourceBridge and its matched proper-prior MLNI comparator have identical
  predictions on all 6,816 recorded cells. Their prediction-error metrics
  therefore coincide; the displayed ranking does not establish a point-error
  advantage over proper MLNI. Procurement, admission and conditional-radius
  guarantees are evaluated separately from these estimator error summaries.
