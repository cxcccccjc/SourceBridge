from release_integrity import verify_identity, resolve_resource
"""Joshi--Boyd (IEEE TSP 2009) robust D-optimal objective, exact small-pool adapter.

Keeps Eq. (21)'s max worst-case logdet objective; combines Sec. V-B's
budget constraint and Sec. V-C's grouped/vector measurements. The shared
calibration value q_i lies in the supplied interval; one selected anchor buys
the complete source group. No target truth, SourceBridge W, or future report
enters selection. This enumerates small pools exactly, and does NOT reproduce
the paper's convex-relaxation algorithm or its timing.

An anchor is (lower, upper, positive_nominal_noise_scale). The default design
weight is 1/scale**2, using the given positive report-error specification as a
declared nominal scale, not claiming a hard bound is a Gaussian variance.
Zero scales are rejected: no arbitrary ridge/noise floor is introduced.
"""
from fractions import Fraction as F
from itertools import combinations
import math
import numbers
import operator
import time

PAPER = 'Joshi and Boyd, Sensor Selection via Convex Optimization, IEEE TSP 2009'
DOI = '10.1109/TSP.2008.2007095'
METHOD = 'JoshiBoyd2009_RobustDOptimal_exact_small_pool'


def _rational(value, label):
    if isinstance(value, bool):
        raise ValueError(label+' must be a finite rational number, not bool')
    try:
        return F(value)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError(label+' must be a finite rational number') from exc


def _model(anchors, costs):
    parsed = []
    for row in anchors:
        try:
            lo, hi, scale = row
        except (TypeError, ValueError) as exc:
            raise ValueError('Each anchor needs (lower, upper, positive scale)') from exc
        lo, hi, scale = (_rational(v, 'anchor entry') for v in (lo, hi, scale))
        if lo > hi or scale <= 0:
            raise ValueError('Require lower <= upper and a strictly positive nominal scale')
        parsed.append((lo, hi, scale))
    costs = tuple(_rational(v, 'cost') for v in costs)
    if len(costs) != len(parsed) or any(v < 0 for v in costs):
        raise ValueError('One nonnegative cost is required for every anchor')
    return tuple(parsed), costs


def minimum_box_determinant(intervals, weights):
    """Exact min det(sum w_i [1,q_i]'[1,q_i]) for q_i in intervals.

    With W=sum w, this is W*min_t sum w_i*dist(t,I_i)**2. Its derivative
    is linear between interval endpoints, so a rational root is sufficient.
    Returns an attaining shared-q witness and a zero derivative certificate.
    """
    intervals = tuple(tuple(_rational(v, 'interval endpoint') for v in row)
                      for row in intervals)
    weights = tuple(_rational(v, 'weight') for v in weights)
    if len(intervals) != len(weights) or any(len(row) != 2 for row in intervals):
        raise ValueError('Need one positive weight per two-endpoint interval')
    if any(lo > hi for lo, hi in intervals) or any(w <= 0 for w in weights):
        raise ValueError('Valid closed intervals and strictly positive weights required')
    if not intervals:
        return dict(determinant=F(0), worst_values=(), center=None,
                    weight_sum=F(0), stationarity=F(0),
                    information_entries=(F(0), F(0), F(0)), robust_full_rank=False)

    def clip(t, interval):
        return max(interval[0], min(interval[1], t))

    def gradient_half(t):
        return sum((w*(t-clip(t, interval))
                    for interval, w in zip(intervals, weights)), F(0))

    common_lo = max(lo for lo, hi in intervals)
    common_hi = min(hi for lo, hi in intervals)
    if common_lo <= common_hi:
        center = (common_lo+common_hi)/2
    else:
        endpoints = sorted({v for interval in intervals for v in interval})
        center = None
        left = endpoints[0]
        g_left = gradient_half(left)
        for right in endpoints[1:]:
            g_right = gradient_half(right)
            if g_left == 0:
                center = left
                break
            if g_right == 0:
                center = right
                break
            if g_left < 0 < g_right:
                center = left-g_left*(right-left)/(g_right-g_left)
                break
            left, g_left = right, g_right
        if center is None:
            raise ArithmeticError('Convex interval-distance objective had no derivative root')
    q = tuple(clip(center, interval) for interval in intervals)
    s0 = sum(weights, F(0))
    s1 = sum((w*x for w, x in zip(weights, q)), F(0))
    s2 = sum((w*x*x for w, x in zip(weights, q)), F(0))
    determinant = s0*s2-s1*s1
    stationarity = gradient_half(center)
    assert stationarity == 0 and determinant >= 0
    assert determinant == s0*sum((w*(x-center)**2 for w, x in zip(weights, q)), F(0))
    return dict(determinant=determinant, worst_values=q, center=center,
                weight_sum=s0, stationarity=stationarity,
                information_entries=(s0, s1, s2), robust_full_rank=determinant > 0)


def build_catalog(anchors, costs):
    """Enumerate every subset; practical for the existing pools of n <= 6."""
    anchors, costs = _model(anchors, costs)
    tick = time.perf_counter()
    entries = []
    for k in range(len(anchors)+1):
        for ids in combinations(range(len(anchors)), k):
            intervals = [(anchors[i][0], anchors[i][1]) for i in ids]
            weights = [1/(anchors[i][2]**2) for i in ids]
            value = minimum_box_determinant(intervals, weights)
            centers = [(lo+hi)/2 for lo, hi in intervals]
            nominal_det = sum((weights[i]*weights[j]*(centers[i]-centers[j])**2
                               for i, j in combinations(range(len(ids)), 2)), F(0))
            entries.append(dict(ids=ids, cost=sum((costs[i] for i in ids), F(0)),
                                nominal_determinant=nominal_det, **value))
    return dict(method=METHOD, anchors=anchors, costs=costs, entries=entries,
                subset_count=len(entries), catalog_seconds=time.perf_counter()-tick,
                exact_semantics='Input finite floats interpreted as exact binary rationals')


def select_from_catalog(catalog, budget, source_count=1):
    """Maximize worst logdet, then nominal midpoint det, cost, count, IDs.

    Total information for m identical-specification sources is I_m kron M,
    hence worst logdet = m*log(det M). Singular cases have score -infinity.
    The declared nominal secondary objective only chooses within the robust
    maximizer set; it does not change the primary objective. A singular robust
    optimum is explicitly flagged even when its nominal information is full rank.
    """
    if isinstance(source_count, bool) or not isinstance(source_count, numbers.Integral):
        raise ValueError('source_count must be a positive integer')
    source_count = operator.index(source_count)
    if source_count < 1:
        raise ValueError('source_count must be a positive integer')
    budget = _rational(budget, 'budget')
    if budget < 0:
        raise ValueError('budget must be nonnegative')
    tick = time.perf_counter()
    eligible = [row for row in catalog['entries'] if row['cost'] <= budget]
    if not eligible:
        raise ValueError('Complete catalog must include the empty zero-cost action')
    selected = min(eligible, key=lambda row:
                   (-row['determinant'], -row['nominal_determinant'],
                    row['cost'], len(row['ids']), row['ids']))
    det = selected['determinant']
    logdet = ('-Infinity' if det == 0 else
              source_count*(math.log(det.numerator)-math.log(det.denominator)))
    return dict(method=METHOD, paper=PAPER, doi=DOI,
                status='ok' if det > 0 else 'robust_degenerate',
                anchor_ids=selected['ids'], cost=selected['cost'], budget=budget,
                worst_determinant_per_source=det, worst_logdet_all_sources=logdet,
                nominal_determinant_per_source=selected['nominal_determinant'],
                robust_full_rank=det > 0, source_count=source_count,
                robust_degenerate=det == 0,
                tie_rule='max robust det, max midpoint nominal det, min cost, min count, lex IDs',
                selected_worst_values=selected['worst_values'],
                selected_inner_center=selected['center'],
                selected_information_entries=selected['information_entries'],
                feasible_subsets=len(eligible), catalog_subsets=catalog['subset_count'],
                selector_seconds=time.perf_counter()-tick,
                solver='exact rational subset enumeration; not original relaxation runtime')


def select_anchors(anchors, costs, budget, source_count=1, catalog=None):
    """Convenience API; a reused catalog must match the exact anchor/cost data."""
    anchors, costs = _model(anchors, costs)
    if catalog is None:
        catalog = build_catalog(anchors, costs)
    elif anchors != catalog['anchors'] or costs != catalog['costs']:
        raise ValueError('Reused catalog does not match these anchors and costs')
    return select_from_catalog(catalog, budget, source_count)
