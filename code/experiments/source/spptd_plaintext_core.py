"""Independent floating-point reconstruction of the S-PPTD numerical core.

Sources: S-PPTD, IEEE TDSC 2025, Eqs. (3)-(5), (17)-(22).
This is not author code, the SenSys original implementation, or a private
two-server/CORDIC implementation. Ambiguous scheduling and undefined-value
handling are explicit caller choices; see spptd_plaintext_reproduction_spec.md.
Reports use shape (sources, objects). Relations are supplied without labels.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
from release_integrity import verify_identity, resolve_resource
import sys
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent / 'deps'))
import numpy as np

NONE, TEMPORAL, SPATIAL = 0, 1, 2


class UndefinedComputation(RuntimeError):
    """The stated equations have no finite defined result for this input."""


@dataclass(frozen=True)
class PlaintextConfig:
    # These mandatory settings are not claimed to be author-selected defaults.
    alpha: float
    temporal_degree: int
    spatial_variance: float
    schedule: str
    max_iterations: int
    absolute_tolerance: float
    relative_tolerance: float = 0.0
    zero_residual_policy: str = 'strict'
    residual_floor: Optional[float] = None
    empty_source_policy: str = 'error'

    def validate(self):
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError('alpha must be finite and nonnegative')
        if not isinstance(self.temporal_degree, int) or isinstance(self.temporal_degree, bool) or self.temporal_degree < 1:
            raise ValueError('temporal_degree must be a positive integer')
        if not np.isfinite(self.spatial_variance) or self.spatial_variance <= 0:
            raise ValueError('spatial_variance means delta squared and must be positive')
        if self.schedule not in ('synchronous', 'grouped'):
            raise ValueError('Explicitly choose synchronous or grouped scheduling')
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError('max_iterations must be a positive integer')
        for name in ('absolute_tolerance', 'relative_tolerance'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(name + ' must be finite and nonnegative')
        if self.zero_residual_policy not in ('strict', 'floor'):
            raise ValueError('zero_residual_policy must be strict or floor')
        if self.zero_residual_policy == 'floor':
            if self.residual_floor is None or not np.isfinite(self.residual_floor) or self.residual_floor <= 0:
                raise ValueError('floor policy requires an explicit positive squared-residual floor')
        elif self.residual_floor is not None:
            raise ValueError('A residual_floor has no effect under strict policy; do not supply one')
        if self.empty_source_policy not in ('error', 'exclude'):
            raise ValueError('empty_source_policy must be error or exclude')


def normalize_by_public_max(reports, observed, public_max):
    """Scale only; caller must supply and record a genuinely public bound.

    No per-test normalization, centering, clipping, or ground-truth access is
    performed. The caller must apply the same declared scale to all methods.
    """
    x = np.asarray(reports, dtype=float)
    mask = np.asarray(observed)
    if mask.dtype != np.bool_ or mask.shape != x.shape or x.ndim != 2:
        raise ValueError('observed must be a boolean mask matching a two-dimensional report matrix')
    if not np.isfinite(public_max) or public_max <= 0:
        raise ValueError('public_max must be finite and positive')
    values = x[mask]
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values >= public_max):
        raise ValueError('Observed values must lie in [0, public_max); no silent clipping')
    return np.where(mask, x / public_max, np.nan)


def independent_groups(relations, order=None):
    """Deterministic greedy graph coloring; our optional convention, not author code.

    Relations may be directed. Either edge direction forbids two objects from
    sharing a group. An explicit object order fixes this otherwise arbitrary
    coloring and may be changed only as a separately recorded sensitivity.
    """
    rel = np.asarray(relations)
    if rel.ndim != 2 or rel.shape[0] != rel.shape[1] or not np.isin(rel, [NONE, TEMPORAL, SPATIAL]).all() or np.any(np.diag(rel)):
        raise ValueError('Relations must be a square 0/1/2 matrix with zero diagonal')
    m = len(rel)
    order = list(range(m)) if order is None else list(order)
    if sorted(order) != list(range(m)):
        raise ValueError('order must contain every object index exactly once')
    result = []
    for j in order:
        for group in result:
            if not any(rel[j, k] or rel[k, j] for k in group):
                group.append(j)
                break
        else:
            result.append([j])
    return result


def _validate_groups(rel, groups, schedule):
    m = len(rel)
    if schedule == 'synchronous':
        if groups is not None:
            raise ValueError('groups are unused in synchronous mode; pass None explicitly')
        return [list(range(m))]
    if groups is None:
        raise ValueError('grouped mode requires an explicit ordered partition')
    result = [list(group) for group in groups]
    if any(not group for group in result) or sorted(j for group in result for j in group) != list(range(m)):
        raise ValueError('groups must form a nonempty, disjoint partition of all object indices')
    for group in result:
        if np.any(rel[np.ix_(group, group)]):
            raise ValueError('Objects within a group must have no relation edges')
    return result


def _prepare(reports, observed, relations, config, groups):
    config.validate()
    x = np.asarray(reports, dtype=float)
    mask = np.asarray(observed)
    if x.ndim != 2 or min(x.shape) < 1 or mask.dtype != np.bool_ or mask.shape != x.shape:
        raise ValueError('Need a nonempty sources-by-objects array and matching boolean observed mask')
    if not np.isfinite(x[mask]).all() or np.any(x[mask] < 0) or np.any(x[mask] >= 1):
        raise ValueError('Core expects observed reports in [0,1); normalize externally using a declared public scale')
    rel = np.asarray(relations)
    if rel.shape != (x.shape[1], x.shape[1]) or not np.isin(rel, [NONE, TEMPORAL, SPATIAL]).all() or np.any(np.diag(rel)):
        raise ValueError('One 0/1/2 relation per ordered object pair is required, with no self edges')
    ordered_groups = _validate_groups(rel, groups, config.schedule)
    missing_objects = np.flatnonzero(~mask.any(axis=0))
    if missing_objects.size:
        raise UndefinedComputation('Eq. (3) has no observed data for objects ' + str(missing_objects.tolist()))
    active = np.flatnonzero(mask.any(axis=1))
    excluded = np.flatnonzero(~mask.any(axis=1))
    if excluded.size and config.empty_source_policy == 'error':
        raise UndefinedComputation('Eq. (5) has zero distance for observationless sources ' + str(excluded.tolist()))
    if len(active) < 2:
        raise UndefinedComputation('Fewer than two nonempty sources: log(sum(D)/D) cannot supply positive next weights')
    return np.where(mask[active], x[active], 0.0), mask[active], rel.astype(np.int8), ordered_groups, active, excluded


def _kernel(j, neighbors, state, relations, config):
    kinds = relations[j, neighbors]
    result = np.empty(len(neighbors), dtype=float)
    temporal = kinds == TEMPORAL
    spatial = kinds == SPATIAL
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        result[temporal] = (state[j] * state[neighbors[temporal]]) ** config.temporal_degree
        result[spatial] = np.exp(-((state[j] - state[neighbors[spatial]]) ** 2) / (2 * config.spatial_variance))
    if not np.isfinite(result).all() or np.any(result < 0):
        raise UndefinedComputation('Nonfinite or negative kernel values')
    return result


def _step(x, mask, weights, relations, groups, config):
    # Eq. (3)/(16): all object estimates use the same old source weights.
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(x),) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('Weights must be finite nonnegative values, one per active source')
    ws = weights @ x
    sw = weights @ mask.astype(float)
    if not np.isfinite(ws).all() or not np.isfinite(sw).all() or np.any(sw <= 0):
        raise UndefinedComputation('Eq. (3) has a nonpositive or nonfinite observed weight sum')
    estimated = ws / sw
    corrected = estimated.copy()
    correction_denominators = sw.copy()
    kernel_masses = np.zeros(len(estimated), dtype=float)
    # Our grouped convention: snapshot each group, then commit all its objects.
    # The synchronous convention is one all-object snapshot. Numerator data terms
    # WS/SW always remain fixed at the current iteration's Eq. (3) values.
    if config.alpha:
        for group in groups:
            state = corrected.copy()
            updates = []
            for j in group:
                neighbors = np.flatnonzero(relations[j])
                if not neighbors.size:
                    updates.append((j, estimated[j]))
                    continue
                kernel = _kernel(j, neighbors, state, relations, config)
                kernel_masses[j] = float(np.sum(kernel))
                denominator = sw[j] + config.alpha * kernel_masses[j]
                numerator = ws[j] + config.alpha * float(kernel @ state[neighbors])
                if not np.isfinite(denominator) or denominator <= 0 or not np.isfinite(numerator):
                    raise UndefinedComputation('Eq. (4) has an invalid corrected numerator or denominator')
                correction_denominators[j] = denominator
                updates.append((j, numerator / denominator))
            for j, value in updates:
                corrected[j] = value
    if not np.isfinite(corrected).all():
        raise UndefinedComputation('Nonfinite corrected truth')
    # Eq. (5)/(21)/(22): sum squared residuals only over observed entries.
    distances = np.sum(np.where(mask, (x - corrected) ** 2, 0.0), axis=1)
    if not np.isfinite(distances).all():
        raise UndefinedComputation('Nonfinite squared residual distance')
    if config.zero_residual_policy == 'strict':
        if np.any(distances <= 0):
            raise UndefinedComputation('Eq. (5) log distance is undefined for zero-residual source(s)')
        effective = distances
        regularized = np.zeros(len(distances), dtype=bool)
    else:
        regularized = distances < config.residual_floor
        effective = np.maximum(distances, config.residual_floor)
    total = float(np.sum(effective))
    if not np.isfinite(total) or total <= 0:
        raise UndefinedComputation('Eq. (5) has a nonpositive/nonfinite total distance')
    next_weights = np.log(total) - np.log(effective)
    if not np.isfinite(next_weights).all() or np.any(next_weights < 0):
        raise UndefinedComputation('Nonfinite or negative source weight')
    return dict(estimated_truth=estimated, corrected_truth=corrected,
                weights_used=weights.copy(), weights_next=next_weights,
                source_squared_residuals=distances, effective_squared_residuals=effective,
                regularized_source_count=int(regularized.sum()), zero_weight_source_count=int(np.sum(next_weights == 0)),
                observed_weight_sums=sw, correction_denominators=correction_denominators,
                kernel_masses=kernel_masses)


def plaintext_iteration(reports, observed, relations, weights, *, config, groups=None):
    """One formula iteration, useful for independent numeric verification.

    weights must refer to nonempty sources when exclusion is explicitly used.
    Undefined equations raise UndefinedComputation; nothing is silently filled.
    """
    x, mask, rel, ordered_groups, active, excluded = _prepare(reports, observed, relations, config, groups)
    try:
        result = _step(x, mask, weights, rel, ordered_groups, config)
    except FloatingPointError as exc:
        raise UndefinedComputation(str(exc)) from exc
    result.update(active_source_indices=active, excluded_source_indices=excluded)
    return result


def run_spptd_plaintext(reports, observed, relations, *, config, groups=None):
    """Run explicit scheduling/stopping conventions and retain a full trace.

    status='undefined' exposes no purported truth result. last_valid_truth is
    diagnostic only, not a fallback prediction. max_iterations is not convergence.
    Source weights are never normalized: doing so changes Eq. (4) relative to alpha.
    """
    config.validate()
    history = []
    try:
        x, mask, rel, ordered_groups, active, excluded = _prepare(reports, observed, relations, config, groups)
    except UndefinedComputation as exc:
        return dict(status='undefined', truth=None, weights=None, last_valid_truth=None,
                    iterations=0, failed_iteration=0, reason=str(exc), history=history,
                    config=asdict(config), scope='independent plaintext core; no cryptography')
    weights = np.ones(len(active))
    previous = None
    status = 'max_iterations'
    for k in range(1, config.max_iterations + 1):
        try:
            row = _step(x, mask, weights, rel, ordered_groups, config)
        except (UndefinedComputation, FloatingPointError) as exc:
            return dict(status='undefined', truth=None, weights=None, last_valid_truth=previous,
                        iterations=len(history), failed_iteration=k, reason=str(exc), history=history,
                        active_source_indices=active, excluded_source_indices=excluded,
                        groups=ordered_groups, config=asdict(config), scope='independent plaintext core; no cryptography')
        current = row['corrected_truth']
        change = None if previous is None else float(np.sum(abs(current - previous)))
        threshold = None if previous is None else config.absolute_tolerance + config.relative_tolerance * float(np.sum(abs(previous)))
        row.update(iteration=k, l1_change=change, stopping_threshold=threshold)
        history.append(row)
        weights = row['weights_next']
        previous = current.copy()
        if change is not None and change <= threshold:
            status = 'converged'
            break
    full_weights = np.zeros(np.asarray(reports).shape[0])
    full_weights[active] = weights
    return dict(status=status, truth=previous, weights=full_weights, last_valid_truth=previous.copy(),
                iterations=len(history), failed_iteration=None, reason=None, history=history,
                active_source_indices=active, excluded_source_indices=excluded,
                groups=ordered_groups, config=asdict(config), scope='independent plaintext core; no cryptography')
