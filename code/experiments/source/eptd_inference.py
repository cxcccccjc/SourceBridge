from release_integrity import verify_identity, resolve_resource
"""Source-traced extra literature inference cores (not full cryptographic systems).

EPTD: Zhang et al., IEEE TIFS 2022, DOI 10.1109/TIFS.2022.3207905.
PDF page 6, discussion following Eq. (13): I-CRH streaming extension.
PDF page 4 Eq. (3): weighted arithmetic mean; page 6 Eq. (13): decayed
sum of squared residuals; immediately following text: log(sum D / D_k).
PDF page 9 accuracy setup: decay=0.5.

This implementation does not claim that EPTD invented I-CRH. EPTD explicitly
attributes that framework to Li et al. (CRH). It also does not implement the
encryption, task matching, initialization authority, or privacy guarantees.

Declared adapters: a time point is a one-task chunk; all sources participate;
initial weights are uniform (streaming discussion does not specify a numeric
initial weight); a 1e-10 distance floor handles identical reports. For group
inputs, the averaging into groups occurs outside this file and is our adapter.
"""

import numpy as np


class EPTDICRH:
    """Incremental real-arithmetic inference supported in published EPTD."""

    def __init__(self, n_sources, decay=0.5, floor=1e-10):
        if n_sources < 2:
            raise ValueError("I-CRH logarithmic weights require at least 2 sources")
        if not 0 <= decay <= 1 or floor <= 0:
            raise ValueError("decay must be in [0,1] and floor positive")
        self.decay = float(decay)
        self.floor = float(floor)
        self.acc = np.zeros(n_sources, dtype=float)
        self.w = np.ones(n_sources, dtype=float)

    def weight(self):
        return self.w / self.w.sum()

    def update(self, x, estimate):
        x = np.asarray(x, dtype=float)
        if x.shape != self.acc.shape or not np.isfinite(x).all():
            raise ValueError("one finite report per source required")
        self.acc = self.decay * self.acc + (x - estimate) ** 2
        distance = np.maximum(self.acc, self.floor)
        self.w = np.maximum(np.log(distance.sum() / distance), self.floor)

    def step(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != self.acc.shape or not np.isfinite(x).all():
            raise ValueError("one finite report per source required")
        estimate = float(x @ self.weight())
        self.update(x, estimate)
        return estimate


def eptd_icrh(reports, decay=0.5, floor=1e-10):
    """reports has shape (time, source); current truth uses previous weights."""
    reports = np.asarray(reports, dtype=float)
    if reports.ndim != 2 or reports.shape[1] < 2:
        raise ValueError("reports must have shape (time, at least 2 sources)")
    method = EPTDICRH(reports.shape[1], decay=decay, floor=floor)
    return np.asarray([method.step(row) for row in reports])


def eptd_crh(reports, max_iterations=100, tolerance=1e-6, floor=1e-10):
    """EPTD main squared-loss CRH core for a simultaneous tasks-by-sources batch.

    EPTD starts from random task truths (PDF page 5). We expose a deterministic
    mean initialization for controlled comparisons; this is an explicit adapter.
    This is NOT the absolute-loss default in TKDE2016's empirical evaluation.
    EPTD itself explicitly uses squared loss (Eq. 9) and sum normalization.
    """
    reports = np.asarray(reports, dtype=float)
    if reports.ndim != 2 or reports.shape[1] < 2 or not np.isfinite(reports).all():
        raise ValueError("finite tasks-by-sources matrix with at least 2 sources required")
    truth = reports.mean(axis=1)
    for iteration in range(max_iterations):
        distance = np.maximum(((reports - truth[:, None]) ** 2).sum(axis=0), floor)
        weights = np.maximum(np.log(distance.sum() / distance), floor)
        next_truth = reports @ (weights / weights.sum())
        change = float(np.max(abs(next_truth - truth)))
        truth = next_truth
        if change < tolerance:
            return truth, {"iterations": iteration + 1, "converged": True}
    return truth, {"iterations": max_iterations, "converged": False}
