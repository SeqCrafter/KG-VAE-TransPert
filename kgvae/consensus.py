"""MODZ-style replicate consensus on Level 3 landmark expression.

Reference: cmap/cmapM sig_tools/math/modzs.m, wt_avg with clip_low_cc=True,
low_thresh_cc=0, clip_low_wt=True, low_thresh_wt=0.01.
This is a documented parameterization, not a claim of exact WAVE preprocessing.
"""
import numpy as np
from scipy.stats import rankdata


def modz(profiles):
    """Return weighted expression [genes] and weights [replicates].

    Spearman similarities determine weights; weights apply to original expression,
    not ranks or z-scores. Constant profiles have zero similarity to other profiles.
    Two replicates receive equal weights; all uncorrelated profiles also fall back
    to equal weights through the positive minimum weight.
    """
    x = np.asarray(profiles, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or x.shape[1] < 2:
        raise ValueError("MODZ expects at least two replicates and two genes")
    if not np.isfinite(x).all():
        raise ValueError("MODZ inputs must be finite")
    ranks = rankdata(x, axis=1, method="average")
    ranks -= ranks.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(ranks, axis=1, keepdims=True)
    ranks = np.divide(ranks, norm, out=np.zeros_like(ranks), where=norm > 0)
    correlations = np.clip(ranks @ ranks.T, 0., 1.)
    np.fill_diagonal(correlations, 0.)
    weights = np.maximum(0.5 * correlations.sum(axis=1), 0.01)
    weights /= weights.sum()
    return (weights @ x).astype(np.float32), weights
