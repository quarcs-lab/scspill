"""Vectorized treatment-effect primitives.

Bite-sized, pure functions for the *effect* side of estimator reporting --
ATT, percent ATT, total effect, and the per-period gap -- kept separate from
the goodness-of-fit primitives in :mod:`scspill.utils.fitutils`. Each is a
vectorized operation returning raw (unrounded) values; callers round only for
display. Copied from ``mlsynth.utils.effectutils`` so effect metrics are
computed identically across both libraries.
"""

from __future__ import annotations

import numpy as np


def _ravel(x: np.ndarray) -> np.ndarray:
    """Flatten to a 1-D float array (the common input shape)."""
    return np.asarray(x, dtype=float).ravel()


def gap(observed: np.ndarray, counterfactual: np.ndarray) -> np.ndarray:
    """Per-period treatment effect ``observed - counterfactual``."""
    return _ravel(observed) - _ravel(counterfactual)


def split_pre_post(arr: np.ndarray, n_pre: int, n_post: int) -> tuple[np.ndarray, np.ndarray]:
    """Slice an array into its pre- and post-treatment segments."""
    a = _ravel(arr)
    return a[:n_pre], a[n_pre : n_pre + n_post]


def att(post_gap: np.ndarray) -> float:
    """Average treatment effect on the treated: mean post-period gap."""
    g = _ravel(post_gap)
    return float(g.mean()) if g.size else float("nan")


def total_effect(post_gap: np.ndarray) -> float:
    """Total treatment effect: summed post-period gap."""
    g = _ravel(post_gap)
    return float(g.sum()) if g.size else float("nan")


def percent_att(att_value: float, counterfactual_post: np.ndarray) -> float:
    """ATT as a percent of the mean post-period counterfactual."""
    cf = _ravel(counterfactual_post)
    mean_cf = float(cf.mean()) if cf.size else 0.0
    return float(100.0 * att_value / mean_cf) if mean_cf != 0 else float("nan")


def percent_gap(post_gap: np.ndarray, counterfactual_post: np.ndarray) -> np.ndarray:
    """Per-period percent effect; ``nan`` where the counterfactual is zero."""
    g = _ravel(post_gap)
    cf = _ravel(counterfactual_post)
    out = np.full_like(g, np.nan)
    nonzero = cf != 0
    out[nonzero] = 100.0 * g[nonzero] / cf[nonzero]
    return out
