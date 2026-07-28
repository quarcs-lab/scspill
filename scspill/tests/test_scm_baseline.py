"""Tests for the classical simplex-SCM comparator (scm_baseline.py)."""

import numpy as np
import pytest

from scspill.utils.scspill_helpers.scm_baseline import (
    classical_scm_counterfactual,
    classical_scm_weights,
)

# ---------------------------------------------------------------------------
# SIMPLEX CONSTRAINTS
# ---------------------------------------------------------------------------


def test_weights_satisfy_simplex():
    rng = np.random.default_rng(0)
    Yc = rng.standard_normal((30, 8)) + 5.0
    y0 = Yc @ np.array([0.2, 0.3, 0.5, 0, 0, 0, 0, 0]) + 0.01 * rng.standard_normal(30)
    w = classical_scm_weights(y0, Yc)
    assert w.shape == (8,)
    assert np.all(w >= 0)
    assert w.sum() == pytest.approx(1.0, abs=1e-8)


def test_recovers_convex_hull_truth():
    rng = np.random.default_rng(1)
    Yc = rng.standard_normal((60, 6)) * 3.0
    w_true = np.array([0.5, 0.3, 0.2, 0.0, 0.0, 0.0])
    y0 = Yc @ w_true
    w = classical_scm_weights(y0, Yc)
    assert np.allclose(w, w_true, atol=1e-4)


def test_ridge_stabilizes_collinear():
    rng = np.random.default_rng(2)
    base = rng.standard_normal(40)
    Yc = np.column_stack([base, base + 1e-9 * rng.standard_normal(40), rng.standard_normal(40)])
    y0 = base + 0.01 * rng.standard_normal(40)
    w = classical_scm_weights(y0, Yc, ridge=1e-8)
    assert np.all(np.isfinite(w))
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_counterfactual_path():
    rng = np.random.default_rng(3)
    Yc = rng.standard_normal((25, 5)) + 10.0
    w_true = np.array([0.4, 0.6, 0.0, 0.0, 0.0])
    y0 = Yc @ w_true
    _w, cf = classical_scm_counterfactual(y0, Yc, T0=20)
    assert cf.shape == (25,)
    assert np.allclose(cf, y0, atol=1e-3)
