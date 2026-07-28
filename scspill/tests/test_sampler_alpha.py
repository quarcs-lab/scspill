"""Tests for the Step-1 horseshoe Gibbs sampler (sampler_alpha.py)."""

import numpy as np
import pytest

from scspill.exceptions import ScspillDataError
from scspill.utils.scspill_helpers.sampler_alpha import alpha_scaling, hs_alpha_gibbs
from scspill.utils.scspill_helpers.structures import AlphaPosterior

from .conftest import assert_chains_reproducible

# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


def _sparse_regression(T0=60, N=12, seed=0, noise=0.05):
    """y = 0.9 x_0 - 0.6 x_1 + eps: two dominant donors, the rest noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T0, N))
    y = 0.9 * X[:, 0] - 0.6 * X[:, 1] + noise * rng.standard_normal(T0)
    return y, X


# ---------------------------------------------------------------------------
# SAMPLER
# ---------------------------------------------------------------------------


def test_shapes_and_finiteness():
    y, X = _sparse_regression()
    post = hs_alpha_gibbs(np.random.default_rng(1), y, X, 400, 200)
    assert isinstance(post, AlphaPosterior)
    assert post.draws.shape == (200, 12)
    assert post.sigma2.shape == (200,)
    assert post.alpha_hat.shape == (12,)
    assert np.all(np.isfinite(post.draws))
    assert np.all(post.sigma2 > 0)
    assert post.iters == 400 and post.burn == 200


def test_seed_reproducibility():
    y, X = _sparse_regression()
    a = hs_alpha_gibbs(np.random.default_rng(42), y, X, 300, 100)
    b = hs_alpha_gibbs(np.random.default_rng(42), y, X, 300, 100)
    assert_chains_reproducible(a.draws, b.draws, atol=0.05)
    c = hs_alpha_gibbs(np.random.default_rng(43), y, X, 300, 100)
    assert not np.array_equal(a.draws, c.draws)


def test_sparse_signal_recovery():
    y, X = _sparse_regression(T0=80, seed=3)
    post = hs_alpha_gibbs(np.random.default_rng(5), y, X, 1500, 500)
    top2 = set(np.argsort(-np.abs(post.alpha_hat))[:2])
    assert top2 == {0, 1}
    assert post.alpha_hat[0] == pytest.approx(0.9, abs=0.1)
    assert post.alpha_hat[1] == pytest.approx(-0.6, abs=0.1)
    # Horseshoe shrinks the noise donors hard.
    assert np.max(np.abs(post.alpha_hat[2:])) < 0.1


def test_back_transform_matches_scaling():
    """Draws are (sy/sx_j) times the standardized-scale draws."""
    y, X = _sparse_regression(T0=40, N=5, seed=9)
    scale = np.array([1.0, 10.0, 0.1, 5.0, 2.0])
    X_scaled = X * scale
    a = hs_alpha_gibbs(np.random.default_rng(7), y, X, 300, 100)
    b = hs_alpha_gibbs(np.random.default_rng(7), y, X_scaled, 300, 100)
    # Same standardized problem -> back-transformed draws differ by 1/scale
    # (up to platform BLAS rounding; see assert_chains_reproducible).
    assert_chains_reproducible(a.draws, b.draws * scale, atol=0.05)
    sy, sx = alpha_scaling(y, X)
    assert sy == pytest.approx(np.std(y, ddof=1))
    assert np.allclose(sx, X.std(axis=0, ddof=1))


def test_perfect_fit_recovers_exact_alpha():
    """When y is exactly in the span of X, alpha concentrates on the truth."""
    rng = np.random.default_rng(11)
    X = rng.standard_normal((50, 8))
    alpha_true = np.array([0.5, -0.2, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
    y = X @ alpha_true
    post = hs_alpha_gibbs(np.random.default_rng(2), y, X, 800, 400)
    assert np.allclose(post.alpha_hat, alpha_true, atol=1e-3)


# ---------------------------------------------------------------------------
# FAILURE MODES
# ---------------------------------------------------------------------------


def test_input_validation():
    y, X = _sparse_regression()
    with pytest.raises(ScspillDataError):
        hs_alpha_gibbs(np.random.default_rng(0), y[:5], X, 100, 50)  # length mismatch
    with pytest.raises(ScspillDataError):
        hs_alpha_gibbs(np.random.default_rng(0), y[:2], X[:2], 100, 50)  # too short
    y_bad = y.copy()
    y_bad[0] = np.nan
    with pytest.raises(ScspillDataError):
        hs_alpha_gibbs(np.random.default_rng(0), y_bad, X, 100, 50)
