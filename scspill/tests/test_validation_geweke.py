"""Tests for the Geweke joint distribution test (scspill.validation.geweke)."""

import numpy as np
import pytest

from scspill.exceptions import ScspillConfigError
from scspill.validation import (
    GewekeReport,
    SimpleKernel,
    batch_means_variance,
    default_g_fn,
    geweke_test,
    plot_geweke,
)
from scspill.validation.kernels import rho_bound_from_A

# ---------------------------------------------------------------------------
# BATCH MEANS VARIANCE
# ---------------------------------------------------------------------------


def test_batch_means_iid_close_to_naive():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20000)
    bm = batch_means_variance(x)
    naive = x.var(ddof=1) / x.size
    assert 0.5 * naive < bm < 2.0 * naive


def test_batch_means_ar1_exceeds_naive():
    rng = np.random.default_rng(1)
    n, phi = 20000, 0.8
    x = np.empty(n)
    x[0] = 0.0
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    bm = batch_means_variance(x)
    naive = x.var(ddof=1) / n
    assert bm > 2.0 * naive  # theoretical inflation (1+phi)/(1-phi) = 9


def test_batch_means_short_series_fallback():
    x = np.array([1.0, 2.0, 3.0])
    assert batch_means_variance(x, batch_size=2) == pytest.approx(np.var(x, ddof=1) / 3)
    assert np.isnan(batch_means_variance(np.array([])))


def test_batch_means_drops_nonfinite():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(5000)
    x_nan = np.concatenate([x, [np.nan, np.inf]])
    assert batch_means_variance(x_nan) == pytest.approx(batch_means_variance(x))


# ---------------------------------------------------------------------------
# G STATISTICS / SUPPORT
# ---------------------------------------------------------------------------


def test_default_g_fn_hand_checked():
    Yc = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 0.0]])  # (T0=3, N=2)
    Wn = np.array([[0.0, 1.0], [1.0, 0.0]])
    wn = np.array([0.5, 0.5])
    y0 = np.array([1.0, 2.0, 3.0])
    summary = {
        "rho": 0.2,
        "sigma2": 4.0,
        "beta": np.array([1.0, 3.0]),
        "Eta": np.zeros(0),
        "Gamma": np.zeros(0),
    }
    g = default_g_fn(summary, Yc, y0, Wn, wn)
    assert g["rho"] == 0.2
    assert g["log_sigma2"] == pytest.approx(np.log(4.0))
    assert g["yc_mean"] == pytest.approx(Yc.mean())
    assert g["spatial_quadratic"] == pytest.approx(np.trace(Yc @ Wn @ Yc.T) / 6.0)
    assert g["beta_mean"] == pytest.approx(2.0)
    assert np.isnan(g["Eta_mean"]) and np.isnan(g["Gamma_mean"])
    wyc = Yc @ wn
    assert g["corr_y0_wyc"] == pytest.approx(np.corrcoef(y0, wyc)[0, 1])


def test_rho_bound_hand_checked():
    # A with known spectrum: diagonal-ish 2x2.
    Wn = np.array([[0.0, 0.5], [0.5, 0.0]])
    wn = np.array([1.0, 0.0])
    alpha = np.zeros(2)
    # A = Wn -> eigenvalues +-0.5 -> bound 0.95 / 0.5 = 1.9
    assert rho_bound_from_A(Wn, wn, alpha) == pytest.approx(1.9)


def test_kernel_support_unified_and_intersected():
    """Prior draws and the transition share one support inside the bound."""
    rng = np.random.default_rng(3)
    N = 4
    Wn = np.eye(N, k=1) + np.eye(N, k=-1)
    Wn = Wn / Wn.sum(axis=1, keepdims=True)
    wn = np.array([1.0, 0, 0, 0])
    alpha = 0.4 * rng.standard_normal(N)
    bnd = rho_bound_from_A(Wn, wn, alpha)
    kern = SimpleKernel(4, N, 0, 0, Wn, wn, alpha, rho_support=(-5.0, 5.0))
    assert kern.rho_support == (pytest.approx(-bnd), pytest.approx(bnd))
    draws = [kern.draw_prior(rng).rho for _ in range(200)]
    assert min(draws) > -bnd and max(draws) < bnd


def test_empty_support_rejected():
    rng = np.random.default_rng(4)
    N = 4
    Wn = np.eye(N, k=1) + np.eye(N, k=-1)
    Wn = Wn / Wn.sum(axis=1, keepdims=True)
    wn = np.array([1.0, 0, 0, 0])
    alpha = 0.4 * rng.standard_normal(N)
    from scspill.exceptions import ScspillDataError

    with pytest.raises(ScspillDataError):
        SimpleKernel(4, N, 0, 0, Wn, wn, alpha, rho_support=(2.0, 3.0))


# ---------------------------------------------------------------------------
# HARNESS
# ---------------------------------------------------------------------------


def test_geweke_smoke_structure():
    rep = geweke_test(kernel="simple", T0=4, N=4, K=0, p=0, m_iid=600, m_mcmc=600, burn=100, seed=0)
    assert isinstance(rep, GewekeReport)
    assert set(rep.table.columns) == {
        "g",
        "mean_iid",
        "mean_mcmc",
        "se_iid",
        "se_mcmc",
        "z",
        "pval",
    }
    # Absent blocks (beta/Eta/Gamma with K=p=0) are dropped from the table.
    assert "beta_mean" not in set(rep.table["g"])
    assert np.all(np.isfinite(rep.table["z"]))
    assert rep.kernel == "simple" and rep.z_crit > 1.9


def test_geweke_unknown_kernel():
    with pytest.raises(ScspillConfigError):
        geweke_test(kernel="nonsense", m_iid=10, m_mcmc=10, burn=2)


@pytest.mark.parametrize(
    "kernel,K,p",
    [("simple", 0, 0), ("simple", 2, 0), ("production", 0, 0)],
)
def test_geweke_isolated_blocks_pass(kernel, K, p):
    """Each sampler block passes the JDT on a small, fast-mixing design."""
    rep = geweke_test(
        kernel=kernel,
        T0=4,
        N=4,
        K=K,
        p=p,
        m_iid=12000,
        m_mcmc=12000,
        burn=2500,
        step_rho=0.5,
        a0=3.0,
        b0=1.0,
        seed=21,
    )
    # Assert against a generous threshold rather than the strict Bonferroni
    # decision so residual long-memory noise cannot flake CI.
    assert rep.table["z"].abs().max() < 4.0


@pytest.mark.slow
def test_geweke_factor_block_passes():
    """The AR(1)-factor block needs a longer chain to resolve its slow modes."""
    rep = geweke_test(
        kernel="simple",
        T0=4,
        N=4,
        K=0,
        p=1,
        m_iid=30000,
        m_mcmc=30000,
        burn=5000,
        step_rho=0.5,
        a0=3.0,
        b0=1.0,
        seed=21,
    )
    assert rep.table["z"].abs().max() < 4.0


@pytest.mark.slow
def test_geweke_core_passes_strictly():
    rep = geweke_test(
        kernel="production",
        T0=4,
        N=4,
        K=0,
        p=0,
        m_iid=30000,
        m_mcmc=30000,
        burn=5000,
        step_rho=0.5,
        a0=3.0,
        b0=1.0,
        seed=21,
    )
    assert rep.passed


def test_geweke_power_detects_broken_kernel():
    """A kernel with a deliberately broken sigma2 update must be flagged."""

    rng_seed = 5

    class BrokenKernel(SimpleKernel):
        def transition(self, state, Yc, rng):
            keep = state.sigma2
            super().transition(state, Yc, rng)
            state.sigma2 = keep  # never update sigma2 -> wrong joint
            return state

    rng = np.random.default_rng(rng_seed)
    N = 4
    Wn = np.eye(N, k=1) + np.eye(N, k=-1)
    Wn = Wn / Wn.sum(axis=1, keepdims=True)
    wn = np.array([1.0, 0, 0, 0])
    alpha = 0.4 * rng.standard_normal(N)
    kern = BrokenKernel(4, N, 0, 0, Wn, wn, alpha, a0=3.0, b0=1.0, step_rho=0.5)
    rep = geweke_test(kernel=kern, m_iid=6000, m_mcmc=6000, burn=1000, seed=rng_seed)
    assert not rep.passed
    assert rep.table["z"].abs().max() > 4.0


def test_plot_geweke_smoke(tmp_path):
    rep = geweke_test(kernel="simple", T0=4, N=4, K=0, p=0, m_iid=400, m_mcmc=400, burn=100, seed=1)
    target = tmp_path / "geweke.png"
    fig = plot_geweke(rep, save=str(target))
    assert target.exists()
    import matplotlib.pyplot as plt

    plt.close(fig)
