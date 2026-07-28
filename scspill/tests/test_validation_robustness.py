"""Tests for prior sensitivity / prior predictive checks (scspill.validation)."""

import numpy as np
import pandas as pd
import pytest

from scspill.exceptions import ScspillConfigError
from scspill.utils.scspill_helpers.setup import prepare_scspill_inputs
from scspill.validation import (
    PosteriorSummary,
    PriorPredictiveResult,
    PriorSensitivityResult,
    plot_prior_predictive,
    ppc_stats,
    prior_predictive,
    prior_sensitivity,
    run_posterior_mcmc,
)
from scspill.validation.robustness import _kurtosis_type3, _skewness_type3

from .conftest import make_sar_panel

# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sar_pre():
    panel = make_sar_panel(N=9, T0=20, T1=2, rho=0.3, K=0, seed=31)
    truth = panel["truth"]
    return {
        "Yc": truth["Yc"][:20],
        "Y0": truth["Y0"][:20],
        "W": panel["spatial_W"].to_numpy(),
        "w": panel["spatial_w"].to_numpy(),
        "alpha": truth["alpha"],
        "rho": truth["rho"],
    }


@pytest.fixture(scope="module")
def ca_pre(california):
    inp = prepare_scspill_inputs(
        california.df,
        california.outcome,
        california.treat,
        california.unitid,
        california.time,
        california.spatial_W,
        california.spatial_w,
    )
    return inp


# ---------------------------------------------------------------------------
# PPC STATISTICS
# ---------------------------------------------------------------------------


def test_type3_moments_hand_checked():
    """e1071 type-3 skewness/kurtosis on a hand-computed sample."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    n = 5
    xc = x - x.mean()
    m2 = np.mean(xc**2)
    m3 = np.mean(xc**3)
    m4 = np.mean(xc**4)
    g1 = m3 / m2**1.5
    g2 = m4 / m2**2 - 3.0
    assert _skewness_type3(x) == pytest.approx(g1 * ((n - 1) / n) ** 1.5)
    assert _kurtosis_type3(x) == pytest.approx((g2 + 3.0) * (1 - 1 / n) ** 2 - 3.0)
    # e1071 parity at tiny n: a two-point sample is symmetric (skewness 0),
    # and only a zero-variance or single-point sample is undefined.
    assert _skewness_type3(np.array([1.0, 2.0])) == pytest.approx(0.0)
    assert np.isnan(_skewness_type3(np.array([1.0])))
    assert np.isnan(_kurtosis_type3(np.ones(10)))  # zero variance


def test_ac2_matches_r_at_two_periods():
    """R's rowSums over empty lag-2 slices give ac2 = 0 at T0 = 2."""
    Yc = np.array([[1.0, 2.0, 4.0], [3.0, 1.0, 2.0]])
    stats = ppc_stats(Yc, np.array([1.0, 2.0]), np.zeros((3, 3)), np.array([1.0, 0, 0]))
    assert stats["ac2"] == pytest.approx(0.0)


def test_ppc_stats_hand_checked():
    Yc = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 1.0], [3.0, 4.0, 2.0], [0.0, 1.0, 3.0]])
    Y0 = np.array([1.0, 2.0, 3.0, 4.0])
    Wn = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    wn = np.array([0.2, 0.3, 0.5])
    stats = ppc_stats(Yc, Y0, Wn, wn)
    assert stats["yc_mean"] == pytest.approx(Yc.mean())
    assert stats["log_yc_var"] == pytest.approx(np.log(Yc.ravel().var(ddof=1) + 1e-12))
    assert stats["spatial_quadratic"] == pytest.approx(np.trace(Yc @ Wn @ Yc.T) / 12.0)
    wyc = Yc @ wn
    assert stats["corr_y0_wyc"] == pytest.approx(np.corrcoef(Y0, wyc)[0, 1])
    # ac1: per-period cross-sectional demeaning, then per-unit lag products.
    Ydc = Yc.T - Yc.T.mean(axis=0, keepdims=True)
    num = np.sum(Ydc[:, 1:] * Ydc[:, :-1], axis=1)
    den = np.sum(Ydc * Ydc, axis=1)
    assert stats["ac1"] == pytest.approx(np.mean(num / den))
    # pve_pc1 from the column-centered covariance eigenvalues.
    Ycc = Yc - Yc.mean(axis=0, keepdims=True)
    eigs = np.sort(np.linalg.eigvalsh(Ycc.T @ Ycc))[::-1]
    assert stats["pve_pc1"] == pytest.approx(eigs[0] / eigs.sum())


def test_ppc_stats_frozen_california_contract(ca_pre):
    """Deterministic pin against the replication package's frozen table."""
    stats = ppc_stats(ca_pre.Yc[: ca_pre.T0], ca_pre.Y0[: ca_pre.T0], ca_pre.Wn, ca_pre.wn)
    frozen = {
        "yc_mean": 131.500,
        "log_yc_var": 6.984,
        "spatial_quadratic": 17844.720,
        "corr_y0_wyc": 0.945,
        "ac1": 0.894,
        "ac2": 0.810,
        "pve_pc1": 0.625,
        "avg_skewness": -0.396,
        "avg_kurtosis": -0.943,
    }
    for name, value in frozen.items():
        assert stats[name] == pytest.approx(value, abs=5e-3), name


# ---------------------------------------------------------------------------
# POSTERIOR RE-RUNS / PRIOR SENSITIVITY
# ---------------------------------------------------------------------------


def test_run_posterior_mcmc_recovers_rho(sar_pre):
    summary = run_posterior_mcmc(
        sar_pre["Yc"],
        sar_pre["W"],
        sar_pre["w"],
        sar_pre["alpha"],
        step_rho=0.1,
        m_burn=500,
        m_keep=1500,
        seed=3,
    )
    assert isinstance(summary, PosteriorSummary)
    row = summary.table[summary.table["parameter"] == "rho"].iloc[0]
    assert row["mean"] == pytest.approx(sar_pre["rho"], abs=0.15)
    assert row["q025"] <= sar_pre["rho"] <= row["q975"]
    assert summary.rho_draws.shape == (1500,)
    assert summary.meta["accept_rate"] > 0.02


def test_run_posterior_mcmc_production_kernel(sar_pre):
    summary = run_posterior_mcmc(
        sar_pre["Yc"],
        sar_pre["W"],
        sar_pre["w"],
        sar_pre["alpha"],
        step_rho=0.1,
        m_burn=300,
        m_keep=600,
        seed=4,
        kernel="production",
    )
    row = summary.table[summary.table["parameter"] == "rho"].iloc[0]
    assert row["mean"] == pytest.approx(sar_pre["rho"], abs=0.2)
    assert summary.meta["kernel"] == "production"


def test_run_posterior_mcmc_thinning_and_support(sar_pre):
    summary = run_posterior_mcmc(
        sar_pre["Yc"],
        sar_pre["W"],
        sar_pre["w"],
        sar_pre["alpha"],
        rho_support=(-0.2, 0.2),
        m_burn=100,
        m_keep=200,
        thin=3,
        seed=5,
    )
    assert summary.rho_draws.shape == (200,)
    assert np.all(np.abs(summary.rho_draws) <= 0.2)


def test_run_posterior_mcmc_unknown_kernel(sar_pre):
    with pytest.raises(ScspillConfigError):
        run_posterior_mcmc(
            sar_pre["Yc"], sar_pre["W"], sar_pre["w"], sar_pre["alpha"], kernel="bad"
        )


def test_prior_sensitivity_grid(sar_pre):
    grid = pd.DataFrame(
        {
            "a0": [3.0, 2.0],
            "b0": [1.0, 0.5],
            "rho_lo": [-0.5, -0.7],
            "rho_hi": [0.5, 0.7],
            "step_rho": [0.1, 0.1],
        }
    )
    res = prior_sensitivity(
        sar_pre["Yc"],
        sar_pre["W"],
        sar_pre["w"],
        sar_pre["alpha"],
        grid,
        m_burn=200,
        m_keep=400,
    )
    assert isinstance(res, PriorSensitivityResult)
    assert len(res.runs) == 2
    assert set(res.table["grid_row"]) == {0, 1}
    for col in ("a0", "b0", "rho_lo", "rho_hi", "step_rho", "parameter", "mean"):
        assert col in res.table.columns
    # Both rows recover rho within their supports.
    for run, (lo, hi) in zip(res.runs, [(-0.5, 0.5), (-0.7, 0.7)], strict=True):
        assert np.all((run.rho_draws >= lo) & (run.rho_draws <= hi))


def test_prior_sensitivity_missing_columns(sar_pre):
    with pytest.raises(ScspillConfigError, match="missing columns"):
        prior_sensitivity(
            sar_pre["Yc"],
            sar_pre["W"],
            sar_pre["w"],
            sar_pre["alpha"],
            pd.DataFrame({"a0": [1.0]}),
        )


# ---------------------------------------------------------------------------
# PRIOR PREDICTIVE
# ---------------------------------------------------------------------------


def test_prior_predictive_smoke_and_pvalues(sar_pre):
    res = prior_predictive(
        sar_pre["Y0"],
        sar_pre["W"],
        sar_pre["w"],
        sar_pre["alpha"],
        Yc_obs=sar_pre["Yc"],
        n_draws=300,
        seed=6,
    )
    assert isinstance(res, PriorPredictiveResult)
    assert res.stats.shape == (300, 9)
    assert res.observed is not None and res.p_values is not None
    for name, p in res.p_values.items():
        assert np.isnan(p) or 0.0 <= p <= 1.0, name
    # The DGP's data are consistent with the model class: the central stats
    # should not sit in the extreme tails.
    assert 0.001 < res.p_values["yc_mean"] < 0.999


def test_prior_predictive_without_observed(sar_pre):
    res = prior_predictive(
        sar_pre["Y0"], sar_pre["W"], sar_pre["w"], sar_pre["alpha"], n_draws=50, seed=7
    )
    assert res.observed is None and res.p_values is None


def test_prior_predictive_purity(sar_pre):
    """Different W arguments must change the results (no hidden globals)."""
    a = prior_predictive(
        sar_pre["Y0"], sar_pre["W"], sar_pre["w"], sar_pre["alpha"], n_draws=100, seed=8
    )
    W_other = np.ones_like(sar_pre["W"]) - np.eye(len(sar_pre["w"]))
    b = prior_predictive(
        sar_pre["Y0"], W_other, sar_pre["w"], sar_pre["alpha"], n_draws=100, seed=8
    )
    assert not np.allclose(
        a.stats["spatial_quadratic"].to_numpy(), b.stats["spatial_quadratic"].to_numpy()
    )


def test_pvalue_definition_hand_checked(sar_pre, monkeypatch):
    """p = mean(sim <= obs) over the finite simulated values."""
    res = prior_predictive(
        sar_pre["Y0"],
        sar_pre["W"],
        sar_pre["w"],
        sar_pre["alpha"],
        Yc_obs=sar_pre["Yc"],
        n_draws=200,
        seed=9,
    )
    sims = res.stats["yc_mean"].to_numpy()
    sims = sims[np.isfinite(sims)]
    expected = float(np.mean(sims <= res.observed["yc_mean"]))
    assert res.p_values["yc_mean"] == pytest.approx(expected)


def test_plot_prior_predictive_smoke(sar_pre, tmp_path):
    res = prior_predictive(
        sar_pre["Y0"],
        sar_pre["W"],
        sar_pre["w"],
        sar_pre["alpha"],
        Yc_obs=sar_pre["Yc"],
        n_draws=100,
        seed=10,
    )
    target = tmp_path / "ppc.png"
    fig = plot_prior_predictive(res, save=str(target))
    assert target.exists()
    import matplotlib.pyplot as plt

    plt.close(fig)
