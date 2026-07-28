"""Tests for MCMC diagnostics (diagnostics.py)."""

import numpy as np
import pytest

from scspill.utils.scspill_helpers.diagnostics import (
    ess_acf,
    geweke_z,
    mcmc_summary,
    mcse_from_ess,
    split_rhat,
)

# ---------------------------------------------------------------------------
# ESS
# ---------------------------------------------------------------------------


def test_ess_iid_near_n():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4000)
    ess = ess_acf(x)
    assert ess == pytest.approx(4000, rel=0.15)


def test_ess_ar1_much_smaller():
    rng = np.random.default_rng(1)
    n, phi = 4000, 0.9
    x = np.empty(n)
    x[0] = 0.0
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    ess = ess_acf(x)
    # Theoretical ESS factor for AR(1): (1-phi)/(1+phi) ~ 0.053 -> ~210.
    assert ess < n / 5
    assert ess > 20


def test_ess_short_chain():
    assert ess_acf(np.array([1.0, 2.0])) == 2.0


# ---------------------------------------------------------------------------
# SPLIT R-HAT
# ---------------------------------------------------------------------------


def test_split_rhat_stationary_near_one():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(4000)
    assert split_rhat(x) < 1.02


def test_split_rhat_mean_shift_flags():
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.standard_normal(1000), 3.0 + rng.standard_normal(1000)])
    assert split_rhat(x) > 1.1


def test_split_rhat_degenerate():
    assert np.isnan(split_rhat(np.array([1.0])))


# ---------------------------------------------------------------------------
# MCSE / GEWEKE
# ---------------------------------------------------------------------------


def test_mcse_iid():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(10000)
    m = mcse_from_ess(x, ess_acf(x))
    assert m == pytest.approx(1.0 / np.sqrt(10000), rel=0.3)


def test_geweke_z_stationary_small():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(4000)
    assert abs(geweke_z(x)) < 3.0


def test_geweke_z_drift_flags():
    x = np.linspace(0.0, 5.0, 2000) + np.random.default_rng(6).standard_normal(2000) * 0.1
    assert abs(geweke_z(x)) > 4.0


# ---------------------------------------------------------------------------
# SUMMARY TABLE
# ---------------------------------------------------------------------------


def test_mcmc_summary_layout():
    rng = np.random.default_rng(7)
    chains = {"rho": rng.standard_normal(500), "sigma2": np.abs(rng.standard_normal(500))}
    table = mcmc_summary(chains)
    assert list(table.index) == ["rho", "sigma2"]
    for col in ("mean", "sd", "q025", "q50", "q975", "ess", "rhat_split", "mcse", "geweke_z"):
        assert col in table.columns
    assert table.loc["rho", "q025"] <= table.loc["rho", "q50"] <= table.loc["rho", "q975"]
