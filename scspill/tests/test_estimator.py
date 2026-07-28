"""End-to-end tests for the SCSPILL estimator."""

import numpy as np
import pytest
from pydantic import ValidationError

from scspill import SCSPILL, SCSPILLResults
from scspill.exceptions import ScspillConfigError, ScspillDataError

from .conftest import assert_chains_reproducible, base_config_kwargs

# ---------------------------------------------------------------------------
# ESTIMATOR (synthetic planted-truth panel)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted(sar_panel):
    return SCSPILL(base_config_kwargs(sar_panel, m_iter=1200, burn=600)).fit()


def test_returns_results(fitted):
    assert isinstance(fitted, SCSPILLResults)


def test_rho_recovery(fitted, sar_panel):
    truth = sar_panel["truth"]
    assert fitted.rho_hat == pytest.approx(truth["rho"], abs=0.1)
    lo, hi = fitted.rho_ci
    assert lo <= truth["rho"] <= hi
    assert lo < hi


def test_att_covers_realized_truth(fitted, sar_panel):
    true_att = float(sar_panel["truth"]["taus"].mean())
    lo, hi = fitted.att_ci
    # Perfect-fit DGP: credible interval is tight around the realized truth.
    assert abs(fitted.att - true_att) < 0.15
    assert lo <= fitted.att <= hi


def test_alpha_recovery(fitted, sar_panel):
    truth = sar_panel["truth"]
    assert np.allclose(fitted.alpha_hat, truth["alpha"], atol=0.05)


def test_flat_accessor_surface(fitted, sar_panel):
    T = sar_panel["truth"]["Y0"].shape[0]
    assert isinstance(fitted.att, float)
    assert fitted.counterfactual.shape == (T,)
    assert fitted.gap.shape == (T,)
    assert set(fitted.donor_weights) == {str(u) for u in sar_panel["units"]}
    assert fitted.pre_rmse >= 0
    ts = fitted.time_series
    assert ts.has_prediction_interval
    assert ts.prediction_interval_kind == "bayesian"
    assert ts.intervention_time == sar_panel["truth"]["T0"]


def test_spillover_panel_labels(fitted, sar_panel):
    panel = fitted.spillover_panel
    assert list(panel.columns) == sorted(sar_panel["units"])
    assert panel.shape[0] == sar_panel["truth"]["Y0"].shape[0]
    assert (fitted.spillover_lower.to_numpy() <= fitted.spillover_upper.to_numpy()).all()


def test_seed_reproducibility(sar_panel):
    a = SCSPILL(base_config_kwargs(sar_panel, m_iter=400, burn=200, seed=5)).fit()
    b = SCSPILL(base_config_kwargs(sar_panel, m_iter=400, burn=200, seed=5)).fit()
    assert_chains_reproducible(a.rho_draws, b.rho_draws, atol=0.05)
    assert_chains_reproducible(a.alpha_draws, b.alpha_draws, atol=0.05)
    assert a.att == pytest.approx(b.att, abs=0.05)
    # A different seed must give genuinely different draws.
    c = SCSPILL(base_config_kwargs(sar_panel, m_iter=400, burn=200, seed=6)).fit()
    assert not np.array_equal(a.rho_draws, c.rho_draws)


def test_diagnostics_method(fitted):
    table = fitted.diagnostics(top_n_alpha=3)
    assert "rho" in table.index and "sigma2" in table.index
    assert sum(name.startswith("alpha[") for name in table.index) == 3
    with pytest.raises(KeyError):
        fitted.diagnostics(which_alpha=["not-a-unit"])


def test_results_frozen(fitted):
    with pytest.raises(ValidationError):  # pydantic frozen model
        fitted.scm_weights = {}
    assert not fitted.rho_draws.flags.writeable
    assert not fitted.alpha_draws.flags.writeable


def test_mcmc_summary_table_precomputed(fitted):
    table = fitted.mcmc_summary_table
    assert table is not None
    assert "rho" in table.index


# ---------------------------------------------------------------------------
# ESTIMATOR (real data smoke)
# ---------------------------------------------------------------------------


def test_california_smoke(california):
    res = SCSPILL(
        {
            **california.config_kwargs(),
            "m_iter": 400,
            "burn": 200,
            "seed": 20251022,
            "display_graphs": False,
            "p_factors": 1,
            "step_rho": 0.05,
        }
    ).fit()
    assert np.isfinite(res.att)
    assert -0.95 < res.rho_hat < 0.95
    assert res.counterfactual.shape == (31,)
    assert set(res.spillover_panel.columns) == set(california.spatial_w.index)
    assert res.sar_posterior.beta.shape[1] == 1  # retprice


# ---------------------------------------------------------------------------
# FAILURE MODES
# ---------------------------------------------------------------------------


def test_two_treated_units_rejected(sar_panel):
    df = sar_panel["df"].copy()
    df.loc[df["unit"] == "u01", "treat"] = 1
    with pytest.raises(ScspillDataError):
        SCSPILL(base_config_kwargs(sar_panel, df=df)).fit()


def test_unbalanced_panel_rejected(sar_panel):
    df = sar_panel["df"].iloc[:-3]
    with pytest.raises(ScspillDataError):
        SCSPILL(base_config_kwargs(sar_panel, df=df)).fit()


def test_bad_config_raises_config_error(sar_panel):
    with pytest.raises(ScspillConfigError):
        SCSPILL(base_config_kwargs(sar_panel, m_iter=50, burn=50))
