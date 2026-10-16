"""Cao--Dowd estimator, inference, and Proposition 99 replication checks.

The numbers checked here come from the authors' repository,
jcao0/synthetic-control-spillover at commit 86004020: the frozen output of
their Proposition 99 application code, and draws and statistics generated
with their simulation code. Their application code uses the interpolated
quantile and the signed-quantile interval, so those checks pass
``APPLICATION_CODE`` explicitly; the package defaults are the paper's (v3)
Section 4 definitions.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scspill import (
    SCSPILL,
    SCSPILLConfig,
    ScspillConfigError,
    ScspillEstimationError,
    SCSPILLResults,
)
from scspill.data import load_california_cd
from scspill.utils.scspill_helpers.cd.scm_core import fit_all_unit_scm

REFERENCE = Path(__file__).resolve().parents[2] / "benchmarks/reference/cao_dowd"
#: The settings of the authors' Proposition 99 application code.
APPLICATION_CODE = {"quantile_method": "linear", "ci_method": "published"}


def test_scm_is_optimal_with_tiny_maxiter_and_falls_back_to_fista(monkeypatch):
    rng = np.random.default_rng(17)
    y = rng.normal(size=(4, 12))
    y[0] = 0.85 * y[1] + 0.15 * y[2] + 0.02 * rng.normal(size=12)
    short_intercepts, short_weights = fit_all_unit_scm(y, maxiter=1, ftol=1e-12)
    regular_intercepts, regular_weights = fit_all_unit_scm(y, maxiter=3000, ftol=1e-12)
    short_loss = np.sum((y[0] - short_intercepts[0] - short_weights[0] @ y) ** 2)
    regular_loss = np.sum((y[0] - regular_intercepts[0] - regular_weights[0] @ y) ** 2)
    assert short_loss <= regular_loss + 1e-5

    # A failed SLSQP falls back to projected gradient with the same optimum.
    from scipy.optimize import OptimizeResult

    def failing(fun, x0, **kwargs):
        return OptimizeResult(x=np.full_like(x0, np.nan), success=False)

    monkeypatch.setattr("scspill.utils.scspill_helpers.cd.scm_core.minimize", failing)
    fb_intercepts, fb_weights = fit_all_unit_scm(y, maxiter=3000, ftol=1e-12)
    fb_loss = np.sum((y[0] - fb_intercepts[0] - fb_weights[0] @ y) ** 2)
    assert fb_loss <= regular_loss + 1e-4
    assert fb_weights[0].sum() == pytest.approx(1)


@pytest.mark.parametrize("spec", [51, 41])
def test_application_against_authors_frozen_output(spec):
    panel = load_california_cd(spec)
    expected = json.loads((REFERENCE / f"results_{spec}.json").read_text())
    result = SCSPILL({**panel.config_kwargs(), **APPLICATION_CODE, "display_graphs": False}).fit()
    assert isinstance(result, SCSPILLResults) and result.method == "cd"
    assert len(result.unit_labels) == spec
    assert result.inputs.T0 == 19
    assert result.cd.rank == expected["k"]
    assert result.cd.condition_number == pytest.approx(expected["cond"], abs=0.03)
    np.testing.assert_allclose(result.effects_panel["CA"], expected["te"], atol=0.03)
    np.testing.assert_allclose(result.counterfactual, expected["sp"], atol=0.03)
    np.testing.assert_allclose(
        result.published_ci["CA"].iloc[0].values - result.effects_panel["CA"].iloc[0],
        expected["te_ci"],
        atol=0.02,
    )
    np.testing.assert_allclose(result.pointwise_p_values["CA"], expected["te_pvalue"])
    for state in ("AZ", "NV", "OR"):
        np.testing.assert_allclose(
            result.effects_panel[state], expected["u" + state][19:], atol=0.03
        )
        np.testing.assert_allclose(
            result.pre_residuals[state], expected["u" + state][:19], atol=0.03
        )
        np.testing.assert_allclose(result.pointwise_p_values[state], expected["p" + state])


def test_defaults_are_v3_section4():
    """Defaults: left-inverse critical values and test-inversion intervals."""
    panel = load_california_cd(51)
    result = SCSPILL({**panel.config_kwargs(), "display_graphs": False}).fit()
    assert (result.cd.quantile_method, result.cd.ci_method) == ("empirical", "inversion")
    assert result.cd.reference_mode == "full"
    assert result.effect_ci is result.pointwise_ci
    width = result.pointwise_ci["CA"].eval("upper - lower")
    assert np.allclose(width, width.iloc[0])  # symmetric, constant half-width
    # Left inverse: with 19 reference periods the 95% critical value is the 19th
    # order statistic, the largest one.
    spill = np.eye(len(result.unit_labels))[1:]
    test = result.test_hypothesis(spill, period=1990)
    assert test.critical_value == pytest.approx(test.reference_statistics.max())
    # Prediction band is the selected interval mapped to the counterfactual scale.
    band = result.time_series.counterfactual_lower
    observed = result.time_series.observed_outcome
    np.testing.assert_allclose(
        band[19:], observed[19:] - result.effect_ci["CA"]["upper"].to_numpy()
    )


def test_application_code_switches_reproduce_the_published_rejections():
    """The authors' code settings reproduce the published joint spillover test."""
    panel = load_california_cd(51)
    result = SCSPILL({**panel.config_kwargs(), **APPLICATION_CODE, "display_graphs": False}).fit()
    assert result.effect_ci is result.published_ci
    spill = np.eye(len(result.unit_labels))[1:]
    years = result.effects_panel.index
    linear = [t for t in years if result.test_hypothesis(spill, period=t).reject]
    paper = [
        t
        for t in years
        if result.test_hypothesis(spill, period=t, quantile_method="empirical").reject
    ]
    assert linear == [1990, 1991, 1992, *range(1994, 2001)]
    assert 1994 not in paper and 1995 not in paper
    band = result.time_series.counterfactual_lower
    observed = result.time_series.observed_outcome
    ci = result.published_ci["CA"]
    np.testing.assert_allclose(band[19:], observed[19:] - ci["upper"].to_numpy())


def test_general_hypothesis_and_distinct_intervals():
    panel = load_california_cd(41)
    result = SCSPILL({**panel.config_kwargs(), "display_graphs": False}).fit()
    c = pd.DataFrame(0.0, index=["CA+NV"], columns=result.unit_labels)
    c.loc["CA+NV", ["CA", "NV"]] = 1
    test = result.test_hypothesis(c, period=1990, d=0)
    assert test.reference_statistics.shape == (19,)
    assert 0 <= test.p_value <= 1
    assert test.statistic >= 0
    assert result.pointwise_ci["CA"].loc[1990, "lower"] != pytest.approx(
        result.published_ci["CA"].loc[1990, "lower"]
    )
    assert result.att_ci is None  # a pointwise interval is not an ATT interval
    assert result.time_series.has_prediction_interval
    assert len(result.kappa_test) == 12


def test_labeled_effect_matrix_and_rank_failure():
    panel = load_california_cd(41)
    labels = list(pd.unique(panel.df["state"]))
    a = pd.DataFrame(0.0, index=labels[::-1], columns=["direct", "shared"])
    a.loc["CA", "direct"] = 1
    a.loc[["AZ", "NV", "OR"], "shared"] = 1
    config = SCSPILLConfig(
        **{
            **panel.config_kwargs(),
            "affected_units": None,
            "spillover_structure": "custom",
            "effect_matrix": a,
            "display_graphs": False,
        }
    )
    result = SCSPILL(config).fit()
    assert result.effect_matrix.loc["CA", "direct"] == 1
    assert result.effects_panel.loc[1990, "AZ"] == pytest.approx(
        result.effects_panel.loc[1990, "NV"]
    )
    bad = pd.DataFrame(np.eye(len(labels)), index=labels, columns=labels)
    with pytest.raises(ScspillEstimationError, match="unidentified"):
        SCSPILL(
            {
                **panel.config_kwargs(),
                "affected_units": None,
                "spillover_structure": "custom",
                "effect_matrix": bad,
                "display_graphs": False,
            }
        ).fit()


def test_leave_one_period_out_and_input_validation():
    panel = load_california_cd(41)
    small = panel.df.loc[
        panel.df["state"].isin(["CA", "AZ", "NV", "OR", "CO"])
        & panel.df["year"].between(1970, 1977)
    ].copy()
    small.loc[small["state"] == "CA", "treated"] = (
        small.loc[small["state"] == "CA", "year"] >= 1976
    ).astype(int)
    cfg = {
        **panel.config_kwargs(),
        "df": small,
        "affected_units": ["AZ", "NV"],
        "display_graphs": False,
    }
    full = SCSPILL(cfg).fit()
    loo = SCSPILL({**cfg, "reference_mode": "leave_one_period_out"}).fit()
    assert not np.allclose(full.placebo_effects, loo.placebo_effects)
    assert loo.kappa_test[1976].reference_statistics.size == 6
    with pytest.raises(ScspillConfigError):
        SCSPILL({**cfg, "affected_units": ["unknown"]}).fit()
    with pytest.raises(ScspillConfigError):
        loo.test_hypothesis(np.ones((1, 2)))
    with pytest.raises(ScspillConfigError):
        loo.test_hypothesis(np.empty((0, len(loo.unit_labels))))
    with pytest.raises(ScspillConfigError):
        loo.test_hypothesis(np.eye(len(loo.unit_labels))[0], d=[0, 1])


def test_monte_carlo_first_replication_against_published_estimates():
    """Check six simulation settings against draws and estimates from the authors' code."""
    with np.load(REFERENCE / "inference_replication_1.npz") as fixture:
        panels = fixture["panels"]
        expected = fixture["sp_estimates"]
    labels = [f"U{i:02d}" for i in range(51)]
    for panel, expected_estimate in zip(panels, expected, strict=True):
        df = pd.DataFrame(panel, index=labels, columns=range(51)).stack().rename("y").reset_index()
        df.columns = ["unit", "time", "y"]
        df["treated"] = ((df["unit"] == "U00") & (df["time"] == 50)).astype(int)
        result = SCSPILL(
            {
                "method": "cd",
                "df": df,
                "outcome": "y",
                "treat": "treated",
                "unitid": "unit",
                "time": "time",
                "affected_units": labels[1:14],
                "display_graphs": False,
            }
        ).fit()
        assert result.effects_panel.iloc[0, 0] == pytest.approx(expected_estimate, abs=0.001)


def test_kappa_against_author_reference():
    """Compare per-unit and homogeneous kappa_A statistics with the authors' code."""
    with np.load(REFERENCE / "inference_replication_1.npz") as fixture:
        panel = fixture["panels"][1]
    expected = json.loads((REFERENCE / "kappa_rep1.json").read_text())
    labels = [f"U{i:02d}" for i in range(51)]
    df = pd.DataFrame(panel, index=labels, columns=range(51)).stack().rename("y").reset_index()
    df.columns = ["unit", "time", "y"]
    df["treated"] = ((df["unit"] == "U00") & (df["time"] == 50)).astype(int)
    base = {
        "method": "cd",
        "df": df,
        "outcome": "y",
        "treat": "treated",
        "unitid": "unit",
        "time": "time",
        "display_graphs": False,
    }
    range_fit = SCSPILL({**base, "affected_units": labels[1:14]}).fit()
    homogeneous = pd.DataFrame(0.0, index=labels, columns=["direct", "shared"])
    homogeneous.loc["U00", "direct"] = 1
    homogeneous.loc[labels[1:14], "shared"] = 1
    homogeneous_fit = SCSPILL(
        {**base, "spillover_structure": "custom", "effect_matrix": homogeneous}
    ).fit()
    for key, result in (("range", range_fit), ("homogeneous", homogeneous_fit)):
        assert result.kappa_test[50].statistic == pytest.approx(
            expected[key]["statistic"], abs=0.001
        )
        np.testing.assert_allclose(
            result.kappa_test[50].reference_statistics,
            expected[key]["reference"],
            atol=0.001,
            rtol=0,
        )


@pytest.mark.slow
def test_inference_leave_one_out_against_author_reference():
    """Check all 50 squared SP-A placebo statistics, not just the effect."""
    with np.load(REFERENCE / "inference_replication_1.npz") as fixture:
        panel = fixture["panels"][1]  # stationary factors, r=0.5, replication 1
    author = np.load(REFERENCE / "loo_reference_stationary_r05_rep1.npy")
    labels = [f"U{i:02d}" for i in range(51)]
    df = pd.DataFrame(panel, index=labels, columns=range(51)).stack().rename("y").reset_index()
    df.columns = ["unit", "time", "y"]
    df["treated"] = ((df["unit"] == "U00") & (df["time"] == 50)).astype(int)
    result = SCSPILL(
        {
            "method": "cd",
            "df": df,
            "outcome": "y",
            "treat": "treated",
            "unitid": "unit",
            "time": "time",
            "affected_units": labels[1:14],
            "reference_mode": "leave_one_period_out",
            "display_graphs": False,
        }
    ).fit()
    observed = result.placebo_effects["U00"].to_numpy() ** 2
    np.testing.assert_allclose(observed, author, atol=0.001, rtol=0)
    # The authors' simulation code uses the paper's left-inverse critical value.
    test = result.test_hypothesis(np.eye(51)[0], period=50, quantile_method="empirical")
    assert test.critical_value == pytest.approx(np.sort(author)[47], abs=0.001)
    assert test.p_value == float(np.mean(author >= test.statistic))
