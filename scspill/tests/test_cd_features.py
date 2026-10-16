"""The cd model's exposure structures, multiple treated units, efficient
weighting, pure-donor sensitivity, and structure selection.

Numbers are checked against mlsynth in test_cd_mlsynth_parity.py; this file
checks the identities that must hold between the features themselves.
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from scspill import SCSPILL
from scspill.data import load_california_cd
from scspill.exceptions import ScspillConfigError, ScspillDataError, ScspillEstimationError
from scspill.utils.scspill_helpers.cd.estimation import (
    effect_operator,
    weighted_effect_operator,
)
from scspill.utils.scspill_helpers.plotter import plot_scspill

from .conftest import base_config_kwargs, make_sar_panel

REFERENCE = Path(__file__).resolve().parents[2] / "benchmarks/reference"
EXPOSED = ["AZ", "NV", "OR"]


def _ca41(**overrides):
    kwargs = {
        **load_california_cd(41).config_kwargs(),
        "affected_units": None,
        "display_graphs": False,
    }
    kwargs.update(overrides)
    return SCSPILL(kwargs).fit()


@pytest.fixture(scope="module")
def homogeneous():
    return _ca41(spillover_structure="homogeneous", affected_units=EXPOSED)


@pytest.fixture(scope="module")
def per_unit():
    return _ca41(affected_units=EXPOSED)


def _assert_same_inference(left, right, atol=1e-8):
    np.testing.assert_allclose(left.effects_panel, right.effects_panel, atol=atol)
    for unit in left.unit_labels:
        np.testing.assert_allclose(left.effect_ci[unit], right.effect_ci[unit], atol=atol)
    np.testing.assert_allclose(left.pointwise_p_values, right.pointwise_p_values)
    for period in left.kappa_test:
        assert left.kappa_test[period].statistic == pytest.approx(
            right.kappa_test[period].statistic, abs=atol
        )


# ---------------------------------------------------------------------------
# EXPOSURE STRUCTURES
# ---------------------------------------------------------------------------


def test_homogeneous_structure_shares_one_spillover(homogeneous):
    a = homogeneous.effect_matrix
    assert list(a.columns) == ["CA", "spillover"]
    effects = homogeneous.effects_panel
    np.testing.assert_allclose(effects["AZ"], effects["NV"])
    np.testing.assert_allclose(effects["AZ"], homogeneous.cd.gamma["spillover"])
    assert homogeneous.inputs.affected_labels == tuple(EXPOSED)


def test_equal_distances_reproduce_the_homogeneous_structure(homogeneous):
    """exp(-d) * b with one d for every exposed unit is a homogeneous spillover."""
    d = 0.7
    decay = _ca41(spillover_structure="distance_decay", unit_distances=dict.fromkeys(EXPOSED, d))
    _assert_same_inference(decay, homogeneous)
    # The parameter is the spillover at distance zero: scaled by exp(d).
    np.testing.assert_allclose(
        decay.cd.gamma["spillover"], homogeneous.cd.gamma["spillover"] * np.exp(d)
    )


def test_distance_decay_scales_spillovers_by_distance():
    decay = _ca41(
        spillover_structure="distance_decay", unit_distances={"NV": 0.0, "AZ": 1.0, "OR": 2.0}
    )
    effects = decay.effects_panel
    np.testing.assert_allclose(effects["AZ"], effects["NV"] * np.exp(-1.0))
    np.testing.assert_allclose(effects["OR"], effects["NV"] * np.exp(-2.0))
    assert decay.inputs.affected_labels == ("AZ", "NV", "OR")
    clean = [u for u in decay.inputs.control_labels if u not in EXPOSED]
    assert np.allclose(effects[clean], 0.0)


def test_one_affected_unit_per_unit_equals_homogeneous():
    left = _ca41(affected_units=["NV"])
    right = _ca41(spillover_structure="homogeneous", affected_units=["NV"])
    _assert_same_inference(left, right)


def test_custom_structure_equal_to_a_built_one(per_unit):
    custom = _ca41(spillover_structure="custom", effect_matrix=per_unit.effect_matrix)
    _assert_same_inference(custom, per_unit)
    # Rescaling a spillover column changes gamma, not the unit effects.
    scaled = per_unit.effect_matrix.copy()
    scaled["NV"] *= 3.0
    rescaled = _ca41(spillover_structure="custom", effect_matrix=scaled)
    np.testing.assert_allclose(rescaled.effects_panel, per_unit.effects_panel, atol=1e-8)
    np.testing.assert_allclose(rescaled.cd.gamma["NV"] * 3.0, per_unit.cd.gamma["NV"])


def test_homogeneous_spillover_on_every_control_is_unidentified():
    """(I - B) 1 = 0 because every SCM row sums to one."""
    controls = list(load_california_cd(41).df["state"].unique())
    controls.remove("CA")
    with pytest.raises(ScspillEstimationError, match="unidentified"):
        _ca41(spillover_structure="homogeneous", affected_units=controls)


def test_homogeneous_against_the_authors_kappa_reference():
    """Authors' kappa_A code, homogeneous structure, built by spillover_structure."""
    with np.load(REFERENCE / "cao_dowd/inference_replication_1.npz") as fixture:
        panel = fixture["panels"][1]
    expected = json.loads((REFERENCE / "cao_dowd/kappa_rep1.json").read_text())["homogeneous"]
    labels = [f"U{i:02d}" for i in range(51)]
    df = pd.DataFrame(panel, index=labels, columns=range(51)).stack().rename("y").reset_index()
    df.columns = ["unit", "time", "y"]
    df["treated"] = ((df["unit"] == "U00") & (df["time"] == 50)).astype(int)
    res = SCSPILL(
        {
            "method": "cd",
            "df": df,
            "outcome": "y",
            "treat": "treated",
            "unitid": "unit",
            "time": "time",
            "spillover_structure": "homogeneous",
            "affected_units": labels[1:14],
            "display_graphs": False,
        }
    ).fit()
    assert res.kappa_test[50].statistic == pytest.approx(expected["statistic"], abs=1e-3)
    np.testing.assert_allclose(
        res.kappa_test[50].reference_statistics, expected["reference"], atol=1e-3
    )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"spillover_structure": "homogeneous"}, "at least one affected unit"),
        ({"spillover_structure": "distance_decay"}, "requires unit_distances"),
        (
            {"spillover_structure": "distance_decay", "unit_distances": {"CA": 1.0}},
            "treated unit",
        ),
        (
            {"spillover_structure": "distance_decay", "unit_distances": {"NV": -1.0}},
            "non-negative",
        ),
        (
            {"spillover_structure": "distance_decay", "unit_distances": {"XX": 1.0}},
            "not found",
        ),
        (
            {
                "spillover_structure": "distance_decay",
                "unit_distances": {"NV": 1.0},
                "affected_units": ["NV"],
            },
            "affected_units is not used",
        ),
        ({"unit_distances": {"NV": 1.0}}, "only with spillover_structure='distance_decay'"),
    ],
)
def test_structure_configuration_errors(overrides, message):
    with pytest.raises(ScspillConfigError, match=message):
        _ca41(**overrides)


# ---------------------------------------------------------------------------
# MULTIPLE TREATED UNITS (v3 S.1.2)
# ---------------------------------------------------------------------------


def _two_treated(rename=None, start=(40, 40)):
    rng = np.random.default_rng(3)
    labels = [f"U{i:02d}" for i in range(10)]
    t = 45
    factors = rng.normal(size=(t, 2)).cumsum(axis=0) * 0.3
    y = 10 + rng.uniform(0.2, 1.5, size=(10, 2)) @ factors.T + rng.normal(0, 0.5, (10, t))
    y[0, 40:] -= 3.0
    y[1, 40:] -= 1.0
    y[2, 40:] += 2.0
    rows = []
    for i, u in enumerate(labels):
        for s in range(t):
            treated = (i == 0 and s >= start[0]) or (i == 1 and s >= start[1])
            rows.append({"unit": u, "time": s, "y": y[i, s], "treated": int(treated)})
    df = pd.DataFrame(rows)
    if rename:
        df["unit"] = df["unit"].replace(rename)
    return df


def _fit_two(df, affected=("U02",)):
    return SCSPILL(
        {
            "method": "cd",
            "df": df,
            "outcome": "y",
            "treat": "treated",
            "unitid": "unit",
            "time": "time",
            "affected_units": list(affected),
            "display_graphs": False,
        }
    ).fit()


def test_multiple_treated_units():
    res = _fit_two(_two_treated())
    assert res.treated_labels == ("U00", "U01")
    assert res.inputs.treated_label == "U00"
    assert list(res.effect_matrix.columns) == ["U00", "U01", "U02"]
    for unit in ("U00", "U01"):
        assert res.att_by_unit[unit] == pytest.approx(res.effects_panel[unit].mean())
    assert res.att == pytest.approx(res.att_by_unit["U00"])
    # The other treated unit is neither a control nor in the spillover panel.
    assert "U01" not in res.inputs.control_labels
    assert "U01" not in res.effects_detail.spill_mean.columns
    # One pointwise interval and P value per treated unit.
    assert {"U00", "U01"} <= set(res.effect_ci) and "U01" in res.pointwise_p_values


def test_headline_choice_does_not_change_the_effects():
    """Renaming so the other treated unit sorts first swaps only the headline."""
    base = _fit_two(_two_treated())
    swapped = _fit_two(_two_treated(rename={"U01": "A01"}))
    assert swapped.inputs.treated_label == "A01"
    np.testing.assert_allclose(swapped.effects_panel["A01"], base.effects_panel["U01"], atol=1e-8)
    np.testing.assert_allclose(swapped.effects_panel["U00"], base.effects_panel["U00"], atol=1e-8)
    assert swapped.att == pytest.approx(base.att_by_unit["U01"], abs=1e-8)


def test_staggered_treatment_rejected():
    with pytest.raises(ScspillDataError, match="common treatment start"):
        _fit_two(_two_treated(start=(40, 42)))


def test_sar_still_rejects_two_treated_units():
    panel = make_sar_panel(N=9, T0=12, T1=4, seed=0)
    df = panel["df"].copy()
    start = df.loc[df["treat"] == 1, "time"].min()
    df.loc[(df["unit"] == panel["units"][0]) & (df["time"] >= start), "treat"] = 1
    with pytest.raises(ScspillDataError, match="exactly one treated"):
        SCSPILL(base_config_kwargs(panel, df=df)).fit()


# ---------------------------------------------------------------------------
# EFFICIENT WEIGHTING (v3 S.1.1)
# ---------------------------------------------------------------------------


def test_identity_weight_is_the_unweighted_estimator(per_unit):
    labels = list(per_unit.unit_labels)
    b = per_unit.scm_matrix.loc[labels, labels].to_numpy()
    a = per_unit.effect_matrix.to_numpy()
    plain = effect_operator(b, a)
    for weight in (np.eye(len(labels)), 4.0 * np.eye(len(labels))):
        weighted = weighted_effect_operator(b, a, weight)
        np.testing.assert_allclose(weighted.unit_map, plain.unit_map, atol=1e-8)


def test_efficient_fit_is_reported_alongside():
    res = _fit_two(_two_treated())
    assert res.cd.efficient is None
    eff_res = SCSPILL({**_two_kwargs(), "weighting": "efficient"}).fit()
    eff = eff_res.cd.efficient
    assert eff is not None and eff_res.cd.weighting == "efficient"
    # The headline stays the identity-weighted fit.
    np.testing.assert_allclose(eff_res.effects_panel, res.effects_panel)
    assert set(eff.att_by_unit) == {"U00", "U01"}
    np.testing.assert_allclose(eff.weight_matrix @ eff.omega_hat, np.eye(10), atol=1e-6)


def _two_kwargs():
    return {
        "method": "cd",
        "df": _two_treated(),
        "outcome": "y",
        "treat": "treated",
        "unitid": "unit",
        "time": "time",
        "affected_units": ["U02"],
        "display_graphs": False,
    }


def test_efficient_warns_when_the_covariance_is_ridge_dominated():
    with pytest.warns(RuntimeWarning, match="singular without its ridge"):
        _ca41(affected_units=EXPOSED, weighting="efficient")


def test_no_warning_with_enough_pre_periods():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        SCSPILL({**_two_kwargs(), "weighting": "efficient"}).fit()


# ---------------------------------------------------------------------------
# PURE-DONOR SENSITIVITY (v3 Section 5.2.2)
# ---------------------------------------------------------------------------


def test_sensitivity_weights(per_unit):
    sens = per_unit.cd.sensitivity
    assert sens.treated_label == "CA"
    assert set(sens.clean_labels) == set(per_unit.inputs.control_labels) - set(EXPOSED)
    assert sens.w_pd.sum() == pytest.approx(1.0)
    assert (sens.w_pd >= 0).all()
    coefficients = [sens.coefficients(p) for p in range(1, sens.n_clean + 1)]
    assert np.all(np.diff([c[0] for c in coefficients]) >= 0)
    assert np.all(np.diff([c[1] for c in coefficients]) >= 0)
    bounds = sens.bias_bounds(2, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(bounds["spillover_aware"], np.array([0, 1, 2]) * coefficients[1][0])
    with pytest.raises(ScspillConfigError):
        sens.coefficients(0)


def test_no_sensitivity_without_clean_controls():
    controls = [u for u in load_california_cd(41).df["state"].unique() if u != "CA"]
    distances = {u: float(k % 5) for k, u in enumerate(controls)}
    res = _ca41(spillover_structure="distance_decay", unit_distances=distances)
    assert res.inputs.clean_labels == ()
    assert res.cd.sensitivity is None


# ---------------------------------------------------------------------------
# STRUCTURE SELECTION BY kappa_A (v3 Sections 5.1.2 and S.1.3)
# ---------------------------------------------------------------------------


def test_selection_reuses_the_fit(per_unit):
    choice = per_unit.select_structure_by_kappa(
        {"fitted": per_unit.effect_matrix, "none": {"affected_units": []}},
        statistic="mean_norm",
    )
    stats = [per_unit.kappa_test[t].statistic for t in per_unit.effects_panel.index]
    assert choice.statistics["fitted"] == pytest.approx(np.mean(stats))
    assert choice.statistic == "mean_norm"
    v3 = per_unit.select_structure_by_kappa(
        {
            "fitted": per_unit.effect_matrix,
            "homogeneous": {"spillover_structure": "homogeneous", "affected_units": EXPOSED},
        }
    )
    assert v3.statistic == "v3" and v3.best in ("fitted", "homogeneous")
    # A per-unit structure nests the homogeneous one, so it fits at least as well.
    assert v3.statistics["fitted"] <= v3.statistics["homogeneous"] + 1e-9


def test_selection_errors(per_unit):
    with pytest.raises(ScspillConfigError, match="unknown keys"):
        per_unit.select_structure_by_kappa({"x": {"affected": ["NV"]}})
    controls = list(per_unit.inputs.control_labels)
    with pytest.raises(ScspillConfigError, match="Candidate 'all'"):
        per_unit.select_structure_by_kappa(
            {"all": {"spillover_structure": "homogeneous", "affected_units": controls}}
        )
    with pytest.raises(ScspillConfigError, match="statistic"):
        per_unit.select_structure_by_kappa({"x": {}}, statistic="other")


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------


def test_sensitivity_and_weights_plots(per_unit):
    ax = plot_scspill(per_unit, kind="sensitivity", display=False)
    assert len(ax.get_lines()) == 4  # two methods x p = 1, 2
    ax = plot_scspill(per_unit, kind="weights", display=False)
    assert "Pure-donor SCM" in [t.get_text() for t in ax.get_legend().get_texts()]
    plt.close("all")
