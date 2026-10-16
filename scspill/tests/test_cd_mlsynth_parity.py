"""Cross-check scspill's ``cd`` against mlsynth's SPILLSYNTH(method="cd").

mlsynth implements the same estimator (Cao & Dowd, arXiv:1902.07343v3) with a
different solver (cvxpy/CLARABEL instead of SLSQP), so the fitted SCM weights
agree only to solver tolerance, and the comparisons use matching tolerances.
mlsynth's inference follows the authors' application code (interpolated
quantiles, signed-quantile intervals), so those settings are pinned here.
The fixtures were frozen by ``tools/freeze_mlsynth_cd_reference.py``; see
``benchmarks/reference/mlsynth_cd/README.md``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scspill import SCSPILL
from scspill.data import load_california_cd

REFERENCE = json.loads(
    (
        Path(__file__).resolve().parents[2] / "benchmarks/reference/mlsynth_cd/reference.json"
    ).read_text()
)
#: mlsynth's inference settings (the authors' application code).
MLSYNTH = {"quantile_method": "linear", "ci_method": "published", "display_graphs": False}


def _panel(name: str) -> dict:
    df = pd.DataFrame(REFERENCE["panels"][name])
    return {"df": df, "outcome": "y", "treat": "treated", "unitid": "unit", "time": "time"}


def _fit(case: str):
    options = dict(REFERENCE["cases"][case]["options"])
    if case.startswith("ca"):
        spec = int(case[2:4])
        base = {**load_california_cd(spec).config_kwargs(), "affected_units": None}
    else:
        base = _panel("synth_two_treated")
    kwargs = {"method": "cd", **base, **MLSYNTH}
    kwargs["spillover_structure"] = options.get("spillover_structure", "per_unit")
    if options.get("affected_units") is not None:
        kwargs["affected_units"] = options["affected_units"]
    if options.get("unit_distances") is not None:
        kwargs["unit_distances"] = options["unit_distances"]
    if options.get("weighting", "identity") != "identity":
        kwargs["weighting"] = options["weighting"]
    return SCSPILL(kwargs).fit()


CASES = [c for c in REFERENCE["cases"] if c != "synth_efficient"]


@pytest.fixture(scope="module", params=CASES)
def pair(request):
    return REFERENCE["cases"][request.param], _fit(request.param)


def test_scm_equations_match(pair):
    ref, res = pair
    units = ref["units"]
    intercepts = np.array([res.intercepts[u] for u in units])
    expected = np.array([ref["a"][u] for u in units], dtype=float)
    scale = np.abs(res.inputs.Y0).max()
    np.testing.assert_allclose(intercepts, expected, atol=1e-5 * scale + 1e-4, rtol=0)
    b = res.scm_matrix.loc[units, units].to_numpy()
    expected_b = np.array([[ref["B"][u][v] for v in units] for u in units])
    np.testing.assert_allclose(b, expected_b, atol=1e-4, rtol=0)


def test_effects_match(pair):
    ref, res = pair
    scale = np.abs(res.inputs.Y0).max()
    for unit in ref["units"]:
        np.testing.assert_allclose(
            res.effects_panel[unit].to_numpy(), ref["alpha"][unit], atol=1e-4 * scale, rtol=0
        )
    for unit, att in ref["att_sp_by_unit"].items():
        assert res.att_by_unit[unit] == pytest.approx(att, abs=1e-4 * scale)
    assert res.cd.condition_number == pytest.approx(ref["cond_AMA"], rel=1e-4)


def test_pointwise_inference_matches(pair):
    ref, res = pair
    t0 = res.inputs.T0
    tests = {**ref["treatment_tests"], **ref["spillover_tests"]}
    for unit, test in tests.items():
        # Rank-based P values can flip only at near-ties.
        np.testing.assert_allclose(
            res.pointwise_p_values[unit].to_numpy(), test["p_value"], atol=1 / t0 + 1e-12
        )
    scale = np.abs(res.inputs.Y0).max()
    for unit, ci in ref["treatment_ci_95"].items():
        np.testing.assert_allclose(res.published_ci[unit].to_numpy(), ci, atol=1e-4 * scale, rtol=0)


def test_joint_spillover_test_matches(pair):
    ref, res = pair
    joint = ref["joint_spillover_test"]
    if joint is None:
        pytest.skip("no declared affected units")
    affected = [u for u in ref["spillover_tests"]]
    c = pd.DataFrame(0.0, index=affected, columns=list(res.unit_labels))
    for unit in affected:
        c.loc[unit, unit] = 1.0
    scale = np.abs(res.inputs.Y0).max() ** 2
    for k, period in enumerate(res.effects_panel.index):
        test = res.test_hypothesis(c, period=period)
        assert test.statistic == pytest.approx(joint["statistic"][k], abs=1e-3 * scale)
        assert test.critical_value == pytest.approx(joint["cutoff_05"], abs=1e-3 * scale)
        assert test.p_value == pytest.approx(joint["p_value"][k], abs=1 / res.inputs.T0 + 1e-12)


def test_kappa_matches(pair):
    ref, res = pair
    kappa = ref["kappa"]
    scale = np.abs(res.inputs.Y0).max()
    stats = [res.kappa_test[t].statistic for t in res.effects_panel.index]
    np.testing.assert_allclose(stats, kappa["statistic"], atol=1e-4 * scale, rtol=0)
    first = res.kappa_test[res.effects_panel.index[0]]
    np.testing.assert_allclose(
        np.sort(first.reference_statistics), np.sort(kappa["reference"]), atol=1e-4 * scale
    )
    assert first.critical_value == pytest.approx(kappa["cutoff_05"], abs=1e-4 * scale)


def test_pure_donor_sensitivity_matches(pair):
    ref, res = pair
    if "sensitivity" not in ref:
        assert res.cd.sensitivity is None
        return
    sens = res.cd.sensitivity
    expected = ref["sensitivity"]
    np.testing.assert_allclose(
        np.sort(np.abs(sens.w_sp.to_numpy()))[::-1], expected["w_sp_sorted_abs"], atol=1e-4
    )
    np.testing.assert_allclose(
        np.sort(np.abs(sens.w_pd.to_numpy()))[::-1], expected["w_pd_sorted_abs"], atol=1e-4
    )


def test_efficient_weighting_matches():
    ref = REFERENCE["cases"]["synth_efficient"]
    res = _fit("synth_efficient")
    eff = res.cd.efficient
    assert eff is not None
    for unit in ref["units"]:
        np.testing.assert_allclose(
            eff.effects_panel[unit].to_numpy(), ref["efficient"]["alpha"][unit], atol=1e-3
        )
    assert eff.att_by_unit[res.inputs.treated_label] == pytest.approx(
        ref["efficient"]["att"], abs=1e-3
    )
    # The identity fit stays the headline.
    for unit in ref["units"]:
        np.testing.assert_allclose(
            res.effects_panel[unit].to_numpy(), ref["alpha"][unit], atol=1e-3
        )


def test_structure_selection_matches():
    """mlsynth's mean-of-norms statistic selects as mlsynth does."""
    selection = REFERENCE["selection"]
    candidates = selection["candidates"]
    res = SCSPILL(
        {
            "method": "cd",
            **_panel("synth_homogeneous"),
            "affected_units": candidates["per_unit_3"]["affected_units"],
            "display_graphs": False,
        }
    ).fit()
    specs = {name: {"spillover_structure": "per_unit", **spec} for name, spec in candidates.items()}
    choice = res.select_structure_by_kappa(specs, statistic="mean_norm")
    assert choice.best == selection["best"]
    for name, value in selection["mean_kappa"].items():
        assert choice.statistics[name] == pytest.approx(value, rel=1e-3)


def test_live_against_installed_mlsynth():
    """Re-run mlsynth itself when it is installed (skipped otherwise)."""
    pytest.importorskip("cvxpy")
    setup = pytest.importorskip("mlsynth.utils.spillsynth_helpers.setup")
    pipeline = pytest.importorskip("mlsynth.utils.spillsynth_helpers.cd.pipeline")
    panel = load_california_cd(41)
    inputs = setup.prepare_spillsynth_inputs(
        panel.df,
        outcome="cigs",
        treat="treated",
        unitid="state",
        time="year",
        affected_units=list(panel.affected_units),
    )
    fit = pipeline.run_cd(inputs)
    labels = [*inputs.treated_labels, *inputs.affected_labels, *inputs.clean_labels]
    res = SCSPILL({**panel.config_kwargs(), **MLSYNTH}).fit()
    for row, unit in enumerate(labels):
        np.testing.assert_allclose(res.effects_panel[unit].to_numpy(), fit.alpha[row], atol=0.02)
