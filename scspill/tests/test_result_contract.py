"""Result-contract conformance for SCSPILL (mirrors mlsynth's contract test)."""

import numpy as np
import pytest
from pydantic import ValidationError

from scspill import SCSPILL, SCSPILLResults
from scspill.config_models import (
    BaseEstimatorResults,
    EffectsResults,
    FitDiagnosticsResults,
    InferenceResults,
    MethodDetailsResults,
    ScspillResult,
    TimeSeriesResults,
    WeightsResults,
)

from .conftest import base_config_kwargs

# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def res(sar_panel):
    return SCSPILL(base_config_kwargs(sar_panel, m_iter=400, burn=200)).fit()


# ---------------------------------------------------------------------------
# TYPE FAMILY
# ---------------------------------------------------------------------------


def test_result_type_family(res):
    assert isinstance(res, SCSPILLResults)
    assert isinstance(res, BaseEstimatorResults)
    assert isinstance(res, ScspillResult)


def test_submodels_populated(res):
    assert isinstance(res.effects, EffectsResults)
    assert isinstance(res.fit_diagnostics, FitDiagnosticsResults)
    assert isinstance(res.time_series, TimeSeriesResults)
    assert isinstance(res.weights, WeightsResults)
    assert isinstance(res.inference, InferenceResults)
    assert isinstance(res.method_details, MethodDetailsResults)
    # "<estimator>/<model>": the suffix names which spillover model ran.
    assert res.method_details.method_name == f"SCSPILL/{res.method}"
    assert res.method == "sar"
    assert res.method_details.parameters_used["method"] == "sar"


# ---------------------------------------------------------------------------
# FLAT ACCESSORS
# ---------------------------------------------------------------------------


def test_flat_accessors(res):
    assert isinstance(res.att, float)
    assert res.att == res.effects.att
    ci = res.att_ci
    assert ci is not None and ci[0] <= ci[1]
    assert res.counterfactual.shape == res.gap.shape
    assert isinstance(res.donor_weights, dict)
    assert res.weight_vector.shape == (len(res.donor_weights),)
    assert np.isfinite(res.pre_rmse)


def test_inference_block(res):
    inf = res.inference
    assert inf.method == "bayesian_posterior"
    assert inf.confidence_level == 0.95
    assert inf.ci == (inf.ci_lower, inf.ci_upper)
    assert inf.se is not None and inf.se > 0


def test_method_parameters_recorded(res):
    params = res.method_details.parameters_used
    for key in ("m_iter", "burn", "p_factors", "beta_prior", "propagate_alpha", "seed"):
        assert key in params


def test_serializability(res):
    """The pydantic surface must serialize without error (numpy encoders)."""
    dumped = res.model_dump(
        exclude={
            "inputs",
            "alpha_posterior",
            "sar_posterior",
            "effects_detail",
            "mcmc_summary_table",
        }
    )
    assert dumped["effects"]["att"] == res.att


# ---------------------------------------------------------------------------
# IMMUTABILITY
# ---------------------------------------------------------------------------


def test_frozen_everything(res):
    with pytest.raises(ValidationError):
        res.effects = None
    with pytest.raises(AttributeError):
        res.inputs.T0 = 1
    with pytest.raises(AttributeError):
        res.alpha_posterior.iters = 1
    with pytest.raises(AttributeError):
        res.effects_detail.att = 0.0
    for arr in (res.alpha_draws, res.rho_draws, res.effects_detail.att_draws):
        assert not arr.flags.writeable


# ---------------------------------------------------------------------------
# THE SAME CONTRACT FOR method="cd"
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cd_res():
    from scspill.data import load_california_cd

    return SCSPILL({**load_california_cd(41).config_kwargs(), "display_graphs": False}).fit()


def test_cd_result_type_family(cd_res):
    assert isinstance(cd_res, SCSPILLResults)
    assert isinstance(cd_res, BaseEstimatorResults)
    assert cd_res.method == "cd"
    assert cd_res.method_details.method_name == "SCSPILL/cd"
    assert cd_res.method_details.parameters_used["method"] == "cd"
    assert cd_res.cd is not None and cd_res.sar is None


def test_cd_flat_accessors(cd_res):
    assert cd_res.att == cd_res.effects.att
    assert cd_res.att == pytest.approx(cd_res.effects_panel["CA"].mean())
    assert cd_res.att_ci is None  # pointwise intervals do not make an ATT interval
    assert cd_res.counterfactual.shape == cd_res.gap.shape
    assert set(cd_res.donor_weights) == {str(u) for u in cd_res.inputs.control_labels}
    assert sum(cd_res.donor_weights.values()) == pytest.approx(1.0)
    assert cd_res.inference.method == "cao_dowd_P_full"
    assert cd_res.att_by_unit == {"CA": pytest.approx(cd_res.att)}


def test_cd_effects_contract(cd_res):
    eff = cd_res.effects_detail
    t0 = cd_res.inputs.T0
    assert eff.interval_kind == "cd_inversion"
    assert np.isnan(eff.cf_lower[:t0]).all() and np.isfinite(eff.cf_lower[t0:]).all()
    # Spillover panel: residuals before treatment, estimated spillovers after.
    np.testing.assert_allclose(eff.spill_mean["NV"].iloc[:t0], cd_res.pre_residuals["NV"])
    np.testing.assert_allclose(eff.spill_mean["NV"].iloc[t0:], cd_res.effects_panel["NV"])
    # An unexposed control has no spillover after treatment.
    clean = next(u for u in cd_res.inputs.control_labels if u not in ("AZ", "NV", "OR"))
    assert np.allclose(eff.spill_mean[clean].iloc[t0:], 0.0)


def test_cd_serializability(cd_res):
    dumped = cd_res.model_dump(exclude={"inputs", "effects_detail"})
    assert dumped["effects"]["att"] == cd_res.att
    assert "cd" not in dumped and "sar" not in dumped


def test_model_specific_accessors_guarded(res, cd_res):
    for name in ("rho_hat", "alpha_hat", "alpha_posterior", "mcmc_summary_table"):
        with pytest.raises(AttributeError, match="only for method='sar'"):
            getattr(cd_res, name)
    for name in ("effects_panel", "effect_ci", "kappa_test", "scm_matrix"):
        with pytest.raises(AttributeError, match="only for method='cd'"):
            getattr(res, name)
    with pytest.raises(AttributeError, match="only for method='sar'"):
        cd_res.diagnostics()
    with pytest.raises(AttributeError, match="only for method='cd'"):
        res.test_hypothesis(np.eye(1))


def test_slot_must_match_method(res, cd_res):
    from scspill.exceptions import ScspillEstimationError

    fields = {k: getattr(res, k) for k in type(res).model_fields if k not in ("cd",)}
    with pytest.raises((ValidationError, ScspillEstimationError)):
        SCSPILLResults(**{**fields, "method": "cd"})
    with pytest.raises((ValidationError, ScspillEstimationError)):
        SCSPILLResults(**{**fields, "cd": cd_res.cd})


def test_cd_frozen(cd_res):
    with pytest.raises(ValidationError):
        cd_res.effects = None
    with pytest.raises(AttributeError):
        cd_res.cd.rank = 0
