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
    assert res.method_details.method_name == "SCSPILL"


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
