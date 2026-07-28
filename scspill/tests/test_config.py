"""Tests for SCSPILLConfig validation."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from scspill import SCSPILL, SCSPILLConfig
from scspill.exceptions import ScspillConfigError, ScspillDataError

from .conftest import base_config_kwargs, make_sar_panel

# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def panel():
    return make_sar_panel(N=9, T0=12, T1=4, seed=0)


def _cfg(panel, **overrides):
    return SCSPILLConfig(**base_config_kwargs(panel, **overrides))


# ---------------------------------------------------------------------------
# CONFIG VALIDATION
# ---------------------------------------------------------------------------


def test_defaults(panel):
    cfg = _cfg(panel)
    assert cfg.m_iter == 600 and cfg.burn == 300  # from the test factory
    assert cfg.step_rho == 0.05
    assert cfg.adapt_rho is True
    assert cfg.target_accept_rho == 0.44
    assert cfg.beta_prior == "horseshoe"
    assert cfg.propagate_alpha is True
    assert cfg.a0 == 1.0 and cfg.b0 == 1.0
    assert cfg.ci == 0.95
    assert cfg.backend == "auto"


def test_extra_fields_forbidden(panel):
    with pytest.raises((ValidationError, ScspillConfigError)):
        _cfg(panel, nonsense_field=1)


def test_burn_must_be_less_than_m_iter(panel):
    with pytest.raises((ScspillConfigError, ValidationError)):
        _cfg(panel, m_iter=100, burn=100)
    with pytest.raises((ScspillConfigError, ValidationError)):
        _cfg(panel, m_iter=100, burn=200)


def test_invalid_numeric_fields(panel):
    with pytest.raises(ValidationError):
        _cfg(panel, step_rho=0.0)
    with pytest.raises(ValidationError):
        _cfg(panel, ci=1.0)
    with pytest.raises(ValidationError):
        _cfg(panel, p_factors=-1)
    with pytest.raises(ValidationError):
        _cfg(panel, target_accept_rho=1.5)


def test_unknown_covariate_rejected(panel):
    with pytest.raises((ScspillConfigError, ValidationError)):
        _cfg(panel, covariates=["not_a_column"])


def test_missing_required_columns():
    df = pd.DataFrame({"unit": ["a", "b"], "time": [1, 1], "y": [1.0, 2.0]})
    with pytest.raises((ScspillDataError, ValidationError)):
        SCSPILLConfig(
            df=df,
            outcome="y",
            treat="treat",  # column absent
            unitid="unit",
            time="time",
            spatial_w=np.ones(1),
            spatial_W=np.zeros((1, 1)),
        )


def test_empty_dataframe_rejected():
    with pytest.raises((ScspillDataError, ValidationError)):
        SCSPILLConfig(
            df=pd.DataFrame(columns=["unit", "time", "y", "treat"]),
            outcome="y",
            treat="treat",
            unitid="unit",
            time="time",
            spatial_w=np.ones(1),
            spatial_W=np.zeros((1, 1)),
        )


def test_estimator_coerces_dict_and_translates_error(panel):
    est = SCSPILL(base_config_kwargs(panel))
    assert isinstance(est.config, SCSPILLConfig)
    with pytest.raises(ScspillConfigError):
        SCSPILL({"df": panel["df"]})  # missing required fields
    with pytest.raises(ScspillConfigError):
        SCSPILL(42)  # neither config nor dict


def test_unknown_backend_rejected(panel):
    with pytest.raises(ValidationError):
        _cfg(panel, backend="fortran")


def test_r_argument_mapping_documented():
    doc = SCSPILLConfig.__doc__
    assert "``M``" in doc and "m_iter" in doc
    assert "treatment_dummy" in doc


# ---------------------------------------------------------------------------
# MODEL SELECTION (the `method` dispatch seam)
# ---------------------------------------------------------------------------


def test_method_defaults_to_sar(panel):
    """Omitting `method` must keep selecting the only implemented model."""
    assert _cfg(panel).method == "sar"


def test_method_accepts_sar_explicitly(panel):
    assert _cfg(panel, method="sar").method == "sar"


def test_planned_methods_are_rejected_not_silently_accepted(panel):
    """The roadmap names must not validate: they are not implemented."""
    for planned in ("cd", "iscm", "grossi"):
        with pytest.raises(ValidationError):
            _cfg(panel, method=planned)
