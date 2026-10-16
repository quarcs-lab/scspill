"""SCSPILLConfig validation for method="cd" and the cross-model field rule."""

import numpy as np
import pandas as pd
import pytest

from scspill import SCSPILL, SCSPILLConfig
from scspill.exceptions import ScspillConfigError

from .conftest import base_config_kwargs, make_sar_panel

# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def panel():
    return make_sar_panel(N=9, T0=12, T1=4, seed=0)


def _cd(panel, **overrides):
    kwargs = {
        "method": "cd",
        "df": panel["df"],
        "outcome": "y",
        "treat": "treat",
        "unitid": "unit",
        "time": "time",
        "display_graphs": False,
    }
    kwargs.update(overrides)
    return SCSPILLConfig(**kwargs)


def _identity_a(panel):
    labels = ["treated", *panel["units"]]
    return pd.DataFrame(np.eye(len(labels)), index=labels, columns=labels)


# ---------------------------------------------------------------------------
# SELECTING THE MODEL
# ---------------------------------------------------------------------------


def test_cd_needs_no_spatial_weights(panel):
    cfg = _cd(panel)
    assert cfg.method == "cd"
    assert cfg.spatial_w is None and cfg.spatial_W is None


def test_sar_still_requires_spatial_weights(panel):
    kwargs = base_config_kwargs(panel)
    del kwargs["spatial_W"]
    with pytest.raises(ScspillConfigError, match="spatial_W"):
        SCSPILLConfig(**kwargs)


def test_burn_check_applies_to_sar_only(panel):
    # m_iter and burn are sar fields, so cd refuses them outright instead.
    with pytest.raises(ScspillConfigError, match="apply only to method='sar'"):
        _cd(panel, m_iter=50, burn=50)


def test_dict_with_method_cd_dispatches_through_scspill(panel):
    model = SCSPILL(
        {
            "method": "cd",
            "df": panel["df"],
            "outcome": "y",
            "treat": "treat",
            "unitid": "unit",
            "time": "time",
            "display_graphs": False,
        }
    )
    assert isinstance(model.config, SCSPILLConfig) and model.method == "cd"


# ---------------------------------------------------------------------------
# CROSS-MODEL FIELDS ARE AN ERROR
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, value",
    [
        ("seed", 1),
        ("covariates", ["x1"]),
        ("p_factors", 2),
        ("backend", "numpy"),
    ],
)
def test_sar_fields_rejected_for_cd(panel, field, value):
    with pytest.raises(ScspillConfigError, match=field):
        _cd(panel, **{field: value})


def test_sar_spatial_weights_rejected_for_cd(panel):
    with pytest.raises(ScspillConfigError, match="spatial_W"):
        _cd(panel, spatial_W=panel["spatial_W"])


def test_empty_covariates_count_as_unset(panel):
    assert _cd(panel, covariates=[]).covariates == []


@pytest.mark.parametrize(
    "field, value",
    [
        ("affected_units", ["u01"]),
        ("reference_mode", "leave_one_period_out"),
        ("ci_method", "published"),
        ("scm_maxiter", 10),
    ],
)
def test_cd_fields_rejected_for_sar(panel, field, value):
    with pytest.raises(ScspillConfigError, match="apply only to method='cd'"):
        SCSPILLConfig(**base_config_kwargs(panel, **{field: value}))


def test_effect_matrix_rejected_for_sar(panel):
    with pytest.raises(ScspillConfigError, match="effect_matrix"):
        SCSPILLConfig(**base_config_kwargs(panel, effect_matrix=_identity_a(panel)))


# ---------------------------------------------------------------------------
# EXPOSURE STRUCTURE
# ---------------------------------------------------------------------------


def test_affected_units_must_be_known_distinct_controls(panel):
    with pytest.raises(ScspillConfigError, match="not found"):
        _cd(panel, affected_units=["nowhere"])
    with pytest.raises(ScspillConfigError, match="repeat"):
        _cd(panel, affected_units=["u01", "u01"])
    with pytest.raises(ScspillConfigError, match="treated unit"):
        _cd(panel, affected_units=["treated", "u01"])


def test_custom_structure_requires_effect_matrix(panel):
    with pytest.raises(ScspillConfigError, match="requires effect_matrix"):
        _cd(panel, spillover_structure="custom")
    with pytest.raises(ScspillConfigError, match="only with spillover_structure='custom'"):
        _cd(panel, effect_matrix=_identity_a(panel))
    with pytest.raises(ScspillConfigError, match="affected_units is not used"):
        _cd(
            panel,
            spillover_structure="custom",
            effect_matrix=_identity_a(panel),
            affected_units=["u01"],
        )
    cfg = _cd(panel, spillover_structure="custom", effect_matrix=_identity_a(panel))
    assert cfg.effect_matrix.shape == (10, 10)


def test_effect_matrix_rows_must_match_units(panel):
    a = _identity_a(panel).iloc[:-1]
    with pytest.raises(ScspillConfigError, match="one row per unit"):
        _cd(panel, spillover_structure="custom", effect_matrix=a)


def test_treated_row_must_be_a_direct_indicator(panel):
    """A spillover column that also loads on the treated unit is rejected."""
    labels = ["treated", *panel["units"]]
    a = pd.DataFrame(0.0, index=labels, columns=["direct", "shared"])
    a.loc["treated", "direct"] = 1
    a.loc[["treated", "u01", "u02"], "shared"] = 1
    with pytest.raises(ScspillConfigError, match="direct-effect indicator"):
        _cd(panel, spillover_structure="custom", effect_matrix=a)
    # A direct column that also loads on a control is rejected too.
    b = pd.DataFrame(0.0, index=labels, columns=["direct"])
    b.loc[["treated", "u01"], "direct"] = 1
    with pytest.raises(ScspillConfigError, match="zero for every other unit"):
        _cd(panel, spillover_structure="custom", effect_matrix=b)
    # A scaled direct column is not an indicator.
    c = pd.DataFrame(0.0, index=labels, columns=["direct"])
    c.loc["treated", "direct"] = 2
    with pytest.raises(ScspillConfigError, match="direct-effect indicator"):
        _cd(panel, spillover_structure="custom", effect_matrix=c)
