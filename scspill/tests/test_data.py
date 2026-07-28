"""Tests for the bundled case-study datasets (scspill.data)."""

import numpy as np
import pandas as pd
import pytest

from scspill.data import SUDAN_COLUMN_MAP, SpillPanel, load_california, load_sudan

# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def california():
    return load_california()


@pytest.fixture(scope="module")
def sudan():
    return load_sudan()


# ---------------------------------------------------------------------------
# CALIFORNIA
# ---------------------------------------------------------------------------


def test_california_panel_shape(california):
    df = california.df
    assert df.shape[0] == 1209
    assert df["state"].nunique() == 39
    assert df["year"].min() == 1970 and df["year"].max() == 2000
    assert set(df.columns) >= {"state", "year", "cigsale", "retprice", "treated"}


def test_california_treatment_indicator(california):
    df = california.df
    # 13 post-treatment years (1988-2000) for California only.
    assert int(df["treated"].sum()) == 13
    treated_rows = df[df["treated"] == 1]
    assert set(treated_rows["state"]) == {"California"}
    assert treated_rows["year"].min() == 1988
    assert california.treated_unit == "California"
    assert california.treatment_time == 1988


def test_california_weights_alignment(california):
    donors = sorted(set(california.df["state"]) - {"California"})
    assert len(donors) == 38
    assert list(california.spatial_w.index) == donors
    assert list(california.spatial_W.index) == donors
    assert list(california.spatial_W.columns) == donors


def test_california_w_nevada_only(california):
    w = california.spatial_w
    assert w["Nevada"] == 1.0
    assert w.sum() == 1.0  # only one contiguous donor
    assert (w >= 0).all()


def test_california_W_rook_properties(california):
    W = california.spatial_W.to_numpy()
    assert np.array_equal(W, W.T)  # symmetric adjacency
    assert np.all(np.diag(W) == 0)
    assert set(np.unique(W)) <= {0.0, 1.0}


def test_california_config_kwargs(california):
    kw = california.config_kwargs()
    assert kw["outcome"] == "cigsale"
    assert kw["covariates"] == ["retprice"]
    assert isinstance(kw["spatial_w"], pd.Series)
    assert isinstance(kw["spatial_W"], pd.DataFrame)


# ---------------------------------------------------------------------------
# SUDAN
# ---------------------------------------------------------------------------


def test_sudan_panel_shape(sudan):
    df = sudan.df
    assert df.shape[0] == 544
    assert df["country"].nunique() == 34
    assert df["year"].min() == 2000 and df["year"].max() == 2015


def test_sudan_treatment_indicator(sudan):
    df = sudan.df
    assert int(df["treated"].sum()) == 5  # 2011-2015
    treated_rows = df[df["treated"] == 1]
    assert set(treated_rows["country"]) == {"Sudan"}
    assert treated_rows["year"].min() == 2011


def test_sudan_renamed_columns(sudan):
    assert sudan.outcome == "gdp_pc"
    assert "gdp_pc" in sudan.df.columns
    assert sudan.covariates == (
        "exports_gdp_share",
        "merchandise_trade_gdp_share",
        "trade_gdp_share",
        "clean_fuels_access",
        "inflation",
        "net_migration",
    )
    assert sudan.column_map == SUDAN_COLUMN_MAP


def test_sudan_raw_names():
    panel = load_sudan(raw_names=True)
    assert panel.outcome == "GDP.per.capita..constant.2015.US.."
    assert panel.column_map == {}
    assert "GDP.per.capita..constant.2015.US.." in panel.df.columns


def test_sudan_weights_alignment_and_raw_scale(sudan):
    donors = sorted(set(sudan.df["country"]) - {"Sudan"})
    assert len(donors) == 33
    assert list(sudan.spatial_w.index) == donors
    assert list(sudan.spatial_W.index) == donors
    assert list(sudan.spatial_W.columns) == donors
    # Raw US$ trade values, deliberately unnormalized.
    assert sudan.spatial_w.max() > 1e12
    assert float(sudan.spatial_W.to_numpy().max()) > 1e12
    # Egypt and Kenya trade most with Sudan (the paper's headline spillover units).
    assert list(sudan.spatial_w.nlargest(2).index) == ["Egypt, Arab Rep.", "Kenya"]


def test_sudan_W_symmetric_zero_diagonal(sudan):
    W = sudan.spatial_W.to_numpy()
    assert np.allclose(W, W.T, rtol=1e-10)
    assert np.all(np.diag(W) == 0)


# ---------------------------------------------------------------------------
# LOADER PURITY / IMMUTABILITY
# ---------------------------------------------------------------------------


def test_loader_purity():
    a = load_california()
    a.df.loc[:, "cigsale"] = -1.0
    b = load_california()
    assert (b.df["cigsale"] > 0).all()


def test_spillpanel_frozen(california):
    with pytest.raises((AttributeError, TypeError)):
        california.outcome = "other"


def test_no_missing_values(california, sudan):
    assert not california.df.isna().any().any()
    assert not sudan.df.isna().any().any()
    assert not california.spatial_W.isna().any().any()
    assert not sudan.spatial_W.isna().any().any()


def test_spillpanel_is_dataclass_instance(california):
    assert isinstance(california, SpillPanel)
