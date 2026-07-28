"""Tests for panel preparation and spatial-weight alignment (setup.py)."""

import numpy as np
import pytest

from scspill.exceptions import ScspillDataError
from scspill.utils.scspill_helpers.setup import (
    normalize_w,
    prepare_scspill_inputs,
    row_normalize,
)

from .conftest import make_sar_panel

# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def panel():
    return make_sar_panel(N=9, T0=12, T1=4, K=2, seed=1)


def _prep(panel, **overrides):
    kwargs = dict(
        df=panel["df"],
        outcome="y",
        treat="treat",
        unitid="unit",
        time="time",
        spatial_W=panel["spatial_W"],
        spatial_w=panel["spatial_w"],
        covariates=panel["covariates"],
    )
    kwargs.update(overrides)
    return prepare_scspill_inputs(**kwargs)


# ---------------------------------------------------------------------------
# ROW NORMALIZATION
# ---------------------------------------------------------------------------


def test_row_normalize_properties():
    W = np.array([[2.0, 1.0, 1.0], [0.0, 5.0, 0.0], [1.0, 1.0, 0.0]])
    Wn = row_normalize(W)
    assert np.all(np.diag(Wn) == 0)
    sums = Wn.sum(axis=1)
    # Row 1 has only a diagonal entry -> becomes all-zero and stays zero.
    assert np.allclose(sums, [1.0, 0.0, 1.0])
    # Input is not modified.
    assert W[0, 0] == 2.0


def test_row_normalize_rejects_nonsquare():
    with pytest.raises(ScspillDataError):
        row_normalize(np.zeros((2, 3)))


def test_normalize_w():
    w = normalize_w(np.array([2.0, 0.0, 2.0]))
    assert np.allclose(w, [0.5, 0.0, 0.5])
    with pytest.raises(ScspillDataError):
        normalize_w(np.zeros(3))


# ---------------------------------------------------------------------------
# SETUP / ALIGNMENT
# ---------------------------------------------------------------------------


def test_shapes_and_donor_order(panel):
    inputs = _prep(panel)
    N = 9
    assert inputs.N == N
    assert inputs.T0 == 12 and inputs.T1 == 4 and inputs.T == 16
    assert inputs.Yc.shape == (16, N)
    assert inputs.X.shape == (16, N, 2)
    assert inputs.treated_label == "treated"
    # Donor order = sorted unit labels minus the treated unit.
    assert list(inputs.control_labels) == sorted(panel["units"])
    # Normalization happened.
    assert np.allclose(inputs.wn.sum(), 1.0)
    row_sums = inputs.Wn.sum(axis=1)
    assert np.all((np.isclose(row_sums, 1.0)) | (np.isclose(row_sums, 0.0)))
    assert np.all(np.diag(inputs.Wn) == 0)


def test_label_alignment_shuffle_invariance(panel):
    """A shuffled spatial_w/W must align by label, not position."""
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(panel["units"]))
    w_shuffled = panel["spatial_w"].iloc[perm]
    W_shuffled = panel["spatial_W"].iloc[perm, perm]
    a = _prep(panel)
    b = _prep(panel, spatial_w=w_shuffled, spatial_W=W_shuffled)
    assert np.array_equal(a.wn, b.wn)
    assert np.array_equal(a.Wn, b.Wn)


def test_missing_donor_label_errors(panel):
    w_missing = panel["spatial_w"].drop(panel["units"][0])
    with pytest.raises(ScspillDataError, match="spatial_w missing"):
        _prep(panel, spatial_w=w_missing)
    W_missing = panel["spatial_W"].drop(index=panel["units"][0])
    with pytest.raises(ScspillDataError, match="spatial_W missing"):
        _prep(panel, spatial_W=W_missing)


def test_bare_array_weights_require_exact_shape(panel):
    N = len(panel["units"])
    with pytest.raises(ScspillDataError):
        _prep(panel, spatial_W=np.zeros((N - 1, N)))
    with pytest.raises(ScspillDataError):
        _prep(panel, spatial_w=np.ones(N - 1))
    # Correct-shape arrays are accepted positionally.
    inputs = _prep(
        panel,
        spatial_W=panel["spatial_W"].to_numpy(),
        spatial_w=panel["spatial_w"].to_numpy(),
    )
    assert inputs.N == N


def test_w_as_dict_and_two_column_frame(panel):
    w_dict = panel["spatial_w"].to_dict()
    a = _prep(panel, spatial_w=w_dict)
    w_frame = panel["spatial_w"].rename("adj").reset_index()
    b = _prep(panel, spatial_w=w_frame)
    assert np.array_equal(a.wn, b.wn)


def test_covariate_cube_values(panel):
    """The (T, N, K) cube must match the raw df cell-for-cell (R bug-1 regression)."""
    inputs = _prep(panel)
    df = panel["df"]
    labels = list(inputs.control_labels)
    for t_probe, unit_probe, k_probe in [(0, 0, 0), (7, 4, 1), (15, 8, 0)]:
        unit = labels[unit_probe]
        t_label = inputs.time_labels[t_probe]
        cell = df[(df["unit"] == unit) & (df["time"] == t_label)][
            panel["covariates"][k_probe]
        ].iloc[0]
        assert inputs.X[t_probe, unit_probe, k_probe] == cell


def test_no_covariates(panel):
    inputs = _prep(panel, covariates=None)
    assert inputs.X is None and inputs.K == 0


def test_error_conditions(panel):
    df = panel["df"]
    # Two treated units.
    df2 = df.copy()
    df2.loc[df2["unit"] == panel["units"][0], "treat"] = 1
    with pytest.raises(ScspillDataError, match="exactly one treated"):
        _prep(panel, df=df2)
    # No treated unit.
    df3 = df.copy()
    df3["treat"] = 0
    with pytest.raises(ScspillDataError, match="exactly one treated"):
        _prep(panel, df=df3)
    # NaN outcome.
    df4 = df.copy()
    df4.loc[0, "y"] = np.nan
    with pytest.raises(ScspillDataError, match="NaN"):
        _prep(panel, df=df4)
    # Unbalanced panel.
    df5 = df.iloc[:-1]
    with pytest.raises(ScspillDataError, match="unbalanced"):
        _prep(panel, df=df5)
    # Too few pre-periods: treated switches on at t=1.
    df6 = df.copy()
    df6.loc[(df6["unit"] == "treated") & (df6["time"] >= 1), "treat"] = 1
    with pytest.raises(ScspillDataError, match="pre-treatment"):
        _prep(panel, df=df6)


def test_california_dimensions(california):
    inputs = prepare_scspill_inputs(
        california.df,
        california.outcome,
        california.treat,
        california.unitid,
        california.time,
        california.spatial_W,
        california.spatial_w,
        covariates=list(california.covariates),
    )
    assert inputs.N == 38
    assert inputs.T0 == 18 and inputs.T1 == 13
    assert inputs.treated_label == "California"
    nevada = list(inputs.control_labels).index("Nevada")
    assert inputs.wn[nevada] == 1.0  # only contiguous donor


def test_inputs_frozen_and_arrays_readonly(panel):
    inputs = _prep(panel)
    with pytest.raises(AttributeError):
        inputs.T0 = 5
    assert not inputs.Y0.flags.writeable
    assert not inputs.Wn.flags.writeable
