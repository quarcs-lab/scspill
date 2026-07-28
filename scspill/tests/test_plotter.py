"""Agg-backend smoke tests for the SCSPILL plotting layer."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scspill import SCSPILL
from scspill.exceptions import ScspillPlottingError
from scspill.utils.scspill_helpers.plotter import plot_scspill

from .conftest import base_config_kwargs

# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted(sar_panel):
    return SCSPILL(base_config_kwargs(sar_panel, m_iter=400, burn=200)).fit()


# ---------------------------------------------------------------------------
# PLOTTER
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["full", "effect", "weights", "rho", "trace"])
def test_single_axis_kinds(fitted, kind):
    ax = plot_scspill(fitted, kind=kind, display=False)
    assert hasattr(ax, "figure")
    plt.close("all")


def test_panel_kind(fitted):
    axes = plot_scspill(fitted, kind="panel", display=False)
    assert np.asarray(axes).size == 3
    plt.close("all")


def test_spill_top_kind(fitted):
    axes = plot_scspill(fitted, kind="spill_top", top_n=5, display=False)
    assert np.asarray(axes).size >= 5
    plt.close("all")


def test_unknown_kind_raises(fitted):
    with pytest.raises(ScspillPlottingError):
        plot_scspill(fitted, kind="nonsense", display=False)


def test_save_writes_file(fitted, tmp_path):
    target = tmp_path / "panel.png"
    plot_scspill(fitted, kind="full", save=str(target), display=False)
    assert target.exists() and target.stat().st_size > 0
    plt.close("all")


def test_results_plot_routing(fitted):
    # Base kinds go through the shared effect-plot contract...
    ax = fitted.plot(kind="counterfactual", display=False)
    assert hasattr(ax, "figure")
    ax = fitted.plot(kind="gap", display=False)
    assert hasattr(ax, "figure")
    # ...estimator-specific kinds route to plot_scspill.
    ax = fitted.plot(kind="rho", display=False)
    assert hasattr(ax, "figure")
    plt.close("all")


def test_display_graphs_fit_path(sar_panel):
    """fit() with display_graphs=True draws the panel without error (Agg)."""
    res = SCSPILL(base_config_kwargs(sar_panel, m_iter=200, burn=100, display_graphs=True)).fit()
    assert res is not None
    plt.close("all")
