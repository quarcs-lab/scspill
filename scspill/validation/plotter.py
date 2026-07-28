"""Plots for the validation suite."""

from __future__ import annotations

import numpy as np

from ..exceptions import ScspillPlottingError
from ..utils.plotting import scspill_style
from .structures import GewekeReport, PriorPredictiveResult


def plot_prior_predictive(
    result: PriorPredictiveResult,
    *,
    save: str | None = None,
    show: bool = False,
):
    """Histogram grid of the prior predictive statistics with observed markers.

    Parameters
    ----------
    result : PriorPredictiveResult
        The output of :func:`scspill.validation.prior_predictive`.
    save : str, optional
        File path to save the figure to.
    show : bool, default False
        Display the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    stats = result.stats
    names = [c for c in stats.columns if stats[c].notna().any()]
    if not names:
        raise ScspillPlottingError("plot_prior_predictive: no finite statistics to plot.")
    ncols = 3
    nrows = int(np.ceil(len(names) / ncols))
    with scspill_style():
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False)
        for k, name in enumerate(names):
            ax = axes[k // ncols][k % ncols]
            vals = stats[name].to_numpy()
            vals = vals[np.isfinite(vals)]
            ax.hist(vals, bins="fd", color="#1f77b4", alpha=0.8)
            if result.observed is not None and np.isfinite(result.observed.get(name, np.nan)):
                ax.axvline(result.observed[name], color="crimson", linewidth=1.4)
                if result.p_values is not None:
                    ax.set_title(f"{name} (p = {result.p_values[name]:.3f})", fontsize=9)
                else:  # pragma: no cover - observed without p-values
                    ax.set_title(name, fontsize=9)
            else:
                ax.set_title(name, fontsize=9)
        for k in range(len(names), nrows * ncols):
            axes[k // ncols][k % ncols].set_visible(False)
        fig.suptitle("Prior predictive checks", fontsize=11)
        fig.tight_layout()
        if save:
            fig.savefig(save, bbox_inches="tight")
        if show:  # pragma: no cover - interactive path
            plt.show()
    return fig


def plot_geweke(report: GewekeReport, *, save: str | None = None, show: bool = False):
    """Dot plot of the Geweke z-scores with the critical band.

    Parameters
    ----------
    report : GewekeReport
        The output of :func:`scspill.validation.geweke_test`.
    save : str, optional
        File path to save the figure to.
    show : bool, default False
        Display the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    table = report.table
    with scspill_style():
        fig, ax = plt.subplots(figsize=(7, 0.5 * len(table) + 1.5))
        ypos = np.arange(len(table))
        ax.axvspan(-report.z_crit, report.z_crit, color="#1f77b4", alpha=0.10)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.plot(table["z"], ypos, "o", color="#1f77b4")
        ax.set_yticks(ypos)
        ax.set_yticklabels(table["g"], fontsize=9)
        ax.set_xlabel("z")
        status = "passed" if report.passed else f"{report.n_flagged} flagged"
        ax.set_title(f"Geweke joint distribution test ({report.kernel} kernel): {status}")
        fig.tight_layout()
        if save:
            fig.savefig(save, bbox_inches="tight")
        if show:  # pragma: no cover - interactive path
            plt.show()
    return fig
