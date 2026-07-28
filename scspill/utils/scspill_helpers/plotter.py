"""Plotting for fitted SCSPILL results.

One entry point, :func:`plot_scspill`, with panel kinds mirroring the R
package's ``plot.scspill`` types:

* ``"full"`` -- observed vs. posterior-mean counterfactual with the credible
  band (R type ``"full"``);
* ``"effect"`` -- the per-period treatment-effect path with its credible band
  (R type ``"effect"``);
* ``"spill_top"`` -- small-multiple spillover paths of the top donors by
  posterior mean magnitude (R type ``"spill_top"``);
* ``"weights"`` -- posterior-mean ``alpha`` against the classical simplex-SCM
  weights (R type ``"weights"``);
* ``"rho"`` -- posterior histogram of the spillover intensity (R type
  ``"rho"``);
* ``"trace"`` -- the ``rho`` chain trace (R type ``"trace"``);
* ``"panel"`` -- the default 1x3 composite ``full | effect | spill_top``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...exceptions import ScspillPlottingError
from ..plotting import Plotter, scspill_style

if TYPE_CHECKING:  # pragma: no cover
    from .structures import SCSPILLResults

_KINDS = ("panel", "full", "effect", "spill_top", "weights", "rho", "trace")


def plot_scspill(
    results: SCSPILLResults,
    *,
    kind: str = "panel",
    top_n: int = 8,
    save: str | None = None,
    display: bool | None = None,
    ax: Any = None,
    **overrides: Any,
) -> Any:
    """Plot a fitted SCSPILL result.

    Parameters
    ----------
    results : SCSPILLResults
        A fitted result.
    kind : str, default "panel"
        One of ``"panel"``, ``"full"``, ``"effect"``, ``"spill_top"``,
        ``"weights"``, ``"rho"``, ``"trace"``.
    top_n : int, default 8
        Number of donors in the ``"spill_top"`` panel (ranked by mean
        post-treatment ``|spillover|``).
    save : str, optional
        File path to save the figure to.
    display : bool, optional
        Show the figure; defaults to the ``display`` of the result's stored
        plot config.
    ax : matplotlib Axes, optional
        Existing axis for single-axis kinds (``"full"``, ``"effect"``,
        ``"rho"``, ``"trace"``, ``"weights"``).
    **overrides
        Cosmetic overrides applied over the stored plot config.

    Returns
    -------
    matplotlib.axes.Axes or np.ndarray of Axes

    Raises
    ------
    ScspillPlottingError
        On an unknown ``kind`` or a plotting failure.
    """
    if kind not in _KINDS:
        raise ScspillPlottingError(f"Unknown plot kind {kind!r}; expected one of {_KINDS}.")
    import matplotlib.pyplot as plt

    pc = results.plot_config
    if pc is not None and overrides:
        pc = pc.model_copy(update=overrides)
    theme = getattr(pc, "theme", None)
    if display is None:
        display = bool(getattr(pc, "display", True))

    try:
        with scspill_style(theme):
            plotter = Plotter.from_config(pc) if pc is not None else Plotter()
            if kind == "panel":
                fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
                _draw_full(results, plotter, axes[0])
                _draw_effect(results, plotter, axes[1])
                _draw_spill_top(results, axes[2], top_n)
                fig.tight_layout()
                out = axes
            elif kind == "spill_top":
                fig, ax_grid = _draw_spill_grid(results, top_n)
                out = ax_grid
            else:
                if ax is None:
                    fig, ax = plt.subplots(figsize=(8, 5))
                else:
                    fig = ax.figure
                if kind == "full":
                    _draw_full(results, plotter, ax)
                elif kind == "effect":
                    _draw_effect(results, plotter, ax)
                elif kind == "weights":
                    _draw_weights(results, ax, top_n=max(top_n, 10))
                elif kind == "rho":
                    _draw_rho(results, ax)
                elif kind == "trace":
                    _draw_trace(results, ax)
                out = ax

            if save:
                fig.savefig(save, bbox_inches="tight")
            if display:
                plt.show()
        return out
    except ScspillPlottingError:
        raise
    except Exception as exc:  # pragma: no cover - defensive translation
        raise ScspillPlottingError(f"SCSPILL plot ({kind}) failed: {exc}") from exc


def _draw_full(results: SCSPILLResults, plotter: Plotter, ax) -> None:
    """Observed vs. counterfactual with the credible band."""
    inputs = results.inputs
    eff = results.effects_detail
    plotter.observed_vs_counterfactual(
        inputs.time_labels,
        inputs.Y0,
        eff.cf_mean,
        treated_label=str(inputs.treated_label),
        intervention=inputs.time_labels[inputs.T0],
        interval=(eff.cf_lower, eff.cf_upper),
        interval_label=f"{round(eff.ci_level * 100)}% credible interval",
        outcome="Outcome",
        time="Time",
        title="Observed vs. synthetic counterfactual",
        ax=ax,
    )


def _draw_effect(results: SCSPILLResults, plotter: Plotter, ax) -> None:
    """Per-period treatment-effect path with its credible band."""
    inputs = results.inputs
    eff = results.effects_detail
    plotter.gap(
        inputs.time_labels,
        inputs.Y0 - eff.cf_mean,
        intervention=inputs.time_labels[inputs.T0],
        interval=(inputs.Y0 - eff.cf_upper, inputs.Y0 - eff.cf_lower),
        interval_label=f"{round(eff.ci_level * 100)}% credible interval",
        outcome="Treatment effect",
        time="Time",
        title="Treatment effect on the treated",
        ax=ax,
    )


def _top_spill_labels(results: SCSPILLResults, top_n: int) -> list:
    """Donors ranked by mean post-treatment absolute spillover."""
    inputs = results.inputs
    spill = results.effects_detail.spill_mean.iloc[inputs.T0 :]
    ranking = spill.abs().mean(axis=0).sort_values(ascending=False)
    return list(ranking.index[:top_n])


def _draw_spill_top(results: SCSPILLResults, ax, top_n: int) -> None:
    """Overlaid spillover paths of the top donors (composite-panel version)."""
    inputs = results.inputs
    spill = results.effects_detail.spill_mean
    for lab in _top_spill_labels(results, top_n):
        ax.plot(inputs.time_labels, spill[lab].to_numpy(), linewidth=1.2, label=str(lab))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(inputs.time_labels[inputs.T0], color="grey", linestyle="-", linewidth=1.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Spillover effect")
    ax.set_title(f"Top {top_n} spillover paths")
    ax.legend(fontsize=8, ncol=2)


def _draw_spill_grid(results: SCSPILLResults, top_n: int):
    """Small-multiple grid of spillover paths with credible bands."""
    import matplotlib.pyplot as plt

    inputs = results.inputs
    eff = results.effects_detail
    labels = _top_spill_labels(results, top_n)
    ncols = min(4, max(1, len(labels)))
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows), sharex=True, squeeze=False
    )
    for k, lab in enumerate(labels):
        ax = axes[k // ncols][k % ncols]
        ax.fill_between(
            inputs.time_labels,
            eff.spill_lower[lab].to_numpy(),
            eff.spill_upper[lab].to_numpy(),
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(inputs.time_labels, eff.spill_mean[lab].to_numpy(), linewidth=1.2)
        ax.axhline(0.0, color="black", linewidth=0.6)
        ax.axvline(inputs.time_labels[inputs.T0], color="grey", linewidth=0.8)
        ax.set_title(str(lab), fontsize=9)
    for k in range(len(labels), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    fig.suptitle("Spillover effects on donors", fontsize=11)
    fig.tight_layout()
    return fig, axes


def _draw_weights(results: SCSPILLResults, ax, top_n: int = 15) -> None:
    """Posterior-mean alpha vs. classical simplex-SCM weights."""
    labels = [str(u) for u in results.inputs.control_labels]
    alpha = results.alpha_hat
    order = np.argsort(-np.abs(alpha))[:top_n][::-1]
    ypos = np.arange(order.size)
    ax.barh(ypos, alpha[order], color="#1f77b4", alpha=0.85, label="SCSPILL (posterior mean)")
    if results.scm_weights:
        scm = np.array([results.scm_weights.get(labels[j], 0.0) for j in order])
        ax.plot(scm, ypos, "o", color="crimson", markersize=5, label="Classical SCM")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels([labels[j] for j in order], fontsize=8)
    ax.set_xlabel("Weight")
    ax.set_title("Donor weights")
    ax.legend(fontsize=9)


def _draw_rho(results: SCSPILLResults, ax) -> None:
    """Posterior histogram of rho with mean and credible-interval markers."""
    rho = results.rho_draws
    ax.hist(rho, bins=40, color="#1f77b4", alpha=0.8, density=True)
    ax.axvline(results.rho_hat, color="black", linestyle="--", linewidth=1.2, label="Mean")
    lo, hi = results.rho_ci
    ax.axvline(lo, color="grey", linestyle=":", linewidth=1.0)
    ax.axvline(hi, color="grey", linestyle=":", linewidth=1.0, label="Credible interval")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("Posterior density")
    ax.set_title("Spillover intensity posterior")
    ax.legend(fontsize=9)


def _draw_trace(results: SCSPILLResults, ax) -> None:
    """Trace of the rho chain."""
    rho = results.rho_draws
    ax.plot(np.arange(rho.size), rho, linewidth=0.6, color="#1f77b4")
    ax.axhline(results.rho_hat, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Draw")
    ax.set_ylabel(r"$\rho$")
    ax.set_title(f"rho trace (acceptance {results.acc_rho:.0%}, ESS {results.rho_ess:.0f})")
