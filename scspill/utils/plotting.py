"""Shared plotting primitives for scspill.

A small, style-carrying :class:`Plotter` with one method per common plot
archetype, adapted from ``mlsynth.utils.plotting`` so scspill figures render
with the same house style. Module-specific plotter helpers build their figures
by calling these archetype methods -- passing an existing Matplotlib ``Axes``
so they can compose multi-panel layouts -- and then add any bespoke panels of
their own.

The class carries the *style* (colors, intervention-line look, figure size) so
that every module routed through it renders consistently, while leaving figure
lifecycle (creation, saving, showing) to the caller.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Union

import numpy as np

_DEFAULT_CF_COLORS = ("red", "blue", "green", "purple", "orange", "brown")

CounterfactualLike = Union[np.ndarray, Sequence[np.ndarray]]  # noqa: UP007 - runtime alias


def _coerce_time_axis(times):
    """Parse date-like time labels to datetimes so Matplotlib uses a date axis.

    Returns ``(values_for_plot, is_datetime)``; numeric labels (e.g. year
    integers) and un-parseable strings are returned unchanged with
    ``is_datetime=False``.
    """
    arr = np.asarray(times)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr, True
    if arr.dtype == object or arr.dtype.kind in ("U", "S"):
        try:
            import pandas as pd

            return pd.to_datetime(arr).to_numpy(), True
        except Exception:  # not date-like -> leave categorical
            return arr, False
    return arr, False


def _coerce_time_scalar(value, is_datetime):
    """Coerce a single time (e.g. an intervention marker) onto the datetime axis."""
    if not is_datetime or value is None:
        return value
    try:
        import pandas as pd

        return pd.to_datetime(value).to_numpy()
    except Exception:  # pragma: no cover - marker stays as-is
        return value


def _style_time_axis(ax, is_datetime, n_ticks):
    """Keep the x-axis legible when there are many periods."""
    import matplotlib.pyplot as plt

    if is_datetime:
        import matplotlib.dates as mdates

        loc = mdates.AutoDateLocator(minticks=4, maxticks=9)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    elif n_ticks > 16:
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))
    if n_ticks > 16 and not is_datetime:
        ax.tick_params(axis="x", labelrotation=30)


#: Preferred display font; falls back through the ``font.sans-serif`` chain
#: below if it is not installed. Override by reassigning before plotting.
SCSPILL_FONT = "Inter"

#: The default scspill plot style (Matplotlib rcParams). Applied per-plot via
#: :func:`scspill_style` (scoped, no global side effects) or globally via
#: :func:`apply_scspill_style`. Identical to mlsynth's house style.
SCSPILL_RC: dict[str, Any] = {
    # Figure
    "figure.dpi": 100,
    "savefig.dpi": 100,
    "figure.facecolor": "white",
    # Grid
    "axes.grid": True,
    "grid.linestyle": "-",
    "grid.alpha": 0.40,
    "grid.color": "#1428A0",
    # Typography
    "font.family": "sans-serif",
    "font.sans-serif": [SCSPILL_FONT, "Arial", "Helvetica", "DejaVu Sans"],
    "font.weight": "medium",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "axes.labelweight": "medium",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    # Legend
    "legend.frameon": True,
    "legend.framealpha": 1,
    "legend.facecolor": "white",
    "legend.edgecolor": "#DDDDDD",
    "legend.fontsize": 12,
    # Lines
    "lines.linewidth": 1,
    "lines.antialiased": True,
    # Clean axes (modern slide look)
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
}


def scspill_style(theme: dict | str | None = None):
    """Context manager applying the scspill plot style to plots created within.

    Scoped via ``matplotlib.rc_context`` so the user's global rcParams are
    untouched. Wrap the *entire* plotting block (figure creation included), as
    rcParams are read when artists are created::

        with scspill_style():
            fig, ax = plt.subplots()
            ...

    Parameters
    ----------
    theme : dict or str, optional
        ``None`` (default) uses the scspill house style :data:`SCSPILL_RC`.
        A dict of rcParams is merged *over* the house style. A string is
        treated as a named Matplotlib style (``plt.style.context``).
    """
    import matplotlib.pyplot as plt

    if theme is None:
        return plt.rc_context(SCSPILL_RC)  # type: ignore[arg-type]
    if isinstance(theme, str):
        return plt.style.context(theme)
    return plt.rc_context({**SCSPILL_RC, **theme})  # type: ignore[arg-type,dict-item]


def apply_scspill_style() -> None:
    """Apply the scspill style globally (mutates ``matplotlib.rcParams``)."""
    import matplotlib as mpl

    mpl.rcParams.update(SCSPILL_RC)  # type: ignore[arg-type]


class Plotter:
    """Render common synthetic-control plot archetypes with a consistent style.

    Parameters
    ----------
    treated_color : str, default "black"
        Color of the observed (treated) series.
    counterfactual_colors : str or sequence of str, optional
        Color cycle for counterfactual lines. A single string is accepted.
        Defaults to a six-color cycle.
    intervention_color : str, default "grey"
        Color of the vertical intervention marker.
    figsize : tuple, default (7, 5)
        Figure size used only when a method must create its own figure.
    treated_linewidth : float, default 1.6
        Line width of the observed series.
    treated_linestyle : str, default "-"
        Matplotlib linestyle of the observed series.
    counterfactual_linewidth : float, default 1.4
        Line width of counterfactual series.
    counterfactual_linestyle : str, default "--"
        Matplotlib linestyle of counterfactual series.
    """

    def __init__(
        self,
        *,
        treated_color: str = "black",
        counterfactual_colors: str | Sequence[str] | None = None,
        intervention_color: str = "grey",
        figsize: tuple = (7, 5),
        treated_linewidth: float = 1.6,
        treated_linestyle: str = "-",
        counterfactual_linewidth: float = 1.4,
        counterfactual_linestyle: str = "--",
    ) -> None:
        self.treated_color = treated_color
        if counterfactual_colors is None:
            counterfactual_colors = list(_DEFAULT_CF_COLORS)
        elif isinstance(counterfactual_colors, str):
            counterfactual_colors = [counterfactual_colors]
        self.counterfactual_colors = list(counterfactual_colors)
        self.intervention_color = intervention_color
        self.figsize = figsize
        self.treated_linewidth = treated_linewidth
        self.treated_linestyle = treated_linestyle
        self.counterfactual_linewidth = counterfactual_linewidth
        self.counterfactual_linestyle = counterfactual_linestyle

    @classmethod
    def from_config(cls, pc: Any) -> Plotter:
        """Build a Plotter from a ``PlotConfig`` (duck-typed; no import).

        Reads the cosmetic fields off ``pc`` so plotting stays decoupled from
        :mod:`scspill.config_models`. Missing attributes fall back to defaults.
        """
        cf = getattr(pc, "counterfactual_colors", None)
        return cls(
            treated_color=getattr(pc, "observed_color", "black"),
            counterfactual_colors=cf if cf else None,
            intervention_color=getattr(pc, "intervention_color", "grey"),
            treated_linewidth=getattr(pc, "observed_linewidth", 1.6),
            treated_linestyle=getattr(pc, "observed_linestyle", "-"),
            counterfactual_linewidth=getattr(pc, "counterfactual_linewidth", 1.4),
            counterfactual_linestyle=getattr(pc, "counterfactual_linestyle", "--"),
        )

    @staticmethod
    def _as_counterfactual_list(counterfactuals: CounterfactualLike) -> list[np.ndarray]:
        """Normalize a single series, a 2D array of columns, or a list to a list."""
        if isinstance(counterfactuals, np.ndarray):
            if counterfactuals.ndim == 1:
                return [counterfactuals]
            return [counterfactuals[:, i] for i in range(counterfactuals.shape[1])]
        return [np.asarray(cf) for cf in counterfactuals]

    def observed_vs_counterfactual(
        self,
        times: Any,
        observed: np.ndarray,
        counterfactuals: CounterfactualLike,
        *,
        labels: Sequence[str] | None = None,
        treated_label: Any = "Treated",
        intervention: Any | None = None,
        interval: Any = None,
        interval_label: str = "Prediction interval",
        outcome: str = "",
        time: str = "",
        title: str = "Observed vs. counterfactual",
        ax: Any = None,
    ) -> Any:
        """Plot the observed series against one or more counterfactual series.

        The workhorse archetype: a treated/observed line plus any number of
        dashed counterfactual lines, with an optional vertical intervention
        marker and an optional shaded interval band around the first
        counterfactual. Pass ``ax`` to draw into an existing subplot (for
        multi-panel figures); otherwise a standalone figure is created.

        Parameters
        ----------
        times : sequence
            X-axis values (time labels), length ``T``.
        observed : np.ndarray
            Observed treated outcome, shape ``(T,)``.
        counterfactuals : np.ndarray or sequence of np.ndarray
            One counterfactual ``(T,)``, a 2D array of column-counterfactuals,
            or a list of ``(T,)`` arrays.
        labels : sequence of str, optional
            Legend labels for the counterfactuals (one per series).
        treated_label : Any, default "Treated"
            Identifier shown for the observed series.
        intervention : Any, optional
            X position of the intervention marker; omitted if None.
        interval : sequence of np.ndarray, optional
            ``(lower, upper)`` band (each shape ``(T,)``) shaded around the
            first counterfactual. ``NaN`` entries are not shaded.
        interval_label : str, default "Prediction interval"
            Legend label for the shaded band.
        outcome, time : str
            Axis labels.
        title : str
            Panel title.
        ax : matplotlib Axes, optional
            Existing axis to draw on. If None, a new figure/axis is created.

        Returns
        -------
        matplotlib.axes.Axes
            The axis drawn on (``ax.figure`` gives the figure).
        """
        import matplotlib.pyplot as plt

        cfs = self._as_counterfactual_list(counterfactuals)
        if labels is None:
            labels = (
                ["Synthetic"]
                if len(cfs) == 1
                else [f"Counterfactual {i + 1}" for i in range(len(cfs))]
            )

        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)

        times, _is_dt = _coerce_time_axis(times)
        intervention = _coerce_time_scalar(intervention, _is_dt)
        ax.plot(
            times,
            np.asarray(observed).reshape(-1),
            color=self.treated_color,
            linewidth=self.treated_linewidth,
            linestyle=self.treated_linestyle,
            label=f"Treated ({treated_label})",
        )
        for i, cf in enumerate(cfs):
            color = self.counterfactual_colors[i % len(self.counterfactual_colors)]
            ax.plot(
                times,
                np.asarray(cf).reshape(-1),
                linewidth=self.counterfactual_linewidth,
                linestyle=self.counterfactual_linestyle,
                color=color,
                label=labels[i],
            )
        band = self.counterfactual_colors[0]
        if interval is not None:
            lower = np.asarray(interval[0], dtype=float).reshape(-1)
            upper = np.asarray(interval[1], dtype=float).reshape(-1)
            ax.fill_between(
                times,
                lower,
                upper,
                where=~np.isnan(lower),
                color=band,
                alpha=0.18,
                linewidth=0,
                label=interval_label,
            )
        if intervention is not None:
            ax.axvline(
                intervention,
                color=self.intervention_color,
                linestyle="-",
                linewidth=1.2,
                label="Intervention",
            )
        ax.set_xlabel(time)
        ax.set_ylabel(outcome)
        ax.set_title(title)
        ax.legend()  # frame styling comes from the active rcParams
        _style_time_axis(ax, _is_dt, len(times))
        return ax

    def gap(
        self,
        times: Any,
        gap: np.ndarray,
        *,
        intervention: Any | None = None,
        interval: Any = None,
        interval_label: str = "Prediction interval",
        outcome: str = "",
        time: str = "",
        title: str = "Estimated gap",
        color: str | None = None,
        ax: Any = None,
    ) -> Any:
        """Plot the per-period effect ``tau_t`` (gap) with a zero reference line.

        Draws the gap series against a horizontal zero line and an optional
        vertical intervention marker -- the standard companion to the
        observed-vs-counterfactual panel. Pass ``interval`` as a
        ``(lower, upper)`` pair (each shape ``(T,)``) to shade an interval
        around the gap; ``NaN`` entries are left unshaded.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)
        times, _is_dt = _coerce_time_axis(times)
        intervention = _coerce_time_scalar(intervention, _is_dt)
        line_color = color or self.counterfactual_colors[0]
        if interval is not None:
            lower = np.asarray(interval[0], dtype=float).reshape(-1)
            upper = np.asarray(interval[1], dtype=float).reshape(-1)
            ax.fill_between(
                times,
                lower,
                upper,
                where=~np.isnan(lower),
                color=line_color,
                alpha=0.18,
                linewidth=0,
                label=interval_label,
            )
        ax.plot(
            times,
            np.asarray(gap).reshape(-1),
            color=line_color,
            linewidth=self.counterfactual_linewidth,
            label="Gap",
        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        if intervention is not None:
            ax.axvline(
                intervention,
                color=self.intervention_color,
                linestyle="-",
                linewidth=1.2,
                label="Intervention",
            )
        ax.set_xlabel(time)
        ax.set_ylabel(outcome or "Treated - counterfactual")
        ax.set_title(title)
        ax.legend()
        _style_time_axis(ax, _is_dt, len(times))
        return ax
