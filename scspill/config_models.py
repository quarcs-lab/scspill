"""Shared configuration bases and standardized result models for scspill.

Adapted from ``mlsynth.config_models`` so that scspill estimators expose the
same configuration and result surface as mlsynth estimators: a pydantic config
inheriting :class:`BaseEstimatorConfig`, and a result object subclassing
:class:`BaseEstimatorResults` that populates the standardized sub-models
(:class:`EffectsResults`, :class:`TimeSeriesResults`, :class:`WeightsResults`,
:class:`InferenceResults`, :class:`FitDiagnosticsResults`,
:class:`MethodDetailsResults`).
"""

import re
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from scspill.exceptions import ScspillDataError, ScspillPlottingError

_DEFAULT_CF_COLORS = ["red", "blue", "green", "purple", "orange", "brown"]


class PlotConfig(BaseModel):
    """Cosmetic configuration for an estimator's plots.

    The single, nested home for everything a user might tune about a figure --
    colors, line thickness and pattern, axis labels, title, and theme -- so the
    look is fully orchestrated from the top config with sensible defaults the
    user never has to touch. Consumed by :meth:`BaseEstimatorResults.plot` and
    the shared :class:`scspill.utils.plotting.Plotter`.
    """

    # Observed (treated) series.
    observed_color: str = Field(
        default="black", description="Color of the observed (treated) line."
    )
    observed_linewidth: float = Field(default=1.6, description="Width of the observed line.")
    observed_linestyle: str = Field(
        default="-", description="Matplotlib linestyle for the observed line."
    )
    # Counterfactual series (cycled / broadcast across multiple counterfactuals).
    counterfactual_colors: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_CF_COLORS),
        description="Color cycle for counterfactual line(s).",
    )
    counterfactual_linewidth: float = Field(
        default=1.4, description="Width of counterfactual line(s)."
    )
    counterfactual_linestyle: str = Field(
        default="--", description="Matplotlib linestyle for counterfactual line(s)."
    )
    # Reference lines.
    show_intervention_line: bool = Field(
        default=True, description="Draw a vertical line at the intervention."
    )
    intervention_color: str = Field(
        default="grey", description="Color of the intervention reference line."
    )
    # Labels / title (None => a sensible default derived from the data/method).
    xlabel: str | None = Field(default=None, description="X-axis label override.")
    ylabel: str | None = Field(default=None, description="Y-axis label override.")
    title: str | None = Field(default=None, description="Plot title override.")
    # Theme: a dict of rcParams merged over the house style, or a named style.
    theme: dict | str | None = Field(
        default=None,
        description="Custom theme: rcParams dict (merged over the scspill style) or a named Matplotlib style.",
    )
    # Lifecycle.
    display: bool = Field(default=True, description="Show the figure.")
    save: bool | str | dict = Field(
        default=False, description="Save config: False, a filename, or a dict."
    )

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"


class BaseEstimatorConfig(BaseModel):
    """Base pydantic model for estimator configurations.

    Includes the common fields required by every panel-data effect estimator:
    the long panel ``df`` and the names of its outcome / treatment-indicator /
    unit / time columns, plus display and plot cosmetics.
    """

    df: pd.DataFrame = Field(..., description="Input panel data as a pandas DataFrame.")
    outcome: str = Field(..., description="Name of the outcome variable column in the DataFrame.")
    treat: str = Field(..., description="Name of the treatment indicator column in the DataFrame.")
    unitid: str = Field(..., description="Name of the unit identifier column in the DataFrame.")
    time: str = Field(..., description="Name of the time period column in the DataFrame.")
    display_graphs: bool = Field(default=True, description="Whether to display plots of results.")
    save: bool | str = Field(
        default=False,
        description=(
            "Configuration for saving plots. If False (default), plots are not saved. "
            "If True, plots are saved with default names. If a string, it's used as "
            "the base filename for saved plots."
        ),
    )
    counterfactual_color: list[str] = Field(
        default_factory=lambda: ["red"],
        description="Color(s) for counterfactual line(s) in plots. (Legacy; prefer `plot`.)",
    )
    treated_color: str = Field(
        default="black",
        description="Color for the treated unit line in plots. (Legacy; prefer `plot`.)",
    )
    plot: PlotConfig = Field(
        default_factory=PlotConfig,
        description="Nested cosmetic plot configuration (colors, line styles, labels, theme).",
    )

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    def resolved_plot(self) -> "PlotConfig":
        """Effective :class:`PlotConfig`, folding in the legacy flat fields.

        If the user supplied a custom ``plot`` it is authoritative; otherwise
        the legacy ``treated_color`` / ``counterfactual_color`` are mapped into
        a fresh ``PlotConfig``. The behavioral ``display_graphs`` / ``save``
        always apply unless overridden on ``plot``.
        """
        if self.plot != PlotConfig():
            pc = self.plot.model_copy()
        else:
            pc = PlotConfig(
                observed_color=self.treated_color,
                counterfactual_colors=list(self.counterfactual_color),
            )
        if pc.display is True:  # default untouched -> honor legacy flag
            pc.display = self.display_graphs
        if pc.save is False:  # default untouched -> honor legacy flag
            pc.save = self.save
        return pc

    @model_validator(mode="after")
    def check_df_and_columns(cls, values: Any) -> Any:
        """Validate that ``df`` is non-empty and holds the named columns."""
        df = values.df
        outcome = values.outcome
        treat = values.treat
        unitid = values.unitid
        time = values.time

        if df.empty:
            raise ScspillDataError("Input DataFrame 'df' cannot be empty.")

        required_columns = {outcome, treat, unitid, time}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ScspillDataError(
                f"Missing required columns in DataFrame 'df': {', '.join(sorted(missing_columns))}"
            )
        return values


# --- Pydantic models for standardized estimator results ---


class ScspillResult(BaseModel):
    """Common base for every ``fit()`` return value in scspill.

    The observational report family (:class:`BaseEstimatorResults`, aliased as
    :class:`EffectResult`) measures a treatment effect on already-observed
    data: ATT, counterfactual, weights, inference. ``isinstance(result,
    ScspillResult)`` always holds for a fitted result, so library behaviour is
    simple and predictable.
    """

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"
        json_encoders = {
            np.ndarray: lambda arr: (
                [None if pd.isna(x) else x for x in arr.tolist()] if arr is not None else None
            )
        }


class EffectsResults(BaseModel):
    """Standardized model for reporting treatment effects."""

    att: float | None = Field(default=None, description="Average Treatment Effect on the Treated.")
    att_percent: float | None = Field(
        default=None, description="Percentage Average Treatment Effect on the Treated."
    )
    att_std_err: float | None = Field(
        default=None, description="Standard error of the ATT estimate."
    )
    additional_effects: dict[str, Any] | None = Field(
        default=None, description="Dictionary for other estimator-specific effects."
    )

    class Config:
        extra = "allow"  # Allow other effect measures to be added dynamically


class FitDiagnosticsResults(BaseModel):
    """Standardized model for reporting goodness-of-fit diagnostics."""

    rmse_pre: float | None = Field(
        default=None, description="Root Mean Squared Error in the pre-treatment period."
    )
    r_squared_pre: float | None = Field(
        default=None, description="R-squared value in the pre-treatment period."
    )
    rmse_post: float | None = Field(
        default=None,
        description="Root Mean Squared Error in the post-treatment period (often std of post-treatment gap).",
    )
    additional_metrics: dict[str, Any] | None = Field(
        default=None, description="Dictionary for other fit metrics."
    )

    class Config:
        extra = "allow"


class TimeSeriesResults(BaseModel):
    """Standardized model for reporting key time series vectors."""

    observed_outcome: np.ndarray | None = Field(
        default=None, description="Observed outcome vector for the treated unit."
    )
    counterfactual_outcome: np.ndarray | None = Field(
        default=None, description="Estimated counterfactual outcome vector."
    )
    estimated_gap: np.ndarray | None = Field(
        default=None, description="Estimated treatment effect vector (observed - counterfactual)."
    )
    time_periods: np.ndarray | None = Field(
        default=None, description="Array of time periods corresponding to the series."
    )
    intervention_time: Any | None = Field(
        default=None,
        description="Time label at the pre/post boundary, for the plot's intervention reference line.",
    )
    # Canonical per-period prediction interval on the counterfactual: aligned to
    # ``time_periods``, NaN where the method has no band there. Populate via
    # :func:`scspill.utils.results_helpers.build_effect_submodels`'s
    # ``prediction_interval`` argument, not by hand.
    counterfactual_lower: np.ndarray | None = Field(
        default=None,
        description="Per-period pointwise lower bound on the counterfactual (NaN where absent).",
    )
    counterfactual_upper: np.ndarray | None = Field(
        default=None,
        description="Per-period pointwise upper bound on the counterfactual (NaN where absent).",
    )
    counterfactual_lower_simultaneous: np.ndarray | None = Field(
        default=None,
        description="Per-period simultaneous (joint-coverage) lower bound on the counterfactual.",
    )
    counterfactual_upper_simultaneous: np.ndarray | None = Field(
        default=None,
        description="Per-period simultaneous (joint-coverage) upper bound on the counterfactual.",
    )
    prediction_interval_level: float | None = Field(
        default=None, description="Nominal coverage of the band (e.g. 0.90 for a 90% interval)."
    )
    prediction_interval_kind: str | None = Field(
        default=None, description="Provenance/type tag for the band, e.g. 'bayesian', 'conformal'."
    )

    @property
    def has_prediction_interval(self) -> bool:
        """True when a per-period pointwise band is present and not all-NaN."""
        lo = self.counterfactual_lower
        if lo is None:
            return False
        return bool(np.any(np.isfinite(np.asarray(lo, dtype=float))))

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


class WeightsResults(BaseModel):
    """Standardized model for reporting estimator weights.

    ``weights`` are not one thing across synthetic-control methods, so this
    model exposes the real variety as optional faces; an estimator populates
    whichever apply:

    * ``donor_weights`` -- ``{donor label: weight}`` (most SCMs);
    * ``time_weights``  -- ``{period: weight}``;
    * ``unit_weights``  -- a weight matrix / array.
    """

    donor_weights: dict[str, float] | None = Field(
        default=None, description="Dictionary mapping donor unit names/IDs to their weights."
    )
    time_weights: dict[Any, float] | None = Field(
        default=None, description="Dictionary mapping time periods to weights."
    )
    unit_weights: np.ndarray | None = Field(
        default=None,
        description="Unit weight matrix/array for estimators whose weights are not a donor mapping.",
    )
    summary_stats: dict[str, Any] | None = Field(
        default=None, description="Summary statistics about weights (e.g., cardinality)."
    )

    @property
    def weight_vector(self) -> np.ndarray | None:
        """The donor/control weights as a dense :class:`numpy.ndarray`.

        A computed view, not a stored copy: the values of ``donor_weights`` in
        their existing key order (use ``list(donor_weights)`` for the aligned
        labels). ``None`` when no donor weights are present.
        """
        if self.donor_weights is None:
            return None
        return np.asarray(list(self.donor_weights.values()), dtype=float)

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


class InferenceResults(BaseModel):
    """Standardized model for reporting statistical inference results."""

    p_value: float | None = Field(default=None, description="P-value for the estimated ATT.")
    ci_lower: float | None = Field(
        default=None, description="Lower bound of the confidence interval for ATT."
    )
    ci_upper: float | None = Field(
        default=None, description="Upper bound of the confidence interval for ATT."
    )
    standard_error: float | None = Field(
        default=None, description="Standard error of the ATT estimate."
    )
    confidence_level: float | None = Field(
        default=None, description="Confidence level used for the CI (e.g., 0.95 for 95%)."
    )
    method: str | None = Field(
        default=None,
        description="Method used for inference (e.g., 'bayesian_posterior', 'placebo').",
    )
    details: Any | None = Field(
        default=None,
        description="More detailed inference results, e.g., posterior summaries or draw arrays.",
    )

    @property
    def se(self) -> float | None:
        """Short read-only alias for :attr:`standard_error`."""
        return self.standard_error

    @property
    def ci(self) -> tuple[float | None, float | None]:
        """Read-only ``(lower, upper)`` view of :attr:`ci_lower` / :attr:`ci_upper`."""
        return (self.ci_lower, self.ci_upper)

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


class MethodDetailsResults(BaseModel):
    """Standardized model for reporting details about the estimation method/variant used."""

    method_name: str | None = Field(
        default=None, description="Name of the specific method or variant used."
    )
    is_recommended: bool | None = Field(
        default=None, description="Flag indicating if this method variant was recommended."
    )
    parameters_used: dict[str, Any] | None = Field(
        default=None, description="Key parameters used for this specific result set."
    )

    class Config:
        extra = "allow"


class BaseEstimatorResults(ScspillResult):
    """The observational report: standardized result of an effect estimator.

    Aliased as :class:`EffectResult`. Carries the standardized sub-models
    (:class:`EffectsResults`, :class:`TimeSeriesResults`,
    :class:`WeightsResults`, :class:`InferenceResults`,
    :class:`FitDiagnosticsResults`) plus flat convenience accessors (``att``,
    ``att_ci``, ``counterfactual``, ``gap``, ``donor_weights``, ``pre_rmse``)
    so every effect estimator exposes one predictable surface.
    """

    effects: EffectsResults | None = None
    fit_diagnostics: FitDiagnosticsResults | None = None
    time_series: TimeSeriesResults | None = None
    weights: WeightsResults | None = None
    inference: InferenceResults | None = None
    method_details: MethodDetailsResults | None = None

    sub_method_results: dict[str, Any] | None = Field(
        default=None, description="Results for sub-methods or variants."
    )
    additional_outputs: dict[str, Any] | None = Field(
        default=None, description="Dictionary for any other outputs specific to the estimator."
    )
    raw_results: dict[str, Any] | None = Field(
        default=None,
        exclude=True,
        description="Original raw results dictionary from the estimator's core logic.",
    )
    execution_summary: dict[str, Any] | None = Field(
        default=None, description="Summary of execution, including any errors or warnings."
    )
    plot_config: PlotConfig | None = Field(
        default=None,
        exclude=True,
        description="Resolved PlotConfig captured at fit time; drives plot().",
    )

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"
        json_encoders = {
            np.ndarray: lambda arr: (
                [None if pd.isna(x) else x for x in arr.tolist()] if arr is not None else None
            )
        }

    # ------------------------------------------------------------------
    # Flat convenience accessors -- the minimum read contract every effect
    # estimator satisfies, delegating to the standardized sub-models.
    # ------------------------------------------------------------------
    @property
    def att(self) -> float | None:
        """Average treatment effect on the treated."""
        return self.effects.att if self.effects else None

    @property
    def att_ci(self) -> tuple | None:
        """``(lower, upper)`` confidence interval for the ATT, if available."""
        inf = self.inference
        if inf is not None and inf.ci_lower is not None and inf.ci_upper is not None:
            return (inf.ci_lower, inf.ci_upper)
        return None

    @property
    def counterfactual(self) -> np.ndarray | None:
        """Estimated counterfactual outcome path."""
        return self.time_series.counterfactual_outcome if self.time_series else None

    @property
    def gap(self) -> np.ndarray | None:
        """Estimated gap (observed minus counterfactual)."""
        return self.time_series.estimated_gap if self.time_series else None

    @property
    def donor_weights(self) -> dict[str, float] | None:
        """Donor weights ``{label: weight}``."""
        return self.weights.donor_weights if self.weights else None

    @property
    def weight_vector(self) -> np.ndarray | None:
        """Donor/control weights as a dense array (see :attr:`WeightsResults.weight_vector`)."""
        return self.weights.weight_vector if self.weights else None

    @property
    def pre_rmse(self) -> float | None:
        """Pre-treatment root-mean-squared fit error."""
        return self.fit_diagnostics.rmse_pre if self.fit_diagnostics else None

    # ------------------------------------------------------------------
    # Standard plotting -- one entry point for every effect estimator,
    # driven by the standardized ``time_series`` sub-model and the resolved
    # ``PlotConfig`` captured at fit time. Bespoke estimators override.
    # ------------------------------------------------------------------
    def plot(self, kind: str = "auto", *, ax: Any = None, **overrides: Any) -> Any:
        """Render the standard effect plot from the standardized result.

        Parameters
        ----------
        kind : {"auto", "counterfactual", "gap"}, default "auto"
            ``"auto"``/``"counterfactual"`` draw observed vs. counterfactual;
            ``"gap"`` draws the per-period effect.
        ax : matplotlib Axes, optional
            Draw into an existing axis (multi-panel composition).
        **overrides
            Per-call cosmetic overrides applied over the stored PlotConfig
            (e.g. ``title=...``, ``observed_color=...``).

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        from .utils.plotting import Plotter, scspill_style

        ts = self.time_series
        if ts is None or ts.observed_outcome is None:
            raise ScspillPlottingError("Result has no time_series.observed_outcome to plot.")

        pc = self.plot_config or PlotConfig()
        pc = pc.model_copy(update=overrides) if overrides else pc

        observed = np.asarray(ts.observed_outcome).reshape(-1)
        times = (
            np.asarray(ts.time_periods)
            if ts.time_periods is not None
            else np.arange(observed.shape[0])
        )
        intervention = ts.intervention_time if pc.show_intervention_line else None
        method = self.method_details.method_name if self.method_details else None
        xlabel = pc.xlabel if pc.xlabel is not None else "Time"
        ylabel = pc.ylabel if pc.ylabel is not None else "Outcome"

        with scspill_style(pc.theme):
            plotter = Plotter.from_config(pc)
            if kind in ("auto", "counterfactual"):
                cf = ts.counterfactual_outcome
                if cf is None:
                    raise ScspillPlottingError("Result has no counterfactual_outcome to plot.")
                interval = None
                if ts.has_prediction_interval:
                    interval = (ts.counterfactual_lower, ts.counterfactual_upper)
                ax = plotter.observed_vs_counterfactual(
                    times,
                    observed,
                    np.asarray(cf).reshape(-1),
                    treated_label=method or "Treated",
                    intervention=intervention,
                    interval=interval,
                    interval_label=(
                        f"{round((ts.prediction_interval_level or 0.95) * 100)}% credible interval"
                        if ts.prediction_interval_kind == "bayesian"
                        else "Prediction interval"
                    ),
                    outcome=ylabel,
                    time=xlabel,
                    title=pc.title
                    or (
                        f"{method}: observed vs. counterfactual"
                        if method
                        else "Observed vs. counterfactual"
                    ),
                    ax=ax,
                )
            elif kind == "gap":
                ax = plotter.gap(
                    times,
                    np.asarray(ts.estimated_gap).reshape(-1),
                    intervention=intervention,
                    outcome=ylabel,
                    time=xlabel,
                    title=pc.title or "Estimated gap",
                    ax=ax,
                )
            else:
                raise ScspillPlottingError(f"Unknown plot kind {kind!r}.")

            fig = ax.figure
            if pc.save:
                # `method` is the method_details name, which carries a
                # "<estimator>/<model>" separator -- slugify it so the default
                # filename never reads as a path into a missing directory.
                slug = re.sub(r"[^\w.-]+", "-", method or "scspill")
                fname = pc.save if isinstance(pc.save, str) else f"{slug}_plot.png"
                fig.savefig(fname, bbox_inches="tight")
            if pc.display:
                plt.show()
        return ax


# ``EffectResult`` is the canonical, intention-revealing name for the
# observational report; ``BaseEstimatorResults`` matches the mlsynth alias.
EffectResult = BaseEstimatorResults


# ---------------------------------------------------------------------------
# Lazy re-exports of per-estimator configs (PEP 562), matching mlsynth's
# convention: configs live next to their helper packages and are re-exported
# from this module so ``from scspill.config_models import SCSPILLConfig`` works.
# ---------------------------------------------------------------------------
_RELOCATED_CONFIGS = {
    "SCSPILLConfig": "scspill.utils.scspill_helpers.config",
}


def __getattr__(name: str):  # PEP 562 module-level attribute hook
    module_path = _RELOCATED_CONFIGS.get(name)
    if module_path is not None:
        import importlib

        return getattr(importlib.import_module(module_path), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
