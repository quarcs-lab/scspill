"""Cao--Dowd synthetic control with spillovers and end-of-sample inference.

Implements Cao and Dowd, "Estimation and Inference for Synthetic Control
Methods with Spillover Effects" (arXiv:1902.07343v3). The defaults are the
paper's Section 4 definitions:

* critical values are the left inverse of the empirical CDF of the
  pre-treatment reference statistics (``quantile_method="empirical"``);
* pointwise intervals invert the squared P test, so they are symmetric
  about the estimate (``ci_method="inversion"``);
* reference statistics come from the full-sample SCM fit
  (``reference_mode="full"``).

The authors' Proposition 99 application code (jcao0/synthetic-control-
spillover at commit 86004020) differs in two places, kept as switches so
its published numbers can be reproduced: the linearly interpolated sample
quantile (``quantile_method="linear"``) and intervals that add the signed
pre-treatment error quantiles to the estimate (``ci_method="published"``).
Both intervals are always stored; ``effect_ci`` is the one selected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from scipy.optimize import minimize

from ...config_models import (
    BaseEstimatorResults,
    InferenceResults,
    MethodDetailsResults,
)
from ...exceptions import ScspillConfigError, ScspillDataError, ScspillEstimationError
from ..results_helpers import build_effect_submodels, make_weights_results

if TYPE_CHECKING:  # pragma: no cover
    from .config import SCSPILLConfig


def validate_effect_matrix(
    effect_matrix: pd.DataFrame, units: list[Any], treated: list[Any]
) -> pd.DataFrame:
    """Check a labeled exposure matrix ``A`` and return it in ``units`` row order.

    Rows must be exactly the unit labels and columns unique and nonempty, with
    finite entries. Each treated unit's row must be its own direct-effect
    indicator: a single nonzero entry, equal to 1, in a column that is zero on
    every other row. Anything else would mix a spillover into the treated
    unit's effect, so the headline counterfactual would no longer be the
    direct effect's.

    Raises
    ------
    ScspillConfigError
        On any violation.
    """
    a = effect_matrix
    if a.index.has_duplicates or set(a.index) != set(units):
        raise ScspillConfigError("effect_matrix needs exactly one row per unit label.")
    if a.columns.has_duplicates or a.shape[1] == 0:
        raise ScspillConfigError("effect_matrix needs unique, nonempty columns.")
    a = a.loc[list(units)].astype(float)
    values = a.to_numpy()
    if not np.isfinite(values).all():
        raise ScspillConfigError("effect_matrix must be finite.")
    rows = {label: k for k, label in enumerate(units)}
    for label in treated:
        row = values[rows[label]]
        support = np.flatnonzero(row != 0)
        if support.size != 1 or row[support[0]] != 1:
            raise ScspillConfigError(
                f"effect_matrix row {label!r} (a treated unit) must be a direct-effect "
                "indicator: a single 1 and zeros elsewhere."
            )
        column = values[:, support[0]]
        if np.count_nonzero(column) != 1:
            raise ScspillConfigError(
                f"effect_matrix column {a.columns[support[0]]!r} is the direct effect of "
                f"{label!r}, so it must be zero for every other unit."
            )
    return a


def _empirical_quantile(values: np.ndarray, probability: float) -> float:
    """Left inverse of the empirical CDF, as defined in v3 Section 4."""
    x = np.sort(np.asarray(values, dtype=float).reshape(-1))
    return float(x[max(0, int(np.ceil(probability * len(x))) - 1)])


def _critical_value(values: np.ndarray, probability: float, method: str) -> float:
    """Critical value under ``quantile_method`` (see ``SCSPILLConfig.quantile_method``)."""
    if method == "empirical":
        return _empirical_quantile(values, probability)
    if method == "linear":
        return float(np.quantile(values, probability))
    raise ScspillConfigError(f"Unknown quantile_method {method!r}.")


class CDTestResult(BaseModel):
    """A P test with its pre-intervention reference distribution."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    statistic: float
    critical_value: float
    p_value: float
    reject: bool
    reference_statistics: np.ndarray


class CDResults(BaseEstimatorResults):
    """Cao--Dowd fit, including effects, SCM equations, and P-test results."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    method: Literal["cd"] = "cd"
    treated_label: Any
    unit_labels: tuple[Any, ...]
    time_labels: np.ndarray
    t_pre: int
    intercepts: pd.Series
    scm_matrix: pd.DataFrame
    effect_matrix: pd.DataFrame
    effects_panel: pd.DataFrame
    spillover_panel: pd.DataFrame
    pre_residuals: pd.DataFrame
    placebo_effects: pd.DataFrame
    pointwise_p_values: pd.DataFrame
    pointwise_ci: dict[Any, pd.DataFrame]
    published_ci: dict[Any, pd.DataFrame]
    kappa_test: dict[Any, CDTestResult]
    rank: int
    condition_number: float
    reference_mode: str
    quantile_method: Literal["linear", "empirical"]
    ci_method: Literal["published", "inversion"]
    ci_level: float

    @property
    def effect_ci(self) -> dict[Any, pd.DataFrame]:
        """Per-unit pointwise intervals selected by ``ci_method``."""
        return self.published_ci if self.ci_method == "published" else self.pointwise_ci

    def _critical(self, refs: np.ndarray, quantile_method: str | None) -> float:
        return _critical_value(refs, self.ci_level, quantile_method or self.quantile_method)

    def _parse_c_d(
        self, C: np.ndarray | pd.DataFrame, d: np.ndarray | float | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate the contrast matrix ``C`` and null vector ``d``."""
        if isinstance(C, pd.DataFrame):
            if set(C.columns) != set(self.unit_labels) or C.columns.has_duplicates:
                raise ScspillConfigError("C columns must match the unit labels exactly.")
            c = C.loc[:, list(self.unit_labels)].to_numpy(dtype=float)
        else:
            c = np.atleast_2d(np.asarray(C, dtype=float))
        if (
            c.ndim != 2
            or c.shape[0] == 0
            or c.shape[1] != len(self.unit_labels)
            or not np.isfinite(c).all()
        ):
            raise ScspillConfigError("C must be a finite matrix with one column per unit.")
        q = c.shape[0]
        try:
            null = np.zeros(q) if d is None else np.broadcast_to(np.asarray(d, dtype=float), (q,))
        except ValueError as exc:
            raise ScspillConfigError("d must have one entry per row of C.") from exc
        if not np.isfinite(null).all():
            raise ScspillConfigError("d must be finite.")
        return c, null

    def test_hypothesis(
        self,
        C: np.ndarray | pd.DataFrame,
        *,
        d: np.ndarray | float | None = None,
        period: Any | None = None,
        weight: np.ndarray | Literal["inverse_covariance"] | None = None,
        quantile_method: Literal["empirical", "linear"] | None = None,
    ) -> CDTestResult:
        """Test ``C @ effect_t = d`` by the pre-intervention P distribution.

        ``C`` has one column per unit in ``unit_labels`` order, or labeled
        columns when passed as a DataFrame. ``period`` is a post-period label.
        The default statistic is the squared Euclidean norm; an explicit
        positive-definite weight or the inverse placebo second moment is allowed.
        ``quantile_method`` overrides the fit's critical-value rule.
        """
        if period is None:
            period = self.effects_panel.index[0]
        if period not in self.effects_panel.index:
            raise ScspillConfigError(f"Unknown post-treatment period {period!r}.")
        c, null = self._parse_c_d(C, d)
        q = c.shape[0]
        pre = c @ self.placebo_effects.to_numpy(dtype=float).T
        if weight is None:
            w = np.eye(q)
        elif isinstance(weight, str):
            if weight != "inverse_covariance" or pre.shape[1] <= q:
                raise ScspillConfigError(
                    "Inverse covariance requires more reference periods than rows of C."
                )
            second_moment = pre @ pre.T / pre.shape[1]
            try:
                w = np.linalg.inv(np.atleast_2d(second_moment))
            except np.linalg.LinAlgError as exc:
                raise ScspillConfigError("Projected placebo second moment is singular.") from exc
        else:
            w = np.asarray(weight, dtype=float)
        if w.shape != (q, q) or not np.isfinite(w).all() or not np.allclose(w, w.T):
            raise ScspillConfigError("weight must be a finite symmetric square matrix.")
        if np.linalg.eigvalsh(w).min() <= 0:
            raise ScspillConfigError("weight must be positive definite.")
        observed = c @ self.effects_panel.loc[period].to_numpy(dtype=float) - null
        statistic = float(observed @ w @ observed)
        refs = np.einsum("it,ij,jt->t", pre, w, pre)
        critical = self._critical(refs, quantile_method)
        return CDTestResult(
            statistic=statistic,
            critical_value=critical,
            p_value=float(np.mean(refs >= statistic)),
            reject=bool(statistic > critical),
            reference_statistics=refs,
        )


def _prepare(
    config: SCSPILLConfig,
) -> tuple[np.ndarray, tuple[Any, ...], np.ndarray, int, Any, pd.DataFrame]:
    """Validate and arrange the balanced panel as units by periods."""
    df = config.df
    if df[[config.unitid, config.time]].isna().any().any():
        raise ScspillDataError("Unit and time labels cannot be missing.")
    if df.duplicated([config.unitid, config.time]).any():
        raise ScspillDataError("Panel has duplicate unit-time observations.")
    if df[[config.outcome, config.treat]].isna().any().any():
        raise ScspillDataError("Outcomes and treatment flags must be complete.")
    units = list(pd.unique(df[config.unitid]))
    times = np.asarray(sorted(pd.unique(df[config.time])))
    y = df.pivot(index=config.unitid, columns=config.time, values=config.outcome).reindex(
        index=units, columns=times
    )
    treat = df.pivot(index=config.unitid, columns=config.time, values=config.treat).reindex(
        index=units, columns=times
    )
    if y.isna().any().any() or treat.isna().any().any():
        raise ScspillDataError("Cao--Dowd requires a balanced panel.")
    if not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ScspillDataError("Outcomes must be finite.")
    flag = treat.to_numpy(dtype=float)
    if not np.isin(flag, [0, 1]).all():
        raise ScspillDataError("Treatment flags must be binary.")
    treated = np.flatnonzero(flag.sum(axis=1))
    if len(treated) != 1:
        raise ScspillDataError("Exactly one treated unit is required.")
    idx = int(treated[0])
    t0 = int(np.argmax(flag[idx] == 1))
    if t0 < 3 or t0 >= len(times) or not np.all(flag[idx, t0:] == 1):
        raise ScspillDataError("Treatment must begin after at least three pre periods and persist.")
    label = units.pop(idx)
    units.insert(0, label)
    yy = y.loc[units].to_numpy(dtype=float)
    if len(units) < 3:
        raise ScspillDataError("At least three units are required for all-unit SCM.")
    if config.spillover_structure == "custom":
        assert config.effect_matrix is not None  # enforced by SCSPILLConfig
        a = validate_effect_matrix(config.effect_matrix, units, [label])
    else:
        affected = list(config.affected_units or [])
        selected = [label, *affected]
        a = pd.DataFrame(0.0, index=units, columns=selected)
        for x in selected:
            a.loc[x, x] = 1.0
    return yy, tuple(units), times, t0, label, a


def _scm_all(y_pre: np.ndarray, *, maxiter: int, ftol: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit an intercept and simplex donor weights for every unit."""
    n = y_pre.shape[0]
    intercepts = np.zeros(n)
    b_matrix = np.zeros((n, n))
    for i in range(n):
        target = y_pre[i]
        donors = np.delete(y_pre, i, axis=0).T
        centered_y = target - target.mean()
        centered_x = donors - donors.mean(axis=0)

        # Scale-free objective with analytic derivatives. SLSQP is the primary
        # solve; projected gradient is the fallback if it fails or is infeasible.
        scale = max(float(np.std(centered_y)), 1.0)
        scaled_y = centered_y / scale
        scaled_x = centered_x / scale

        def objective(
            candidate: np.ndarray,
            target: np.ndarray = scaled_y,
            donor_matrix: np.ndarray = scaled_x,
        ) -> float:
            return float(np.sum((target - donor_matrix @ candidate) ** 2))

        def gradient(
            candidate: np.ndarray,
            target: np.ndarray = scaled_y,
            donor_matrix: np.ndarray = scaled_x,
        ) -> np.ndarray:
            return 2 * donor_matrix.T @ (donor_matrix @ candidate - target)

        fit = minimize(
            objective,
            np.full(n - 1, 1 / (n - 1)),
            jac=gradient,
            method="SLSQP",
            bounds=[(0, None)] * (n - 1),
            constraints=[
                {"type": "eq", "fun": lambda b: b.sum() - 1, "jac": lambda b: np.ones(n - 1)}
            ],
            options={"maxiter": maxiter, "ftol": ftol},
        )
        weights = np.clip(fit.x, 0, None)
        if (
            not fit.success
            or not np.isfinite(weights).all()
            or weights.sum() <= 0
            or abs(weights.sum() - 1) > 1e-4
        ):
            weights = _simplex_fista(centered_y, centered_x)
        weights /= weights.sum()
        intercepts[i] = target.mean() - donors.mean(axis=0) @ weights
        b_matrix[i] = np.insert(weights, i, 0.0)
    return intercepts, b_matrix


def _simplex_project(values: np.ndarray) -> np.ndarray:
    """Project one vector onto the nonnegative unit simplex."""
    ordered = np.sort(values)[::-1]
    thresholds = (np.cumsum(ordered) - 1) / np.arange(1, len(values) + 1)
    active = np.flatnonzero(ordered > thresholds)
    if active.size == 0:
        raise ScspillEstimationError("Simplex projection has no active coordinate.")
    projected = np.maximum(values - thresholds[active[-1]], 0)
    return projected / projected.sum()


def _simplex_fista(target: np.ndarray, donors: np.ndarray) -> np.ndarray:
    """Solve the centered SCM problem by restarted projected gradient."""
    n = donors.shape[1]
    current = np.full(n, 1 / n)
    extrapolated = current.copy()
    momentum = 1.0
    lipschitz = max(2 * float(np.linalg.norm(donors, ord=2)) ** 2, np.finfo(float).eps)
    previous_loss = float("inf")
    for _ in range(50_000):
        gradient = 2 * donors.T @ (donors @ extrapolated - target)
        candidate = _simplex_project(extrapolated - gradient / lipschitz)
        loss = float(np.sum((target - donors @ candidate) ** 2))
        if loss > previous_loss + 1e-13:
            extrapolated = current.copy()
            momentum = 1.0
            gradient = 2 * donors.T @ (donors @ extrapolated - target)
            candidate = _simplex_project(extrapolated - gradient / lipschitz)
            loss = float(np.sum((target - donors @ candidate) ** 2))
        candidate_gradient = 2 * donors.T @ (donors @ candidate - target)
        fixed_point = _simplex_project(candidate - candidate_gradient / lipschitz)
        if np.max(np.abs(candidate - fixed_point)) <= 1e-9:
            return candidate
        next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        extrapolated = candidate + (momentum - 1) / next_momentum * (candidate - current)
        current = candidate
        momentum = next_momentum
        previous_loss = loss
    raise ScspillEstimationError("Simplex SCM did not converge after 50,000 fallback iterations.")


def _projection(b: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Build the effect map G and residual-space projector after rank checks."""
    r = np.eye(len(b)) - b
    ra = r @ a
    singular = np.linalg.svd(ra, compute_uv=False)
    tol = max(ra.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.sum(singular > tol))
    if rank != a.shape[1]:
        raise ScspillEstimationError(
            f"Cao--Dowd effects unidentified: rank((I-B)A)={rank} < {a.shape[1]}."
        )
    pinv = np.linalg.pinv(ra)
    g = a @ pinv
    projector = ra @ pinv
    # The paper reports the condition number of A'(I-B)'(I-B)A, whose
    # eigenvalue ratio is the square of the singular-value ratio of (I-B)A.
    return g, projector, rank, float((singular[0] / singular[-1]) ** 2)


def fit_cd(config: SCSPILLConfig) -> CDResults:
    """Fit Cao--Dowd and compute pointwise and specification-test references."""
    y, labels, times, t0, treated, a_df = _prepare(config)
    a_matrix = a_df.to_numpy(dtype=float)
    intercepts, b = _scm_all(y[:, :t0], maxiter=config.scm_maxiter, ftol=config.scm_ftol)
    r = np.eye(len(labels)) - b
    g, projector, rank, condition = _projection(b, a_matrix)
    residuals = r @ y[:, :t0] - intercepts[:, None]
    post_target = r @ y[:, t0:] - intercepts[:, None]
    alpha = g @ post_target
    if config.reference_mode == "full":
        placebo = g @ residuals
        kappa_pre_vectors = (np.eye(len(labels)) - projector) @ residuals
    else:
        placebo = np.empty((len(labels), t0))
        kappa_pre_vectors = np.empty((len(labels), t0))
        for held in range(t0):
            kept = np.arange(t0) != held
            fold_a, fold_b = _scm_all(
                y[:, :t0][:, kept], maxiter=config.scm_maxiter, ftol=config.scm_ftol
            )
            fold_g, fold_projector, _, _ = _projection(fold_b, a_matrix)
            fold_resid = (np.eye(len(labels)) - fold_b) @ y[:, held] - fold_a
            placebo[:, held] = fold_g @ fold_resid
            kappa_pre_vectors[:, held] = (np.eye(len(labels)) - fold_projector) @ fold_resid
    kappa_post_vectors = (np.eye(len(labels)) - projector) @ post_target
    kappa_ref = np.linalg.norm(kappa_pre_vectors, axis=0)
    kappa_post = np.linalg.norm(kappa_post_vectors, axis=0)
    post_times = times[t0:]
    effects = pd.DataFrame(alpha.T, index=post_times, columns=labels)
    placebo_df = pd.DataFrame(placebo.T, index=times[:t0], columns=labels)
    residual_df = pd.DataFrame(residuals.T, index=times[:t0], columns=labels)
    pointwise = {}
    published = {}
    pvalues = pd.DataFrame(index=post_times, columns=labels, dtype=float)
    for j, label in enumerate(labels):
        pre = placebo[j]
        threshold = np.sqrt(_critical_value(pre**2, config.ci, config.quantile_method))
        pointwise[label] = pd.DataFrame(
            {"lower": alpha[j] - threshold, "upper": alpha[j] + threshold}, index=post_times
        )
        qlo, qhi = np.quantile(pre, [(1 - config.ci) / 2, (1 + config.ci) / 2])
        published[label] = pd.DataFrame(
            {"lower": alpha[j] + qlo, "upper": alpha[j] + qhi}, index=post_times
        )
        pvalues[label] = np.mean(pre[:, None] ** 2 >= alpha[j][None, :] ** 2, axis=0)
    kappa = {}
    critical_kappa = _critical_value(kappa_ref, config.ci, config.quantile_method)
    for period, statistic in zip(post_times, kappa_post, strict=True):
        kappa[period] = CDTestResult(
            statistic=float(statistic),
            critical_value=critical_kappa,
            p_value=float(np.mean(kappa_ref >= statistic)),
            reject=bool(statistic > critical_kappa),
            reference_statistics=kappa_ref.copy(),
        )
    selected_ci = published if config.ci_method == "published" else pointwise
    cf = np.r_[intercepts[0] + b[0] @ y[:, :t0], y[0, t0:] - alpha[0]]
    pre_spill = np.zeros((t0, len(labels) - 1))
    spill = pd.DataFrame(np.vstack([pre_spill, alpha[1:].T]), index=times, columns=list(labels[1:]))
    weights = make_weights_results(
        {labels[j]: b[0, j] for j in range(1, len(labels))},
        "intercept-adjusted simplex SCM",
    )
    submodels = build_effect_submodels(
        observed_outcome=y[0],
        counterfactual_outcome=cf,
        n_pre_periods=t0,
        n_post_periods=len(times) - t0,
        time_periods=times,
        intervention_time=times[t0],
        weights=weights,
        inference=InferenceResults(
            method=f"cao_dowd_P_{config.reference_mode}",
            confidence_level=config.ci,
            details="Pointwise and joint inference is in CDResults; no ATT interval is implied.",
        ),
        prediction_interval={
            "lower": y[0, t0:] - selected_ci[treated]["upper"].to_numpy(),
            "upper": y[0, t0:] - selected_ci[treated]["lower"].to_numpy(),
            "periods": post_times,
            "level": config.ci,
            "kind": (
                "published_signed_quantile"
                if config.ci_method == "published"
                else "pointwise_P_inversion"
            ),
        },
    )
    return CDResults(
        **submodels,
        method_details=MethodDetailsResults(
            method_name="SCSPILL/cd",
            is_recommended=True,
            parameters_used={
                "method": "cd",
                "reference_mode": config.reference_mode,
                "quantile_method": config.quantile_method,
                "ci_method": config.ci_method,
                "ci": config.ci,
                "scm_maxiter": config.scm_maxiter,
                "scm_ftol": config.scm_ftol,
            },
        ),
        treated_label=treated,
        unit_labels=labels,
        time_labels=times,
        t_pre=t0,
        intercepts=pd.Series(intercepts, index=labels),
        scm_matrix=pd.DataFrame(b, index=labels, columns=labels),
        effect_matrix=a_df,
        effects_panel=effects,
        spillover_panel=spill,
        pre_residuals=residual_df,
        placebo_effects=placebo_df,
        pointwise_p_values=pvalues,
        pointwise_ci=pointwise,
        published_ci=published,
        kappa_test=kappa,
        rank=rank,
        condition_number=condition,
        reference_mode=config.reference_mode,
        quantile_method=config.quantile_method,
        ci_method=config.ci_method,
        ci_level=config.ci,
        plot_config=config.resolved_plot(),
    )
