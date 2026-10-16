"""End-of-sample P-test inference for the Cao--Dowd model.

Andrews' (2003) end-of-sample instability test, generalized by Cao & Dowd
(arXiv:1902.07343v3, Section 4) to hypotheses ``C alpha_t = d`` under
spillovers. A post-treatment statistic is compared with the same quadratic
form evaluated on the pre-treatment residuals, the "reference statistics";
inverting the test gives pointwise intervals. The ``kappa_A`` statistic
(Section 5.1) applies the same reference logic to the part of the residuals
that the assumed exposure structure cannot explain.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd

from ....exceptions import ScspillConfigError, ScspillEstimationError
from .estimation import effect_operator
from .setup import build_exposure_matrix, validate_effect_matrix
from .structures import CDFit, CDTestResult, KappaSelection


def empirical_quantile(values: np.ndarray, probability: float) -> float:
    """Left inverse of the empirical CDF, as defined in v3 Section 4."""
    x = np.sort(np.asarray(values, dtype=float).reshape(-1))
    return float(x[max(0, int(np.ceil(probability * len(x))) - 1)])


def critical_value(values: np.ndarray, probability: float, method: str) -> float:
    """Critical value under ``quantile_method`` (see ``SCSPILLConfig``).

    ``"empirical"`` is the paper's left inverse of the empirical CDF;
    ``"linear"`` is NumPy's linearly interpolated quantile, used by the
    authors' Proposition 99 application code.
    """
    if method == "empirical":
        return empirical_quantile(values, probability)
    if method == "linear":
        return float(np.quantile(values, probability))
    raise ScspillConfigError(f"Unknown quantile_method {method!r}.")


def p_test(
    statistic: float,
    references: np.ndarray,
    *,
    level: float,
    quantile_method: str,
) -> CDTestResult:
    """Compare one statistic with its pre-treatment reference distribution.

    The P value is the share of reference statistics at least as large as the
    observed one; the test rejects when the statistic exceeds the
    ``level``-quantile of the references.
    """
    refs = np.asarray(references, dtype=float)
    critical = critical_value(refs, level, quantile_method)
    return CDTestResult(
        statistic=float(statistic),
        critical_value=critical,
        p_value=float(np.mean(refs >= statistic)),
        reject=bool(statistic > critical),
        reference_statistics=refs,
    )


def quadratic_test(
    c: np.ndarray,
    null: np.ndarray,
    effect: np.ndarray,
    placebo: np.ndarray,
    *,
    weight: np.ndarray | str | None,
    level: float,
    quantile_method: str,
) -> CDTestResult:
    """Test ``C alpha_t = d`` with the statistic ``(C alpha_t - d)' W (C alpha_t - d)``.

    Parameters
    ----------
    c : np.ndarray
        Contrast matrix, shape ``(q, n)``.
    null : np.ndarray
        Null value ``d``, shape ``(q,)``.
    effect : np.ndarray
        Estimated unit effects of one post-treatment period, shape ``(n,)``.
    placebo : np.ndarray
        Pre-treatment placebo effects ``G u_t``, shape ``(n, T0)``.
    weight : array, "inverse_covariance", or None
        ``W``: the identity by default, an explicit positive-definite matrix,
        or the inverse second moment of the projected placebo effects.
    level, quantile_method
        Test level and critical-value rule.
    """
    q = c.shape[0]
    pre = c @ placebo
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
    observed = c @ effect - null
    refs = np.einsum("it,ij,jt->t", pre, w, pre)
    return p_test(
        float(observed @ w @ observed), refs, level=level, quantile_method=quantile_method
    )


def parse_contrast(
    C: np.ndarray | pd.DataFrame,
    d: np.ndarray | float | None,
    unit_labels: tuple[Any, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the contrast matrix ``C`` and null vector ``d``.

    ``C`` has one column per unit in ``unit_labels`` order, or labeled
    columns when passed as a DataFrame.
    """
    if isinstance(C, pd.DataFrame):
        if set(C.columns) != set(unit_labels) or C.columns.has_duplicates:
            raise ScspillConfigError("C columns must match the unit labels exactly.")
        c = C.loc[:, list(unit_labels)].to_numpy(dtype=float)
    else:
        c = np.atleast_2d(np.asarray(C, dtype=float))
    if c.ndim != 2 or c.shape[0] == 0 or c.shape[1] != len(unit_labels) or not np.isfinite(c).all():
        raise ScspillConfigError("C must be a finite matrix with one column per unit.")
    q = c.shape[0]
    try:
        null = np.zeros(q) if d is None else np.broadcast_to(np.asarray(d, dtype=float), (q,))
    except ValueError as exc:
        raise ScspillConfigError("d must have one entry per row of C.") from exc
    if not np.isfinite(null).all():
        raise ScspillConfigError("d must be finite.")
    return c, null


def pointwise_intervals(
    alpha: np.ndarray,
    placebo: np.ndarray,
    *,
    labels: tuple[Any, ...],
    periods: np.ndarray,
    level: float,
    quantile_method: str,
) -> tuple[dict[Any, pd.DataFrame], dict[Any, pd.DataFrame]]:
    """Per-unit pointwise intervals: by test inversion, and as published.

    The inversion interval is ``alpha_t +/- sqrt(c)``, with ``c`` the critical
    value of the squared placebo effects. The published interval adds the
    signed placebo quantiles to the estimate, as the authors' application
    code does.

    Returns
    -------
    inversion, published : dict
        ``{unit label: DataFrame(lower, upper) indexed by post period}``.
    """
    inversion: dict[Any, pd.DataFrame] = {}
    published: dict[Any, pd.DataFrame] = {}
    for j, label in enumerate(labels):
        pre = placebo[j]
        threshold = np.sqrt(critical_value(pre**2, level, quantile_method))
        inversion[label] = pd.DataFrame(
            {"lower": alpha[j] - threshold, "upper": alpha[j] + threshold}, index=periods
        )
        qlo, qhi = np.quantile(pre, [(1 - level) / 2, (1 + level) / 2])
        published[label] = pd.DataFrame(
            {"lower": alpha[j] + qlo, "upper": alpha[j] + qhi}, index=periods
        )
    return inversion, published


def pointwise_p_values(
    alpha: np.ndarray, placebo: np.ndarray, *, labels: tuple[Any, ...], periods: np.ndarray
) -> pd.DataFrame:
    """P value of ``alpha_{i,t} = 0`` for every unit and post period."""
    out = pd.DataFrame(index=periods, columns=list(labels), dtype=float)
    for j, label in enumerate(labels):
        out[label] = np.mean(placebo[j][:, None] ** 2 >= alpha[j][None, :] ** 2, axis=0)
    return out


def kappa_tests(
    pre_vectors: np.ndarray,
    post_vectors: np.ndarray,
    *,
    periods: np.ndarray,
    level: float,
    quantile_method: str,
) -> dict[Any, CDTestResult]:
    """Run the ``kappa_A`` structure test in every post-treatment period.

    The statistic is the norm of the residual part that ``A`` cannot explain,
    ``(I - P_A)(R Y_t - a)``; the references are the same norm on the
    pre-treatment residuals.
    """
    refs = np.linalg.norm(pre_vectors, axis=0)
    stats = np.linalg.norm(post_vectors, axis=0)
    return {
        period: p_test(float(stat), refs.copy(), level=level, quantile_method=quantile_method)
        for period, stat in zip(periods, stats, strict=True)
    }


def reference_residuals(
    y: np.ndarray,
    a_matrix: np.ndarray,
    *,
    t0: int,
    intercepts: np.ndarray,
    b: np.ndarray,
    mode: str,
    scm: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-treatment placebo effects and ``kappa_A`` reference vectors.

    ``"full"`` evaluates the full-sample fit on every pre-treatment period;
    ``"leave_one_period_out"`` refits every SCM equation without each period
    and evaluates the refit on the held-out one.

    Returns
    -------
    placebo : np.ndarray
        ``G u_t``, shape ``(n, T0)``.
    kappa_pre : np.ndarray
        ``(I - P_A) u_t``, shape ``(n, T0)``.
    """
    n = len(a_matrix)
    eye = np.eye(n)
    if mode == "full":
        op = effect_operator(b, a_matrix)
        residuals = (eye - b) @ y[:, :t0] - intercepts[:, None]
        return op.unit_map @ residuals, (eye - op.projector) @ residuals
    if mode != "leave_one_period_out":
        raise ScspillConfigError(f"Unknown reference_mode {mode!r}.")
    if scm is None:
        raise ScspillConfigError("leave_one_period_out needs an SCM solver.")
    placebo = np.empty((n, t0))
    kappa_pre = np.empty((n, t0))
    for held in range(t0):
        kept = np.arange(t0) != held
        fold_a, fold_b = scm(y[:, :t0][:, kept])
        fold_op = effect_operator(fold_b, a_matrix)
        fold_resid = (eye - fold_b) @ y[:, held] - fold_a
        placebo[:, held] = fold_op.unit_map @ fold_resid
        kappa_pre[:, held] = (eye - fold_op.projector) @ fold_resid
    return placebo, kappa_pre


#: Keys a structure specification may hold in :func:`select_structure_by_kappa`.
_SPEC_KEYS = {"spillover_structure", "affected_units", "unit_distances", "effect_matrix"}


def select_structure_by_kappa(
    fit: CDFit,
    candidates: Mapping[str, pd.DataFrame | Mapping[str, Any]],
    *,
    statistic: Literal["v3", "mean_norm"] = "v3",
) -> KappaSelection:
    """Choose the exposure structure with the smallest ``kappa_A`` statistic.

    Cao & Dowd v3 Section 5.1.2 proposes ``argmin_A kappa_A`` over a set of
    candidate structures, and Section S.1.3 extends it to several
    post-treatment periods, where the choice becomes consistent. The SCM
    equations of ``fit`` are reused; only ``A`` changes.

    Parameters
    ----------
    fit : CDFit
        A fitted ``cd`` model.
    candidates : mapping
        ``{name: A}`` with ``A`` a labeled exposure matrix, or ``{name:
        spec}`` with ``spec`` a dict of ``spillover_structure`` (default
        ``"per_unit"``), ``affected_units``, ``unit_distances`` and
        ``effect_matrix``, as in ``SCSPILLConfig``.
    statistic : {"v3", "mean_norm"}, default "v3"
        ``"v3"`` is the paper's ``||sum_s (I - Gamma_A) u_s|| / sqrt(T1)``
        (Section S.1.3); ``"mean_norm"`` averages the per-period norms, as
        mlsynth's ``select_A_by_kappa`` does.

    Returns
    -------
    KappaSelection

    Raises
    ------
    ScspillConfigError
        On an invalid or unidentified candidate.
    """
    if statistic not in ("v3", "mean_norm"):
        raise ScspillConfigError(f"Unknown statistic {statistic!r}; use 'v3' or 'mean_norm'.")
    if not candidates:
        raise ScspillConfigError("Pass at least one candidate structure.")
    inputs = fit.inputs
    labels = list(inputs.unit_labels)
    b = fit.scm_matrix.loc[labels, labels].to_numpy(dtype=float)
    a = fit.intercepts.loc[labels].to_numpy(dtype=float)
    eye = np.eye(len(labels))
    post = (eye - b) @ inputs.Y_units[:, inputs.T0 :] - a[:, None]
    stats: dict[str, float] = {}
    matrices: dict[str, pd.DataFrame] = {}
    for name, candidate in candidates.items():
        if isinstance(candidate, pd.DataFrame):
            matrix = validate_effect_matrix(candidate, labels, inputs.treated_labels)
        elif isinstance(candidate, Mapping):
            unknown = set(candidate) - _SPEC_KEYS
            if unknown:
                raise ScspillConfigError(f"Candidate {name!r} has unknown keys {sorted(unknown)}.")
            spec = {"spillover_structure": "per_unit", **candidate}
            try:
                matrix = build_exposure_matrix(inputs, **spec)
            except ScspillConfigError as exc:
                raise ScspillConfigError(f"Candidate {name!r}: {exc}") from exc
        else:
            raise ScspillConfigError(
                f"Candidate {name!r} must be a labeled DataFrame or a structure dict."
            )
        try:
            op = effect_operator(b, matrix.to_numpy(dtype=float))
        except ScspillEstimationError as exc:
            raise ScspillConfigError(f"Candidate {name!r}: {exc}") from exc
        unexplained = (eye - op.projector) @ post
        if statistic == "v3":
            stats[name] = float(np.linalg.norm(unexplained.sum(axis=1)) / np.sqrt(post.shape[1]))
        else:
            stats[name] = float(np.linalg.norm(unexplained, axis=0).mean())
        matrices[name] = matrix
    series = pd.Series(stats, name=f"kappa_A ({statistic})")
    return KappaSelection(
        statistics=series, best=str(series.idxmin()), statistic=statistic, matrices=matrices
    )


__all__ = [
    "critical_value",
    "empirical_quantile",
    "kappa_tests",
    "p_test",
    "parse_contrast",
    "pointwise_intervals",
    "pointwise_p_values",
    "quadratic_test",
    "reference_residuals",
    "select_structure_by_kappa",
]
