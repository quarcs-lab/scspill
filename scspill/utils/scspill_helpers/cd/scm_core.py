"""Intercept-adjusted simplex synthetic control, one equation per unit.

Cao & Dowd (arXiv:1902.07343v3, Section 3) fit, for every unit ``i``, an
intercept ``a_i`` and simplex weights ``b_i`` over all other units on the
pre-treatment outcomes. Centering absorbs the intercept, so each equation is
a simplex-constrained least-squares problem. SLSQP with analytic gradients on
a scale-free objective is the primary solve; restarted projected gradient
(FISTA) on the simplex is the fallback when SLSQP fails or leaves the
simplex.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ....exceptions import ScspillEstimationError


def fit_simplex_scm(
    target: np.ndarray, donors: np.ndarray, *, maxiter: int, ftol: float
) -> tuple[float, np.ndarray]:
    """Fit one intercept-adjusted simplex SCM equation.

    Parameters
    ----------
    target : np.ndarray
        Pre-treatment outcomes of the unit being fitted, shape ``(T0,)``.
    donors : np.ndarray
        Pre-treatment outcomes of its donors, shape ``(T0, J)``.
    maxiter, ftol : int, float
        SLSQP iteration limit and tolerance.

    Returns
    -------
    intercept : float
    weights : np.ndarray
        Nonnegative donor weights summing to one, shape ``(J,)``.
    """
    n_donors = donors.shape[1]
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
        np.full(n_donors, 1 / n_donors),
        jac=gradient,
        method="SLSQP",
        bounds=[(0, None)] * n_donors,
        constraints=[
            {"type": "eq", "fun": lambda b: b.sum() - 1, "jac": lambda b: np.ones(n_donors)}
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
    return float(target.mean() - donors.mean(axis=0) @ weights), weights


def fit_all_unit_scm(
    y_pre: np.ndarray, *, maxiter: int, ftol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an intercept and simplex donor weights for every unit.

    Parameters
    ----------
    y_pre : np.ndarray
        Pre-treatment outcomes, units by periods, shape ``(n, T0)``.
    maxiter, ftol : int, float
        SLSQP iteration limit and tolerance.

    Returns
    -------
    intercepts : np.ndarray
        Shape ``(n,)``.
    b_matrix : np.ndarray
        Shape ``(n, n)``; row ``i`` holds unit ``i``'s weights on the other
        units and a zero on the diagonal.
    """
    n = y_pre.shape[0]
    intercepts = np.zeros(n)
    b_matrix = np.zeros((n, n))
    for i in range(n):
        donors = np.delete(y_pre, i, axis=0).T
        intercepts[i], weights = fit_simplex_scm(y_pre[i], donors, maxiter=maxiter, ftol=ftol)
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
