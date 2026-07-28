"""Classical simplex-constrained synthetic control comparator.

The Abadie-style weights (non-negative, summing to one) fitted on the
pre-treatment outcomes, used as the no-spillover benchmark in the weights
plot, the simulation engine, and the benchmarks. Mirrors the R package's
``get_w`` (SLSQP with a penalized L-BFGS-B fallback).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from ...exceptions import ScspillEstimationError


def classical_scm_weights(
    y0_pre: np.ndarray,
    Yc_pre: np.ndarray,
    ridge: float = 0.0,
) -> np.ndarray:
    """Simplex-constrained synthetic control weights.

    Minimizes ``||y0 - Yc w||^2 + ridge ||w||^2`` subject to ``w >= 0`` and
    ``sum(w) = 1``.

    Parameters
    ----------
    y0_pre : np.ndarray
        Treated pre-treatment outcomes, shape ``(T0,)``.
    Yc_pre : np.ndarray
        Control pre-treatment outcomes, shape ``(T0, N)``.
    ridge : float, default 0.0
        Ridge penalty on the weights (the simulation engine uses ``1e-8``,
        matching the R Monte Carlo comparator).

    Returns
    -------
    np.ndarray
        The weight vector, shape ``(N,)``, non-negative and summing to one.

    Raises
    ------
    ScspillEstimationError
        If both the SLSQP solve and the penalized fallback fail.
    """
    y0_pre = np.asarray(y0_pre, dtype=float).ravel()
    Yc_pre = np.asarray(Yc_pre, dtype=float)
    _T0, N = Yc_pre.shape
    Dmat = Yc_pre.T @ Yc_pre + ridge * np.eye(N)
    dvec = Yc_pre.T @ y0_pre

    def objective(w):
        return float(w @ Dmat @ w - 2.0 * dvec @ w + y0_pre @ y0_pre)

    def grad(w):
        return 2.0 * (Dmat @ w - dvec)

    w0 = np.full(N, 1.0 / N)
    res = minimize(
        objective,
        w0,
        jac=grad,
        method="SLSQP",
        bounds=Bounds(0.0, 1.0),
        constraints=[LinearConstraint(np.ones(N), 1.0, 1.0)],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if res.success:
        w = np.clip(res.x, 0.0, None)
        return w / w.sum()

    # Penalized fallback (the R package's L-BFGS-B route).
    lam = 1e3

    def penalized(w):
        return objective(w) + lam * (w.sum() - 1.0) ** 2

    def penalized_grad(w):
        return grad(w) + 2.0 * lam * (w.sum() - 1.0) * np.ones(N)

    res2 = minimize(
        penalized,
        w0,
        jac=penalized_grad,
        method="L-BFGS-B",
        bounds=Bounds(0.0, 1.0),
        options={"maxiter": 2000},
    )
    if not res2.success:  # pragma: no cover - both solvers failing is pathological
        raise ScspillEstimationError(
            f"classical_scm_weights: optimization failed ({res.message} / {res2.message})."
        )
    w = np.clip(res2.x, 0.0, None)
    total = w.sum()
    if total <= 0:  # pragma: no cover - degenerate fallback output
        raise ScspillEstimationError("classical_scm_weights: degenerate zero weights.")
    return w / total


def classical_scm_counterfactual(
    y0: np.ndarray,
    Yc: np.ndarray,
    T0: int,
    ridge: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Classical SCM weights and full-length counterfactual path.

    Fits the simplex weights on the first ``T0`` periods and applies them to
    every period.

    Parameters
    ----------
    y0 : np.ndarray
        Treated outcomes, shape ``(T,)``.
    Yc : np.ndarray
        Control outcomes, shape ``(T, N)``.
    T0 : int
        Number of pre-treatment periods.
    ridge : float, default 0.0
        Ridge penalty passed to :func:`classical_scm_weights`.

    Returns
    -------
    (np.ndarray, np.ndarray)
        The weights ``(N,)`` and the counterfactual path ``(T,)``.
    """
    y0 = np.asarray(y0, dtype=float).ravel()
    Yc = np.asarray(Yc, dtype=float)
    w = classical_scm_weights(y0[:T0], Yc[:T0], ridge=ridge)
    return w, Yc @ w
