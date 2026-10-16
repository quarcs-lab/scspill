"""The Cao--Dowd effect estimator under a known exposure structure.

With ``R = I - B`` and the fitted intercepts ``a``, the post-treatment
residual equations read ``R Y_t - a = R A gamma_t + u_t``, so

    gamma_t = (A' R' R A)^{-1} A' R' (R Y_t - a),    alpha_t = A gamma_t

(Cao & Dowd, arXiv:1902.07343v3, Section 3.2). ``gamma_t`` holds one effect
per column of ``A``; ``alpha_t`` maps them back to one effect per unit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ....exceptions import ScspillEstimationError


@dataclass(frozen=True)
class EffectOperator:
    """The linear maps from residual equations to effects.

    Attributes
    ----------
    gamma_map : np.ndarray
        ``(A'R'RA)^{-1} A'R'``, shape ``(k, n)``: residuals to parameters.
    unit_map : np.ndarray
        ``G = A gamma_map``, shape ``(n, n)``: residuals to unit effects.
    projector : np.ndarray
        ``RA gamma_map``, shape ``(n, n)``: the projection onto the span of
        ``RA``; ``I - projector`` drives the ``kappa_A`` test.
    rank : int
        Numerical rank of ``RA``.
    condition_number : float
        Condition number of ``A'R'RA``.
    """

    gamma_map: np.ndarray
    unit_map: np.ndarray
    projector: np.ndarray
    rank: int
    condition_number: float


def effect_operator(b: np.ndarray, a: np.ndarray) -> EffectOperator:
    """Build the effect maps after checking that ``A`` is identified.

    Parameters
    ----------
    b : np.ndarray
        All-unit SCM weight matrix ``B``, shape ``(n, n)``.
    a : np.ndarray
        Exposure matrix ``A``, shape ``(n, k)``.

    Raises
    ------
    ScspillEstimationError
        When ``(I - B) A`` does not have full column rank.
    """
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
    # The paper reports the condition number of A'(I-B)'(I-B)A, whose
    # eigenvalue ratio is the square of the singular-value ratio of (I-B)A.
    return EffectOperator(
        gamma_map=pinv,
        unit_map=a @ pinv,
        projector=ra @ pinv,
        rank=rank,
        condition_number=float((singular[0] / singular[-1]) ** 2),
    )


#: Ridge added to the residual covariance before inverting it; keeps the
#: efficient weight positive definite when ``T0`` does not exceed the number
#: of units (the same value as mlsynth).
OMEGA_RIDGE = 1e-6


def residual_covariance(residuals: np.ndarray, *, ridge: float = OMEGA_RIDGE) -> np.ndarray:
    """Estimate ``Omega = Cov[u_t]`` from the pre-treatment residuals.

    ``Omega_hat = T0^{-1} sum_t u_t u_t' + ridge * I`` with
    ``u_t = (I - B) Y_t - a``, shape ``(n, n)``.
    """
    n, t0 = residuals.shape
    return residuals @ residuals.T / t0 + ridge * np.eye(n)


def weighted_effect_operator(b: np.ndarray, a: np.ndarray, weight: np.ndarray) -> EffectOperator:
    """Build the effect maps under a weighting matrix ``W`` (v3 Section S.1.1).

    ``gamma_W = (A' R' W R A)^{-1} A' R' W (R Y_t - a)``: minimizing
    ``||W^{1/2} u_t||`` instead of ``||u_t||``. With ``W`` a consistent
    estimate of ``Omega^{-1}`` this is the lower-variance estimator of
    Proposition S.1. ``projector`` is then the ``W``-orthogonal projection
    onto the span of ``R A``.

    Raises
    ------
    ScspillEstimationError
        When ``(I - B) A`` does not have full column rank.
    """
    ra = (np.eye(len(b)) - b) @ a
    rank = int(np.linalg.matrix_rank(ra))
    if rank != a.shape[1]:
        raise ScspillEstimationError(
            f"Cao--Dowd effects unidentified: rank((I-B)A)={rank} < {a.shape[1]}."
        )
    gram = ra.T @ weight @ ra
    gamma_map = np.linalg.solve(gram, ra.T @ weight)
    return EffectOperator(
        gamma_map=gamma_map,
        unit_map=a @ gamma_map,
        projector=ra @ gamma_map,
        rank=rank,
        condition_number=float(np.linalg.cond(gram)),
    )
