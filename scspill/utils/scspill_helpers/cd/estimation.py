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
