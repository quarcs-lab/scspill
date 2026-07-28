r"""Step 1 of the SCSPILL sampler: horseshoe Gibbs for the synthetic weights.

Bayesian regression of the treated unit's pre-treatment outcomes on the
control units' pre-treatment outcomes,

.. math::

    Y_{0t} = \\sum_{i=1}^{N} \\alpha_i Y_{it} + \\varepsilon_t,

with a horseshoe prior on the unconstrained weights ``alpha`` (Carvalho,
Polson & Scott 2010; Kim, Lee & Gupta 2020), sampled with the Makalic &
Schmidt (2015) inverse-gamma auxiliary-variable representation so every full
conditional is closed form.

Faithful port of the reference ``hs_alpha_gibbs_cpp``: the outcomes are
scaled by their standard deviations only (no centering, no intercept), and
draws are back-transformed by ``sd(y) / sd(x_j)``.
"""

from __future__ import annotations

import numpy as np

from ...exceptions import ScspillDataError
from ._kernels import resolve_backend
from .structures import AlphaPosterior


def hs_alpha_gibbs(
    rng: np.random.Generator,
    y_pre: np.ndarray,
    Yc_pre: np.ndarray,
    iters: int,
    burn: int,
    *,
    kernels=None,
) -> AlphaPosterior:
    """Draw the synthetic weights ``alpha`` with a horseshoe Gibbs sampler.

    Parameters
    ----------
    rng : numpy.random.Generator
        The random generator (thread it from the estimator's ``seed``).
    y_pre : np.ndarray
        Treated pre-treatment outcomes, shape ``(T0,)``.
    Yc_pre : np.ndarray
        Control pre-treatment outcomes, shape ``(T0, N)``.
    iters, burn : int
        Total iterations and burn-in; ``iters - burn`` draws are retained.
    kernels : namespace, optional
        Kernel backend from
        :func:`scspill.utils.scspill_helpers._kernels.resolve_backend`;
        defaults to the NumPy reference kernels.

    Returns
    -------
    AlphaPosterior
        Post-burn draws (back-transformed to the original outcome units),
        the Step-1 error-variance draws, and the posterior mean
        ``alpha_hat``.

    Raises
    ------
    ScspillDataError
        If the inputs are too short or contain non-finite values.
    """
    y_pre = np.asarray(y_pre, dtype=float).ravel()
    Yc_pre = np.asarray(Yc_pre, dtype=float)
    if Yc_pre.ndim != 2 or y_pre.shape[0] != Yc_pre.shape[0]:
        raise ScspillDataError(
            f"hs_alpha_gibbs: y_pre has length {y_pre.shape[0]} but Yc_pre has "
            f"shape {Yc_pre.shape}."
        )
    T0, _N = Yc_pre.shape
    if T0 < 3:
        raise ScspillDataError(f"hs_alpha_gibbs: needs at least 3 pre-periods, got {T0}.")
    if not (np.all(np.isfinite(y_pre)) and np.all(np.isfinite(Yc_pre))):
        raise ScspillDataError("hs_alpha_gibbs: inputs contain non-finite values.")
    if kernels is None:
        kernels = resolve_backend("numpy")

    # Scale by standard deviations only -- no centering, no intercept
    # (the reference sampler's convention; centering would change alpha).
    sx = np.maximum(Yc_pre.std(axis=0, ddof=1), 1e-8)
    sy = max(float(y_pre.std(ddof=1)), 1e-8)
    Xs = Yc_pre / sx
    ys = y_pre / sy

    alpha_scaled, s2_draws = kernels.hs_alpha_loop(rng, ys, Xs, int(iters), int(burn))
    draws = alpha_scaled * (sy / sx)  # back-transform to outcome units
    return AlphaPosterior(
        draws=draws,
        sigma2=s2_draws,
        alpha_hat=draws.mean(axis=0),
        iters=int(iters),
        burn=int(burn),
    )


def alpha_scaling(y_pre: np.ndarray, Yc_pre: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the ``(sd(y), sd(x_j))`` scaling used by the Step-1 sampler.

    Exposed for the validation suite, which needs ``alpha`` on the
    standardized scale (``alpha_scaled_j = alpha_j * sx_j / sy``).

    Parameters
    ----------
    y_pre : np.ndarray
        Treated pre-treatment outcomes, shape ``(T0,)``.
    Yc_pre : np.ndarray
        Control pre-treatment outcomes, shape ``(T0, N)``.

    Returns
    -------
    (float, np.ndarray)
        ``sy`` and the vector ``sx`` (floored at ``1e-8``).
    """
    y_pre = np.asarray(y_pre, dtype=float).ravel()
    Yc_pre = np.asarray(Yc_pre, dtype=float)
    sx = np.maximum(Yc_pre.std(axis=0, ddof=1), 1e-8)
    sy = max(float(y_pre.std(ddof=1)), 1e-8)
    return sy, sx


__all__ = ["alpha_scaling", "hs_alpha_gibbs"]
