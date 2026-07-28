r"""The paper's SAR data-generating process (Section 5).

Pre-treatment, controls follow the spatial-autoregressive panel

.. math::

    Y^c_t = (I - \rho W - \rho w \alpha')^{-1} (X_t \beta + e_t),

and the treated unit is the exact synthetic combination
``Y0_t = alpha' Y^c_t`` (the perfect-fit assumption holds by construction).
Post-treatment, the treated outcome gains ``tau_t ~ N(mu_tau, sd_tau)`` and
the control outcomes are regenerated through the SAR carrying the treated
unit's *treated* outcome -- so the controls are contaminated and classical
SCM is biased whenever ``rho != 0``.

Faithful port of the replication package's ``scspill_sim_dgp`` (same random
draw order: ``X_pre``, ``X_post``, errors, ``tau``; NumPy seeds are not
compatible with R's, only the distributions match).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..exceptions import ScspillDataError
from ..utils.scspill_helpers.setup import row_normalize


def rook_W(nrow: int, ncol: int, normalize: bool = False) -> np.ndarray:
    """Binary rook adjacency on an ``nrow x ncol`` lattice (row-major ids).

    Parameters
    ----------
    nrow, ncol : int
        Lattice dimensions; the matrix has ``nrow * ncol`` units.
    normalize : bool, default False
        Row-normalize the adjacency before returning.

    Returns
    -------
    np.ndarray
        The ``(N, N)`` adjacency: 1 where two cells share an edge.
    """
    N = nrow * ncol
    W = np.zeros((N, N))
    for i in range(N):
        r, c = divmod(i, ncol)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < nrow and 0 <= cc < ncol:
                W[i, rr * ncol + cc] = 1.0
    return row_normalize(W) if normalize else W


def make_w(N: int, treated=(0, 1, 2, 3)) -> np.ndarray:
    """Build the indicator exposure vector linking the treated unit to selected controls.

    Parameters
    ----------
    N : int
        Number of control units.
    treated : int or sequence of int, default (0, 1, 2, 3)
        Zero-based indices of the exposed controls. The default matches the
        paper's simulation design (the first four donors exposed, R's
        ``1:4``); note the R *function* default is a single exposed donor.

    Returns
    -------
    np.ndarray
        A 0/1 vector of length ``N``.
    """
    w = np.zeros(N)
    idx = np.atleast_1d(np.asarray(treated, dtype=int))
    w[idx] = 1.0
    return w


def paper_alpha(N: int) -> np.ndarray:
    """Return the paper's planted synthetic weights.

    ``alpha = (0.5, -0.2, 0.4, 0.4, 0.1/6 x 6, 0, ...)`` -- sparse, signed,
    and not on the simplex.

    Parameters
    ----------
    N : int
        Number of control units (must be at least 10).

    Returns
    -------
    np.ndarray
    """
    if N < 10:
        raise ScspillDataError(f"paper_alpha needs N >= 10, got {N}.")
    alpha = np.zeros(N)
    alpha[0] = 0.5
    alpha[1] = -0.2
    alpha[2] = 0.4
    alpha[3] = 0.4
    alpha[4:10] = 0.1 / 6.0
    return alpha


@dataclass(frozen=True)
class SimTruth:
    """Ground truth of one simulated panel."""

    rho: float
    sigma2: float
    alpha: np.ndarray
    beta: np.ndarray
    tau_post: np.ndarray  # realized per-period treatment effects, (T1,)
    y0_cf_post: np.ndarray  # treated unit's untreated post-period path, (T1,)


@dataclass(frozen=True)
class SimDGP:
    """One simulated spillover panel (matrices plus ground truth).

    Attributes
    ----------
    Y0_pre, Y0_post : np.ndarray
        Treated outcomes, shapes ``(T0,)`` and ``(T1,)``.
    Yc_pre, Yc_post : np.ndarray
        Control outcomes, shapes ``(T0, N)`` and ``(T1, N)``.
    X_pre, X_post : np.ndarray or None
        Covariate cubes ``(T0, N, K)`` / ``(T1, N, K)``, or None when K=0.
    Wn, wn : np.ndarray
        Normalized spatial weights used by the DGP.
    truth : SimTruth
        The planted parameters and realized effects.
    errors : np.ndarray or None
        The error draws ``(T0+T1, N)``, kept when ``keep_internals=True``
        (enables exact-identity tests on the DGP).
    """

    Y0_pre: np.ndarray
    Y0_post: np.ndarray
    Yc_pre: np.ndarray
    Yc_post: np.ndarray
    X_pre: np.ndarray | None
    X_post: np.ndarray | None
    Wn: np.ndarray
    wn: np.ndarray
    truth: SimTruth
    errors: np.ndarray | None = None

    @property
    def T0(self) -> int:
        """Number of pre-treatment periods."""
        return int(self.Y0_pre.shape[0])

    @property
    def T1(self) -> int:
        """Number of post-treatment periods."""
        return int(self.Y0_post.shape[0])

    @property
    def N(self) -> int:
        """Number of control units."""
        return int(self.Yc_pre.shape[1])

    @property
    def K(self) -> int:
        """Number of covariates."""
        return 0 if self.X_pre is None else int(self.X_pre.shape[2])


def scspill_sim_dgp(
    T0: int,
    T1: int,
    N: int,
    W: np.ndarray,
    w: np.ndarray,
    rho: float,
    sigma2: float,
    alpha: np.ndarray,
    K: int = 0,
    beta: np.ndarray | None = None,
    seed=None,
    mu_tau: float = 1.0,
    sd_tau: float = 1.0,
    keep_internals: bool = False,
) -> SimDGP:
    """Simulate one spillover panel from the paper's SAR DGP.

    Parameters
    ----------
    T0, T1 : int
        Pre- and post-treatment lengths.
    N : int
        Number of control units.
    W : np.ndarray
        Raw control-to-control weights, ``(N, N)`` (row-normalized inside).
    w : np.ndarray
        Raw treated-to-control exposure, ``(N,)`` (sum-normalized inside).
    rho : float
        True spillover intensity.
    sigma2 : float
        Error variance.
    alpha : np.ndarray
        True synthetic weights, ``(N,)``.
    K : int, default 0
        Number of iid ``N(0, 1)`` covariates.
    beta : np.ndarray, optional
        Covariate coefficients (required when ``K > 0``).
    seed : int or numpy.random.SeedSequence, optional
        Seed for ``numpy.random.default_rng``.
    mu_tau, sd_tau : float, default 1.0
        Mean and sd of the per-period treatment effects.
    keep_internals : bool, default False
        Keep the error draws on the returned object (for identity tests).

    Returns
    -------
    SimDGP

    Raises
    ------
    ScspillDataError
        On shape mismatches or a near-singular SAR system.
    """
    W = np.asarray(W, dtype=float)
    w = np.asarray(w, dtype=float).ravel()
    alpha = np.asarray(alpha, dtype=float).ravel()
    if W.shape != (N, N):
        raise ScspillDataError(f"scspill_sim_dgp: W has shape {W.shape}, expected ({N}, {N}).")
    if w.shape[0] != N or alpha.shape[0] != N:
        raise ScspillDataError("scspill_sim_dgp: w and alpha must have length N.")
    if K > 0 and beta is None:
        raise ScspillDataError("scspill_sim_dgp: beta must be provided when K > 0.")
    beta_arr = np.zeros(0) if K == 0 else np.asarray(beta, dtype=float).ravel()
    if K > 0 and beta_arr.shape[0] != K:
        raise ScspillDataError("scspill_sim_dgp: beta must have length K.")

    rng = np.random.default_rng(seed)
    I_N = np.eye(N)
    Wn = row_normalize(W)
    wsum = float(w.sum())
    wn = w / wsum if np.isfinite(wsum) and wsum > 1e-12 else w

    A_pre = I_N - rho * Wn - rho * np.outer(wn, alpha)
    if 1.0 / np.linalg.cond(A_pre, p=1) < 1e-10:
        raise ScspillDataError("scspill_sim_dgp: A_pre is near singular.")
    A_pre_inv = np.linalg.inv(A_pre)

    # Draw order matches the R reference: X_pre, X_post, errors, tau.
    X_pre = rng.standard_normal((T0, N, K)) if K > 0 else None
    X_post = rng.standard_normal((T1, N, K)) if K > 0 else None
    TT = T0 + T1
    errors = np.sqrt(max(sigma2, 1e-12)) * rng.standard_normal((TT, N))

    Yc0_all = np.empty((TT, N))
    Y00_all = np.empty(TT)
    for t in range(TT):
        if K > 0:
            assert X_pre is not None and X_post is not None  # implied by K > 0
            Xt = X_pre[t] if t < T0 else X_post[t - T0]
            rhs = Xt @ beta_arr + errors[t]
        else:
            rhs = errors[t]
        Yc0_all[t] = A_pre_inv @ rhs
        Y00_all[t] = float(alpha @ Yc0_all[t])

    A_post = I_N - rho * Wn
    if 1.0 / np.linalg.cond(A_post, p=1) < 1e-10:
        raise ScspillDataError("scspill_sim_dgp: A_post is near singular.")
    A_post_inv = np.linalg.inv(A_post)

    tau_post = rng.normal(mu_tau, sd_tau, size=T1)
    Y01_post = Y00_all[T0:] + tau_post

    Yc1_post = np.empty((T1, N))
    for tt in range(T1):
        rhs = rho * wn * Y01_post[tt]
        if K > 0:
            assert X_post is not None  # implied by K > 0
            rhs = rhs + X_post[tt] @ beta_arr
        rhs = rhs + errors[T0 + tt]
        Yc1_post[tt] = A_post_inv @ rhs

    return SimDGP(
        Y0_pre=Y00_all[:T0],
        Y0_post=Y01_post,
        Yc_pre=Yc0_all[:T0],
        Yc_post=Yc1_post,
        X_pre=X_pre,
        X_post=X_post,
        Wn=Wn,
        wn=wn,
        truth=SimTruth(
            rho=float(rho),
            sigma2=float(sigma2),
            alpha=alpha,
            beta=beta_arr,
            tau_post=Y01_post - Y00_all[T0:],
            y0_cf_post=Y00_all[T0:],
        ),
        errors=errors if keep_internals else None,
    )
