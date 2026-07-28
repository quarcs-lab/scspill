r"""Posterior treatment and spillover effects for SCSPILL (Theorems 3.1-3.2).

Given the spillover intensity ``rho`` and the synthetic weights ``alpha``,
the no-treatment counterfactual of the control outcomes is

.. math::

    \\mathbf Y^c_t(\\mathbf 0)
    = (\\mathbf I_N - \\rho \\mathbf W - \\rho \\mathbf w \\boldsymbol\\alpha^\\top)^{-1}
      \\big((\\mathbf I_N - \\rho \\mathbf W)\\, \\mathbf Y^c_t
            - \\rho\\, \\mathbf w\\, Y_{0t}\\big),

from which the treated unit's counterfactual is
``alpha' Y^c_t(0)`` and the spillover effect on each control is
``Y^c_t - Y^c_t(0)``. Only ``(alpha, rho, w, W)`` and the observed outcomes
enter -- no error-model parameters.

The posterior sweep evaluates these formulas for every retained draw. The
:class:`RhoSolver` exploits the structure of the inverse: a one-time
eigendecomposition of ``Wn`` turns each ``(I - rho Wn)^{-1}`` application
into a diagonal rescaling, and the rank-one term ``rho w alpha'`` is absorbed
by the Sherman-Morrison identity -- so one posterior draw costs
``O(N^2 T)`` instead of an ``O(N^3)`` factorization. Ill-conditioned cases
fall back to an explicit ridge-escalated solve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ....exceptions import ScspillEstimationError
from ..structures import SCSPILLEffects, SCSPILLInputs


def _robust_solve(Amat: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve ``Amat @ X = B`` with escalating ridge, then pseudo-inverse.

    Port of the R package's ``robust_solve`` ladder: plain solve, then
    ridge-regularized solves with ``1e-12 * 10^k`` on the diagonal, then the
    Moore-Penrose pseudo-inverse as a last resort.
    """
    try:
        return np.linalg.solve(Amat, B)
    except np.linalg.LinAlgError:
        pass
    n = Amat.shape[0]
    for k in range(6):
        try:
            return np.linalg.solve(Amat + (1e-12 * 10.0**k) * np.eye(n), B)
        except np.linalg.LinAlgError:
            continue
    return np.linalg.pinv(Amat) @ B


def counterfactual_controls(
    Y0: np.ndarray,
    Yc: np.ndarray,
    Wn: np.ndarray,
    wn: np.ndarray,
    alpha: np.ndarray,
    rho: float,
) -> np.ndarray:
    """No-treatment counterfactual of the control outcomes (direct solve).

    The reference implementation of the identification formula -- one dense
    solve per call. The posterior sweep uses :class:`RhoSolver` instead; this
    function is the ground truth it is tested against, and the reusable seam
    for the simulation engine.

    Parameters
    ----------
    Y0 : np.ndarray
        Treated outcomes over the evaluated periods, shape ``(T,)``.
    Yc : np.ndarray
        Control outcomes over the evaluated periods, shape ``(T, N)``.
    Wn, wn : np.ndarray
        Normalized spatial weights.
    alpha : np.ndarray
        Synthetic weights, shape ``(N,)``.
    rho : float
        Spillover intensity.

    Returns
    -------
    np.ndarray
        Counterfactual control outcomes ``Y^c(0)``, shape ``(T, N)``.
    """
    Y0 = np.asarray(Y0, dtype=float).ravel()
    Yc = np.asarray(Yc, dtype=float)
    N = Wn.shape[0]
    I_N = np.eye(N)
    Mmat = I_N - rho * Wn - rho * np.outer(wn, alpha)
    RHS = (I_N - rho * Wn) @ Yc.T - rho * np.outer(wn, Y0)
    return _robust_solve(Mmat, RHS).T


def treated_counterfactual(
    Y0: np.ndarray,
    Yc: np.ndarray,
    Wn: np.ndarray,
    wn: np.ndarray,
    alpha: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Treated unit's no-treatment counterfactual path (Theorem 3.1).

    ``alpha' @ Y^c(0)`` over the evaluated periods; see
    :func:`counterfactual_controls` for the parameters.

    Returns
    -------
    np.ndarray
        Counterfactual treated outcomes, shape ``(T,)``.
    """
    return counterfactual_controls(Y0, Yc, Wn, wn, alpha, rho) @ np.asarray(alpha, dtype=float)


def spillover_effects(
    Y0: np.ndarray,
    Yc: np.ndarray,
    Wn: np.ndarray,
    wn: np.ndarray,
    alpha: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Per-control spillover effects (Theorem 3.2) over the evaluated periods.

    ``Y^c - Y^c(0)``; see :func:`counterfactual_controls` for the parameters.

    Returns
    -------
    np.ndarray
        Spillover effects, shape ``(T, N)``.
    """
    return np.asarray(Yc, dtype=float) - counterfactual_controls(Y0, Yc, Wn, wn, alpha, rho)


class RhoSolver:
    r"""Fast per-draw application of the identification inverse.

    Diagonalizes ``Wn = P diag(lam) P^{-1}`` once, so
    ``(I - rho Wn)^{-1} x = P ((P^{-1} x) / (1 - rho lam))`` costs
    ``O(N^2)`` per column, and absorbs the rank-one spillover term with
    Sherman-Morrison:

    .. math::

        (\\mathbf B - \\rho \\mathbf w \\boldsymbol\\alpha^\\top)^{-1}
        = \\mathbf B^{-1} + \\frac{\\mathbf B^{-1} \\rho \\mathbf w\\,
          \\boldsymbol\\alpha^\\top \\mathbf B^{-1}}
          {1 - \\boldsymbol\\alpha^\\top \\mathbf B^{-1} \\rho \\mathbf w},
        \\qquad \\mathbf B = \\mathbf I - \\rho \\mathbf W.

    When the eigenvector basis is ill-conditioned, an eigenvalue collides
    with ``1/rho``, or the Sherman-Morrison denominator vanishes, the solver
    falls back to the explicit ridge-escalated solve of
    :func:`counterfactual_controls`.

    Parameters
    ----------
    Wn : np.ndarray
        Row-normalized control-to-control weight matrix, ``(N, N)``.
    wn : np.ndarray
        Sum-normalized treated-to-control exposure vector, ``(N,)``.
    """

    #: Guards for the fast path.
    _COND_MAX = 1e8
    _DENOM_MIN = 1e-10

    def __init__(self, Wn: np.ndarray, wn: np.ndarray) -> None:
        self.Wn = np.asarray(Wn, dtype=float)
        self.wn = np.asarray(wn, dtype=float).ravel()
        lam, P = np.linalg.eig(self.Wn)
        self.lam = lam.astype(np.complex128)
        self.P = P.astype(np.complex128)
        self.Pinv: np.ndarray | None
        self.Pinv_wn: np.ndarray | None
        try:
            self.Pinv = np.linalg.inv(self.P)
            self.cond_ok = bool(np.linalg.cond(self.P) < self._COND_MAX)
        except np.linalg.LinAlgError:  # pragma: no cover - defective eigenbasis
            self.Pinv = None
            self.cond_ok = False
        self.Pinv_wn = (
            self.Pinv @ self.wn.astype(np.complex128)
            if self.cond_ok and self.Pinv is not None
            else None
        )

    def counterfactual_controls(
        self,
        Y0: np.ndarray,
        Yc: np.ndarray,
        alpha: np.ndarray,
        rho: float,
        *,
        _cache: dict | None = None,
    ) -> np.ndarray:
        """Counterfactual control outcomes ``Y^c(0)``, shape ``(T, N)``.

        Parameters
        ----------
        Y0, Yc : np.ndarray
            Outcomes over the evaluated periods, shapes ``(T,)`` / ``(T, N)``.
        alpha : np.ndarray
            Synthetic weights for this draw, shape ``(N,)``.
        rho : float
            Spillover intensity for this draw.
        _cache : dict, optional
            Reusable ``P^{-1}``-transformed data blocks; pass the same dict
            across draws that share ``(Y0, Yc)`` to skip re-transformation
            (the posterior sweep does).
        """
        alpha = np.asarray(alpha, dtype=float).ravel()
        if not self.cond_ok:
            return counterfactual_controls(Y0, Yc, self.Wn, self.wn, alpha, rho)
        assert self.Pinv is not None and self.Pinv_wn is not None  # implied by cond_ok

        if _cache is None:
            _cache = {}
        if "PinvYcT" not in _cache:
            Y0 = np.asarray(Y0, dtype=float).ravel()
            Yc = np.asarray(Yc, dtype=float)
            _cache["PinvYcT"] = self.Pinv @ Yc.T.astype(np.complex128)
            _cache["PinvWYcT"] = self.Pinv @ (self.Wn @ Yc.T).astype(np.complex128)
            _cache["PinvwY0"] = self.Pinv @ np.outer(self.wn, Y0).astype(np.complex128)
            _cache["Yc"] = Yc
            _cache["Y0"] = Y0

        denom_eig = 1.0 - rho * self.lam
        if np.min(np.abs(denom_eig)) < self._DENOM_MIN:
            return counterfactual_controls(
                _cache_y0(_cache), _cache["Yc"], self.Wn, self.wn, alpha, rho
            )

        # B(rho)^{-1} RHS with RHS = (I - rho Wn) Yc' - rho w Y0'
        PinvRHS = _cache["PinvYcT"] - rho * _cache["PinvWYcT"] - rho * _cache["PinvwY0"]
        Binv_RHS = self.P @ (PinvRHS / denom_eig[:, None])
        # Sherman-Morrison for the rank-one term rho w alpha'
        Binv_u = rho * (self.P @ (self.Pinv_wn / denom_eig))
        sm_denom = 1.0 - complex(alpha.astype(np.complex128) @ Binv_u)
        if abs(sm_denom) < self._DENOM_MIN:
            return counterfactual_controls(
                _cache_y0(_cache), _cache["Yc"], self.Wn, self.wn, alpha, rho
            )
        correction = np.outer(Binv_u, (alpha.astype(np.complex128) @ Binv_RHS) / sm_denom)
        return np.real(Binv_RHS + correction).T


def _cache_y0(cache: dict) -> np.ndarray:
    """Recover ``Y0`` from the cached ``P^{-1} (w Y0')`` block's source data."""
    # The cache stores Yc directly; Y0 is recoverable from PinvwY0 only up to
    # the weight pattern, so keep it explicitly on first use.
    if "Y0" not in cache:
        raise ScspillEstimationError(
            "RhoSolver cache is missing Y0; populate the cache via "
            "posterior_effects, which stores it."
        )
    return cache["Y0"]


def posterior_effects(
    inputs: SCSPILLInputs,
    alpha_draws: np.ndarray,
    rho_draws: np.ndarray,
    alpha_hat: np.ndarray,
    rho_hat: float,
    *,
    ci: float = 0.95,
    propagate_alpha: bool = True,
    max_draws: int | None = None,
    solver: RhoSolver | None = None,
) -> SCSPILLEffects:
    """Sweep the posterior draws through the identification formulas.

    For each retained draw ``m``, plugs ``(alpha^(m), rho^(m))`` (or
    ``(alpha_hat, rho^(m))`` when ``propagate_alpha=False``, the R package's
    convention) into Theorems 3.1-3.2 over the full period range, and
    summarizes the resulting treated-counterfactual and spillover paths with
    posterior means and equal-tailed pointwise credible bands.

    Parameters
    ----------
    inputs : SCSPILLInputs
        Prepared estimation inputs.
    alpha_draws : np.ndarray
        Step-1 draws, shape ``(M, N)``.
    rho_draws : np.ndarray
        Step-2 draws, shape ``(M,)``. Must match ``alpha_draws`` in length
        when ``propagate_alpha=True`` (both steps retain ``m_iter - burn``
        draws, so this holds by construction).
    alpha_hat : np.ndarray
        Posterior mean weights, shape ``(N,)``.
    rho_hat : float
        Posterior mean spillover intensity.
    ci : float, default 0.95
        Credible level of all intervals.
    propagate_alpha : bool, default True
        Pair the two chains by index (paper procedure) or hold ``alpha`` at
        ``alpha_hat`` (R behavior).
    max_draws : int, optional
        Evenly-spaced thinning cap on the number of draws swept.
    solver : RhoSolver, optional
        Reusable solver; constructed from ``inputs`` when omitted.

    Returns
    -------
    SCSPILLEffects

    Raises
    ------
    ScspillEstimationError
        If ``propagate_alpha=True`` and the two chains have different lengths.
    """
    Y0, Yc = inputs.Y0, inputs.Yc
    T, T0, N = inputs.T, inputs.T0, inputs.N
    alpha_draws = np.asarray(alpha_draws, dtype=float)
    rho_draws = np.asarray(rho_draws, dtype=float).ravel()

    if propagate_alpha and alpha_draws.shape[0] != rho_draws.shape[0]:
        raise ScspillEstimationError(
            f"posterior_effects: alpha_draws ({alpha_draws.shape[0]}) and rho_draws "
            f"({rho_draws.shape[0]}) differ in length; cannot pair draws. "
            "Both steps retain m_iter - burn draws, so this indicates truncated input."
        )

    M_total = rho_draws.shape[0]
    if max_draws is not None and max_draws < M_total:
        idx = np.unique(np.linspace(0, M_total - 1, int(max_draws)).round().astype(int))
    else:
        idx = np.arange(M_total)
    M = idx.size

    if solver is None:
        solver = RhoSolver(inputs.Wn, inputs.wn)

    cache: dict = {"Y0": Y0}
    cf_draws = np.empty((M, T))
    spill_draws = np.empty((M, T, N))
    for out_i, m in enumerate(idx):
        a = alpha_draws[m] if propagate_alpha else alpha_hat
        CF = solver.counterfactual_controls(Y0, Yc, a, float(rho_draws[m]), _cache=cache)
        cf_draws[out_i] = CF @ a
        spill_draws[out_i] = Yc - CF

    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q

    att_draws = (Y0[T0:][None, :] - cf_draws[:, T0:]).mean(axis=1)
    att = float(att_draws.mean())
    att_ci = (float(np.quantile(att_draws, lo_q)), float(np.quantile(att_draws, hi_q)))

    # Plug-in comparators.
    cf_plugin = treated_counterfactual(Y0, Yc, inputs.Wn, inputs.wn, alpha_hat, rho_hat)
    att_plugin = float((Y0[T0:] - cf_plugin[T0:]).mean())
    att_scm = float((Y0[T0:] - Yc[T0:] @ alpha_hat).mean())

    time_index = pd.Index(inputs.time_labels, name="time")
    cols = pd.Index(inputs.control_labels, name="unit")
    spill_mean = pd.DataFrame(spill_draws.mean(axis=0), index=time_index, columns=cols)
    spill_lower = pd.DataFrame(
        np.quantile(spill_draws, lo_q, axis=0), index=time_index, columns=cols
    )
    spill_upper = pd.DataFrame(
        np.quantile(spill_draws, hi_q, axis=0), index=time_index, columns=cols
    )

    return SCSPILLEffects(
        att=att,
        att_ci=att_ci,
        att_draws=att_draws,
        att_plugin=att_plugin,
        att_scm=att_scm,
        cf_mean=cf_draws.mean(axis=0),
        cf_lower=np.quantile(cf_draws, lo_q, axis=0),
        cf_upper=np.quantile(cf_draws, hi_q, axis=0),
        spill_mean=spill_mean,
        spill_lower=spill_lower,
        spill_upper=spill_upper,
        ci_level=float(ci),
        propagate_alpha=bool(propagate_alpha),
        n_draws_used=int(M),
    )
