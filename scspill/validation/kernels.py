"""Transition kernels for the Geweke joint distribution test.

Two interchangeable kernels drive :func:`scspill.validation.geweke.geweke_test`:

* :class:`SimpleKernel` -- an exact port of the replication package's
  ``scspill_one_step_cpp``: the SAR model with *unified simple priors*
  (``beta ~ N(0, I)``, ``Gamma_t ~ N(0, I)`` with no AR(1), ``Eta_i ~
  N(0, I)``, ``sigma2 ~ IG(a0, b0)``, ``rho`` flat on a bounded support).
  This is the appendix's test target and is directly comparable to the R
  package's frozen Geweke table.
* :class:`ProductionKernel` -- wraps the production Step-2 sampler's
  :func:`~scspill.utils.scspill_helpers.sar.sampler_sar.one_sweep` and
  :func:`~scspill.utils.scspill_helpers.sar.sampler_sar.draw_prior_state`, so
  the test validates the sampler users actually run (AR(1) latent factors,
  horseshoe hierarchies, half-Cauchy scale mixtures).

Both share the forward data simulator :func:`simulate_yc_forward`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..exceptions import ScspillDataError
from ..utils.scspill_helpers.sar._kernels import _clip, _ig
from ..utils.scspill_helpers.sar.sampler_sar import (
    SARData,
    SARState,
    draw_prior_state,
    make_sar_data,
    one_sweep,
)


def rho_bound_from_A(
    Wn: np.ndarray, wn: np.ndarray, alpha: np.ndarray, c_stab: float = 0.95
) -> float:
    """Spectral stability bound ``c / max|eig(W + w alpha')|``.

    Parameters
    ----------
    Wn, wn : np.ndarray
        Normalized spatial weights.
    alpha : np.ndarray
        Synthetic weights, shape ``(N,)``.
    c_stab : float, default 0.95
        Stability margin.

    Returns
    -------
    float
    """
    A = Wn + np.outer(wn, alpha)
    max_abs = float(np.max(np.abs(np.linalg.eigvals(A))))
    if not np.isfinite(max_abs) or max_abs < 1e-12:
        max_abs = 1e-12
    return c_stab / max_abs


def simulate_yc_forward(
    rng: np.random.Generator,
    T0: int,
    Wn: np.ndarray,
    wn: np.ndarray,
    alpha: np.ndarray,
    rho: float,
    sigma2: float,
    X: np.ndarray | None = None,
    beta: np.ndarray | None = None,
    Eta: np.ndarray | None = None,
    Gamma: np.ndarray | None = None,
) -> np.ndarray:
    """Forward-simulate the control panel from the SAR model.

    ``Yc_t = (I - rho A)^{-1} (X_t beta + Eta Gamma_t + eps_t)`` with
    ``A = W + w alpha'`` and ``eps_t ~ N(0, sigma2 I)``; a ridge-regularized
    normal-equations solve is the fallback when the direct solve fails
    (mirroring ``simulate_Yc_forward_cpp``).

    Parameters
    ----------
    rng : numpy.random.Generator
    T0 : int
        Number of periods to simulate.
    Wn, wn, alpha : np.ndarray
        Normalized spatial weights and synthetic weights.
    rho, sigma2 : float
        Spillover intensity and error variance.
    X : np.ndarray, optional
        Covariate cube ``(T0, N, K)``.
    beta : np.ndarray, optional
        Covariate coefficients ``(K,)``.
    Eta : np.ndarray, optional
        Factor loadings ``(N, p)``.
    Gamma : np.ndarray, optional
        Factor paths ``(p, T0)``.

    Returns
    -------
    np.ndarray
        Simulated outcomes, shape ``(T0, N)``.
    """
    N = Wn.shape[0]
    A = Wn + np.outer(wn, alpha)
    Mmat = np.eye(N) - rho * A
    sd = float(np.sqrt(max(1e-12, sigma2)))

    Mu = np.zeros((N, T0))
    if X is not None and beta is not None and beta.size > 0:
        # X is (T0, N, K); contract over K then transpose to (N, T0).
        Mu += np.einsum("tnk,k->tn", X, beta).T
    if Eta is not None and Gamma is not None and Eta.size > 0:
        Mu += Eta @ Gamma

    rhs = Mu + sd * rng.standard_normal((N, T0))
    try:
        sol = np.linalg.solve(Mmat, rhs)
        if not np.all(np.isfinite(sol)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:  # pragma: no cover - near-singular fallback
        AtA = Mmat.T @ Mmat + 1e-10 * np.eye(N)
        sol = np.linalg.solve(AtA, Mmat.T @ rhs)
    return sol.T


@dataclass
class SimpleState:
    """Parameter state of the simplified (appendix) SAR model."""

    rho: float
    sigma2: float
    beta: np.ndarray  # (K,)
    Eta: np.ndarray  # (N, p)
    Gamma: np.ndarray  # (p, T0)


class SimpleKernel:
    """The appendix's simplified SAR kernel (port of ``scspill_one_step_cpp``).

    Priors: ``rho`` flat on ``rho_support`` (intersected with the spectral
    bound of ``A``); ``sigma2 ~ IG(a0, b0)``; ``beta ~ N(0, I_K)``;
    ``Eta_ik ~ N(0, 1)``; ``Gamma_kt ~ N(0, 1)`` (no AR(1) dynamics).

    Parameters
    ----------
    T0, N, K, p : int
        Panel and model dimensions.
    Wn, wn : np.ndarray
        Normalized spatial weights (``Wn`` row-stochastic, ``wn`` sums to 1).
    alpha : np.ndarray
        Fixed synthetic weights, shape ``(N,)``.
    X : np.ndarray, optional
        Fixed covariate cube ``(T0, N, K)`` (required when ``K > 0``).
    a0, b0 : float, default 1.0
        Inverse-gamma prior for ``sigma2``.
    step_rho : float, default 0.05
        Fixed random-walk Metropolis step (no adaptation: the Geweke test
        needs a fixed Markov kernel).
    rho_support : (float, float), optional
        Support of ``rho``; defaults to the spectral bound of ``A`` and is
        always intersected with it.
    """

    name = "simple"

    def __init__(
        self,
        T0: int,
        N: int,
        K: int,
        p: int,
        Wn: np.ndarray,
        wn: np.ndarray,
        alpha: np.ndarray,
        *,
        X: np.ndarray | None = None,
        a0: float = 1.0,
        b0: float = 1.0,
        step_rho: float = 0.05,
        rho_support: tuple[float, float] | None = None,
    ) -> None:
        if K > 0 and (X is None or X.shape != (T0, N, K)):
            raise ScspillDataError(f"SimpleKernel: K={K} requires X with shape ({T0}, {N}, {K}).")
        self.T0, self.N, self.K, self.p = T0, N, K, p
        self.Wn = np.asarray(Wn, dtype=float)
        self.wn = np.asarray(wn, dtype=float).ravel()
        self.alpha = np.asarray(alpha, dtype=float).ravel()
        self.X = X
        self.a0, self.b0 = float(a0), float(b0)
        self.step_rho = float(step_rho)

        self.A = self.Wn + np.outer(self.wn, self.alpha)
        self.evA = np.linalg.eigvals(self.A).astype(np.complex128)
        bnd = rho_bound_from_A(self.Wn, self.wn, self.alpha)
        lo, hi = rho_support if rho_support is not None else (-bnd, bnd)
        # One canonical support for prior draws AND the transition kernel.
        self.rho_support = (max(float(lo), -bnd), min(float(hi), bnd))
        if not self.rho_support[0] < self.rho_support[1]:
            raise ScspillDataError(f"SimpleKernel: empty rho support {self.rho_support}.")
        if K > 0:
            assert X is not None  # validated above
            self._X2 = X.reshape(T0 * N, K)
            self._XtX = self._X2.T @ self._X2
        else:
            self._X2 = np.zeros((0, 0))
            self._XtX = np.zeros((0, 0))

    # -- Geweke protocol ---------------------------------------------------
    def draw_prior(self, rng: np.random.Generator) -> SimpleState:
        """One draw of the full parameter state from the prior."""
        lo, hi = self.rho_support
        return SimpleState(
            rho=float(rng.uniform(lo, hi)),
            sigma2=float(_ig(rng, self.a0, self.b0)),
            beta=rng.standard_normal(self.K),
            Eta=rng.standard_normal((self.N, self.p)),
            Gamma=rng.standard_normal((self.p, self.T0)),
        )

    def simulate_data(self, state: SimpleState, rng: np.random.Generator) -> np.ndarray:
        """Simulate ``Yc ~ p(y | theta)``, shape ``(T0, N)``."""
        return simulate_yc_forward(
            rng,
            self.T0,
            self.Wn,
            self.wn,
            self.alpha,
            state.rho,
            state.sigma2,
            X=self.X,
            beta=state.beta,
            Eta=state.Eta,
            Gamma=state.Gamma,
        )

    def transition(
        self, state: SimpleState, Yc: np.ndarray, rng: np.random.Generator
    ) -> SimpleState:
        """One full Gibbs/Metropolis sweep of the simplified model (in place)."""
        T0, N, K, p = self.T0, self.N, self.K, self.p
        rho, sigma2 = state.rho, state.sigma2
        beta, Eta, Gamma = state.beta, state.Eta, state.Gamma
        s2c = _clip(sigma2)

        M_rho = np.eye(N) - rho * self.A
        MY = Yc @ M_rho.T  # (T0, N): row t = (M_rho @ Y_t)'
        YA = Yc @ self.A.T
        Xbeta = (self._X2 @ beta).reshape(T0, N) if K > 0 else np.zeros((T0, N))
        ystar_base = MY - Xbeta

        # (1) Gamma_t | rest ~ N(Vt Eta' y*_t / s2, Vt), prior N(0, I_p)
        if p > 0:
            EtE = Eta.T @ Eta
            Vt = np.linalg.inv(0.5 * ((EtE / s2c + np.eye(p)) + (EtE / s2c + np.eye(p)).T))
            Lt = np.linalg.cholesky(0.5 * (Vt + Vt.T) + 1e-12 * np.eye(p))
            for t in range(T0):
                mt = Vt @ (Eta.T @ ystar_base[t] / s2c)
                Gamma[:, t] = mt + Lt @ rng.standard_normal(p)

            # (2) Eta_i | rest, prior N(0, I_p)
            GtG = Gamma @ Gamma.T
            Vi = np.linalg.inv(0.5 * ((GtG / s2c + np.eye(p)) + (GtG / s2c + np.eye(p)).T))
            Li = np.linalg.cholesky(0.5 * (Vi + Vi.T) + 1e-12 * np.eye(p))
            for i in range(N):
                rhs = Gamma @ ystar_base[:, i]
                mi = Vi @ (rhs / s2c)
                Eta[i] = mi + Li @ rng.standard_normal(p)

        # (3) beta | rest, prior N(0, I_K)
        if K > 0:
            fac = (Eta @ Gamma).T if p > 0 else np.zeros((T0, N))
            lhs = MY - fac
            XtY = self._X2.T @ lhs.ravel()
            P_beta = self._XtX / _clip(sigma2) + np.eye(K)
            V_beta = np.linalg.inv(0.5 * (P_beta + P_beta.T))
            m_beta = V_beta @ (XtY / _clip(sigma2))
            Lb = np.linalg.cholesky(0.5 * (V_beta + V_beta.T) + 1e-12 * np.eye(K))
            beta = m_beta + Lb @ rng.standard_normal(K)
            state.beta = beta
            Xbeta = (self._X2 @ beta).reshape(T0, N)

        # (4) sigma2 | rest
        fac = (Eta @ Gamma).T if p > 0 else np.zeros((T0, N))
        resid = MY - Xbeta - fac
        ss = float(np.sum(resid * resid))
        sigma2 = float(_ig(rng, self.a0 + 0.5 * (T0 * N), self.b0 + 0.5 * ss))
        state.sigma2 = sigma2

        # (5) rho | rest via fixed-step random-walk Metropolis
        mu_mat = Xbeta + fac
        lo, hi = self.rho_support

        def logpost(r: float) -> float:
            if r < lo or r > hi:
                return -np.inf
            mags = np.abs(1.0 - r * self.evA)
            if np.any(mags <= 1e-18) or not np.all(np.isfinite(mags)):
                return -np.inf
            ldet = float(np.sum(np.log(mags)))
            MY_r = MY + (rho - r) * YA
            resid_r = MY_r - mu_mat
            ssum = float(np.sum(resid_r * resid_r))
            return T0 * ldet - 0.5 * (T0 * N) * np.log(_clip(sigma2)) - 0.5 * ssum / _clip(sigma2)

        rho_prop = rho + self.step_rho * rng.standard_normal()
        lcur = logpost(rho)
        lprp = logpost(rho_prop)
        if np.log(rng.random()) < lprp - lcur:
            state.rho = rho_prop
        return state

    def state_summary(self, state: SimpleState) -> dict:
        """Named parameter blocks consumed by the g-statistics."""
        return {
            "rho": state.rho,
            "sigma2": state.sigma2,
            "beta": state.beta,
            "Eta": state.Eta,
            "Gamma": state.Gamma,
        }

    def extra_stats(self, state: SimpleState) -> dict:
        """Kernel-specific extra g-statistics (none for the simple kernel)."""
        return {}


class ProductionKernel:
    """Geweke kernel wrapping the production Step-2 sampler.

    Uses :func:`~scspill.utils.scspill_helpers.sar.sampler_sar.draw_prior_state`
    for the marginal-conditional side and
    :func:`~scspill.utils.scspill_helpers.sar.sampler_sar.one_sweep` (fixed
    Metropolis step, no adaptation) as the transition -- so a passing test
    certifies the sampler users actually run, including the AR(1) factor
    block and the horseshoe-on-``beta`` hierarchy.

    Parameters mirror :class:`SimpleKernel`; ``beta_prior`` selects the
    horseshoe (default) or ridge conditional.
    """

    name = "production"

    def __init__(
        self,
        T0: int,
        N: int,
        K: int,
        p: int,
        Wn: np.ndarray,
        wn: np.ndarray,
        alpha: np.ndarray,
        *,
        X: np.ndarray | None = None,
        a0: float = 1.0,
        b0: float = 1.0,
        step_rho: float = 0.05,
        rho_support: tuple[float, float] | None = None,
        beta_prior: str = "horseshoe",
    ) -> None:
        if K > 0 and (X is None or X.shape != (T0, N, K)):
            raise ScspillDataError(
                f"ProductionKernel: K={K} requires X with shape ({T0}, {N}, {K})."
            )
        self.T0, self.N, self.K, self.p = T0, N, K, p
        self.Wn = np.asarray(Wn, dtype=float)
        self.wn = np.asarray(wn, dtype=float).ravel()
        self.alpha = np.asarray(alpha, dtype=float).ravel()
        self.X = X
        bnd = rho_bound_from_A(self.Wn, self.wn, self.alpha)
        lo, hi = rho_support if rho_support is not None else (-bnd, bnd)
        self.rho_support = (max(float(lo), -bnd), min(float(hi), bnd))
        if not self.rho_support[0] < self.rho_support[1]:
            raise ScspillDataError(f"ProductionKernel: empty rho support {self.rho_support}.")
        # Template SARData with a placeholder panel; per-sweep data swaps Yc/AYc.
        self._data = make_sar_data(
            np.zeros((T0, N)),
            self.alpha,
            self.wn,
            self.Wn,
            X=X,
            p=p,
            step_rho=step_rho,
            a0=a0,
            b0=b0,
            beta_prior=beta_prior,
            rho_support=self.rho_support,
        )
        self.A = self.Wn + np.outer(self.wn, self.alpha)

    def _with_panel(self, Yc: np.ndarray) -> SARData:
        """Return the fixed sampler inputs with the panel swapped in."""
        return replace(self._data, Yc=Yc, AYc=(self.A @ Yc.T).T)

    # -- Geweke protocol ---------------------------------------------------
    def draw_prior(self, rng: np.random.Generator) -> SARState:
        """One draw of the full production parameter state from its prior."""
        return draw_prior_state(self._data, rng)

    def simulate_data(self, state: SARState, rng: np.random.Generator) -> np.ndarray:
        """Simulate ``Yc ~ p(y | theta)``, shape ``(T0, N)``."""
        return simulate_yc_forward(
            rng,
            self.T0,
            self.Wn,
            self.wn,
            self.alpha,
            state.rho,
            state.s2,
            X=self.X,
            beta=state.beta,
            Eta=state.Eta,
            Gamma=state.Gamma,
        )

    def transition(self, state: SARState, Yc: np.ndarray, rng: np.random.Generator) -> SARState:
        """One full production Gibbs/Metropolis sweep on the supplied panel."""
        return one_sweep(state, self._with_panel(Yc), rng)

    def state_summary(self, state: SARState) -> dict:
        """Named parameter blocks consumed by the g-statistics."""
        return {
            "rho": state.rho,
            "sigma2": state.s2,
            "beta": state.beta,
            "Eta": state.Eta,
            "Gamma": state.Gamma,
        }

    def extra_stats(self, state: SARState) -> dict:
        """Hyperparameter statistics that only exist in the production model."""
        out: dict = {}
        if self.p > 0:
            out["phi_g"] = float(state.phi_g)
            out["log_s2_g"] = float(np.log(max(state.s2_g, 1e-12)))
            out["log_s2_eta"] = float(np.log(max(state.s2_eta, 1e-12)))
        if self.K > 0 and self._data.beta_hs:
            out["log_psi2"] = float(np.log(max(state.psi2, 1e-12)))
        return out
