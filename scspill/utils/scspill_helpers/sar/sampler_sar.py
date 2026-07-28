r"""Step 2 of the SCSPILL sampler: the SAR block conditional on ``alpha_hat``.

Samples the spatial-autoregressive panel model on the control outcomes,

.. math::

    (\\mathbf I_N - \\rho \\mathbf A)\\, \\mathbf Y^c_t
    = \\mathbf X_t \\boldsymbol\\beta + \\boldsymbol\\eta \\boldsymbol\\gamma_t
      + \\boldsymbol\\varepsilon_t,
    \\qquad \\mathbf A = \\mathbf W + \\mathbf w \\hat{\\boldsymbol\\alpha}^\\top,

with an AR(1) latent factor block (forward-filter/backward-sample), a
horseshoe (paper) or flat-plus-ridge (R code) prior on ``beta``, an
inverse-gamma prior on ``sigma^2``, and a random-walk Metropolis step for the
spillover intensity ``rho`` whose Jacobian ``log|I - rho A|`` is evaluated in
O(N) from the cached complex eigenvalues of ``A``.

Besides the batch sampler :func:`sar_step2_sampler`, this module exposes two
seams used by the Geweke (2004) joint-distribution test in
:mod:`scspill.validation`:

* :func:`draw_prior_state` -- one draw of the full parameter state from the
  generative prior;
* :func:`one_sweep` -- one full Gibbs/Metropolis sweep, consuming the random
  generator in exactly the same order as one iteration of the batch loop (a
  parity test pins this).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ....exceptions import ScspillDataError
from ..structures import SARPosterior
from ._kernels import FLO, _chol_sym, _clip, _ig, _rho_loglik, _sym, resolve_backend


def rho_stability_bound(Wn: np.ndarray) -> float:
    """Half-width of the default ``rho`` support, ``0.95 / max(1, max|eig(Wn)|)``.

    Matches the reference C++ sampler's spectral bound (computed from the
    eigenvalues of the row-normalized ``W``, not of ``A``). For a
    row-stochastic ``Wn`` the bound is 0.95.

    Parameters
    ----------
    Wn : np.ndarray
        Row-normalized control-to-control weight matrix, ``(N, N)``.

    Returns
    -------
    float
    """
    max_abs = float(np.max(np.abs(np.linalg.eigvals(Wn)))) if Wn.size else 0.0
    return 0.95 / max(1.0, max_abs)


@dataclass(frozen=True)
class SARData:
    """Fixed inputs of the Step-2 sampler (data, weights, priors, support)."""

    Yc: np.ndarray  # (T0, N)
    AYc: np.ndarray  # (T0, N) = (A @ Yc.T).T
    evA: np.ndarray  # (N,) complex eigenvalues of A
    X2: np.ndarray  # (T0*N, K); (0, 0) when K == 0
    K: int
    p: int
    rho_lo: float
    rho_hi: float
    a0: float
    b0: float
    beta_hs: bool
    step_rho: float

    @property
    def T0(self) -> int:
        """Number of pre-treatment periods."""
        return int(self.Yc.shape[0])

    @property
    def N(self) -> int:
        """Number of control units."""
        return int(self.Yc.shape[1])


@dataclass
class SARState:
    """Mutable parameter state of the Step-2 sampler (one point in the chain)."""

    rho: float
    s2: float
    beta: np.ndarray  # (K,)
    Eta: np.ndarray  # (N, p)
    Gamma: np.ndarray  # (p, T0)
    phi_g: float = 0.0
    s2_g: float = 1.0
    nu_s2_g: float = 1.0
    omega: np.ndarray = field(default_factory=lambda: np.ones(0))
    nu_omega: np.ndarray = field(default_factory=lambda: np.ones(0))
    s2_eta: float = 1.0
    nu_s2_eta: float = 1.0
    kappa2: np.ndarray = field(default_factory=lambda: np.ones(0))
    nu_kappa: np.ndarray = field(default_factory=lambda: np.ones(0))
    psi2: float = 1.0
    nu_psi: float = 1.0


def make_sar_data(
    Yc_pre: np.ndarray,
    alpha_hat: np.ndarray,
    wn: np.ndarray,
    Wn: np.ndarray,
    *,
    X: np.ndarray | None = None,
    p: int = 1,
    step_rho: float = 0.05,
    a0: float = 1.0,
    b0: float = 1.0,
    beta_prior: str = "horseshoe",
    rho_support: tuple[float, float] | None = None,
) -> SARData:
    """Precompute the fixed sampler inputs (``A``, its eigenvalues, support).

    Parameters
    ----------
    Yc_pre : np.ndarray
        Control pre-treatment outcomes, shape ``(T0, N)``.
    alpha_hat : np.ndarray
        Plug-in synthetic weights from Step 1, shape ``(N,)``.
    wn, Wn : np.ndarray
        Normalized spatial weights (``wn`` sums to one, ``Wn`` row-stochastic).
    X : np.ndarray, optional
        Covariate cube over the pre-period, shape ``(T0, N, K)``.
    p : int, default 1
        Number of AR(1) latent factors.
    step_rho : float, default 0.05
        Random-walk Metropolis step (initial value under adaptation).
    a0, b0 : float, default 1.0
        Inverse-gamma prior shape/scale for ``sigma^2``.
    beta_prior : {"horseshoe", "ridge"}, default "horseshoe"
        Prior on the covariate coefficients.
    rho_support : (float, float), optional
        Explicit support for ``rho``; defaults to the symmetric spectral
        bound ``(-b, b)`` with ``b = 0.95 / max(1, max|eig(Wn)|)``.

    Returns
    -------
    SARData
    """
    Yc_pre = np.asarray(Yc_pre, dtype=float)
    alpha_hat = np.asarray(alpha_hat, dtype=float).ravel()
    T0, N = Yc_pre.shape
    if alpha_hat.shape[0] != N:
        raise ScspillDataError(
            f"make_sar_data: alpha_hat has length {alpha_hat.shape[0]}, expected {N}."
        )
    A = Wn + np.outer(wn, alpha_hat)
    evA = np.linalg.eigvals(A).astype(np.complex128)
    if rho_support is None:
        bnd = rho_stability_bound(Wn)
        rho_lo, rho_hi = -bnd, bnd
    else:
        rho_lo, rho_hi = float(rho_support[0]), float(rho_support[1])
        if not rho_lo < rho_hi:
            raise ScspillDataError(f"make_sar_data: invalid rho_support ({rho_lo}, {rho_hi}).")
    K = 0
    X2 = np.zeros((0, 0))
    if X is not None:
        X = np.asarray(X, dtype=float)
        if X.shape[:2] != (T0, N):
            raise ScspillDataError(
                f"make_sar_data: X has shape {X.shape}, expected ({T0}, {N}, K)."
            )
        K = int(X.shape[2])
        if K > 0:
            X2 = X.reshape(T0 * N, K)
    return SARData(
        Yc=Yc_pre,
        AYc=(A @ Yc_pre.T).T,
        evA=evA,
        X2=X2,
        K=K,
        p=int(p),
        rho_lo=rho_lo,
        rho_hi=rho_hi,
        a0=float(a0),
        b0=float(b0),
        beta_hs=(beta_prior == "horseshoe"),
        step_rho=float(step_rho),
    )


def sar_step2_sampler(
    rng: np.random.Generator,
    Yc_pre: np.ndarray,
    alpha_hat: np.ndarray,
    wn: np.ndarray,
    Wn: np.ndarray,
    iters: int,
    burn: int,
    *,
    X: np.ndarray | None = None,
    p: int = 1,
    step_rho: float = 0.05,
    adapt_rho: bool = True,
    target_accept: float = 0.44,
    beta_prior: str = "horseshoe",
    a0: float = 1.0,
    b0: float = 1.0,
    rho_support: tuple[float, float] | None = None,
    kernels=None,
) -> SARPosterior:
    """Sample ``(rho, sigma^2, beta)`` from the SAR block given ``alpha_hat``.

    Parameters
    ----------
    rng : numpy.random.Generator
    Yc_pre : np.ndarray
        Control pre-treatment outcomes, shape ``(T0, N)``.
    alpha_hat : np.ndarray
        Plug-in synthetic weights from Step 1, shape ``(N,)``.
    wn, Wn : np.ndarray
        Normalized spatial weights.
    iters, burn : int
        Total iterations and burn-in.
    X : np.ndarray, optional
        Covariate cube over the pre-period, ``(T0, N, K)``.
    p : int, default 1
        Number of AR(1) latent factors (0 disables the block).
    step_rho : float, default 0.05
        Random-walk Metropolis step (initial value when ``adapt_rho``).
    adapt_rho : bool, default True
        Robbins-Monro adaptation of the log step during burn-in only.
    target_accept : float, default 0.44
        Adaptation target acceptance rate.
    beta_prior : {"horseshoe", "ridge"}, default "horseshoe"
        Prior on the covariate coefficients.
    a0, b0 : float, default 1.0
        Inverse-gamma prior for ``sigma^2``.
    rho_support : (float, float), optional
        Explicit ``rho`` support; defaults to the spectral bound.
    kernels : namespace, optional
        Kernel backend; defaults to the NumPy reference kernels.

    Returns
    -------
    SARPosterior
        Post-burn draws of ``rho`` / ``sigma^2`` / ``beta`` plus the
        acceptance rate and the (possibly adapted) final step size.
    """
    if kernels is None:
        kernels = resolve_backend("numpy")
    data = make_sar_data(
        Yc_pre,
        alpha_hat,
        wn,
        Wn,
        X=X,
        p=p,
        step_rho=step_rho,
        a0=a0,
        b0=b0,
        beta_prior=beta_prior,
        rho_support=rho_support,
    )
    rho_out, s2_out, beta_out, acc, step_final = kernels.sar_step2_loop(
        rng,
        data.Yc,
        data.AYc,
        data.evA,
        data.X2,
        data.K,
        data.p,
        int(iters),
        int(burn),
        float(step_rho),
        bool(adapt_rho),
        float(target_accept),
        data.rho_lo,
        data.rho_hi,
        data.a0,
        data.b0,
        data.beta_hs,
    )
    M = max(1, int(iters) - int(burn))
    return SARPosterior(
        rho=rho_out,
        sigma2=s2_out,
        beta=beta_out if data.K > 0 else None,
        rho_hat=float(rho_out.mean()),
        acc_rho=float(acc) / M,
        step_rho_final=float(step_final),
        rho_bound=float(max(abs(data.rho_lo), abs(data.rho_hi))),
        beta_prior=beta_prior,
        p_factors=int(p),
        iters=int(iters),
        burn=int(burn),
    )


# ---------------------------------------------------------------------------
# Geweke seams: prior state draws and a single transition sweep
# ---------------------------------------------------------------------------
def initial_state(data: SARData) -> SARState:
    """Build the batch loop's deterministic initial state (for parity testing).

    Parameters
    ----------
    data : SARData
        Fixed sampler inputs (provides the dimensions).

    Returns
    -------
    SARState
        ``rho = 0``, ``sigma^2 = 1``, zero coefficients/factors, unit scales
        -- exactly how the batch loop initializes.
    """
    return SARState(
        rho=0.0,
        s2=1.0,
        beta=np.zeros(data.K),
        Eta=np.zeros((data.N, data.p)),
        Gamma=np.zeros((data.p, data.T0)),
        phi_g=0.0,
        s2_g=1.0,
        nu_s2_g=1.0,
        omega=np.ones(data.p),
        nu_omega=np.ones(data.p),
        s2_eta=1.0,
        nu_s2_eta=1.0,
        kappa2=np.ones(data.K),
        nu_kappa=np.ones(data.K),
        psi2=1.0,
        nu_psi=1.0,
    )


def draw_prior_state(data: SARData, rng: np.random.Generator) -> SARState:
    """Draw the full Step-2 parameter state from its generative prior.

    The priors are the ones whose full conditionals the sampler implements:
    ``rho`` flat on ``(rho_lo, rho_hi)``; ``sigma^2 ~ IG(a0, b0)``; the AR(1)
    factor block with ``phi ~ U(-1, 1)``, half-Cauchy scales via their
    inverse-gamma mixture representations, stationary ``gamma_1``; loadings
    ``eta_i ~ N(0, s2_eta diag(omega)^{-1})``; and the horseshoe hierarchy on
    ``beta`` (or the near-flat ``N(0, 1e6)`` ridge prior).

    Parameters
    ----------
    data : SARData
        Fixed sampler inputs (dimensions, priors, support).
    rng : numpy.random.Generator

    Returns
    -------
    SARState
    """
    T0, N, K, p = data.T0, data.N, data.K, data.p

    state = SARState(
        rho=float(rng.uniform(data.rho_lo, data.rho_hi)),
        s2=_clip(_ig(rng, data.a0, data.b0)),
        beta=np.zeros(K),
        Eta=np.zeros((N, p)),
        Gamma=np.zeros((p, T0)),
    )

    if p > 0:
        state.phi_g = float(rng.uniform(-1.0, 1.0))
        state.nu_s2_g = _clip(_ig(rng, 0.5, 1.0 / 100.0))
        state.s2_g = _clip(_ig(rng, 0.5, 1.0 / state.nu_s2_g))
        # omega_k ~ C+(0, 10) via the inverse-gamma mixture (the paper's prior).
        state.nu_omega = np.array([_clip(_ig(rng, 0.5, 1.0 / 100.0)) for _ in range(p)])
        state.omega = np.array([_clip(_ig(rng, 0.5, 1.0 / state.nu_omega[k])) for k in range(p)])
        state.nu_s2_eta = _clip(_ig(rng, 0.5, 1.0 / 100.0))
        state.s2_eta = _clip(_ig(rng, 0.5, 1.0 / state.nu_s2_eta))
        eta_sd = np.sqrt(state.s2_eta * state.omega)  # var = s2_eta * omega_k
        state.Eta = rng.standard_normal((N, p)) * eta_sd[None, :]
        # gamma_0 = 0: gamma_1 ~ N(0, s2_g I), then the AR(1) recursion.
        g = np.empty((p, T0))
        sd_g = np.sqrt(_clip(state.s2_g))
        g[:, 0] = sd_g * rng.standard_normal(p)
        for t in range(1, T0):
            g[:, t] = state.phi_g * g[:, t - 1] + sd_g * rng.standard_normal(p)
        state.Gamma = g

    if K > 0:
        if data.beta_hs:
            state.nu_psi = _clip(_ig(rng, 0.5, 1.0 / 100.0))
            state.psi2 = _clip(_ig(rng, 0.5, 1.0 / state.nu_psi))
            state.nu_kappa = np.array([_clip(_ig(rng, 0.5, 1.0 / state.psi2)) for _ in range(K)])
            state.kappa2 = np.array(
                [_clip(_ig(rng, 0.5, 1.0 / state.nu_kappa[k])) for k in range(K)]
            )
            state.beta = rng.standard_normal(K) * np.sqrt(state.kappa2)
        else:
            state.beta = rng.standard_normal(K) * 1e3  # ridge prior N(0, 1e6)
            state.kappa2 = np.ones(K)
            state.nu_kappa = np.ones(K)
    return state


def one_sweep(state: SARState, data: SARData, rng: np.random.Generator) -> SARState:
    """One full Gibbs/Metropolis sweep of the Step-2 sampler (in place).

    Mirrors one iteration of the batch loop in
    :mod:`scspill.utils.scspill_helpers.sar._kernels` exactly -- same block
    order, same conditionals, same random-generator consumption -- with a
    fixed Metropolis step (no adaptation; the Geweke test requires a fixed
    Markov transition kernel). A parity test asserts draw-for-draw equality
    with the batch loop.

    Parameters
    ----------
    state : SARState
        Current parameter state; mutated in place.
    data : SARData
        Fixed sampler inputs.
    rng : numpy.random.Generator

    Returns
    -------
    SARState
        The same (mutated) ``state``, for chaining.
    """
    Yc, AYc, evA, X2 = data.Yc, data.AYc, data.evA, data.X2
    T0, N, K, p = data.T0, data.N, data.K, data.p
    I_p = np.eye(p) if p > 0 else np.zeros((0, 0))
    I_K = np.eye(K) if K > 0 else np.zeros((0, 0))
    XtX = X2.T @ X2 if K > 0 else np.zeros((0, 0))

    rho, s2 = state.rho, state.s2
    beta, Eta, Gamma = state.beta, state.Eta, state.Gamma

    Xbeta = (X2 @ beta).reshape(T0, N) if K > 0 else np.zeros((T0, N))

    # (1) AR(1) factors Gamma via forward-filter/backward-sample
    if p > 0:
        phi_g, s2_g = state.phi_g, state.s2_g
        Ystar = Yc - rho * AYc - Xbeta
        EtaT = Eta.T.copy()
        HtH = (EtaT @ Eta) / _clip(s2)
        Ht = EtaT / _clip(s2)

        m_f = np.zeros((T0, p))
        V_f = np.zeros((T0, p, p))
        pred_f = np.zeros((T0, p))
        Ppred_f = np.zeros((T0, p, p))
        gprev = np.zeros(p)
        Pprev = 0.0 * I_p  # gamma_0 = 0 -> gamma_1 ~ N(0, s2_g I); see _kernels

        for t in range(T0):
            pred = phi_g * gprev
            Ppred = (phi_g * phi_g) * Pprev + _clip(s2_g) * I_p
            Pi = np.linalg.inv(_sym(Ppred))
            V = np.linalg.inv(_sym(Pi + HtH))
            m = V @ (Pi @ pred + Ht @ Ystar[t])
            m_f[t] = m
            V_f[t] = V
            pred_f[t] = pred
            Ppred_f[t] = Ppred
            gprev = m
            Pprev = V
        Gamma[:, T0 - 1] = m_f[T0 - 1] + _chol_sym(V_f[T0 - 1]) @ rng.standard_normal(p)
        for t in range(T0 - 2, -1, -1):
            J = (V_f[t] * phi_g) @ np.linalg.inv(_sym(Ppred_f[t + 1]))
            ms = m_f[t] + J @ (Gamma[:, t + 1] - pred_f[t + 1])
            Vs = V_f[t] - phi_g * (J @ V_f[t])
            Gamma[:, t] = ms + _chol_sym(Vs) @ rng.standard_normal(p)

        den = 0.0
        num = 0.0
        for t in range(1, T0):
            gl = Gamma[:, t - 1]
            den += float(gl @ gl)
            num += float(gl @ Gamma[:, t])
        mean_phi = num / den if den > 0 else 0.0
        var_phi = _clip(s2_g) / den if den > 0 else 1.0
        sd_phi = np.sqrt(var_phi)
        cand = rng.normal(mean_phi, sd_phi)
        while abs(cand) > 1.0:
            cand = rng.normal(mean_phi, sd_phi)
        state.phi_g = phi_g = cand
        sc_g = 0.0
        for t in range(T0):
            diff = Gamma[:, 0] if t == 0 else Gamma[:, t] - phi_g * Gamma[:, t - 1]
            sc_g += 0.5 * float(diff @ diff)
        state.s2_g = s2_g = _clip(_ig(rng, 0.5 + 0.5 * p * T0, sc_g + 1.0 / _clip(state.nu_s2_g)))
        state.nu_s2_g = _clip(_ig(rng, 1.0, 1.0 / _clip(s2_g) + 1.0 / 100.0))

        # (2) factor loadings Eta: eta_i ~ N(0, s2_eta diag(omega)); see _kernels
        GtG = Gamma @ Gamma.T.copy()
        Dom_inv = np.diag(1.0 / np.maximum(state.omega, FLO))
        Vrow = np.linalg.inv(_sym(GtG / _clip(s2) + Dom_inv / _clip(state.s2_eta)))
        Lrow = _chol_sym(Vrow)
        RHS = Gamma @ Ystar  # (p, N)
        Z = rng.standard_normal((p, N))
        state.Eta = Eta = (Vrow @ (RHS / _clip(s2)) + Lrow @ Z).T.copy()

        sc_eta = 0.0
        for i in range(N):
            ei = Eta[i]
            sc_eta += float(ei @ (Dom_inv @ ei))
        state.s2_eta = _clip(
            _ig(rng, 0.5 + 0.5 * p * N, 0.5 * sc_eta + 1.0 / _clip(state.nu_s2_eta))
        )
        state.nu_s2_eta = _clip(_ig(rng, 1.0, 1.0 / _clip(state.s2_eta) + 1.0 / 100.0))
        for k in range(p):
            tmp = 0.0
            for i in range(N):
                tmp += 0.5 * Eta[i, k] * Eta[i, k] / _clip(state.s2_eta)
            state.omega[k] = _clip(_ig(rng, 0.5 * (N + 1.0), 1.0 / _clip(state.nu_omega[k]) + tmp))
            # 1/100 encodes the paper's omega_k ~ C+(0, 10); see _kernels.
            state.nu_omega[k] = _clip(_ig(rng, 1.0, 1.0 / 100.0 + 1.0 / _clip(state.omega[k])))

    EG = (Eta @ Gamma).T.copy() if p > 0 else np.zeros((T0, N))

    # (3) covariate coefficients beta
    if K > 0:
        Bt = Yc - rho * AYc - EG
        Bb = X2.T @ Bt.ravel()
        if data.beta_hs:
            Ab = XtX.copy()
            for k in range(K):
                Ab[k, k] += _clip(s2) / _clip(state.kappa2[k])
        else:
            Ab = XtX + (_clip(s2) * 1e-6) * I_K
        Ainv = _sym(np.linalg.inv(_sym(Ab)))
        state.beta = beta = Ainv @ Bb + _chol_sym(_clip(s2) * Ainv) @ rng.standard_normal(K)
        if data.beta_hs:
            for k in range(K):
                state.kappa2[k] = _clip(
                    _ig(rng, 1.0, 0.5 * beta[k] * beta[k] + 1.0 / max(state.nu_kappa[k], FLO))
                )
            for k in range(K):
                state.nu_kappa[k] = _clip(
                    _ig(rng, 1.0, 1.0 / max(state.kappa2[k], FLO) + 1.0 / max(state.psi2, FLO))
                )
            sc_psi = 1.0 / max(state.nu_psi, FLO)
            for k in range(K):
                sc_psi += 1.0 / _clip(state.nu_kappa[k])
            state.psi2 = _clip(_ig(rng, 0.5 * (K + 1.0), sc_psi))
            state.nu_psi = _clip(_ig(rng, 1.0, 1.0 / max(state.psi2, FLO) + 1.0 / 100.0))
        Xbeta = (X2 @ beta).reshape(T0, N)

    # (4) sigma^2
    Resid0 = Yc - Xbeta - EG
    U = Resid0 - rho * AYc
    ss = float(np.sum(U * U))
    state.s2 = s2 = _ig(rng, data.a0 + 0.5 * (T0 * N), data.b0 + 0.5 * ss)

    # (5) rho via random-walk Metropolis (fixed step)
    c0 = float(np.sum(Resid0 * Resid0))
    c1 = float(np.sum(Resid0 * AYc))
    c2 = float(np.sum(AYc * AYc))
    lcur = _rho_loglik(rho, evA, c0, c1, c2, s2, T0, N, data.rho_lo, data.rho_hi)
    # exp(log(step)) mirrors the batch loop's arithmetic bit-for-bit, so the
    # one_sweep/batch parity test can assert exact equality.
    prop = rho + np.exp(np.log(data.step_rho)) * rng.standard_normal()
    lprp = _rho_loglik(prop, evA, c0, c1, c2, s2, T0, N, data.rho_lo, data.rho_hi)
    if np.log(rng.random()) < lprp - lcur:
        state.rho = prop
    return state
