"""Sampler loop kernels for the SCSPILL estimator.

The two hot MCMC loops -- the Step-1 horseshoe Gibbs sampler and the Step-2
SAR sampler -- written in a numba-compatible NumPy style (no dicts, no
``einsum``, preallocated arrays, scalar inverse-gamma draws). The same source
functions serve both backends:

* ``"numpy"`` uses the plain Python functions below (the reference path);
* ``"numba"`` wraps the very same functions with ``numba.njit`` at dispatch
  time, so the two backends share one implementation and cannot drift.

:func:`resolve_backend` performs the dispatch. Compilation is attempted once
and cached; under ``backend="auto"`` any numba failure (missing package,
unsupported feature on an old numba) silently falls back to NumPy, while an
explicit ``backend="numba"`` raises.

Everything above the loops (standardization, eigenvalue precomputation,
structure assembly) lives in :mod:`sampler_alpha` / :mod:`sampler_sar`.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np

from ...exceptions import ScspillConfigError

#: Universal clipping bounds for scale parameters, matching the C++ reference.
FLO = 1e-12
FHI = 1e12

try:  # pragma: no cover - depends on the optional numba extra
    # ``register_jitable`` keeps a helper usable as plain Python while letting
    # numba inline it when the loop functions are njit-compiled -- the bridge
    # that makes the single-source dual-backend design work.
    from numba.extending import register_jitable as _jitable
except ImportError:  # pragma: no cover

    def _jitable(func):
        """Identity decorator when numba is not installed."""
        return func


@_jitable
def _ig(rng, shape, scale):
    """Draw from an inverse-gamma(shape, scale) distribution (scalar)."""
    return 1.0 / rng.gamma(shape, 1.0 / max(scale, 1e-300))


@_jitable
def _clip(x):
    """Clip a scalar to the universal ``[FLO, FHI]`` stability bounds."""
    return min(max(x, FLO), FHI)


@_jitable
def _sym(A):
    """Symmetrize a square matrix, returning a C-contiguous result.

    The explicit copies keep every operand contiguous so numba's fused array
    expressions never need an any-layout ('A') allocation.
    """
    B = A.copy()
    return 0.5 * (B + B.T.copy())


@_jitable
def _chol_sym(A, jitter=1e-10):
    """Cholesky of a symmetrized matrix with a small fixed jitter.

    The matrices factorized inside the loops (posterior covariances) are
    positive definite by construction; the jitter guards against rounding.
    """
    n = A.shape[0]
    return np.linalg.cholesky(_sym(A) + jitter * np.eye(n))


# ---------------------------------------------------------------------------
# Step 1: horseshoe Gibbs for alpha (standardized scale)
# ---------------------------------------------------------------------------
def hs_alpha_loop(rng, ys, Xs, iters, burn):
    """Horseshoe-Gibbs loop for the synthetic weights on standardized data.

    Exact port of the reference ``hs_alpha_gibbs_cpp`` (Makalic & Schmidt
    2015 auxiliary-variable representation; all full conditionals closed
    form). ``ys`` and ``Xs`` are already scaled by their standard deviations
    (no centering); the caller back-transforms the draws.

    Parameters
    ----------
    rng : numpy.random.Generator
    ys : np.ndarray
        Standardized treated pre-treatment outcomes, shape ``(T0,)``.
    Xs : np.ndarray
        Standardized control pre-treatment outcomes, shape ``(T0, N)``.
    iters, burn : int
        Total iterations and burn-in.

    Returns
    -------
    (np.ndarray, np.ndarray)
        ``alpha`` draws on the standardized scale, shape
        ``(iters - burn, N)``, and the error-variance draws ``(iters - burn,)``.
    """
    T0, N = Xs.shape
    XtX = Xs.T @ Xs
    Xty = Xs.T @ ys
    I_N = np.eye(N)

    s2i = np.ones(N)
    nus_i = np.ones(N)
    tau2 = 1.0
    nu_tau = 1.0
    s2 = np.var(ys) * T0 / max(T0 - 1, 1)  # ddof=1 sample variance
    if not np.isfinite(s2) or s2 <= 0.0:
        s2 = 1.0
    nu_s = 1.0

    M = iters - burn
    alpha_out = np.empty((M, N))
    s2_out = np.empty(M)

    for it in range(iters):
        # alpha | rest ~ N(D^-1 X'y, s2 D^-1), D = X'X + s2 diag(1/s2_i)
        sig2c = _clip(s2)
        D = XtX.copy()
        for j in range(N):
            D[j, j] += sig2c / _clip(s2i[j])
        Dinv = _sym(np.linalg.solve(D, I_N))
        L = _chol_sym(max(sig2c, 1e-12) * Dinv)
        alpha = Dinv @ Xty + L @ rng.standard_normal(N)

        # local scales and their auxiliaries (Makalic-Schmidt)
        for j in range(N):
            s2i[j] = _clip(_ig(rng, 1.0, 0.5 * alpha[j] * alpha[j] + 1.0 / max(nus_i[j], FLO)))
        for j in range(N):
            nus_i[j] = _clip(_ig(rng, 1.0, 1.0 / max(s2i[j], FLO) + 1.0 / max(tau2, FLO)))

        # global scale and its auxiliary
        sc_tau = 1.0 / max(nu_tau, FLO)
        for j in range(N):
            sc_tau += 1.0 / _clip(nus_i[j])
        tau2 = _clip(_ig(rng, 0.5 * (N + 1.0), sc_tau))
        nu_tau = _clip(_ig(rng, 1.0, 1.0 / max(tau2, FLO) + 1.0 / max(s2, FLO)))

        # error variance and the C+(0, 10) auxiliary
        res = ys - Xs @ alpha
        sse = float(res @ res)
        s2 = _clip(
            _ig(rng, 1.0 + 0.5 * T0, 1.0 / max(nu_tau, FLO) + 1.0 / max(nu_s, FLO) + 0.5 * sse)
        )
        nu_s = _clip(_ig(rng, 1.0, 1.0 / max(s2, FLO) + 1.0 / 100.0))

        if it >= burn:
            alpha_out[it - burn] = alpha
            s2_out[it - burn] = s2

    return alpha_out, s2_out


# ---------------------------------------------------------------------------
# Step 2: SAR sampler conditional on alpha_hat
# ---------------------------------------------------------------------------
def sar_step2_loop(
    rng,
    Yc,
    AYc,
    evA,
    X2,
    K,
    p,
    iters,
    burn,
    step_rho0,
    adapt,
    target_accept,
    rho_lo,
    rho_hi,
    a0,
    b0,
    beta_hs,
):
    """Step-2 SAR Gibbs/Metropolis loop conditional on ``alpha_hat``.

    Samples the latent AR(1) factor block (forward-filter/backward-sample),
    the covariate coefficients ``beta`` (horseshoe or ridge prior), the
    innovation variance ``sigma2``, and the spillover intensity ``rho``
    (random-walk Metropolis with the ``log|I - rho A|`` Jacobian evaluated in
    O(N) from the precomputed eigenvalues of ``A = W + w alpha'``), in the
    reference C++ block order.

    Parameters
    ----------
    rng : numpy.random.Generator
    Yc : np.ndarray
        Control pre-treatment outcomes, shape ``(T0, N)``.
    AYc : np.ndarray
        Precomputed ``(A @ Yc.T).T``, shape ``(T0, N)``.
    evA : np.ndarray
        Complex eigenvalues of ``A``, shape ``(N,)``.
    X2 : np.ndarray
        Covariates flattened to ``(T0 * N, K)`` (row-major over ``(t, i)``);
        pass ``np.zeros((0, 0))`` when ``K == 0``.
    K, p : int
        Number of covariates and latent factors.
    iters, burn : int
        Total iterations and burn-in.
    step_rho0 : float
        Initial random-walk Metropolis step for ``rho``.
    adapt : bool
        Robbins-Monro adaptation of the log step during burn-in.
    target_accept : float
        Target acceptance rate for the adaptation.
    rho_lo, rho_hi : float
        Support of ``rho``: proposals outside ``(rho_lo, rho_hi)`` are
        rejected (flat prior on the interval).
    a0, b0 : float
        Inverse-gamma prior shape/scale for ``sigma2``.
    beta_hs : bool
        Horseshoe prior on ``beta`` (True, the paper) or flat-plus-ridge
        (False, the R code).

    Returns
    -------
    tuple
        ``(rho_draws, s2_draws, beta_draws, acc_count, step_final)`` with
        post-burn draw arrays of shapes ``(M,)``, ``(M,)``, ``(M, K)``, the
        post-burn acceptance count (int), and the final step size (float).
    """
    T0, N = Yc.shape
    M = iters - burn

    XtX = X2.T @ X2 if K > 0 else np.zeros((0, 0))
    I_K = np.eye(K) if K > 0 else np.zeros((0, 0))
    I_p = np.eye(p) if p > 0 else np.zeros((0, 0))

    # states
    rho = 0.0
    s2 = 1.0
    beta = np.zeros(K)
    Eta = np.zeros((N, p))
    Gamma = np.zeros((p, T0))
    phi_g = 0.0
    s2_g = 1.0
    nu_s2_g = 1.0
    omega = np.ones(p)
    nu_omega = np.ones(p)
    s2_eta = 1.0
    nu_s2_eta = 1.0
    # horseshoe-on-beta auxiliaries
    kappa2 = np.ones(K)
    nu_kappa = np.ones(K)
    psi2 = 1.0
    nu_psi = 1.0

    log_step = np.log(step_rho0)
    log_step_lo = np.log(1e-6)
    log_step_hi = 0.0

    rho_out = np.empty(M)
    s2_out = np.empty(M)
    beta_out = np.zeros((M, K))
    acc = 0

    for it in range(iters):
        Xbeta = (X2 @ beta).reshape(T0, N) if K > 0 else np.zeros((T0, N))

        # (1) AR(1) factors Gamma via forward-filter/backward-sample
        if p > 0:
            Ystar = Yc - rho * AYc - Xbeta
            EtaT = Eta.T.copy()
            HtH = (EtaT @ Eta) / _clip(s2)
            Ht = EtaT / _clip(s2)

            m_f = np.zeros((T0, p))
            V_f = np.zeros((T0, p, p))
            pred_f = np.zeros((T0, p))
            Ppred_f = np.zeros((T0, p, p))
            gprev = np.zeros(p)
            # gamma_0 = 0, so gamma_1 ~ N(0, s2_g I): P_0 = 0. (The C++ used a
            # stationary P_0 while its phi/s2_g conditionals assume gamma_1 ~
            # N(0, s2_g) -- mutually inconsistent; the Geweke joint
            # distribution test flags that combination, so the coherent
            # initialization is used here.)
            Pprev = 0.0 * I_p
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
                g_next = Gamma[:, t + 1].copy()
                ms = m_f[t] + J @ (g_next - pred_f[t + 1])
                Vs = V_f[t] - phi_g * (J @ V_f[t])
                Gamma[:, t] = ms + _chol_sym(Vs) @ rng.standard_normal(p)

            den = 0.0
            num = 0.0
            for t in range(1, T0):
                gl = Gamma[:, t - 1].copy()
                den += float(gl @ gl)
                num += float(gl @ Gamma[:, t].copy())
            mean_phi = num / den if den > 0 else 0.0
            var_phi = _clip(s2_g) / den if den > 0 else 1.0
            sd_phi = np.sqrt(var_phi)
            cand = rng.normal(mean_phi, sd_phi)
            while abs(cand) > 1.0:
                cand = rng.normal(mean_phi, sd_phi)
            phi_g = cand
            sc_g = 0.0
            for t in range(T0):
                if t == 0:
                    diff = Gamma[:, 0].copy()
                else:
                    diff = Gamma[:, t].copy() - phi_g * Gamma[:, t - 1].copy()
                sc_g += 0.5 * float(diff @ diff)
            s2_g = _clip(_ig(rng, 0.5 + 0.5 * p * T0, sc_g + 1.0 / _clip(nu_s2_g)))
            nu_s2_g = _clip(_ig(rng, 1.0, 1.0 / _clip(s2_g) + 1.0 / 100.0))

            # (2) factor loadings Eta (shared row covariance).
            # Paper parametrization: eta_i ~ N(0, s2_eta * diag(omega)), so the
            # prior precision is diag(1/omega) / s2_eta. (The C++ used
            # diag(omega) here while its omega update assumed the paper's
            # form -- incoherent conditionals flagged by the Geweke test.)
            GtG = Gamma @ Gamma.T.copy()
            Dom_inv = np.diag(1.0 / np.maximum(omega, FLO))
            Vrow = np.linalg.inv(_sym(GtG / _clip(s2) + Dom_inv / _clip(s2_eta)))
            Lrow = _chol_sym(Vrow)
            RHS = Gamma @ Ystar  # (p, N)
            Z = rng.standard_normal((p, N))
            Eta = (Vrow @ (RHS / _clip(s2)) + Lrow @ Z).T.copy()

            sc_eta = 0.0
            for i in range(N):
                ei = Eta[i]
                sc_eta += float(ei @ (Dom_inv @ ei))
            s2_eta = _clip(_ig(rng, 0.5 + 0.5 * p * N, 0.5 * sc_eta + 1.0 / _clip(nu_s2_eta)))
            nu_s2_eta = _clip(_ig(rng, 1.0, 1.0 / _clip(s2_eta) + 1.0 / 100.0))
            for k in range(p):
                tmp = 0.0
                for i in range(N):
                    tmp += 0.5 * Eta[i, k] * Eta[i, k] / _clip(s2_eta)
                omega[k] = _clip(_ig(rng, 0.5 * (N + 1.0), 1.0 / _clip(nu_omega[k]) + tmp))
                # 1/100 encodes the paper's omega_k ~ C+(0, 10); the C++
                # reference used 1.0 (C+(0, 1)) here, unlike its own sibling
                # s2_eta hierarchy.
                nu_omega[k] = _clip(_ig(rng, 1.0, 1.0 / 100.0 + 1.0 / _clip(omega[k])))

        EG = (Eta @ Gamma).T.copy() if p > 0 else np.zeros((T0, N))

        # (3) covariate coefficients beta
        if K > 0:
            Bt = Yc - rho * AYc - EG
            Bb = X2.T @ Bt.ravel()
            if beta_hs:
                Ab = XtX.copy()
                for k in range(K):
                    Ab[k, k] += _clip(s2) / _clip(kappa2[k])
            else:
                Ab = XtX + (_clip(s2) * 1e-6) * I_K
            Ainv = _sym(np.linalg.inv(_sym(Ab)))
            beta = Ainv @ Bb + _chol_sym(_clip(s2) * Ainv) @ rng.standard_normal(K)
            if beta_hs:
                for k in range(K):
                    kappa2[k] = _clip(
                        _ig(rng, 1.0, 0.5 * beta[k] * beta[k] + 1.0 / max(nu_kappa[k], FLO))
                    )
                for k in range(K):
                    nu_kappa[k] = _clip(
                        _ig(rng, 1.0, 1.0 / max(kappa2[k], FLO) + 1.0 / max(psi2, FLO))
                    )
                sc_psi = 1.0 / max(nu_psi, FLO)
                for k in range(K):
                    sc_psi += 1.0 / _clip(nu_kappa[k])
                psi2 = _clip(_ig(rng, 0.5 * (K + 1.0), sc_psi))
                nu_psi = _clip(_ig(rng, 1.0, 1.0 / max(psi2, FLO) + 1.0 / 100.0))
            Xbeta = (X2 @ beta).reshape(T0, N)

        # (4) sigma^2
        Resid0 = Yc - Xbeta - EG
        U = Resid0 - rho * AYc
        ss = float(np.sum(U * U))
        s2 = _ig(rng, a0 + 0.5 * (T0 * N), b0 + 0.5 * ss)

        # (5) rho via random-walk Metropolis (optional burn-in adaptation)
        c0 = float(np.sum(Resid0 * Resid0))
        c1 = float(np.sum(Resid0 * AYc))
        c2 = float(np.sum(AYc * AYc))

        lcur = _rho_loglik(rho, evA, c0, c1, c2, s2, T0, N, rho_lo, rho_hi)
        step = np.exp(log_step)
        prop = rho + step * rng.standard_normal()
        lprp = _rho_loglik(prop, evA, c0, c1, c2, s2, T0, N, rho_lo, rho_hi)
        loga = lprp - lcur
        if np.log(rng.random()) < loga:
            rho = prop
            if it >= burn:
                acc += 1
        if adapt and it < burn:
            a_prob = np.exp(min(0.0, loga)) if np.isfinite(loga) else 0.0
            log_step += (it + 1.0) ** (-0.6) * (a_prob - target_accept)
            log_step = min(max(log_step, log_step_lo), log_step_hi)

        if it >= burn:
            m = it - burn
            rho_out[m] = rho
            s2_out[m] = s2
            if K > 0:
                beta_out[m] = beta

    return rho_out, s2_out, beta_out, acc, np.exp(log_step)


@_jitable
def _rho_loglik(r, evA, c0, c1, c2, s2, T0, N, rho_lo, rho_hi):
    """Profile log-likelihood of ``rho`` given the residual quadratic form.

    ``ss(r) = c0 - 2 r c1 + r^2 c2`` and the Jacobian term is
    ``T0 * sum(log |1 - r lambda_i(A)|)`` from the cached complex eigenvalues.
    Returns ``-inf`` outside the support ``(rho_lo, rho_hi)``.
    """
    if r <= rho_lo or r >= rho_hi:
        return -np.inf
    ldet = float(np.sum(np.log(np.abs(1.0 - r * evA))))
    if not np.isfinite(ldet):
        return -np.inf
    ss = c0 - 2.0 * r * c1 + r * r * c2
    return T0 * ldet - 0.5 * (N * T0) * np.log(s2) - 0.5 * ss / s2


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------
_NUMPY_KERNELS = SimpleNamespace(
    name="numpy",
    hs_alpha_loop=hs_alpha_loop,
    sar_step2_loop=sar_step2_loop,
)

_numba_kernels = None
_numba_error: Exception | None = None


def _build_numba_kernels():
    """Compile the loop kernels with numba (cached module-level)."""
    global _numba_kernels, _numba_error
    if _numba_kernels is not None:
        return _numba_kernels
    if _numba_error is not None:
        raise _numba_error
    try:
        from .numba_kernels import build_kernels

        _numba_kernels = build_kernels()
        return _numba_kernels
    except Exception as exc:  # ImportError or a numba compile failure
        _numba_error = exc
        raise


def resolve_backend(requested: str = "auto") -> SimpleNamespace:
    """Resolve the sampler kernel backend.

    Parameters
    ----------
    requested : {"auto", "numpy", "numba"}, default "auto"
        ``"numpy"`` returns the reference kernels; ``"numba"`` returns
        JIT-compiled versions of the same source functions (raising
        :class:`ScspillConfigError` if numba is unavailable or compilation
        fails); ``"auto"`` prefers numba when importable and silently falls
        back to NumPy otherwise.

    Returns
    -------
    types.SimpleNamespace
        With attributes ``name``, ``hs_alpha_loop``, ``sar_step2_loop``.
    """
    if requested == "numpy":
        return _NUMPY_KERNELS
    if requested == "numba":
        try:
            return _build_numba_kernels()
        except Exception as exc:
            raise ScspillConfigError(
                "backend='numba' was requested but the numba kernels are unavailable: "
                f"{exc}. Install the extra with `pip install 'scspill[numba]'`."
            ) from exc
    if requested == "auto":
        import importlib.util

        if importlib.util.find_spec("numba") is not None:
            try:
                return _build_numba_kernels()
            except Exception as exc:  # pragma: no cover - depends on numba version
                warnings.warn(
                    f"numba is installed but kernel compilation failed ({exc}); "
                    "falling back to the NumPy backend.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return _NUMPY_KERNELS
        return _NUMPY_KERNELS
    raise ScspillConfigError(f"Unknown backend {requested!r}.")
