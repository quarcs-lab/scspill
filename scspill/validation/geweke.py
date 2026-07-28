"""Geweke (2004) joint distribution test of the Step-2 sampler.

Compares two simulators of the joint distribution ``p(theta, y)``:

* the *marginal-conditional* simulator draws ``theta ~ p(theta)`` and
  ``y ~ p(y | theta)`` independently each iteration (exact iid draws);
* the *successive-conditional* simulator alternates ``y_m ~ p(y |
  theta_{m-1})`` with one full sweep of the posterior transition kernel
  ``theta_m ~ K(theta | theta_{m-1}, y_m)``.

If the sampler's conditionals are mutually consistent with the priors, both
simulators target the same joint distribution, so every test statistic
``g(theta, y)`` must agree in expectation. The z-statistics compare the two
means with an iid standard error on one side and a batch-means MCMC standard
error on the other.

This port fixes one inconsistency in the R reference: there, the
marginal-conditional prior drew ``rho`` on the spectral bound of ``A`` while
the transition kernel truncated ``rho`` to a support derived from ``W`` --
two different joints by construction. Here one canonical support (the
kernel's ``rho_support``) is used on both sides.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from threadpoolctl import threadpool_limits

from ..exceptions import ScspillConfigError
from .kernels import ProductionKernel, SimpleKernel
from .structures import GewekeReport


def batch_means_variance(x: np.ndarray, batch_size: int | None = None) -> float:
    """Batch-means estimate of the variance of an MCMC sample mean.

    Non-finite values are dropped. With batch size ``b`` (default
    ``max(2, floor(sqrt(M)))``) and ``a = floor(M / b)`` full batches, the
    estimate is ``b * var(batch means, ddof=1) / (a * b)``; when fewer than
    two batches fit, the naive ``var / M`` is returned. Exact port of the R
    ``var_mcmc_batchmeans``.

    Parameters
    ----------
    x : np.ndarray
        The chain.
    batch_size : int, optional
        Batch length ``b``.

    Returns
    -------
    float
        The estimated variance of ``mean(x)``.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    M = x.size
    if M == 0:
        return float("nan")
    if batch_size is None:
        batch_size = max(2, int(np.floor(np.sqrt(M))))
    b = int(batch_size)
    a = M // b
    if a < 2:
        return float(np.var(x, ddof=1) / M) if M > 1 else float("nan")
    bm = x[: a * b].reshape(a, b).mean(axis=1)
    tau2 = b * float(np.var(bm, ddof=1))
    return tau2 / (a * b)


def default_g_fn(
    summary: dict,
    Yc: np.ndarray,
    Y0_pre: np.ndarray,
    Wn: np.ndarray,
    wn: np.ndarray,
) -> dict:
    """Compute the reference set of Geweke test statistics ``g(theta, y)``.

    Port of the R ``default_g_fn``: the spillover intensity, log error
    variance, panel mean and log variance, the spatial quadratic form
    ``tr(Yc W Yc') / (N T0)``, the correlation between the pseudo treated
    series and the exposure-weighted panel, and the means of ``beta`` /
    ``Eta`` / ``Gamma`` (NaN when the block is absent).

    Parameters
    ----------
    summary : dict
        Named parameter blocks from the kernel's ``state_summary``.
    Yc : np.ndarray
        Simulated panel, shape ``(T0, N)``.
    Y0_pre : np.ndarray
        Fixed pseudo treated pre-period series, shape ``(T0,)``.
    Wn, wn : np.ndarray
        Normalized spatial weights.

    Returns
    -------
    dict
        Statistic name -> value.
    """
    yc_vec = Yc.ravel()
    T0, N = Yc.shape
    wyc = Yc @ wn
    spatial_q = float(np.trace(Yc @ Wn @ Yc.T)) / (N * T0)
    sd0 = float(np.std(Y0_pre, ddof=1))
    sdw = float(np.std(wyc, ddof=1))
    corr = float(np.corrcoef(Y0_pre, wyc)[0, 1]) if sd0 > 0 and sdw > 0 else float("nan")

    beta = np.asarray(summary.get("beta", np.zeros(0)))
    eta = np.asarray(summary.get("Eta", np.zeros(0)))
    gamma = np.asarray(summary.get("Gamma", np.zeros(0)))
    return {
        "rho": float(summary["rho"]),
        "log_sigma2": float(np.log(max(summary["sigma2"], 1e-12))),
        "yc_mean": float(yc_vec.mean()),
        "log_yc_var": float(np.log(max(yc_vec.var(ddof=1), 1e-12))),
        "spatial_quadratic": spatial_q,
        "corr_y0_wyc": corr,
        "beta_mean": float(beta.mean()) if beta.size else float("nan"),
        "Eta_mean": float(eta.mean()) if eta.size else float("nan"),
        "Gamma_mean": float(gamma.mean()) if gamma.size else float("nan"),
    }


def _chain_graph(N: int) -> np.ndarray:
    """Path-graph adjacency (the R Geweke driver's toy ``W``)."""
    W = np.zeros((N, N))
    for i in range(N - 1):
        W[i, i + 1] = W[i + 1, i] = 1.0
    return W


def geweke_test(
    *,
    kernel: str | object = "simple",
    T0: int = 15,
    N: int = 8,
    K: int = 2,
    p: int = 1,
    spatial_W: np.ndarray | None = None,
    spatial_w: np.ndarray | None = None,
    alpha: np.ndarray | None = None,
    X: np.ndarray | None = None,
    y0_pre: np.ndarray | None = None,
    m_iid: int = 20_000,
    m_mcmc: int = 20_000,
    burn: int = 5_000,
    a0: float = 1.0,
    b0: float = 1.0,
    step_rho: float = 0.05,
    rho_support: tuple[float, float] | None = None,
    beta_prior: str = "horseshoe",
    g_fn: Callable | None = None,
    batch_size: int | None = None,
    alpha_level: float = 0.05,
    bonferroni: bool = True,
    seed: int | None = None,
    verbose: bool = False,
) -> GewekeReport:
    """Run the Geweke joint distribution test of the Step-2 sampler.

    Defaults mirror the replication package's Geweke driver: a
    ``T0 = 15 x N = 8`` panel on a chain graph with ``w = e_1``, ``K = 2``
    iid standard-normal covariates, ``p = 1`` latent factor, and standardized
    synthetic weights drawn ``N(0, 0.4^2)``.

    Parameters
    ----------
    kernel : {"simple", "production"} or kernel object, default "simple"
        ``"simple"`` tests the appendix's simplified model (comparable to the
        R package's frozen table); ``"production"`` tests the sampler users
        actually run. A custom object implementing ``draw_prior`` /
        ``simulate_data`` / ``transition`` / ``state_summary`` /
        ``extra_stats`` (see :mod:`scspill.validation.kernels`) is accepted.
    T0, N, K, p : int
        Panel and model dimensions (ignored for a custom kernel object).
    spatial_W, spatial_w, alpha, X, y0_pre : arrays, optional
        Overrides of the toy design; drawn/derived from ``seed`` when
        omitted.
    m_iid, m_mcmc, burn : int
        Draw counts of the two simulators and the transition burn-in.
    a0, b0, step_rho, rho_support, beta_prior : sampler settings
        Passed to the kernel (``beta_prior`` is production-only).
    g_fn : callable, optional
        ``g_fn(summary, Yc, y0_pre, Wn, wn) -> dict`` replacing
        :func:`default_g_fn`.
    batch_size : int, optional
        Batch length for the MCMC standard error (default ``floor(sqrt(m))``).
    alpha_level : float, default 0.05
        Familywise test level.
    bonferroni : bool, default True
        Bonferroni-adjust the critical value across statistics.
    seed : int, optional
        Seed for all randomness (design draws included).
    verbose : bool, default False
        Print stage progress.

    Returns
    -------
    GewekeReport
        The per-statistic z table and the pass decision.

    Notes
    -----
    The successive-conditional simulator mixes slowly along two well-known
    directions, and an under-resolved run flags them as spurious failures:

    * the ``rho`` chain moves only as far per sweep as its data-conditional
      posterior allows, so *informative designs* (large ``T0 * N``) make it
      diffuse slowly -- keep the test panel small (e.g. ``T0=4, N=4``);
    * with ``K > 0`` and ``p > 0`` simultaneously, the ``X beta`` and
      ``Eta Gamma`` mean components trade off along a ridge whose relaxation
      dominates the global data statistics -- test the blocks in isolation
      first, and give joint configurations long chains with a large
      ``batch_size``;
    * the production kernel's half-Cauchy scale hierarchies are funnel-shaped
      and effectively untestable at feasible chain lengths (the reason the
      replication package only ever tested the simplified kernel, at two
      million draws).

    A genuine incoherence shows up as a *stable, sign-consistent* z across
    seeds and scales; mixing artifacts flip sign and shrink as the chain
    grows.
    """
    rng = np.random.default_rng(seed)

    if isinstance(kernel, str):
        from ..utils.scspill_helpers.setup import normalize_w, row_normalize

        W_raw = spatial_W if spatial_W is not None else _chain_graph(N)
        Wn = row_normalize(np.asarray(W_raw, dtype=float))
        if spatial_w is None:
            w_raw = np.zeros(N)
            w_raw[0] = 1.0
        else:
            w_raw = np.asarray(spatial_w, dtype=float).ravel()
        wn = normalize_w(w_raw)
        alpha_use = (
            np.asarray(alpha, dtype=float).ravel()
            if alpha is not None
            else 0.4 * rng.standard_normal(N)
        )
        X_use = X if X is not None else (rng.standard_normal((T0, N, K)) if K > 0 else None)
        kern: Any
        if kernel == "simple":
            kern = SimpleKernel(
                T0,
                N,
                K,
                p,
                Wn,
                wn,
                alpha_use,
                X=X_use,
                a0=a0,
                b0=b0,
                step_rho=step_rho,
                rho_support=rho_support,
            )
        elif kernel == "production":
            kern = ProductionKernel(
                T0,
                N,
                K,
                p,
                Wn,
                wn,
                alpha_use,
                beta_prior=beta_prior,
                X=X_use,
                a0=a0,
                b0=b0,
                step_rho=step_rho,
                rho_support=rho_support,
            )
        else:
            raise ScspillConfigError(f"geweke_test: unknown kernel {kernel!r}.")
    else:
        kern = kernel
        Wn, wn = kern.Wn, kern.wn
        T0 = kern.T0

    if y0_pre is None:
        y0_pre = rng.standard_normal(T0)
    if g_fn is None:
        g_fn = default_g_fn

    def _g(state, Yc) -> dict:
        vals = dict(g_fn(kern.state_summary(state), Yc, y0_pre, Wn, wn))
        vals.update(kern.extra_stats(state))
        return vals

    with threadpool_limits(limits=1, user_api="blas"):
        # ---------- marginal-conditional (iid) side ----------
        if verbose:  # pragma: no cover - cosmetic
            print(f"[JDT] marginal-conditional side ({m_iid} iid draws) ...")
        rows_iid = []
        for _ in range(m_iid):
            st = kern.draw_prior(rng)
            Yc = kern.simulate_data(st, rng)
            rows_iid.append(_g(st, Yc))
        g_iid = pd.DataFrame(rows_iid)

        # ---------- successive-conditional side ----------
        if verbose:  # pragma: no cover - cosmetic
            print(f"[JDT] successive-conditional side ({burn} burn + {m_mcmc}) ...")
        state = kern.draw_prior(rng)
        for _ in range(burn):
            Yc = kern.simulate_data(state, rng)
            state = kern.transition(state, Yc, rng)
        rows_mcmc = []
        for _ in range(m_mcmc):
            Yc = kern.simulate_data(state, rng)
            state = kern.transition(state, Yc, rng)
            rows_mcmc.append(_g(state, Yc))
        g_mcmc = pd.DataFrame(rows_mcmc)

    # Drop statistics that are entirely NaN (absent blocks).
    keep = [c for c in g_iid.columns if g_iid[c].notna().any() or g_mcmc[c].notna().any()]
    g_iid, g_mcmc = g_iid[keep], g_mcmc[keep]

    if batch_size is None:
        batch_size = max(2, int(np.floor(np.sqrt(m_mcmc))))

    mean_iid = g_iid.mean(skipna=True)
    mean_mcmc = g_mcmc.mean(skipna=True)
    n_iid = g_iid.notna().sum().clip(lower=1)
    se_iid = np.sqrt(g_iid.var(skipna=True, ddof=1) / n_iid)
    se_mcmc = pd.Series(
        {c: np.sqrt(batch_means_variance(g_mcmc[c].to_numpy(), batch_size)) for c in keep}
    )
    z = (mean_iid - mean_mcmc) / np.sqrt(se_iid**2 + se_mcmc**2)
    pval = 2.0 * norm.sf(np.abs(z))

    table = pd.DataFrame(
        {
            "g": keep,
            "mean_iid": np.asarray(mean_iid, dtype=float),
            "mean_mcmc": np.asarray(mean_mcmc, dtype=float),
            "se_iid": np.asarray(se_iid, dtype=float),
            "se_mcmc": np.asarray(se_mcmc, dtype=float),
            "z": np.asarray(z, dtype=float),
            "pval": pval,
        }
    )

    z_arr = np.asarray(table["z"], dtype=float)
    finite_z = z_arr[np.isfinite(z_arr)]
    n_stats = max(1, finite_z.size)
    z_crit = float(
        norm.ppf(1.0 - alpha_level / (2.0 * n_stats))
        if bonferroni
        else norm.ppf(1.0 - alpha_level / 2.0)
    )
    n_flagged = int(np.sum(np.abs(finite_z) > z_crit))

    return GewekeReport(
        table=table,
        kernel=getattr(kern, "name", "custom"),
        m_iid=int(m_iid),
        m_mcmc=int(m_mcmc),
        burn=int(burn),
        batch_size=int(batch_size),
        rho_support=(float(kern.rho_support[0]), float(kern.rho_support[1])),
        z_crit=z_crit,
        n_flagged=n_flagged,
        passed=(n_flagged == 0),
    )
