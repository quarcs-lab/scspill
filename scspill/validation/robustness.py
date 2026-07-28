"""Prior sensitivity and prior predictive checks for the Step-2 model.

Pure-function ports of the replication package's ``run_mcmc_for_posterior``,
``prior_sensitivity``, and ``prior_predictive`` -- reading only their
explicit arguments (the R versions accidentally captured global ``W``/``w``
bindings, which this port fixes) -- plus the nine prior-predictive summary
statistics of :func:`ppc_stats`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..exceptions import ScspillConfigError, ScspillDataError
from ..utils.scspill_helpers._kernels import _ig
from ..utils.scspill_helpers.sampler_sar import sar_step2_sampler
from ..utils.scspill_helpers.setup import normalize_w, row_normalize
from .kernels import SimpleKernel, SimpleState
from .structures import PosteriorSummary, PriorPredictiveResult, PriorSensitivityResult

_STAT_NAMES = (
    "yc_mean",
    "log_yc_var",
    "spatial_quadratic",
    "corr_y0_wyc",
    "ac1",
    "ac2",
    "pve_pc1",
    "avg_skewness",
    "avg_kurtosis",
)


def _skewness_type3(x: np.ndarray) -> float:
    """e1071 type-3 skewness: ``g1 * ((n - 1) / n)^{3/2}``."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return float("nan")
    xc = x - x.mean()
    m2 = float(np.mean(xc**2))
    m3 = float(np.mean(xc**3))
    if m2 <= 0:
        return float("nan")
    g1 = m3 / m2**1.5
    return g1 * ((n - 1.0) / n) ** 1.5


def _kurtosis_type3(x: np.ndarray) -> float:
    """e1071 type-3 excess kurtosis: ``(g2 + 3) (1 - 1/n)^2 - 3``."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return float("nan")
    xc = x - x.mean()
    m2 = float(np.mean(xc**2))
    m4 = float(np.mean(xc**4))
    if m2 <= 0:
        return float("nan")
    g2 = m4 / m2**2 - 3.0
    return (g2 + 3.0) * (1.0 - 1.0 / n) ** 2 - 3.0


def ppc_stats(Yc: np.ndarray, Y0_pre: np.ndarray, Wn: np.ndarray, wn: np.ndarray) -> dict:
    """Nine summary statistics of a pre-treatment donor panel.

    Exact port of the R ``ppc_stats``: panel mean and log variance, the
    spatial quadratic form ``tr(Yc W Yc') / (N T0)``, the correlation of the
    treated series with the exposure-weighted panel, average lag-1 and lag-2
    autocorrelations of the per-period cross-sectionally demeaned panel, the
    share of variance in the first principal component, and the average
    e1071 type-3 skewness and excess kurtosis across donors.

    Parameters
    ----------
    Yc : np.ndarray
        Donor panel, shape ``(T0, N)``.
    Y0_pre : np.ndarray
        Treated pre-treatment series, shape ``(T0,)``.
    Wn, wn : np.ndarray
        Normalized spatial weights.

    Returns
    -------
    dict
        Statistic name -> value (NaN where degenerate).
    """
    Yc = np.asarray(Yc, dtype=float)
    Y0_pre = np.asarray(Y0_pre, dtype=float).ravel()
    T0, N = Yc.shape
    yc_vec = Yc.ravel()

    wyc = Yc @ wn
    spatial_q = float(np.trace(Yc @ Wn @ Yc.T)) / (N * T0)
    sd0 = float(np.std(Y0_pre, ddof=1))
    sdw = float(np.std(wyc, ddof=1))
    corr = float(np.corrcoef(Y0_pre, wyc)[0, 1]) if sd0 > 0 and sdw > 0 else float("nan")

    # Per-period cross-sectional demeaning (the R code's scale() on t(Yc)),
    # then average per-unit lag correlations.
    Yd = Yc.T  # (N, T0)
    Ydc = Yd - Yd.mean(axis=0, keepdims=True)
    den = np.sum(Ydc * Ydc, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ac1 = float(np.nanmean(np.sum(Ydc[:, 1:] * Ydc[:, :-1], axis=1) / den))
        ac2 = (
            float(np.nanmean(np.sum(Ydc[:, 2:] * Ydc[:, :-2], axis=1) / den))
            if T0 > 2
            else float("nan")
        )

    pve_pc1 = float("nan")
    if N > 1 and T0 > 1:
        Ycc = Yc - Yc.mean(axis=0, keepdims=True)
        sv = np.linalg.svd(Ycc, compute_uv=False)
        eigs = sv**2
        total = float(eigs.sum())
        if total > 0:
            pve_pc1 = float(eigs[0] / total)

    avg_skew = float(np.nanmean([_skewness_type3(Yc[:, j]) for j in range(N)]))
    avg_kurt = float(np.nanmean([_kurtosis_type3(Yc[:, j]) for j in range(N)]))

    return {
        "yc_mean": float(yc_vec.mean()),
        "log_yc_var": float(np.log(yc_vec.var(ddof=1) + 1e-12)),
        "spatial_quadratic": spatial_q,
        "corr_y0_wyc": corr,
        "ac1": ac1,
        "ac2": ac2,
        "pve_pc1": pve_pc1,
        "avg_skewness": avg_skew,
        "avg_kurtosis": avg_kurt,
    }


def _prep_weights(spatial_W, spatial_w) -> tuple[np.ndarray, np.ndarray]:
    """Normalize raw spatial weights (explicit arguments, no globals)."""
    Wn = row_normalize(np.asarray(spatial_W, dtype=float))
    wn = normalize_w(np.asarray(spatial_w, dtype=float).ravel())
    return Wn, wn


def run_posterior_mcmc(
    Yc_obs: np.ndarray,
    spatial_W: np.ndarray,
    spatial_w: np.ndarray,
    alpha: np.ndarray,
    *,
    X: np.ndarray | None = None,
    p: int = 0,
    a0: float = 1.0,
    b0: float = 1.0,
    rho_support: tuple[float, float] | None = None,
    step_rho: float = 0.05,
    m_burn: int = 5_000,
    m_keep: int = 20_000,
    thin: int = 1,
    seed: int = 123,
    kernel: str = "simple",
) -> PosteriorSummary:
    """Sample the Step-2 posterior on a fixed panel under explicit priors.

    ``kernel="simple"`` iterates the appendix's simplified sweep
    (:class:`~scspill.validation.kernels.SimpleKernel`) on the observed
    panel, mirroring the R ``run_mcmc_for_posterior``; ``kernel="production"``
    re-runs the production Step-2 sampler with the overridden priors and
    support.

    Parameters
    ----------
    Yc_obs : np.ndarray
        Observed donor pre-period panel, shape ``(T0, N)``.
    spatial_W, spatial_w : np.ndarray
        Raw spatial weights (normalized internally).
    alpha : np.ndarray
        Fixed synthetic weights, shape ``(N,)``.
    X : np.ndarray, optional
        Covariate cube ``(T0, N, K)``.
    p : int, default 0
        Number of latent factors.
    a0, b0 : float, default 1.0
        Inverse-gamma prior for ``sigma^2``.
    rho_support : (float, float), optional
        Support for ``rho`` (defaults to the spectral bound of ``A``).
    step_rho : float, default 0.05
        Fixed Metropolis step.
    m_burn, m_keep, thin : int
        Burn-in, retained draws, and thinning interval.
    seed : int, default 123
        Random seed.
    kernel : {"simple", "production"}, default "simple"

    Returns
    -------
    PosteriorSummary
    """
    Yc_obs = np.asarray(Yc_obs, dtype=float)
    T0, N = Yc_obs.shape
    K = 0 if X is None else int(X.shape[2])
    Wn, wn = _prep_weights(spatial_W, spatial_w)
    alpha = np.asarray(alpha, dtype=float).ravel()
    rng = np.random.default_rng(seed)

    if kernel == "production":
        iters = m_burn + m_keep * thin
        post = sar_step2_sampler(
            rng,
            Yc_obs,
            alpha,
            wn,
            Wn,
            iters,
            m_burn,
            X=X,
            p=p,
            step_rho=step_rho,
            adapt_rho=False,
            a0=a0,
            b0=b0,
            rho_support=rho_support,
        )
        rho_draws = post.rho[::thin]
        s2_draws = post.sigma2[::thin]
        beta_draws = post.beta[::thin] if post.beta is not None else None
        accept = post.acc_rho
    elif kernel == "simple":
        kern = SimpleKernel(
            T0,
            N,
            K,
            p,
            Wn,
            wn,
            alpha,
            X=X,
            a0=a0,
            b0=b0,
            step_rho=step_rho,
            rho_support=rho_support,
        )
        lo, hi = kern.rho_support
        # Initial state per the R reference: rho ~ U(support), sigma2 from its
        # prior, beta ~ N(0, I), factors at zero.
        state = SimpleState(
            rho=float(rng.uniform(lo, hi)),
            sigma2=float(_ig(rng, a0, b0)),
            beta=rng.standard_normal(K),
            Eta=np.zeros((N, p)),
            Gamma=np.zeros((p, T0)),
        )
        rho_list, s2_list, beta_list = [], [], []
        moved = 0
        with threadpool_limits(limits=1, user_api="blas"):
            for _ in range(m_burn):
                kern.transition(state, Yc_obs, rng)
            last_rho = state.rho
            for m in range(m_keep * thin):
                kern.transition(state, Yc_obs, rng)
                if state.rho != last_rho:
                    moved += 1
                last_rho = state.rho
                if (m + 1) % thin == 0:
                    rho_list.append(state.rho)
                    s2_list.append(state.sigma2)
                    if K > 0:
                        beta_list.append(state.beta.copy())
        rho_draws = np.asarray(rho_list)
        s2_draws = np.asarray(s2_list)
        beta_draws = np.asarray(beta_list) if K > 0 else None
        accept = moved / max(1, m_keep * thin)
    else:
        raise ScspillConfigError(f"run_posterior_mcmc: unknown kernel {kernel!r}.")

    rows = {
        "rho": rho_draws,
        "sigma2": s2_draws,
    }
    if beta_draws is not None:
        for k in range(beta_draws.shape[1]):
            rows[f"beta[{k + 1}]"] = beta_draws[:, k]
    table = pd.DataFrame(
        {
            "parameter": list(rows),
            "mean": [float(np.mean(v)) for v in rows.values()],
            "sd": [float(np.std(v, ddof=1)) if len(v) > 1 else float("nan") for v in rows.values()],
            "q025": [float(np.quantile(v, 0.025)) for v in rows.values()],
            "q975": [float(np.quantile(v, 0.975)) for v in rows.values()],
        }
    )
    return PosteriorSummary(
        table=table,
        rho_draws=rho_draws,
        sigma2_draws=s2_draws,
        meta={
            "kernel": kernel,
            "m_burn": m_burn,
            "m_keep": m_keep,
            "thin": thin,
            "a0": a0,
            "b0": b0,
            "rho_support": rho_support,
            "step_rho": step_rho,
            "seed": seed,
            "accept_rate": float(accept),
        },
    )


def prior_sensitivity(
    Yc_obs: np.ndarray,
    spatial_W: np.ndarray,
    spatial_w: np.ndarray,
    alpha: np.ndarray,
    grid: pd.DataFrame,
    *,
    X: np.ndarray | None = None,
    p: int = 0,
    m_burn: int = 5_000,
    m_keep: int = 20_000,
    thin: int = 1,
    base_seed: int = 1000,
    kernel: str = "simple",
) -> PriorSensitivityResult:
    """Re-run the Step-2 posterior across a grid of prior settings.

    Parameters
    ----------
    Yc_obs, spatial_W, spatial_w, alpha, X, p : as in :func:`run_posterior_mcmc`
    grid : pd.DataFrame
        One row per setting with columns ``a0``, ``b0``, ``rho_lo``,
        ``rho_hi``, ``step_rho`` (the replication package's grid layout).
    m_burn, m_keep, thin : int
        MCMC budget shared by every row.
    base_seed : int, default 1000
        Row ``i`` uses seed ``base_seed + i``.
    kernel : {"simple", "production"}, default "simple"

    Returns
    -------
    PriorSensitivityResult
        The grid-joined posterior summaries and the per-row runs.
    """
    required = {"a0", "b0", "rho_lo", "rho_hi", "step_rho"}
    missing = required - set(grid.columns)
    if missing:
        raise ScspillConfigError(f"prior_sensitivity: grid is missing columns {sorted(missing)}.")
    runs = []
    frames = []
    for i, row in grid.reset_index(drop=True).iterrows():
        summary = run_posterior_mcmc(
            Yc_obs,
            spatial_W,
            spatial_w,
            alpha,
            X=X,
            p=p,
            a0=float(row["a0"]),
            b0=float(row["b0"]),
            rho_support=(float(row["rho_lo"]), float(row["rho_hi"])),
            step_rho=float(row["step_rho"]),
            m_burn=m_burn,
            m_keep=m_keep,
            thin=thin,
            seed=base_seed + int(i),
            kernel=kernel,
        )
        runs.append(summary)
        joined = summary.table.copy()
        for col in ("a0", "b0", "rho_lo", "rho_hi", "step_rho"):
            joined.insert(0, col, row[col])
        joined.insert(0, "grid_row", i)
        frames.append(joined)
    return PriorSensitivityResult(table=pd.concat(frames, ignore_index=True), runs=tuple(runs))


def prior_predictive(
    Y0_pre: np.ndarray,
    spatial_W: np.ndarray,
    spatial_w: np.ndarray,
    alpha: np.ndarray,
    *,
    Yc_obs: np.ndarray | None = None,
    X: np.ndarray | None = None,
    p: int = 0,
    a0: float = 3.0,
    b0: float = 1.0,
    rho_support: tuple[float, float] | None = None,
    n_draws: int = 2000,
    seed: int = 123,
) -> PriorPredictiveResult:
    """Prior predictive check of the Step-2 model.

    Draws parameters from the appendix's simple priors, forward-simulates a
    donor panel for each draw, computes the nine :func:`ppc_stats`, and --
    when an observed panel is supplied -- reports per-statistic prior
    predictive p-values ``P(h_sim <= h_obs)`` over the finite simulated
    values.

    Parameters
    ----------
    Y0_pre : np.ndarray
        Treated pre-treatment series, shape ``(T0,)``.
    spatial_W, spatial_w : np.ndarray
        Raw spatial weights (normalized internally).
    alpha : np.ndarray
        Fixed synthetic weights, shape ``(N,)``.
    Yc_obs : np.ndarray, optional
        Observed donor panel ``(T0, N)`` for the observed column/p-values.
    X : np.ndarray, optional
        Covariate cube ``(T0, N, K)``.
    p : int, default 0
        Number of latent factors.
    a0, b0 : float, default 3.0 / 1.0
        Inverse-gamma prior for ``sigma^2`` (the replication package's
        prior-predictive defaults).
    rho_support : (float, float), optional
        Support of the flat ``rho`` prior (defaults to the spectral bound).
    n_draws : int, default 2000
        Number of prior draws.
    seed : int, default 123
        Random seed.

    Returns
    -------
    PriorPredictiveResult
    """
    Y0_pre = np.asarray(Y0_pre, dtype=float).ravel()
    T0 = Y0_pre.shape[0]
    Wn, wn = _prep_weights(spatial_W, spatial_w)
    N = Wn.shape[0]
    K = 0 if X is None else int(X.shape[2])
    alpha = np.asarray(alpha, dtype=float).ravel()
    if Yc_obs is not None:
        Yc_obs = np.asarray(Yc_obs, dtype=float)
        if Yc_obs.shape != (T0, N):
            raise ScspillDataError(
                f"prior_predictive: Yc_obs has shape {Yc_obs.shape}, expected ({T0}, {N})."
            )

    kern = SimpleKernel(T0, N, K, p, Wn, wn, alpha, X=X, a0=a0, b0=b0, rho_support=rho_support)
    rng = np.random.default_rng(seed)
    rows = []
    with threadpool_limits(limits=1, user_api="blas"):
        for _ in range(n_draws):
            st = kern.draw_prior(rng)
            Yc_sim = kern.simulate_data(st, rng)
            rows.append(ppc_stats(Yc_sim, Y0_pre, Wn, wn))
    stats = pd.DataFrame(rows, columns=list(_STAT_NAMES))

    observed = None
    p_values = None
    if Yc_obs is not None:
        observed = ppc_stats(Yc_obs, Y0_pre, Wn, wn)
        p_values = {}
        for name in _STAT_NAMES:
            sims = stats[name].to_numpy()
            sims = sims[np.isfinite(sims)]
            obs = observed[name]
            p_values[name] = (
                float(np.mean(sims <= obs)) if sims.size and np.isfinite(obs) else float("nan")
            )
    return PriorPredictiveResult(stats=stats, observed=observed, p_values=p_values)
