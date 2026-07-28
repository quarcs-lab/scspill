"""Single-replication and Monte Carlo runners for the simulation study.

Each replication fits three estimators on one simulated panel -- the
classical simplex SCM, the Bayesian horseshoe SCM (Step 1 only), and SCSPILL
(both steps plus the identification formulas) -- and scores them against the
realized treatment-effect path: bias and MSE of the per-period path and of
the ATE, plus 95% credible-interval coverage for the Bayesian methods
(coverage is undefined for the point-estimate SCM).

The SCSPILL effect draws pair ``(alpha^(m), rho^(m))`` by common index over
``min(M_alpha, M_rho)`` draws; the R reference samples random index pairs,
which is statistically equivalent under the two-step cut posterior.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..exceptions import ScspillConfigError
from ..utils.scm_baseline import classical_scm_weights
from ..utils.scspill_helpers.sar._kernels import resolve_backend
from ..utils.scspill_helpers.sar.effects import RhoSolver
from ..utils.scspill_helpers.sar.sampler_alpha import hs_alpha_gibbs
from ..utils.scspill_helpers.sar.sampler_sar import sar_step2_sampler
from .dgp import SimDGP, SimTruth, make_w, paper_alpha, rook_W, scspill_sim_dgp

_METHODS = ("SCM", "BSCM", "SCSPILL")


@dataclass(frozen=True)
class SimRunResult:
    """Result of one simulation replication.

    Attributes
    ----------
    metrics : pd.DataFrame
        One row per method (SCM / BSCM / SCSPILL) with columns ``bias_ate``,
        ``mse_ate``, ``bias_point``, ``mse_point``, ``cover95_ate``,
        ``cover95_point``, ``method``. Coverage is NaN for SCM.
    truth : SimTruth or None
        The replication's ground truth (kept when ``keep_full``).
    draws : dict or None
        Posterior draw arrays (kept when ``keep_full``).
    per_time : dict or None
        Per-period effect paths and credible bands (kept when ``keep_full``).
    """

    metrics: pd.DataFrame
    truth: SimTruth | None = None
    draws: dict | None = None
    per_time: dict | None = None


def _resolve_dgp(dgp: SimDGP | None, dgp_args: dict | None, seed) -> SimDGP:
    """Build the DGP from explicit args when one is not supplied."""
    if dgp is not None:
        return dgp
    if dgp_args is None:
        raise ScspillConfigError("run_one_sim: provide either `dgp` or `dgp_args`.")
    args = dict(dgp_args)
    if "W" not in args:
        grid = args.pop("grid", None)
        if grid is None:
            raise ScspillConfigError("run_one_sim: dgp_args needs `W` or `grid=(nrow, ncol)`.")
        args["W"] = rook_W(int(grid[0]), int(grid[1]))
        args["N"] = args["W"].shape[0]
    args.setdefault("N", np.asarray(args["W"]).shape[0])
    if "w" not in args:
        # Default exposure: the paper's simulation design (first four donors
        # exposed). Note the R *function* default is a single exposed donor;
        # the R study driver overrides it to the paper's 1:4.
        args["w"] = make_w(args["N"], treated=args.pop("treated", (0, 1, 2, 3)))
    args.pop("treated", None)
    if "alpha" not in args:
        args["alpha"] = paper_alpha(args["N"])
    args.setdefault("K", 0)
    if args["K"] <= 0:
        args["beta"] = None
    args["seed"] = seed
    return scspill_sim_dgp(**args)


def run_one_sim(
    dgp: SimDGP | None = None,
    dgp_args: dict | None = None,
    *,
    m_iter: int = 2000,
    burn: int = 1000,
    step_rho: float = 0.02,
    a0: float = 1.0,
    b0: float = 1.0,
    seed=None,
    keep_full: bool = False,
    backend: str = "auto",
) -> SimRunResult:
    """Run one simulation replication: DGP, three estimators, metrics.

    Parameters
    ----------
    dgp : SimDGP, optional
        A pre-simulated panel; when omitted, one is generated from
        ``dgp_args`` (which accepts either ``W`` or ``grid=(nrow, ncol)``,
        plus the :func:`~scspill.simulate.dgp.scspill_sim_dgp` arguments;
        ``alpha`` defaults to the paper's planted weights).
    dgp_args : dict, optional
        Arguments for the DGP when ``dgp`` is not given.
    m_iter, burn : int
        MCMC budget for both sampler steps.
    step_rho : float, default 0.02
        Random-walk Metropolis step for ``rho`` (the R simulation's fixed
        step; adaptation is off to mirror the reference study).
    a0, b0 : float, default 1.0
        Inverse-gamma prior for ``sigma^2``.
    seed : int or numpy.random.SeedSequence, optional
        Seeds the replication. The DGP (when generated here) and the
        samplers draw from two *independent* child streams of this seed, so
        the simulated data and the MCMC noise are never correlated.
    keep_full : bool, default False
        Keep the draw arrays and per-period paths on the result.
    backend : {"auto", "numpy", "numba"}, default "auto"
        Sampler kernel backend.

    Returns
    -------
    SimRunResult
    """
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    dgp_ss, sampler_ss = ss.spawn(2)
    dgp = _resolve_dgp(dgp, dgp_args, dgp_ss)
    kernels = resolve_backend(backend)
    rng = np.random.default_rng(sampler_ss)

    Y0_pre, Yc_pre = dgp.Y0_pre, dgp.Yc_pre
    Y0_post, Yc_post = dgp.Y0_post, dgp.Yc_post
    te_true = dgp.truth.tau_post
    ate_true = float(te_true.mean())

    with threadpool_limits(limits=1, user_api="blas"):
        # --- classical simplex SCM ---
        alpha_scm = classical_scm_weights(Y0_pre, Yc_pre, ridge=1e-8)
        te_scm = Y0_post - Yc_post @ alpha_scm

        # --- Bayesian horseshoe SCM (Step 1 only) ---
        alpha_post = hs_alpha_gibbs(rng, Y0_pre, Yc_pre, m_iter, burn, kernels=kernels)
        te_bscm_mat = Y0_post[:, None] - Yc_post @ alpha_post.draws.T  # (T1, M)
        te_bscm_mean = te_bscm_mat.mean(axis=1)
        ate_bscm_draws = te_bscm_mat.mean(axis=0)

        # --- SCSPILL (Step 2 + identification formulas, p = 0 as in the paper) ---
        sar_post = sar_step2_sampler(
            rng,
            Yc_pre,
            alpha_post.alpha_hat,
            dgp.wn,
            dgp.Wn,
            m_iter,
            burn,
            X=dgp.X_pre,
            p=0,
            step_rho=step_rho,
            adapt_rho=False,
            a0=a0,
            b0=b0,
            kernels=kernels,
        )
        S = min(alpha_post.draws.shape[0], sar_post.rho.shape[0])
        solver = RhoSolver(dgp.Wn, dgp.wn)
        cache: dict = {}
        te_spill_mat = np.empty((dgp.T1, S))
        for s in range(S):
            CF = solver.counterfactual_controls(
                Y0_post, Yc_post, alpha_post.draws[s], float(sar_post.rho[s]), _cache=cache
            )
            te_spill_mat[:, s] = Y0_post - CF @ alpha_post.draws[s]
        te_spill_mean = te_spill_mat.mean(axis=1)
        ate_spill_draws = te_spill_mat.mean(axis=0)

    def _ci_cols(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.quantile(mat, 0.025, axis=1), np.quantile(mat, 0.975, axis=1)

    def _cover_point(mat: np.ndarray) -> float:
        lo, hi = _ci_cols(mat)
        return float(np.mean((lo <= te_true) & (te_true <= hi)))

    def _cover_ate(draws: np.ndarray) -> float:
        lo, hi = np.quantile(draws, [0.025, 0.975])
        return float(lo <= ate_true <= hi)

    def _row(te_hat: np.ndarray, ate_hat: float, cov_ate: float, cov_pt: float) -> dict:
        err_ate = ate_true - ate_hat
        return {
            "bias_ate": err_ate,
            "mse_ate": err_ate**2,
            "bias_point": float(np.mean(te_true - te_hat)),
            "mse_point": float(np.mean((te_true - te_hat) ** 2)),
            "cover95_ate": cov_ate,
            "cover95_point": cov_pt,
        }

    metrics = pd.DataFrame(
        [
            _row(te_scm, float(te_scm.mean()), np.nan, np.nan),
            _row(
                te_bscm_mean,
                float(te_bscm_mean.mean()),
                _cover_ate(ate_bscm_draws),
                _cover_point(te_bscm_mat),
            ),
            _row(
                te_spill_mean,
                float(te_spill_mean.mean()),
                _cover_ate(ate_spill_draws),
                _cover_point(te_spill_mat),
            ),
        ]
    )
    metrics["method"] = list(_METHODS)

    if not keep_full:
        return SimRunResult(metrics=metrics)

    bscm_lo, bscm_hi = _ci_cols(te_bscm_mat)
    spill_lo, spill_hi = _ci_cols(te_spill_mat)
    return SimRunResult(
        metrics=metrics,
        truth=dgp.truth,
        draws={
            "alpha_bscm": alpha_post.draws,
            "rho": sar_post.rho,
            "ate": {"bscm": ate_bscm_draws, "scspill": ate_spill_draws},
        },
        per_time={
            "true": te_true,
            "scm": te_scm,
            "bscm": te_bscm_mean,
            "scspill": te_spill_mean,
            "ci": {
                "bscm": {"lower": bscm_lo, "upper": bscm_hi},
                "scspill": {"lower": spill_lo, "upper": spill_hi},
            },
        },
    )


def _worker(payload: tuple[dict, int | None, dict]) -> pd.DataFrame:
    """Process-pool worker: one replication, metrics only (picklable)."""
    dgp_args, seed, run_kwargs = payload
    return run_one_sim(dgp_args=dgp_args, seed=seed, keep_full=False, **run_kwargs).metrics


def run_many_sim(
    n_sims: int,
    dgp_args: dict,
    *,
    seeds: list | None = None,
    n_jobs: int | None = 1,
    keep_full: bool = False,
    **run_kwargs: Any,
) -> list[SimRunResult]:
    """Run many replications, optionally in parallel.

    Parameters
    ----------
    n_sims : int
        Number of replications.
    dgp_args : dict
        Arguments for the DGP (see :func:`run_one_sim`).
    seeds : list of int or of numpy.random.SeedSequence, optional
        One seed per replication; independent ``SeedSequence`` children are
        spawned when omitted.
    n_jobs : int, optional
        Worker processes; ``1`` runs serially, ``None`` reads the
        ``SCSPILL_NWORKERS`` environment variable (R-parity convenience).
        Parallel runs return metrics-only results.
    keep_full : bool, default False
        Keep draw arrays on each result (serial runs only).
    **run_kwargs
        Forwarded to :func:`run_one_sim` (``m_iter``, ``burn``, ...).

    Returns
    -------
    list of SimRunResult
        In replication order.
    """
    if seeds is not None and len(seeds) != n_sims:
        raise ScspillConfigError(f"run_many_sim: got {len(seeds)} seeds for {n_sims} replications.")
    if seeds is None:
        # Keep the SeedSequence children themselves: collapsing them to a
        # single uint32 would discard the collision-free 128-bit spawn keys.
        seeds = list(np.random.SeedSequence().spawn(n_sims))
    if n_jobs is None:
        n_jobs = int(os.environ.get("SCSPILL_NWORKERS", "1"))

    if n_jobs <= 1:
        return [
            run_one_sim(dgp_args=dgp_args, seed=s, keep_full=keep_full, **run_kwargs) for s in seeds
        ]

    payloads = [(dgp_args, s, run_kwargs) for s in seeds]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        frames = list(pool.map(_worker, payloads))
    return [SimRunResult(metrics=m) for m in frames]
