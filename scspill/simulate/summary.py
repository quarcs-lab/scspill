"""Aggregation and grid drivers for the Monte Carlo study."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from ..exceptions import ScspillConfigError, ScspillDataError
from .dgp import paper_alpha
from .runner import SimRunResult, run_many_sim

_METRIC_COLS = (
    "bias_ate",
    "mse_ate",
    "bias_point",
    "mse_point",
    "cover95_ate",
    "cover95_point",
)


def summarize_many(results: Iterable[SimRunResult]) -> pd.DataFrame:
    """Aggregate replication metrics into per-method summary rows.

    Parameters
    ----------
    results : iterable of SimRunResult
        The output of :func:`~scspill.simulate.runner.run_many_sim`.

    Returns
    -------
    pd.DataFrame
        One row per method (ordered SCM < BSCM < SCSPILL) with the NaN-aware
        mean of each metric, its across-replication standard deviation
        (``*_sd`` columns), and the derived ``rmse_ate`` / ``rmse_point``
        (square roots of the mean squared errors).
    """
    frames = [r.metrics for r in results]
    if not frames:
        raise ScspillDataError("summarize_many: no results to summarize.")
    stacked = pd.concat(frames, ignore_index=True)
    method = pd.CategoricalDtype(["SCM", "BSCM", "SCSPILL"], ordered=True)
    stacked["method"] = stacked["method"].astype(method)
    grouped = stacked.groupby("method", observed=True)
    means = grouped[list(_METRIC_COLS)].mean()
    sds = grouped[list(_METRIC_COLS)].std(ddof=1).add_suffix("_sd")
    out = means.join(sds)
    out["rmse_ate"] = np.sqrt(out["mse_ate"])
    out["rmse_point"] = np.sqrt(out["mse_point"])
    out["n_sims"] = grouped.size()
    return out.reset_index()


def mc_grid(
    *,
    Ns: tuple[int, ...] = (16, 36, 64),
    T0s: tuple[int, ...] = (20, 50),
    T1: int = 10,
    rhos: tuple[float, ...] = (-0.8, -0.3, -0.1, 0.0, 0.1, 0.3, 0.8),
    sims_per: int = 1000,
    K: int = 1,
    beta: tuple[float, ...] = (1.0,),
    sigma2: float = 0.1,
    treated=(0, 1, 2, 3),
    m_iter: int = 6000,
    burn: int = 1000,
    step_rho: float = 0.05,
    n_jobs: int | None = 1,
    seed: int | None = None,
    backend: str = "auto",
    progress: bool = False,
) -> pd.DataFrame:
    """Run the paper's Monte Carlo grid (Tables 1-2 design).

    Defaults reproduce the replication package's full study: rook lattices of
    ``N in {16, 36, 64}`` units, pre-periods ``T0 in {20, 50}``, spillover
    intensities ``rho in {-0.8, ..., 0.8}``, 1000 replications per cell, one
    ``N(0, 1)`` covariate with coefficient 1, error variance 0.1, and the
    paper's planted ``alpha``. Reduce ``sims_per`` / the grids for quicker
    runs.

    Parameters
    ----------
    Ns : tuple of int
        Numbers of control units; each must be a perfect square.
    T0s : tuple of int
        Pre-treatment lengths.
    T1 : int, default 10
        Post-treatment length.
    rhos : tuple of float
        True spillover intensities.
    sims_per : int, default 1000
        Replications per scenario.
    K, beta, sigma2, treated : DGP settings
        See :func:`~scspill.simulate.dgp.scspill_sim_dgp`.
    m_iter, burn, step_rho : MCMC budget
        Sampler settings per replication (the paper uses 6000/1000/0.05).
    n_jobs : int, optional
        Parallel workers per scenario (see
        :func:`~scspill.simulate.runner.run_many_sim`).
    seed : int, optional
        Master seed; per-scenario seed lists are spawned deterministically.
    backend : {"auto", "numpy", "numba"}, default "auto"
        Sampler kernel backend.
    progress : bool, default False
        Print one line per completed scenario.

    Returns
    -------
    pd.DataFrame
        Tidy results in the frozen R schema: ``N``, ``T0``, ``T1``, ``rho``,
        ``method``, ``bias_point``, ``rmse_point``, ``cover95_point``.
    """
    for N in Ns:
        side = round(float(np.sqrt(N)))
        if side * side != N:
            raise ScspillConfigError(f"mc_grid: N={N} is not a perfect square (rook lattice).")

    master = np.random.SeedSequence(seed)
    rows = []
    for N in Ns:
        side = round(float(np.sqrt(N)))
        for T0 in T0s:
            for rho in rhos:
                scen_ss = master.spawn(1)[0]
                seeds = list(scen_ss.spawn(sims_per))
                dgp_args = {
                    "grid": (side, side),
                    "treated": treated,
                    "T0": T0,
                    "T1": T1,
                    "rho": rho,
                    "sigma2": sigma2,
                    "alpha": paper_alpha(N),
                    "K": K,
                    "beta": np.asarray(beta, dtype=float) if K > 0 else None,
                }
                results = run_many_sim(
                    sims_per,
                    dgp_args,
                    seeds=seeds,
                    n_jobs=n_jobs,
                    m_iter=m_iter,
                    burn=burn,
                    step_rho=step_rho,
                    backend=backend,
                )
                summary = summarize_many(results)
                for _, r in summary.iterrows():
                    rows.append(
                        {
                            "N": N,
                            "T0": T0,
                            "T1": T1,
                            "rho": rho,
                            "method": str(r["method"]),
                            "bias_point": float(r["bias_point"]),
                            "rmse_point": float(r["rmse_point"]),
                            "cover95_point": (
                                float(r["cover95_point"])
                                if np.isfinite(r["cover95_point"])
                                else np.nan
                            ),
                        }
                    )
                if progress:  # pragma: no cover - cosmetic
                    print(f"mc_grid: N={N} T0={T0} rho={rho:+.1f} done ({sims_per} reps)")
    return pd.DataFrame(rows)


def load_r_mc_reference(directory: str | Path) -> pd.DataFrame:
    """Load the frozen R Monte Carlo results for comparison.

    Parameters
    ----------
    directory : str or Path
        Folder holding the replication package's
        ``mc_study_N=<N>T0=<T0>T1=<T1>.csv`` files (the frozen copies live
        under ``benchmarks/reference/mc_result/`` in this repository).

    Returns
    -------
    pd.DataFrame
        The concatenated tidy frame in the same schema as :func:`mc_grid`,
        with the R method label ``Proposed``/``SCSPILL`` normalized to
        ``SCSPILL``.
    """
    directory = Path(directory)
    files = sorted(directory.glob("mc_study_*.csv"))
    if not files:
        raise ScspillDataError(f"load_r_mc_reference: no mc_study_*.csv under {directory}.")
    frames = [pd.read_csv(f) for f in files]
    out = pd.concat(frames, ignore_index=True)
    out["method"] = out["method"].replace({"Proposed": "SCSPILL"})
    return out
