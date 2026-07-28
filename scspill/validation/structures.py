"""Typed result containers for the validation suite."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GewekeReport:
    """Result of the Geweke (2004) joint distribution test.

    Attributes
    ----------
    table : pd.DataFrame
        One row per test statistic ``g`` with columns ``mean_iid``,
        ``mean_mcmc``, ``se_iid``, ``se_mcmc``, ``z``, ``pval``.
    kernel : str
        ``"simple"`` or ``"production"``.
    m_iid, m_mcmc, burn : int
        Draw counts of the two simulators and the successive-conditional
        burn-in.
    batch_size : int
        Batch size of the batch-means MCMC standard error.
    rho_support : tuple of float
        The canonical ``rho`` support used by *both* simulators (the prior
        draws and the transition kernel; the R implementation used slightly
        different supports on the two sides, which this port deliberately
        unifies).
    z_crit : float
        Two-sided critical value used for the pass decision
        (Bonferroni-adjusted across the statistics by default).
    n_flagged : int
        Number of statistics with ``|z| > z_crit``.
    passed : bool
        ``n_flagged == 0``.
    """

    table: pd.DataFrame
    kernel: str
    m_iid: int
    m_mcmc: int
    burn: int
    batch_size: int
    rho_support: tuple[float, float]
    z_crit: float
    n_flagged: int
    passed: bool


@dataclass(frozen=True)
class PosteriorSummary:
    """Posterior summary of one Step-2 re-run (prior-sensitivity row).

    Attributes
    ----------
    table : pd.DataFrame
        One row per parameter (``rho``, ``sigma2``, ``beta[k]``) with columns
        ``mean``, ``sd``, ``q025``, ``q975``.
    rho_draws, sigma2_draws : np.ndarray
        The retained chains.
    meta : dict
        The run settings (kernel, burn/keep/thin, priors, support, step,
        seed, acceptance rate).
    """

    table: pd.DataFrame
    rho_draws: np.ndarray
    sigma2_draws: np.ndarray
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PriorSensitivityResult:
    """Result of a prior-sensitivity grid sweep.

    Attributes
    ----------
    table : pd.DataFrame
        One row per (grid row, parameter) pair: the grid settings joined with
        the posterior summary.
    runs : tuple of PosteriorSummary
        The per-row posterior summaries, in grid order.
    """

    table: pd.DataFrame
    runs: tuple[PosteriorSummary, ...]


@dataclass(frozen=True)
class PriorPredictiveResult:
    """Result of a prior predictive check.

    Attributes
    ----------
    stats : pd.DataFrame
        One row per prior draw, one column per summary statistic.
    observed : dict or None
        The statistics of the observed panel (when supplied).
    p_values : dict or None
        Per-statistic ``P(h_sim <= h_obs)`` over the finite simulated values.
    """

    stats: pd.DataFrame
    observed: dict | None
    p_values: dict | None
