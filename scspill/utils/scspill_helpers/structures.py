"""Typed containers for the SCSPILL estimator.

Frozen dataclasses for the estimation inputs and posterior blocks, plus the
:class:`SCSPILLResults` pydantic model that exposes the standardized
effect-result surface (``att``, ``att_ci``, ``counterfactual``, ``gap``,
``donor_weights``) alongside the estimator-specific posterior objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ConfigDict

from ...config_models import BaseEstimatorResults


def _freeze_array(arr: np.ndarray | None) -> None:
    """Mark an array read-only so frozen containers stay effectively immutable."""
    if isinstance(arr, np.ndarray):
        arr.setflags(write=False)


@dataclass(frozen=True)
class SCSPILLInputs:
    """Prepared estimation inputs for the SCSPILL estimator.

    Attributes
    ----------
    Y0 : np.ndarray
        Treated-unit outcome path, shape ``(T,)``.
    Yc : np.ndarray
        Control-unit outcome matrix, shape ``(T, N)`` in ``control_labels``
        column order.
    Wn : np.ndarray
        Row-normalized control-to-control spatial weight matrix, ``(N, N)``.
    wn : np.ndarray
        Sum-normalized treated-to-control exposure vector, ``(N,)``.
    W_raw, w_raw : np.ndarray
        The user-supplied spatial weights before normalization (label-aligned).
    T0 : int
        Number of pre-treatment periods.
    X : np.ndarray or None
        Covariate cube ``(T, N, K)`` for the control units, or ``None``.
    treated_label : Any
        Label of the treated unit.
    control_labels : tuple
        Donor labels in column order of ``Yc`` / ``Wn`` / ``wn``.
    time_labels : np.ndarray
        Sorted time labels, length ``T``.
    covariate_names : tuple of str
        Names of the covariate columns stacked into ``X``.
    """

    Y0: np.ndarray
    Yc: np.ndarray
    Wn: np.ndarray
    wn: np.ndarray
    W_raw: np.ndarray
    w_raw: np.ndarray
    T0: int
    X: np.ndarray | None
    treated_label: Any
    control_labels: tuple[Any, ...]
    time_labels: np.ndarray
    covariate_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate shape coherence and freeze the arrays."""
        N = len(self.control_labels)
        T = self.Y0.shape[0]
        assert self.Yc.shape == (T, N), "Yc shape does not match (T, N)"
        assert self.Wn.shape == (N, N), "Wn shape does not match (N, N)"
        assert self.wn.shape == (N,), "wn shape does not match (N,)"
        assert self.time_labels.shape[0] == T, "time_labels length does not match T"
        if self.X is not None:
            assert self.X.shape[:2] == (T, N), "X shape does not match (T, N, K)"
        for arr in (
            self.Y0,
            self.Yc,
            self.Wn,
            self.wn,
            self.W_raw,
            self.w_raw,
            self.X,
            self.time_labels,
        ):
            _freeze_array(arr)

    @property
    def N(self) -> int:
        """Number of control units."""
        return len(self.control_labels)

    @property
    def T(self) -> int:
        """Total number of time periods."""
        return int(self.Y0.shape[0])

    @property
    def T1(self) -> int:
        """Number of post-treatment periods."""
        return self.T - self.T0

    @property
    def K(self) -> int:
        """Number of covariates."""
        return 0 if self.X is None else int(self.X.shape[2])

    @property
    def time_pre(self) -> np.ndarray:
        """Pre-treatment time labels."""
        return self.time_labels[: self.T0]

    @property
    def time_post(self) -> np.ndarray:
        """Post-treatment time labels."""
        return self.time_labels[self.T0 :]


@dataclass(frozen=True)
class AlphaPosterior:
    """Step-1 posterior: horseshoe draws of the synthetic weights ``alpha``.

    Attributes
    ----------
    draws : np.ndarray
        Post-burn ``alpha`` draws in original outcome units, shape ``(M, N)``.
    sigma2 : np.ndarray
        Post-burn draws of the Step-1 error variance (on the standardized
        scale), shape ``(M,)``.
    alpha_hat : np.ndarray
        Posterior mean of ``alpha``, shape ``(N,)`` -- the plug-in weights
        used by Step 2.
    iters, burn : int
        Total iterations and burn-in of the sampler.
    """

    draws: np.ndarray
    sigma2: np.ndarray
    alpha_hat: np.ndarray
    iters: int
    burn: int

    def __post_init__(self) -> None:
        """Freeze the draw arrays."""
        for arr in (self.draws, self.sigma2, self.alpha_hat):
            _freeze_array(arr)


@dataclass(frozen=True)
class SARPosterior:
    """Step-2 posterior: the SAR block conditional on ``alpha_hat``.

    Attributes
    ----------
    rho : np.ndarray
        Post-burn draws of the spillover intensity, shape ``(M,)``.
    sigma2 : np.ndarray
        Post-burn draws of the innovation variance, shape ``(M,)``.
    beta : np.ndarray or None
        Post-burn draws of the covariate coefficients, shape ``(M, K)``, or
        ``None`` when there are no covariates.
    rho_hat : float
        Posterior mean of ``rho``.
    acc_rho : float
        Post-burn acceptance rate of the random-walk Metropolis step.
    step_rho_final : float
        Random-walk step size after burn-in adaptation (equals the initial
        ``step_rho`` when adaptation is off).
    rho_bound : float
        Half-width of the admissible ``rho`` support,
        ``0.95 / max(1, max |eig(Wn)|)``.
    beta_prior : str
        Prior used for ``beta``: ``"horseshoe"`` (paper) or ``"ridge"``
        (R-code compatibility).
    p_factors : int
        Number of AR(1) latent factors in the error model.
    iters, burn : int
        Total iterations and burn-in of the sampler.
    """

    rho: np.ndarray
    sigma2: np.ndarray
    beta: np.ndarray | None
    rho_hat: float
    acc_rho: float
    step_rho_final: float
    rho_bound: float
    beta_prior: str
    p_factors: int
    iters: int
    burn: int

    def __post_init__(self) -> None:
        """Freeze the draw arrays."""
        for arr in (self.rho, self.sigma2, self.beta):
            _freeze_array(arr)


@dataclass(frozen=True)
class SCSPILLEffects:
    """Posterior treatment and spillover effects (Theorems 3.1-3.2).

    All paths cover the full sample (pre and post periods); pre-period rows of
    the spillover panel are fit residuals by construction, not causal
    spillovers.

    Attributes
    ----------
    att : float
        Posterior mean of the average treatment effect on the treated
        (mean over the ``att_draws``).
    att_ci : tuple of float
        Equal-tailed credible interval for the ATT at level ``ci_level``.
    att_draws : np.ndarray
        One ATT draw per retained posterior draw, shape ``(M,)``.
    att_plugin : float
        ATT evaluated at the posterior means ``(alpha_hat, rho_hat)``.
    att_scm : float
        ATT of the ``rho = 0`` special case (plain Bayesian horseshoe SCM at
        ``alpha_hat``) -- the no-spillover comparator.
    cf_mean, cf_lower, cf_upper : np.ndarray
        Posterior mean and pointwise credible band of the treated unit's
        untreated counterfactual path, each shape ``(T,)``.
    spill_mean, spill_lower, spill_upper : pd.DataFrame
        Posterior mean and pointwise credible band of the spillover effects,
        each ``(T, N)`` indexed by time labels with donor-label columns.
    ci_level : float
        Credible level of all intervals.
    propagate_alpha : bool
        True when effect draws paired ``(alpha^(m), rho^(m))``; False when
        ``alpha`` was held at ``alpha_hat`` (the R package's convention).
    n_draws_used : int
        Number of posterior draws swept in the effects computation.
    """

    att: float
    att_ci: tuple[float, float]
    att_draws: np.ndarray
    att_plugin: float
    att_scm: float
    cf_mean: np.ndarray
    cf_lower: np.ndarray
    cf_upper: np.ndarray
    spill_mean: pd.DataFrame
    spill_lower: pd.DataFrame
    spill_upper: pd.DataFrame
    ci_level: float
    propagate_alpha: bool
    n_draws_used: int

    def __post_init__(self) -> None:
        """Freeze the draw arrays."""
        for arr in (self.att_draws, self.cf_mean, self.cf_lower, self.cf_upper):
            _freeze_array(arr)


@dataclass(frozen=True)
class SCSPILLFit:
    """Bundle of everything the pipeline produced (pre-results assembly)."""

    inputs: SCSPILLInputs
    alpha_posterior: AlphaPosterior
    sar_posterior: SARPosterior
    effects: SCSPILLEffects
    scm_weights: dict[str, float] = field(default_factory=dict)
    mcmc_summary_table: pd.DataFrame | None = None


class SCSPILLResults(BaseEstimatorResults):
    """Standardized results of the SCSPILL estimator.

    Subclasses :class:`scspill.config_models.BaseEstimatorResults`, so the
    flat effect surface (``att``, ``att_ci``, ``counterfactual``, ``gap``,
    ``donor_weights``, ``pre_rmse``, ``.plot()``) resolves through the shared
    contract, while the Bayesian detail lives in typed fields:

    * :attr:`inputs` -- the prepared panel and spatial weights;
    * :attr:`alpha_posterior` / :attr:`sar_posterior` -- the two posterior
      blocks with their full draw arrays;
    * :attr:`effects_detail` -- ATT draws, counterfactual band, and the
      per-donor spillover panel;
    * :attr:`scm_weights` -- classical simplex-SCM comparator weights.

    :attr:`method` records which spillover model produced the fit. It mirrors
    ``SCSPILLConfig.method`` and the suffix of
    ``method_details.method_name`` (``"SCSPILL/<method>"``). Only ``"sar"``
    exists today; a model added later declares its own posterior blocks
    alongside :attr:`alpha_posterior` / :attr:`sar_posterior`, which are
    SAR-specific, while :attr:`inputs`, :attr:`effects_detail` and the flat
    effect surface are the cross-model contract.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    method: str = "sar"
    inputs: SCSPILLInputs
    alpha_posterior: AlphaPosterior
    sar_posterior: SARPosterior
    effects_detail: SCSPILLEffects
    scm_weights: dict[str, float] | None = None
    mcmc_summary_table: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Convenience accessors over the posterior blocks
    # ------------------------------------------------------------------
    @property
    def alpha_draws(self) -> np.ndarray:
        """Step-1 ``alpha`` draws, shape ``(M, N)``."""
        return self.alpha_posterior.draws

    @property
    def alpha_hat(self) -> np.ndarray:
        """Posterior mean of the synthetic weights ``alpha``."""
        return self.alpha_posterior.alpha_hat

    @property
    def rho_draws(self) -> np.ndarray:
        """Step-2 ``rho`` draws, shape ``(M,)``."""
        return self.sar_posterior.rho

    @property
    def rho_hat(self) -> float:
        """Posterior mean of the spillover intensity ``rho``."""
        return self.sar_posterior.rho_hat

    @property
    def rho_ci(self) -> tuple[float, float]:
        """Equal-tailed credible interval for ``rho`` at the fit's ``ci`` level."""
        lo_q = (1.0 - self.effects_detail.ci_level) / 2.0
        rho = self.sar_posterior.rho
        return (float(np.quantile(rho, lo_q)), float(np.quantile(rho, 1.0 - lo_q)))

    @property
    def rho_ess(self) -> float:
        """Effective sample size of the ``rho`` chain."""
        from ..diagnostics import ess_acf

        return ess_acf(self.sar_posterior.rho)

    @property
    def acc_rho(self) -> float:
        """Post-burn acceptance rate of the ``rho`` Metropolis step."""
        return self.sar_posterior.acc_rho

    @property
    def spillover_panel(self) -> pd.DataFrame:
        """Posterior-mean spillover effects, ``(T, N)`` time-by-donor."""
        return self.effects_detail.spill_mean

    @property
    def spillover_lower(self) -> pd.DataFrame:
        """Pointwise lower credible band of the spillover effects."""
        return self.effects_detail.spill_lower

    @property
    def spillover_upper(self) -> pd.DataFrame:
        """Pointwise upper credible band of the spillover effects."""
        return self.effects_detail.spill_upper

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def diagnostics(
        self,
        which_alpha: list | None = None,
        top_n_alpha: int = 6,
        which_beta: list | None = None,
        top_n_beta: int = 6,
    ) -> pd.DataFrame:
        """Posterior summary and convergence diagnostics table.

        Builds one row per monitored chain (``rho``, ``sigma2``, selected
        ``alpha`` components, and covariate coefficients when present) with
        posterior mean, sd, quantiles, effective sample size, split-chain
        R-hat, Monte Carlo standard error, and a Geweke z-score -- mirroring
        the R package's ``diagnostics()`` summary.

        Parameters
        ----------
        which_alpha : list, optional
            Donor labels whose ``alpha`` chains to include. Defaults to the
            ``top_n_alpha`` donors by absolute posterior-mean weight.
        top_n_alpha : int, default 6
            Number of top-|alpha| donors monitored when ``which_alpha`` is None.
        which_beta : list, optional
            Covariate names whose ``beta`` chains to include. Defaults to the
            first ``top_n_beta`` covariates.
        top_n_beta : int, default 6
            Number of covariates monitored when ``which_beta`` is None.

        Returns
        -------
        pd.DataFrame
            One row per parameter with columns ``mean``, ``sd``, ``q025``,
            ``q50``, ``q975``, ``ess``, ``rhat_split``, ``mcse``,
            ``geweke_z``.
        """
        from ..diagnostics import mcmc_summary

        chains: dict[str, np.ndarray] = {
            "rho": self.sar_posterior.rho,
            "sigma2": self.sar_posterior.sigma2,
        }

        labels = list(self.inputs.control_labels)
        if which_alpha is None:
            order = np.argsort(-np.abs(self.alpha_hat))[:top_n_alpha]
            which_alpha = [labels[j] for j in order]
        for lab in which_alpha:
            if lab not in labels:
                raise KeyError(f"Unknown donor label {lab!r} in which_alpha.")
            chains[f"alpha[{lab}]"] = self.alpha_posterior.draws[:, labels.index(lab)]

        if self.sar_posterior.beta is not None:
            cov_names = list(self.inputs.covariate_names) or [
                f"x{k}" for k in range(self.sar_posterior.beta.shape[1])
            ]
            if which_beta is None:
                which_beta = cov_names[:top_n_beta]
            for name in which_beta:
                if name not in cov_names:
                    raise KeyError(f"Unknown covariate name {name!r} in which_beta.")
                chains[f"beta[{name}]"] = self.sar_posterior.beta[:, cov_names.index(name)]

        return mcmc_summary(chains)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot(self, kind: str = "auto", *, ax: Any = None, **overrides: Any) -> Any:
        """Plot the fitted result.

        Parameters
        ----------
        kind : str, default "auto"
            ``"auto"`` / ``"counterfactual"`` / ``"gap"`` use the shared
            effect-plot contract; ``"panel"``, ``"full"``, ``"effect"``,
            ``"spill_top"``, ``"weights"``, ``"rho"``, and ``"trace"`` route
            to :func:`scspill.utils.scspill_helpers.plotter.plot_scspill`.
        ax : matplotlib Axes, optional
            Draw into an existing axis (single-panel kinds only).
        **overrides
            Cosmetic overrides forwarded to the plotting layer.

        Returns
        -------
        matplotlib.axes.Axes or np.ndarray of Axes
        """
        if kind in ("auto", "counterfactual", "gap"):
            return super().plot(kind=kind, ax=ax, **overrides)
        from .plotter import plot_scspill

        return plot_scspill(self, kind=kind, ax=ax, **overrides)


SCSPILLResults.model_rebuild()
