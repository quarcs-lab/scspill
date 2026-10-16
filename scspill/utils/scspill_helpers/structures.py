"""The public result of a SCSPILL fit.

:class:`SCSPILLResults` is the pydantic model that exposes the standardized
effect-result surface (``att``, ``att_ci``, ``counterfactual``, ``gap``,
``donor_weights``) alongside each model's typed fit. The dataclass containers
live next to the code that fills them -- the shared ones in
:mod:`.containers`, each model's in its subpackage -- and are re-exported here
under their historical import path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import ConfigDict

from ...config_models import BaseEstimatorResults
from .containers import PanelInputs, SCSPILLEffects, _freeze_array
from .sar.structures import (
    AlphaPosterior,
    SARFit,
    SARInputs,
    SARPosterior,
    SCSPILLFit,
    SCSPILLInputs,
)

__all__ = [
    "AlphaPosterior",
    "PanelInputs",
    "SARFit",
    "SARInputs",
    "SARPosterior",
    "SCSPILLEffects",
    "SCSPILLFit",
    "SCSPILLInputs",
    "SCSPILLResults",
    "_freeze_array",
]


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
    uses this result; another model declares its own estimation blocks
    alongside :attr:`alpha_posterior` / :attr:`sar_posterior`, which are
    SAR-specific, while :attr:`inputs`, :attr:`effects_detail` and the flat
    effect surface are the cross-model contract.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    method: str = "sar"
    inputs: SARInputs
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
