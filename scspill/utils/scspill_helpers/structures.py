"""The public result of a SCSPILL fit.

:class:`SCSPILLResults` is the pydantic model that exposes the standardized
effect-result surface (``att``, ``att_ci``, ``counterfactual``, ``gap``,
``donor_weights``) alongside each model's typed fit. The dataclass containers
live next to the code that fills them -- the shared ones in
:mod:`.containers`, each model's in its subpackage -- and are re-exported here
under their historical import path.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field, model_validator

from ...config_models import BaseEstimatorResults
from ...exceptions import ScspillEstimationError
from .cd.structures import CDFit, CDInputs, CDTestResult
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
    "CDFit",
    "CDInputs",
    "CDTestResult",
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
    """Standardized results of a SCSPILL fit, for any model.

    Subclasses :class:`scspill.config_models.BaseEstimatorResults`, so the
    flat effect surface (``att``, ``att_ci``, ``counterfactual``, ``gap``,
    ``donor_weights``, ``pre_rmse``, ``.plot()``) resolves through the shared
    contract for every model. The rest is typed:

    * :attr:`method` -- the model that produced the fit; it mirrors
      ``SCSPILLConfig.method`` and the suffix of
      ``method_details.method_name`` (``"SCSPILL/<method>"``);
    * :attr:`inputs` -- the prepared panel (with the spatial weights for
      ``sar``, the exposure matrix for ``cd``);
    * :attr:`effects_detail` -- the cross-model effect contract: the
      counterfactual path and band, and the time-by-control spillover panel
      with its band;
    * :attr:`sar` / :attr:`cd` -- the model's own fit. Exactly the slot named
      by :attr:`method` is set; the other is ``None``.

    Model-specific accessors read the matching slot -- ``alpha_hat``,
    ``rho_hat``, ``diagnostics()`` for ``sar``; ``effects_panel``,
    ``effect_ci``, ``test_hypothesis()`` for ``cd`` -- and raise
    :class:`AttributeError` on a result of the other model.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    method: Literal["sar", "cd"] = "sar"
    inputs: SARInputs | CDInputs
    effects_detail: SCSPILLEffects
    scm_weights: dict[str, float] | None = None
    sar: SARFit | None = Field(default=None, exclude=True, repr=False)
    cd: CDFit | None = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def _check_model_slot(self) -> SCSPILLResults:
        """Check that exactly the slot named by ``method`` holds the fit."""
        expected = {"sar": (SARFit, SARInputs), "cd": (CDFit, CDInputs)}
        for name, (fit_type, inputs_type) in expected.items():
            fit = getattr(self, name)
            if name == self.method:
                if not isinstance(fit, fit_type) or not isinstance(self.inputs, inputs_type):
                    raise ScspillEstimationError(
                        f"A method={self.method!r} result needs its {fit_type.__name__} "
                        f"and {inputs_type.__name__}."
                    )
            elif fit is not None:
                raise ScspillEstimationError(
                    f"A method={self.method!r} result must not carry a {name!r} fit."
                )
        return self

    # ------------------------------------------------------------------
    # Shared across models
    # ------------------------------------------------------------------
    @property
    def treated_labels(self) -> tuple[Any, ...]:
        """Every treated unit; the first is the headline of the effect surface."""
        return self.inputs.treated_labels

    @property
    def unit_labels(self) -> tuple[Any, ...]:
        """Every unit: the treated units first, then the controls."""
        return self.inputs.unit_labels

    @property
    def att_scm(self) -> float | None:
        """ATT of the no-spillover synthetic-control comparator."""
        return self.effects_detail.att_scm

    @property
    def att_by_unit(self) -> dict[Any, float]:
        """Average post-treatment effect on every treated unit."""
        if self.cd is not None:
            return dict(self.cd.att_by_unit)
        return {self.inputs.treated_label: float(self.effects_detail.att)}

    # ------------------------------------------------------------------
    # method="cd"
    # ------------------------------------------------------------------
    def _require_cd(self, name: str) -> CDFit:
        """Return the ``cd`` fit, or raise AttributeError naming the accessor."""
        if self.cd is None:
            raise AttributeError(
                f"{name!r} is defined only for method='cd'; this result has method={self.method!r}."
            )
        return self.cd

    @property
    def effects_panel(self) -> pd.DataFrame:
        """Estimated effect on every unit by post period (``cd`` only).

        Treated columns are direct effects, control columns spillovers.
        """
        return self._require_cd("effects_panel").effects_panel

    @property
    def effect_ci(self) -> dict[Any, pd.DataFrame]:
        """Pointwise interval of every unit's effect, by ``ci_method`` (``cd`` only)."""
        return self._require_cd("effect_ci").effect_ci

    @property
    def pointwise_ci(self) -> dict[Any, pd.DataFrame]:
        """Pointwise intervals by P-test inversion (``cd`` only)."""
        return self._require_cd("pointwise_ci").pointwise_ci

    @property
    def published_ci(self) -> dict[Any, pd.DataFrame]:
        """Pointwise intervals by the authors' signed-quantile rule (``cd`` only)."""
        return self._require_cd("published_ci").published_ci

    @property
    def pointwise_p_values(self) -> pd.DataFrame:
        """P value of every unit's effect by post period (``cd`` only)."""
        return self._require_cd("pointwise_p_values").pointwise_p_values

    @property
    def placebo_effects(self) -> pd.DataFrame:
        """Pre-treatment placebo effects, the P-test references (``cd`` only)."""
        return self._require_cd("placebo_effects").placebo_effects

    @property
    def pre_residuals(self) -> pd.DataFrame:
        """Pre-treatment SCM residuals of every unit (``cd`` only)."""
        return self._require_cd("pre_residuals").pre_residuals

    @property
    def kappa_test(self) -> dict[Any, CDTestResult]:
        """The ``kappa_A`` structure test by post period (``cd`` only)."""
        return self._require_cd("kappa_test").kappa_test

    @property
    def scm_matrix(self) -> pd.DataFrame:
        """All-unit SCM weight matrix ``B`` (``cd`` only)."""
        return self._require_cd("scm_matrix").scm_matrix

    @property
    def intercepts(self) -> pd.Series:
        """Intercepts of every unit's SCM equation (``cd`` only)."""
        return self._require_cd("intercepts").intercepts

    @property
    def effect_matrix(self) -> pd.DataFrame:
        """The exposure matrix ``A`` used by the fit (``cd`` only)."""
        return self._require_cd("effect_matrix").inputs.A

    def test_hypothesis(
        self,
        C: np.ndarray | pd.DataFrame,
        *,
        d: np.ndarray | float | None = None,
        period: Any | None = None,
        weight: np.ndarray | Literal["inverse_covariance"] | None = None,
        quantile_method: Literal["empirical", "linear"] | None = None,
    ) -> CDTestResult:
        """Test ``C @ alpha_t = d`` by the pre-treatment P distribution (``cd`` only).

        See :meth:`scspill.utils.scspill_helpers.cd.structures.CDFit.test_hypothesis`.
        """
        return self._require_cd("test_hypothesis").test_hypothesis(
            C, d=d, period=period, weight=weight, quantile_method=quantile_method
        )

    # ------------------------------------------------------------------
    # method="sar"
    # ------------------------------------------------------------------

    def _require_sar(self, name: str) -> SARFit:
        """Return the ``sar`` fit, or raise AttributeError naming the accessor."""
        if self.sar is None:
            raise AttributeError(
                f"{name!r} is defined only for method='sar'; this result has "
                f"method={self.method!r}."
            )
        return self.sar

    @property
    def alpha_posterior(self) -> AlphaPosterior:
        """Step-1 posterior block of the synthetic weights (``sar`` only)."""
        return self._require_sar("alpha_posterior").alpha_posterior

    @property
    def sar_posterior(self) -> SARPosterior:
        """Step-2 posterior block of the SAR process (``sar`` only)."""
        return self._require_sar("sar_posterior").sar_posterior

    @property
    def mcmc_summary_table(self) -> pd.DataFrame | None:
        """Precomputed posterior summary table (``sar`` only)."""
        return self._require_sar("mcmc_summary_table").mcmc_summary_table

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
            ``"spill_top"``, ``"weights"`` (both models), ``"rho"``,
            ``"trace"`` (``sar``), ``"spillover"`` and ``"kappa"`` (``cd``)
            route to :func:`scspill.utils.scspill_helpers.plotter.plot_scspill`.
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
