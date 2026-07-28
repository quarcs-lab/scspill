"""Configuration model for the SCSPILL estimator."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import ScspillConfigError


class SCSPILLConfig(BaseEstimatorConfig):
    """Configuration for the Bayesian spatial-spillover synthetic control (SCSPILL).

    Implements:

        Sakaguchi, S., & Tagawa, H. (2026). "Identification and Bayesian
        Inference for Synthetic Control Methods with Spillover Effects." The
        Econometrics Journal. https://doi.org/10.1093/ectj/utag006

    The panel is a long DataFrame (one row per unit-period). The treated unit
    and the treatment date are inferred from the 0/1 ``treat`` indicator
    column (1 for the treated unit in post-treatment periods), following the
    mlsynth convention. The spatial structure is supplied through
    ``spatial_w`` (treated-to-control exposure) and ``spatial_W``
    (control-to-control weights), aligned to donor units by label.

    Mapping from the R package's ``sc_spillover()`` arguments:

    ============== ======================= ====================================
    R argument      SCSPILLConfig field     Notes
    ============== ======================= ====================================
    ``data``        ``df``                  long panel
    ``y``           ``outcome``             column name
    ``unit_col``    ``unitid``              column name
    ``time_col``    ``time``                column name
    ``treatment_dummy`` ``treat``           column name; also identifies the
                                            treated unit and ``T0``
    ``treated_unit``/``T0``  --             inferred from ``treat``
    ``w``           ``spatial_w``           label-aligned Series/dict/array
    ``W``           ``spatial_W``           label-aligned DataFrame/array
    ``X``           ``covariates``          column names in ``df``
    ``p_factors``   ``p_factors``
    ``M``           ``m_iter``
    ``burn``        ``burn``
    ``step_rho``    ``step_rho``            initial value when ``adapt_rho``
    ``seed``        ``seed``
    ============== ======================= ====================================

    Differences from the R code (paper-correct defaults, each with an
    R-compatibility escape hatch):

    * ``beta_prior="horseshoe"`` implements the paper's Section 4.2 horseshoe
      prior on the covariate coefficients; ``"ridge"`` reproduces the R
      code's flat-plus-ridge conditional.
    * ``propagate_alpha=True`` pairs ``(alpha^(m), rho^(m))`` posterior draws
      in the effect formulas (the paper's stated procedure); ``False``
      reproduces the R package's intervals, which hold ``alpha`` fixed at its
      posterior mean and vary only ``rho``.
    * ``adapt_rho=True`` tunes the ``rho`` random-walk step during burn-in
      toward ``target_accept_rho`` (Robbins-Monro), then freezes it; the R
      sampler uses a fixed step.
    * Covariates are carried as a proper ``(T, N, K)`` array throughout,
      fixing the R package's covariate memory-layout mismatch.
    """

    spatial_w: Any = Field(
        ...,
        description=(
            "Treated-to-control spatial exposure weights: a pandas Series or dict "
            "keyed by donor unit label, a two-column DataFrame (label, weight), or "
            "an (N,) array in donor-label order. Normalized to sum to one internally."
        ),
    )
    spatial_W: Any = Field(
        ...,
        description=(
            "Control-to-control spatial weight matrix: a labelled (N, N) DataFrame "
            "(optionally with a leading label column), or an (N, N) array in "
            "donor-label order. Row-normalized internally (zero rows are kept zero)."
        ),
    )
    covariates: list[str] | None = Field(
        default=None,
        description="Names of time-varying covariate columns in `df` (the R `X`).",
    )
    p_factors: int = Field(
        default=1,
        ge=0,
        description="Number of AR(1) latent factors in the Step-2 error model (0 disables).",
    )
    m_iter: int = Field(
        default=2000,
        ge=10,
        description="Total MCMC iterations per step (the R `M`).",
    )
    burn: int = Field(
        default=1000,
        ge=0,
        description="Burn-in iterations per step; `m_iter - burn` draws are retained.",
    )
    step_rho: float = Field(
        default=0.05,
        gt=0,
        description="Random-walk Metropolis step for rho (initial value when adapt_rho).",
    )
    adapt_rho: bool = Field(
        default=True,
        description=(
            "Adapt the rho step size during burn-in toward `target_accept_rho` "
            "(Robbins-Monro), frozen afterwards. Set False for the R sampler's fixed step."
        ),
    )
    target_accept_rho: float = Field(
        default=0.44,
        gt=0,
        lt=1,
        description="Target acceptance rate for the adaptive rho step (scalar-optimal 0.44).",
    )
    beta_prior: Literal["horseshoe", "ridge"] = Field(
        default="horseshoe",
        description=(
            "Prior on the covariate coefficients: 'horseshoe' (paper Section 4.2) or "
            "'ridge' (the R code's flat-plus-ridge conditional, for cross-validation)."
        ),
    )
    propagate_alpha: bool = Field(
        default=True,
        description=(
            "Pair (alpha, rho) posterior draws in the effect formulas (paper procedure). "
            "False holds alpha at its posterior mean, reproducing the R intervals."
        ),
    )
    a0: float = Field(default=1.0, gt=0, description="Inverse-gamma prior shape for sigma^2.")
    b0: float = Field(default=1.0, gt=0, description="Inverse-gamma prior scale for sigma^2.")
    seed: int | None = Field(default=None, description="Seed for numpy's default_rng.")
    ci: float = Field(
        default=0.95,
        gt=0,
        lt=1,
        description="Credible level for all posterior intervals.",
    )
    top_n_spill: int = Field(
        default=8,
        ge=1,
        description="Default number of donors shown in the spillover plot panel.",
    )
    max_effect_draws: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Cap on posterior draws swept in the effects computation (evenly-spaced "
            "thinning). None uses every retained draw."
        ),
    )
    backend: Literal["auto", "numpy", "numba"] = Field(
        default="auto",
        description=(
            "Sampler kernel backend: 'numpy' (reference), 'numba' (JIT, requires the "
            "scspill[numba] extra), or 'auto' (numba when importable, else numpy)."
        ),
    )
    verbose: bool = Field(default=False, description="Print progress messages during sampling.")

    @model_validator(mode="after")
    def _check_mcmc_and_covariates(self) -> SCSPILLConfig:
        """Validate the MCMC budget and covariate column names."""
        if self.burn >= self.m_iter:
            raise ScspillConfigError(
                f"burn={self.burn} must be strictly less than m_iter={self.m_iter}."
            )
        if self.covariates:
            missing = [c for c in self.covariates if c not in self.df.columns]
            if missing:
                raise ScspillConfigError(f"Covariate column(s) not found in df: {missing}.")
        return self
