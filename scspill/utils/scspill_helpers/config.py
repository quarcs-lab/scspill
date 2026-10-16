"""Configuration model for the SCSPILL estimator."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import ScspillConfigError

#: Fields read only by ``method="sar"``.
SAR_FIELDS = (
    "spatial_w",
    "spatial_W",
    "covariates",
    "p_factors",
    "m_iter",
    "burn",
    "step_rho",
    "adapt_rho",
    "target_accept_rho",
    "beta_prior",
    "propagate_alpha",
    "a0",
    "b0",
    "seed",
    "max_effect_draws",
    "backend",
)

#: Fields read only by ``method="cd"``.
CD_FIELDS = (
    "affected_units",
    "spillover_structure",
    "effect_matrix",
    "reference_mode",
    "quantile_method",
    "ci_method",
    "scm_maxiter",
    "scm_ftol",
)


class SCSPILLConfig(BaseEstimatorConfig):
    """Configuration for a spillover synthetic-control fit, any model.

    ``method`` selects the model; every other field belongs either to all
    models or to exactly one, marked ``(method='sar')`` or ``(method='cd')``
    in its description. Setting a field of a model that is not selected is an
    error rather than being silently ignored.

    ``method="sar"`` (the default) implements:

        Sakaguchi, S., & Tagawa, H. (2026). "Identification and Bayesian
        Inference for Synthetic Control Methods with Spillover Effects." The
        Econometrics Journal. https://doi.org/10.1093/ectj/utag006

    ``method="cd"`` implements:

        Cao, J., & Dowd, C. "Estimation and Inference for Synthetic Control
        Methods with Spillover Effects." arXiv:1902.07343v3.

    The panel is a long DataFrame (one row per unit-period). The treated unit
    and the treatment date are inferred from the 0/1 ``treat`` indicator
    column (1 for the treated unit in post-treatment periods), following the
    mlsynth convention. For ``sar`` the spatial structure is supplied through
    ``spatial_w`` (treated-to-control exposure) and ``spatial_W``
    (control-to-control weights), aligned to donor units by label. For ``cd``
    it is a fixed exposure structure: the units that may receive a spillover
    (``affected_units``) or a labeled exposure matrix (``effect_matrix``).

    Mapping from the R package's ``sc_spillover()`` arguments (``sar``):

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

    ``sar`` differences from the R code (paper-correct defaults, each with an
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

    method: Literal["sar", "cd"] = Field(
        default="sar",
        description=(
            "Spillover model to fit. 'sar' (default) = Sakaguchi & Tagawa (2026), "
            "the spatial-autoregressive Bayesian SCM: donor outcomes follow a SAR "
            "process under the supplied weights (spatial_w, spatial_W), and a "
            "two-step sampler estimates the synthetic weights alpha and the "
            "spillover intensity rho. 'cd' = Cao & Dowd (arXiv:1902.07343v3): an "
            "intercept-adjusted simplex SCM for every unit and a fixed exposure "
            "structure (affected_units or effect_matrix) identify period-specific "
            "direct and spillover effects, with Andrews-type P-test inference. See "
            "https://quarcs-lab.github.io/scspill/models/ for the catalogue."
        ),
    )
    spatial_w: Any = Field(
        default=None,
        description=(
            "(method='sar', required) Treated-to-control spatial exposure weights: a pandas Series or dict "
            "keyed by donor unit label, a two-column DataFrame (label, weight), or "
            "an (N,) array in donor-label order. Normalized to sum to one internally."
        ),
    )
    spatial_W: Any = Field(
        default=None,
        description=(
            "(method='sar', required) Control-to-control spatial weight matrix: a labelled (N, N) DataFrame "
            "(optionally with a leading label column), or an (N, N) array in "
            "donor-label order. Row-normalized internally (zero rows are kept zero)."
        ),
    )
    covariates: list[str] | None = Field(
        default=None,
        description="(method='sar') Names of time-varying covariate columns in `df` (the R `X`).",
    )
    p_factors: int = Field(
        default=1,
        ge=0,
        description="(method='sar') Number of AR(1) latent factors in the Step-2 error model (0 disables).",
    )
    m_iter: int = Field(
        default=2000,
        ge=10,
        description="(method='sar') Total MCMC iterations per step (the R `M`).",
    )
    burn: int = Field(
        default=1000,
        ge=0,
        description="(method='sar') Burn-in iterations per step; `m_iter - burn` draws are retained.",
    )
    step_rho: float = Field(
        default=0.05,
        gt=0,
        description="(method='sar') Random-walk Metropolis step for rho (initial value when adapt_rho).",
    )
    adapt_rho: bool = Field(
        default=True,
        description=(
            "(method='sar') Adapt the rho step size during burn-in toward `target_accept_rho` "
            "(Robbins-Monro), frozen afterwards. Set False for the R sampler's fixed step."
        ),
    )
    target_accept_rho: float = Field(
        default=0.44,
        gt=0,
        lt=1,
        description="(method='sar') Target acceptance rate for the adaptive rho step (scalar-optimal 0.44).",
    )
    beta_prior: Literal["horseshoe", "ridge"] = Field(
        default="horseshoe",
        description=(
            "(method='sar') Prior on the covariate coefficients: 'horseshoe' (paper Section 4.2) or "
            "'ridge' (the R code's flat-plus-ridge conditional, for cross-validation)."
        ),
    )
    propagate_alpha: bool = Field(
        default=True,
        description=(
            "(method='sar') Pair (alpha, rho) posterior draws in the effect formulas (paper procedure). "
            "False holds alpha at its posterior mean, reproducing the R intervals."
        ),
    )
    a0: float = Field(
        default=1.0, gt=0, description="(method='sar') Inverse-gamma prior shape for sigma^2."
    )
    b0: float = Field(
        default=1.0, gt=0, description="(method='sar') Inverse-gamma prior scale for sigma^2."
    )
    seed: int | None = Field(
        default=None, description="(method='sar') Seed for numpy's default_rng."
    )
    ci: float = Field(
        default=0.95,
        gt=0,
        lt=1,
        description=(
            "Level of every interval: posterior credible intervals (sar), P-test "
            "confidence intervals and critical values (cd)."
        ),
    )
    top_n_spill: int = Field(
        default=8,
        ge=1,
        description="Default number of control units shown in the spillover plot panel.",
    )
    max_effect_draws: int | None = Field(
        default=None,
        ge=1,
        description=(
            "(method='sar') Cap on posterior draws swept in the effects computation (evenly-spaced "
            "thinning). None uses every retained draw."
        ),
    )
    backend: Literal["auto", "numpy", "numba"] = Field(
        default="auto",
        description=(
            "(method='sar') Sampler kernel backend: 'numpy' (reference), 'numba' (JIT, requires the "
            "scspill[numba] extra), or 'auto' (numba when importable, else numpy)."
        ),
    )
    verbose: bool = Field(default=False, description="Print progress messages during fitting.")

    # ------------------------------------------------------------------
    # method="cd"
    # ------------------------------------------------------------------
    affected_units: list[Any] | None = Field(
        default=None,
        description=(
            "(method='cd') Labels of the control units that may receive a "
            "spillover. Each gets its own effect under "
            "spillover_structure='per_unit'. The treated unit must not appear "
            "here; omitting the field means no unit is exposed."
        ),
    )
    spillover_structure: Literal["per_unit", "custom"] = Field(
        default="per_unit",
        description=(
            "(method='cd') How the exposure matrix A is built: 'per_unit' gives "
            "the treated unit and every affected unit its own effect; 'custom' "
            "takes A from effect_matrix."
        ),
    )
    effect_matrix: pd.DataFrame | None = Field(
        default=None,
        description=(
            "(method='cd', required for spillover_structure='custom') A labeled "
            "exposure matrix A: one row per unit label, one column per effect "
            "parameter. Each treated unit's row must be its own direct-effect "
            "indicator: a single 1, in a column that is zero for every other unit."
        ),
    )
    reference_mode: Literal["full", "leave_one_period_out"] = Field(
        default="full",
        description=(
            "(method='cd') Pre-treatment reference statistics of the P tests: "
            "'full' uses the full-sample SCM fit (Cao & Dowd v3, Section 4); "
            "'leave_one_period_out' refits every SCM equation without each "
            "pre-treatment period in turn."
        ),
    )
    quantile_method: Literal["linear", "empirical"] = Field(
        default="linear",
        description=(
            "(method='cd') Critical-value rule of every P and kappa_A test: "
            "'empirical' is the left inverse of the empirical CDF (Cao & Dowd "
            "v3, Section 4); 'linear' is the linearly interpolated sample "
            "quantile of the authors' application code."
        ),
    )
    ci_method: Literal["published", "inversion"] = Field(
        default="published",
        description=(
            "(method='cd') Pointwise interval reported as effect_ci: "
            "'inversion' inverts the squared P test, giving an interval "
            "symmetric about the estimate (Cao & Dowd v3, Section 4); "
            "'published' adds the signed pre-treatment error quantiles to the "
            "estimate, as the authors' application code does."
        ),
    )
    scm_maxiter: int = Field(
        default=3000,
        ge=1,
        description="(method='cd') Iteration limit of the SLSQP simplex SCM solve.",
    )
    scm_ftol: float = Field(
        default=1e-12,
        gt=0,
        description="(method='cd') Convergence tolerance of the SLSQP simplex SCM solve.",
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def _check_inactive_model_fields(self) -> SCSPILLConfig:
        """Reject fields that belong to a model other than the selected one."""
        inactive = CD_FIELDS if self.method == "sar" else SAR_FIELDS
        owner = "cd" if self.method == "sar" else "sar"
        stray = [name for name in inactive if not _is_default(self, name)]
        if stray:
            raise ScspillConfigError(
                f"Field(s) {stray} apply only to method={owner!r}; remove them "
                f"or set method={owner!r}."
            )
        return self

    @model_validator(mode="after")
    def _check_sar(self) -> SCSPILLConfig:
        """``sar``: spatial weights, the MCMC budget, and covariate columns."""
        if self.method != "sar":
            return self
        missing = [n for n in ("spatial_w", "spatial_W") if getattr(self, n) is None]
        if missing:
            raise ScspillConfigError(f"method='sar' requires {missing}.")
        if self.burn >= self.m_iter:
            raise ScspillConfigError(
                f"burn={self.burn} must be strictly less than m_iter={self.m_iter}."
            )
        if self.covariates:
            absent = [c for c in self.covariates if c not in self.df.columns]
            if absent:
                raise ScspillConfigError(f"Covariate column(s) not found in df: {absent}.")
        return self

    @model_validator(mode="after")
    def _check_cd(self) -> SCSPILLConfig:
        """``cd``: the exposure structure and its labels."""
        if self.method != "cd":
            return self
        from .cd import validate_effect_matrix

        custom = self.spillover_structure == "custom"
        if custom and self.effect_matrix is None:
            raise ScspillConfigError("spillover_structure='custom' requires effect_matrix.")
        if not custom and self.effect_matrix is not None:
            raise ScspillConfigError(
                "effect_matrix is used only with spillover_structure='custom'."
            )
        units = set(self.df[self.unitid].unique())
        treated = set(self.df.loc[self.df[self.treat] == 1, self.unitid].unique())
        if self.effect_matrix is not None:
            if self.affected_units is not None:
                raise ScspillConfigError(
                    "affected_units is not used with spillover_structure='custom'; "
                    "the exposure is the effect_matrix."
                )
            validate_effect_matrix(self.effect_matrix, sorted(units), sorted(treated))
        elif self.affected_units is not None:
            labels = list(self.affected_units)
            if len(set(labels)) != len(labels):
                raise ScspillConfigError("affected_units must not repeat a label.")
            unknown = [u for u in labels if u not in units]
            if unknown:
                raise ScspillConfigError(f"affected_units not found in df: {unknown}.")
            if treated & set(labels):
                raise ScspillConfigError(
                    f"affected_units must not include a treated unit: "
                    f"{sorted(treated & set(labels))}."
                )
        return self


def _is_default(config: SCSPILLConfig, name: str) -> bool:
    """Whether field ``name`` holds its declared default (``[]`` counts as ``None``)."""
    value = getattr(config, name)
    default = type(config).model_fields[name].get_default(call_default_factory=True)
    if default is None:
        return value is None or (isinstance(value, list) and not value)
    if isinstance(value, pd.DataFrame | pd.Series):
        return False
    return bool(value == default)
