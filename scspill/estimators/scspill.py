"""The SCSPILL estimator: spillover-aware synthetic control models.

One estimator, several models, selected by ``SCSPILLConfig.method``:

``"sar"`` (default)
    Sakaguchi, S., & Tagawa, H. (2026). "Identification and Bayesian
    Inference for Synthetic Control Methods with Spillover Effects." The
    Econometrics Journal. https://doi.org/10.1093/ectj/utag006

    The treatment can spill over to the donor pool through a
    spatial-autoregressive structure with user-supplied weights ``(w, W)``
    and a single spillover intensity ``rho``; a two-step Bayesian sampler
    estimates the synthetic weights and ``rho``. At ``rho = 0`` it collapses
    exactly to the Bayesian horseshoe synthetic control.

``"cd"``
    Cao, J., & Dowd, C. "Estimation and Inference for Synthetic Control
    Methods with Spillover Effects." arXiv:1902.07343v3.

    Every unit gets an intercept-adjusted simplex synthetic control; a known
    exposure structure then separates the direct effect from the spillovers
    in every post-treatment period, with end-of-sample P-test inference.

Both models share panel preparation, the exception contract, the result
type, and plotting; see ``scspill.utils.scspill_helpers.sar`` and
``scspill.utils.scspill_helpers.cd`` for each model's algorithmic pieces.
"""

from __future__ import annotations

from pydantic import ValidationError

from ..exceptions import (
    ScspillConfigError,
    ScspillDataError,
    ScspillEstimationError,
    ScspillPlottingError,
)
from ..utils.datautils import balance
from ..utils.scspill_helpers.cd import build_cd_report, prepare_cd_inputs, run_cd
from ..utils.scspill_helpers.config import SCSPILLConfig
from ..utils.scspill_helpers.sar.pipeline import run_scspill
from ..utils.scspill_helpers.sar.report import build_sar_report
from ..utils.scspill_helpers.sar.setup import prepare_sar_inputs
from ..utils.scspill_helpers.setup import prepare_panel_inputs
from ..utils.scspill_helpers.structures import SCSPILLResults


class SCSPILL:
    """Spillover-aware synthetic control estimator (models ``sar`` and ``cd``).

    The default ``method="sar"`` fits the Bayesian spatial-autoregressive
    model of Sakaguchi and Tagawa (2026); ``method="cd"`` fits Cao and Dowd's
    all-unit SCM with a known exposure structure and end-of-sample P-test
    inference (arXiv:1902.07343v3). Both return a :class:`SCSPILLResults`.

    Parameters
    ----------
    config : SCSPILLConfig or dict
        The estimation configuration; a dict is coerced into
        :class:`scspill.config_models.SCSPILLConfig`.

    Examples
    --------
    ```python
    from scspill import SCSPILL
    from scspill.data import load_california, load_california_cd

    panel = load_california()
    result = SCSPILL(
        {**panel.config_kwargs(), "m_iter": 2000, "burn": 1000, "seed": 42}
    ).fit()
    result.att, result.att_ci
    result.rho_hat, result.rho_ci
    result.spillover_panel["Nevada"]

    cd = SCSPILL(load_california_cd(51).config_kwargs()).fit()
    cd.effects_panel.loc[1990, ["CA", "NV"]], cd.effect_ci["CA"]
    ```
    """

    def __init__(self, config: SCSPILLConfig | dict) -> None:
        if isinstance(config, dict):
            try:
                config = SCSPILLConfig(**config)
            except ValidationError as exc:
                raise ScspillConfigError(f"Invalid SCSPILL configuration: {exc}") from exc
        if not isinstance(config, SCSPILLConfig):
            raise ScspillConfigError("config must be a SCSPILLConfig or a dict of its fields.")
        self.config = config
        # Convenience attribute unpacking (mlsynth convention).
        self.df = config.df
        self.outcome = config.outcome
        self.treat = config.treat
        self.unitid = config.unitid
        self.time = config.time
        self.method = config.method
        self.display_graphs = config.display_graphs

    def fit(self) -> SCSPILLResults:
        """Estimate the selected model and return standardized results.

        Runs the balance check and panel preparation shared by every model,
        the model's own estimation (``sar``: the two-step sampler and the
        posterior effect sweep; ``cd``: the all-unit SCM equations, the
        effects, and the P-test inference), and (optionally) the default plot
        panel.

        Returns
        -------
        SCSPILLResults
            With the model's fit in the slot named by ``method``.

        Raises
        ------
        ScspillDataError
            On invalid panel data or misaligned spatial weights.
        ScspillEstimationError
            On an estimation failure (including an unidentified ``cd``
            exposure structure).
        ScspillPlottingError
            On plot generation failure (only when ``display_graphs``).
        """
        cfg = self.config
        sar = cfg.method == "sar"

        try:
            balance(cfg.df, cfg.unitid, cfg.time)
            panel = prepare_panel_inputs(
                cfg.df,
                cfg.outcome,
                cfg.treat,
                cfg.unitid,
                cfg.time,
                covariates=cfg.covariates if sar else None,
                min_pre=2 if sar else 3,
                min_controls=3 if sar else 2,
                allow_multiple_treated=not sar,
                require_persistent=not sar,
            )
            if sar:
                sar_inputs = prepare_sar_inputs(panel, cfg.spatial_W, cfg.spatial_w)
            else:
                cd_inputs = prepare_cd_inputs(
                    panel,
                    spillover_structure=cfg.spillover_structure,
                    affected_units=cfg.affected_units,
                    unit_distances=cfg.unit_distances,
                    effect_matrix=cfg.effect_matrix,
                )
        except (ScspillDataError, ScspillConfigError):
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise ScspillDataError(f"SCSPILL data preparation failed: {exc}") from exc

        try:
            if sar:
                sar_fit = run_scspill(
                    sar_inputs,
                    p_factors=cfg.p_factors,
                    m_iter=cfg.m_iter,
                    burn=cfg.burn,
                    step_rho=cfg.step_rho,
                    adapt_rho=cfg.adapt_rho,
                    target_accept_rho=cfg.target_accept_rho,
                    beta_prior=cfg.beta_prior,
                    propagate_alpha=cfg.propagate_alpha,
                    a0=cfg.a0,
                    b0=cfg.b0,
                    ci=cfg.ci,
                    max_effect_draws=cfg.max_effect_draws,
                    backend=cfg.backend,
                    seed=cfg.seed,
                    verbose=cfg.verbose,
                )
                report = build_sar_report(cfg, sar_fit)
            else:
                cd_fit = run_cd(
                    cd_inputs,
                    reference_mode=cfg.reference_mode,
                    quantile_method=cfg.quantile_method,
                    ci_method=cfg.ci_method,
                    ci=cfg.ci,
                    weighting=cfg.weighting,
                    scm_maxiter=cfg.scm_maxiter,
                    scm_ftol=cfg.scm_ftol,
                )
                report = build_cd_report(cfg, cd_fit)
            results = SCSPILLResults(**report, plot_config=cfg.resolved_plot())
        except (ScspillDataError, ScspillConfigError, ScspillEstimationError):
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise ScspillEstimationError(f"SCSPILL estimation failed: {exc}") from exc

        if cfg.display_graphs:
            try:
                from ..utils.scspill_helpers.plotter import plot_scspill

                plot_scspill(
                    results,
                    kind="panel",
                    top_n=cfg.top_n_spill,
                    save=cfg.save if isinstance(cfg.save, str) else None,
                )
            except ScspillPlottingError:
                raise
            except Exception as exc:  # pragma: no cover - defensive translation
                raise ScspillPlottingError(f"SCSPILL plotting failed: {exc}") from exc

        return results
