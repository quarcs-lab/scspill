"""Bayesian spatial-spillover synthetic control (SCSPILL).

Implements:

    Sakaguchi, S., & Tagawa, H. "Identification and Bayesian Inference for
    Synthetic Control Methods with Spillover Effects." The Econometrics
    Journal.

The method relaxes SUTVA: the treatment can spill over to the donor pool
through a spatial-autoregressive structure with user-supplied weights
``(w, W)`` and a single spillover intensity ``rho``. Both the treatment
effect on the treated unit and the spillover effect received by every
control unit are identified from ``(alpha, rho, w, W)`` alone, and are
estimated with a two-step Bayesian sampler: a horseshoe Gibbs sampler for
the unconstrained synthetic weights ``alpha`` on the pre-treatment fit,
then a SAR block (latent AR(1) factors, covariates, random-walk Metropolis
for ``rho``) conditional on the posterior mean of ``alpha``. At ``rho = 0``
the method collapses exactly to the Bayesian horseshoe synthetic control.

See ``scspill.utils.scspill_helpers`` for the algorithmic pieces.
"""

from __future__ import annotations

from pydantic import ValidationError

from ..config_models import MethodDetailsResults
from ..exceptions import (
    ScspillConfigError,
    ScspillDataError,
    ScspillEstimationError,
    ScspillPlottingError,
)
from ..utils.datautils import balance
from ..utils.results_helpers import build_effect_submodels, make_weights_results
from ..utils.scspill_helpers.config import SCSPILLConfig
from ..utils.scspill_helpers.inference import build_inference, prediction_interval_spec
from ..utils.scspill_helpers.pipeline import run_scspill
from ..utils.scspill_helpers.setup import prepare_scspill_inputs
from ..utils.scspill_helpers.structures import SCSPILLResults


class SCSPILL:
    """Bayesian spatial-spillover synthetic control estimator.

    Parameters
    ----------
    config : SCSPILLConfig or dict
        The estimation configuration; a dict is coerced into
        :class:`scspill.config_models.SCSPILLConfig`.

    Examples
    --------
    ```python
    from scspill import SCSPILL
    from scspill.data import load_california

    panel = load_california()
    result = SCSPILL(
        {**panel.config_kwargs(), "m_iter": 2000, "burn": 1000, "seed": 42}
    ).fit()
    result.att, result.att_ci
    result.rho_hat, result.rho_ci
    result.spillover_panel["Nevada"]
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
        self.display_graphs = config.display_graphs

    def fit(self) -> SCSPILLResults:
        """Estimate the model and return standardized results.

        Runs the balance check, panel preparation, the two-step sampler, the
        posterior effect sweep, and (optionally) the default plot panel.

        Returns
        -------
        SCSPILLResults

        Raises
        ------
        ScspillDataError
            On invalid panel data or misaligned spatial weights.
        ScspillEstimationError
            On sampler or effect-computation failure.
        ScspillPlottingError
            On plot generation failure (only when ``display_graphs``).
        """
        cfg = self.config

        try:
            balance(cfg.df, cfg.unitid, cfg.time)
            inputs = prepare_scspill_inputs(
                cfg.df,
                cfg.outcome,
                cfg.treat,
                cfg.unitid,
                cfg.time,
                cfg.spatial_W,
                cfg.spatial_w,
                covariates=cfg.covariates,
            )
        except (ScspillDataError, ScspillConfigError):
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise ScspillDataError(f"SCSPILL data preparation failed: {exc}") from exc

        try:
            fit = run_scspill(
                inputs,
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
        except (ScspillDataError, ScspillConfigError, ScspillEstimationError):
            raise
        except Exception as exc:  # pragma: no cover - defensive translation
            raise ScspillEstimationError(f"SCSPILL estimation failed: {exc}") from exc

        effects = fit.effects
        weights = make_weights_results(
            dict(
                zip(
                    (str(u) for u in inputs.control_labels),
                    fit.alpha_posterior.alpha_hat,
                    strict=True,
                )
            ),
            "unconstrained horseshoe weights (posterior mean)",
            extra={
                "scm_simplex_weights": fit.scm_weights,
                "rho_hat": fit.sar_posterior.rho_hat,
                "acc_rho": fit.sar_posterior.acc_rho,
            },
        )
        submodels = build_effect_submodels(
            observed_outcome=inputs.Y0,
            counterfactual_outcome=effects.cf_mean,
            n_pre_periods=inputs.T0,
            n_post_periods=inputs.T1,
            time_periods=inputs.time_labels,
            intervention_time=inputs.time_labels[inputs.T0],
            weights=weights,
            inference=build_inference(effects),
            effects_overrides={"att": effects.att},
            additional_effects={
                "att_plugin": effects.att_plugin,
                "att_scm": effects.att_scm,
                "rho_hat": fit.sar_posterior.rho_hat,
            },
            prediction_interval=prediction_interval_spec(effects),
        )
        results = SCSPILLResults(
            **submodels,
            method_details=MethodDetailsResults(
                method_name="SCSPILL",
                is_recommended=True,
                parameters_used={
                    "m_iter": cfg.m_iter,
                    "burn": cfg.burn,
                    "p_factors": cfg.p_factors,
                    "step_rho": cfg.step_rho,
                    "adapt_rho": cfg.adapt_rho,
                    "beta_prior": cfg.beta_prior,
                    "propagate_alpha": cfg.propagate_alpha,
                    "seed": cfg.seed,
                    "ci": cfg.ci,
                    "backend": cfg.backend,
                },
            ),
            inputs=inputs,
            alpha_posterior=fit.alpha_posterior,
            sar_posterior=fit.sar_posterior,
            effects_detail=effects,
            scm_weights=fit.scm_weights,
            mcmc_summary_table=fit.mcmc_summary_table,
            plot_config=cfg.resolved_plot(),
        )

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
