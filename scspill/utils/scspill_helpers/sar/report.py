"""Assemble a ``sar`` fit into the fields of :class:`SCSPILLResults`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ....config_models import MethodDetailsResults
from ...results_helpers import build_effect_submodels, make_weights_results
from .inference import build_inference, prediction_interval_spec
from .structures import SARFit

if TYPE_CHECKING:  # pragma: no cover
    from ..config import SCSPILLConfig


def build_sar_report(cfg: SCSPILLConfig, fit: SARFit) -> dict[str, Any]:
    """Map a ``sar`` fit onto the standardized result surface.

    Parameters
    ----------
    cfg : SCSPILLConfig
        The configuration that produced the fit.
    fit : SARFit
        The pipeline output.

    Returns
    -------
    dict
        Keyword arguments for :class:`SCSPILLResults` (everything except
        ``plot_config``).
    """
    inputs = fit.inputs
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
    return {
        **submodels,
        "method_details": MethodDetailsResults(
            method_name="SCSPILL/sar",
            is_recommended=True,
            parameters_used={
                "method": "sar",
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
        "method": "sar",
        "inputs": inputs,
        "effects_detail": effects,
        "scm_weights": fit.scm_weights,
        "sar": fit,
    }
