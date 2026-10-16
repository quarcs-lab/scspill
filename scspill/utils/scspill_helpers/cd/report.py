"""Assemble a ``cd`` fit into the fields of :class:`SCSPILLResults`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ....config_models import InferenceResults, MethodDetailsResults
from ...results_helpers import build_effect_submodels, make_weights_results
from ..containers import SCSPILLEffects
from .structures import CDFit

if TYPE_CHECKING:  # pragma: no cover
    from ..config import SCSPILLConfig


def cd_effects(fit: CDFit) -> SCSPILLEffects:
    """Fill the cross-model effect contract from a ``cd`` fit.

    The headline is the first treated unit. Its counterfactual is its SCM
    fit ``a_1 + B_1 Y_t`` before treatment and ``Y_1 - alpha_1`` after; the
    band, defined after treatment only, maps the selected pointwise interval
    of ``alpha_1`` onto the counterfactual scale. The spillover panel holds
    every control's SCM residuals before treatment and its estimated
    spillover after, as in the ``sar`` model.
    """
    inputs = fit.inputs
    t0 = inputs.T0
    treated = inputs.treated_label
    controls = list(inputs.control_labels)
    effect_ci = fit.effect_ci
    alpha0 = fit.effects_panel[treated].to_numpy()

    cf_mean = np.r_[fit.scm_paths[treated].to_numpy()[:t0], inputs.Y0[t0:] - alpha0]
    nan_pre = np.full(t0, np.nan)
    cf_lower = np.r_[nan_pre, inputs.Y0[t0:] - effect_ci[treated]["upper"].to_numpy()]
    cf_upper = np.r_[nan_pre, inputs.Y0[t0:] - effect_ci[treated]["lower"].to_numpy()]

    time_index = pd.Index(inputs.time_labels, name="time")
    cols = pd.Index(controls, name="unit")
    spill_mean = pd.DataFrame(
        np.vstack([fit.pre_residuals[controls].to_numpy(), fit.effects_panel[controls].to_numpy()]),
        index=time_index,
        columns=cols,
    )
    pre_band = np.full((t0, len(controls)), np.nan)
    lower = np.column_stack([effect_ci[c]["lower"].to_numpy() for c in controls])
    upper = np.column_stack([effect_ci[c]["upper"].to_numpy() for c in controls])
    return SCSPILLEffects(
        att=fit.att_by_unit[treated],
        att_ci=None,
        cf_mean=cf_mean,
        cf_lower=cf_lower,
        cf_upper=cf_upper,
        spill_mean=spill_mean,
        spill_lower=pd.DataFrame(np.vstack([pre_band, lower]), index=time_index, columns=cols),
        spill_upper=pd.DataFrame(np.vstack([pre_band, upper]), index=time_index, columns=cols),
        ci_level=fit.ci_level,
        interval_kind="cd_published" if fit.ci_method == "published" else "cd_inversion",
        att_scm=fit.att_scm_by_unit[treated],
    )


def build_cd_report(cfg: SCSPILLConfig, fit: CDFit) -> dict[str, Any]:
    """Map a ``cd`` fit onto the standardized result surface.

    Parameters
    ----------
    cfg : SCSPILLConfig
        The configuration that produced the fit.
    fit : CDFit
        The pipeline output.

    Returns
    -------
    dict
        Keyword arguments for :class:`SCSPILLResults` (everything except
        ``plot_config``).
    """
    inputs = fit.inputs
    treated = inputs.treated_label
    effects = cd_effects(fit)
    t0 = inputs.T0
    weights = make_weights_results(
        {c: fit.scm_matrix.loc[treated, c] for c in inputs.control_labels},
        "intercept-adjusted simplex SCM (the treated unit's equation)",
        extra={"intercept": float(fit.intercepts[treated])},
    )
    submodels = build_effect_submodels(
        observed_outcome=inputs.Y0,
        counterfactual_outcome=effects.cf_mean,
        n_pre_periods=t0,
        n_post_periods=inputs.T1,
        time_periods=inputs.time_labels,
        intervention_time=inputs.time_labels[t0],
        weights=weights,
        inference=InferenceResults(
            method=f"cao_dowd_P_{fit.reference_mode}",
            confidence_level=fit.ci_level,
            details=(
                "Pointwise intervals and P tests per period are in the cd fit "
                "(effect_ci, pointwise_p_values, test_hypothesis); no ATT "
                "interval is implied."
            ),
        ),
        effects_overrides={"att": effects.att},
        additional_effects={"att_scm": effects.att_scm, "att_by_unit": fit.att_by_unit},
        prediction_interval={
            "lower": effects.cf_lower[t0:],
            "upper": effects.cf_upper[t0:],
            "periods": inputs.time_post,
            "level": fit.ci_level,
            "kind": effects.interval_kind,
        },
    )
    return {
        **submodels,
        "method_details": MethodDetailsResults(
            method_name="SCSPILL/cd",
            is_recommended=True,
            parameters_used={
                "method": "cd",
                "spillover_structure": cfg.spillover_structure,
                "reference_mode": cfg.reference_mode,
                "quantile_method": cfg.quantile_method,
                "ci_method": cfg.ci_method,
                "ci": cfg.ci,
                "scm_maxiter": cfg.scm_maxiter,
                "scm_ftol": cfg.scm_ftol,
            },
        ),
        "method": "cd",
        "inputs": inputs,
        "effects_detail": effects,
        "cd": fit,
    }
