"""Inference assembly for SCSPILL results.

Maps the posterior effect summaries into the standardized
:class:`~scspill.config_models.InferenceResults` model and the canonical
per-period credible-band specification consumed by
:func:`scspill.utils.results_helpers.build_effect_submodels`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...config_models import InferenceResults
from .structures import SCSPILLEffects


def build_inference(effects: SCSPILLEffects) -> InferenceResults:
    """Standardized inference block from the posterior effect summaries.

    Parameters
    ----------
    effects : SCSPILLEffects
        The posterior effects detail.

    Returns
    -------
    InferenceResults
        With ``method="bayesian_posterior"``, the ATT credible interval, the
        posterior standard deviation of the ATT draws as the standard error,
        and the effects detail attached as ``details``.
    """
    att_draws = np.asarray(effects.att_draws, dtype=float)
    return InferenceResults(
        ci_lower=effects.att_ci[0],
        ci_upper=effects.att_ci[1],
        standard_error=float(att_draws.std(ddof=1)) if att_draws.size > 1 else None,
        confidence_level=effects.ci_level,
        method="bayesian_posterior",
        details=effects,
    )


def prediction_interval_spec(effects: SCSPILLEffects) -> dict[str, Any]:
    """Canonical band specification for the counterfactual credible interval.

    Parameters
    ----------
    effects : SCSPILLEffects
        The posterior effects detail.

    Returns
    -------
    dict
        The ``prediction_interval`` argument for
        :func:`~scspill.utils.results_helpers.build_effect_submodels`.
    """
    return {
        "lower": effects.cf_lower,
        "upper": effects.cf_upper,
        "level": effects.ci_level,
        "kind": "bayesian",
    }
