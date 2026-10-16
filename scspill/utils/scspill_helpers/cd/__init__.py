"""The ``method="cd"`` model: Cao & Dowd's synthetic control with spillovers.

Implements:

    Cao, J., & Dowd, C. "Estimation and Inference for Synthetic Control
    Methods with Spillover Effects." arXiv:1902.07343v3.

Every unit gets an intercept-adjusted simplex synthetic control fitted on the
pre-treatment outcomes. A known exposure structure ``A`` then separates the
treatment effect on the treated unit from the spillovers on the exposed
controls in every post-treatment period, and Andrews' end-of-sample P test,
generalized to hypotheses ``C alpha_t = d``, gives inference from the
pre-treatment residuals alone.

The defaults are the paper's Section 4 definitions: the left inverse of the
empirical CDF as the critical value, intervals by test inversion, and
full-sample reference statistics. The authors' Proposition 99 application
code (jcao0/synthetic-control-spillover at commit 86004020) uses an
interpolated quantile and intervals that add the signed placebo quantiles to
the estimate; both are kept as switches (``quantile_method="linear"``,
``ci_method="published"``).

Module layout:

* :mod:`.setup` -- the exposure matrix ``A`` and :func:`prepare_cd_inputs`.
* :mod:`.scm_core` -- the intercept-adjusted simplex SCM equations.
* :mod:`.estimation` -- the effect maps from the residual equations.
* :mod:`.inference` -- P tests, intervals, the ``kappa_A`` test, and
  structure selection by ``kappa_A``.
* :mod:`.sensitivity` -- bias bounds under missed spillovers.
* :mod:`.pipeline` -- :func:`run_cd`, the driver.
* :mod:`.report` -- the fit mapped onto the standardized result.
* :mod:`.structures` -- :class:`CDInputs`, :class:`CDTestResult`,
  :class:`CDFit`, :class:`CDEfficientFit`, :class:`PureDonorSensitivity`,
  :class:`KappaSelection`.
"""

from .inference import select_structure_by_kappa
from .pipeline import run_cd
from .report import build_cd_report, cd_effects
from .sensitivity import pure_donor_sensitivity
from .setup import (
    build_A_distance_decay,
    build_A_homogeneous,
    build_A_per_unit,
    build_exposure_matrix,
    decay_weights,
    prepare_cd_inputs,
    validate_effect_matrix,
)
from .structures import (
    CDEfficientFit,
    CDFit,
    CDInputs,
    CDTestResult,
    KappaSelection,
    PureDonorSensitivity,
)

__all__ = [
    "CDEfficientFit",
    "CDFit",
    "CDInputs",
    "CDTestResult",
    "KappaSelection",
    "PureDonorSensitivity",
    "build_A_distance_decay",
    "build_A_homogeneous",
    "build_A_per_unit",
    "build_cd_report",
    "build_exposure_matrix",
    "cd_effects",
    "decay_weights",
    "prepare_cd_inputs",
    "pure_donor_sensitivity",
    "run_cd",
    "select_structure_by_kappa",
    "validate_effect_matrix",
]
